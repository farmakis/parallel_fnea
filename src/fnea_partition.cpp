/**
 * @file fnea_partition.cpp
 * @brief Implementation of parallel FNEA partitioning algorithm
 */

#include "fnea_partition.hpp"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

#define TPL template <typename real_t, typename index_t>
#define FNEA_RESULT FNEAResult<real_t, index_t>

using namespace std;

namespace fnea {

// Helper struct for hashing pairs (C++11 compatible)
template<typename index_t>
struct PairHash {
    std::size_t operator()(const std::pair<index_t, index_t>& p) const {
        return std::hash<index_t>{}(p.first) ^ (std::hash<index_t>{}(p.second) << 1);
    }
};

template<typename real_t>
void compute_merged_bounding_box(
    const real_t* c1,
    const real_t* c2,
    const real_t* bb1,
    const real_t* bb2,
    real_t* bb_merged_out) {
    
    for (int i = 0; i < 3; ++i) {
        real_t minbound1 = c1[i] - bb1[i] / 2;
        real_t minbound2 = c2[i] - bb2[i] / 2;
        real_t maxbound1 = c1[i] + bb1[i] / 2;
        real_t maxbound2 = c2[i] + bb2[i] / 2;
        
        real_t minbound = std::min(minbound1, minbound2);
        real_t maxbound = std::max(maxbound1, maxbound2);
        
        bb_merged_out[i] = maxbound - minbound;
    }
}

TPL void compute_feature_heterogeneity(
    index_t num_edges,
    index_t num_features,
    const index_t* edges,
    const real_t* n,
    const real_t* x,
    const real_t* h,
    real_t* hf_out) {
    
    #pragma omp parallel for
    for (index_t e = 0; e < num_edges; ++e) {
        index_t u = edges[2 * e];
        index_t v = edges[2 * e + 1];
        
        real_t n1 = n[u];
        real_t n2 = n[v];
        real_t h1 = 0;
        real_t h2 = 0;
        
        // Compute heterogeneity for each feature dimension
        for (index_t f = 0; f < num_features; ++f) {
            h1 += h[u * num_features + f];
            h2 += h[v * num_features + f];
        }
        
        // Compute merged heterogeneity (standard deviation of the two nodes)
        real_t hm = 0;
        for (index_t f = 0; f < num_features; ++f) {
            real_t xu = x[u * num_features + f];
            real_t xv = x[v * num_features + f];
            real_t diff = xu - xv;
            hm += diff * diff;
        }
        hm = std::sqrt(hm / num_features);
        
        // Compute heterogeneity increase
        hf_out[e] = (n1 + n2) * hm - (n1 * h1 + n2 * h2);
    }
}

TPL void compute_shape_heterogeneity(
    index_t num_edges,
    const index_t* edges,
    const real_t* n,
    const real_t* pos,
    const real_t* bb,
    real_t compactness,
    real_t* hs_out) {
    
    #pragma omp parallel for
    for (index_t e = 0; e < num_edges; ++e) {
        index_t u = edges[2 * e];
        index_t v = edges[2 * e + 1];
        
        real_t n1 = n[u];
        real_t n2 = n[v];
        
        // Get node positions and bounding boxes
        const real_t* pos_u = &pos[u * 3];
        const real_t* pos_v = &pos[v * 3];
        const real_t* bb_u = &bb[u * 3];
        const real_t* bb_v = &bb[v * 3];
        
        // Compute merged bounding box
        real_t bb_merged[3];
        compute_merged_bounding_box(pos_u, pos_v, bb_u, bb_v, bb_merged);
        
        // Compute compactness measures
        real_t comp1 = 0, comp2 = 0, compm = 0;
        for (int i = 0; i < 3; ++i) {
            comp1 += bb_u[i];
            comp2 += bb_v[i];
            compm += bb_merged[i];
        }
        comp1 = (comp1 / 3.0) / std::cbrt(n1);
        comp2 = (comp2 / 3.0) / std::cbrt(n2);
        compm = (compm / 3.0) / std::cbrt(n1 + n2);
        
        // Compute shape heterogeneity increase
        hs_out[e] = (n1 + n2) * compm - (n1 * comp1 + n2 * comp2);
    }
}

template<typename index_t>
index_t rebuild_edges(
    index_t num_original_edges,
    const index_t* original_edges,
    const index_t* super_index,
    std::vector<std::pair<index_t, index_t>>& new_edges_out) {
    
    std::unordered_set<std::pair<index_t, index_t>, PairHash<index_t>> edge_set;
    
    new_edges_out.clear();
    
    for (index_t e = 0; e < num_original_edges; ++e) {
        index_t u_orig = original_edges[2 * e];
        index_t v_orig = original_edges[2 * e + 1];
        
        index_t u_super = super_index[u_orig];
        index_t v_super = super_index[v_orig];
        
        // Skip self-loops
        if (u_super == v_super) continue;
        
        // Ensure consistent edge ordering (u < v)
        if (u_super > v_super) {
            std::swap(u_super, v_super);
        }
        
        std::pair<index_t, index_t> edge(u_super, v_super);
        if (edge_set.find(edge) == edge_set.end()) {
            edge_set.insert(edge);
            new_edges_out.push_back(edge);
        }
    }
    
    return new_edges_out.size();
}

TPL FNEA_RESULT fnea_partition_level(
    index_t num_nodes,
    index_t num_features,
    const real_t* pos,
    const real_t* x,
    const real_t* h,
    const real_t* bb,
    const real_t* rgb,
    const index_t* source_csr,
    const index_t* target,
    const real_t* edge_weights,
    const real_t* vert_weights,
    real_t scale_factor,
    real_t compactness,
    real_t spatial_weight,
    bool verbose,
    int max_num_threads,
    bool balance_parallel_split,
    bool compute_time,
    bool compute_list,
    bool compute_graph) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    FNEA_RESULT result;
    result.times.clear();
    
    #ifdef _OPENMP
    if (max_num_threads > 0) {
        omp_set_num_threads(max_num_threads);
    }
    #endif
    
    // Initialize working arrays
    std::vector<real_t> n(vert_weights, vert_weights + num_nodes);
    std::vector<real_t> current_pos(pos, pos + num_nodes * 3);
    std::vector<real_t> current_x(x, x + num_nodes * num_features);
    std::vector<real_t> current_h(h, h + num_nodes * num_features);
    std::vector<real_t> current_bb(bb, bb + num_nodes * 3);
    std::vector<real_t> current_rgb(rgb, rgb + num_nodes * 3);
    
    std::vector<index_t> super_index(num_nodes);
    std::iota(super_index.begin(), super_index.end(), 0);
    
    // Copy initial graph structure
    index_t num_edges = source_csr[num_nodes] - source_csr[0];
    std::vector<index_t> current_source_csr(source_csr, source_csr + num_nodes + 1);
    std::vector<index_t> current_target(target, target + num_edges);
    std::vector<real_t> current_edge_weights(edge_weights, edge_weights + num_edges);
    
    // Create original edge list for rebuilding
    std::vector<index_t> original_edges(num_edges * 2);
    for (index_t u = 0; u < num_nodes; ++u) {
        for (index_t e = source_csr[u]; e < source_csr[u + 1]; ++e) {
            original_edges[2 * e] = u;
            original_edges[2 * e + 1] = target[e];
        }
    }
    
    index_t current_num_nodes = num_nodes;
    index_t iteration = 0;
    real_t scale_threshold = scale_factor * scale_factor;
    
    if (compute_time) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<real_t>(current_time - start_time).count();
        result.times.push_back(elapsed);
    }
    
    while (true) {
        ++iteration;
        
        if (verbose) {
            printf("FNEA Iteration %d: Processing %d nodes\\n", (int)iteration, (int)current_num_nodes);
        }
        
        // Step 1: Find best merge candidate for each node (parallel)
        std::vector<index_t> merge_candidate(current_num_nodes, static_cast<index_t>(-1));
        
        #pragma omp parallel for
        for (index_t u = 0; u < current_num_nodes; ++u) {
            index_t start = current_source_csr[u];
            index_t end = current_source_csr[u + 1];
            
            if (end > start) {
                real_t min_weight = scale_threshold;
                index_t best_neighbor = static_cast<index_t>(-1);
                
                for (index_t e = start; e < end; ++e) {
                    index_t v = current_target[e];
                    real_t weight = current_edge_weights[e];
                    
                    if (weight < min_weight) {
                        min_weight = weight;
                        best_neighbor = v;
                    }
                }
                
                merge_candidate[u] = best_neighbor;
            }
        }
        
        // Step 2: Find mutual best-fitting pairs (sequential to avoid conflicts)
        std::vector<std::pair<index_t, index_t>> merge_pairs;
        std::vector<bool> visited(current_num_nodes, false);
        
        for (index_t u = 0; u < current_num_nodes; ++u) {
            if (visited[u]) continue;
            
            index_t v = merge_candidate[u];
            if (v == static_cast<index_t>(-1)) continue;
            
            if (merge_candidate[v] == u) {
                merge_pairs.push_back({u, v});
                visited[u] = true;
                visited[v] = true;
            }
        }
        
        if (merge_pairs.empty()) {
            if (verbose) {
                printf("No more merges possible. Stopping.\\n");
            }
            break;
        }
        
        if (verbose) {
            printf("Iteration %d: %d merges\\n", (int)iteration, (int)merge_pairs.size());
        }
        
        // Step 3: Perform merges (can be parallelized with care)
        #pragma omp parallel for
        for (size_t i = 0; i < merge_pairs.size(); ++i) {
            index_t u = merge_pairs[i].first;
            index_t v = merge_pairs[i].second;
            
            real_t n_u = n[u];
            real_t n_v = n[v];
            real_t total_n = n_u + n_v;
            
            // Update super_index mapping
            #pragma omp critical
            {
                for (size_t k = 0; k < super_index.size(); ++k) {
                    if (super_index[k] == v) {
                        super_index[k] = u;
                    }
                }
            }
            
            // Update node features (weighted average)
            for (index_t f = 0; f < num_features; ++f) {
                current_x[u * num_features + f] = 
                    (n_u * current_x[u * num_features + f] + n_v * current_x[v * num_features + f]) / total_n;
            }
            
            // Update heterogeneity (standard deviation of merged features)
            for (index_t f = 0; f < num_features; ++f) {
                real_t xu = current_x[u * num_features + f];
                real_t xv = current_x[v * num_features + f];
                current_h[u * num_features + f] = std::abs(xu - xv);
            }
            
            // Update positions (weighted average)
            for (int d = 0; d < 3; ++d) {
                current_pos[u * 3 + d] = 
                    (n_u * current_pos[u * 3 + d] + n_v * current_pos[v * 3 + d]) / total_n;
            }
            
            // Update colors (weighted average)
            for (int d = 0; d < 3; ++d) {
                current_rgb[u * 3 + d] = 
                    (n_u * current_rgb[u * 3 + d] + n_v * current_rgb[v * 3 + d]) / total_n;
            }
            
            // Update bounding box (merged extents)
            compute_merged_bounding_box(
                &current_pos[u * 3], &current_pos[v * 3],
                &current_bb[u * 3], &current_bb[v * 3],
                &current_bb[u * 3]
            );
            
            // Update node weight
            n[u] = total_n;
            
            // Deactivate node v
            n[v] = 0;
            for (index_t f = 0; f < num_features; ++f) {
                current_x[v * num_features + f] = 0;
                current_h[v * num_features + f] = 0;
            }
            for (int d = 0; d < 3; ++d) {
                current_pos[v * 3 + d] = 0;
                current_rgb[v * 3 + d] = 0;
                current_bb[v * 3 + d] = 0;
            }
        }
        
        // Step 4: Compact representation (remove deactivated nodes)
        std::vector<bool> active_mask(current_num_nodes);
        for (index_t i = 0; i < current_num_nodes; ++i) {
            active_mask[i] = n[i] > 0;
        }
        
        std::vector<index_t> new_labels(current_num_nodes, static_cast<index_t>(-1));
        index_t new_index = 0;
        for (index_t i = 0; i < current_num_nodes; ++i) {
            if (active_mask[i]) {
                new_labels[i] = new_index++;
            }
        }
        
        // Compact arrays
        index_t new_num_nodes = new_index;
        std::vector<real_t> new_n(new_num_nodes);
        std::vector<real_t> new_pos(new_num_nodes * 3);
        std::vector<real_t> new_x(new_num_nodes * num_features);
        std::vector<real_t> new_h(new_num_nodes * num_features);
        std::vector<real_t> new_bb(new_num_nodes * 3);
        std::vector<real_t> new_rgb(new_num_nodes * 3);
        
        new_index = 0;
        for (index_t i = 0; i < current_num_nodes; ++i) {
            if (active_mask[i]) {
                new_n[new_index] = n[i];
                for (int d = 0; d < 3; ++d) {
                    new_pos[new_index * 3 + d] = current_pos[i * 3 + d];
                    new_bb[new_index * 3 + d] = current_bb[i * 3 + d];
                    new_rgb[new_index * 3 + d] = current_rgb[i * 3 + d];
                }
                for (index_t f = 0; f < num_features; ++f) {
                    new_x[new_index * num_features + f] = current_x[i * num_features + f];
                    new_h[new_index * num_features + f] = current_h[i * num_features + f];
                }
                new_index++;
            }
        }
        
        // Update super_index mapping
        for (index_t& si : super_index) {
            si = new_labels[si];
        }
        
        // Step 5: Rebuild graph structure
        std::vector<std::pair<index_t, index_t>> new_edges;
        rebuild_edges(static_cast<index_t>(original_edges.size() / 2), original_edges.data(), super_index.data(), new_edges);
        
        // Convert to CSR format (simplified - would need proper CSR conversion)
        current_source_csr.assign(new_num_nodes + 1, 0);
        current_target.clear();
        current_edge_weights.clear();
        
        // Build CSR structure
        for (const auto& edge : new_edges) {
            current_source_csr[edge.first + 1]++;
        }
        
        // Convert counts to offsets
        for (index_t i = 1; i <= new_num_nodes; ++i) {
            current_source_csr[i] += current_source_csr[i - 1];
        }
        
        current_target.resize(new_edges.size());
        current_edge_weights.resize(new_edges.size());
        
        std::vector<index_t> edge_counters(new_num_nodes, 0);
        for (const auto& edge : new_edges) {
            index_t pos = current_source_csr[edge.first] + edge_counters[edge.first]++;
            current_target[pos] = edge.second;
        }
        
        // Step 6: Recompute edge weights
        std::vector<index_t> edge_list(new_edges.size() * 2);
        for (size_t i = 0; i < new_edges.size(); ++i) {
            edge_list[2 * i] = new_edges[i].first;
            edge_list[2 * i + 1] = new_edges[i].second;
        }
        
        std::vector<real_t> hf(new_edges.size());
        std::vector<real_t> hs(new_edges.size());
        
        compute_feature_heterogeneity(
            static_cast<index_t>(new_edges.size()), num_features, edge_list.data(),
            new_n.data(), new_x.data(), new_h.data(), hf.data()
        );
        
        compute_shape_heterogeneity(
            static_cast<index_t>(new_edges.size()), edge_list.data(),
            new_n.data(), new_pos.data(), new_bb.data(),
            compactness, hs.data()
        );
        
        for (size_t i = 0; i < new_edges.size(); ++i) {
            current_edge_weights[i] = (1 - spatial_weight) * hf[i] + spatial_weight * hs[i];
        }
        
        // Update working arrays
        n = std::move(new_n);
        current_pos = std::move(new_pos);
        current_x = std::move(new_x);
        current_h = std::move(new_h);
        current_bb = std::move(new_bb);
        current_rgb = std::move(new_rgb);
        current_num_nodes = new_num_nodes;
        
        if (compute_time) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<real_t>(current_time - start_time).count();
            result.times.push_back(elapsed);
        }
    }
    
    // Prepare results
    result.super_index = std::move(super_index);
    result.pos = std::move(current_pos);
    result.bb = std::move(current_bb);
    result.rgb = std::move(current_rgb);
    result.source_csr = std::move(current_source_csr);
    result.target = std::move(current_target);
    result.edge_weights = std::move(current_edge_weights);
    result.num_iterations = iteration;
    result.final_num_nodes = current_num_nodes;
    
    // Concatenate features and heterogeneity - keep in (nodes, features) format
    result.x_c.resize(current_num_nodes * (num_features * 2));
    for (index_t i = 0; i < current_num_nodes; ++i) {
        for (index_t f = 0; f < num_features; ++f) {
            result.x_c[i * (num_features * 2) + f] = current_x[i * num_features + f];
            result.x_c[i * (num_features * 2) + (num_features + f)] = current_h[i * num_features + f];
        }
    }
    
    // Build cluster information if requested
    if (compute_list) {
        result.cluster.resize(current_num_nodes);
        for (index_t i = 0; i < num_nodes; ++i) {
            if (result.super_index[i] < current_num_nodes) {
                result.cluster[result.super_index[i]].push_back(i);
            }
        }
    }
    
    return result;
}

/**  instantiate for compilation  **/
#if defined _OPENMP && _OPENMP < 200805
/* use of unsigned counter in parallel loops requires OpenMP 3.0;
 * although published in 2008, MSVC still does not support it as of 2020 */
template FNEAResult<float, int32_t> fnea_partition_level<float, int32_t>(
    int32_t, int32_t, const float*, const float*, const float*, const float*, const float*,
    const int32_t*, const int32_t*, const float*, const float*,
    float, float, float, bool, int, bool, bool, bool, bool);

template FNEAResult<double, int32_t> fnea_partition_level<double, int32_t>(
    int32_t, int32_t, const double*, const double*, const double*, const double*, const double*,
    const int32_t*, const int32_t*, const double*, const double*,
    double, double, double, bool, int, bool, bool, bool, bool);

template void compute_feature_heterogeneity<float, int32_t>(
    int32_t, int32_t, const int32_t*, const float*, const float*, const float*, float*);

template void compute_feature_heterogeneity<double, int32_t>(
    int32_t, int32_t, const int32_t*, const double*, const double*, const double*, double*);

template void compute_shape_heterogeneity<float, int32_t>(
    int32_t, const int32_t*, const float*, const float*, const float*, float, float*);

template void compute_shape_heterogeneity<double, int32_t>(
    int32_t, const int32_t*, const double*, const double*, const double*, double, double*);

template int32_t rebuild_edges<int32_t>(
    int32_t, const int32_t*, const int32_t*, std::vector<std::pair<int32_t, int32_t>>&);

template void compute_merged_bounding_box<float>(
    const float*, const float*, const float*, const float*, float*);

template void compute_merged_bounding_box<double>(
    const double*, const double*, const double*, const double*, double*);
#else
template FNEAResult<float, uint32_t> fnea_partition_level<float, uint32_t>(
    uint32_t, uint32_t, const float*, const float*, const float*, const float*, const float*,
    const uint32_t*, const uint32_t*, const float*, const float*,
    float, float, float, bool, int, bool, bool, bool, bool);

template FNEAResult<double, uint32_t> fnea_partition_level<double, uint32_t>(
    uint32_t, uint32_t, const double*, const double*, const double*, const double*, const double*,
    const uint32_t*, const uint32_t*, const double*, const double*,
    double, double, double, bool, int, bool, bool, bool, bool);

template void compute_feature_heterogeneity<float, uint32_t>(
    uint32_t, uint32_t, const uint32_t*, const float*, const float*, const float*, float*);

template void compute_feature_heterogeneity<double, uint32_t>(
    uint32_t, uint32_t, const uint32_t*, const double*, const double*, const double*, double*);

template void compute_shape_heterogeneity<float, uint32_t>(
    uint32_t, const uint32_t*, const float*, const float*, const float*, float, float*);

template void compute_shape_heterogeneity<double, uint32_t>(
    uint32_t, const uint32_t*, const double*, const double*, const double*, double, double*);

template uint32_t rebuild_edges<uint32_t>(
    uint32_t, const uint32_t*, const uint32_t*, std::vector<std::pair<uint32_t, uint32_t>>&);

template void compute_merged_bounding_box<float>(
    const float*, const float*, const float*, const float*, float*);

template void compute_merged_bounding_box<double>(
    const double*, const double*, const double*, const double*, double*);
#endif

} // namespace fnea