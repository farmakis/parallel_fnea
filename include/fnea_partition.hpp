#pragma once

#include <vector>
#include <cstdint>
#include <memory>

/**
 * @file fnea_partition.hpp
 * @brief Parallel Fractal Net Evolution Approach (FNEA) for graph partitioning
 * 
 * This header defines the interface for a parallelized implementation of the
 * FNEA algorithm, which performs hierarchical graph partitioning by iteratively
 * merging nodes based on mutual best-fitting criteria.
 */

namespace fnea {

/**
 * @brief Result structure for FNEA partition level computation
 */
template<typename real_t, typename index_t>
struct FNEAResult {
    // Output data
    std::vector<index_t> super_index;     ///< Mapping from original to super nodes
    std::vector<real_t> pos;              ///< Updated node positions
    std::vector<real_t> bb;               ///< Updated bounding boxes
    std::vector<real_t> rgb;              ///< Updated node colors
    std::vector<real_t> x_c;              ///< Concatenated features and heterogeneity
    
    // Graph structure
    std::vector<index_t> source_csr;      ///< CSR source array
    std::vector<index_t> target;          ///< CSR target array
    std::vector<real_t> edge_weights;     ///< Updated edge weights
    
    // Clustering information
    std::vector<std::vector<index_t>> cluster; ///< Node clusters
    
    // Timing information
    std::vector<real_t> times;            ///< Computation times per iteration
    
    // Statistics
    index_t num_iterations;               ///< Number of iterations performed
    index_t final_num_nodes;              ///< Final number of super nodes
};

/**
 * @brief Parallel FNEA partition level computation
 * 
 * Performs one level of hierarchical FNEA partitioning using OpenMP parallelization.
 * The algorithm iteratively merges nodes based on mutual best-fitting criteria
 * until no more beneficial merges can be found.
 * 
 * @tparam real_t Floating point type (float or double)
 * @tparam index_t Integer type for indices (int32_t or uint32_t)
 * 
 * @param num_nodes Number of nodes in the graph
 * @param num_features Number of feature dimensions
 * @param pos Node positions [num_nodes * 3]
 * @param x Node features [num_nodes * num_features]
 * @param h Node heterogeneity [num_nodes * num_features]
 * @param bb Node bounding boxes [num_nodes * 3]
 * @param rgb Node colors [num_nodes * 3]
 * @param source_csr CSR source indices [num_nodes + 1]
 * @param target CSR target indices [num_edges]
 * @param edge_weights Edge weights [num_edges]
 * @param vert_weights Vertex weights (node sizes) [num_nodes]
 * @param scale_factor Scale factor for merge threshold
 * @param compactness Compactness parameter [0, 1]
 * @param spatial_weight Weight for spatial vs feature heterogeneity [0, 1]
 * @param verbose Enable verbose output
 * @param max_num_threads Maximum number of OpenMP threads
 * @param balance_parallel_split Balance parallel workload
 * @param compute_time Compute timing information
 * @param compute_list Compute cluster lists
 * @param compute_graph Compute output graph
 * 
 * @return FNEAResult containing the partition results
 */
template<typename real_t, typename index_t>
FNEAResult<real_t, index_t> fnea_partition_level(
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
    bool verbose = false,
    int max_num_threads = 0,
    bool balance_parallel_split = true,
    bool compute_time = true,
    bool compute_list = true,
    bool compute_graph = true
);

/**
 * @brief Compute feature heterogeneity increase for edge merges
 * 
 * @tparam real_t Floating point type
 * @tparam index_t Integer type for indices
 * @param num_edges Number of edges
 * @param num_features Number of feature dimensions
 * @param edges Edge list [num_edges * 2]
 * @param n Node sizes [num_nodes]
 * @param x Node features [num_nodes * num_features]
 * @param h Node heterogeneity [num_nodes * num_features]
 * @param hf_out Output feature heterogeneity increases [num_edges]
 */
template<typename real_t, typename index_t>
void compute_feature_heterogeneity(
    index_t num_edges,
    index_t num_features,
    const index_t* edges,
    const real_t* n,
    const real_t* x,
    const real_t* h,
    real_t* hf_out
);

/**
 * @brief Compute shape heterogeneity increase for edge merges
 * 
 * @tparam real_t Floating point type
 * @tparam index_t Integer type for indices
 * @param num_edges Number of edges
 * @param edges Edge list [num_edges * 2]
 * @param n Node sizes [num_nodes]
 * @param pos Node positions [num_nodes * 3]
 * @param bb Node bounding boxes [num_nodes * 3]
 * @param compactness Compactness parameter
 * @param hs_out Output shape heterogeneity increases [num_edges]
 */
template<typename real_t, typename index_t>
void compute_shape_heterogeneity(
    index_t num_edges,
    const index_t* edges,
    const real_t* n,
    const real_t* pos,
    const real_t* bb,
    real_t compactness,
    real_t* hs_out
);

/**
 * @brief Rebuild graph edges after node merging
 * 
 * @tparam index_t Integer type for indices
 * @param num_original_edges Number of original edges
 * @param original_edges Original edge list [num_original_edges * 2]
 * @param super_index Super node mapping [num_original_nodes]
 * @param new_edges_out Output new edge list
 * @return Number of new edges
 */
template<typename index_t>
index_t rebuild_edges(
    index_t num_original_edges,
    const index_t* original_edges,
    const index_t* super_index,
    std::vector<std::pair<index_t, index_t>>& new_edges_out
);

/**
 * @brief Compute merged bounding box extents
 * 
 * @tparam real_t Floating point type
 * @param c1 Center of first box [3]
 * @param c2 Center of second box [3]
 * @param bb1 Extents of first box [3]
 * @param bb2 Extents of second box [3]
 * @param bb_merged_out Output merged bounding box [3]
 */
template<typename real_t>
void compute_merged_bounding_box(
    const real_t* c1,
    const real_t* c2,
    const real_t* bb1,
    const real_t* bb2,
    real_t* bb_merged_out
);

} // namespace fnea