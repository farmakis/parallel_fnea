/**
 * @file fnea_partition_cpy.cpp
 * @brief Python C extension for parallel FNEA partitioning
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "../include/fnea_partition.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

// Helper function to check array type and contiguity
template<typename T>
bool check_array_type_and_contiguity(PyArrayObject* array, int expected_dtype) {
    return PyArray_TYPE(array) == expected_dtype && 
           PyArray_FLAGS(array) & NPY_ARRAY_F_CONTIGUOUS;
}

// Helper function to create numpy array from std::vector
template<typename T>
PyObject* vector_to_numpy_array(const std::vector<T>& vec, int numpy_type, npy_intp* dims, int ndim) {
    PyObject* array = PyArray_SimpleNew(ndim, dims, numpy_type);
    if (!array) return nullptr;
    
    T* data = static_cast<T*>(PyArray_DATA((PyArrayObject*)array));
    std::copy(vec.begin(), vec.end(), data);
    
    return array;
}

// Main FNEA partition function
static PyObject* fnea_partition_level_cpy(PyObject* self, PyObject* args, PyObject* kwargs) {
    // Input arguments
    PyArrayObject *pos_array = nullptr, *x_array = nullptr, *h_array = nullptr;
    PyArrayObject *bb_array = nullptr, *rgb_array = nullptr;
    PyArrayObject *source_csr_array = nullptr, *target_array = nullptr;
    PyArrayObject *edge_weights_array = nullptr, *vert_weights_array = nullptr;
    double scale_factor = 10.0, compactness = 0.2, spatial_weight = 0.5;
    int verbose = 0, max_num_threads = 0, balance_parallel_split = 1;
    int compute_time = 1, compute_list = 1, compute_graph = 1;
    int is_double = 0;
    
    static char* kwlist[] = {
        "pos", "x", "h", "bb", "rgb", "source_csr", "target", "edge_weights", "vert_weights",
        "scale_factor", "compactness", "spatial_weight", "verbose", "max_num_threads",
        "balance_parallel_split", "is_double", "compute_time", "compute_list", "compute_graph",
        nullptr
    };
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!O!O!O!O!O!O!ddd|iiiiiii", kwlist,
                                     &PyArray_Type, &pos_array,
                                     &PyArray_Type, &x_array,
                                     &PyArray_Type, &h_array,
                                     &PyArray_Type, &bb_array,
                                     &PyArray_Type, &rgb_array,
                                     &PyArray_Type, &source_csr_array,
                                     &PyArray_Type, &target_array,
                                     &PyArray_Type, &edge_weights_array,
                                     &PyArray_Type, &vert_weights_array,
                                     &scale_factor, &compactness, &spatial_weight,
                                     &verbose, &max_num_threads, &balance_parallel_split,
                                     &is_double, &compute_time, &compute_list, &compute_graph)) {
        return nullptr;
    }
    
    try {
        // Determine data types
        int float_type = is_double ? NPY_FLOAT64 : NPY_FLOAT32;
        int index_type = NPY_UINT32; // Use uint32 for indices
        
        // Validate input arrays - now expecting C-contiguous
        if (PyArray_TYPE(pos_array) != float_type || !(PyArray_FLAGS(pos_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(x_array) != float_type || !(PyArray_FLAGS(x_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(h_array) != float_type || !(PyArray_FLAGS(h_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(bb_array) != float_type || !(PyArray_FLAGS(bb_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(rgb_array) != float_type || !(PyArray_FLAGS(rgb_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(source_csr_array) != index_type || !(PyArray_FLAGS(source_csr_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(target_array) != index_type || !(PyArray_FLAGS(target_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(edge_weights_array) != float_type || !(PyArray_FLAGS(edge_weights_array) & NPY_ARRAY_C_CONTIGUOUS) ||
            PyArray_TYPE(vert_weights_array) != float_type || !(PyArray_FLAGS(vert_weights_array) & NPY_ARRAY_C_CONTIGUOUS)) {
            PyErr_SetString(PyExc_TypeError, "Input arrays must be C-contiguous and of correct type");
            return nullptr;
        }
        
        // Get array dimensions - expect (num_nodes, features) layout for efficiency
        npy_intp num_nodes = PyArray_DIM(pos_array, 0);
        npy_intp num_features = PyArray_DIM(x_array, 1);
        
        // Validate dimensions - expect (num_nodes, features) layout
        if (PyArray_DIM(pos_array, 1) != 3 ||
            PyArray_DIM(x_array, 0) != num_nodes ||
            PyArray_DIM(h_array, 0) != num_nodes || PyArray_DIM(h_array, 1) != num_features ||
            PyArray_DIM(bb_array, 0) != num_nodes || PyArray_DIM(bb_array, 1) != 3 ||
            PyArray_DIM(rgb_array, 0) != num_nodes || PyArray_DIM(rgb_array, 1) != 3 ||
            PyArray_DIM(source_csr_array, 0) != num_nodes + 1 ||
            PyArray_DIM(vert_weights_array, 0) != num_nodes) {
            PyErr_SetString(PyExc_ValueError, "Array dimensions are inconsistent");
            return nullptr;
        }
        
        // Get data pointers
        void* pos_data = PyArray_DATA(pos_array);
        void* x_data = PyArray_DATA(x_array);
        void* h_data = PyArray_DATA(h_array);
        void* bb_data = PyArray_DATA(bb_array);
        void* rgb_data = PyArray_DATA(rgb_array);
        uint32_t* source_csr_data = static_cast<uint32_t*>(PyArray_DATA(source_csr_array));
        uint32_t* target_data = static_cast<uint32_t*>(PyArray_DATA(target_array));
        void* edge_weights_data = PyArray_DATA(edge_weights_array);
        void* vert_weights_data = PyArray_DATA(vert_weights_array);
        
        PyObject* result = nullptr;
        
        if (is_double) {
            // Double precision version
            // Input arrays are already in (num_nodes, features) format - no transpose needed
            const double* pos_ptr = static_cast<const double*>(pos_data);
            const double* x_ptr = static_cast<const double*>(x_data);
            const double* h_ptr = static_cast<const double*>(h_data);
            const double* bb_ptr = static_cast<const double*>(bb_data);
            const double* rgb_ptr = static_cast<const double*>(rgb_data);
            
            auto fnea_result = fnea::fnea_partition_level<double, uint32_t>(
                static_cast<uint32_t>(num_nodes),
                static_cast<uint32_t>(num_features),
                pos_ptr,
                x_ptr,
                h_ptr,
                bb_ptr,
                rgb_ptr,
                source_csr_data,
                target_data,
                static_cast<const double*>(edge_weights_data),
                static_cast<const double*>(vert_weights_data),
                static_cast<double>(scale_factor),
                static_cast<double>(compactness),
                static_cast<double>(spatial_weight),
                verbose != 0,
                max_num_threads,
                balance_parallel_split != 0,
                compute_time != 0,
                compute_list != 0,
                compute_graph != 0
            );
            
            // Convert results to Python objects
            npy_intp super_index_dims[] = {static_cast<npy_intp>(fnea_result.super_index.size())};
            PyObject* super_index_array = vector_to_numpy_array(
                fnea_result.super_index, NPY_UINT32, super_index_dims, 1);
            
            // Create arrays in (num_nodes, features) format to match input layout
            npy_intp pos_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), 3};
            PyObject* pos_out_array = vector_to_numpy_array(
                fnea_result.pos, NPY_FLOAT64, pos_dims, 2);
            
            npy_intp bb_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), 3};
            PyObject* bb_out_array = vector_to_numpy_array(
                fnea_result.bb, NPY_FLOAT64, bb_dims, 2);
            
            npy_intp rgb_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), 3};
            PyObject* rgb_out_array = vector_to_numpy_array(
                fnea_result.rgb, NPY_FLOAT64, rgb_dims, 2);
            
            npy_intp x_c_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), static_cast<npy_intp>(num_features * 2)};
            PyObject* x_c_array = vector_to_numpy_array(
                fnea_result.x_c, NPY_FLOAT64, x_c_dims, 2);
            
            // Build cluster list
            PyObject* cluster_list = PyList_New(fnea_result.cluster.size());
            for (size_t i = 0; i < fnea_result.cluster.size(); ++i) {
                npy_intp cluster_dims[] = {static_cast<npy_intp>(fnea_result.cluster[i].size())};
                PyObject* cluster_array = vector_to_numpy_array(
                    fnea_result.cluster[i], NPY_UINT32, cluster_dims, 1);
                PyList_SetItem(cluster_list, i, cluster_array);
            }
            
            // Build edges tuple
            npy_intp source_csr_dims[] = {static_cast<npy_intp>(fnea_result.source_csr.size())};
            PyObject* source_csr_out = vector_to_numpy_array(
                fnea_result.source_csr, NPY_UINT32, source_csr_dims, 1);
            
            npy_intp target_dims[] = {static_cast<npy_intp>(fnea_result.target.size())};
            PyObject* target_out = vector_to_numpy_array(
                fnea_result.target, NPY_UINT32, target_dims, 1);
                
            npy_intp edge_weights_dims[] = {static_cast<npy_intp>(fnea_result.edge_weights.size())};
            PyObject* edge_weights_out = vector_to_numpy_array(
                fnea_result.edge_weights, NPY_FLOAT64, edge_weights_dims, 1);
            
            PyObject* edges_tuple = PyTuple_New(3);
            PyTuple_SetItem(edges_tuple, 0, source_csr_out);
            PyTuple_SetItem(edges_tuple, 1, target_out);
            PyTuple_SetItem(edges_tuple, 2, edge_weights_out);
            
            // Build times array
            PyObject* times_array = nullptr;
            if (compute_time && !fnea_result.times.empty()) {
                npy_intp times_dims[] = {static_cast<npy_intp>(fnea_result.times.size())};
                times_array = vector_to_numpy_array(
                    fnea_result.times, NPY_FLOAT64, times_dims, 1);
            } else {
                Py_INCREF(Py_None);
                times_array = Py_None;
            }
            
            // Build result tuple
            result = PyTuple_New(8);
            PyTuple_SetItem(result, 0, super_index_array);
            PyTuple_SetItem(result, 1, pos_out_array);
            PyTuple_SetItem(result, 2, bb_out_array);
            PyTuple_SetItem(result, 3, rgb_out_array);
            PyTuple_SetItem(result, 4, x_c_array);
            PyTuple_SetItem(result, 5, cluster_list);
            PyTuple_SetItem(result, 6, edges_tuple);
            PyTuple_SetItem(result, 7, times_array);
            
        } else {
            // Single precision version
            // Input arrays are already in (num_nodes, features) format - no transpose needed
            const float* pos_ptr = static_cast<const float*>(pos_data);
            const float* x_ptr = static_cast<const float*>(x_data);
            const float* h_ptr = static_cast<const float*>(h_data);
            const float* bb_ptr = static_cast<const float*>(bb_data);
            const float* rgb_ptr = static_cast<const float*>(rgb_data);
            
            auto fnea_result = fnea::fnea_partition_level<float, uint32_t>(
                static_cast<uint32_t>(num_nodes),
                static_cast<uint32_t>(num_features),
                pos_ptr,
                x_ptr,
                h_ptr,
                bb_ptr,
                rgb_ptr,
                source_csr_data,
                target_data,
                static_cast<const float*>(edge_weights_data),
                static_cast<const float*>(vert_weights_data),
                static_cast<float>(scale_factor),
                static_cast<float>(compactness),
                static_cast<float>(spatial_weight),
                verbose != 0,
                max_num_threads,
                balance_parallel_split != 0,
                compute_time != 0,
                compute_list != 0,
                compute_graph != 0
            );
            
            // Convert results to Python objects (similar to double version but with NPY_FLOAT32)
            npy_intp super_index_dims[] = {static_cast<npy_intp>(fnea_result.super_index.size())};
            PyObject* super_index_array = vector_to_numpy_array(
                fnea_result.super_index, NPY_UINT32, super_index_dims, 1);
            
            npy_intp pos_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), 3};
            PyObject* pos_out_array = vector_to_numpy_array(
                fnea_result.pos, NPY_FLOAT32, pos_dims, 2);
            
            npy_intp bb_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), 3};
            PyObject* bb_out_array = vector_to_numpy_array(
                fnea_result.bb, NPY_FLOAT32, bb_dims, 2);
            
            npy_intp rgb_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), 3};
            PyObject* rgb_out_array = vector_to_numpy_array(
                fnea_result.rgb, NPY_FLOAT32, rgb_dims, 2);
            
            npy_intp x_c_dims[] = {static_cast<npy_intp>(fnea_result.final_num_nodes), static_cast<npy_intp>(num_features * 2)};
            PyObject* x_c_array = vector_to_numpy_array(
                fnea_result.x_c, NPY_FLOAT32, x_c_dims, 2);
            
            // Build cluster list
            PyObject* cluster_list = PyList_New(fnea_result.cluster.size());
            for (size_t i = 0; i < fnea_result.cluster.size(); ++i) {
                npy_intp cluster_dims[] = {static_cast<npy_intp>(fnea_result.cluster[i].size())};
                PyObject* cluster_array = vector_to_numpy_array(
                    fnea_result.cluster[i], NPY_UINT32, cluster_dims, 1);
                PyList_SetItem(cluster_list, i, cluster_array);
            }
            
            // Build edges tuple
            npy_intp source_csr_dims[] = {static_cast<npy_intp>(fnea_result.source_csr.size())};
            PyObject* source_csr_out = vector_to_numpy_array(
                fnea_result.source_csr, NPY_UINT32, source_csr_dims, 1);
            
            npy_intp target_dims[] = {static_cast<npy_intp>(fnea_result.target.size())};
            PyObject* target_out = vector_to_numpy_array(
                fnea_result.target, NPY_UINT32, target_dims, 1);
                
            npy_intp edge_weights_dims[] = {static_cast<npy_intp>(fnea_result.edge_weights.size())};
            PyObject* edge_weights_out = vector_to_numpy_array(
                fnea_result.edge_weights, NPY_FLOAT32, edge_weights_dims, 1);
            
            PyObject* edges_tuple = PyTuple_New(3);
            PyTuple_SetItem(edges_tuple, 0, source_csr_out);
            PyTuple_SetItem(edges_tuple, 1, target_out);
            PyTuple_SetItem(edges_tuple, 2, edge_weights_out);
            
            // Build times array
            PyObject* times_array = nullptr;
            if (compute_time && !fnea_result.times.empty()) {
                npy_intp times_dims[] = {static_cast<npy_intp>(fnea_result.times.size())};
                times_array = vector_to_numpy_array(
                    fnea_result.times, NPY_FLOAT32, times_dims, 1);
            } else {
                Py_INCREF(Py_None);
                times_array = Py_None;
            }
            
            // Build result tuple
            result = PyTuple_New(8);
            PyTuple_SetItem(result, 0, super_index_array);
            PyTuple_SetItem(result, 1, pos_out_array);
            PyTuple_SetItem(result, 2, bb_out_array);
            PyTuple_SetItem(result, 3, rgb_out_array);
            PyTuple_SetItem(result, 4, x_c_array);
            PyTuple_SetItem(result, 5, cluster_list);
            PyTuple_SetItem(result, 6, edges_tuple);
            PyTuple_SetItem(result, 7, times_array);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unknown error occurred in FNEA partition");
        return nullptr;
    }
}

// Method definitions
static PyMethodDef fnea_methods[] = {
    {"fnea_partition_level_cpy", (PyCFunction)fnea_partition_level_cpy, 
     METH_VARARGS | METH_KEYWORDS, "Parallel FNEA partition level computation"},
    {nullptr, nullptr, 0, nullptr}
};

// Module definition
static struct PyModuleDef fnea_module = {
    PyModuleDef_HEAD_INIT,
    "fnea_partition_cpy",
    "Parallel FNEA partitioning C extension",
    -1,
    fnea_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_fnea_partition_cpy(void) {
    import_array();
    return PyModule_Create(&fnea_module);
}