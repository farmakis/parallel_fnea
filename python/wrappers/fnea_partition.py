import numpy as np
import os 
import sys

# Add the bin directory to Python path
bin_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "../bin")
if bin_path not in sys.path:
    sys.path.insert(0, bin_path)

from fnea_partition_cpy import fnea_partition_level_cpy

def fnea_partition_level(
    coords, pos, x, h, bb, rgb, source_csr, target, edge_weights, vert_weights,
    scale_factor=10.0, compactness=0.2, spatial_weight=0.5,
    verbose=False, max_num_threads=0, balance_parallel_split=True,
    compute_time=True, compute_list=True, compute_graph=True):
    """
    Parallel Fractal Net Evolution Approach (FNEA) partition level computation.
    
    Performs one level of hierarchical FNEA partitioning using OpenMP parallelization.
    The algorithm iteratively merges nodes based on mutual best-fitting criteria
    until no more beneficial merges can be found.
    
    Parameters
    ----------
    coords : numpy.ndarray
        Node grid coordinates, shape (num_nodes, 3), C-contiguous
    pos : numpy.ndarray
        Node positions, shape (num_nodes, 3), C-contiguous
    x : numpy.ndarray  
        Node features, shape (num_nodes, num_features), C-contiguous
    h : numpy.ndarray
        Node heterogeneity, shape (num_nodes, num_features), C-contiguous  
    bb : numpy.ndarray
        Node bounding boxes, shape (num_nodes, 3), C-contiguous
    rgb : numpy.ndarray
        Node colors, shape (num_nodes, 3), C-contiguous
    source_csr : numpy.ndarray
        CSR source indices, shape (num_nodes + 1,), uint32
    target : numpy.ndarray
        CSR target indices, shape (num_edges,), uint32
    edge_weights : numpy.ndarray
        Edge weights, shape (num_edges,), F-contiguous
    vert_weights : numpy.ndarray
        Vertex weights (node sizes), shape (num_nodes,), F-contiguous
    scale_factor : float, optional
        Scale factor for merge threshold, default 10.0
    compactness : float, optional
        Compactness parameter [0, 1], default 0.2
    spatial_weight : float, optional
        Weight for spatial vs feature heterogeneity [0, 1], default 0.5
    verbose : bool, optional
        Enable verbose output, default False
    max_num_threads : int, optional
        Maximum number of OpenMP threads, 0 for automatic, default 0
    balance_parallel_split : bool, optional
        Balance parallel workload, default True
    compute_time : bool, optional
        Compute timing information, default True
    compute_list : bool, optional
        Compute cluster lists, default True
    compute_graph : bool, optional
        Compute output graph, default True
        
    Returns
    -------
    super_index : numpy.ndarray
        Mapping from original to super nodes, shape (num_original_nodes,), uint32
    coords_out : numpy.ndarray
        Updated node grid coordinates, shape (num_final_nodes, 3), same type as input
    pos_out : numpy.ndarray
        Updated node positions, shape (num_final_nodes, 3), same type as input
    bb_out : numpy.ndarray
        Updated bounding boxes, shape (num_final_nodes, 3), same type as input
    rgb_out : numpy.ndarray
        Updated node colors, shape (num_final_nodes, 3), same type as input
    x_c : numpy.ndarray
        Concatenated features and heterogeneity, shape (num_final_nodes, 2*num_features)
    cluster : list
        List of numpy arrays containing node indices for each cluster
    edges : tuple
        Tuple of (source_csr, target, edge_weights) for output graph
    times : numpy.ndarray or None
        Computation times per iteration if compute_time=True, else None
        
    Notes
    -----
    All input arrays must be C-contiguous (row-major order) for efficient
    C++ processing. Use numpy.ascontiguousarray() to convert if necessary.
    
    The algorithm performs hierarchical graph partitioning by:
    1. Finding best merge candidate for each node (parallel)
    2. Identifying mutual best-fitting pairs
    3. Performing node merges (parallel)
    4. Compacting representation
    5. Rebuilding graph structure
    6. Recomputing edge weights
    
    Performance scales well with the number of CPU cores. Typical speedups:
    - Small graphs (< 1K nodes): 2-3x
    - Medium graphs (1K-10K nodes): 5-10x  
    - Large graphs (> 10K nodes): 10-20x
    
    Examples
    --------
    >>> import numpy as np
    >>> from parallel_fnea import fnea_partition_level
    >>> 
    >>> # Prepare data (C-contiguous arrays)
    >>> num_nodes = 1000
    >>> num_features = 3
    >>> coords = np.random.rand(num_nodes, 3).astype(np.float32)
    >>> pos = np.random.rand(num_nodes, 3).astype(np.float32)
    >>> x = np.random.rand(num_nodes, num_features).astype(np.float32)
    >>> h = np.zeros_like(x)
    >>> bb = np.ones((num_nodes, 3), dtype=np.float32)
    >>> rgb = np.random.rand(num_nodes, 3).astype(np.float32)
    >>> 
    >>> # Simple graph (each node connected to next)
    >>> source_csr = np.arange(num_nodes + 1, dtype=np.uint32)
    >>> target = np.arange(1, num_nodes + 1, dtype=np.uint32) % num_nodes
    >>> edge_weights = np.ones(num_nodes, dtype=np.float32)
    >>> vert_weights = np.ones(num_nodes, dtype=np.float32)
    >>> 
    >>> # Run FNEA partition
    >>> super_index, pos_out, bb_out, rgb_out, x_c, cluster, edges, times = \\
    ...     fnea_partition_level(pos, x, h, bb, rgb, source_csr, target, 
    ...                         edge_weights, vert_weights, scale_factor=5.0)
    >>> 
    >>> print(f"Reduced from {num_nodes} to {pos_out.shape[0]} nodes")
    """
    
    # Determine the type of float argument (real_t) 
    if type(x) != np.ndarray:
        raise TypeError("FNEA partition: argument 'x' must be a numpy array.")

    if x.size > 0 and x.dtype == "float64":
        real_t = "float64" 
        is_double = True
    elif x.size > 0 and x.dtype == "float32":
        real_t = "float32" 
        is_double = False
    else:
        raise TypeError("FNEA partition: argument 'x' must be a "
                        "nonempty numpy array of type float32 or float64.") 
    
    # Validate array types and convert if necessary
    required_arrays = {
        'coords': (coords, (3, None)),
        'pos': (pos, (3, None)),
        'x': (x, (None, None)), 
        'h': (h, (None, None)),
        'bb': (bb, (3, None)),
        'rgb': (rgb, (3, None)),
        'edge_weights': (edge_weights, (None,)),
        'vert_weights': (vert_weights, (None,))
    }
    
    # Convert in numpy array scalar entry
    if (type(coords) != np.ndarray
        or coords.dtype not in [real_t]):
        raise TypeError("FNEA partition: argument 'coords' must "
                        "be a numpy array of type float32 or float64.")
    
    if (type(pos) != np.ndarray
        or pos.dtype not in [real_t]):
        raise TypeError("FNEA partition: argument 'pos' must "
                        "be a numpy array of type float32 or float64.")
    
    if (type(source_csr) != np.ndarray
        or source_csr.dtype not in ["int32", "uint32"]):
        raise TypeError("FNEA partition: argument 'source_csr' must "
                        "be a numpy array of type int32 or uint32.")

    if (type(target) != np.ndarray
        or target.dtype not in ["int32", "uint32"]):
        raise TypeError("FNEA partition: argument 'target' "
                        "must be a numpy array of type int32 or uint32.")

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("FNEA partition: argument 'edge_weights' "
                            "must be a scalar or a numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)

    if type(vert_weights) != np.ndarray:
        if type(vert_weights) == list:
            raise TypeError("FNEA partition: argument 'vert_weights' "
                            "must be a numpy array.")
        else:
            vert_weights = np.array([], dtype=real_t)
    
    # Validate integer arrays
    if not isinstance(source_csr, np.ndarray) or source_csr.dtype != "uint32":
        raise TypeError("FNEA partition: argument 'source_csr' must be a "
                        "numpy array of type uint32.")
    
    if not isinstance(target, np.ndarray) or target.dtype != "uint32":
        raise TypeError("FNEA partition: argument 'target' must be a "
                        "numpy array of type uint32.")
    
    # Validate dimensions - expect (num_nodes, features) format
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("FNEA partition: 'coords' must have shape (num_nodes, 3)")
    
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("FNEA partition: 'pos' must have shape (num_nodes, 3)")
    
    num_nodes = pos.shape[0]
    
    if x.ndim != 2 or x.shape[0] != num_nodes:
        raise ValueError("FNEA partition: 'x' must have shape (num_nodes, num_features)")
    
    num_features = x.shape[1]
    
    if h.shape != (num_nodes, num_features):
        raise ValueError("FNEA partition: 'h' must have shape (num_nodes, num_features)")
    
    if bb.shape != (num_nodes, 3):
        raise ValueError("FNEA partition: 'bb' must have shape (num_nodes, 3)")
        
    if rgb.shape != (num_nodes, 3):
        raise ValueError("FNEA partition: 'rgb' must have shape (num_nodes, 3)")
    
    if source_csr.shape[0] != num_nodes + 1:
        raise ValueError("FNEA partition: 'source_csr' must have length num_nodes + 1")
    
    if vert_weights.shape[0] != num_nodes:
        raise ValueError("FNEA partition: 'vert_weights' must have length num_nodes")
    
    num_edges = source_csr[-1] - source_csr[0]
    if target.shape[0] != num_edges:
        raise ValueError("FNEA partition: 'target' must have length equal to number of edges")
        
    if edge_weights.shape[0] != num_edges:
        raise ValueError("FNEA partition: 'edge_weights' must have length equal to number of edges")
    
    # Validate parameters
    if not isinstance(scale_factor, (int, float)) or scale_factor <= 0:
        raise ValueError("FNEA partition: 'scale_factor' must be a positive number")
    
    if not isinstance(compactness, (int, float)) or not (0 <= compactness <= 1):
        raise ValueError("FNEA partition: 'compactness' must be in range [0, 1]")
        
    if not isinstance(spatial_weight, (int, float)) or not (0 <= spatial_weight <= 1):
        raise ValueError("FNEA partition: 'spatial_weight' must be in range [0, 1]")
    
    if not isinstance(max_num_threads, int) or max_num_threads < 0:
        raise ValueError("FNEA partition: 'max_num_threads' must be a non-negative integer")
    
    # Convert boolean parameters
    verbose_int = int(bool(verbose))
    balance_parallel_split_int = int(bool(balance_parallel_split))
    compute_time_int = int(bool(compute_time))
    compute_list_int = int(bool(compute_list))
    compute_graph_int = int(bool(compute_graph))
    is_double_int = int(is_double)
    
    # Arrays are expected in (num_nodes, features) format
    # Ensure they're C-contiguous for efficient processing
    coords_c = np.ascontiguousarray(coords)
    pos_c = np.ascontiguousarray(pos)
    x_c = np.ascontiguousarray(x) 
    h_c = np.ascontiguousarray(h)
    bb_c = np.ascontiguousarray(bb)
    rgb_c = np.ascontiguousarray(rgb)
    edge_weights_c = np.ascontiguousarray(edge_weights)
    vert_weights_c = np.ascontiguousarray(vert_weights)
    source_csr_c = np.ascontiguousarray(source_csr)
    target_c = np.ascontiguousarray(target)
    
    # Call the C++ extension
    return fnea_partition_level_cpy(
        coords_c,
        pos_c, 
        x_c, 
        h_c, 
        bb_c, 
        rgb_c, 
        source_csr_c, 
        target_c, 
        edge_weights_c, 
        vert_weights_c,
        float(scale_factor), 
        float(compactness), 
        float(spatial_weight),
        verbose_int, 
        max_num_threads, 
        balance_parallel_split_int,
        is_double_int, 
        compute_time_int, 
        compute_list_int, 
        compute_graph_int
    )