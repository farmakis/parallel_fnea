"""
Parallel FNEA (Fractal Net Evolution Approach) partitioning package.

This package provides a parallelized C++ implementation of the FNEA algorithm
for hierarchical graph partitioning, with Python bindings.
"""

from .fnea_partition import fnea_partition_level

__all__ = ['fnea_partition_level']
__version__ = '1.0.0'