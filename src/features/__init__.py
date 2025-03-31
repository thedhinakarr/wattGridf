"""
Feature engineering modules for the WattGrid project.

This package contains utilities for creating advanced features and
spatial relationships between Points of Connection (POCs) using
Dynamic Time Warping (DTW).
"""

from .build_features import (
    create_pivot_table,
    compute_dtw_distances,
    create_adjacency_matrix,
    extract_static_node_features,
    create_adjacency_list
)

__all__ = [
    'create_pivot_table',
    'compute_dtw_distances',
    'create_adjacency_matrix',
    'extract_static_node_features',
    'create_adjacency_list'
]
