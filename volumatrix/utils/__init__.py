"""
Utilities module for Volumatrix.

Contains preprocessing, conversion, and transformation utilities.
"""

from .transforms import normalize, rescale, rotate
from .conversion import (
    voxelize,
    devoxelize,
    mesh_to_pointcloud,
    pointcloud_to_mesh,
    mesh_to_voxel,
    voxel_to_mesh,
)

__all__ = [
    "normalize",
    "rescale", 
    "rotate",
    "voxelize",
    "devoxelize",
    "mesh_to_pointcloud",
    "pointcloud_to_mesh",
    "mesh_to_voxel",
    "voxel_to_mesh",
] 