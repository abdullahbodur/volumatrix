"""
Core module for Volumatrix.

Contains the fundamental 3D object representations and scene management.
"""

from .object import VolumatrixObject
from .scene import Scene
from .representations import Mesh, Voxel, PointCloud

__all__ = ["VolumatrixObject", "Scene", "Mesh", "Voxel", "PointCloud"] 