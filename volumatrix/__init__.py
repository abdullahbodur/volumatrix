"""
Volumatrix: AI-Native 3D Object Generation Library

A Python library that makes 3D model generation as accessible and fast as image generation,
while maintaining Blender-grade versatility.
"""

__version__ = "0.1.0"
__author__ = "Volumatrix Team"
__email__ = "team@volumatrix.ai"

# Core imports
from .core.object import VolumatrixObject
from .core.scene import Scene
from .core.representations import Mesh, Voxel, PointCloud

# Main API functions
from .api.generate import generate, generate_batch, generate_and_show
from .api.io import load, export, save

# Utility functions
from .utils.transforms import normalize, rescale, rotate
from .utils.conversion import (
    voxelize,
    devoxelize,
    mesh_to_pointcloud,
    pointcloud_to_mesh,
    mesh_to_voxel,
    voxel_to_mesh,
)

# Model interfaces
from .models.base import BaseModel
from .models.registry import register_model, get_model, list_models

# Rendering
from .rendering.preview import preview, preview_jupyter, show
# from .rendering.webgl import WebGLRenderer  # TODO: Implement WebGLRenderer

__all__ = [
    # Core classes
    "VolumatrixObject",
    "Scene",
    "Mesh",
    "Voxel", 
    "PointCloud",
    # Main API
    "generate",
    "generate_batch",
    "generate_and_show",
    "load",
    "export",
    "save",
    # Utilities
    "normalize",
    "rescale",
    "rotate",
    "voxelize",
    "devoxelize",
    "mesh_to_pointcloud",
    "pointcloud_to_mesh",
    "mesh_to_voxel",
    "voxel_to_mesh",
    # Models
    "BaseModel",
    "register_model",
    "get_model",
    "list_models",
    # Rendering
    "preview",
    "preview_jupyter",
    "show",
    # "WebGLRenderer",  # TODO: Add back when implemented
] 