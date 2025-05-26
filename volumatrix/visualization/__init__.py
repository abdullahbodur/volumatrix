"""
Volumatrix Custom Visualization Module

This module provides a custom 3D visualization backend for Volumatrix.
It uses OpenGL for rendering and provides a simple, efficient way to visualize 3D objects.
"""

from .renderer import VolumatrixRenderer
from .window import VolumatrixWindow
from .shaders import ShaderProgram
from .camera import Camera

__all__ = ['VolumatrixRenderer', 'VolumatrixWindow', 'ShaderProgram', 'Camera'] 