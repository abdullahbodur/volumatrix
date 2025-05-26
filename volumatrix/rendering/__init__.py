"""
Rendering module for Volumatrix.

Contains visualization and preview tools for 3D objects.
"""

from .preview import preview, preview_jupyter, show
# from .webgl import WebGLRenderer  # TODO: Implement WebGLRenderer

__all__ = ["preview", "preview_jupyter", "show"]  # "WebGLRenderer" when implemented 