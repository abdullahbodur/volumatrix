"""
Window management for Volumatrix visualization.
"""

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
from typing import Optional, Tuple


class VolumatrixWindow:
  """Manages the OpenGL window and context for Volumatrix visualization."""

  def __init__(self, width: int = 800, height: int = 600, title: str = "Volumatrix Viewer"):
    """Initialize the window with given dimensions and title."""
    if not glfw.init():
      raise RuntimeError("Failed to initialize GLFW")

    # Configure GLFW
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create window
    self.window = glfw.create_window(width, height, title, None, None)
    if not self.window:
      glfw.terminate()
      raise RuntimeError("Failed to create GLFW window")

    # Make the window's context current
    glfw.make_context_current(self.window)

    # Set up viewport
    glViewport(0, 0, width, height)

    # Enable depth testing
    glEnable(GL_DEPTH_TEST)

    # Store window dimensions
    self.width = width
    self.height = height

  def should_close(self) -> bool:
    """Check if the window should close."""
    return glfw.window_should_close(self.window)

  def swap_buffers(self):
    """Swap front and back buffers."""
    glfw.swap_buffers(self.window)

  def poll_events(self):
    """Poll for and process events."""
    glfw.poll_events()

  def get_size(self) -> Tuple[int, int]:
    """Get the current window size."""
    return glfw.get_window_size(self.window)

  def set_size(self, width: int, height: int):
    """Set the window size."""
    glfw.set_window_size(self.window, width, height)
    glViewport(0, 0, width, height)
    self.width = width
    self.height = height

  def cleanup(self):
    """Clean up resources."""
    glfw.destroy_window(self.window)
    glfw.terminate()

  def __enter__(self):
    """Context manager entry."""
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit."""
    self.cleanup()
