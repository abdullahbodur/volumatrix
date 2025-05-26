#!/usr/bin/env python3
"""
Example: Custom Visualization Backend in Volumatrix

This script demonstrates how to use Volumatrix's custom OpenGL-based visualization backend.
"""

import numpy as np
from volumatrix.visualization import VolumatrixRenderer


def create_cube_vertices():
  """Create vertices for a cube with positions, normals, and texture coordinates."""
  vertices = np.array([
      # positions          # normals           # texture coords
      -0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.0,
      0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 0.0,
      0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 1.0,
      0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 1.0, 1.0,
      -0.5, 0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 1.0,
      -0.5, -0.5, -0.5, 0.0, 0.0, -1.0, 0.0, 0.0,

      -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
      0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
      0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
      0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
      -0.5, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0,
      -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0,

      -0.5, 0.5, 0.5, -1.0, 0.0, 0.0, 1.0, 0.0,
      -0.5, 0.5, -0.5, -1.0, 0.0, 0.0, 1.0, 1.0,
      -0.5, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 1.0,
      -0.5, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 1.0,
      -0.5, -0.5, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0,
      -0.5, 0.5, 0.5, -1.0, 0.0, 0.0, 1.0, 0.0,

      0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
      0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
      0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
      0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
      0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,
      0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,

      -0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 0.0, 1.0,
      0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 1.0, 1.0,
      0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0,
      0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 1.0, 0.0,
      -0.5, -0.5, 0.5, 0.0, -1.0, 0.0, 0.0, 0.0,
      -0.5, -0.5, -0.5, 0.0, -1.0, 0.0, 0.0, 1.0,

      -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
      0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0,
      0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
      0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
      -0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0,
      -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0
  ], dtype=np.float32)
  return vertices


def main():
  """Main function demonstrating the custom visualization backend."""
  # Create renderer
  renderer = VolumatrixRenderer(width=800, height=600)

  # Create cube vertices
  vertices = create_cube_vertices()
  renderer.set_vertices(vertices)

  # Main loop
  while not renderer.window.should_close():
    renderer.render()

  # Cleanup
  renderer.cleanup()


if __name__ == "__main__":
  main()
