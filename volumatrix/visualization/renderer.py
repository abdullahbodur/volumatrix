"""
Main renderer for Volumatrix visualization.
"""

import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import glfw
import ctypes
from typing import Optional, List, Tuple

from .window import VolumatrixWindow
from .shaders import ShaderProgram
from .camera import Camera


class VolumatrixRenderer:
  """Main renderer class for Volumatrix visualization."""

  def __init__(self, width: int = 800, height: int = 600):
    """Initialize the renderer with window dimensions."""
    self.window = VolumatrixWindow(width, height)

    # Initialize VAO and VBO first
    self.vao = glGenVertexArrays(1)
    self.vbo = glGenBuffers(1)
    glBindVertexArray(self.vao)

    # Initialize grid VAO and VBO
    self.grid_vao = glGenVertexArrays(1)
    self.grid_vbo = glGenBuffers(1)

    # Now initialize shader (after VAO is bound)
    self.shader = ShaderProgram()

    # Camera settings
    self.default_camera_position = np.array([3.0, 3.0, 3.0])
    self.default_camera_up = np.array([0.0, 1.0, 0.0])
    self.default_camera_yaw = -45.0
    self.default_camera_pitch = -35.0

    # Initialize camera with position outside the object
    self.camera = Camera(
        position=self.default_camera_position,
        up=self.default_camera_up,
        yaw=self.default_camera_yaw,
        pitch=self.default_camera_pitch
    )

    # Camera control settings
    self.orbit_mode = False
    self.orbit_center = np.array([0.0, 0.0, 0.0])
    self.orbit_radius = 5.0
    self.orbit_angle = 0.0
    self.orbit_speed = 1.0

    # Set up callbacks
    glfw.set_cursor_pos_callback(self.window.window, self._mouse_callback)
    glfw.set_scroll_callback(self.window.window, self._scroll_callback)
    glfw.set_input_mode(self.window.window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    # Initialize timing
    self.last_frame = 0.0
    self.delta_time = 0.0

    # Initialize visualization settings
    self.show_grid = True
    self.grid_size = 10.0
    self.grid_divisions = 20
    self.grid_color = np.array([0.5, 0.5, 0.5])
    self.grid_alpha = 0.3

    # Initialize grid
    self._setup_grid()

  def reset_camera(self):
    """Reset camera to default position and orientation."""
    self.camera = Camera(
        position=self.default_camera_position,
        up=self.default_camera_up,
        yaw=self.default_camera_yaw,
        pitch=self.default_camera_pitch
    )
    self.orbit_mode = False

  def toggle_orbit_mode(self):
    """Toggle between free camera and orbit mode."""
    self.orbit_mode = not self.orbit_mode
    if self.orbit_mode:
      # Calculate orbit radius from current position
      self.orbit_radius = np.linalg.norm(
        self.camera.position - self.orbit_center)
      # Calculate orbit angle
      direction = self.camera.position - self.orbit_center
      self.orbit_angle = np.arctan2(direction[0], direction[2])

  def _update_orbit_camera(self):
    """Update camera position in orbit mode."""
    if self.orbit_mode:
      # Update orbit angle
      self.orbit_angle += self.orbit_speed * self.delta_time

      # Calculate new position
      x = self.orbit_center[0] + self.orbit_radius * np.sin(self.orbit_angle)
      z = self.orbit_center[2] + self.orbit_radius * np.cos(self.orbit_angle)
      y = self.orbit_center[1] + self.orbit_radius * \
          0.5  # Keep camera slightly above center

      self.camera.position = np.array([x, y, z])
      # Make camera look at center
      self.camera.front = self.orbit_center - self.camera.position
      self.camera.front = self.camera.front / np.linalg.norm(self.camera.front)

  def _mouse_callback(self, window, xpos: float, ypos: float):
    """Handle mouse movement."""
    if not hasattr(self, 'last_x'):
      self.last_x = xpos
      self.last_y = ypos

    xoffset = xpos - self.last_x
    yoffset = self.last_y - ypos
    self.last_x = xpos
    self.last_y = ypos

    if not self.orbit_mode:
      self.camera.process_mouse_movement(xoffset, yoffset)

  def _scroll_callback(self, window, xoffset: float, yoffset: float):
    """Handle mouse scroll."""
    if self.orbit_mode:
      # Adjust orbit radius
      self.orbit_radius = max(1.0, self.orbit_radius - yoffset * 0.5)
    else:
      self.camera.process_mouse_scroll(yoffset)

  def set_vertices(self, vertices: np.ndarray):
    """Set the vertices for rendering."""
    glBindVertexArray(self.vao)
    glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, None)
    glEnableVertexAttribArray(0)
    # Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          8 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)
    # Texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                          8 * 4, ctypes.c_void_p(6 * 4))
    glEnableVertexAttribArray(2)

  def render(self):
    """Main rendering loop."""
    current_frame = glfw.get_time()
    self.delta_time = current_frame - self.last_frame
    self.last_frame = current_frame

    # Process input
    if glfw.get_key(self.window.window, glfw.KEY_W) == glfw.PRESS and not self.orbit_mode:
      self.camera.process_keyboard("FORWARD", self.delta_time)
    if glfw.get_key(self.window.window, glfw.KEY_S) == glfw.PRESS and not self.orbit_mode:
      self.camera.process_keyboard("BACKWARD", self.delta_time)
    if glfw.get_key(self.window.window, glfw.KEY_A) == glfw.PRESS and not self.orbit_mode:
      self.camera.process_keyboard("LEFT", self.delta_time)
    if glfw.get_key(self.window.window, glfw.KEY_D) == glfw.PRESS and not self.orbit_mode:
      self.camera.process_keyboard("RIGHT", self.delta_time)
    if glfw.get_key(self.window.window, glfw.KEY_G) == glfw.PRESS:
      self.show_grid = not self.show_grid
      glfw.wait_events_timeout(0.2)  # Prevent multiple toggles
    if glfw.get_key(self.window.window, glfw.KEY_O) == glfw.PRESS:
      self.toggle_orbit_mode()
      glfw.wait_events_timeout(0.2)  # Prevent multiple toggles
    if glfw.get_key(self.window.window, glfw.KEY_R) == glfw.PRESS:
      self.reset_camera()
      glfw.wait_events_timeout(0.2)  # Prevent multiple toggles

    # Update orbit camera if in orbit mode
    self._update_orbit_camera()

    # Clear the screen
    glClearColor(0.2, 0.3, 0.3, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Render grid if enabled
    if self.show_grid:
      self._render_grid()

    # Activate shader
    self.shader.use()

    # Set uniforms
    projection = self._get_projection_matrix()
    view = self.camera.get_view_matrix()
    model = np.identity(4)

    self.shader.set_uniform_matrix4fv("projection", projection)
    self.shader.set_uniform_matrix4fv("view", view)
    self.shader.set_uniform_matrix4fv("model", model)

    # Set light properties
    self.shader.set_uniform_3fv("lightPos", np.array([1.2, 1.0, 2.0]))
    self.shader.set_uniform_3fv("viewPos", self.camera.position)
    self.shader.set_uniform_3fv("lightColor", np.array([1.0, 1.0, 1.0]))
    self.shader.set_uniform_3fv("objectColor", np.array([1.0, 0.5, 0.2]))

    # Draw the object
    glBindVertexArray(self.vao)
    glDrawArrays(GL_TRIANGLES, 0, 36)

    # Swap buffers and poll events
    self.window.swap_buffers()
    self.window.poll_events()

  def _render_grid(self):
    """Render the grid."""
    self.shader.use_grid()

    # Set grid uniforms
    projection = self._get_projection_matrix()
    view = self.camera.get_view_matrix()

    self.shader.set_grid_matrix4fv("projection", projection)
    self.shader.set_grid_matrix4fv("view", view)
    self.shader.set_grid_uniforms(self.grid_color, self.grid_alpha)

    # Draw grid
    glBindVertexArray(self.grid_vao)
    glDrawArrays(GL_LINES, 0, (self.grid_divisions * 2 + 1) * 4)

  def _get_projection_matrix(self) -> np.ndarray:
    """Get the projection matrix based on camera zoom."""
    aspect = self.window.width / self.window.height
    return self._perspective(self.camera.zoom, aspect, 0.1, 100.0)

  def _perspective(self, fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create a perspective projection matrix."""
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    result = np.zeros((4, 4))

    result[0, 0] = f / aspect
    result[1, 1] = f
    result[2, 2] = (far + near) / (near - far)
    result[2, 3] = -1.0
    result[3, 2] = (2.0 * far * near) / (near - far)

    return result

  def cleanup(self):
    """Clean up resources."""
    if hasattr(self, 'vao'):
      glDeleteVertexArrays(1, [self.vao])
    if hasattr(self, 'vbo'):
      glDeleteBuffers(1, [self.vbo])
    if hasattr(self, 'grid_vao'):
      glDeleteVertexArrays(1, [self.grid_vao])
    if hasattr(self, 'grid_vbo'):
      glDeleteBuffers(1, [self.grid_vbo])
    if hasattr(self, 'shader'):
      self.shader.cleanup()
    if hasattr(self, 'window'):
      self.window.cleanup()
