"""
Camera system for Volumatrix visualization.
"""

import numpy as np
from typing import Tuple


class Camera:
  """Manages the camera for 3D viewing in Volumatrix."""

  def __init__(self, position: np.ndarray = np.array([0.0, 0.0, 3.0]),
               front: np.ndarray = np.array([0.0, 0.0, -1.0]),
               up: np.ndarray = np.array([0.0, 1.0, 0.0]),
               yaw: float = -90.0,
               pitch: float = 0.0):
    """Initialize camera with position and orientation."""
    self.position = position
    self.front = front
    self.up = up
    self.yaw = yaw
    self.pitch = pitch

    # Camera options
    self.movement_speed = 2.5
    self.mouse_sensitivity = 0.1
    self.zoom = 45.0

    self._update_camera_vectors()

  def _update_camera_vectors(self):
    """Update camera vectors based on yaw and pitch."""
    # Calculate new front vector
    front = np.array([
        np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
        np.sin(np.radians(self.pitch)),
        np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
    ])
    self.front = front / np.linalg.norm(front)

    # Recalculate right and up vectors
    self.right = np.cross(self.front, self.up)
    self.right = self.right / np.linalg.norm(self.right)
    self.up = np.cross(self.right, self.front)
    self.up = self.up / np.linalg.norm(self.up)

  def get_view_matrix(self) -> np.ndarray:
    """Get the view matrix for the camera."""
    return self._look_at(self.position, self.position + self.front, self.up)

  def _look_at(self, eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a look-at view matrix."""
    f = (center - eye) / np.linalg.norm(center - eye)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    result = np.identity(4)
    result[0, 0:3] = s
    result[1, 0:3] = u
    result[2, 0:3] = -f
    result[0:3, 3] = -np.array([
        np.dot(s, eye),
        np.dot(u, eye),
        np.dot(-f, eye)
    ])

    return result

  def process_keyboard(self, direction: str, delta_time: float):
    """Process keyboard input for camera movement."""
    velocity = self.movement_speed * delta_time

    if direction == "FORWARD":
      self.position += self.front * velocity
    if direction == "BACKWARD":
      self.position -= self.front * velocity
    if direction == "LEFT":
      self.position -= self.right * velocity
    if direction == "RIGHT":
      self.position += self.right * velocity

  def process_mouse_movement(self, xoffset: float, yoffset: float, constrain_pitch: bool = True):
    """Process mouse movement for camera rotation."""
    xoffset *= self.mouse_sensitivity
    yoffset *= self.mouse_sensitivity

    self.yaw += xoffset
    self.pitch += yoffset

    if constrain_pitch:
      self.pitch = np.clip(self.pitch, -89.0, 89.0)

    self._update_camera_vectors()

  def process_mouse_scroll(self, yoffset: float):
    """Process mouse scroll for camera zoom."""
    self.zoom -= yoffset
    self.zoom = np.clip(self.zoom, 1.0, 45.0)
