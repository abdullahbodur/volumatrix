"""
Transformation utilities for Volumatrix.

This module provides functions for normalizing, rescaling, and rotating 3D objects.
"""

from typing import Optional, Union
import numpy as np

from ..core.object import VolumatrixObject


def normalize(obj: VolumatrixObject, center: bool = True, scale: bool = True) -> VolumatrixObject:
  """
  Normalize a 3D object to fit within a unit cube.

  Args:
      obj: The VolumatrixObject to normalize
      center: Whether to center the object at the origin
      scale: Whether to scale the object to fit in a unit cube

  Returns:
      A new normalized VolumatrixObject

  Examples:
      >>> normalized_obj = normalize(obj)
      >>> centered_obj = normalize(obj, center=True, scale=False)
  """
  # Get current bounds
  min_coords, max_coords = obj.bounds()
  current_center = (min_coords + max_coords) / 2
  current_size = max_coords - min_coords
  max_dimension = np.max(current_size)

  # Build transformation matrix
  transform = np.eye(4)

  if center:
    # Translate to center at origin
    transform[:3, 3] = -current_center

  if scale and max_dimension > 0:
    # Scale to fit in unit cube
    scale_factor = 2.0 / max_dimension  # Scale to [-1, 1] range
    scale_matrix = np.eye(4)
    scale_matrix[:3, :3] *= scale_factor
    transform = scale_matrix @ transform

  return obj.transform(transform)


def rescale(obj: VolumatrixObject, scale_factor: Union[float, np.ndarray],
            center: Optional[np.ndarray] = None) -> VolumatrixObject:
  """
  Rescale a 3D object by the given factor(s).

  Args:
      obj: The VolumatrixObject to rescale
      scale_factor: Scale factor (uniform) or scale vector [sx, sy, sz]
      center: Center point for scaling (uses object center if None)

  Returns:
      A new rescaled VolumatrixObject

  Examples:
      >>> larger_obj = rescale(obj, 2.0)
      >>> stretched_obj = rescale(obj, [2.0, 1.0, 0.5])
  """
  return obj.scale(scale_factor, center)


def rotate(obj: VolumatrixObject, rotation: Union[np.ndarray, list],
           center: Optional[np.ndarray] = None, degrees: bool = False) -> VolumatrixObject:
  """
  Rotate a 3D object.

  Args:
      obj: The VolumatrixObject to rotate
      rotation: Rotation as 3x3 matrix or Euler angles [rx, ry, rz]
      center: Center point for rotation (uses object center if None)
      degrees: Whether Euler angles are in degrees (default: radians)

  Returns:
      A new rotated VolumatrixObject

  Examples:
      >>> rotated_obj = rotate(obj, [0, np.pi/4, 0])  # Rotate 45° around Y
      >>> rotated_obj = rotate(obj, [0, 45, 0], degrees=True)
      >>> rotated_obj = rotate(obj, rotation_matrix)
  """
  if isinstance(rotation, (list, tuple)) or (isinstance(rotation, np.ndarray) and rotation.shape == (3,)):
    # Convert Euler angles to rotation matrix
    rotation = np.array(rotation)
    if degrees:
      rotation = np.radians(rotation)

    rx, ry, rz = rotation

    # Create rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])

    rotation_matrix = Rz @ Ry @ Rx
  else:
    rotation_matrix = np.array(rotation)
    if rotation_matrix.shape != (3, 3):
      raise ValueError("Rotation matrix must be 3x3")

  return obj.rotate(rotation_matrix, center)


def translate(obj: VolumatrixObject, translation: Union[np.ndarray, list]) -> VolumatrixObject:
  """
  Translate a 3D object.

  Args:
      obj: The VolumatrixObject to translate
      translation: Translation vector [dx, dy, dz]

  Returns:
      A new translated VolumatrixObject

  Examples:
      >>> moved_obj = translate(obj, [1.0, 0.0, 0.5])
  """
  translation = np.array(translation)
  return obj.translate(translation)


def align_to_axes(obj: VolumatrixObject, method: str = "pca") -> VolumatrixObject:
  """
  Align object to coordinate axes using principal component analysis.

  Args:
      obj: The VolumatrixObject to align
      method: Alignment method ("pca" for principal component analysis)

  Returns:
      A new aligned VolumatrixObject

  Examples:
      >>> aligned_obj = align_to_axes(obj)
  """
  if method != "pca":
    raise ValueError("Only 'pca' method is currently supported")

  # Get vertices from primary representation
  primary_repr = obj.primary_representation

  if hasattr(primary_repr, 'vertices'):
    vertices = primary_repr.vertices
  elif hasattr(primary_repr, 'points'):
    vertices = primary_repr.points
  elif hasattr(primary_repr, 'get_occupied_coordinates'):
    vertices = primary_repr.get_occupied_coordinates()
  else:
    raise ValueError("Cannot extract vertices from object representation")

  # Center the vertices
  center = np.mean(vertices, axis=0)
  centered_vertices = vertices - center

  # Compute PCA
  covariance_matrix = np.cov(centered_vertices.T)
  eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

  # Sort by eigenvalues (largest first)
  idx = np.argsort(eigenvalues)[::-1]
  eigenvectors = eigenvectors[:, idx]

  # Ensure right-handed coordinate system
  if np.linalg.det(eigenvectors) < 0:
    eigenvectors[:, -1] *= -1

  # Create transformation matrix
  transform = np.eye(4)
  transform[:3, :3] = eigenvectors.T  # Transpose to align with axes
  transform[:3, 3] = -eigenvectors.T @ center  # Apply centering

  return obj.transform(transform)


def fit_in_box(obj: VolumatrixObject, box_size: Union[float, np.ndarray],
               center: bool = True) -> VolumatrixObject:
  """
  Scale and optionally center an object to fit within a specified box.

  Args:
      obj: The VolumatrixObject to fit
      box_size: Size of the target box (uniform or [width, height, depth])
      center: Whether to center the object

  Returns:
      A new fitted VolumatrixObject

  Examples:
      >>> fitted_obj = fit_in_box(obj, 2.0)  # Fit in 2x2x2 box
      >>> fitted_obj = fit_in_box(obj, [4, 2, 1])  # Fit in 4x2x1 box
  """
  if isinstance(box_size, (int, float)):
    box_size = np.array([box_size, box_size, box_size])
  else:
    box_size = np.array(box_size)

  # Get current bounds
  min_coords, max_coords = obj.bounds()
  current_size = max_coords - min_coords

  # Calculate scale factors for each dimension
  # Add small epsilon to avoid division by zero
  scale_factors = box_size / (current_size + 1e-8)

  # Use the minimum scale factor to ensure object fits in all dimensions
  uniform_scale = np.min(scale_factors)

  # Apply transformations
  result = obj

  if center:
    # Center at origin first
    current_center = (min_coords + max_coords) / 2
    result = result.translate(-current_center)

  # Scale to fit
  result = result.scale(uniform_scale)

  return result
