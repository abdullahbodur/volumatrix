"""
Scene management for Volumatrix.

This module defines the Scene class for composing and managing multiple
VolumatrixObjects with transformations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

from .object import VolumatrixObject


class SceneNode(BaseModel):
  """A node in the scene graph containing an object and its transformation."""

  object: VolumatrixObject = Field(..., description="The 3D object")
  transform: np.ndarray = Field(
      default_factory=lambda: np.eye(4),
      description="4x4 transformation matrix"
  )
  visible: bool = Field(
    default=True, description="Whether the object is visible")
  metadata: Dict[str, Any] = Field(
    default_factory=dict, description="Node metadata")

  class Config:
    arbitrary_types_allowed = True

  def get_transformed_object(self) -> VolumatrixObject:
    """Get the object with the node's transformation applied."""
    if np.allclose(self.transform, np.eye(4)):
      return self.object
    return self.object.transform(self.transform)


class Scene(BaseModel):
  """
  Scene container for managing multiple 3D objects.

  A Scene holds multiple VolumatrixObjects with their transformations,
  providing functionality for composition, manipulation, and export.
  """

  name: str = Field(default="Scene", description="Name of the scene")
  nodes: Dict[str, SceneNode] = Field(
    default_factory=dict, description="Scene nodes")
  metadata: Dict[str, Any] = Field(
    default_factory=dict, description="Scene metadata")

  class Config:
    arbitrary_types_allowed = True

  def add(
      self,
      obj: VolumatrixObject,
      name: Optional[str] = None,
      position: Optional[Union[List[float], np.ndarray]] = None,
      rotation: Optional[Union[List[float], np.ndarray]] = None,
      scale: Optional[Union[float, List[float], np.ndarray]] = None,
      transform: Optional[np.ndarray] = None,
      **kwargs
  ) -> str:
    """
    Add an object to the scene.

    Args:
        obj: The VolumatrixObject to add
        name: Name for the object in the scene (auto-generated if None)
        position: Translation vector [x, y, z]
        rotation: Rotation as Euler angles [rx, ry, rz] in radians
        scale: Scale factor (uniform) or scale vector [sx, sy, sz]
        transform: Direct 4x4 transformation matrix (overrides other transforms)
        **kwargs: Additional metadata for the scene node

    Returns:
        The name assigned to the object in the scene
    """
    if name is None:
      name = obj.name
      counter = 1
      while name in self.nodes:
        name = f"{obj.name}_{counter}"
        counter += 1

    if name in self.nodes:
      raise ValueError(f"Object with name '{name}' already exists in scene")

    # Build transformation matrix
    if transform is not None:
      if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")
      final_transform = transform.copy()
    else:
      final_transform = np.eye(4)

      # Apply scale
      if scale is not None:
        if isinstance(scale, (int, float)):
          scale_matrix = np.diag([scale, scale, scale, 1.0])
        else:
          if len(scale) != 3:
            raise ValueError("Scale must be a scalar or 3-element vector")
          scale_matrix = np.diag([*scale, 1.0])
        final_transform = final_transform @ scale_matrix

      # Apply rotation (Euler angles)
      if rotation is not None:
        if len(rotation) != 3:
          raise ValueError("Rotation must be a 3-element vector (Euler angles)")
        rx, ry, rz = rotation

        # Create rotation matrices
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(rx), -np.sin(rx), 0],
            [0, np.sin(rx), np.cos(rx), 0],
            [0, 0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry), 0],
            [0, 1, 0, 0],
            [-np.sin(ry), 0, np.cos(ry), 0],
            [0, 0, 0, 1]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0, 0],
            [np.sin(rz), np.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        rotation_matrix = Rz @ Ry @ Rx
        final_transform = final_transform @ rotation_matrix

      # Apply translation
      if position is not None:
        if len(position) != 3:
          raise ValueError("Position must be a 3-element vector")
        final_transform[:3, 3] = position

    # Create scene node
    node = SceneNode(
        object=obj,
        transform=final_transform,
        metadata=kwargs
    )

    self.nodes[name] = node
    return name

  def remove(self, name: str) -> bool:
    """Remove an object from the scene. Returns True if removed, False if not found."""
    if name in self.nodes:
      del self.nodes[name]
      return True
    return False

  def get(self, name: str) -> Optional[SceneNode]:
    """Get a scene node by name."""
    return self.nodes.get(name)

  def get_object(self, name: str) -> Optional[VolumatrixObject]:
    """Get an object by name."""
    node = self.get(name)
    return node.object if node else None

  def get_transformed_object(self, name: str) -> Optional[VolumatrixObject]:
    """Get an object with its scene transformation applied."""
    node = self.get(name)
    return node.get_transformed_object() if node else None

  def list_objects(self) -> List[str]:
    """List all object names in the scene."""
    return list(self.nodes.keys())

  def set_visibility(self, name: str, visible: bool) -> bool:
    """Set the visibility of an object. Returns True if successful, False if not found."""
    if name in self.nodes:
      self.nodes[name].visible = visible
      return True
    return False

  def set_transform(self, name: str, transform: np.ndarray) -> bool:
    """Set the transformation matrix for an object."""
    if name in self.nodes:
      if transform.shape != (4, 4):
        raise ValueError("Transform must be a 4x4 matrix")
      self.nodes[name].transform = transform.copy()
      return True
    return False

  def translate(self, name: str, translation: np.ndarray) -> bool:
    """Translate an object in the scene."""
    if name not in self.nodes:
      return False

    if len(translation) != 3:
      raise ValueError("Translation must be a 3D vector")

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    self.nodes[name].transform = translation_matrix @ self.nodes[name].transform
    return True

  def rotate(self, name: str, rotation: np.ndarray, center: Optional[np.ndarray] = None) -> bool:
    """Rotate an object in the scene."""
    if name not in self.nodes:
      return False

    if rotation.shape != (3, 3):
      raise ValueError("Rotation must be a 3x3 matrix")

    if center is None:
      # Use object center
      obj = self.get_transformed_object(name)
      center = obj.center()

    # Create rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation

    # Apply rotation around center
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -center

    translate_back = np.eye(4)
    translate_back[:3, 3] = center

    final_matrix = translate_back @ rotation_matrix @ translate_to_origin
    self.nodes[name].transform = final_matrix @ self.nodes[name].transform
    return True

  def scale(self, name: str, scale_factor: Union[float, np.ndarray], center: Optional[np.ndarray] = None) -> bool:
    """Scale an object in the scene."""
    if name not in self.nodes:
      return False

    if center is None:
      # Use object center
      obj = self.get_transformed_object(name)
      center = obj.center()

    if isinstance(scale_factor, (int, float)):
      scale_matrix = np.eye(3) * scale_factor
    else:
      if len(scale_factor) != 3:
        raise ValueError("Scale factor must be a scalar or 3D vector")
      scale_matrix = np.diag(scale_factor)

    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = scale_matrix

    # Apply scaling around center
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -center

    translate_back = np.eye(4)
    translate_back[:3, 3] = center

    final_matrix = translate_back @ matrix @ translate_to_origin
    self.nodes[name].transform = final_matrix @ self.nodes[name].transform
    return True

  def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of the entire scene."""
    if not self.nodes:
      return np.zeros(3), np.zeros(3)

    all_bounds = []
    for node in self.nodes.values():
      if node.visible:
        obj = node.get_transformed_object()
        min_coords, max_coords = obj.bounds()
        all_bounds.extend([min_coords, max_coords])

    if not all_bounds:
      return np.zeros(3), np.zeros(3)

    all_bounds = np.array(all_bounds)
    return np.min(all_bounds, axis=0), np.max(all_bounds, axis=0)

  def center(self) -> np.ndarray:
    """Get the center point of the scene."""
    min_coords, max_coords = self.bounds()
    return (min_coords + max_coords) / 2

  def merge_objects(self, names: Optional[List[str]] = None) -> VolumatrixObject:
    """
    Merge multiple objects into a single VolumatrixObject.

    Args:
        names: List of object names to merge. If None, merges all visible objects.

    Returns:
        A new VolumatrixObject containing the merged geometry.
    """
    if names is None:
      names = [name for name, node in self.nodes.items() if node.visible]

    if not names:
      raise ValueError("No objects to merge")

    # Start with the first object
    merged_obj = self.get_transformed_object(names[0])
    if merged_obj is None:
      raise ValueError(f"Object '{names[0]}' not found")

    merged_obj = merged_obj.copy()

    # Merge remaining objects
    for name in names[1:]:
      obj = self.get_transformed_object(name)
      if obj is None:
        raise ValueError(f"Object '{name}' not found")
      merged_obj = merged_obj.merge(obj)

    merged_obj.name = f"{self.name}_merged"
    return merged_obj

  def copy(self) -> "Scene":
    """Create a deep copy of the scene."""
    copied_nodes = {}
    for name, node in self.nodes.items():
      copied_nodes[name] = SceneNode(
          object=node.object.copy(),
          transform=node.transform.copy(),
          visible=node.visible,
          metadata=node.metadata.copy()
      )

    return Scene(
        name=self.name,
        nodes=copied_nodes,
        metadata=self.metadata.copy()
    )

  def export(self, filepath: str, format: Optional[str] = None, **kwargs) -> None:
    """
    Export the scene to a file.

    Args:
        filepath: Output file path
        format: Export format (inferred from extension if None)
        **kwargs: Additional export options
    """
    # This will be implemented in the export module
    from ..api.io import export

    # Merge all visible objects for export
    merged_obj = self.merge_objects()
    export(merged_obj, filepath, format=format, **kwargs)

  def __len__(self) -> int:
    """Number of objects in the scene."""
    return len(self.nodes)

  def __contains__(self, name: str) -> bool:
    """Check if an object exists in the scene."""
    return name in self.nodes

  def __iter__(self):
    """Iterate over object names in the scene."""
    return iter(self.nodes.keys())

  def __str__(self) -> str:
    """String representation of the scene."""
    visible_count = sum(1 for node in self.nodes.values() if node.visible)
    return f"Scene(name='{self.name}', objects={len(self.nodes)}, visible={visible_count})"

  def __repr__(self) -> str:
    """Detailed representation of the scene."""
    return self.__str__()
