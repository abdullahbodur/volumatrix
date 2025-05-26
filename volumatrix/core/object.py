"""
VolumatrixObject: The main 3D object container for Volumatrix.

This module defines the VolumatrixObject class, which serves as the primary
interface for working with 3D objects in various representations.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field

from .representations import BaseRepresentation, Mesh, Voxel, PointCloud


class VolumatrixObject(BaseModel):
  """
  Main container for 3D objects in Volumatrix.

  A VolumatrixObject can contain multiple representations of the same 3D object
  (mesh, voxel, point cloud) and provides a unified interface for manipulation.
  """

  name: str = Field(default="VolumatrixObject",
                    description="Name of the object")
  representations: Dict[str, BaseRepresentation] = Field(
      default_factory=dict,
      description="Dictionary of different representations"
  )
  metadata: Dict[str, Any] = Field(
    default_factory=dict, description="Object metadata")

  class Config:
    arbitrary_types_allowed = True

  def __init__(self, **data):
    super().__init__(**data)
    if not self.representations:
      raise ValueError("VolumatrixObject must have at least one representation")

  @classmethod
  def from_mesh(
      cls,
      vertices: np.ndarray,
      faces: np.ndarray,
      name: str = "MeshObject",
      **kwargs
  ) -> "VolumatrixObject":
    """Create a VolumatrixObject from mesh data."""
    mesh = Mesh(vertices=vertices, faces=faces, **kwargs)
    return cls(name=name, representations={"mesh": mesh})

  @classmethod
  def from_voxel(
      cls,
      grid: np.ndarray,
      resolution: tuple,
      name: str = "VoxelObject",
      **kwargs
  ) -> "VolumatrixObject":
    """Create a VolumatrixObject from voxel data."""
    voxel = Voxel(grid=grid, resolution=resolution, **kwargs)
    return cls(name=name, representations={"voxel": voxel})

  @classmethod
  def from_pointcloud(
      cls,
      points: np.ndarray,
      name: str = "PointCloudObject",
      **kwargs
  ) -> "VolumatrixObject":
    """Create a VolumatrixObject from point cloud data."""
    pointcloud = PointCloud(points=points, **kwargs)
    return cls(name=name, representations={"pointcloud": pointcloud})

  def add_representation(self, name: str, representation: BaseRepresentation) -> None:
    """Add a new representation to the object."""
    self.representations[name] = representation

  def get_representation(self, name: str) -> Optional[BaseRepresentation]:
    """Get a specific representation by name."""
    return self.representations.get(name)

  def has_representation(self, name: str) -> bool:
    """Check if the object has a specific representation."""
    return name in self.representations

  def list_representations(self) -> List[str]:
    """List all available representation names."""
    return list(self.representations.keys())

  def remove_representation(self, name: str) -> bool:
    """Remove a representation. Returns True if removed, False if not found."""
    if name in self.representations:
      del self.representations[name]
      return True
    return False

  @property
  def primary_representation(self) -> BaseRepresentation:
    """Get the primary (first) representation."""
    if not self.representations:
      raise ValueError("No representations available")
    return next(iter(self.representations.values()))

  @property
  def mesh(self) -> Optional[Mesh]:
    """Get the mesh representation if available."""
    return self.get_representation("mesh")

  @property
  def voxel(self) -> Optional[Voxel]:
    """Get the voxel representation if available."""
    return self.get_representation("voxel")

  @property
  def pointcloud(self) -> Optional[PointCloud]:
    """Get the point cloud representation if available."""
    return self.get_representation("pointcloud")

  def bounds(self) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of the object using the primary representation."""
    return self.primary_representation.bounds()

  def center(self) -> np.ndarray:
    """Get the center point of the object using the primary representation."""
    return self.primary_representation.center()

  def transform(self, matrix: np.ndarray) -> "VolumatrixObject":
    """Apply a transformation matrix to all representations."""
    transformed_representations = {}
    for name, representation in self.representations.items():
      transformed_representations[name] = representation.transform(matrix)

    return VolumatrixObject(
        name=self.name,
        representations=transformed_representations,
        metadata=self.metadata.copy()
    )

  def translate(self, translation: np.ndarray) -> "VolumatrixObject":
    """Translate the object by the given vector."""
    if len(translation) != 3:
      raise ValueError("Translation must be a 3D vector")

    matrix = np.eye(4)
    matrix[:3, 3] = translation
    return self.transform(matrix)

  def rotate(self, rotation: np.ndarray, center: Optional[np.ndarray] = None) -> "VolumatrixObject":
    """Rotate the object using a 3x3 rotation matrix."""
    if rotation.shape != (3, 3):
      raise ValueError("Rotation must be a 3x3 matrix")

    if center is None:
      center = self.center()

    # Create 4x4 transformation matrix
    matrix = np.eye(4)
    matrix[:3, :3] = rotation

    # Apply rotation around center
    translate_to_origin = np.eye(4)
    translate_to_origin[:3, 3] = -center

    translate_back = np.eye(4)
    translate_back[:3, 3] = center

    final_matrix = translate_back @ matrix @ translate_to_origin
    return self.transform(final_matrix)

  def scale(self, scale_factor: Union[float, np.ndarray], center: Optional[np.ndarray] = None) -> "VolumatrixObject":
    """Scale the object by the given factor(s)."""
    if center is None:
      center = self.center()

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
    return self.transform(final_matrix)

  def copy(self) -> "VolumatrixObject":
    """Create a deep copy of the object."""
    copied_representations = {}
    for name, representation in self.representations.items():
      copied_representations[name] = representation.copy()

    return VolumatrixObject(
        name=self.name,
        representations=copied_representations,
        metadata=self.metadata.copy()
    )

  def merge(self, other: "VolumatrixObject") -> "VolumatrixObject":
    """Merge another VolumatrixObject with this one."""
    # This is a simplified merge - in practice, you'd need more sophisticated
    # logic for combining different representations
    merged_representations = self.representations.copy()

    for name, representation in other.representations.items():
      if name in merged_representations:
        # Handle conflicts by renaming
        counter = 1
        new_name = f"{name}_{counter}"
        while new_name in merged_representations:
          counter += 1
          new_name = f"{name}_{counter}"
        merged_representations[new_name] = representation.copy()
      else:
        merged_representations[name] = representation.copy()

    merged_metadata = self.metadata.copy()
    merged_metadata.update(other.metadata)

    return VolumatrixObject(
        name=f"{self.name}_merged_{other.name}",
        representations=merged_representations,
        metadata=merged_metadata
    )

  def __str__(self) -> str:
    """String representation of the object."""
    repr_info = []
    for name, representation in self.representations.items():
      if isinstance(representation, Mesh):
        repr_info.append(
          f"{name}: {representation.num_vertices} vertices, {representation.num_faces} faces")
      elif isinstance(representation, Voxel):
        repr_info.append(
          f"{name}: {representation.resolution} resolution, {representation.num_occupied} occupied")
      elif isinstance(representation, PointCloud):
        repr_info.append(f"{name}: {representation.num_points} points")
      else:
        repr_info.append(f"{name}: {type(representation).__name__}")

    return f"VolumatrixObject(name='{self.name}', representations=[{', '.join(repr_info)}])"

  def __repr__(self) -> str:
    """Detailed representation of the object."""
    return self.__str__()
