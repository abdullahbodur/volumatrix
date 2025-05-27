"""
Core 3D representations for Volumatrix.

This module defines the fundamental data structures for representing 3D objects:
- Mesh: Triangle-based surface representation
- Voxel: Volumetric grid representation  
- PointCloud: Point-based representation
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class BaseRepresentation(BaseModel, ABC):
  """Abstract base class for all 3D representations."""

  metadata: Dict[str, Any] = Field(default_factory=dict)

  class Config:
    arbitrary_types_allowed = True

  @abstractmethod
  def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return the bounding box as (min_coords, max_coords)."""
    pass

  @abstractmethod
  def center(self) -> np.ndarray:
    """Return the center point of the object."""
    pass

  @abstractmethod
  def transform(self, matrix: np.ndarray) -> "BaseRepresentation":
    """Apply a 4x4 transformation matrix to the representation."""
    pass

  @abstractmethod
  def copy(self) -> "BaseRepresentation":
    """Create a deep copy of the representation."""
    pass


class Mesh(BaseRepresentation):
  """Triangle mesh representation."""

  vertices: np.ndarray = Field(...,
                               description="Nx3 array of vertex coordinates")
  faces: np.ndarray = Field(...,
                            description="Mx3 array of triangle face indices")
  normals: Optional[np.ndarray] = Field(
    None, description="Nx3 array of vertex normals")
  colors: Optional[np.ndarray] = Field(
    None, description="Nx3 or Nx4 array of vertex colors")
  texcoords: Optional[np.ndarray] = Field(
    None, description="Nx2 array of texture coordinates")

  def __init__(self, **data):
    super().__init__(**data)
    self._validate_mesh()

  def _validate_mesh(self) -> None:
    """Validate mesh data consistency."""
    if self.vertices.shape[1] != 3:
      raise ValueError("Vertices must be Nx3 array")

    if self.faces.shape[1] != 3:
      raise ValueError("Faces must be Mx3 array")

    if np.max(self.faces) >= len(self.vertices):
      raise ValueError("Face indices exceed vertex count")

    if self.normals is not None and self.normals.shape != self.vertices.shape:
      raise ValueError("Normals must match vertex count and dimensions")

    if self.colors is not None:
      if self.colors.shape[0] != len(self.vertices):
        raise ValueError("Colors must match vertex count")
      if self.colors.shape[1] not in [3, 4]:
        raise ValueError("Colors must be RGB or RGBA")

  def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return the bounding box as (min_coords, max_coords)."""
    return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

  def center(self) -> np.ndarray:
    """Return the center point of the mesh."""
    min_coords, max_coords = self.bounds()
    return (min_coords + max_coords) / 2

  def transform(self, matrix: np.ndarray) -> "Mesh":
    """Apply a 4x4 transformation matrix to the mesh."""
    if matrix.shape != (4, 4):
      raise ValueError("Transformation matrix must be 4x4")

    # Transform vertices
    vertices_homo = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
    transformed_vertices = (matrix @ vertices_homo.T).T[:, :3]

    # Transform normals if present
    transformed_normals = None
    if self.normals is not None:
      # Use inverse transpose for normals
      normal_matrix = np.linalg.inv(matrix[:3, :3]).T
      transformed_normals = (normal_matrix @ self.normals.T).T
      # Renormalize
      norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
      transformed_normals = transformed_normals / (norms + 1e-8)

    return Mesh(
        vertices=transformed_vertices,
        faces=self.faces.copy(),
        normals=transformed_normals,
        colors=self.colors.copy() if self.colors is not None else None,
        texcoords=self.texcoords.copy() if self.texcoords is not None else None,
        metadata=self.metadata.copy()
    )

  def copy(self) -> "Mesh":
    """Create a deep copy of the mesh."""
    return Mesh(
        vertices=self.vertices.copy(),
        faces=self.faces.copy(),
        normals=self.normals.copy() if self.normals is not None else None,
        colors=self.colors.copy() if self.colors is not None else None,
        texcoords=self.texcoords.copy() if self.texcoords is not None else None,
        metadata=self.metadata.copy()
    )

  @property
  def num_vertices(self) -> int:
    """Number of vertices in the mesh."""
    return len(self.vertices)

  @property
  def num_faces(self) -> int:
    """Number of faces in the mesh."""
    return len(self.faces)

  def compute_normals(self) -> None:
    """Compute vertex normals from face geometry."""
    # Initialize vertex normals
    vertex_normals = np.zeros_like(self.vertices)

    # Compute face normals and accumulate to vertices
    for face in self.faces:
      v0, v1, v2 = self.vertices[face]
      normal = np.cross(v1 - v0, v2 - v0)
      normal = normal / (np.linalg.norm(normal) + 1e-8)

      vertex_normals[face] += normal

    # Normalize vertex normals
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    self.normals = vertex_normals / (norms + 1e-8)


class Voxel(BaseRepresentation):
  """Voxel grid representation."""

  grid: np.ndarray = Field(...,
                           description="3D boolean or float array representing occupancy")
  resolution: Tuple[int, int,
                    int] = Field(..., description="Grid resolution (x, y, z)")
  origin: np.ndarray = Field(default_factory=lambda: np.array([0.0, 0.0, 0.0]),
                             description="World coordinates of grid origin")
  spacing: Union[float, np.ndarray] = Field(default=1.0,
                                            description="Voxel spacing (uniform or per-axis)")

  def __init__(self, **data):
    super().__init__(**data)
    self._validate_voxel()

  def _validate_voxel(self) -> None:
    """Validate voxel data consistency."""
    if len(self.grid.shape) != 3:
      raise ValueError("Voxel grid must be 3D")

    if self.grid.shape != self.resolution:
      raise ValueError("Grid shape must match resolution")

    if isinstance(self.spacing, np.ndarray) and len(self.spacing) != 3:
      raise ValueError("Spacing array must have 3 elements")

  def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return the bounding box as (min_coords, max_coords)."""
    spacing_array = np.array(
      [self.spacing] * 3) if isinstance(self.spacing, float) else self.spacing
    max_coords = self.origin + np.array(self.resolution) * spacing_array
    return self.origin.copy(), max_coords

  def center(self) -> np.ndarray:
    """Return the center point of the voxel grid."""
    min_coords, max_coords = self.bounds()
    return (min_coords + max_coords) / 2

  def transform(self, matrix: np.ndarray) -> "Voxel":
    """Apply a 4x4 transformation matrix to the voxel grid."""
    # For voxels, we transform the origin and spacing
    # Note: This is a simplified transformation that doesn't handle rotation properly
    # A full implementation would need to resample the grid

    origin_homo = np.append(self.origin, 1.0)
    transformed_origin = (matrix @ origin_homo)[:3]

    # Extract scale from transformation matrix (simplified)
    scale = np.linalg.norm(matrix[:3, :3], axis=0)

    if isinstance(self.spacing, float):
      transformed_spacing = self.spacing * np.mean(scale)
    else:
      transformed_spacing = self.spacing * scale

    return Voxel(
        grid=self.grid.copy(),
        resolution=self.resolution,
        origin=transformed_origin,
        spacing=transformed_spacing,
        metadata=self.metadata.copy()
    )

  def copy(self) -> "Voxel":
    """Create a deep copy of the voxel grid."""
    return Voxel(
        grid=self.grid.copy(),
        resolution=self.resolution,
        origin=self.origin.copy(),
        spacing=self.spacing.copy() if isinstance(
          self.spacing, np.ndarray) else self.spacing,
        metadata=self.metadata.copy()
    )

  @property
  def num_voxels(self) -> int:
    """Total number of voxels in the grid."""
    return np.prod(self.resolution)

  @property
  def num_occupied(self) -> int:
    """Number of occupied voxels."""
    return np.sum(self.grid > 0)

  def get_occupied_coordinates(self) -> np.ndarray:
    """Get world coordinates of occupied voxels."""
    indices = np.where(self.grid > 0)
    spacing_array = np.array(
      [self.spacing] * 3) if isinstance(self.spacing, float) else self.spacing
    coordinates = np.column_stack(indices) * spacing_array + self.origin
    return coordinates


class PointCloud(BaseRepresentation):
  """Point cloud representation."""

  points: np.ndarray = Field(..., description="Nx3 array of point coordinates")
  colors: Optional[np.ndarray] = Field(
    None, description="Nx3 or Nx4 array of point colors")
  normals: Optional[np.ndarray] = Field(
    None, description="Nx3 array of point normals")
  intensities: Optional[np.ndarray] = Field(
    None, description="N array of point intensities")

  def __init__(self, **data):
    super().__init__(**data)
    self._validate_pointcloud()

  def _validate_pointcloud(self) -> None:
    """Validate point cloud data consistency."""
    if self.points.shape[1] != 3:
      raise ValueError("Points must be Nx3 array")

    num_points = len(self.points)

    if self.colors is not None:
      if self.colors.shape[0] != num_points:
        raise ValueError("Colors must match point count")
      if self.colors.shape[1] not in [3, 4]:
        raise ValueError("Colors must be RGB or RGBA")

    if self.normals is not None and self.normals.shape != self.points.shape:
      raise ValueError("Normals must match point count and dimensions")

    if self.intensities is not None and len(self.intensities) != num_points:
      raise ValueError("Intensities must match point count")

  def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return the bounding box as (min_coords, max_coords)."""
    return np.min(self.points, axis=0), np.max(self.points, axis=0)

  def center(self) -> np.ndarray:
    """Return the center point of the point cloud."""
    return np.mean(self.points, axis=0)

  def transform(self, matrix: np.ndarray) -> "PointCloud":
    """Apply a 4x4 transformation matrix to the point cloud."""
    if matrix.shape != (4, 4):
      raise ValueError("Transformation matrix must be 4x4")

    # Transform points
    points_homo = np.hstack([self.points, np.ones((len(self.points), 1))])
    transformed_points = (matrix @ points_homo.T).T[:, :3]

    # Transform normals if present
    transformed_normals = None
    if self.normals is not None:
      normal_matrix = np.linalg.inv(matrix[:3, :3]).T
      transformed_normals = (normal_matrix @ self.normals.T).T
      norms = np.linalg.norm(transformed_normals, axis=1, keepdims=True)
      transformed_normals = transformed_normals / (norms + 1e-8)

    return PointCloud(
        points=transformed_points,
        colors=self.colors.copy() if self.colors is not None else None,
        normals=transformed_normals,
        intensities=self.intensities.copy() if self.intensities is not None else None,
        metadata=self.metadata.copy()
    )

  def copy(self) -> "PointCloud":
    """Create a deep copy of the point cloud."""
    return PointCloud(
        points=self.points.copy(),
        colors=self.colors.copy() if self.colors is not None else None,
        normals=self.normals.copy() if self.normals is not None else None,
        intensities=self.intensities.copy() if self.intensities is not None else None,
        metadata=self.metadata.copy()
    )

  @property
  def num_points(self) -> int:
    """Number of points in the cloud."""
    return len(self.points)

  def subsample(self, num_points: int, method: str = "random") -> "PointCloud":
    """Subsample the point cloud to a target number of points."""
    if num_points >= self.num_points:
      return self.copy()

    if method == "random":
      indices = np.random.choice(self.num_points, num_points, replace=False)
    elif method == "uniform":
      indices = np.linspace(0, self.num_points - 1, num_points, dtype=int)
    else:
      raise ValueError(f"Unknown subsampling method: {method}")

    return PointCloud(
        points=self.points[indices],
        colors=self.colors[indices] if self.colors is not None else None,
        normals=self.normals[indices] if self.normals is not None else None,
        intensities=self.intensities[indices] if self.intensities is not None else None,
        metadata=self.metadata.copy()
    )
