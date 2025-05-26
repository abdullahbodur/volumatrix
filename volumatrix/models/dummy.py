"""
Dummy model for testing and demonstration.

This module provides a simple dummy model that generates basic geometric shapes
for testing the Volumatrix framework without requiring actual AI models.
"""

from typing import Optional
import numpy as np

from .base import BaseModel, ModelConfig
from ..core.object import VolumatrixObject
from ..core.representations import Mesh, Voxel, PointCloud


class DummyModel(BaseModel):
  """
  A dummy model that generates simple geometric shapes.

  This model is useful for testing and demonstration purposes.
  It generates basic shapes like cubes, spheres, and cylinders
  based on keywords in the prompt.
  """

  def __init__(self, **kwargs):
    config = ModelConfig(
        name="DummyModel",
        description="A dummy model that generates simple geometric shapes",
        version="1.0.0",
        supported_formats=["mesh", "voxel", "pointcloud"],
        max_resolution=256,
        requires_gpu=False,
        model_type="geometric"
    )
    super().__init__(config, **kwargs)

  def _setup(self, **kwargs) -> None:
    """Setup the dummy model."""
    self.is_loaded = True

  def generate(
      self,
      prompt: str,
      seed: Optional[int] = None,
      resolution: int = 64,
      output_format: str = "mesh",
      **kwargs
  ) -> VolumatrixObject:
    """
    Generate a simple geometric shape based on the prompt.

    Args:
        prompt: Text description (keywords: cube, sphere, cylinder, chair, table)
        seed: Random seed for reproducible generation
        resolution: Output resolution for voxel-based models
        output_format: Preferred output format
        **kwargs: Additional parameters

    Returns:
        A VolumatrixObject containing the generated shape
    """
    # Validate parameters
    validated_params = self.validate_parameters(
        prompt=prompt,
        seed=seed,
        resolution=resolution,
        output_format=output_format,
        **kwargs
    )

    # Set random seed if provided
    if seed is not None:
      np.random.seed(seed)

    # Determine shape based on prompt keywords
    prompt_lower = prompt.lower()

    if "sphere" in prompt_lower or "ball" in prompt_lower:
      shape_type = "sphere"
    elif "cylinder" in prompt_lower or "tube" in prompt_lower:
      shape_type = "cylinder"
    elif "chair" in prompt_lower:
      shape_type = "chair"
    elif "table" in prompt_lower:
      shape_type = "table"
    else:
      shape_type = "cube"  # Default

    # Generate the shape
    if output_format == "mesh":
      representation = self._generate_mesh(shape_type)
    elif output_format == "voxel":
      representation = self._generate_voxel(shape_type, resolution)
    elif output_format == "pointcloud":
      representation = self._generate_pointcloud(shape_type)
    else:
      raise ValueError(f"Unsupported output format: {output_format}")

    # Create VolumatrixObject
    obj = VolumatrixObject(
        name=f"Dummy_{shape_type}",
        representations={output_format: representation}
    )

    return obj

  def _generate_mesh(self, shape_type: str) -> Mesh:
    """Generate a mesh for the given shape type."""
    if shape_type == "cube":
      return self._create_cube_mesh()
    elif shape_type == "sphere":
      return self._create_sphere_mesh()
    elif shape_type == "cylinder":
      return self._create_cylinder_mesh()
    elif shape_type == "chair":
      return self._create_chair_mesh()
    elif shape_type == "table":
      return self._create_table_mesh()
    else:
      return self._create_cube_mesh()

  def _generate_voxel(self, shape_type: str, resolution: int) -> Voxel:
    """Generate a voxel grid for the given shape type."""
    grid = np.zeros((resolution, resolution, resolution), dtype=bool)

    center = resolution // 2
    radius = resolution // 4

    if shape_type == "sphere":
      # Create sphere
      for x in range(resolution):
        for y in range(resolution):
          for z in range(resolution):
            dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
            if dist <= radius:
              grid[x, y, z] = True
    elif shape_type == "cylinder":
      # Create cylinder
      for x in range(resolution):
        for y in range(resolution):
          for z in range(resolution):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist <= radius and abs(z - center) <= radius:
              grid[x, y, z] = True
    else:
      # Create cube (default)
      start = center - radius
      end = center + radius
      grid[start:end, start:end, start:end] = True

    return Voxel(
        grid=grid,
        resolution=(resolution, resolution, resolution),
        origin=np.array([-1.0, -1.0, -1.0]),
        spacing=2.0 / resolution
    )

  def _generate_pointcloud(self, shape_type: str, num_points: int = 1000) -> PointCloud:
    """Generate a point cloud for the given shape type."""
    if shape_type == "sphere":
      # Generate points on sphere surface
      phi = np.random.uniform(0, 2 * np.pi, num_points)
      costheta = np.random.uniform(-1, 1, num_points)
      theta = np.arccos(costheta)

      x = np.sin(theta) * np.cos(phi)
      y = np.sin(theta) * np.sin(phi)
      z = np.cos(theta)

      points = np.column_stack([x, y, z])
    elif shape_type == "cylinder":
      # Generate points on cylinder surface
      theta = np.random.uniform(0, 2 * np.pi, num_points)
      z = np.random.uniform(-1, 1, num_points)

      x = np.cos(theta)
      y = np.sin(theta)

      points = np.column_stack([x, y, z])
    else:
      # Generate points on cube surface (default)
      points = np.random.uniform(-1, 1, (num_points, 3))

      # Project to cube surface
      abs_coords = np.abs(points)
      max_coord = np.max(abs_coords, axis=1, keepdims=True)
      points = points / max_coord

    # Add some color variation
    colors = np.random.uniform(0.5, 1.0, (num_points, 3))

    return PointCloud(points=points, colors=colors)

  def _create_cube_mesh(self) -> Mesh:
    """Create a simple cube mesh."""
    # Cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
    ], dtype=np.float32)

    # Cube faces (triangles)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 7, 6], [4, 6, 5],  # Top
        [0, 4, 5], [0, 5, 1],  # Front
        [2, 6, 7], [2, 7, 3],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 5, 6], [1, 6, 2]   # Right
    ], dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces)

  def _create_sphere_mesh(self, subdivisions: int = 2) -> Mesh:
    """Create a simple sphere mesh using icosphere subdivision."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=np.float32)

    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int32)

    return Mesh(vertices=vertices, faces=faces)

  def _create_cylinder_mesh(self, segments: int = 16) -> Mesh:
    """Create a simple cylinder mesh."""
    vertices = []
    faces = []

    # Create vertices
    for i in range(segments):
      angle = 2 * np.pi * i / segments
      x = np.cos(angle)
      y = np.sin(angle)

      # Bottom circle
      vertices.append([x, y, -1])
      # Top circle
      vertices.append([x, y, 1])

    # Add center vertices
    vertices.append([0, 0, -1])  # Bottom center
    vertices.append([0, 0, 1])   # Top center

    vertices = np.array(vertices, dtype=np.float32)

    # Create faces
    bottom_center = len(vertices) - 2
    top_center = len(vertices) - 1

    for i in range(segments):
      next_i = (i + 1) % segments

      # Bottom face
      faces.append([bottom_center, 2 * i, 2 * next_i])

      # Top face
      faces.append([top_center, 2 * next_i + 1, 2 * i + 1])

      # Side faces
      faces.append([2 * i, 2 * i + 1, 2 * next_i + 1])
      faces.append([2 * i, 2 * next_i + 1, 2 * next_i])

    faces = np.array(faces, dtype=np.int32)
    return Mesh(vertices=vertices, faces=faces)

  def _create_chair_mesh(self) -> Mesh:
    """Create a simple chair mesh (combination of cubes)."""
    # This is a simplified chair made of rectangular parts
    vertices = []
    faces = []

    # Seat (flat rectangle)
    seat_verts = np.array([
        [-0.8, -0.8, 0.4], [0.8, -0.8, 0.4], [0.8, 0.8, 0.4], [-0.8, 0.8, 0.4],
        [-0.8, -0.8, 0.5], [0.8, -0.8, 0.5], [0.8, 0.8, 0.5], [-0.8, 0.8, 0.5]
    ])

    # Backrest
    back_verts = np.array([
        [-0.8, 0.7, 0.5], [0.8, 0.7, 0.5], [0.8, 0.8, 0.5], [-0.8, 0.8, 0.5],
        [-0.8, 0.7, 1.5], [0.8, 0.7, 1.5], [0.8, 0.8, 1.5], [-0.8, 0.8, 1.5]
    ])

    # Combine vertices
    vertices = np.vstack([seat_verts, back_verts])

    # Create faces for both parts
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ])

    # Seat faces
    seat_faces = cube_faces.copy()

    # Backrest faces (offset by 8 vertices)
    back_faces = cube_faces + 8

    faces = np.vstack([seat_faces, back_faces])

    return Mesh(vertices=vertices, faces=faces)

  def _create_table_mesh(self) -> Mesh:
    """Create a simple table mesh."""
    # Table top
    top_verts = np.array([
        [-1.5, -1.0, 0.7], [1.5, -1.0, 0.7], [1.5, 1.0, 0.7], [-1.5, 1.0, 0.7],
        [-1.5, -1.0, 0.8], [1.5, -1.0, 0.8], [1.5, 1.0, 0.8], [-1.5, 1.0, 0.8]
    ])

    # Table legs (simplified as one block for now)
    leg_verts = np.array([
        [-1.3, -0.8, 0.0], [1.3, -0.8, 0.0], [1.3, 0.8, 0.0], [-1.3, 0.8, 0.0],
        [-1.3, -0.8, 0.7], [1.3, -0.8, 0.7], [1.3, 0.8, 0.7], [-1.3, 0.8, 0.7]
    ])

    vertices = np.vstack([top_verts, leg_verts])

    # Create faces
    cube_faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ])

    top_faces = cube_faces.copy()
    leg_faces = cube_faces + 8

    faces = np.vstack([top_faces, leg_faces])

    return Mesh(vertices=vertices, faces=faces)
