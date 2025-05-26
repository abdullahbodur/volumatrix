"""
Tests for core Volumatrix functionality.
"""

import pytest
import numpy as np

from volumatrix.core.representations import Mesh, Voxel, PointCloud
from volumatrix.core.object import VolumatrixObject
from volumatrix.core.scene import Scene


class TestMesh:
  """Test the Mesh representation."""

  def test_create_simple_mesh(self):
    """Test creating a simple triangle mesh."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    mesh = Mesh(vertices=vertices, faces=faces)

    assert mesh.num_vertices == 3
    assert mesh.num_faces == 1
    assert np.array_equal(mesh.vertices, vertices)
    assert np.array_equal(mesh.faces, faces)

  def test_mesh_bounds(self):
    """Test mesh bounding box calculation."""
    vertices = np.array([
        [-1.0, -2.0, -3.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    mesh = Mesh(vertices=vertices, faces=faces)
    min_coords, max_coords = mesh.bounds()

    np.testing.assert_array_equal(min_coords, [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(max_coords, [1.0, 2.0, 3.0])

  def test_mesh_center(self):
    """Test mesh center calculation."""
    vertices = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    mesh = Mesh(vertices=vertices, faces=faces)
    center = mesh.center()

    np.testing.assert_array_equal(center, [0.0, 0.0, 0.0])

  def test_mesh_transform(self):
    """Test mesh transformation."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    mesh = Mesh(vertices=vertices, faces=faces)

    # Translation matrix
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]

    transformed_mesh = mesh.transform(transform)

    expected_vertices = vertices + [1.0, 2.0, 3.0]
    np.testing.assert_array_almost_equal(
      transformed_mesh.vertices, expected_vertices)

  def test_compute_normals(self):
    """Test normal computation."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    mesh = Mesh(vertices=vertices, faces=faces)
    mesh.compute_normals()

    assert mesh.normals is not None
    assert mesh.normals.shape == (3, 3)


class TestVoxel:
  """Test the Voxel representation."""

  def test_create_voxel_grid(self):
    """Test creating a voxel grid."""
    resolution = (8, 8, 8)
    grid = np.zeros(resolution, dtype=bool)
    grid[2:6, 2:6, 2:6] = True  # Create a cube in the center

    voxel = Voxel(grid=grid, resolution=resolution)

    assert voxel.resolution == resolution
    assert voxel.num_voxels == 8 * 8 * 8
    assert voxel.num_occupied == 4 * 4 * 4

  def test_voxel_bounds(self):
    """Test voxel bounding box calculation."""
    resolution = (4, 4, 4)
    grid = np.ones(resolution, dtype=bool)
    origin = np.array([1.0, 2.0, 3.0])
    spacing = 0.5

    voxel = Voxel(grid=grid, resolution=resolution,
                  origin=origin, spacing=spacing)
    min_coords, max_coords = voxel.bounds()

    expected_max = origin + np.array(resolution) * spacing
    np.testing.assert_array_equal(min_coords, origin)
    np.testing.assert_array_equal(max_coords, expected_max)

  def test_get_occupied_coordinates(self):
    """Test getting occupied voxel coordinates."""
    resolution = (3, 3, 3)
    grid = np.zeros(resolution, dtype=bool)
    grid[1, 1, 1] = True  # Single occupied voxel

    voxel = Voxel(grid=grid, resolution=resolution, spacing=1.0)
    coords = voxel.get_occupied_coordinates()

    assert len(coords) == 1
    np.testing.assert_array_equal(coords[0], [1.0, 1.0, 1.0])


class TestPointCloud:
  """Test the PointCloud representation."""

  def test_create_pointcloud(self):
    """Test creating a point cloud."""
    points = np.random.rand(100, 3)
    colors = np.random.rand(100, 3)

    pc = PointCloud(points=points, colors=colors)

    assert pc.num_points == 100
    assert np.array_equal(pc.points, points)
    assert np.array_equal(pc.colors, colors)

  def test_pointcloud_bounds(self):
    """Test point cloud bounding box calculation."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-1.0, -2.0, -3.0]
    ])

    pc = PointCloud(points=points)
    min_coords, max_coords = pc.bounds()

    np.testing.assert_array_equal(min_coords, [-1.0, -2.0, -3.0])
    np.testing.assert_array_equal(max_coords, [1.0, 2.0, 3.0])

  def test_pointcloud_subsample(self):
    """Test point cloud subsampling."""
    points = np.random.rand(100, 3)
    pc = PointCloud(points=points)

    subsampled = pc.subsample(50, method="random")

    assert subsampled.num_points == 50
    assert subsampled.points.shape == (50, 3)


class TestVolumatrixObject:
  """Test the VolumatrixObject class."""

  def test_create_from_mesh(self):
    """Test creating object from mesh data."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    obj = VolumatrixObject.from_mesh(vertices, faces, name="TestTriangle")

    assert obj.name == "TestTriangle"
    assert obj.has_representation("mesh")
    assert obj.mesh is not None
    assert obj.mesh.num_vertices == 3

  def test_create_from_pointcloud(self):
    """Test creating object from point cloud data."""
    points = np.random.rand(50, 3)

    obj = VolumatrixObject.from_pointcloud(points, name="TestPC")

    assert obj.name == "TestPC"
    assert obj.has_representation("pointcloud")
    assert obj.pointcloud is not None
    assert obj.pointcloud.num_points == 50

  def test_object_transform(self):
    """Test object transformation."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    obj = VolumatrixObject.from_mesh(vertices, faces)

    # Test translation
    translated = obj.translate([1.0, 2.0, 3.0])

    assert translated.name == obj.name
    assert translated.mesh is not None
    expected_vertices = vertices + [1.0, 2.0, 3.0]
    np.testing.assert_array_almost_equal(
      translated.mesh.vertices, expected_vertices)

  def test_object_merge(self):
    """Test merging objects."""
    # Create two simple objects
    vertices1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    faces1 = np.array([[0, 1, 2]])
    obj1 = VolumatrixObject.from_mesh(vertices1, faces1, name="Obj1")

    vertices2 = np.array([[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.5, 1.0, 0.0]])
    faces2 = np.array([[0, 1, 2]])
    obj2 = VolumatrixObject.from_mesh(vertices2, faces2, name="Obj2")

    merged = obj1.merge(obj2)

    assert "merged" in merged.name.lower()
    assert len(merged.representations) >= 1


class TestScene:
  """Test the Scene class."""

  def test_create_empty_scene(self):
    """Test creating an empty scene."""
    scene = Scene(name="TestScene")

    assert scene.name == "TestScene"
    assert len(scene) == 0
    assert len(scene.list_objects()) == 0

  def test_add_object_to_scene(self):
    """Test adding objects to a scene."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])
    obj = VolumatrixObject.from_mesh(vertices, faces, name="Triangle")

    scene = Scene()
    name = scene.add(obj, name="MyTriangle")

    assert name == "MyTriangle"
    assert len(scene) == 1
    assert "MyTriangle" in scene
    assert scene.get_object("MyTriangle") is not None

  def test_scene_transformations(self):
    """Test scene-level transformations."""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])
    obj = VolumatrixObject.from_mesh(vertices, faces)

    scene = Scene()
    scene.add(obj, name="Triangle", position=[1.0, 2.0, 3.0])

    transformed_obj = scene.get_transformed_object("Triangle")
    assert transformed_obj is not None

    # Check that the object was transformed
    original_center = obj.center()
    transformed_center = transformed_obj.center()

    # The center should be shifted by the position
    expected_center = original_center + [1.0, 2.0, 3.0]
    np.testing.assert_array_almost_equal(
      transformed_center, expected_center, decimal=5)

  def test_scene_visibility(self):
    """Test object visibility in scenes."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    obj = VolumatrixObject.from_mesh(vertices, faces)

    scene = Scene()
    scene.add(obj, name="Triangle")

    # Test visibility
    assert scene.nodes["Triangle"].visible == True

    scene.set_visibility("Triangle", False)
    assert scene.nodes["Triangle"].visible == False

  def test_scene_bounds(self):
    """Test scene bounding box calculation."""
    # Create two objects at different positions
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])

    obj1 = VolumatrixObject.from_mesh(vertices, faces)
    obj2 = VolumatrixObject.from_mesh(vertices, faces)

    scene = Scene()
    scene.add(obj1, name="Obj1", position=[0.0, 0.0, 0.0])
    scene.add(obj2, name="Obj2", position=[5.0, 5.0, 5.0])

    min_coords, max_coords = scene.bounds()

    # The scene should encompass both objects
    assert min_coords[0] <= 0.0
    assert max_coords[0] >= 6.0  # 5.0 + 1.0 (object width)
    assert min_coords[1] <= 0.0
    assert max_coords[1] >= 6.0  # 5.0 + 1.0 (object height)


if __name__ == "__main__":
  pytest.main([__file__])
