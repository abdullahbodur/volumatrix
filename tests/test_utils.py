"""
Tests for Volumatrix utilities functionality.
"""

import pytest
import numpy as np

import volumatrix as vm
from volumatrix.utils.transforms import normalize, rescale, rotate, translate, align_to_axes, fit_in_box
from volumatrix.utils.conversion import (
    voxelize, devoxelize, mesh_to_pointcloud, pointcloud_to_mesh,
    mesh_to_voxel, voxel_to_mesh
)
from volumatrix.core.object import VolumatrixObject
from volumatrix.core.representations import Mesh, Voxel, PointCloud


class TestTransforms:
  """Test transformation utilities."""

  def test_normalize(self):
    """Test object normalization."""
    # Create a cube that's not normalized
    vertices = np.array([
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
        [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
    ])

    obj = VolumatrixObject.from_mesh(vertices, faces)
    normalized = normalize(obj)

    # Check that the object is now centered and scaled
    min_coords, max_coords = normalized.bounds()
    center = normalized.center()

    np.testing.assert_array_almost_equal(center, [0, 0, 0], decimal=5)
    assert np.max(max_coords - min_coords) <= 2.0  # Should fit in [-1, 1] range

  def test_rescale(self):
    """Test object rescaling."""
    obj = vm.generate("cube")
    original_size = np.max(
      np.array(obj.bounds()[1]) - np.array(obj.bounds()[0]))

    # Scale by 2x
    scaled = rescale(obj, 2.0)
    new_size = np.max(
      np.array(scaled.bounds()[1]) - np.array(scaled.bounds()[0]))

    assert abs(new_size - original_size * 2.0) < 0.1

  def test_rescale_non_uniform(self):
    """Test non-uniform rescaling."""
    obj = vm.generate("cube")

    # Scale differently in each dimension
    scaled = rescale(obj, [2.0, 1.0, 0.5])

    min_coords, max_coords = scaled.bounds()
    size = max_coords - min_coords

    # Check relative proportions (approximately)
    assert size[0] > size[1]  # X should be larger than Y
    assert size[1] > size[2]  # Y should be larger than Z

  def test_rotate(self):
    """Test object rotation."""
    obj = vm.generate("cube")

    # Rotate 90 degrees around Z axis
    rotated = rotate(obj, [0, 0, np.pi / 2])

    # Object should be rotated but same size
    original_size = np.array(obj.bounds()[1]) - np.array(obj.bounds()[0])
    rotated_size = np.array(rotated.bounds()[1]) - np.array(rotated.bounds()[0])

    # Size should be approximately the same (allowing for numerical precision)
    np.testing.assert_array_almost_equal(
        np.sort(original_size), np.sort(rotated_size), decimal=3
    )

  def test_rotate_degrees(self):
    """Test rotation with degrees."""
    obj = vm.generate("cube")

    # Rotate 90 degrees around Z axis
    rotated = rotate(obj, [0, 0, 90], degrees=True)

    assert isinstance(rotated, VolumatrixObject)

  def test_translate(self):
    """Test object translation."""
    obj = vm.generate("cube")
    original_center = obj.center()

    translation = [1.0, 2.0, 3.0]
    translated = translate(obj, translation)
    new_center = translated.center()

    expected_center = original_center + np.array(translation)
    np.testing.assert_array_almost_equal(new_center, expected_center, decimal=5)

  def test_fit_in_box(self):
    """Test fitting object in a box."""
    obj = vm.generate("cube")

    # Fit in a 1x1x1 box
    fitted = fit_in_box(obj, 1.0)

    min_coords, max_coords = fitted.bounds()
    size = max_coords - min_coords

    # Should fit within the box
    assert np.all(size <= 1.0 + 1e-6)  # Allow small numerical error

  def test_fit_in_non_uniform_box(self):
    """Test fitting object in a non-uniform box."""
    obj = vm.generate("cube")

    # Fit in a 2x1x0.5 box
    fitted = fit_in_box(obj, [2.0, 1.0, 0.5])

    min_coords, max_coords = fitted.bounds()
    size = max_coords - min_coords

    # Should fit within the box
    assert size[0] <= 2.0 + 1e-6
    assert size[1] <= 1.0 + 1e-6
    assert size[2] <= 0.5 + 1e-6


class TestConversions:
  """Test conversion utilities."""

  def test_voxelize_mesh(self):
    """Test converting mesh to voxels."""
    mesh_obj = vm.generate("cube", output_format="mesh")
    voxel_obj = voxelize(mesh_obj, resolution=16)

    assert voxel_obj.has_representation("voxel")
    assert voxel_obj.voxel.num_occupied > 0
    assert voxel_obj.voxel.resolution[0] <= 16  # May be smaller due to bounds

  def test_devoxelize(self):
    """Test converting voxels to mesh."""
    voxel_obj = vm.generate("cube", output_format="voxel", resolution=16)
    mesh_obj = devoxelize(voxel_obj)

    assert mesh_obj.has_representation("mesh")
    assert mesh_obj.mesh.num_vertices > 0
    assert mesh_obj.mesh.num_faces > 0

  def test_mesh_to_pointcloud(self):
    """Test converting mesh to point cloud."""
    mesh_obj = vm.generate("sphere", output_format="mesh")
    pc_obj = mesh_to_pointcloud(mesh_obj, num_points=500)

    assert pc_obj.has_representation("pointcloud")
    # May be fewer if mesh has fewer vertices
    assert pc_obj.pointcloud.num_points <= 500

  def test_mesh_to_pointcloud_methods(self):
    """Test different mesh to point cloud conversion methods."""
    mesh_obj = vm.generate("cube", output_format="mesh")

    # Test different methods
    pc_surface = mesh_to_pointcloud(mesh_obj, num_points=100, method="surface")
    pc_vertices = mesh_to_pointcloud(
      mesh_obj, num_points=100, method="vertices")
    pc_random = mesh_to_pointcloud(mesh_obj, num_points=100, method="random")

    for pc_obj in [pc_surface, pc_vertices, pc_random]:
      assert pc_obj.has_representation("pointcloud")
      assert pc_obj.pointcloud.num_points > 0

  def test_pointcloud_to_mesh(self):
    """Test converting point cloud to mesh."""
    pc_obj = vm.generate("sphere", output_format="pointcloud")
    mesh_obj = pointcloud_to_mesh(pc_obj, method="delaunay")

    assert mesh_obj.has_representation("mesh")
    assert mesh_obj.mesh.num_vertices > 0
    assert mesh_obj.mesh.num_faces > 0

  def test_conversion_aliases(self):
    """Test conversion alias functions."""
    mesh_obj = vm.generate("cube", output_format="mesh")

    # Test aliases
    voxel_obj = mesh_to_voxel(mesh_obj, resolution=16)
    mesh_obj2 = voxel_to_mesh(voxel_obj)

    assert voxel_obj.has_representation("voxel")
    assert mesh_obj2.has_representation("mesh")

  def test_conversion_preserves_original(self):
    """Test that conversions preserve original representations."""
    mesh_obj = vm.generate("cube", output_format="mesh")
    original_mesh = mesh_obj.mesh

    # Convert to point cloud
    pc_obj = mesh_to_pointcloud(mesh_obj)

    # Original mesh should still be there
    assert pc_obj.has_representation("mesh")
    assert pc_obj.has_representation("pointcloud")
    assert pc_obj.mesh is original_mesh  # Should be the same object


class TestConversionErrors:
  """Test error handling in conversions."""

  def test_voxelize_without_mesh_or_pointcloud(self):
    """Test voxelizing object without mesh or point cloud."""
    # Create object with only voxel representation
    grid = np.ones((8, 8, 8), dtype=bool)
    voxel_obj = VolumatrixObject.from_voxel(grid, (8, 8, 8))

    with pytest.raises(ValueError, match="Cannot voxelize representation type"):
      voxelize(voxel_obj)

  def test_devoxelize_without_voxel(self):
    """Test devoxelizing object without voxel representation."""
    mesh_obj = vm.generate("cube", output_format="mesh")

    with pytest.raises(ValueError, match="must have a voxel representation"):
      devoxelize(mesh_obj)

  def test_mesh_to_pointcloud_without_mesh(self):
    """Test converting to point cloud without mesh."""
    points = np.random.rand(100, 3)
    pc_obj = VolumatrixObject.from_pointcloud(points)

    with pytest.raises(ValueError, match="must have a mesh representation"):
      mesh_to_pointcloud(pc_obj)

  def test_pointcloud_to_mesh_without_pointcloud(self):
    """Test converting to mesh without point cloud."""
    mesh_obj = vm.generate("cube", output_format="mesh")

    with pytest.raises(ValueError, match="must have a point cloud representation"):
      pointcloud_to_mesh(mesh_obj)


class TestIntegrationWorkflows:
  """Test complete workflows combining multiple utilities."""

  def test_full_conversion_pipeline(self):
    """Test a complete conversion pipeline."""
    # Start with a mesh
    mesh_obj = vm.generate("sphere", output_format="mesh")

    # Convert to point cloud
    pc_obj = mesh_to_pointcloud(mesh_obj, num_points=500)

    # Convert to voxels
    voxel_obj = voxelize(pc_obj, resolution=16)

    # Convert back to mesh
    final_mesh = devoxelize(voxel_obj)

    # Should have all representations
    assert final_mesh.has_representation("mesh")
    assert final_mesh.has_representation("pointcloud")
    assert final_mesh.has_representation("voxel")

  def test_transform_and_convert_workflow(self):
    """Test combining transformations and conversions."""
    obj = vm.generate("cube", output_format="mesh")

    # Transform the object
    transformed = normalize(obj)
    transformed = rescale(transformed, 2.0)
    transformed = rotate(transformed, [0, 0, np.pi / 4])

    # Convert to different representations
    pc_obj = mesh_to_pointcloud(transformed, num_points=200)
    voxel_obj = voxelize(transformed, resolution=16)

    # All should be valid
    assert pc_obj.has_representation("pointcloud")
    assert voxel_obj.has_representation("voxel")

  def test_scene_with_conversions(self):
    """Test using conversions in a scene workflow."""
    # Create objects with different representations
    mesh_obj = vm.generate("cube", output_format="mesh")
    pc_obj = mesh_to_pointcloud(mesh_obj, num_points=100)
    voxel_obj = voxelize(mesh_obj, resolution=8)

    # Add to scene
    scene = vm.Scene()
    scene.add(mesh_obj, name="MeshCube", position=[0, 0, 0])
    scene.add(pc_obj, name="PCCube", position=[2, 0, 0])
    scene.add(voxel_obj, name="VoxelCube", position=[4, 0, 0])

    assert len(scene) == 3

    # Test scene bounds
    min_coords, max_coords = scene.bounds()
    assert max_coords[0] > 4.0  # Should span across all objects


if __name__ == "__main__":
  pytest.main([__file__])
