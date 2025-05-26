"""
Tests for Volumatrix API functionality.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

import volumatrix as vm
from volumatrix.api.generate import generate, generate_batch
from volumatrix.api.io import export, load
from volumatrix.core.object import VolumatrixObject


class TestGeneration:
  """Test the generation API."""

  def test_simple_generation(self):
    """Test basic object generation."""
    obj = generate("cube")

    assert isinstance(obj, VolumatrixObject)
    assert obj.name.startswith("Dummy_")
    assert len(obj.representations) > 0

  def test_generation_with_seed(self):
    """Test generation with reproducible seed."""
    obj1 = generate("sphere", seed=42)
    obj2 = generate("sphere", seed=42)

    # Objects should be identical with same seed
    assert obj1.name == obj2.name
    assert len(obj1.representations) == len(obj2.representations)

  def test_generation_different_formats(self):
    """Test generation with different output formats."""
    mesh_obj = generate("cube", output_format="mesh")
    voxel_obj = generate("cube", output_format="voxel")
    pc_obj = generate("cube", output_format="pointcloud")

    assert mesh_obj.has_representation("mesh")
    assert voxel_obj.has_representation("voxel")
    assert pc_obj.has_representation("pointcloud")

  def test_batch_generation(self):
    """Test batch generation."""
    prompts = ["cube", "sphere", "cylinder"]
    objects = generate_batch(prompts)

    assert len(objects) == 3
    assert all(isinstance(obj, VolumatrixObject) for obj in objects)
    assert all(obj is not None for obj in objects)

  def test_batch_generation_with_seeds(self):
    """Test batch generation with seeds."""
    prompts = ["cube", "sphere"]
    seeds = [42, 123]

    objects = generate_batch(prompts, seeds=seeds)

    assert len(objects) == 2
    assert all(isinstance(obj, VolumatrixObject) for obj in objects)

  def test_generation_keywords(self):
    """Test generation with different keywords."""
    keywords = ["cube", "sphere", "cylinder", "chair", "table"]

    for keyword in keywords:
      obj = generate(keyword)
      assert isinstance(obj, VolumatrixObject)
      assert keyword.lower() in obj.name.lower()


class TestIO:
  """Test I/O functionality."""

  def test_export_obj(self):
    """Test exporting to OBJ format."""
    obj = generate("cube")

    with tempfile.TemporaryDirectory() as tmpdir:
      filepath = Path(tmpdir) / "test_cube.obj"
      export(obj, str(filepath))

      assert filepath.exists()
      assert filepath.stat().st_size > 0

  def test_export_different_formats(self):
    """Test exporting to different formats."""
    obj = generate("sphere")

    formats = ["obj", "stl"]  # Test formats that don't require trimesh

    with tempfile.TemporaryDirectory() as tmpdir:
      for fmt in formats:
        filepath = Path(tmpdir) / f"test_sphere.{fmt}"
        export(obj, str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

  def test_export_without_mesh(self):
    """Test exporting object without mesh representation."""
    # Create object with only point cloud
    points = np.random.rand(100, 3)
    obj = VolumatrixObject.from_pointcloud(points)

    with tempfile.TemporaryDirectory() as tmpdir:
      filepath = Path(tmpdir) / "test_pc.obj"

      with pytest.raises(ValueError, match="must have a mesh representation"):
        export(obj, str(filepath))


class TestIntegration:
  """Integration tests combining multiple components."""

  def test_generate_and_export(self):
    """Test generating and exporting an object."""
    obj = generate("chair")

    with tempfile.TemporaryDirectory() as tmpdir:
      filepath = Path(tmpdir) / "generated_chair.obj"
      export(obj, str(filepath))

      assert filepath.exists()
      assert filepath.stat().st_size > 0

  def test_scene_workflow(self):
    """Test complete scene workflow."""
    # Generate objects
    chair = generate("chair")
    table = generate("table")

    # Create scene
    scene = vm.Scene(name="FurnitureScene")
    scene.add(chair, name="Chair", position=[0, 0, 0])
    scene.add(table, name="Table", position=[2, 0, 0])

    # Test scene properties
    assert len(scene) == 2
    assert "Chair" in scene
    assert "Table" in scene

    # Test scene bounds
    min_coords, max_coords = scene.bounds()
    assert len(min_coords) == 3
    assert len(max_coords) == 3

    # Test merging
    merged = scene.merge_objects()
    assert isinstance(merged, VolumatrixObject)
    assert "merged" in merged.name.lower()

  def test_transformations_workflow(self):
    """Test transformation workflow."""
    obj = generate("cube")

    # Test various transformations
    normalized = vm.normalize(obj)
    scaled = vm.rescale(obj, 2.0)
    rotated = vm.rotate(obj, [0, np.pi / 4, 0])

    # All should be valid objects
    for transformed_obj in [normalized, scaled, rotated]:
      assert isinstance(transformed_obj, VolumatrixObject)
      assert len(transformed_obj.representations) > 0

  def test_conversion_workflow(self):
    """Test conversion between representations."""
    # Start with mesh
    mesh_obj = generate("sphere", output_format="mesh")

    # Convert to point cloud
    pc_obj = vm.mesh_to_pointcloud(mesh_obj, num_points=1000)
    assert pc_obj.has_representation("pointcloud")
    assert pc_obj.pointcloud.num_points == 1000

    # Convert to voxels
    voxel_obj = vm.voxelize(mesh_obj, resolution=32)
    assert voxel_obj.has_representation("voxel")
    assert voxel_obj.voxel.resolution == (32, 32, 32)


class TestErrorHandling:
  """Test error handling and edge cases."""

  def test_invalid_model_name(self):
    """Test generation with invalid model name."""
    with pytest.raises(ValueError, match="not found"):
      generate("cube", model="nonexistent_model")

  def test_invalid_output_format(self):
    """Test generation with invalid output format."""
    with pytest.raises(ValueError, match="not supported"):
      generate("cube", output_format="invalid_format")

  def test_export_nonexistent_directory(self):
    """Test exporting to nonexistent directory."""
    obj = generate("cube")

    # Should create directory automatically
    with tempfile.TemporaryDirectory() as tmpdir:
      filepath = Path(tmpdir) / "subdir" / "test.obj"
      export(obj, str(filepath))

      assert filepath.exists()

  def test_empty_batch_generation(self):
    """Test batch generation with empty prompt list."""
    objects = generate_batch([])
    assert objects == []

  def test_mismatched_seeds_length(self):
    """Test batch generation with mismatched seeds length."""
    prompts = ["cube", "sphere"]
    seeds = [42]  # Wrong length

    with pytest.raises(ValueError, match="Number of seeds must match"):
      generate_batch(prompts, seeds=seeds)


if __name__ == "__main__":
  pytest.main([__file__])
