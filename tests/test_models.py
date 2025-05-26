"""
Tests for Volumatrix models functionality.
"""

import pytest
import numpy as np

from volumatrix.models.base import BaseModel, ModelConfig
from volumatrix.models.dummy import DummyModel
from volumatrix.models.registry import (
    register_model, unregister_model, get_model, list_models,
    set_default_model, get_default_model, clear_registry
)
from volumatrix.core.object import VolumatrixObject


class TestModelConfig:
  """Test the ModelConfig class."""

  def test_create_config(self):
    """Test creating a model configuration."""
    config = ModelConfig(
        name="TestModel",
        description="A test model",
        version="1.0.0",
        supported_formats=["mesh", "voxel"],
        max_resolution=128
    )

    assert config.name == "TestModel"
    assert config.description == "A test model"
    assert config.version == "1.0.0"
    assert config.supported_formats == ["mesh", "voxel"]
    assert config.max_resolution == 128


class TestDummyModel:
  """Test the DummyModel implementation."""

  def test_create_dummy_model(self):
    """Test creating a dummy model."""
    model = DummyModel()

    assert model.name == "DummyModel"
    assert model.is_loaded == True
    assert "mesh" in model.supported_formats
    assert "voxel" in model.supported_formats
    assert "pointcloud" in model.supported_formats

  def test_generate_cube(self):
    """Test generating a cube."""
    model = DummyModel()
    obj = model.generate("cube")

    assert isinstance(obj, VolumatrixObject)
    assert "cube" in obj.name.lower()
    assert obj.has_representation("mesh")

  def test_generate_sphere(self):
    """Test generating a sphere."""
    model = DummyModel()
    obj = model.generate("sphere")

    assert isinstance(obj, VolumatrixObject)
    assert "sphere" in obj.name.lower()
    assert obj.has_representation("mesh")

  def test_generate_different_formats(self):
    """Test generating objects in different formats."""
    model = DummyModel()

    mesh_obj = model.generate("cube", output_format="mesh")
    voxel_obj = model.generate("cube", output_format="voxel")
    pc_obj = model.generate("cube", output_format="pointcloud")

    assert mesh_obj.has_representation("mesh")
    assert voxel_obj.has_representation("voxel")
    assert pc_obj.has_representation("pointcloud")

  def test_generate_with_seed(self):
    """Test reproducible generation with seed."""
    model = DummyModel()

    obj1 = model.generate("sphere", seed=42)
    obj2 = model.generate("sphere", seed=42)

    # Should generate identical objects
    assert obj1.name == obj2.name
    assert obj1.mesh.num_vertices == obj2.mesh.num_vertices

  def test_generate_different_shapes(self):
    """Test generating different shapes."""
    model = DummyModel()
    shapes = ["cube", "sphere", "cylinder", "chair", "table"]

    for shape in shapes:
      obj = model.generate(shape)
      assert isinstance(obj, VolumatrixObject)
      assert shape in obj.name.lower()

  def test_validate_parameters(self):
    """Test parameter validation."""
    model = DummyModel()

    # Valid parameters
    params = model.validate_parameters(
        output_format="mesh",
        resolution=64
    )
    assert params["output_format"] == "mesh"
    assert params["resolution"] == 64

    # Invalid output format
    with pytest.raises(ValueError, match="not supported"):
      model.validate_parameters(output_format="invalid")

    # Invalid resolution
    with pytest.raises(ValueError, match="exceeds maximum"):
      model.validate_parameters(resolution=1000)


class TestModelRegistry:
  """Test the model registry functionality."""

  def setup_method(self):
    """Setup for each test method."""
    clear_registry()

  def teardown_method(self):
    """Cleanup after each test method."""
    clear_registry()

  def test_register_model_instance(self):
    """Test registering a model instance."""
    model = DummyModel()
    name = register_model(model, name="test_dummy")

    assert name == "test_dummy"
    assert "test_dummy" in list_models()
    assert get_model("test_dummy") is model

  def test_register_model_class(self):
    """Test registering a model class."""
    name = register_model(DummyModel, name="test_dummy_class")

    assert name == "test_dummy_class"
    assert "test_dummy_class" in list_models()

    model = get_model("test_dummy_class")
    assert isinstance(model, DummyModel)

  def test_register_as_default(self):
    """Test registering a model as default."""
    register_model(DummyModel, name="default_model", set_as_default=True)

    assert get_default_model() == "default_model"
    assert get_model() is not None  # Should return default model

  def test_unregister_model(self):
    """Test unregistering a model."""
    register_model(DummyModel, name="temp_model")
    assert "temp_model" in list_models()

    success = unregister_model("temp_model")
    assert success == True
    assert "temp_model" not in list_models()

    # Try to unregister non-existent model
    success = unregister_model("nonexistent")
    assert success == False

  def test_set_default_model(self):
    """Test setting default model."""
    register_model(DummyModel, name="model1")
    register_model(DummyModel, name="model2")

    success = set_default_model("model2")
    assert success == True
    assert get_default_model() == "model2"

    # Try to set non-existent model as default
    success = set_default_model("nonexistent")
    assert success == False

  def test_multiple_models(self):
    """Test registering multiple models."""
    register_model(DummyModel, name="model1")
    register_model(DummyModel, name="model2")
    register_model(DummyModel, name="model3")

    models = list_models()
    assert len(models) == 3
    assert "model1" in models
    assert "model2" in models
    assert "model3" in models

  def test_model_overwrite(self):
    """Test overwriting an existing model."""
    register_model(DummyModel, name="test_model")
    original_model = get_model("test_model")

    # Register another model with same name
    register_model(DummyModel, name="test_model")
    new_model = get_model("test_model")

    # Should be different instances
    assert new_model is not original_model

  def test_clear_registry(self):
    """Test clearing the registry."""
    register_model(DummyModel, name="model1")
    register_model(DummyModel, name="model2")

    assert len(list_models()) == 2

    clear_registry()

    assert len(list_models()) == 0
    assert get_default_model() is None


class TestModelIntegration:
  """Integration tests for models with the rest of the system."""

  def setup_method(self):
    """Setup for each test method."""
    clear_registry()
    register_model(DummyModel, name="test_model", set_as_default=True)

  def teardown_method(self):
    """Cleanup after each test method."""
    clear_registry()

  def test_generation_through_api(self):
    """Test using models through the generation API."""
    from volumatrix.api.generate import generate

    obj = generate("cube")
    assert isinstance(obj, VolumatrixObject)
    assert obj.has_representation("mesh")

  def test_model_switching(self):
    """Test switching between different models."""
    from volumatrix.api.generate import generate

    # Register another dummy model
    register_model(DummyModel, name="model2")

    obj1 = generate("cube", model="test_model")
    obj2 = generate("cube", model="model2")

    # Both should work
    assert isinstance(obj1, VolumatrixObject)
    assert isinstance(obj2, VolumatrixObject)

  def test_model_info(self):
    """Test getting model information."""
    from volumatrix.models.registry import get_model_info, list_model_info

    info = get_model_info("test_model")
    assert info is not None
    assert info["name"] == "DummyModel"
    assert "supported_formats" in info

    all_info = list_model_info()
    assert "test_model" in all_info
    assert all_info["test_model"]["name"] == "DummyModel"


class TestCustomModel:
  """Test creating and using custom models."""

  def test_custom_model_implementation(self):
    """Test implementing a custom model."""

    class SimpleModel(BaseModel):
      def _setup(self, **kwargs):
        self.is_loaded = True

      def generate(self, prompt, **kwargs):
        # Create a simple cube regardless of prompt
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5],
            [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
            [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
        ])

        return VolumatrixObject.from_mesh(vertices, faces, name="SimpleCube")

    # Test the custom model
    config = ModelConfig(name="SimpleModel", supported_formats=["mesh"])
    model = SimpleModel(config)

    obj = model.generate("anything")
    assert isinstance(obj, VolumatrixObject)
    assert obj.name == "SimpleCube"
    assert obj.has_representation("mesh")


if __name__ == "__main__":
  pytest.main([__file__])
