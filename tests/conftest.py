"""
Pytest configuration and fixtures for Volumatrix tests.
"""

import pytest
import numpy as np
import warnings

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


@pytest.fixture
def simple_cube_vertices():
  """Fixture providing vertices for a simple cube."""
  return np.array([
      [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom face
      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top face
  ], dtype=np.float32)


@pytest.fixture
def simple_cube_faces():
  """Fixture providing faces for a simple cube."""
  return np.array([
      [0, 1, 2], [0, 2, 3],  # Bottom
      [4, 7, 6], [4, 6, 5],  # Top
      [0, 4, 5], [0, 5, 1],  # Front
      [2, 6, 7], [2, 7, 3],  # Back
      [0, 3, 7], [0, 7, 4],  # Left
      [1, 5, 6], [1, 6, 2]   # Right
  ], dtype=np.int32)


@pytest.fixture
def random_points():
  """Fixture providing random 3D points."""
  np.random.seed(42)  # For reproducible tests
  return np.random.rand(100, 3) * 2 - 1  # Points in [-1, 1] range


@pytest.fixture
def sample_voxel_grid():
  """Fixture providing a sample voxel grid."""
  grid = np.zeros((8, 8, 8), dtype=bool)
  grid[2:6, 2:6, 2:6] = True  # Cube in the center
  return grid


@pytest.fixture(autouse=True)
def reset_random_seed():
  """Reset random seed before each test for reproducibility."""
  np.random.seed(42)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
  """Setup test environment once per session."""
  # Ensure the dummy model is available for tests
  from volumatrix.models.registry import clear_registry, register_model
  from volumatrix.models.dummy import DummyModel

  # Clear and setup fresh registry for tests
  clear_registry()
  register_model(DummyModel, name="dummy", set_as_default=True)

  yield

  # Cleanup after all tests
  clear_registry()


def pytest_configure(config):
  """Configure pytest with custom markers."""
  config.addinivalue_line(
      "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
  )
  config.addinivalue_line(
      "markers", "integration: marks tests as integration tests"
  )
  config.addinivalue_line(
      "markers", "requires_trimesh: marks tests that require trimesh library"
  )
