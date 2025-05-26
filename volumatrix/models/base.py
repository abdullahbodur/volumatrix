"""
Base model interface for Volumatrix.

This module defines the abstract base class that all AI model backends
must implement to be compatible with Volumatrix.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel as PydanticBaseModel, Field

from ..core.object import VolumatrixObject


class ModelConfig(PydanticBaseModel):
  """Configuration for a model."""

  name: str = Field(..., description="Model name")
  description: str = Field(default="", description="Model description")
  version: str = Field(default="1.0.0", description="Model version")
  supported_formats: List[str] = Field(
      default=["mesh"],
      description="Supported output formats"
  )
  max_resolution: int = Field(
    default=256, description="Maximum output resolution")
  requires_gpu: bool = Field(
    default=False, description="Whether model requires GPU")
  model_type: str = Field(
    default="unknown", description="Type of model (diffusion, vae, etc.)")
  parameters: Dict[str, Any] = Field(
      default_factory=dict,
      description="Model-specific parameters"
  )


class BaseModel(ABC):
  """
  Abstract base class for all AI model backends in Volumatrix.

  All model implementations must inherit from this class and implement
  the required abstract methods.
  """

  def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
    """
    Initialize the model.

    Args:
        config: Model configuration
        **kwargs: Additional initialization parameters
    """
    self.config = config or ModelConfig(name=self.__class__.__name__)
    self.is_loaded = False
    self._setup(**kwargs)

  @abstractmethod
  def _setup(self, **kwargs) -> None:
    """
    Setup the model (load weights, initialize components, etc.).

    This method is called during initialization and should handle
    all model-specific setup logic.
    """
    pass

  @abstractmethod
  def generate(
      self,
      prompt: str,
      seed: Optional[int] = None,
      resolution: int = 64,
      output_format: str = "mesh",
      **kwargs
  ) -> VolumatrixObject:
    """
    Generate a 3D object from a text prompt.

    Args:
        prompt: Text description of the object to generate
        seed: Random seed for reproducible generation
        resolution: Output resolution for voxel-based models
        output_format: Preferred output format ("mesh", "voxel", "pointcloud")
        **kwargs: Additional model-specific parameters

    Returns:
        A VolumatrixObject containing the generated 3D object
    """
    pass

  def load(self) -> None:
    """Load the model if not already loaded."""
    if not self.is_loaded:
      self._load_model()
      self.is_loaded = True

  def unload(self) -> None:
    """Unload the model to free memory."""
    if self.is_loaded:
      self._unload_model()
      self.is_loaded = False

  def _load_model(self) -> None:
    """
    Load model weights and components.

    Override this method to implement model loading logic.
    """
    pass

  def _unload_model(self) -> None:
    """
    Unload model weights and components.

    Override this method to implement model unloading logic.
    """
    pass

  def validate_parameters(self, **kwargs) -> Dict[str, Any]:
    """
    Validate and process generation parameters.

    Args:
        **kwargs: Generation parameters to validate

    Returns:
        Validated and processed parameters
    """
    validated = {}

    # Check output format
    output_format = kwargs.get("output_format", "mesh")
    if output_format not in self.config.supported_formats:
      raise ValueError(
          f"Output format '{output_format}' not supported by model '{self.config.name}'. "
          f"Supported formats: {self.config.supported_formats}"
      )
    validated["output_format"] = output_format

    # Check resolution
    resolution = kwargs.get("resolution", 64)
    if resolution > self.config.max_resolution:
      raise ValueError(
          f"Resolution {resolution} exceeds maximum {self.config.max_resolution} "
          f"for model '{self.config.name}'"
      )
    validated["resolution"] = resolution

    # Add other parameters
    for key, value in kwargs.items():
      if key not in validated:
        validated[key] = value

    return validated

  def preprocess_prompt(self, prompt: str) -> str:
    """
    Preprocess the input prompt.

    Override this method to implement prompt preprocessing logic
    (e.g., cleaning, tokenization, etc.).

    Args:
        prompt: Raw input prompt

    Returns:
        Processed prompt
    """
    return prompt.strip()

  def postprocess_output(self, output: Any, **kwargs) -> VolumatrixObject:
    """
    Postprocess the model output into a VolumatrixObject.

    Override this method to implement output postprocessing logic.

    Args:
        output: Raw model output
        **kwargs: Additional processing parameters

    Returns:
        A VolumatrixObject containing the processed output
    """
    raise NotImplementedError("Subclasses must implement postprocess_output")

  @property
  def name(self) -> str:
    """Get the model name."""
    return self.config.name

  @property
  def description(self) -> str:
    """Get the model description."""
    return self.config.description

  @property
  def version(self) -> str:
    """Get the model version."""
    return self.config.version

  @property
  def supported_formats(self) -> List[str]:
    """Get the supported output formats."""
    return self.config.supported_formats

  def get_info(self) -> Dict[str, Any]:
    """Get comprehensive model information."""
    return {
        "name": self.name,
        "description": self.description,
        "version": self.version,
        "supported_formats": self.supported_formats,
        "max_resolution": self.config.max_resolution,
        "requires_gpu": self.config.requires_gpu,
        "model_type": self.config.model_type,
        "is_loaded": self.is_loaded,
        "parameters": self.config.parameters
    }

  def __str__(self) -> str:
    """String representation of the model."""
    return f"{self.config.name} v{self.config.version} ({self.config.model_type})"

  def __repr__(self) -> str:
    """Detailed representation of the model."""
    return (
        f"{self.__class__.__name__}("
        f"name='{self.config.name}', "
        f"version='{self.config.version}', "
        f"loaded={self.is_loaded})"
    )
