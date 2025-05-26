"""
Model registry for Volumatrix.

This module provides functionality to register, discover, and manage
AI model backends.
"""

from typing import Dict, List, Optional, Type, Union
import logging
from threading import Lock

from .base import BaseModel

logger = logging.getLogger(__name__)

# Global model registry
_model_registry: Dict[str, BaseModel] = {}
_model_classes: Dict[str, Type[BaseModel]] = {}
_default_model: Optional[str] = None
_registry_lock = Lock()


def register_model(
    model: Union[BaseModel, Type[BaseModel]],
    name: Optional[str] = None,
    set_as_default: bool = False,
    **kwargs
) -> str:
  """
  Register a model in the global registry.

  Args:
      model: Model instance or class to register
      name: Name to register the model under (uses model.name if None)
      set_as_default: Whether to set this as the default model
      **kwargs: Additional arguments for model instantiation (if model is a class)

  Returns:
      The name the model was registered under

  Examples:
      >>> from volumatrix.models import DummyModel
      >>> register_model(DummyModel(), name="my_dummy")
      >>> register_model(DummyModel, name="another_dummy", set_as_default=True)
  """
  global _default_model

  with _registry_lock:
    # Handle class vs instance
    if isinstance(model, type):
      # It's a class, instantiate it
      model_instance = model(**kwargs)
      model_class = model
    else:
      # It's an instance
      model_instance = model
      model_class = type(model)

    # Determine the registration name
    if name is None:
      name = model_instance.name

    if name in _model_registry:
      logger.warning(f"Overwriting existing model '{name}' in registry")

    # Register the model
    _model_registry[name] = model_instance
    _model_classes[name] = model_class

    # Set as default if requested or if it's the first model
    if set_as_default or _default_model is None:
      _default_model = name

    logger.info(f"Registered model '{name}' ({model_class.__name__})")

    return name


def unregister_model(name: str) -> bool:
  """
  Unregister a model from the registry.

  Args:
      name: Name of the model to unregister

  Returns:
      True if the model was unregistered, False if not found
  """
  global _default_model

  with _registry_lock:
    if name not in _model_registry:
      return False

    # Unload the model if it's loaded
    model = _model_registry[name]
    if model.is_loaded:
      model.unload()

    # Remove from registry
    del _model_registry[name]
    del _model_classes[name]

    # Update default if necessary
    if _default_model == name:
      _default_model = next(iter(_model_registry.keys())
                            ) if _model_registry else None

    logger.info(f"Unregistered model '{name}'")
    return True


def get_model(name: Optional[str] = None) -> Optional[BaseModel]:
  """
  Get a model from the registry.

  Args:
      name: Name of the model to get (uses default if None)

  Returns:
      The model instance, or None if not found

  Examples:
      >>> model = get_model("my_dummy")
      >>> default_model = get_model()  # Gets default model
  """
  with _registry_lock:
    if name is None:
      name = _default_model

    if name is None:
      return None

    return _model_registry.get(name)


def list_models() -> List[str]:
  """
  List all registered model names.

  Returns:
      List of registered model names
  """
  with _registry_lock:
    return list(_model_registry.keys())


def get_model_info(name: Optional[str] = None) -> Optional[Dict]:
  """
  Get information about a model.

  Args:
      name: Name of the model (uses default if None)

  Returns:
      Model information dictionary, or None if not found
  """
  model = get_model(name)
  return model.get_info() if model else None


def list_model_info() -> Dict[str, Dict]:
  """
  Get information about all registered models.

  Returns:
      Dictionary mapping model names to their information
  """
  with _registry_lock:
    return {name: model.get_info() for name, model in _model_registry.items()}


def set_default_model(name: str) -> bool:
  """
  Set the default model.

  Args:
      name: Name of the model to set as default

  Returns:
      True if successful, False if model not found
  """
  global _default_model

  with _registry_lock:
    if name not in _model_registry:
      return False

    _default_model = name
    logger.info(f"Set default model to '{name}'")
    return True


def get_default_model() -> Optional[str]:
  """
  Get the name of the default model.

  Returns:
      Name of the default model, or None if no default set
  """
  with _registry_lock:
    return _default_model


def clear_registry() -> None:
  """Clear all models from the registry."""
  global _default_model

  with _registry_lock:
    # Unload all models
    for model in _model_registry.values():
      if model.is_loaded:
        model.unload()

    _model_registry.clear()
    _model_classes.clear()
    _default_model = None

    logger.info("Cleared model registry")


def auto_discover_models() -> List[str]:
  """
  Automatically discover and register available models.

  This function looks for common model backends and registers them
  if they are available.

  Returns:
      List of names of models that were discovered and registered
  """
  discovered = []

  # Try to register the dummy model for testing
  try:
    from .dummy import DummyModel
    name = register_model(DummyModel, name="dummy", set_as_default=True)
    discovered.append(name)
  except ImportError:
    pass

  # TODO: Add discovery for other model backends
  # Try to discover and register other models like:
  # - Stability AI models
  # - OpenAI 3D models
  # - DreamFusion
  # - Custom models in plugins directory

  logger.info(f"Auto-discovered {len(discovered)} models: {discovered}")
  return discovered


def load_model(name: Optional[str] = None) -> bool:
  """
  Load a model into memory.

  Args:
      name: Name of the model to load (uses default if None)

  Returns:
      True if successful, False if model not found or already loaded
  """
  model = get_model(name)
  if model is None:
    return False

  if model.is_loaded:
    return True

  try:
    model.load()
    logger.info(f"Loaded model '{model.name}'")
    return True
  except Exception as e:
    logger.error(f"Failed to load model '{model.name}': {e}")
    return False


def unload_model(name: Optional[str] = None) -> bool:
  """
  Unload a model from memory.

  Args:
      name: Name of the model to unload (uses default if None)

  Returns:
      True if successful, False if model not found or not loaded
  """
  model = get_model(name)
  if model is None:
    return False

  if not model.is_loaded:
    return True

  try:
    model.unload()
    logger.info(f"Unloaded model '{model.name}'")
    return True
  except Exception as e:
    logger.error(f"Failed to unload model '{model.name}': {e}")
    return False


def reload_model(name: Optional[str] = None) -> bool:
  """
  Reload a model (unload then load).

  Args:
      name: Name of the model to reload (uses default if None)

  Returns:
      True if successful, False otherwise
  """
  if unload_model(name):
    return load_model(name)
  return False


# Initialize with auto-discovery
def _initialize_registry():
  """Initialize the registry with auto-discovered models."""
  try:
    auto_discover_models()
  except Exception as e:
    logger.warning(f"Failed to auto-discover models: {e}")


# Auto-initialize when module is imported
_initialize_registry()
