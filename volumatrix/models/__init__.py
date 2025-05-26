"""
Models module for Volumatrix.

Contains the base model interface and registry for AI model backends.
"""

from .base import BaseModel
from .registry import register_model, get_model, list_models
from .dummy import DummyModel

__all__ = ["BaseModel", "register_model", "get_model", "list_models", "DummyModel"] 