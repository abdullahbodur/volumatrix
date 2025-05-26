"""
Generation API for Volumatrix.

This module provides the main user-facing functions for generating 3D objects
using AI models.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..core.object import VolumatrixObject
from ..models.registry import get_model, list_models
from ..models.base import BaseModel

logger = logging.getLogger(__name__)


def generate(
    prompt: str,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    resolution: int = 64,
    output_format: str = "mesh",
    auto_preview: bool = False,
    preview_backend: str = "auto",
    **kwargs
) -> VolumatrixObject:
    """
    Generate a 3D object from a text prompt.
    
    Args:
        prompt: Text description of the object to generate
        model: Name of the model to use (uses default if None)
        seed: Random seed for reproducible generation
        resolution: Output resolution for voxel-based models
        output_format: Preferred output format ("mesh", "voxel", "pointcloud")
        auto_preview: Whether to automatically open a visualization window
        preview_backend: Backend to use for auto preview ("auto", "pyvista", "plotly", etc.)
        **kwargs: Additional model-specific parameters
    
    Returns:
        A VolumatrixObject containing the generated 3D object
    
    Examples:
        >>> obj = generate("a red futuristic drone")
        >>> chair = generate("wooden chair", model="diffusion-3d", seed=42)
        >>> voxel_obj = generate("tree", output_format="voxel", resolution=128)
        >>> # Automatically show the object in a window
        >>> cube = generate("cube", auto_preview=True)
    """
    try:
        # Get the model instance
        model_instance = get_model(model)
        if model_instance is None:
            available_models = list_models()
            if not available_models:
                raise ValueError(
                    "No models available. Please register a model first using "
                    "volumatrix.register_model() or install a model backend."
                )
            else:
                raise ValueError(
                    f"Model '{model}' not found. Available models: {available_models}"
                )
        
        logger.info(f"Generating 3D object with prompt: '{prompt}' using model: {model_instance.name}")
        
        # Prepare generation parameters
        generation_params = {
            "prompt": prompt,
            "seed": seed,
            "resolution": resolution,
            "output_format": output_format,
            **kwargs
        }
        
        # Generate the object
        result = model_instance.generate(**generation_params)
        
        if not isinstance(result, VolumatrixObject):
            raise TypeError(f"Model returned {type(result)}, expected VolumatrixObject")
        
        # Add generation metadata
        result.metadata.update({
            "prompt": prompt,
            "model": model_instance.name,
            "seed": seed,
            "resolution": resolution,
            "output_format": output_format,
            "generation_params": generation_params
        })
        
        logger.info(f"Successfully generated object: {result}")
        
        # Auto preview if requested
        if auto_preview:
            try:
                from ..rendering.preview import preview
                logger.info(f"Auto-previewing generated object: {result.name}")
                preview(result, backend=preview_backend, window_title=f"Generated: {prompt}")
            except Exception as e:
                logger.warning(f"Failed to auto-preview object: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate object with prompt '{prompt}': {e}")
        raise


def generate_batch(
    prompts: List[str],
    model: Optional[str] = None,
    seeds: Optional[List[int]] = None,
    resolution: int = 64,
    output_format: str = "mesh",
    auto_preview: bool = False,
    preview_backend: str = "auto",
    max_workers: Optional[int] = None,
    show_progress: bool = True,
    **kwargs
) -> List[VolumatrixObject]:
    """
    Generate multiple 3D objects from a list of text prompts.
    
    Args:
        prompts: List of text descriptions
        model: Name of the model to use (uses default if None)
        seeds: List of random seeds (one per prompt, optional)
        resolution: Output resolution for voxel-based models
        output_format: Preferred output format ("mesh", "voxel", "pointcloud")
        auto_preview: Whether to automatically preview each generated object
        preview_backend: Backend to use for auto preview ("auto", "pyvista", "plotly", etc.)
        max_workers: Maximum number of parallel workers (None for auto)
        show_progress: Whether to show a progress bar
        **kwargs: Additional model-specific parameters
    
    Returns:
        List of VolumatrixObjects containing the generated 3D objects
    
    Examples:
        >>> objects = generate_batch([
        ...     "red sports car",
        ...     "blue bicycle", 
        ...     "green tree"
        ... ])
        >>> objects = generate_batch(
        ...     ["chair", "table"],
        ...     seeds=[42, 123],
        ...     model="diffusion-3d"
        ... )
        >>> # Auto-preview each object as it's generated
        >>> objects = generate_batch(["cube", "sphere"], auto_preview=True)
    """
    if not prompts:
        return []
    
    if seeds is not None and len(seeds) != len(prompts):
        raise ValueError("Number of seeds must match number of prompts")
    
    logger.info(f"Starting batch generation of {len(prompts)} objects")
    
    # Prepare generation tasks
    tasks = []
    for i, prompt in enumerate(prompts):
        seed = seeds[i] if seeds else None
        task_kwargs = kwargs.copy()
        task_kwargs.update({
            "model": model,
            "seed": seed,
            "resolution": resolution,
            "output_format": output_format,
            "auto_preview": auto_preview,
            "preview_backend": preview_backend
        })
        tasks.append((prompt, task_kwargs))
    
    results = []
    failed_tasks = []
    
    # Force sequential processing if auto_preview is enabled to avoid GUI threading issues
    if auto_preview or max_workers == 1:
        # Sequential processing
        iterator = enumerate(tasks)
        if show_progress:
            iterator = tqdm(iterator, total=len(tasks), desc="Generating objects")
        
        for i, (prompt, task_kwargs) in iterator:
            try:
                result = generate(prompt, **task_kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate object {i} with prompt '{prompt}': {e}")
                failed_tasks.append((i, prompt, str(e)))
                results.append(None)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, (prompt, task_kwargs) in enumerate(tasks):
                future = executor.submit(generate, prompt, **task_kwargs)
                future_to_index[future] = (i, prompt)
            
            # Collect results
            results = [None] * len(prompts)
            completed_futures = as_completed(future_to_index.keys())
            
            if show_progress:
                completed_futures = tqdm(completed_futures, total=len(tasks), desc="Generating objects")
            
            for future in completed_futures:
                i, prompt = future_to_index[future]
                try:
                    result = future.result()
                    results[i] = result
                except Exception as e:
                    logger.error(f"Failed to generate object {i} with prompt '{prompt}': {e}")
                    failed_tasks.append((i, prompt, str(e)))
                    results[i] = None
    
    # Report results
    successful_count = sum(1 for r in results if r is not None)
    logger.info(f"Batch generation completed: {successful_count}/{len(prompts)} successful")
    
    if failed_tasks:
        logger.warning(f"Failed to generate {len(failed_tasks)} objects:")
        for i, prompt, error in failed_tasks:
            logger.warning(f"  {i}: '{prompt}' - {error}")
    
    return results


def generate_variations(
    base_prompt: str,
    variations: List[str],
    model: Optional[str] = None,
    base_seed: Optional[int] = None,
    **kwargs
) -> List[VolumatrixObject]:
    """
    Generate variations of a base prompt.
    
    Args:
        base_prompt: Base text description
        variations: List of variation modifiers
        model: Name of the model to use
        base_seed: Base seed for reproducible generation
        **kwargs: Additional generation parameters
    
    Returns:
        List of VolumatrixObjects with variations
    
    Examples:
        >>> variations = generate_variations(
        ...     "a chair",
        ...     ["wooden", "metal", "plastic", "leather"]
        ... )
        >>> colors = generate_variations(
        ...     "a sports car",
        ...     ["red", "blue", "green", "black"]
        ... )
    """
    prompts = [f"{variation} {base_prompt}" for variation in variations]
    
    # Generate seeds based on base_seed if provided
    seeds = None
    if base_seed is not None:
        import random
        random.seed(base_seed)
        seeds = [random.randint(0, 2**32 - 1) for _ in prompts]
    
    return generate_batch(prompts, model=model, seeds=seeds, **kwargs)


def generate_from_template(
    template: str,
    parameters: Dict[str, List[str]],
    model: Optional[str] = None,
    **kwargs
) -> List[VolumatrixObject]:
    """
    Generate objects from a template with parameter substitution.
    
    Args:
        template: Template string with {parameter} placeholders
        parameters: Dictionary mapping parameter names to lists of values
        model: Name of the model to use
        **kwargs: Additional generation parameters
    
    Returns:
        List of VolumatrixObjects generated from template combinations
    
    Examples:
        >>> objects = generate_from_template(
        ...     "a {color} {material} {object}",
        ...     {
        ...         "color": ["red", "blue"],
        ...         "material": ["wooden", "metal"],
        ...         "object": ["chair", "table"]
        ...     }
        ... )
    """
    import itertools
    
    # Generate all combinations of parameters
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    
    prompts = []
    for combination in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combination))
        prompt = template.format(**param_dict)
        prompts.append(prompt)
    
    return generate_batch(prompts, model=model, **kwargs)


def generate_and_show(prompt: str, **kwargs) -> VolumatrixObject:
    """
    Generate a 3D object and immediately show it in an interactive window.
    
    This is a convenience function equivalent to generate(..., auto_preview=True).
    
    Args:
        prompt: Text description of the object to generate
        **kwargs: All arguments supported by generate()
    
    Returns:
        A VolumatrixObject containing the generated 3D object
    
    Examples:
        >>> cube = generate_and_show("cube")  # Generates and shows cube
        >>> chair = generate_and_show("wooden chair", preview_backend="pyvista")
    """
    kwargs.setdefault("auto_preview", True)
    return generate(prompt, **kwargs) 