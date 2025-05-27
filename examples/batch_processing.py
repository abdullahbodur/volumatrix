#!/usr/bin/env python3
"""
Batch Processing Examples

This example demonstrates how to efficiently process multiple 3D objects
in Volumatrix using batch operations and parallel processing techniques.
"""

import time
from pathlib import Path

from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def basic_batch_generation():
    """Demonstrate basic batch generation."""
    log.info("Basic batch generation...")

    # Generate multiple objects at once
    prompts = ["cube", "sphere", "cylinder", "chair", "table"]
    log.info(f"Generating {len(prompts)} objects...")

    # Generate all objects
    objects = vm.generate_batch(prompts)

    log.info(f"Batch generation completed:")
    for prompt, obj in zip(prompts, objects):
        log.info(f"   - {prompt}: {obj.name}")

    return objects


def batch_generation_with_seeds():
    """Demonstrate batch generation with seeds."""
    log.info("Batch generation with seeds...")

    # Generate same objects with specific seeds
    prompts = ["dragon", "vase", "lamp"]
    seeds = [42, 123, 456]

    log.info(f"Generating {len(prompts)} objects with specific seeds...")
    objects = vm.generate_batch(prompts, seeds=seeds)

    log.info(f"Generated objects with seeds:")
    for prompt, seed, obj in zip(prompts, seeds, objects):
        log.info(f"   - {prompt} (seed {seed}): {obj.name}")

    # Verify reproducibility
    log.info(f"Verifying reproducibility...")
    for prompt, seed in zip(prompts, seeds):
        obj1 = vm.generate(prompt, seed=seed)
        obj2 = vm.generate(prompt, seed=seed)
        log.info(f"   - {prompt}: {obj1.name == obj2.name}")

    return objects


def batch_transformations():
    """Demonstrate batch transformations."""
    log.info("Batch transformations...")

    # Generate base objects
    base_objects = vm.generate_batch(["cube", "sphere", "cylinder"])
    log.info(f"Generated {len(base_objects)} base objects")

    # Define transformations
    transformations = [
        ("normalize", lambda x: vm.normalize(x)),
        ("scale_2x", lambda x: vm.rescale(x, 2.0)),
        ("rotate_45", lambda x: vm.rotate(x, [0, 0, 3.14159 / 4])),
        ("translate", lambda x: vm.translate(x, [1, 1, 1])),
    ]

    # Apply transformations
    log.info(f"Applying transformations...")
    transformed_objects = {}
    for name, transform in transformations:
        transformed_objects[name] = [transform(obj) for obj in base_objects]

    # Compare results
    log.info(f"Transformation comparison:")
    for name, objects in transformed_objects.items():
        log.info(f"   - {name}:")
        for i, obj in enumerate(objects):
            log.info(f"     * {base_objects[i].name} -> {obj.name}")

    return base_objects, transformed_objects


def batch_conversions():
    """Demonstrate batch conversions between representations."""
    log.info("Batch conversions...")

    # Generate objects in different formats
    mesh_objects = vm.generate_batch(["chair", "table", "lamp"], output_format="mesh")
    voxel_objects = vm.generate_batch(
        ["cube", "sphere", "cylinder"], output_format="voxel"
    )
    pc_objects = vm.generate_batch(
        ["vase", "bookshelf", "dragon"], output_format="pointcloud"
    )

    all_objects = mesh_objects + voxel_objects + pc_objects
    log.info(f"Generated {len(all_objects)} objects with different representations")

    # Convert all to mesh
    log.info(f"Converting all to mesh representation...")
    mesh_converted = []
    for obj in all_objects:
        if obj.has_representation("mesh"):
            mesh_converted.append(obj)
        elif obj.has_representation("voxel"):
            mesh_converted.append(vm.voxel_to_mesh(obj))
        elif obj.has_representation("pointcloud"):
            mesh_converted.append(vm.pointcloud_to_mesh(obj))

    # Add point cloud representation to all
    log.info(f"Adding point cloud representation to all...")
    pc_converted = []
    for obj in mesh_converted:
        if obj.has_representation("pointcloud"):
            pc_converted.append(obj)
        else:
            pc_converted.append(vm.mesh_to_pointcloud(obj))

    # Show final representations
    log.info(f"Final representations:")
    for i, (obj, pc) in enumerate(zip(mesh_converted, pc_converted)):
        log.info(f"   - Object {i}:")
        log.info(f"     * Mesh: {obj.mesh.num_vertices} vertices")
        log.info(f"     * Point cloud: {pc.pointcloud.num_points} points")

    return mesh_converted, pc_converted


def batch_export():
    """Demonstrate batch export operations."""
    log.info("Batch export...")

    # Generate furniture objects
    furniture_objects = vm.generate_batch(
        ["modern chair", "coffee table", "floor lamp", "bookshelf", "dining table"]
    )
    log.info(f"Generated {len(furniture_objects)} furniture objects")

    # Create temporary directory for exports
    temp_path = Path("temp_exports")
    temp_path.mkdir(exist_ok=True)
    log.info(f"Exporting to: {temp_path}")

    # Export to different formats
    export_formats = ["obj", "stl", "ply"]
    export_results = {}

    for fmt in export_formats:
        export_results[fmt] = []
        for i, obj in enumerate(furniture_objects):
            filename = temp_path / f"furniture_{i}.{fmt}"
            vm.export(obj, filename)
            export_results[fmt].append(filename)

    log.info(f"Batch export completed:")
    for fmt, files in export_results.items():
        log.info(f"   - {fmt.upper()}: {len(files)} files")

    # Calculate total size
    total_size = sum(
        f.stat().st_size for files in export_results.values() for f in files
    )
    log.info(f"Total exported size: {total_size} bytes")

    return export_results


def batch_scene_creation():
    """Demonstrate batch scene creation."""
    log.info("Batch scene creation...")

    # Define scene configurations
    scene_configs = [
        {
            "name": "LivingRoom",
            "objects": [
                ("sofa", [0, 0, 0]),
                ("coffee table", [0, 2, 0]),
                ("tv stand", [0, 4, 0]),
                ("bookshelf", [-3, 0, 0]),
                ("floor lamp", [-2, 2, 0]),
            ],
        },
        {
            "name": "DiningRoom",
            "objects": [
                ("dining table", [0, 0, 0]),
                ("chair", [-1, -1, 0]),
                ("chair", [1, -1, 0]),
                ("chair", [-1, 1, 0]),
                ("chair", [1, 1, 0]),
            ],
        },
        {
            "name": "Office",
            "objects": [
                ("desk", [0, 0, 0]),
                ("office chair", [0, -1, 0]),
                ("bookshelf", [3, 0, 0]),
                ("lamp", [0, 0, 0.8]),
                ("computer", [0, 0, 0.8]),
            ],
        },
    ]

    # Create scenes
    scenes = []
    for config in scene_configs:
        scene = vm.Scene(name=config["name"])
        for prompt, position in config["objects"]:
            obj = vm.generate(prompt)
            scene.add(obj, position=position)
        scenes.append(scene)
        log.info(f"Created {config['name']} with {len(scene)} objects")

        log.info(f"Scene analysis:")
        log.info(f"   - Bounds: {scene.bounds()}")
        log.info(f"   - Center: {scene.center()}")

    return scenes


def performance_comparison():
    """Compare individual vs batch operations."""
    log.info("Performance comparison...")

    # Test objects
    prompts = ["cube", "sphere", "cylinder", "chair", "table"] * 2
    n_objects = len(prompts)

    # Individual generation
    log.info(f"Individual generation...")
    start_time = time.time()
    individual_objects = []
    for prompt in prompts:
        obj = vm.generate(prompt)
        individual_objects.append(obj)
    individual_time = time.time() - start_time

    # Batch generation
    log.info(f"Batch generation...")
    start_time = time.time()
    batch_objects = vm.generate_batch(prompts)
    batch_time = time.time() - start_time

    log.info(f"Performance comparison:")
    log.info(f"   - Individual: {individual_time:.2f}s for {n_objects} objects")
    log.info(f"   - Batch: {batch_time:.2f}s for {n_objects} objects")
    log.info(f"   - Speedup: {individual_time/batch_time:.1f}x")

    return individual_objects, batch_objects


def main():
    """Run all batch processing examples."""
    log.info("Volumatrix Batch Processing Examples")
    log.info("=" * 50)

    try:
        # Basic batch generation
        basic_objects = basic_batch_generation()

        # Batch generation with seeds
        seeded_objects = batch_generation_with_seeds()

        # Batch transformations
        base_objects, transformed_objects = batch_transformations()

        # Batch conversions
        mesh_objects, pc_objects = batch_conversions()

        # Batch export
        export_results = batch_export()

        # Batch scene creation
        scenes = batch_scene_creation()

        # Performance comparison
        individual_objects, batch_objects = performance_comparison()

        log.info("All batch processing examples completed successfully!")

    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
