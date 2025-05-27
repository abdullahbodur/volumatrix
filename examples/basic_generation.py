#!/usr/bin/env python3
"""
Basic Generation Examples
This example demonstrates the fundamental object generation capabilities of Volumatrix.
Learn how to generate simple 3D objects from text prompts.
"""
from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def generate_simple_objects():
    """Generate basic geometric shapes."""
    log.info("Starting basic geometric shapes generation")
    # Generate a cube
    cube = vm.generate("cube")
    log.debug(f"Generated cube: {cube.name}")
    log.debug(f"   - Representations: {list(cube.representations.keys())}")
    log.debug(f"   - Bounds: {cube.bounds()}")
    # Generate a sphere
    sphere = vm.generate("sphere")
    log.debug(f"Generated sphere: {sphere.name}")
    log.debug(f"   - Center: {sphere.center()}")
    # Generate a cylinder
    cylinder = vm.generate("cylinder")
    log.debug(f"Generated cylinder: {cylinder.name}")
    log.info(f"Successfully generated {3} basic shapes")
    return cube, sphere, cylinder


def generate_with_different_formats():
    """Generate objects in different output formats."""
    log.info("Starting generation with different formats")
    # Generate as mesh (default)
    mesh_obj = vm.generate("chair", output_format="mesh")
    log.debug(f"Generated mesh chair: {mesh_obj.name}")
    log.debug(f"   - Has mesh: {mesh_obj.has_representation('mesh')}")
    # Generate as voxels
    voxel_obj = vm.generate("table", output_format="voxel", resolution=32)
    log.debug(f"Generated voxel table: {voxel_obj.name}")
    log.debug(f"   - Has voxels: {voxel_obj.has_representation('voxel')}")
    log.debug(f"   - Voxel resolution: {voxel_obj.voxel.resolution}")
    # Generate as point cloud
    pc_obj = vm.generate("lamp", output_format="pointcloud")
    log.debug(f"Generated point cloud lamp: {pc_obj.name}")
    log.debug(f"   - Has point cloud: {pc_obj.has_representation('pointcloud')}")
    log.debug(f"   - Number of points: {pc_obj.pointcloud.num_points}")
    log.info("Successfully generated objects in different formats")
    return mesh_obj, voxel_obj, pc_obj


def generate_with_seeds():
    """Generate reproducible objects using seeds."""
    log.info("Starting generation with seeds")
    # Generate the same object twice with the same seed
    obj1 = vm.generate("dragon", seed=42)
    obj2 = vm.generate("dragon", seed=42)
    log.debug(f"Generated dragon 1: {obj1.name}")
    log.debug(f"Generated dragon 2: {obj2.name}")
    log.debug(f"   - Names match: {obj1.name == obj2.name}")
    # Generate with different seed
    obj3 = vm.generate("dragon", seed=123)
    log.debug(f"Generated dragon 3: {obj3.name}")
    log.debug(f"   - Different from first: {obj1.name != obj3.name}")
    log.info("Successfully generated reproducible objects with seeds")
    return obj1, obj2, obj3


def generate_complex_objects():
    """Generate complex objects with various properties."""
    log.info("Starting generation of complex objects")
    # Generate complex objects
    objects = []
    prompts = [
        "detailed chair with armrests",
        "modern table with glass top",
        "ornate vase with patterns",
        "realistic tree with leaves",
    ]
    for prompt in prompts:
        obj = vm.generate(prompt)
        objects.append(obj)
        log.debug(f"Generated {prompt}: {obj.name}")
    log.info(f"Successfully generated {len(objects)} complex objects")
    return objects


def main():
    """Run all basic generation examples."""
    log.info("Starting Volumatrix Basic Generation Examples")
    log.info("=" * 50)
    try:
        # Basic shapes
        cube, sphere, cylinder = generate_simple_objects()
        # Different formats
        mesh_obj, voxel_obj, pc_obj = generate_with_different_formats()
        # Reproducible generation
        dragon1, dragon2, dragon3 = generate_with_seeds()
        # Complex objects
        complex_objects = generate_complex_objects()
        log.info("All examples completed successfully!")
        log.info(f"Total objects generated: {3 + 3 + 3 + len(complex_objects)}")
    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
