#!/usr/bin/env python3
"""
Transformation Examples

This example demonstrates how to transform and manipulate 3D objects in Volumatrix.
Learn about scaling, rotation, translation, normalization, and more.
"""

import numpy as np
from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def basic_transformations():
    """Demonstrate basic transformation operations."""
    log.info("Basic transformations...")

    # Generate a cube to work with
    cube = vm.generate("cube")
    log.info(f"Original cube: {cube.name}")
    log.info(f"   - Bounds: {cube.bounds()}")
    log.info(f"   - Center: {cube.center()}")

    # Normalize the object (center and scale to fit in [-1, 1])
    normalized = vm.normalize(cube)
    log.info(f"Normalized cube:")
    log.info(f"   - New bounds: {normalized.bounds()}")
    log.info(f"   - New center: {normalized.center()}")

    # Scale the object
    scaled = vm.rescale(cube, 2.0)
    log.info(f"Scaled cube (2x):")
    log.info(f"   - New bounds: {scaled.bounds()}")

    # Non-uniform scaling
    stretched = vm.rescale(cube, [3.0, 1.0, 0.5])
    log.info(f"Stretched cube (3x, 1x, 0.5x):")
    log.info(f"   - New bounds: {stretched.bounds()}")

    return cube, normalized, scaled, stretched


def rotation_examples():
    """Demonstrate rotation operations."""
    log.info("Rotation examples...")

    # Generate a chair to make rotations more visible
    chair = vm.generate("chair")
    log.info(f"Original chair: {chair.name}")

    # Rotate around Z-axis (90 degrees)
    rotated_z = vm.rotate(chair, [0, 0, np.pi / 2])
    log.info(f"Rotated 90° around Z-axis")

    # Rotate around Y-axis (45 degrees)
    rotated_y = vm.rotate(chair, [0, np.pi / 4, 0])
    log.info(f"Rotated 45° around Y-axis")

    # Rotate using degrees instead of radians
    rotated_degrees = vm.rotate(chair, [0, 0, 90], degrees=True)
    log.info(f"Rotated 90° using degrees")

    # Complex rotation (around all axes)
    complex_rotation = vm.rotate(chair, [np.pi / 6, np.pi / 4, np.pi / 3])
    log.info(f"Complex rotation around all axes")

    return chair, rotated_z, rotated_y, rotated_degrees, complex_rotation


def translation_examples():
    """Demonstrate translation operations."""
    log.info("Translation examples...")

    # Generate objects to translate
    sphere = vm.generate("sphere")
    log.info(f"Original sphere center: {sphere.center()}")

    # Simple translation
    translated = vm.translate(sphere, [2.0, 3.0, 1.0])
    log.info(f"Translated sphere center: {translated.center()}")

    # Multiple translations
    twice_translated = vm.translate(translated, [-1.0, -1.0, 2.0])
    log.info(f"Twice translated center: {twice_translated.center()}")

    return sphere, translated, twice_translated


def fitting_and_alignment():
    """Demonstrate fitting objects into specific spaces."""
    log.info("Fitting and alignment examples...")

    # Generate a large object
    table = vm.generate("table")
    original_bounds = table.bounds()
    log.info(f"Original table bounds: {original_bounds}")

    # Fit into a unit cube
    fitted_cube = vm.fit_in_box(table, 1.0)
    new_bounds = fitted_cube.bounds()
    log.info(f"Fitted in unit cube: {new_bounds}")

    # Fit into a rectangular box
    fitted_rect = vm.fit_in_box(table, [2.0, 1.0, 0.5])
    rect_bounds = fitted_rect.bounds()
    log.info(f"Fitted in rectangle: {rect_bounds}")

    return table, fitted_cube, fitted_rect


def chained_transformations():
    """Demonstrate chaining multiple transformations."""
    log.info("Chained transformations...")

    # Start with a lamp
    lamp = vm.generate("lamp")
    log.info(f"Original lamp: {lamp.name}")
    log.info(f"   - Bounds: {lamp.bounds()}")

    # Chain multiple transformations
    transformed = lamp

    # Step 1: Normalize
    transformed = vm.normalize(transformed)
    log.info(f"Step 1 - Normalized")

    # Step 2: Scale up
    transformed = vm.rescale(transformed, 1.5)
    log.info(f"Step 2 - Scaled 1.5x")

    # Step 3: Rotate
    transformed = vm.rotate(transformed, [0, np.pi / 4, 0])
    log.info(f"Step 3 - Rotated 45° around Y")

    # Step 4: Translate
    transformed = vm.translate(transformed, [1.0, 0.5, -0.5])
    log.info(f"Step 4 - Translated")

    # Step 5: Fit in box
    transformed = vm.fit_in_box(transformed, [2.0, 2.0, 2.0])
    log.info(f"Step 5 - Fitted in 2x2x2 box")

    log.info(f"   - Final bounds: {transformed.bounds()}")
    log.info(f"   - Final center: {transformed.center()}")

    return lamp, transformed


def transformation_preservation():
    """Show how transformations preserve object properties."""
    log.info("Transformation preservation...")

    # Generate object with multiple representations
    obj = vm.generate("sphere", output_format="mesh")

    # Add more representations
    obj = vm.mesh_to_pointcloud(obj, num_points=500)
    obj = vm.voxelize(obj, resolution=16)

    log.info(f"Original object representations: {list(obj.representations.keys())}")

    # Apply transformation
    transformed = vm.rescale(obj, 2.0)

    log.info(
        f"Transformed object representations: {list(transformed.representations.keys())}"
    )
    log.info("   - All representations are preserved and transformed!")

    return obj, transformed


def main():
    """Run all transformation examples."""
    log.info("Volumatrix Transformation Examples")
    log.info("=" * 50)

    try:
        # Basic transformations
        cube, normalized, scaled, stretched = basic_transformations()

        # Rotations
        chair, rot_z, rot_y, rot_deg, rot_complex = rotation_examples()

        # Translations
        sphere, translated, twice_translated = translation_examples()

        # Fitting and alignment
        table, fitted_cube, fitted_rect = fitting_and_alignment()

        # Chained transformations
        lamp, final_lamp = chained_transformations()

        # Preservation
        original, transformed = transformation_preservation()

        log.info("All transformation examples completed successfully!")
        log.info(
            "Tip: All transformations return new objects, preserving the originals"
        )

    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
