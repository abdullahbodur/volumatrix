#!/usr/bin/env python3
"""
Transformation Examples

This example demonstrates how to transform and manipulate 3D objects in Volumatrix.
Learn about scaling, rotation, translation, normalization, and more.
"""

import numpy as np
import volumatrix as vm


def basic_transformations():
    """Demonstrate basic transformation operations."""
    print("🎯 Basic transformations...")
    
    # Generate a cube to work with
    cube = vm.generate("cube")
    print(f"✅ Original cube: {cube.name}")
    print(f"   - Bounds: {cube.bounds()}")
    print(f"   - Center: {cube.center()}")
    
    # Normalize the object (center and scale to fit in [-1, 1])
    normalized = vm.normalize(cube)
    print(f"✅ Normalized cube:")
    print(f"   - New bounds: {normalized.bounds()}")
    print(f"   - New center: {normalized.center()}")
    
    # Scale the object
    scaled = vm.rescale(cube, 2.0)
    print(f"✅ Scaled cube (2x):")
    print(f"   - New bounds: {scaled.bounds()}")
    
    # Non-uniform scaling
    stretched = vm.rescale(cube, [3.0, 1.0, 0.5])
    print(f"✅ Stretched cube (3x, 1x, 0.5x):")
    print(f"   - New bounds: {stretched.bounds()}")
    
    return cube, normalized, scaled, stretched


def rotation_examples():
    """Demonstrate rotation operations."""
    print("\n🎯 Rotation examples...")
    
    # Generate a chair to make rotations more visible
    chair = vm.generate("chair")
    print(f"✅ Original chair: {chair.name}")
    
    # Rotate around Z-axis (90 degrees)
    rotated_z = vm.rotate(chair, [0, 0, np.pi/2])
    print(f"✅ Rotated 90° around Z-axis")
    
    # Rotate around Y-axis (45 degrees)
    rotated_y = vm.rotate(chair, [0, np.pi/4, 0])
    print(f"✅ Rotated 45° around Y-axis")
    
    # Rotate using degrees instead of radians
    rotated_degrees = vm.rotate(chair, [0, 0, 90], degrees=True)
    print(f"✅ Rotated 90° using degrees")
    
    # Complex rotation (around all axes)
    complex_rotation = vm.rotate(chair, [np.pi/6, np.pi/4, np.pi/3])
    print(f"✅ Complex rotation around all axes")
    
    return chair, rotated_z, rotated_y, rotated_degrees, complex_rotation


def translation_examples():
    """Demonstrate translation operations."""
    print("\n🎯 Translation examples...")
    
    # Generate objects to translate
    sphere = vm.generate("sphere")
    print(f"✅ Original sphere center: {sphere.center()}")
    
    # Simple translation
    translated = vm.translate(sphere, [2.0, 3.0, 1.0])
    print(f"✅ Translated sphere center: {translated.center()}")
    
    # Multiple translations
    twice_translated = vm.translate(translated, [-1.0, -1.0, 2.0])
    print(f"✅ Twice translated center: {twice_translated.center()}")
    
    return sphere, translated, twice_translated


def fitting_and_alignment():
    """Demonstrate fitting objects into specific spaces."""
    print("\n🎯 Fitting and alignment examples...")
    
    # Generate a large object
    table = vm.generate("table")
    original_bounds = table.bounds()
    print(f"✅ Original table bounds: {original_bounds}")
    
    # Fit into a unit cube
    fitted_cube = vm.fit_in_box(table, 1.0)
    new_bounds = fitted_cube.bounds()
    print(f"✅ Fitted in unit cube: {new_bounds}")
    
    # Fit into a rectangular box
    fitted_rect = vm.fit_in_box(table, [2.0, 1.0, 0.5])
    rect_bounds = fitted_rect.bounds()
    print(f"✅ Fitted in rectangle: {rect_bounds}")
    
    return table, fitted_cube, fitted_rect


def chained_transformations():
    """Demonstrate chaining multiple transformations."""
    print("\n🎯 Chained transformations...")
    
    # Start with a lamp
    lamp = vm.generate("lamp")
    print(f"✅ Original lamp: {lamp.name}")
    print(f"   - Bounds: {lamp.bounds()}")
    
    # Chain multiple transformations
    transformed = lamp
    
    # Step 1: Normalize
    transformed = vm.normalize(transformed)
    print(f"✅ Step 1 - Normalized")
    
    # Step 2: Scale up
    transformed = vm.rescale(transformed, 1.5)
    print(f"✅ Step 2 - Scaled 1.5x")
    
    # Step 3: Rotate
    transformed = vm.rotate(transformed, [0, np.pi/4, 0])
    print(f"✅ Step 3 - Rotated 45° around Y")
    
    # Step 4: Translate
    transformed = vm.translate(transformed, [1.0, 0.5, -0.5])
    print(f"✅ Step 4 - Translated")
    
    # Step 5: Fit in box
    transformed = vm.fit_in_box(transformed, [2.0, 2.0, 2.0])
    print(f"✅ Step 5 - Fitted in 2x2x2 box")
    
    print(f"   - Final bounds: {transformed.bounds()}")
    print(f"   - Final center: {transformed.center()}")
    
    return lamp, transformed


def transformation_preservation():
    """Show how transformations preserve object properties."""
    print("\n🎯 Transformation preservation...")
    
    # Generate object with multiple representations
    obj = vm.generate("sphere", output_format="mesh")
    
    # Add more representations
    obj = vm.mesh_to_pointcloud(obj, num_points=500)
    obj = vm.voxelize(obj, resolution=16)
    
    print(f"✅ Original object representations: {list(obj.representations.keys())}")
    
    # Apply transformation
    transformed = vm.rescale(obj, 2.0)
    
    print(f"✅ Transformed object representations: {list(transformed.representations.keys())}")
    print("   - All representations are preserved and transformed!")
    
    return obj, transformed


def main():
    """Run all transformation examples."""
    print("🚀 Volumatrix Transformation Examples")
    print("=" * 50)
    
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
        
        print("\n🎉 All transformation examples completed successfully!")
        print("💡 Tip: All transformations return new objects, preserving the originals")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main() 