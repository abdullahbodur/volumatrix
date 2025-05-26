#!/usr/bin/env python3
"""
Demo: Automatic 3D Visualization in Volumatrix

This script demonstrates how Volumatrix can automatically open 
interactive 3D windows when generating objects.
"""

import volumatrix as vm


def main():
  print("Volumatrix Auto-Visualization Demo")
  print("=" * 50)

  # Method 1: Generate with auto-preview
  print("Method 1: Auto-preview with generate()")
  print("Generating cube with automatic visualization...")
  cube = vm.generate("cube", auto_preview=True)
  print(f"Generated: {cube.name}")

  # Method 2: Convenience function
  print("\nMethod 2: Using generate_and_show()")
  print("Generating sphere with automatic visualization...")
  sphere = vm.generate_and_show("sphere")
  print(f"Generated: {sphere.name}")

  # Method 3: Manual visualization
  print("\nMethod 3: Manual visualization with show()")
  print("Generating cylinder and then showing it...")
  cylinder = vm.generate("cylinder")
  vm.show(cylinder)  # Manually show
  print(f"Generated and showed: {cylinder.name}")

  # Method 4: Scene visualization
  print("\nMethod 4: Scene visualization")
  print("Creating a scene with multiple objects...")
  scene = vm.Scene(name="AutoVisualizationDemo")
  scene.add(cube, name="Cube", position=[0, 0, 0])
  scene.add(sphere, name="Sphere", position=[2, 0, 0])
  scene.add(cylinder, name="Cylinder", position=[0, 2, 0])

  print(f"Scene created with {len(scene)} objects")
  vm.show(scene)  # Show the entire scene
  print("Scene visualization opened!")

  print("\nDemo Complete!")


if __name__ == "__main__":
  main()
