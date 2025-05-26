#!/usr/bin/env python3
"""
Basic Generation Examples

This example demonstrates the fundamental object generation capabilities of Volumatrix.
Learn how to generate simple 3D objects from text prompts.
"""

import volumatrix as vm


def generate_simple_objects():
  """Generate basic geometric shapes."""
  print("🎯 Generating basic geometric shapes...")

  # Generate a cube
  cube = vm.generate("cube")
  print(f"✅ Generated cube: {cube.name}")
  print(f"   - Representations: {list(cube.representations.keys())}")
  print(f"   - Bounds: {cube.bounds()}")

  # Generate a sphere
  sphere = vm.generate("sphere")
  print(f"✅ Generated sphere: {sphere.name}")
  print(f"   - Center: {sphere.center()}")

  # Generate a cylinder
  cylinder = vm.generate("cylinder")
  print(f"✅ Generated cylinder: {cylinder.name}")

  return cube, sphere, cylinder


def generate_with_different_formats():
  """Generate objects in different output formats."""
  print("\n🎯 Generating objects in different formats...")

  # Generate as mesh (default)
  mesh_obj = vm.generate("chair", output_format="mesh")
  print(f"✅ Generated mesh chair: {mesh_obj.name}")
  print(f"   - Has mesh: {mesh_obj.has_representation('mesh')}")

  # Generate as voxels
  voxel_obj = vm.generate("table", output_format="voxel", resolution=32)
  print(f"✅ Generated voxel table: {voxel_obj.name}")
  print(f"   - Has voxels: {voxel_obj.has_representation('voxel')}")
  print(f"   - Voxel resolution: {voxel_obj.voxel.resolution}")

  # Generate as point cloud
  pc_obj = vm.generate("lamp", output_format="pointcloud")
  print(f"✅ Generated point cloud lamp: {pc_obj.name}")
  print(f"   - Has point cloud: {pc_obj.has_representation('pointcloud')}")
  print(f"   - Number of points: {pc_obj.pointcloud.num_points}")

  return mesh_obj, voxel_obj, pc_obj


def generate_with_seeds():
  """Generate reproducible objects using seeds."""
  print("\n🎯 Generating reproducible objects with seeds...")

  # Generate the same object twice with the same seed
  obj1 = vm.generate("dragon", seed=42)
  obj2 = vm.generate("dragon", seed=42)

  print(f"✅ Generated dragon 1: {obj1.name}")
  print(f"✅ Generated dragon 2: {obj2.name}")
  print(f"   - Names match: {obj1.name == obj2.name}")

  # Generate with different seed
  obj3 = vm.generate("dragon", seed=123)
  print(f"✅ Generated dragon 3: {obj3.name}")
  print(f"   - Different from first: {obj1.name != obj3.name}")

  return obj1, obj2, obj3


def generate_complex_objects():
  """Generate more complex objects."""
  print("\n🎯 Generating complex objects...")

  complex_prompts = [
      "wooden chair with armrests",
      "modern coffee table",
      "vintage lamp with shade",
      "decorative vase",
      "simple bookshelf"
  ]

  objects = []
  for prompt in complex_prompts:
    obj = vm.generate(prompt)
    objects.append(obj)
    print(f"✅ Generated: {obj.name}")

  return objects


def main():
  """Run all basic generation examples."""
  print("🚀 Volumatrix Basic Generation Examples")
  print("=" * 50)

  try:
    # Basic shapes
    cube, sphere, cylinder = generate_simple_objects()

    # Different formats
    mesh_obj, voxel_obj, pc_obj = generate_with_different_formats()

    # Reproducible generation
    dragon1, dragon2, dragon3 = generate_with_seeds()

    # Complex objects
    complex_objects = generate_complex_objects()

    print("\n🎉 All examples completed successfully!")
    print(f"📊 Total objects generated: {3 + 3 + 3 + len(complex_objects)}")

  except Exception as e:
    print(f"❌ Error running examples: {e}")
    raise


if __name__ == "__main__":
  main()
