#!/usr/bin/env python3
"""
Conversion Examples

This example demonstrates how to convert between different 3D representations
in Volumatrix: meshes, voxels, and point clouds.
"""

import numpy as np
import volumatrix as vm


def mesh_to_other_formats():
  """Convert mesh to other representations."""
  print("Converting mesh to other formats...")

  # Start with a mesh
  sphere = vm.generate("sphere", output_format="mesh")
  print(f"Generated mesh sphere: {sphere.name}")
  print(f"   - Vertices: {sphere.mesh.num_vertices}")
  print(f"   - Faces: {sphere.mesh.num_faces}")

  # Convert to point cloud
  pc_sphere = vm.mesh_to_pointcloud(sphere, num_points=1000)
  print(f"Converted to point cloud:")
  print(f"   - Points: {pc_sphere.pointcloud.num_points}")
  print(f"   - Has mesh: {pc_sphere.has_representation('mesh')}")
  print(f"   - Has point cloud: {pc_sphere.has_representation('pointcloud')}")

  # Convert to voxels
  voxel_sphere = vm.voxelize(sphere, resolution=32)
  print(f"Converted to voxels:")
  print(f"   - Resolution: {voxel_sphere.voxel.resolution}")
  print(f"   - Occupied voxels: {voxel_sphere.voxel.num_occupied}")
  print(f"   - Has mesh: {voxel_sphere.has_representation('mesh')}")
  print(f"   - Has voxels: {voxel_sphere.has_representation('voxel')}")

  return sphere, pc_sphere, voxel_sphere


def pointcloud_conversions():
  """Convert point clouds to other formats."""
  print("\nPoint cloud conversions...")

  # Start with a point cloud
  chair = vm.generate("chair", output_format="pointcloud")
  print(f"Generated point cloud chair: {chair.name}")
  print(f"   - Points: {chair.pointcloud.num_points}")

  # Convert point cloud to mesh using different methods
  mesh_delaunay = vm.pointcloud_to_mesh(chair, method="delaunay")
  print(f"Converted to mesh (Delaunay):")
  print(f"   - Vertices: {mesh_delaunay.mesh.num_vertices}")
  print(f"   - Faces: {mesh_delaunay.mesh.num_faces}")

  # Try different point cloud sampling methods
  cube = vm.generate("cube", output_format="mesh")

  # Surface sampling
  pc_surface = vm.mesh_to_pointcloud(cube, num_points=500, method="surface")
  print(f"Surface sampling: {pc_surface.pointcloud.num_points} points")

  # Vertex sampling
  pc_vertices = vm.mesh_to_pointcloud(cube, num_points=100, method="vertices")
  print(f"Vertex sampling: {pc_vertices.pointcloud.num_points} points")

  # Random sampling
  pc_random = vm.mesh_to_pointcloud(cube, num_points=300, method="random")
  print(f"Random sampling: {pc_random.pointcloud.num_points} points")

  return chair, mesh_delaunay, pc_surface, pc_vertices, pc_random


def voxel_conversions():
  """Convert voxels to other formats."""
  print("\nVoxel conversions...")

  # Start with voxels
  table = vm.generate("table", output_format="voxel", resolution=24)
  print(f"Generated voxel table: {table.name}")
  print(f"   - Resolution: {table.voxel.resolution}")
  print(f"   - Occupied: {table.voxel.num_occupied}")

  # Convert voxels to mesh
  mesh_table = vm.devoxelize(table)
  print(f"Converted to mesh:")
  print(f"   - Vertices: {mesh_table.mesh.num_vertices}")
  print(f"   - Faces: {mesh_table.mesh.num_faces}")

  # Test different voxel resolutions
  lamp = vm.generate("lamp", output_format="mesh")

  low_res = vm.voxelize(lamp, resolution=8)
  med_res = vm.voxelize(lamp, resolution=16)
  high_res = vm.voxelize(lamp, resolution=32)

  print(f"Voxelization at different resolutions:")
  print(f"   - 8³: {low_res.voxel.num_occupied} occupied voxels")
  print(f"   - 16³: {med_res.voxel.num_occupied} occupied voxels")
  print(f"   - 32³: {high_res.voxel.num_occupied} occupied voxels")

  return table, mesh_table, low_res, med_res, high_res


def conversion_pipeline():
  """Demonstrate a complete conversion pipeline."""
  print("\nComplete conversion pipeline...")

  # Start with a mesh
  original = vm.generate("vase", output_format="mesh")
  print(f"Original mesh: {original.name}")
  print(f"   - Representations: {list(original.representations.keys())}")

  # Step 1: Mesh → Point Cloud
  step1 = vm.mesh_to_pointcloud(original, num_points=800)
  print(f"Step 1 - Added point cloud")
  print(f"   - Representations: {list(step1.representations.keys())}")

  # Step 2: Point Cloud → Mesh (reconstruction)
  step2 = vm.pointcloud_to_mesh(step1, method="delaunay")
  print(f"Step 2 - Reconstructed mesh from point cloud")

  # Step 3: Mesh → Voxels
  step3 = vm.voxelize(step2, resolution=20)
  print(f"Step 3 - Added voxel representation")
  print(f"   - Representations: {list(step3.representations.keys())}")

  # Step 4: Voxels → Mesh (reconstruction)
  final = vm.devoxelize(step3)
  print(f"Step 4 - Reconstructed mesh from voxels")
  print(f"   - Final representations: {list(final.representations.keys())}")

  # Compare original and final
  orig_bounds = original.bounds()
  final_bounds = final.bounds()
  print(f"Comparison:")
  print(f"   - Original bounds: {orig_bounds}")
  print(f"   - Final bounds: {final_bounds}")

  return original, step1, step2, step3, final


def conversion_with_parameters():
  """Show conversion with different parameters."""
  print("\nConversion parameters...")

  # Generate a complex object
  bookshelf = vm.generate("bookshelf", output_format="mesh")
  print(f"Generated bookshelf: {bookshelf.name}")

  # Point cloud with different densities
  sparse_pc = vm.mesh_to_pointcloud(bookshelf, num_points=200)
  dense_pc = vm.mesh_to_pointcloud(bookshelf, num_points=2000)

  print(f"Point cloud densities:")
  print(f"   - Sparse: {sparse_pc.pointcloud.num_points} points")
  print(f"   - Dense: {dense_pc.pointcloud.num_points} points")

  # Voxelization with different resolutions
  coarse_voxels = vm.voxelize(bookshelf, resolution=12)
  fine_voxels = vm.voxelize(bookshelf, resolution=48)

  print(f"Voxel resolutions:")
  print(f"   - Coarse (12³): {coarse_voxels.voxel.num_occupied} occupied")
  print(f"   - Fine (48³): {fine_voxels.voxel.num_occupied} occupied")

  return bookshelf, sparse_pc, dense_pc, coarse_voxels, fine_voxels


def conversion_aliases():
  """Demonstrate conversion alias functions."""
  print("\nConversion aliases...")

  # Generate object
  cylinder = vm.generate("cylinder", output_format="mesh")
  print(f"Generated cylinder: {cylinder.name}")

  # Use alias functions
  voxel_cylinder = vm.mesh_to_voxel(cylinder, resolution=16)
  print(f"mesh_to_voxel alias: {voxel_cylinder.voxel.num_occupied} voxels")

  mesh_cylinder = vm.voxel_to_mesh(voxel_cylinder)
  print(f"voxel_to_mesh alias: {mesh_cylinder.mesh.num_vertices} vertices")

  # Show that these are equivalent to the full function names
  voxel_full = vm.voxelize(cylinder, resolution=16)
  mesh_full = vm.devoxelize(voxel_full)

  print(f"Equivalence check:")
  print(
    f"   - Alias voxels == Full voxels: {voxel_cylinder.voxel.num_occupied == voxel_full.voxel.num_occupied}")
  print(
    f"   - Alias mesh == Full mesh: {mesh_cylinder.mesh.num_vertices == mesh_full.mesh.num_vertices}")

  return cylinder, voxel_cylinder, mesh_cylinder


def main():
  """Run all conversion examples."""
  print("Volumatrix Conversion Examples")
  print("=" * 50)

  try:
    # Mesh conversions
    sphere, pc_sphere, voxel_sphere = mesh_to_other_formats()

    # Point cloud conversions
    chair, mesh_chair, pc_surf, pc_vert, pc_rand = pointcloud_conversions()

    # Voxel conversions
    table, mesh_table, low_res, med_res, high_res = voxel_conversions()

    # Complete pipeline
    orig, step1, step2, step3, final = conversion_pipeline()

    # Parameter variations
    bookshelf, sparse, dense, coarse, fine = conversion_with_parameters()

    # Aliases
    cylinder, voxel_cyl, mesh_cyl = conversion_aliases()

    print("\nAll conversion examples completed successfully!")
    print("Tips:")
    print("   - Conversions preserve original representations")
    print("   - Higher resolutions = more detail but larger memory usage")
    print("   - Different methods suit different use cases")

  except Exception as e:
    print(f"Error running examples: {e}")
    raise


if __name__ == "__main__":
  main()
