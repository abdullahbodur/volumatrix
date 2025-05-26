#!/usr/bin/env python3
"""
Batch Processing Examples

This example demonstrates how to efficiently process multiple 3D objects
in Volumatrix using batch operations and parallel processing techniques.
"""

import time
import tempfile
from pathlib import Path
import volumatrix as vm


def basic_batch_generation():
    """Generate multiple objects in batch."""
    print("🎯 Basic batch generation...")
    
    # Define prompts for batch generation
    prompts = [
        "cube", "sphere", "cylinder", "chair", "table", 
        "lamp", "vase", "bookshelf", "sofa", "bed"
    ]
    
    print(f"✅ Generating {len(prompts)} objects...")
    start_time = time.time()
    
    # Generate all objects in batch
    objects = vm.generate_batch(prompts)
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    print(f"✅ Batch generation completed:")
    print(f"   - Objects: {len(objects)}")
    print(f"   - Time: {generation_time:.2f} seconds")
    print(f"   - Average: {generation_time/len(objects):.2f} seconds per object")
    
    # Show details for each object
    for prompt, obj in zip(prompts, objects):
        print(f"   - {prompt}: {obj.name} ({list(obj.representations.keys())})")
    
    return objects, prompts


def batch_with_seeds():
    """Generate batch with reproducible seeds."""
    print("\n🎯 Batch generation with seeds...")
    
    prompts = ["dragon", "castle", "tree", "car", "airplane"]
    seeds = [42, 123, 456, 789, 101112]
    
    print(f"✅ Generating {len(prompts)} objects with specific seeds...")
    
    # Generate with seeds
    objects = vm.generate_batch(prompts, seeds=seeds)
    
    print(f"✅ Generated objects with seeds:")
    for prompt, seed, obj in zip(prompts, seeds, objects):
        print(f"   - {prompt} (seed {seed}): {obj.name}")
    
    # Verify reproducibility by generating again
    print(f"✅ Verifying reproducibility...")
    objects2 = vm.generate_batch(prompts, seeds=seeds)
    
    matches = 0
    for obj1, obj2 in zip(objects, objects2):
        if obj1.name == obj2.name:
            matches += 1
    
    print(f"   - Matching names: {matches}/{len(objects)}")
    
    return objects, objects2


def batch_transformations():
    """Apply transformations to multiple objects in batch."""
    print("\n🎯 Batch transformations...")
    
    # Generate base objects
    base_objects = vm.generate_batch(["cube", "sphere", "cylinder", "chair"])
    print(f"✅ Generated {len(base_objects)} base objects")
    
    # Apply different transformations to each
    transformations = [
        lambda obj: vm.normalize(obj),
        lambda obj: vm.rescale(obj, 2.0),
        lambda obj: vm.rotate(obj, [0, 0, 3.14159/4]),
        lambda obj: vm.translate(obj, [1, 1, 1])
    ]
    
    transformation_names = ["normalized", "scaled_2x", "rotated_45deg", "translated"]
    
    print(f"✅ Applying transformations...")
    transformed_objects = []
    
    for obj, transform, name in zip(base_objects, transformations, transformation_names):
        start_time = time.time()
        transformed = transform(obj)
        end_time = time.time()
        
        transformed_objects.append(transformed)
        print(f"   - {name}: {end_time - start_time:.3f}s")
    
    # Compare bounds before and after
    print(f"✅ Transformation comparison:")
    for i, (original, transformed, name) in enumerate(zip(base_objects, transformed_objects, transformation_names)):
        orig_bounds = original.bounds()
        trans_bounds = transformed.bounds()
        print(f"   - Object {i} ({name}):")
        print(f"     * Original bounds: {orig_bounds}")
        print(f"     * Transformed bounds: {trans_bounds}")
    
    return base_objects, transformed_objects


def batch_conversions():
    """Convert multiple objects between representations."""
    print("\n🎯 Batch conversions...")
    
    # Generate objects with different initial formats
    mesh_objects = vm.generate_batch(["sphere", "cube"], output_format="mesh")
    voxel_objects = vm.generate_batch(["cylinder", "chair"], output_format="voxel", resolution=16)
    pc_objects = vm.generate_batch(["table", "lamp"], output_format="pointcloud")
    
    all_objects = mesh_objects + voxel_objects + pc_objects
    object_types = ["mesh", "mesh", "voxel", "voxel", "pointcloud", "pointcloud"]
    
    print(f"✅ Generated {len(all_objects)} objects with different representations")
    
    # Convert all to have mesh representation
    print(f"✅ Converting all to mesh representation...")
    mesh_converted = []
    
    for obj, obj_type in zip(all_objects, object_types):
        start_time = time.time()
        
        if obj_type == "voxel" and not obj.has_representation("mesh"):
            converted = vm.devoxelize(obj)
        elif obj_type == "pointcloud" and not obj.has_representation("mesh"):
            converted = vm.pointcloud_to_mesh(obj)
        else:
            converted = obj  # Already has mesh
        
        end_time = time.time()
        mesh_converted.append(converted)
        
        print(f"   - {obj_type} → mesh: {end_time - start_time:.3f}s")
    
    # Add point cloud representation to all
    print(f"✅ Adding point cloud representation to all...")
    pc_added = []
    
    for obj in mesh_converted:
        if not obj.has_representation("pointcloud"):
            converted = vm.mesh_to_pointcloud(obj, num_points=500)
        else:
            converted = obj
        pc_added.append(converted)
    
    # Show final representations
    print(f"✅ Final representations:")
    for i, obj in enumerate(pc_added):
        representations = list(obj.representations.keys())
        print(f"   - Object {i}: {representations}")
    
    return all_objects, mesh_converted, pc_added


def batch_export():
    """Export multiple objects in batch."""
    print("\n🎯 Batch export...")
    
    # Generate objects for export
    furniture_prompts = ["chair", "table", "sofa", "bed", "desk", "lamp"]
    furniture_objects = vm.generate_batch(furniture_prompts)
    
    print(f"✅ Generated {len(furniture_objects)} furniture objects")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"✅ Exporting to: {temp_path}")
        
        # Export all objects
        start_time = time.time()
        exported_files = []
        
        for prompt, obj in zip(furniture_prompts, furniture_objects):
            obj_file = temp_path / f"{prompt}.obj"
            vm.export(obj, str(obj_file))
            exported_files.append(obj_file)
        
        end_time = time.time()
        export_time = end_time - start_time
        
        print(f"✅ Batch export completed:")
        print(f"   - Files: {len(exported_files)}")
        print(f"   - Time: {export_time:.2f} seconds")
        print(f"   - Average: {export_time/len(exported_files):.2f} seconds per file")
        
        # Show file sizes
        total_size = 0
        for prompt, file_path in zip(furniture_prompts, exported_files):
            file_size = file_path.stat().st_size
            total_size += file_size
            print(f"   - {prompt}: {file_size} bytes")
        
        print(f"✅ Total exported size: {total_size} bytes")
        
        return exported_files


def batch_scene_creation():
    """Create multiple scenes with batch-generated objects."""
    print("\n🎯 Batch scene creation...")
    
    # Generate objects for different room types
    living_room_objects = vm.generate_batch(["sofa", "coffee table", "tv stand", "lamp"])
    bedroom_objects = vm.generate_batch(["bed", "nightstand", "dresser", "chair"])
    kitchen_objects = vm.generate_batch(["dining table", "chair", "chair", "refrigerator"])
    
    room_configs = [
        {
            "name": "LivingRoom",
            "objects": living_room_objects,
            "names": ["Sofa", "CoffeeTable", "TVStand", "Lamp"],
            "positions": [[0, 0, 0], [0, 2, 0], [0, 4, 0], [-2, 0, 0]]
        },
        {
            "name": "Bedroom", 
            "objects": bedroom_objects,
            "names": ["Bed", "Nightstand", "Dresser", "Chair"],
            "positions": [[0, 0, 0], [2, 0, 0], [0, 3, 0], [-2, 2, 0]]
        },
        {
            "name": "Kitchen",
            "objects": kitchen_objects,
            "names": ["DiningTable", "Chair1", "Chair2", "Refrigerator"],
            "positions": [[0, 0, 0], [-1, -1, 0], [1, -1, 0], [3, 0, 0]]
        }
    ]
    
    scenes = []
    
    for config in room_configs:
        scene = vm.Scene(name=config["name"])
        
        for obj, name, position in zip(config["objects"], config["names"], config["positions"]):
            scene.add(obj, name=name, position=position)
        
        scenes.append(scene)
        print(f"✅ Created {config['name']} with {len(scene)} objects")
    
    # Analyze all scenes
    print(f"✅ Scene analysis:")
    for scene in scenes:
        bounds = scene.bounds()
        print(f"   - {scene.name}: {len(scene)} objects, bounds {bounds}")
    
    return scenes


def performance_comparison():
    """Compare performance of batch vs individual operations."""
    print("\n🎯 Performance comparison...")
    
    prompts = ["cube", "sphere", "cylinder", "chair", "table"]
    
    # Individual generation
    print(f"✅ Individual generation...")
    start_time = time.time()
    individual_objects = []
    for prompt in prompts:
        obj = vm.generate(prompt)
        individual_objects.append(obj)
    individual_time = time.time() - start_time
    
    # Batch generation
    print(f"✅ Batch generation...")
    start_time = time.time()
    batch_objects = vm.generate_batch(prompts)
    batch_time = time.time() - start_time
    
    # Compare results
    print(f"✅ Performance comparison:")
    print(f"   - Individual: {individual_time:.2f} seconds")
    print(f"   - Batch: {batch_time:.2f} seconds")
    print(f"   - Speedup: {individual_time/batch_time:.2f}x")
    print(f"   - Objects match: {len(individual_objects) == len(batch_objects)}")
    
    return individual_objects, batch_objects, individual_time, batch_time


def main():
    """Run all batch processing examples."""
    print("🚀 Volumatrix Batch Processing Examples")
    print("=" * 50)
    
    try:
        # Basic batch generation
        objects, prompts = basic_batch_generation()
        
        # Batch with seeds
        seeded_objects, seeded_objects2 = batch_with_seeds()
        
        # Batch transformations
        base_objs, transformed_objs = batch_transformations()
        
        # Batch conversions
        orig_objs, mesh_objs, pc_objs = batch_conversions()
        
        # Batch export
        exported_files = batch_export()
        
        # Batch scene creation
        scenes = batch_scene_creation()
        
        # Performance comparison
        individual, batch, ind_time, batch_time = performance_comparison()
        
        print("\n🎉 All batch processing examples completed successfully!")
        print("💡 Tips:")
        print("   - Batch operations are more efficient than individual calls")
        print("   - Use seeds for reproducible batch generation")
        print("   - Combine batch generation with transformations and exports")
        print("   - Batch processing is ideal for creating datasets")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main() 