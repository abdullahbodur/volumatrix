#!/usr/bin/env python3
"""
Export Formats Examples

This example demonstrates how to export 3D objects to various file formats
using Volumatrix. Learn about different export options and file formats.
"""

import tempfile
import os
from pathlib import Path
import volumatrix as vm


def basic_export_formats():
    """Export objects to basic file formats."""
    print("🎯 Basic export formats...")
    
    # Generate a test object
    chair = vm.generate("chair")
    print(f"✅ Generated chair: {chair.name}")
    
    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"✅ Using temporary directory: {temp_path}")
        
        # Export to OBJ format
        obj_file = temp_path / "chair.obj"
        vm.export(chair, str(obj_file))
        print(f"✅ Exported to OBJ: {obj_file.name} ({obj_file.stat().st_size} bytes)")
        
        # Export to STL format
        stl_file = temp_path / "chair.stl"
        vm.export(chair, str(stl_file))
        print(f"✅ Exported to STL: {stl_file.name} ({stl_file.stat().st_size} bytes)")
        
        # Export to PLY format (if supported)
        try:
            ply_file = temp_path / "chair.ply"
            vm.export(chair, str(ply_file))
            print(f"✅ Exported to PLY: {ply_file.name} ({ply_file.stat().st_size} bytes)")
        except Exception as e:
            print(f"⚠️  PLY export not available: {e}")
        
        # List all exported files
        exported_files = list(temp_path.glob("chair.*"))
        print(f"✅ Total exported files: {len(exported_files)}")
        
        return exported_files


def export_different_objects():
    """Export different types of objects."""
    print("\n🎯 Exporting different object types...")
    
    # Generate different objects
    objects = {
        "sphere": vm.generate("sphere"),
        "cube": vm.generate("cube"),
        "table": vm.generate("table"),
        "lamp": vm.generate("lamp")
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for name, obj in objects.items():
            # Export each object to OBJ
            obj_file = temp_path / f"{name}.obj"
            vm.export(obj, str(obj_file))
            
            file_size = obj_file.stat().st_size
            print(f"✅ Exported {name}: {file_size} bytes")
            
            # Check mesh properties
            if obj.has_representation("mesh"):
                mesh = obj.mesh
                print(f"   - Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}")
        
        # List all files
        all_files = list(temp_path.glob("*.obj"))
        print(f"✅ Total objects exported: {len(all_files)}")
        
        return all_files


def export_with_transformations():
    """Export objects after applying transformations."""
    print("\n🎯 Exporting transformed objects...")
    
    # Generate base object
    vase = vm.generate("vase")
    print(f"✅ Generated base vase: {vase.name}")
    
    # Apply different transformations
    normalized = vm.normalize(vase)
    scaled = vm.rescale(vase, 2.0)
    rotated = vm.rotate(vase, [0, 0, 3.14159/4])  # 45 degrees
    
    transformations = {
        "original": vase,
        "normalized": normalized,
        "scaled_2x": scaled,
        "rotated_45deg": rotated
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for name, obj in transformations.items():
            obj_file = temp_path / f"vase_{name}.obj"
            vm.export(obj, str(obj_file))
            
            bounds = obj.bounds()
            center = obj.center()
            file_size = obj_file.stat().st_size
            
            print(f"✅ Exported {name}:")
            print(f"   - File: {obj_file.name} ({file_size} bytes)")
            print(f"   - Bounds: {bounds}")
            print(f"   - Center: {center}")
        
        return list(temp_path.glob("vase_*.obj"))


def export_different_representations():
    """Export objects with different representations."""
    print("\n🎯 Exporting different representations...")
    
    # Generate object with multiple representations
    sphere = vm.generate("sphere", output_format="mesh")
    
    # Add point cloud representation
    sphere_with_pc = vm.mesh_to_pointcloud(sphere, num_points=1000)
    
    # Add voxel representation
    sphere_with_voxels = vm.voxelize(sphere_with_pc, resolution=32)
    
    print(f"✅ Created sphere with representations: {list(sphere_with_voxels.representations.keys())}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export the mesh representation
        mesh_file = temp_path / "sphere_mesh.obj"
        vm.export(sphere_with_voxels, str(mesh_file))
        print(f"✅ Exported mesh representation: {mesh_file.stat().st_size} bytes")
        
        # Note: Point cloud and voxel exports would require specific formats
        # For now, the export function uses the mesh representation
        
        return [mesh_file]


def export_scene_objects():
    """Export objects from a scene."""
    print("\n🎯 Exporting scene objects...")
    
    # Create a scene with multiple objects
    scene = vm.Scene(name="ExportDemo")
    
    # Add objects to scene
    chair = vm.generate("chair")
    table = vm.generate("table")
    lamp = vm.generate("lamp")
    
    scene.add(chair, name="Chair", position=[0, 0, 0])
    scene.add(table, name="Table", position=[2, 0, 0])
    scene.add(lamp, name="Lamp", position=[2, 0, 1])
    
    print(f"✅ Created scene with {len(scene)} objects")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export individual objects from scene
        for obj_name in scene.list_objects():
            # Get the transformed object
            obj = scene.get_transformed_object(obj_name)
            
            # Export to file
            obj_file = temp_path / f"scene_{obj_name.lower()}.obj"
            vm.export(obj, str(obj_file))
            
            file_size = obj_file.stat().st_size
            position = scene.nodes[obj_name].position
            print(f"✅ Exported {obj_name}:")
            print(f"   - File: {obj_file.name} ({file_size} bytes)")
            print(f"   - Scene position: {position}")
        
        # Export merged scene as single object
        merged_scene = scene.merge_objects()
        merged_file = temp_path / "merged_scene.obj"
        vm.export(merged_scene, str(merged_file))
        print(f"✅ Exported merged scene: {merged_file.stat().st_size} bytes")
        
        return list(temp_path.glob("*.obj"))


def export_batch_objects():
    """Export multiple objects in batch."""
    print("\n🎯 Batch export...")
    
    # Generate multiple objects
    prompts = ["cube", "sphere", "cylinder", "chair", "table", "lamp"]
    objects = vm.generate_batch(prompts)
    
    print(f"✅ Generated {len(objects)} objects")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Export all objects
        exported_files = []
        for i, (prompt, obj) in enumerate(zip(prompts, objects)):
            obj_file = temp_path / f"{prompt}_{i:02d}.obj"
            vm.export(obj, str(obj_file))
            exported_files.append(obj_file)
            
            file_size = obj_file.stat().st_size
            print(f"✅ Exported {prompt}: {obj_file.name} ({file_size} bytes)")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in exported_files)
        print(f"✅ Batch export complete:")
        print(f"   - Files: {len(exported_files)}")
        print(f"   - Total size: {total_size} bytes")
        
        return exported_files


def export_error_handling():
    """Demonstrate export error handling."""
    print("\n🎯 Export error handling...")
    
    # Generate test object
    cube = vm.generate("cube")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Valid export
        valid_file = temp_path / "cube.obj"
        try:
            vm.export(cube, str(valid_file))
            print(f"✅ Valid export successful: {valid_file.name}")
        except Exception as e:
            print(f"❌ Unexpected error in valid export: {e}")
        
        # Test automatic directory creation
        nested_dir = temp_path / "nested" / "directory"
        nested_file = nested_dir / "cube.obj"
        try:
            vm.export(cube, str(nested_file))
            print(f"✅ Nested directory export successful: {nested_file}")
        except Exception as e:
            print(f"❌ Nested directory export failed: {e}")
        
        # Test unsupported format (should still work with warning)
        try:
            unknown_file = temp_path / "cube.unknown"
            vm.export(cube, str(unknown_file))
            print(f"✅ Unknown format handled: {unknown_file.name}")
        except Exception as e:
            print(f"⚠️  Unknown format error (expected): {e}")
        
        return [valid_file, nested_file]


def main():
    """Run all export format examples."""
    print("🚀 Volumatrix Export Format Examples")
    print("=" * 50)
    
    try:
        # Basic formats
        basic_files = basic_export_formats()
        
        # Different objects
        object_files = export_different_objects()
        
        # Transformed objects
        transform_files = export_with_transformations()
        
        # Different representations
        repr_files = export_different_representations()
        
        # Scene objects
        scene_files = export_scene_objects()
        
        # Batch export
        batch_files = export_batch_objects()
        
        # Error handling
        error_files = export_error_handling()
        
        print("\n🎉 All export format examples completed successfully!")
        print("💡 Tips:")
        print("   - OBJ format is widely supported")
        print("   - STL is good for 3D printing")
        print("   - Export preserves transformations")
        print("   - Directories are created automatically")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main() 