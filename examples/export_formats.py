#!/usr/bin/env python3
"""
Export Formats Examples
This example demonstrates how to export 3D objects to various file formats
using Volumatrix. Learn about different export options and file formats.
"""
import tempfile
from pathlib import Path

from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def basic_export_formats():
    """Export objects to basic file formats."""
    log.info("Starting basic export formats demonstration")
    # Generate a test object
    chair = vm.generate("chair")
    log.debug(f"Generated chair: {chair.name}")
    # Create temporary directory for exports
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        log.debug(f"Using temporary directory: {temp_path}")
        # Export to OBJ format
        obj_file = temp_path / "chair.obj"
        vm.export(chair, str(obj_file))
        log.debug(f"Exported to OBJ: {obj_file.name} ({obj_file.stat().st_size} bytes)")
        # Export to STL format
        stl_file = temp_path / "chair.stl"
        vm.export(chair, str(stl_file))
        log.debug(f"Exported to STL: {stl_file.name} ({stl_file.stat().st_size} bytes)")
        # Export to PLY format (if supported)
        try:
            ply_file = temp_path / "chair.ply"
            vm.export(chair, str(ply_file))
            log.debug(
                f"Exported to PLY: {ply_file.name} ({ply_file.stat().st_size} bytes)"
            )
        except Exception as e:
            log.warning(f"PLY export not available: {e}")
        # List all exported files
        exported_files = list(temp_path.glob("chair.*"))
        log.info(f"Successfully exported {len(exported_files)} files in basic formats")
        return exported_files


def export_different_objects():
    """Export different types of objects."""
    log.info("Starting export of different object types")
    # Generate different objects
    objects = {
        "sphere": vm.generate("sphere"),
        "cube": vm.generate("cube"),
        "table": vm.generate("table"),
        "lamp": vm.generate("lamp"),
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for name, obj in objects.items():
            # Export each object to OBJ
            obj_file = temp_path / f"{name}.obj"
            vm.export(obj, str(obj_file))
            file_size = obj_file.stat().st_size
            log.debug(f"Exported {name}: {file_size} bytes")
            # Check mesh properties
            if obj.has_representation("mesh"):
                mesh = obj.mesh
                log.debug(
                    f"   - Vertices: {mesh.num_vertices}, Faces: {mesh.num_faces}"
                )
        # List all files
        all_files = list(temp_path.glob("*.obj"))
        log.info(f"Successfully exported {len(all_files)} different objects")
        return all_files


def export_with_transformations():
    """Export objects after applying transformations."""
    log.info("Starting export of transformed objects")
    # Generate base object
    vase = vm.generate("vase")
    log.debug(f"Generated base vase: {vase.name}")
    # Apply different transformations
    normalized = vm.normalize(vase)
    scaled = vm.rescale(vase, 2.0)
    rotated = vm.rotate(vase, [0, 0, 3.14159 / 4])  # 45 degrees
    transformations = {
        "original": vase,
        "normalized": normalized,
        "scaled_2x": scaled,
        "rotated_45deg": rotated,
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for name, obj in transformations.items():
            obj_file = temp_path / f"vase_{name}.obj"
            vm.export(obj, str(obj_file))
            bounds = obj.bounds()
            center = obj.center()
            file_size = obj_file.stat().st_size
            log.debug(f"Exported {name}:")
            log.debug(f"   - File: {obj_file.name} ({file_size} bytes)")
            log.debug(f"   - Bounds: {bounds}")
            log.debug(f"   - Center: {center}")
        log.info(
            f"Successfully exported vase with {len(transformations)} transformations"
        )
        return list(temp_path.glob("vase_*.obj"))


def export_different_representations():
    """Export objects with different representations."""
    log.info("Starting export of different representations")
    # Generate object with multiple representations
    sphere = vm.generate("sphere", output_format="mesh")
    # Add point cloud representation
    sphere_with_pc = vm.mesh_to_pointcloud(sphere, num_points=1000)
    # Add voxel representation
    sphere_with_voxels = vm.voxelize(sphere_with_pc, resolution=32)
    log.debug(
        f"Created sphere with representations: {list(sphere_with_voxels.representations.keys())}"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Export the mesh representation
        mesh_file = temp_path / "sphere_mesh.obj"
        vm.export(sphere_with_voxels, str(mesh_file))
        log.debug(f"Exported mesh representation: {mesh_file.stat().st_size} bytes")
        log.info("Successfully exported sphere with multiple representations")
        return [mesh_file]


def export_scene_objects():
    """Export objects from a scene."""
    log.info("Starting export of scene objects")
    # Create a scene with multiple objects
    scene = vm.Scene(name="ExportDemo")
    # Add objects to scene
    chair = vm.generate("chair")
    table = vm.generate("table")
    lamp = vm.generate("lamp")
    scene.add(chair, name="Chair", position=[0, 0, 0])
    scene.add(table, name="Table", position=[2, 0, 0])
    scene.add(lamp, name="Lamp", position=[2, 0, 1])
    log.debug(f"Created scene with {len(scene)} objects")
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
            log.debug(f"Exported {obj_name}:")
            log.debug(f"   - File: {obj_file.name} ({file_size} bytes)")
            log.debug(f"   - Scene position: {position}")
        # Export merged scene as single object
        merged_scene = scene.merge_objects()
        merged_file = temp_path / "merged_scene.obj"
        vm.export(merged_scene, str(merged_file))
        log.debug(f"Exported merged scene: {merged_file.stat().st_size} bytes")
        log.info(
            f"Successfully exported scene with {len(scene)} objects and merged scene"
        )
        return list(temp_path.glob("*.obj"))


def export_batch_objects():
    """Export multiple objects in batch."""
    log.info("Starting batch export")
    # Generate multiple objects
    prompts = ["cube", "sphere", "cylinder", "chair", "table", "lamp"]
    objects = vm.generate_batch(prompts)
    log.debug(f"Generated {len(objects)} objects")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Export all objects
        exported_files = []
        for i, (prompt, obj) in enumerate(zip(prompts, objects)):
            obj_file = temp_path / f"{prompt}_{i:02d}.obj"
            vm.export(obj, str(obj_file))
            exported_files.append(obj_file)
            file_size = obj_file.stat().st_size
            log.debug(f"Exported {prompt}: {obj_file.name} ({file_size} bytes)")
        # Calculate total size
        total_size = sum(f.stat().st_size for f in exported_files)
        log.info(f"Batch export completed successfully:")
        log.debug(f"   - Files: {len(exported_files)}")
        log.debug(f"   - Total size: {total_size} bytes")
        return exported_files


def export_error_handling():
    """Demonstrate export error handling."""
    log.info("Starting export error handling demonstration")
    # Generate test object
    cube = vm.generate("cube")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Valid export
        valid_file = temp_path / "cube.obj"
        try:
            vm.export(cube, str(valid_file))
            log.debug(f"Valid export successful: {valid_file.name}")
        except Exception as e:
            log.error(f"Unexpected error in valid export: {e}")
        # Test automatic directory creation
        nested_dir = temp_path / "nested" / "directory"
        nested_file = nested_dir / "cube.obj"
        try:
            vm.export(cube, str(nested_file))
            log.debug(f"Nested directory export successful: {nested_file}")
        except Exception as e:
            log.error(f"Nested directory export failed: {e}")
        # Test unsupported format (should still work with warning)
        try:
            unknown_file = temp_path / "cube.unknown"
            vm.export(cube, str(unknown_file))
            log.debug(f"Unknown format handled: {unknown_file.name}")
        except Exception as e:
            log.warning(f"Unknown format error (expected): {e}")
        log.info("Export error handling demonstration completed")
        return [valid_file, nested_file]


def main():
    """Run all export format examples."""
    log.info("Starting Volumatrix Export Format Examples")
    log.info("=" * 50)
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
        log.info("All export format examples completed successfully!")
        log.info("Export format tips:")
        log.info("   - OBJ format is widely supported")
        log.info("   - STL is good for 3D printing")
        log.info("   - Export preserves transformations")
        log.info("   - Directories are created automatically")
    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
