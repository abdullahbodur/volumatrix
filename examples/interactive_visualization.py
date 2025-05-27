#!/usr/bin/env python3
"""
Interactive Visualization Examples

This example demonstrates the real-time interactive visualization capabilities
of Volumatrix. Objects automatically open in interactive windows when generated
or when explicitly previewed.
"""

import time

from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def basic_interactive_viewing():
    """Demonstrate basic interactive viewing of objects."""
    log.info("Basic interactive viewing...")

    # Generate objects and view them interactively
    log.info("Generating cube...")
    cube = vm.generate("cube")

    log.info("Opening interactive window for cube...")
    vm.show(cube)  # Opens interactive window

    log.info("Generating sphere...")
    sphere = vm.generate("sphere")

    log.info("Opening interactive window for sphere...")
    vm.show(sphere)

    return cube, sphere


def visualization_backends():
    """Try different visualization backends."""
    log.info("Testing different visualization backends...")

    # Generate a test object
    chair = vm.generate("chair")
    log.info(f"Generated chair: {chair.name}")

    # Try PyVista (best interactive experience)
    try:
        log.info("Opening with PyVista (recommended)...")
        vm.preview(chair, backend="pyvista", window_title="PyVista Viewer")
        time.sleep(1)  # Brief pause between windows
    except Exception as e:
        log.error(f"PyVista not available: {e}")

    # Try Plotly (web-based interactive)
    try:
        log.info("Opening with Plotly...")
        vm.preview(chair, backend="plotly", window_title="Plotly Viewer")
        time.sleep(1)
    except Exception as e:
        log.error(f"Plotly not available: {e}")

    # Try Trimesh (simple but efficient visualization)
    try:
        log.info("Opening with Trimesh...")
        vm.preview(chair, backend="trimesh", window_title="Trimesh Viewer")
        time.sleep(1)
    except Exception as e:
        log.error(f"Trimesh not available: {e}")

    # Try Matplotlib (basic 3D)
    try:
        log.info("Opening with Matplotlib...")
        vm.preview(chair, backend="matplotlib", window_title="Matplotlib Viewer")
    except Exception as e:
        log.error(f"Matplotlib not available: {e}")

    return chair


def scene_visualization():
    """Demonstrate interactive scene visualization."""
    log.info("Interactive scene visualization...")

    # Create a scene with multiple objects
    scene = vm.Scene(name="InteractiveDemo")

    # Add objects to the scene
    cube = vm.generate("cube")
    sphere = vm.generate("sphere")
    cylinder = vm.generate("cylinder")

    scene.add(cube, name="Cube", position=[0, 0, 0])
    scene.add(sphere, name="Sphere", position=[3, 0, 0])
    scene.add(cylinder, name="Cylinder", position=[0, 3, 0])

    log.info(f"Created scene with {len(scene)} objects")

    # Visualize the entire scene
    log.info("Opening interactive scene viewer...")
    vm.show(scene, window_title="Interactive Scene")

    return scene


def different_representations():
    """Show objects with different 3D representations."""
    log.info("Visualizing different representations...")

    # Start with a mesh
    original = vm.generate("table", output_format="mesh")
    log.info("Viewing original mesh...")
    vm.show(original, window_title="Mesh Representation")

    # Convert to point cloud
    pc_version = vm.mesh_to_pointcloud(original, num_points=1000)
    log.info("Viewing as point cloud...")
    vm.show(pc_version, window_title="Point Cloud Representation")

    # Convert to voxels
    voxel_version = vm.voxelize(original, resolution=16)
    log.info("Viewing as voxels...")
    vm.show(voxel_version, window_title="Voxel Representation")

    return original, pc_version, voxel_version


def transformation_visualization():
    """Visualize objects undergoing transformations."""
    log.info("Transformation visualization...")

    # Generate base object
    lamp = vm.generate("lamp")
    log.info("Original lamp...")
    vm.show(lamp, window_title="Original Lamp")

    # Apply and visualize transformations
    normalized = vm.normalize(lamp)
    log.info("Normalized lamp...")
    vm.show(normalized, window_title="Normalized Lamp")

    scaled = vm.rescale(lamp, 2.0)
    log.info("Scaled lamp (2x)...")
    vm.show(scaled, window_title="Scaled Lamp")

    rotated = vm.rotate(lamp, [0, 0, 3.14159 / 4])  # 45 degrees
    log.info("Rotated lamp (45°)...")
    vm.show(rotated, window_title="Rotated Lamp")

    return lamp, normalized, scaled, rotated


def batch_visualization():
    """Visualize multiple objects from batch generation."""
    log.info("Batch visualization...")

    # Generate multiple objects
    prompts = ["cube", "sphere", "cylinder", "chair"]
    objects = vm.generate_batch(prompts)

    log.info(f"Generated {len(objects)} objects")

    # Visualize each object individually
    for prompt, obj in zip(prompts, objects):
        log.info(f"Viewing {prompt}...")
        vm.show(obj, window_title=f"Batch Object: {prompt.title()}")
        time.sleep(0.5)  # Brief pause between windows

    # Create a scene with all objects
    batch_scene = vm.Scene(name="BatchScene")
    for i, (prompt, obj) in enumerate(zip(prompts, objects)):
        batch_scene.add(obj, name=prompt.title(), position=[i * 2, 0, 0])

    log.info("Viewing all objects in one scene...")
    vm.show(batch_scene, window_title="Batch Scene")

    return objects, batch_scene


def custom_visualization_options():
    """Demonstrate custom visualization options."""
    log.info("Custom visualization options...")

    # Generate object
    vase = vm.generate("vase")

    # PyVista with custom options
    try:
        log.info("Custom PyVista visualization...")
        vm.preview(
            vase,
            backend="pyvista",
            window_title="Custom Vase Viewer",
            window_size=(1000, 800),
            background_color="lightgray",
            show_axes=True,
            show_grid=True,
        )
    except Exception as e:
        log.error(f"Custom PyVista not available: {e}")

    # Plotly with custom size
    try:
        log.info("Custom Plotly visualization...")
        vm.preview(
            vase,
            backend="plotly",
            window_title="Custom Vase (Plotly)",
            window_size=(900, 700),
        )
    except Exception as e:
        log.error(f"Custom Plotly not available: {e}")

    return vase


def auto_backend_selection():
    """Demonstrate automatic backend selection."""
    log.info("Automatic backend selection...")

    # Generate object
    bookshelf = vm.generate("bookshelf")

    # Let Volumatrix choose the best available backend
    log.info("Using automatic backend selection...")
    vm.show(bookshelf)  # Will use the best available backend

    # You can also use vm.preview() with backend="auto"
    log.info("Explicit auto backend...")
    vm.preview(bookshelf, backend="auto", window_title="Auto Backend Selection")

    return bookshelf


def main():
    """Run all interactive visualization examples."""
    log.info("Volumatrix Interactive Visualization Examples")
    log.info("=" * 50)
    log.info("Note: Multiple windows will open - close them to continue")
    log.info("=" * 50)

    try:
        # Basic viewing
        cube, sphere = basic_interactive_viewing()

        # Different backends
        chair = visualization_backends()

        # Scene visualization
        scene = scene_visualization()

        # Different representations
        original, pc, voxels = different_representations()

        # Transformations
        lamp, normalized, scaled, rotated = transformation_visualization()

        # Batch visualization
        batch_objects, batch_scene = batch_visualization()

        # Custom options
        vase = custom_visualization_options()

        # Auto backend
        bookshelf = auto_backend_selection()

        log.info("All interactive visualization examples completed!")

    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
