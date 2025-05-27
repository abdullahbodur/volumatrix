#!/usr/bin/env python3
"""
Scene Management Examples

This example demonstrates how to create and manage scenes with multiple 3D objects
in Volumatrix. Learn about positioning, transformations, visibility, and scene operations.
"""

import numpy as np
from logger import setup_logger

import volumatrix as vm

log = setup_logger(__name__)


def basic_scene_creation():
    """Demonstrate basic scene creation and management."""
    log.info("Basic scene creation...")

    # Create a new scene
    scene = vm.Scene(name="BasicDemo")
    log.info(f"Created scene: {scene.name}")

    # Add some objects
    cube = vm.generate("cube")
    sphere = vm.generate("sphere")
    cylinder = vm.generate("cylinder")

    scene.add(cube, name="Cube1", position=[0, 0, 0])
    scene.add(sphere, name="Sphere1", position=[2, 0, 0])
    scene.add(cylinder, name="Cylinder1", position=[0, 2, 0])

    log.info(f"Added objects to scene:")
    for obj in scene:
        log.info(f"   - {obj.name} at {obj.position}")

    log.info(f"Scene info:")
    log.info(f"   - Number of objects: {len(scene)}")
    log.info(f"   - Scene bounds: {scene.bounds()}")

    return scene


def scene_transformations():
    """Demonstrate scene transformations."""
    log.info("Scene transformations...")

    # Create a scene
    scene = vm.Scene(name="TransformDemo")

    # Generate base objects
    chair = vm.generate("chair")
    table = vm.generate("table")
    lamp = vm.generate("lamp")

    # Add objects with transformations
    scene.add(chair, name="Chair1", position=[0, 0, 0], rotation=[0, 0, 0])
    scene.add(table, name="Table1", position=[2, 0, 0], rotation=[0, 0, 3.14159 / 4])
    scene.add(lamp, name="Lamp1", position=[0, 2, 0], rotation=[0, 3.14159 / 4, 0])

    log.info(f"Added objects with transformations:")
    for obj in scene:
        log.info(f"   - {obj.name}:")
        log.info(f"     Position: {obj.position}")
        log.info(f"     Rotation: {obj.rotation}")

    # Get object centers after transformation
    log.info(f"Transformed object centers:")
    for obj in scene:
        center = obj.center()
        log.info(f"   - {obj.name}: {center}")

    return scene


def scene_visibility():
    """Demonstrate scene visibility control."""
    log.info("Scene visibility control...")

    # Create a scene
    scene = vm.Scene(name="VisibilityDemo")

    # Generate and add objects
    objects = []
    for i in range(3):
        chair = vm.generate("chair")
        sphere = vm.generate("sphere")
        objects.extend([chair, sphere])
        scene.add(chair, name=f"Chair{i+1}", position=[i * 2, 0, 0])
        scene.add(sphere, name=f"Sphere{i+1}", position=[i * 2, 2, 0])

    log.info(f"Added {len(objects)} objects to scene")

    # Check initial visibility
    log.info(f"Initial visibility:")
    for obj in scene:
        log.info(f"   - {obj.name}: {'visible' if obj.visible else 'hidden'}")

    # Hide some objects
    scene.hide("Sphere1")
    scene.hide("Chair3")

    log.info(f"After hiding Sphere1 and Chair3:")
    for obj in scene:
        log.info(f"   - {obj.name}: {'visible' if obj.visible else 'hidden'}")

    # Get visible objects
    visible_objects = scene.visible_objects()
    log.info(f"Visible objects: {visible_objects}")

    return scene


def scene_operations():
    """Demonstrate scene operations."""
    log.info("Scene operations...")

    # Create a scene with multiple objects
    scene = vm.Scene(name="OperationsDemo")
    chairs = []
    for i in range(3):
        chair = vm.generate("chair")
        chairs.append(chair)
        scene.add(chair, name=f"Chair{i+1}", position=[i * 2, 0, 0])

    log.info(f"Created scene with {len(scene)} objects")

    # Merge all objects into one
    merged = scene.merge()
    log.info(f"Merged scene into single object: {merged.name}")

    # Merge specific objects
    merged_chairs = scene.merge(["Chair1", "Chair2"])
    log.info(f"Merged chairs: {merged_chairs.name}")

    # Clear scene and add merged object
    scene.clear()
    scene.add(merged, name="MergedAll")
    log.info(f"Cleared scene and added merged object")

    return scene


def complex_scene_layout():
    """Create a complex scene layout."""
    log.info("Complex scene layout...")

    # Define furniture layout
    furniture = [
        {
            "name": "Sofa",
            "prompt": "modern sofa",
            "position": [0, 0, 0],
            "rotation": [0, 0, 0],
        },
        {
            "name": "CoffeeTable",
            "prompt": "coffee table",
            "position": [0, 2, 0],
            "rotation": [0, 0, 0],
        },
        {
            "name": "TVStand",
            "prompt": "tv stand",
            "position": [0, 4, 0],
            "rotation": [0, 0, 0],
        },
        {
            "name": "Bookshelf1",
            "prompt": "bookshelf",
            "position": [-3, 0, 0],
            "rotation": [0, 0, 0],
        },
        {
            "name": "Bookshelf2",
            "prompt": "bookshelf",
            "position": [3, 0, 0],
            "rotation": [0, 0, 3.14159],
        },
        {
            "name": "Lamp1",
            "prompt": "floor lamp",
            "position": [-2, 2, 0],
            "rotation": [0, 0, 0],
        },
        {
            "name": "Lamp2",
            "prompt": "floor lamp",
            "position": [2, 2, 0],
            "rotation": [0, 0, 3.14159],
        },
    ]

    # Create scene
    scene = vm.Scene(name="LivingRoom")

    # Add furniture
    for item in furniture:
        obj = vm.generate(item["prompt"])
        scene.add(
            obj, name=item["name"], position=item["position"], rotation=item["rotation"]
        )
        log.info(f"Added {item['name']}")

    log.info(f"Created complex living room scene:")
    log.info(f"   - Number of objects: {len(scene)}")
    log.info(f"   - Scene bounds: {scene.bounds()}")
    log.info(f"   - Object names: {[obj.name for obj in scene]}")

    return scene


def scene_analysis():
    """Demonstrate scene analysis capabilities."""
    log.info("Scene analysis...")

    # Create a scene with various objects
    scene = vm.Scene(name="AnalysisDemo")

    # Add objects with different properties
    objects = [
        ("Cube1", "cube", [0, 0, 0]),
        ("Sphere1", "sphere", [2, 0, 0]),
        ("Cylinder1", "cylinder", [0, 2, 0]),
        ("Chair1", "chair", [2, 2, 0]),
        ("Table1", "table", [0, 0, 2]),
    ]

    for name, prompt, pos in objects:
        obj = vm.generate(prompt)
        scene.add(obj, name=name, position=pos)

    log.info(f"Created analysis scene with {len(scene)} objects")

    # Analyze scene
    log.info(f"Scene analysis:")
    log.info(f"   - Total objects: {len(scene)}")
    log.info(f"   - Scene bounds: {scene.bounds()}")
    log.info(f"   - Scene center: {scene.center()}")
    log.info(f"   - Object types: {scene.object_types()}")

    # Analyze individual objects
    log.info(f"Object analysis:")
    for obj in scene:
        log.info(f"   - {obj.name}:")
        log.info(f"     Center: {obj.center()}")
        log.info(f"     Bounds: {obj.bounds()}")
        log.info(f"     Volume: {obj.volume():.2f}")

    return scene


def main():
    """Run all scene management examples."""
    log.info("Volumatrix Scene Management Examples")
    log.info("=" * 50)

    try:
        # Basic scene creation
        basic_scene = basic_scene_creation()

        # Scene transformations
        transform_scene = scene_transformations()

        # Scene visibility
        visibility_scene = scene_visibility()

        # Scene operations
        operations_scene = scene_operations()

        # Complex scene layout
        living_room = complex_scene_layout()

        # Scene analysis
        analysis_scene = scene_analysis()

        log.info("All scene management examples completed successfully!")

    except Exception as e:
        log.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
