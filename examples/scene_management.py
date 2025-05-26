#!/usr/bin/env python3
"""
Scene Management Examples

This example demonstrates how to create and manage scenes with multiple 3D objects
in Volumatrix. Learn about positioning, transformations, visibility, and scene operations.
"""

import numpy as np
import volumatrix as vm


def basic_scene_creation():
  """Create a basic scene with multiple objects."""
  print("🎯 Basic scene creation...")

  # Create a new scene
  scene = vm.Scene(name="FurnitureRoom")
  print(f"✅ Created scene: {scene.name}")

  # Generate some furniture objects
  chair = vm.generate("chair")
  table = vm.generate("table")
  lamp = vm.generate("lamp")

  # Add objects to the scene
  chair_name = scene.add(chair, name="DiningChair", position=[0, 0, 0])
  table_name = scene.add(table, name="DiningTable", position=[2, 0, 0])
  lamp_name = scene.add(lamp, name="TableLamp", position=[2, 0, 1])

  print(f"✅ Added objects to scene:")
  print(f"   - {chair_name} at [0, 0, 0]")
  print(f"   - {table_name} at [2, 0, 0]")
  print(f"   - {lamp_name} at [2, 0, 1]")

  # Scene information
  print(f"✅ Scene info:")
  print(f"   - Number of objects: {len(scene)}")
  print(f"   - Object names: {scene.list_objects()}")
  print(f"   - Scene bounds: {scene.bounds()}")

  return scene, chair, table, lamp


def scene_transformations():
  """Demonstrate scene-level transformations."""
  print("\n🎯 Scene transformations...")

  # Create scene with objects
  scene = vm.Scene(name="TransformDemo")

  # Add objects with different transformations
  cube = vm.generate("cube")
  sphere = vm.generate("sphere")
  cylinder = vm.generate("cylinder")

  # Add with position only
  scene.add(cube, name="Cube", position=[0, 0, 0])

  # Add with position and rotation
  scene.add(sphere, name="Sphere",
            position=[3, 0, 0],
            rotation=[0, 0, np.pi / 4])

  # Add with position, rotation, and scale
  scene.add(cylinder, name="Cylinder",
            position=[0, 3, 0],
            rotation=[np.pi / 2, 0, 0],
            scale=[1.5, 1.5, 1.5])

  print(f"✅ Added objects with transformations:")
  print(f"   - Cube: position only")
  print(f"   - Sphere: position + rotation")
  print(f"   - Cylinder: position + rotation + scale")

  # Get transformed objects
  transformed_sphere = scene.get_transformed_object("Sphere")
  transformed_cylinder = scene.get_transformed_object("Cylinder")

  print(f"✅ Transformed object centers:")
  print(f"   - Original sphere center: {sphere.center()}")
  print(f"   - Transformed sphere center: {transformed_sphere.center()}")
  print(f"   - Original cylinder center: {cylinder.center()}")
  print(f"   - Transformed cylinder center: {transformed_cylinder.center()}")

  return scene


def scene_visibility():
  """Demonstrate visibility control."""
  print("\n🎯 Scene visibility control...")

  scene = vm.Scene(name="VisibilityDemo")

  # Add several objects
  objects = []
  for i, shape in enumerate(["cube", "sphere", "cylinder", "chair"]):
    obj = vm.generate(shape)
    name = scene.add(obj, name=f"{shape.capitalize()}{i}",
                     position=[i * 2, 0, 0])
    objects.append(name)

  print(f"✅ Added {len(objects)} objects to scene")

  # Check initial visibility
  print(f"✅ Initial visibility:")
  for obj_name in objects:
    visible = scene.nodes[obj_name].visible
    print(f"   - {obj_name}: {visible}")

  # Hide some objects
  scene.set_visibility("Sphere1", False)
  scene.set_visibility("Chair3", False)

  print(f"✅ After hiding Sphere1 and Chair3:")
  for obj_name in objects:
    visible = scene.nodes[obj_name].visible
    print(f"   - {obj_name}: {visible}")

  # Get only visible objects
  visible_objects = [name for name in objects
                     if scene.nodes[name].visible]
  print(f"✅ Visible objects: {visible_objects}")

  return scene, objects


def scene_operations():
  """Demonstrate scene operations like merging."""
  print("\n🎯 Scene operations...")

  scene = vm.Scene(name="OperationsDemo")

  # Create a small furniture set
  chair1 = vm.generate("chair")
  chair2 = vm.generate("chair")
  table = vm.generate("table")

  scene.add(chair1, name="Chair1", position=[-1, 0, 0])
  scene.add(chair2, name="Chair2", position=[1, 0, 0])
  scene.add(table, name="Table", position=[0, 1, 0])

  print(f"✅ Created scene with {len(scene)} objects")

  # Merge all objects into one
  merged = scene.merge_objects()
  print(f"✅ Merged scene into single object: {merged.name}")
  print(f"   - Merged object bounds: {merged.bounds()}")

  # Merge only specific objects
  chair_names = ["Chair1", "Chair2"]
  merged_chairs = scene.merge_objects(object_names=chair_names)
  print(f"✅ Merged chairs: {merged_chairs.name}")

  # Clear scene and add merged object
  scene.clear()
  scene.add(merged, name="FurnitureSet", position=[0, 0, 0])
  print(f"✅ Cleared scene and added merged object")
  print(f"   - Scene now has {len(scene)} object(s)")

  return scene, merged, merged_chairs


def complex_scene_layout():
  """Create a complex scene layout."""
  print("\n🎯 Complex scene layout...")

  scene = vm.Scene(name="LivingRoom")

  # Define room layout
  furniture_layout = [
      # Living area
      {"object": "sofa", "name": "Sofa", "position": [
        0, 0, 0], "rotation": [0, 0, 0]},
      {"object": "coffee table", "name": "CoffeeTable",
       "position": [0, 2, 0], "rotation": [0, 0, 0]},
      {"object": "lamp", "name": "FloorLamp",
       "position": [-2, 0, 0], "rotation": [0, 0, 0]},

      # Dining area
      {"object": "dining table", "name": "DiningTable",
       "position": [5, 0, 0], "rotation": [0, 0, 0]},
      {"object": "chair", "name": "Chair1",
       "position": [4, -1, 0], "rotation": [0, 0, 0]},
      {"object": "chair", "name": "Chair2",
       "position": [6, -1, 0], "rotation": [0, 0, 0]},
      {"object": "chair", "name": "Chair3", "position": [
        4, 1, 0], "rotation": [0, 0, np.pi]},
      {"object": "chair", "name": "Chair4", "position": [
        6, 1, 0], "rotation": [0, 0, np.pi]},

      # Decorative items
      {"object": "vase", "name": "Vase", "position": [
        0, 2, 0.5], "scale": [0.5, 0.5, 0.5]},
      {"object": "lamp", "name": "TableLamp", "position": [
        5, 0, 0.8], "scale": [0.7, 0.7, 0.7]},
  ]

  # Generate and place all furniture
  for item in furniture_layout:
    obj = vm.generate(item["object"])

    kwargs = {
        "name": item["name"],
        "position": item["position"]
    }

    if "rotation" in item:
      kwargs["rotation"] = item["rotation"]
    if "scale" in item:
      kwargs["scale"] = item["scale"]

    scene.add(obj, **kwargs)
    print(f"✅ Added {item['name']}")

  print(f"✅ Created complex living room scene:")
  print(f"   - Total objects: {len(scene)}")
  print(f"   - Scene bounds: {scene.bounds()}")
  print(f"   - Object list: {scene.list_objects()}")

  return scene


def scene_analysis():
  """Analyze scene properties."""
  print("\n🎯 Scene analysis...")

  # Create a test scene
  scene = vm.Scene(name="AnalysisDemo")

  # Add objects at different positions and scales
  positions = [
      [0, 0, 0], [3, 0, 0], [0, 3, 0], [3, 3, 0],
      [1.5, 1.5, 2]  # One elevated object
  ]

  scales = [1.0, 1.5, 0.8, 1.2, 0.6]
  shapes = ["cube", "sphere", "cylinder", "chair", "lamp"]

  for i, (shape, pos, scale) in enumerate(zip(shapes, positions, scales)):
    obj = vm.generate(shape)
    scene.add(obj, name=f"{shape.capitalize()}{i}",
              position=pos, scale=[scale, scale, scale])

  print(f"✅ Created analysis scene with {len(scene)} objects")

  # Analyze scene bounds
  min_coords, max_coords = scene.bounds()
  scene_size = np.array(max_coords) - np.array(min_coords)
  scene_center = (np.array(max_coords) + np.array(min_coords)) / 2

  print(f"✅ Scene analysis:")
  print(f"   - Bounds: {min_coords} to {max_coords}")
  print(f"   - Size: {scene_size}")
  print(f"   - Center: {scene_center}")
  print(f"   - Volume: {np.prod(scene_size):.2f}")

  # Analyze individual objects
  print(f"✅ Object analysis:")
  for obj_name in scene.list_objects():
    node = scene.nodes[obj_name]
    obj = scene.get_transformed_object(obj_name)
    print(f"   - {obj_name}:")
    print(f"     * Position: {node.position}")
    print(f"     * Scale: {node.scale}")
    print(f"     * Visible: {node.visible}")
    print(f"     * Bounds: {obj.bounds()}")

  return scene


def main():
  """Run all scene management examples."""
  print("🚀 Volumatrix Scene Management Examples")
  print("=" * 50)

  try:
    # Basic scene creation
    furniture_scene, chair, table, lamp = basic_scene_creation()

    # Scene transformations
    transform_scene = scene_transformations()

    # Visibility control
    visibility_scene, objects = scene_visibility()

    # Scene operations
    ops_scene, merged, merged_chairs = scene_operations()

    # Complex layout
    living_room = complex_scene_layout()

    # Scene analysis
    analysis_scene = scene_analysis()

    print("\n🎉 All scene management examples completed successfully!")
    print("💡 Tips:")
    print("   - Use scenes to organize multiple objects")
    print("   - Transformations are applied when objects are added")
    print("   - Visibility can be controlled per object")
    print("   - Scenes can be merged into single objects")

  except Exception as e:
    print(f"❌ Error running examples: {e}")
    raise


if __name__ == "__main__":
  main()
