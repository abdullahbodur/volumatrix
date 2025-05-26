#!/usr/bin/env python3
"""
Demo: Automatic 3D Visualization in Volumatrix

This script demonstrates how Volumatrix can automatically open 
interactive 3D windows when generating objects.
"""

import volumatrix as vm

def main():
    print("🎯 Volumatrix Auto-Visualization Demo")
    print("=" * 50)
    print("This demo shows how 3D objects can automatically open in viewers")
    print()
    
    # Method 1: Generate with auto-preview
    print("📱 Method 1: Auto-preview with generate()")
    print("Generating cube with automatic visualization...")
    cube = vm.generate("cube", auto_preview=True)
    print(f"✅ Generated: {cube.name}")
    print("   → A 3D viewer window should have opened!")
    print()
    
    # Method 2: Convenience function
    print("📱 Method 2: Using generate_and_show()")
    print("Generating sphere with automatic visualization...")
    sphere = vm.generate_and_show("sphere")  
    print(f"✅ Generated: {sphere.name}")
    print("   → Another 3D viewer window should have opened!")
    print()
    
    # Method 3: Manual visualization
    print("📱 Method 3: Manual visualization with show()")
    print("Generating cylinder and then showing it...")
    cylinder = vm.generate("cylinder")
    vm.show(cylinder)  # Manually show
    print(f"✅ Generated and showed: {cylinder.name}")
    print()
    
    # Method 4: Scene visualization
    print("📱 Method 4: Scene visualization")
    print("Creating a scene with multiple objects...")
    scene = vm.Scene(name="AutoVisualizationDemo")
    scene.add(cube, name="Cube", position=[0, 0, 0])
    scene.add(sphere, name="Sphere", position=[2, 0, 0])
    scene.add(cylinder, name="Cylinder", position=[0, 2, 0])
    
    print(f"Scene created with {len(scene)} objects")
    vm.show(scene)  # Show the entire scene
    print("✅ Scene visualization opened!")
    print()
    
    print("🎉 Demo Complete!")
    print()
    print("💡 Key Features Demonstrated:")
    print("   • vm.generate(..., auto_preview=True)")
    print("   • vm.generate_and_show(...)")
    print("   • vm.show(object_or_scene)")
    print("   • Automatic backend selection")
    print("   • Scene visualization")
    print()
    print("🎨 Available Backends:")
    print("   • pyvista (best interactive experience)")
    print("   • plotly (web-based, opens in browser)")
    print("   • trimesh (simple windowed viewer)")
    print("   • matplotlib (basic 3D plotting)")
    print()
    print("📚 Next Steps:")
    print("   1. Try: vm.generate_and_show('your_object_here')")
    print("   2. Experiment with different backends")
    print("   3. Create complex scenes with multiple objects")
    print("   4. Use transformations and visualize results")

if __name__ == "__main__":
    main() 