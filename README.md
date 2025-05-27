# Volumatrix

**AI-Native 3D Object Generation Library**

Volumatrix is a Python library that makes 3D model generation as accessible and fast as image generation, while maintaining Blender-grade versatility. Generate, manipulate, and export 3D objects using AI models with a simple, intuitive API.

## Features

### Generation
- AI-driven 3D object generation
- Multiple object representations (meshes, voxels, point clouds)
- Universal export (.obj, .glb, .ply, .stl)
- Scene composition and management

### Visualization Options
1. Built-in OpenGL Renderer:
   - Interactive 3D visualization
   - Camera controls:
     - WASD: Move camera
     - Mouse: Look around
     - Scroll: Zoom in/out
     - O: Toggle orbit mode
     - R: Reset camera
     - G: Toggle grid
   - Real-time rendering
   - Grid reference system
   - Orbit mode for object inspection

2. Third-party Visualizers:
   - PyVista
   - Plotly
   - Trimesh

## Installation

```bash
pip install volumatrix
```

## Usage

### Object Generation
```python
import volumatrix as vm

# Generate a 3D object from text
obj = vm.generate("a red futuristic drone")

# Export to various formats
vm.export(obj, "drone.obj")
vm.export(obj, "drone.glb")
```

### Visualization Options

#### 1. Built-in OpenGL Renderer
```python
from volumatrix.visualization import VolumatrixRenderer

# Create renderer
renderer = VolumatrixRenderer(width=800, height=600)

# Set your data
vertices = ...  # Your 3D vertices
renderer.set_vertices(vertices)

# Start visualization
while not renderer.window.should_close():
    renderer.render()

# Cleanup
renderer.cleanup()
```

#### 2. Third-party Visualizers
```python
import volumatrix as vm

# Generate and visualize with PyVista
obj = vm.generate("cube")
vm.show(obj, backend="pyvista")

# Generate and visualize with Plotly
obj = vm.generate("sphere")
vm.show(obj, backend="plotly")

# Generate and visualize with Trimesh
obj = vm.generate("cylinder")
vm.show(obj, backend="trimesh")
```

### Scene Management
```python
import volumatrix as vm

# Create a scene
scene = vm.Scene()

# Add objects with transformations
chair = vm.generate("wooden chair")
table = vm.generate("wooden table")

scene.add(chair, position=(0, 0, 0))
scene.add(table, position=(2, 0, 0), rotation=(0, 45, 0))

# Visualize the scene with your preferred backend
vm.show(scene, backend="pyvista")
# or
vm.show(scene, backend="opengl")
```

## Requirements

- Python 3.8+
- OpenGL (for built-in renderer)
- NumPy
- GLFW (for built-in renderer)
- PyTorch/TensorFlow (for AI models)
- Trimesh (3D processing)
- Optional: PyVista, Plotly for additional visualization options

## License

MIT License

## Author

Abdullah Bodur
