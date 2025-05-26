# Volumatrix 🎯

**AI-Native 3D Object Generation Library**

Volumatrix is a Python library that makes 3D model generation as accessible and fast as image generation, while maintaining Blender-grade versatility. Generate, manipulate, and export 3D objects using AI models with a simple, intuitive API.

## 🚀 Quick Start

```python
import volumatrix as vm

# Generate a 3D object from text
obj = vm.generate("a red futuristic drone")

# Export to various formats
vm.export(obj, "drone.obj")
vm.export(obj, "drone.glb")

# Generate and immediately visualize in an interactive window
cube = vm.generate_and_show("cube")  # Opens 3D viewer automatically

# Or use the auto_preview parameter
sphere = vm.generate("sphere", auto_preview=True)
```

## 🎯 Features

### Core Capabilities
- **AI-Driven Generation**: Plug-and-play interface for diffusion models, transformers, and generative networks
- **Multiple Representations**: Support for meshes, voxels, and point clouds
- **Universal Export**: Export to .obj, .glb, .ply, .stl formats
- **Scene Management**: Compose multiple objects with transformations

### Model Integration
- Clean API for external generative models
- Text-to-3D generation with simple prompts
- Support for various AI backends (Stability AI, OpenAI 3D, DreamFusion, SDF networks)

### Utilities & Processing
- Mesh normalization, rescaling, rotation, and combination
- Voxelization and de-voxelization tools
- Format conversion between point clouds, meshes, and voxels

### Visualization & Rendering
- Real-time preview in Jupyter notebooks
- WebGL-based browser rendering
- Optional Blender integration for professional rendering

### Performance & Deployment
- GPU acceleration with CUDA/Metal backends
- Batch generation and multiprocessing support
- REST API for server-side deployment

## 📦 Installation

```bash
pip install volumatrix
```

### Development Installation

```bash
git clone https://github.com/your-username/volumatrix.git
cd volumatrix
pip install -e ".[dev]"
```

## 📚 Examples

Check out the [`examples/`](examples/) directory for comprehensive usage examples:

- **[Basic Generation](examples/basic_generation.py)**: Learn the fundamentals
- **[Transformations](examples/transformations.py)**: Object manipulation and transformations
- **[Conversions](examples/conversions.py)**: Convert between 3D representations
- **[Scene Management](examples/scene_management.py)**: Work with multiple objects
- **[Export Formats](examples/export_formats.py)**: Save to different file formats
- **[Batch Processing](examples/batch_processing.py)**: Efficient bulk operations

```bash
# Run any example
python examples/basic_generation.py
```

## 🛠️ Usage

### Basic Generation

```python
import volumatrix as vm

# Generate from text prompt
chair = vm.generate("wooden chair")

# Generate with immediate visualization
drone = vm.generate_and_show("futuristic drone")  # Opens in 3D viewer

# Generate with specific backend
table = vm.generate("table", auto_preview=True, preview_backend="pyvista")

# Batch generation with auto-visualization
objects = vm.generate_batch([
    "red sports car",
    "blue bicycle", 
    "green tree"
], auto_preview=True)
```

### Scene Composition

```python
import volumatrix as vm

# Create a scene
scene = vm.Scene()

# Add objects with transformations
chair = vm.generate("wooden chair")
table = vm.generate("wooden table")

scene.add(chair, position=(0, 0, 0))
scene.add(table, position=(2, 0, 0), rotation=(0, 45, 0))

# Visualize the scene interactively
vm.show(scene)  # Opens 3D viewer with entire scene

# Export entire scene
vm.export(scene, "living_room.glb")
```

### Advanced Manipulation

```python
import volumatrix as vm

# Load and process existing object
obj = vm.load("model.obj")

# Normalize and rescale
obj = vm.normalize(obj, scale=2.0)

# Convert between representations and visualize
voxels = vm.voxelize(obj, resolution=64)
vm.show(voxels)  # See the voxelized version

points = vm.mesh_to_pointcloud(obj, num_points=10000)
vm.show(points)  # See the point cloud version
```

### Interactive Visualization

```python
import volumatrix as vm

# Generate and immediately show in 3D viewer
cube = vm.generate_and_show("cube")

# Generate with auto-preview
sphere = vm.generate("sphere", auto_preview=True)

# Show existing objects
chair = vm.generate("chair")
vm.show(chair)  # Opens interactive window

# Choose visualization backend
table = vm.generate("table")
vm.preview(table, backend="pyvista")  # Best interactive experience
vm.preview(table, backend="plotly")   # Web-based viewer
vm.preview(table, backend="trimesh")  # Simple windowed viewer

# Batch generation with automatic visualization
objects = vm.generate_batch(
    ["cube", "sphere", "cylinder"], 
    auto_preview=True
)

# Scene visualization
scene = vm.Scene()
scene.add(vm.generate("chair"), position=[0, 0, 0])
scene.add(vm.generate("table"), position=[2, 0, 0])
vm.show(scene)  # Show entire scene in 3D viewer
```

### CLI Usage

```bash
# Generate from command line
volumatrix generate --prompt "wooden chair" --output chair.obj

# Batch generation
volumatrix generate --prompts prompts.txt --output-dir ./models/

# Convert formats
volumatrix convert input.obj output.glb
```

## 🏗️ Architecture

```
volumatrix/
├── core/           # Core 3D representations and scene management
├── models/         # AI model interfaces and backends
├── utils/          # Preprocessing and conversion utilities
├── rendering/      # Visualization and preview tools
├── export/         # Format export handlers
├── cli/           # Command-line interface
└── plugins/       # Extensible plugin system
```

## 🔧 Requirements

- Python 3.10+
- NumPy, PyTorch/TensorFlow (for AI models)
- Trimesh (3D processing)
- Optional: CUDA for GPU acceleration

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built for researchers and ML developers who want to make 3D generation accessible to everyone. 