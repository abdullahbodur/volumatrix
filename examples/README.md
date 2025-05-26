# Volumatrix Examples

This directory contains practical examples demonstrating how to use the Volumatrix library for 3D object generation, manipulation, and export. Each example is self-contained and can be run independently.

## 📁 Example Files

### 🎯 [basic_generation.py](basic_generation.py)
**Learn the fundamentals of 3D object generation**

- Generate simple geometric shapes (cube, sphere, cylinder)
- Use different output formats (mesh, voxel, pointcloud)
- Generate reproducible objects with seeds
- Create complex objects from text prompts

```bash
python examples/basic_generation.py
```

### 🔄 [transformations.py](transformations.py)
**Master object transformations and manipulations**

- Basic transformations (normalize, scale, rotate, translate)
- Chain multiple transformations
- Fit objects into specific spaces
- Preserve object properties during transformations

```bash
python examples/transformations.py
```

### 🔀 [conversions.py](conversions.py)
**Convert between different 3D representations**

- Convert meshes to point clouds and voxels
- Convert point clouds to meshes
- Convert voxels to meshes
- Complete conversion pipelines
- Different conversion parameters and methods

```bash
python examples/conversions.py
```

### 🎬 [scene_management.py](scene_management.py)
**Work with scenes and multiple objects**

- Create and manage scenes
- Position and transform objects in scenes
- Control object visibility
- Merge scene objects
- Analyze scene properties

```bash
python examples/scene_management.py
```

### 💾 [export_formats.py](export_formats.py)
**Export objects to various file formats**

- Export to OBJ, STL, PLY formats
- Export transformed objects
- Export scene objects
- Batch export operations
- Handle export errors gracefully

```bash
python examples/export_formats.py
```

### ⚡ [batch_processing.py](batch_processing.py)
**Efficiently process multiple objects**

- Batch generation with and without seeds
- Batch transformations and conversions
- Batch export operations
- Performance comparisons
- Create multiple scenes efficiently

```bash
python examples/batch_processing.py
```

### 🖼️ [interactive_visualization.py](interactive_visualization.py)
**Real-time interactive 3D visualization**

- Interactive windowed viewing of 3D objects
- Multiple visualization backends (PyVista, Plotly, Trimesh, Matplotlib)
- Scene visualization with multiple objects
- Custom visualization options and controls
- Automatic backend selection

```bash
python examples/interactive_visualization.py
```

### 🎯 [run_all_examples.py](run_all_examples.py)
**Run all examples in sequence**

- Execute all examples in the recommended learning order
- Get a comprehensive overview of Volumatrix capabilities
- Perfect for testing and demonstration

```bash
python examples/run_all_examples.py
```

## 🚀 Quick Start

1. **Install Volumatrix** (if not already installed):
   ```bash
   pip install -e .
   ```

2. **Run any example**:
   ```bash
   python examples/basic_generation.py
   ```

3. **Run all examples** (recommended for first-time users):
   ```bash
   python examples/run_all_examples.py
   ```

4. **Explore the code** to understand how each feature works

## 📚 Learning Path

We recommend following this order when learning Volumatrix:

1. **Start with basics**: `basic_generation.py`
2. **Learn transformations**: `transformations.py`
3. **Understand conversions**: `conversions.py`
4. **Work with scenes**: `scene_management.py`
5. **Export your work**: `export_formats.py`
6. **Scale up**: `batch_processing.py`

## 🎯 Common Use Cases

### Generate a simple object
```python
import volumatrix as vm
cube = vm.generate("cube")
```

### Transform an object
```python
scaled_cube = vm.rescale(cube, 2.0)
rotated_cube = vm.rotate(cube, [0, 0, 3.14159/4])
```

### Convert between representations
```python
point_cloud = vm.mesh_to_pointcloud(cube, num_points=1000)
voxels = vm.voxelize(cube, resolution=32)
```

### Create a scene
```python
scene = vm.Scene()
scene.add(cube, name="MyCube", position=[0, 0, 0])
```

### Export to file
```python
vm.export(cube, "my_cube.obj")
```

### Batch processing
```python
objects = vm.generate_batch(["cube", "sphere", "cylinder"])
```

## 🔧 Customization

Each example can be easily modified to suit your needs:

- **Change prompts**: Modify the text descriptions to generate different objects
- **Adjust parameters**: Change resolutions, scales, positions, etc.
- **Add new transformations**: Combine existing operations in new ways
- **Export different formats**: Try different file formats for your use case

## 🐛 Troubleshooting

If you encounter issues:

1. **Check dependencies**: Ensure all required packages are installed
2. **Verify installation**: Make sure Volumatrix is properly installed
3. **Check file paths**: Ensure you have write permissions for export operations
4. **Review error messages**: Most errors include helpful information

## 💡 Tips

- **Start simple**: Begin with basic examples before moving to complex workflows
- **Experiment**: Modify parameters to see how they affect the results
- **Combine features**: Mix and match different operations for powerful workflows
- **Use batch operations**: They're more efficient for processing multiple objects
- **Save your work**: Export objects you want to keep for later use

## 📖 Additional Resources

- **Main Documentation**: See the project README for comprehensive documentation
- **API Reference**: Check the source code for detailed function documentation
- **Tests**: Look at the `tests/` directory for more usage examples

## 🤝 Contributing

Found a bug or want to add a new example? Contributions are welcome! Please:

1. Fork the repository
2. Create a new branch for your changes
3. Add your example with clear documentation
4. Submit a pull request

---

Happy 3D modeling with Volumatrix! 🎨✨ 