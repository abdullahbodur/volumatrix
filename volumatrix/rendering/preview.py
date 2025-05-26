"""
Preview and visualization utilities for Volumatrix.

This module provides functions for previewing 3D objects in various environments
with interactive windowed visualization capabilities.
"""

from typing import Optional, Union, List, Dict, Any
import logging
import numpy as np

from ..core.object import VolumatrixObject
from ..core.scene import Scene

logger = logging.getLogger(__name__)


def preview(obj: Union[VolumatrixObject, Scene],
            backend: str = "auto",
            window_title: Optional[str] = None,
            window_size: tuple = (800, 600),
            auto_show: bool = True,
            **kwargs) -> None:
  """
  Preview a 3D object or scene in an interactive window.

  Args:
      obj: VolumatrixObject or Scene to preview
      backend: Rendering backend ("auto", "pyvista", "trimesh", "plotly", "matplotlib")
      window_title: Title for the visualization window
      window_size: Size of the window (width, height)
      auto_show: Whether to automatically show the window
      **kwargs: Additional rendering options

  Examples:
      >>> preview(obj)  # Opens interactive window
      >>> preview(scene, backend="pyvista", window_title="My Scene")
  """
  if backend == "auto":
    # Auto-select best available backend
    backend = _select_best_backend()

  if window_title is None:
    if isinstance(obj, Scene):
      window_title = f"Volumatrix Scene: {obj.name}"
    else:
      window_title = f"Volumatrix Object: {obj.name}"

  if backend == "pyvista":
    _preview_pyvista(obj, window_title=window_title,
                     window_size=window_size, auto_show=auto_show, **kwargs)
  elif backend == "trimesh":
    _preview_trimesh(obj, window_title=window_title,
                     auto_show=auto_show, **kwargs)
  elif backend == "plotly":
    _preview_plotly(obj, window_title=window_title,
                    window_size=window_size, **kwargs)
  elif backend == "matplotlib":
    _preview_matplotlib(obj, window_title=window_title,
                        window_size=window_size, **kwargs)
  else:
    raise ValueError(f"Unknown backend: {backend}")


def show(obj: Union[VolumatrixObject, Scene], **kwargs) -> None:
  """
  Convenience function to immediately show an object in an interactive window.

  Args:
      obj: VolumatrixObject or Scene to show
      **kwargs: Arguments passed to preview()

  Examples:
      >>> cube = vm.generate("cube")
      >>> vm.show(cube)  # Immediately opens window
  """
  preview(obj, **kwargs)


def preview_jupyter(obj: Union[VolumatrixObject, Scene],
                    backend: str = "plotly",
                    **kwargs) -> None:
  """
  Preview a 3D object or scene in Jupyter notebook.

  Args:
      obj: VolumatrixObject or Scene to preview
      backend: Rendering backend ("plotly", "pythreejs", "k3d")
      **kwargs: Additional rendering options

  Examples:
      >>> preview_jupyter(obj)
      >>> preview_jupyter(scene, backend="pythreejs")
  """
  if backend == "plotly":
    _preview_plotly_jupyter(obj, **kwargs)
  elif backend == "pythreejs":
    _preview_pythreejs(obj, **kwargs)
  elif backend == "k3d":
    _preview_k3d(obj, **kwargs)
  else:
    raise ValueError(f"Unknown Jupyter backend: {backend}")


def _select_best_backend() -> str:
  """Select the best available backend for interactive visualization."""
  try:
    import pyvista
    return "pyvista"
  except ImportError:
    pass

  try:
    import trimesh
    return "trimesh"
  except ImportError:
    pass

  try:
    import plotly
    return "plotly"
  except ImportError:
    pass

  try:
    import matplotlib
    return "matplotlib"
  except ImportError:
    pass

  raise RuntimeError(
    "No visualization backend available. Install pyvista, trimesh, plotly, or matplotlib.")


def _preview_pyvista(obj: Union[VolumatrixObject, Scene],
                     window_title: str = "Volumatrix Viewer",
                     window_size: tuple = (800, 600),
                     auto_show: bool = True,
                     background_color: str = "white",
                     show_axes: bool = True,
                     show_grid: bool = False,
                     **kwargs) -> None:
  """Preview using PyVista (best interactive experience)."""
  try:
    import pyvista as pv

    # Configure PyVista
    pv.set_plot_theme("document")

    # Create plotter
    plotter = pv.Plotter(
        title=window_title,
        window_size=window_size,
        off_screen=not auto_show
    )

    plotter.set_background(background_color)

    if isinstance(obj, Scene):
      # Render all visible objects in the scene
      for name, node in obj.nodes.items():
        if node.visible:
          transformed_obj = node.get_transformed_object()
          _add_object_pyvista(plotter, transformed_obj, name=name)
    else:
      _add_object_pyvista(plotter, obj)

    if show_axes:
      plotter.show_axes()

    if show_grid:
      plotter.show_grid()

    # Add camera controls info
    plotter.add_text(
        "Mouse: Rotate | Shift+Mouse: Pan | Scroll: Zoom | R: Reset View",
        position="lower_left",
        font_size=10,
        color="gray"
    )

    if auto_show:
      plotter.show(interactive=True)

    return plotter

  except ImportError:
    logger.error(
      "PyVista is required for pyvista backend. Install with: pip install pyvista")
    raise


def _preview_trimesh(obj: Union[VolumatrixObject, Scene],
                     window_title: str = "Volumatrix Viewer",
                     auto_show: bool = True,
                     **kwargs) -> None:
  """Preview using trimesh."""
  try:
    import trimesh

    meshes = []
    mesh_names = []

    if isinstance(obj, Scene):
      # Collect all visible meshes
      for name, node in obj.nodes.items():
        if node.visible:
          transformed_obj = node.get_transformed_object()
          mesh_repr = transformed_obj.mesh
          if mesh_repr is not None:
            trimesh_mesh = trimesh.Trimesh(
                vertices=mesh_repr.vertices,
                faces=mesh_repr.faces,
                vertex_normals=mesh_repr.normals
            )
            meshes.append(trimesh_mesh)
            mesh_names.append(name)
    else:
      mesh_repr = obj.mesh
      if mesh_repr is not None:
        trimesh_mesh = trimesh.Trimesh(
            vertices=mesh_repr.vertices,
            faces=mesh_repr.faces,
            vertex_normals=mesh_repr.normals
        )
        meshes.append(trimesh_mesh)
        mesh_names.append(obj.name)

    if meshes:
      if auto_show:
        if len(meshes) == 1:
          meshes[0].show(caption=window_title)
        else:
          scene = trimesh.Scene(meshes)
          scene.show(caption=window_title)
      else:
        logger.info(
          "Trimesh visualization prepared but not shown (auto_show=False)")
    else:
      logger.warning("No mesh representations found for preview")

  except ImportError:
    logger.error(
      "trimesh is required for trimesh backend. Install with: pip install trimesh")
    raise


def _preview_plotly(obj: Union[VolumatrixObject, Scene],
                    window_title: str = "Volumatrix Viewer",
                    window_size: tuple = (800, 600),
                    **kwargs) -> None:
  """Preview using plotly."""
  try:
    import plotly.graph_objects as go

    fig = go.Figure()

    if isinstance(obj, Scene):
      # Render all visible objects in the scene
      for name, node in obj.nodes.items():
        if node.visible:
          transformed_obj = node.get_transformed_object()
          _add_object_plotly(fig, transformed_obj, name=name)
    else:
      _add_object_plotly(fig, obj)

    fig.update_layout(
        title=window_title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=window_size[0],
        height=window_size[1],
        showlegend=True
    )

    fig.show()

  except ImportError:
    logger.error(
      "plotly is required for plotly backend. Install with: pip install plotly")
    raise


def _preview_matplotlib(obj: Union[VolumatrixObject, Scene],
                        window_title: str = "Volumatrix Viewer",
                        window_size: tuple = (800, 600),
                        **kwargs) -> None:
  """Preview using matplotlib."""
  try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(window_size[0] / 100, window_size[1] / 100))
    fig.suptitle(window_title)
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(obj, Scene):
      # Render all visible objects in the scene
      for name, node in obj.nodes.items():
        if node.visible:
          transformed_obj = node.get_transformed_object()
          _plot_object_matplotlib(ax, transformed_obj, label=name)
    else:
      _plot_object_matplotlib(ax, obj)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if kwargs.get('show_legend', True):
      ax.legend()

    plt.show()

  except ImportError:
    logger.error(
      "matplotlib is required for matplotlib backend. Install with: pip install matplotlib")
    raise


def _add_object_pyvista(plotter, obj: VolumatrixObject, name: Optional[str] = None) -> None:
  """Add a VolumatrixObject to a PyVista plotter."""
  import pyvista as pv

  display_name = name or obj.name

  # Try different representations in order of preference
  if obj.mesh is not None:
    mesh = obj.mesh

    # Create PyVista mesh
    pv_mesh = pv.PolyData(mesh.vertices,
                          faces=np.column_stack([np.full(len(mesh.faces), 3), mesh.faces]))

    # Add to plotter with nice styling
    plotter.add_mesh(
        pv_mesh,
        name=display_name,
        show_edges=True,
        edge_color="black",
        line_width=0.5,
        opacity=0.8,
        color="lightblue" if name is None else None,
        lighting=True,
        smooth_shading=True
    )

  elif obj.pointcloud is not None:
    pc = obj.pointcloud

    # Create PyVista point cloud
    pv_points = pv.PolyData(pc.points)

    # Add colors if available
    if pc.colors is not None:
      pv_points["colors"] = (pc.colors[:, :3] * 255).astype(np.uint8)
      plotter.add_mesh(pv_points, name=display_name, scalars="colors",
                       point_size=5, render_points_as_spheres=True)
    else:
      plotter.add_mesh(pv_points, name=display_name, color="red",
                       point_size=5, render_points_as_spheres=True)

  elif obj.voxel is not None:
    voxel = obj.voxel

    # Get occupied voxel coordinates
    coords = voxel.get_occupied_coordinates()
    if len(coords) > 0:
      # Create small cubes for each voxel
      pv_points = pv.PolyData(coords)

      # Create small cubes
      cubes = pv_points.glyph(geom=pv.Cube(), scale=False,
                              factor=voxel.spacing * 0.8)
      plotter.add_mesh(cubes, name=display_name, color="green", opacity=0.7)

  else:
    logger.warning(f"No supported representation found for object: {obj.name}")


def _plot_object_matplotlib(ax, obj: VolumatrixObject, label: Optional[str] = None) -> None:
  """Plot a VolumatrixObject using matplotlib."""
  # Try different representations in order of preference
  if obj.mesh is not None:
    _plot_mesh_matplotlib(ax, obj.mesh, label)
  elif obj.pointcloud is not None:
    _plot_pointcloud_matplotlib(ax, obj.pointcloud, label)
  elif obj.voxel is not None:
    _plot_voxel_matplotlib(ax, obj.voxel, label)
  else:
    logger.warning(f"No supported representation found for object: {obj.name}")


def _plot_mesh_matplotlib(ax, mesh, label: Optional[str] = None) -> None:
  """Plot a mesh using matplotlib."""
  # Plot vertices as scatter points
  ax.scatter(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
             s=1, alpha=0.6, label=label)

  # Optionally plot edges (simplified)
  if len(mesh.faces) < 1000:  # Only for small meshes to avoid clutter
    for face in mesh.faces[:100]:  # Limit number of faces
      triangle = mesh.vertices[face]
      # Plot triangle edges
      for i in range(3):
        start, end = triangle[i], triangle[(i + 1) % 3]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                'b-', alpha=0.3, linewidth=0.5)


def _plot_pointcloud_matplotlib(ax, pointcloud, label: Optional[str] = None) -> None:
  """Plot a point cloud using matplotlib."""
  colors = pointcloud.colors[:, :3] if pointcloud.colors is not None else 'blue'
  ax.scatter(pointcloud.points[:, 0], pointcloud.points[:, 1], pointcloud.points[:, 2],
             c=colors, s=1, alpha=0.6, label=label)


def _plot_voxel_matplotlib(ax, voxel, label: Optional[str] = None) -> None:
  """Plot a voxel grid using matplotlib."""
  # Get occupied voxel coordinates
  coords = voxel.get_occupied_coordinates()
  if len(coords) > 0:
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=10, alpha=0.6, label=label, marker='s')


def _add_object_plotly(fig, obj: VolumatrixObject, name: Optional[str] = None) -> None:
  """Add a VolumatrixObject to a plotly figure."""
  if obj.mesh is not None:
    _add_mesh_plotly(fig, obj.mesh, name)
  elif obj.pointcloud is not None:
    _add_pointcloud_plotly(fig, obj.pointcloud, name)
  elif obj.voxel is not None:
    _add_voxel_plotly(fig, obj.voxel, name)
  else:
    logger.warning(f"No supported representation found for object: {obj.name}")


def _add_mesh_plotly(fig, mesh, name: Optional[str] = None) -> None:
  """Add a mesh to a plotly figure."""
  import plotly.graph_objects as go

  fig.add_trace(go.Mesh3d(
      x=mesh.vertices[:, 0],
      y=mesh.vertices[:, 1],
      z=mesh.vertices[:, 2],
      i=mesh.faces[:, 0],
      j=mesh.faces[:, 1],
      k=mesh.faces[:, 2],
      name=name or "Mesh",
      opacity=0.8,
      showscale=False
  ))


def _add_pointcloud_plotly(fig, pointcloud, name: Optional[str] = None) -> None:
  """Add a point cloud to a plotly figure."""
  import plotly.graph_objects as go

  marker_dict = dict(size=2, opacity=0.6)
  if pointcloud.colors is not None:
    # Convert colors to RGB strings
    colors = pointcloud.colors[:, :3]  # Take only RGB, ignore alpha
    marker_dict['color'] = [f'rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})'
                            for c in colors]

  fig.add_trace(go.Scatter3d(
      x=pointcloud.points[:, 0],
      y=pointcloud.points[:, 1],
      z=pointcloud.points[:, 2],
      mode='markers',
      marker=marker_dict,
      name=name or "Point Cloud"
  ))


def _add_voxel_plotly(fig, voxel, name: Optional[str] = None) -> None:
  """Add a voxel grid to a plotly figure."""
  import plotly.graph_objects as go

  # Get occupied voxel coordinates
  coords = voxel.get_occupied_coordinates()
  if len(coords) > 0:
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(size=4, opacity=0.6, symbol='square'),
        name=name or "Voxels"
    ))


def _preview_plotly_jupyter(obj: Union[VolumatrixObject, Scene], **kwargs) -> None:
  """Preview using plotly in Jupyter."""
  # Same as regular plotly preview
  _preview_plotly(obj, **kwargs)


def _preview_pythreejs(obj: Union[VolumatrixObject, Scene], **kwargs) -> None:
  """Preview using pythreejs in Jupyter."""
  try:
    import pythreejs as p3js
    from IPython.display import display

    # This is a placeholder - full pythreejs implementation would be more complex
    logger.warning("pythreejs preview is not fully implemented yet")

    # Fallback to plotly
    _preview_plotly(obj, **kwargs)

  except ImportError:
    logger.error(
      "pythreejs is required for pythreejs backend. Install with: pip install pythreejs")
    raise


def _preview_k3d(obj: Union[VolumatrixObject, Scene], **kwargs) -> None:
  """Preview using k3d in Jupyter."""
  try:
    import k3d

    # This is a placeholder - full k3d implementation would be more complex
    logger.warning("k3d preview is not fully implemented yet")

    # Fallback to plotly
    _preview_plotly(obj, **kwargs)

  except ImportError:
    logger.error(
      "k3d is required for k3d backend. Install with: pip install k3d")
    raise
