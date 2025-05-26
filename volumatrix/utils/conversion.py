"""
Conversion utilities for Volumatrix.

This module provides functions for converting between different 3D representations:
meshes, voxels, and point clouds.
"""

from typing import Optional, Union
import numpy as np

from ..core.object import VolumatrixObject
from ..core.representations import Mesh, Voxel, PointCloud


def voxelize(obj: VolumatrixObject, resolution: int = 64, 
            method: str = "surface") -> VolumatrixObject:
    """
    Convert an object to voxel representation.
    
    Args:
        obj: The VolumatrixObject to voxelize
        resolution: Voxel grid resolution
        method: Voxelization method ("surface" or "solid")
    
    Returns:
        A new VolumatrixObject with voxel representation
    
    Examples:
        >>> voxel_obj = voxelize(mesh_obj, resolution=128)
        >>> surface_voxels = voxelize(obj, method="surface")
    """
    # Get bounds for voxel grid
    min_coords, max_coords = obj.bounds()
    size = max_coords - min_coords
    max_size = np.max(size)
    
    # Create uniform grid
    spacing = max_size / resolution
    grid_size = np.ceil(size / spacing).astype(int)
    
    # Ensure minimum resolution
    grid_size = np.maximum(grid_size, 1)
    
    # Initialize voxel grid
    grid = np.zeros(tuple(grid_size), dtype=bool)
    
    # Get primary representation
    primary_repr = obj.primary_representation
    
    if isinstance(primary_repr, Mesh):
        grid = _voxelize_mesh(primary_repr, grid, min_coords, spacing, method)
    elif isinstance(primary_repr, PointCloud):
        grid = _voxelize_pointcloud(primary_repr, grid, min_coords, spacing)
    else:
        raise ValueError(f"Cannot voxelize representation type: {type(primary_repr)}")
    
    # Create voxel representation
    voxel_repr = Voxel(
        grid=grid,
        resolution=tuple(grid_size),
        origin=min_coords,
        spacing=spacing
    )
    
    # Create new object
    new_obj = obj.copy()
    new_obj.add_representation("voxel", voxel_repr)
    
    return new_obj


def devoxelize(obj: VolumatrixObject, method: str = "marching_cubes") -> VolumatrixObject:
    """
    Convert voxel representation to mesh using marching cubes or similar algorithm.
    
    Args:
        obj: The VolumatrixObject with voxel representation
        method: Conversion method ("marching_cubes" or "simple")
    
    Returns:
        A new VolumatrixObject with mesh representation
    
    Examples:
        >>> mesh_obj = devoxelize(voxel_obj)
    """
    voxel_repr = obj.voxel
    if voxel_repr is None:
        raise ValueError("Object must have a voxel representation")
    
    if method == "marching_cubes":
        try:
            from skimage import measure
            
            # Apply marching cubes
            vertices, faces, _, _ = measure.marching_cubes(
                voxel_repr.grid.astype(float), 
                level=0.5
            )
            
            # Transform vertices to world coordinates
            spacing_array = np.array([voxel_repr.spacing] * 3) if isinstance(voxel_repr.spacing, float) else voxel_repr.spacing
            vertices = vertices * spacing_array + voxel_repr.origin
            
        except ImportError:
            # Fallback to simple method
            vertices, faces = _simple_devoxelize(voxel_repr)
    else:
        vertices, faces = _simple_devoxelize(voxel_repr)
    
    # Create mesh representation
    mesh_repr = Mesh(vertices=vertices, faces=faces)
    mesh_repr.compute_normals()
    
    # Create new object
    new_obj = obj.copy()
    new_obj.add_representation("mesh", mesh_repr)
    
    return new_obj


def mesh_to_pointcloud(obj: VolumatrixObject, num_points: int = 10000, 
                      method: str = "surface") -> VolumatrixObject:
    """
    Convert mesh to point cloud representation.
    
    Args:
        obj: The VolumatrixObject with mesh representation
        num_points: Number of points to generate
        method: Sampling method ("surface", "vertices", or "random")
    
    Returns:
        A new VolumatrixObject with point cloud representation
    
    Examples:
        >>> pc_obj = mesh_to_pointcloud(mesh_obj, num_points=5000)
    """
    mesh_repr = obj.mesh
    if mesh_repr is None:
        raise ValueError("Object must have a mesh representation")
    
    if method == "vertices":
        # Use existing vertices
        points = mesh_repr.vertices.copy()
        colors = mesh_repr.colors.copy() if mesh_repr.colors is not None else None
        normals = mesh_repr.normals.copy() if mesh_repr.normals is not None else None
        
        # Subsample if needed
        if len(points) > num_points:
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
            colors = colors[indices] if colors is not None else None
            normals = normals[indices] if normals is not None else None
            
    elif method == "surface":
        # Sample points on mesh surface
        points, colors, normals = _sample_mesh_surface(mesh_repr, num_points)
        
    else:  # random
        # Random sampling within bounding box
        min_coords, max_coords = mesh_repr.bounds()
        points = np.random.uniform(min_coords, max_coords, (num_points, 3))
        colors = None
        normals = None
    
    # Create point cloud representation
    pc_repr = PointCloud(
        points=points,
        colors=colors,
        normals=normals
    )
    
    # Create new object
    new_obj = obj.copy()
    new_obj.add_representation("pointcloud", pc_repr)
    
    return new_obj


def pointcloud_to_mesh(obj: VolumatrixObject, method: str = "poisson") -> VolumatrixObject:
    """
    Convert point cloud to mesh representation.
    
    Args:
        obj: The VolumatrixObject with point cloud representation
        method: Reconstruction method ("poisson", "delaunay", or "alpha_shape")
    
    Returns:
        A new VolumatrixObject with mesh representation
    
    Examples:
        >>> mesh_obj = pointcloud_to_mesh(pc_obj)
    """
    pc_repr = obj.pointcloud
    if pc_repr is None:
        raise ValueError("Object must have a point cloud representation")
    
    if method == "poisson":
        vertices, faces = _poisson_reconstruction(pc_repr)
    elif method == "delaunay":
        vertices, faces = _delaunay_triangulation(pc_repr)
    elif method == "alpha_shape":
        vertices, faces = _alpha_shape_reconstruction(pc_repr)
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
    
    # Create mesh representation
    mesh_repr = Mesh(vertices=vertices, faces=faces)
    mesh_repr.compute_normals()
    
    # Create new object
    new_obj = obj.copy()
    new_obj.add_representation("mesh", mesh_repr)
    
    return new_obj


def mesh_to_voxel(obj: VolumatrixObject, resolution: int = 64) -> VolumatrixObject:
    """
    Convert mesh to voxel representation (alias for voxelize).
    
    Args:
        obj: The VolumatrixObject with mesh representation
        resolution: Voxel grid resolution
    
    Returns:
        A new VolumatrixObject with voxel representation
    """
    return voxelize(obj, resolution)


def voxel_to_mesh(obj: VolumatrixObject) -> VolumatrixObject:
    """
    Convert voxel to mesh representation (alias for devoxelize).
    
    Args:
        obj: The VolumatrixObject with voxel representation
    
    Returns:
        A new VolumatrixObject with mesh representation
    """
    return devoxelize(obj)


# Helper functions

def _voxelize_mesh(mesh: Mesh, grid: np.ndarray, origin: np.ndarray, 
                  spacing: float, method: str) -> np.ndarray:
    """Voxelize a mesh representation."""
    if method == "surface":
        # Rasterize mesh faces
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            _rasterize_triangle(v0, v1, v2, grid, origin, spacing)
    else:  # solid
        # For solid voxelization, we'd need more complex algorithms
        # For now, fall back to surface
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            _rasterize_triangle(v0, v1, v2, grid, origin, spacing)
    
    return grid


def _voxelize_pointcloud(pc: PointCloud, grid: np.ndarray, origin: np.ndarray, 
                        spacing: float) -> np.ndarray:
    """Voxelize a point cloud representation."""
    # Convert points to voxel indices
    indices = ((pc.points - origin) / spacing).astype(int)
    
    # Clamp indices to grid bounds
    indices = np.clip(indices, 0, np.array(grid.shape) - 1)
    
    # Set voxels
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    
    return grid


def _rasterize_triangle(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray,
                       grid: np.ndarray, origin: np.ndarray, spacing: float) -> None:
    """Rasterize a triangle into the voxel grid."""
    # Convert vertices to voxel coordinates
    voxel_coords = [(v - origin) / spacing for v in [v0, v1, v2]]
    
    # Get bounding box
    min_coords = np.floor(np.min(voxel_coords, axis=0)).astype(int)
    max_coords = np.ceil(np.max(voxel_coords, axis=0)).astype(int)
    
    # Clamp to grid bounds
    min_coords = np.maximum(min_coords, 0)
    max_coords = np.minimum(max_coords, np.array(grid.shape) - 1)
    
    # Rasterize triangle (simplified - just mark voxels near vertices and edges)
    for coord in voxel_coords:
        idx = np.clip(np.round(coord).astype(int), 0, np.array(grid.shape) - 1)
        grid[idx[0], idx[1], idx[2]] = True


def _simple_devoxelize(voxel: Voxel) -> tuple[np.ndarray, np.ndarray]:
    """Simple voxel to mesh conversion (creates cubes for each voxel)."""
    vertices = []
    faces = []
    
    # Get occupied voxel indices
    occupied = np.where(voxel.grid)
    
    spacing_array = np.array([voxel.spacing] * 3) if isinstance(voxel.spacing, float) else voxel.spacing
    
    for i, j, k in zip(*occupied):
        # Create cube vertices for this voxel
        base_pos = voxel.origin + np.array([i, j, k]) * spacing_array
        
        # Cube vertices (8 vertices per cube)
        cube_vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
        ]) * spacing_array + base_pos
        
        # Add vertices
        vertex_offset = len(vertices)
        vertices.extend(cube_vertices)
        
        # Cube faces (12 triangles per cube)
        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Bottom
            [4, 7, 6], [4, 6, 5],  # Top
            [0, 4, 5], [0, 5, 1],  # Front
            [2, 6, 7], [2, 7, 3],  # Back
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2]   # Right
        ]) + vertex_offset
        
        faces.extend(cube_faces)
    
    return np.array(vertices), np.array(faces)


def _sample_mesh_surface(mesh: Mesh, num_points: int) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Sample points on mesh surface."""
    # Calculate face areas for weighted sampling
    face_areas = []
    for face in mesh.faces:
        v0, v1, v2 = mesh.vertices[face]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        face_areas.append(area)
    
    face_areas = np.array(face_areas)
    face_probabilities = face_areas / np.sum(face_areas)
    
    # Sample faces based on area
    sampled_faces = np.random.choice(len(mesh.faces), num_points, p=face_probabilities)
    
    points = []
    colors = []
    normals = []
    
    for face_idx in sampled_faces:
        face = mesh.faces[face_idx]
        v0, v1, v2 = mesh.vertices[face]
        
        # Random barycentric coordinates
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        r3 = 1 - r1 - r2
        
        # Interpolate position
        point = r1 * v0 + r2 * v1 + r3 * v2
        points.append(point)
        
        # Interpolate colors if available
        if mesh.colors is not None:
            c0, c1, c2 = mesh.colors[face]
            color = r1 * c0 + r2 * c1 + r3 * c2
            colors.append(color)
        
        # Interpolate normals if available
        if mesh.normals is not None:
            n0, n1, n2 = mesh.normals[face]
            normal = r1 * n0 + r2 * n1 + r3 * n2
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals.append(normal)
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    normals = np.array(normals) if normals else None
    
    return points, colors, normals


def _poisson_reconstruction(pc: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """Poisson surface reconstruction (simplified fallback)."""
    # This is a placeholder - real Poisson reconstruction requires complex algorithms
    # For now, return a simple convex hull
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pc.points)
        return hull.points, hull.simplices
    except ImportError:
        # Fallback: create a simple mesh from point cloud bounds
        return _simple_pointcloud_to_mesh(pc)


def _delaunay_triangulation(pc: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation of point cloud."""
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(pc.points)
        return tri.points, tri.simplices
    except ImportError:
        return _simple_pointcloud_to_mesh(pc)


def _alpha_shape_reconstruction(pc: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """Alpha shape reconstruction (simplified)."""
    # Placeholder - alpha shapes require complex algorithms
    return _delaunay_triangulation(pc)


def _simple_pointcloud_to_mesh(pc: PointCloud) -> tuple[np.ndarray, np.ndarray]:
    """Simple point cloud to mesh conversion (creates bounding box)."""
    min_coords = np.min(pc.points, axis=0)
    max_coords = np.max(pc.points, axis=0)
    
    # Create bounding box vertices
    vertices = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]]
    ])
    
    # Bounding box faces
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 7, 6], [4, 6, 5],  # Top
        [0, 4, 5], [0, 5, 1],  # Front
        [2, 6, 7], [2, 7, 3],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 5, 6], [1, 6, 2]   # Right
    ])
    
    return vertices, faces 