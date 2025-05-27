"""
I/O API for Volumatrix.

This module provides functions for loading and exporting 3D objects
in various formats.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

from ..core.object import VolumatrixObject

logger = logging.getLogger(__name__)


def load(filepath: str, **kwargs) -> VolumatrixObject:
    """
    Load a 3D object from a file.

    Args:
        filepath: Path to the file to load
        **kwargs: Additional loading options

    Returns:
        A VolumatrixObject containing the loaded 3D object

    Examples:
        >>> obj = load("model.obj")
        >>> obj = load("model.glb")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Determine format from extension
    extension = filepath.suffix.lower()

    if extension == ".obj":
        return _load_obj(filepath, **kwargs)
    elif extension in [".glb", ".gltf"]:
        return _load_gltf(filepath, **kwargs)
    elif extension == ".ply":
        return _load_ply(filepath, **kwargs)
    elif extension == ".stl":
        return _load_stl(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def export(
    obj: VolumatrixObject, filepath: str, format: Optional[str] = None, **kwargs
) -> None:
    """
    Export a VolumatrixObject to a file.

    Args:
        obj: The VolumatrixObject to export
        filepath: Output file path
        format: Export format (inferred from extension if None)
        **kwargs: Additional export options

    Examples:
        >>> export(obj, "output.obj")
        >>> export(obj, "output.glb", format="glb")
    """
    filepath = Path(filepath)

    # Create output directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format is None:
        format = filepath.suffix.lower().lstrip(".")

    if format == "obj":
        _export_obj(obj, filepath, **kwargs)
    elif format in ["glb", "gltf"]:
        _export_gltf(obj, filepath, **kwargs)
    elif format == "ply":
        _export_ply(obj, filepath, **kwargs)
    elif format == "stl":
        _export_stl(obj, filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")

    logger.info(f"Exported object to {filepath}")


def save(obj: VolumatrixObject, filepath: str, **kwargs) -> None:
    """
    Save a VolumatrixObject (alias for export).

    Args:
        obj: The VolumatrixObject to save
        filepath: Output file path
        **kwargs: Additional save options
    """
    export(obj, filepath, **kwargs)


def _load_obj(filepath: Path, **kwargs) -> VolumatrixObject:
    """Load an OBJ file."""
    try:
        import trimesh

        mesh = trimesh.load(str(filepath))

        from ..core.representations import Mesh

        volumatrix_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.vertex_normals if hasattr(mesh, "vertex_normals") else None,
        )

        return VolumatrixObject(
            name=filepath.stem, representations={"mesh": volumatrix_mesh}
        )
    except ImportError:
        raise ImportError(
            "trimesh is required for OBJ file support. Install with: pip install trimesh"
        )


def _load_gltf(filepath: Path, **kwargs) -> VolumatrixObject:
    """Load a GLTF/GLB file."""
    try:
        import trimesh

        mesh = trimesh.load(str(filepath))

        from ..core.representations import Mesh

        volumatrix_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.vertex_normals if hasattr(mesh, "vertex_normals") else None,
        )

        return VolumatrixObject(
            name=filepath.stem, representations={"mesh": volumatrix_mesh}
        )
    except ImportError:
        raise ImportError(
            "trimesh is required for GLTF file support. Install with: pip install trimesh"
        )


def _load_ply(filepath: Path, **kwargs) -> VolumatrixObject:
    """Load a PLY file."""
    try:
        import trimesh

        mesh = trimesh.load(str(filepath))

        from ..core.representations import Mesh

        volumatrix_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.vertex_normals if hasattr(mesh, "vertex_normals") else None,
        )

        return VolumatrixObject(
            name=filepath.stem, representations={"mesh": volumatrix_mesh}
        )
    except ImportError:
        raise ImportError(
            "trimesh is required for PLY file support. Install with: pip install trimesh"
        )


def _load_stl(filepath: Path, **kwargs) -> VolumatrixObject:
    """Load an STL file."""
    try:
        import trimesh

        mesh = trimesh.load(str(filepath))

        from ..core.representations import Mesh

        volumatrix_mesh = Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            normals=mesh.vertex_normals if hasattr(mesh, "vertex_normals") else None,
        )

        return VolumatrixObject(
            name=filepath.stem, representations={"mesh": volumatrix_mesh}
        )
    except ImportError:
        raise ImportError(
            "trimesh is required for STL file support. Install with: pip install trimesh"
        )


def _export_obj(obj: VolumatrixObject, filepath: Path, **kwargs) -> None:
    """Export to OBJ format."""
    # Get mesh representation
    mesh = obj.mesh
    if mesh is None:
        raise ValueError("Object must have a mesh representation to export as OBJ")

    try:
        import trimesh

        # Create trimesh object
        trimesh_mesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.normals
        )

        # Export
        trimesh_mesh.export(str(filepath))

    except ImportError:
        # Fallback: simple OBJ writer
        _write_simple_obj(mesh, filepath)


def _export_gltf(obj: VolumatrixObject, filepath: Path, **kwargs) -> None:
    """Export to GLTF/GLB format."""
    mesh = obj.mesh
    if mesh is None:
        raise ValueError("Object must have a mesh representation to export as GLTF")

    try:
        import trimesh

        trimesh_mesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.normals
        )

        trimesh_mesh.export(str(filepath))

    except ImportError:
        raise ImportError(
            "trimesh is required for GLTF export. Install with: pip install trimesh"
        )


def _export_ply(obj: VolumatrixObject, filepath: Path, **kwargs) -> None:
    """Export to PLY format."""
    mesh = obj.mesh
    if mesh is None:
        raise ValueError("Object must have a mesh representation to export as PLY")

    try:
        import trimesh

        trimesh_mesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.normals
        )

        trimesh_mesh.export(str(filepath))

    except ImportError:
        raise ImportError(
            "trimesh is required for PLY export. Install with: pip install trimesh"
        )


def _export_stl(obj: VolumatrixObject, filepath: Path, **kwargs) -> None:
    """Export to STL format."""
    mesh = obj.mesh
    if mesh is None:
        raise ValueError("Object must have a mesh representation to export as STL")

    try:
        import trimesh

        trimesh_mesh = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, vertex_normals=mesh.normals
        )

        trimesh_mesh.export(str(filepath))

    except ImportError:
        # Fallback: simple STL writer
        _write_simple_stl(mesh, filepath)


def _write_simple_obj(mesh, filepath: Path) -> None:
    """Simple OBJ file writer (fallback when trimesh is not available)."""
    with open(filepath, "w") as f:
        f.write(f"# Volumatrix OBJ export")
        f.write(f"# Vertices: {len(mesh.vertices)}")
        f.write(f"# Faces: {len(mesh.faces)}")

        # Write vertices
        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")

        # Write normals if available
        if mesh.normals is not None:
            for normal in mesh.normals:
                f.write(f"vn {normal[0]:.6f} {normal[1]:.6f} {normal[2]:.6f}")

        f.write("")

        # Write faces (OBJ uses 1-based indexing)
        for face in mesh.faces:
            if mesh.normals is not None:
                f.write(
                    f"f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}"
                )
            else:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")


def _write_simple_stl(mesh, filepath: Path) -> None:
    """Simple STL file writer (fallback when trimesh is not available)."""
    import struct

    with open(filepath, "wb") as f:
        # Write header (80 bytes)
        header = b"Volumatrix STL export" + b"\0" * (80 - 21)
        f.write(header)

        # Write number of triangles
        f.write(struct.pack("<I", len(mesh.faces)))

        # Write triangles
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]

            # Calculate face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)

            # Write normal
            f.write(struct.pack("<fff", normal[0], normal[1], normal[2]))

            # Write vertices
            f.write(struct.pack("<fff", v0[0], v0[1], v0[2]))
            f.write(struct.pack("<fff", v1[0], v1[1], v1[2]))
            f.write(struct.pack("<fff", v2[0], v2[1], v2[2]))

            # Write attribute byte count (unused)
            f.write(struct.pack("<H", 0))
