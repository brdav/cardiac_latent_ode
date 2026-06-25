from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import torch
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from cardiac_latent_ode.utils.pylogger import RichLogger
from cardiac_latent_ode.utils.mesh_processing import as_pv_polydata
import random

log = RichLogger(__name__)


def load_vtk_as_trimesh(path: str | Path) -> trimesh.Trimesh:
    """Load a VTK mesh file via PyVista and convert it to :class:`trimesh.Trimesh`."""
    pv_mesh = pv.read(path)
    vertices = np.asarray(pv_mesh.points)
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]  # drop face-length prefix
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return tm


def save_trimesh_as_vtk(mesh: trimesh.Trimesh, filename: str | Path) -> None:
    """Save a :class:`trimesh.Trimesh` instance to a VTK file via PyVista."""
    pv_mesh = as_pv_polydata(mesh.vertices, mesh.faces)
    pv_mesh.save(filename)


def compute_merge_map(
    vertices: np.ndarray, threshold_percent: float = 0.01
) -> tuple[np.ndarray, int]:
    """
    Compute a mapping from old vertices to new vertices based on distance merging.
    Replicates the logic of merging close vertices on a template mesh.

    Args:
        vertices: (N, 3) array of vertex positions
        threshold_percent: Percentage of bounding box diagonal to use as distance threshold.

    Returns:
        labels: (N,) array where labels[i] is the new vertex index for old vertex i.
        n_unique: Number of unique vertices after merging.
    """
    # Calculate threshold distance based on bounding box diagonal
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    r = diag * (threshold_percent / 100.0)

    # Find pairs of vertices closer than r
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(r)
    pairs = np.array(list(pairs))

    n = len(vertices)
    if len(pairs) > 0:
        # Build graph where edges connect close vertices
        row = pairs[:, 0]
        col = pairs[:, 1]
        data = np.ones(len(pairs), dtype=bool)
        # Symmetric adjacency
        adj = coo_matrix((data, (row, col)), shape=(n, n))
        adj = adj + adj.T

        # Find connected components (these become the new merged vertices)
        n_unique, labels = connected_components(adj, directed=False)
    else:
        n_unique = n
        labels = np.arange(n)

    return labels, n_unique


def apply_merge_map(
    vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, n_unique: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a pre-computed vertex merge mapping to a mesh.
    Ensures topological consistency with the template used to generate the map.
    """
    # 1. Compute new vertex positions (average of merged vertices)
    # We use bincount to sum coordinates for each label
    count = np.bincount(labels, minlength=n_unique)
    # Avoid division by zero (connected_components ensures count >= 1)
    count = np.maximum(count, 1).reshape(-1, 1)

    new_x = np.bincount(labels, weights=vertices[:, 0], minlength=n_unique)
    new_y = np.bincount(labels, weights=vertices[:, 1], minlength=n_unique)
    new_z = np.bincount(labels, weights=vertices[:, 2], minlength=n_unique)

    new_vertices = np.stack([new_x, new_y, new_z], axis=1) / count

    # 2. Re-index faces using the map
    new_faces = labels[faces]

    # 3. Remove degenerate faces (where vertices merged into same index)
    # A face is degenerate if v0==v1 or v1==v2 or v2==v0
    valid_mask = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )

    new_faces = new_faces[valid_mask]

    return new_vertices, new_faces


def seed_everything(seed: int | None = None) -> int:
    """Function that sets seed for pseudo-random number generators.

    Args:
        seed: the integer value seed for global random state.
            If `None`, will select it randomly.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        log.warning(
            f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = random.randint(min_seed_value, max_seed_value)

    log.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
