import warnings
from typing import Any, Sequence, cast

import numpy as np
import pymeshlab as ml
import torch
import trimesh
from scipy.sparse import coo_matrix, csc_matrix
from sklearn.neighbors import KDTree


def _require_triangular_faces(faces: np.ndarray) -> None:
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape [F, 3], got {tuple(faces.shape)}")


def _validate_seq_params(seq_length: int, dilation: int) -> None:
    if seq_length <= 0:
        raise ValueError(f"seq_length must be positive, got {seq_length}")
    if dilation <= 0:
        raise ValueError(f"dilation must be positive, got {dilation}")


def scipy_to_torch_sparse(scp_matrix: Any) -> torch.Tensor:
    """Convert a SciPy COO sparse matrix to a torch sparse COO tensor."""

    if not isinstance(scp_matrix, coo_matrix):
        scp_matrix = scp_matrix.tocoo()

    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.as_tensor(indices, dtype=torch.long)
    v = torch.as_tensor(values, dtype=torch.float32)
    matrix_shape = getattr(scp_matrix, "shape", None)
    if matrix_shape is None:
        raise ValueError("Sparse matrix must define a shape")
    shape = tuple(int(dim) for dim in matrix_shape)

    # Suppress the sparse invariant checks warning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Sparse invariant checks.*")
        sparse_tensor = torch.sparse_coo_tensor(
            i, v, torch.Size(shape), check_invariants=False
        )

    return sparse_tensor


def get_vert_connectivity(mesh_v: np.ndarray, mesh_f: np.ndarray) -> csc_matrix:
    """Build vertex–vertex connectivity as a sparse adjacency matrix.

    Args:
        mesh_v: (N, 3) array of vertex positions.
        mesh_f: (F, 3) array of triangle vertex indices.

    Returns:
        vpv: (N, N) CSC sparse matrix, where vpv[i, j] > 0 means
            vertices i and j share an edge.
    """
    _require_triangular_faces(mesh_f)

    num_verts = mesh_v.shape[0]
    vpv: csc_matrix = csc_matrix((num_verts, num_verts))

    # For each directed edge in each triangle, add adjacency.
    for i in range(3):
        src = mesh_f[:, i]
        dst = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(src), dtype=np.float64)

        ij = np.vstack((src.flatten().reshape(1, -1), dst.flatten().reshape(1, -1)))
        mtx = csc_matrix((data, ij), shape=vpv.shape)

        # Add both directions to make it symmetric.
        vpv = vpv + mtx + mtx.T

    return vpv


def setup_deformation_transfer(
    source_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
) -> csc_matrix:
    """Build a sparse matrix mapping source vertices to target vertices.

    Each target vertex is represented as a barycentric combination of three
    vertices on the source mesh (the vertices of the closest face).

    Args:
        source_mesh: The source mesh (coarse).
        target_mesh: The target mesh (fine).

    Returns:
        A (N_target, N_source) CSC sparse weight matrix.
    """
    n_target = target_mesh.vertices.shape[0]
    n_source = source_mesh.vertices.shape[0]

    # 1. Closest points on the source surface.
    query = trimesh.proximity.ProximityQuery(source_mesh)
    closest_points, _, face_indices = query.on_surface(target_mesh.vertices)

    # 2. Barycentric coordinates of closest points in their triangles.
    closest_triangles = source_mesh.vertices[source_mesh.faces[face_indices]]
    barycentric_coordinates = trimesh.triangles.points_to_barycentric(
        triangles=closest_triangles,
        points=closest_points,
    )

    # 3. Build the sparse matrix.
    rows = np.repeat(np.arange(n_target), 3)
    cols = source_mesh.faces[face_indices].flatten()
    coeffs_v = barycentric_coordinates.flatten()

    matrix = csc_matrix(
        (coeffs_v, (rows, cols)),
        shape=(n_target, n_source),
    )

    return matrix


def quadric_decimator_transformer(
    mesh: trimesh.Trimesh,
    factor: float,
) -> tuple[np.ndarray, csc_matrix]:
    """Simplify a mesh using quadric decimation.

    Args:
        mesh: Trimesh mesh to decimate.
        factor: Fraction of original faces to keep (0 < factor < 1).

    Returns:
        new_faces: (F_new, 3) decimated faces (with new contiguous vertex ids).
        mtx: (N_new, N_orig) CSC sparse transform mapping original vertices
            to decimated vertices.
    """
    if not (0.0 < factor <= 1.0):
        raise ValueError(f"factor must be in (0, 1], got {factor}")

    meshset_cls = cast(Any, getattr(ml, "MeshSet"))
    mesh_cls = cast(Any, getattr(ml, "Mesh"))
    meshlab_ms = meshset_cls()
    meshlab_ms.add_mesh(mesh_cls(mesh.vertices, mesh.faces))
    meshlab_ms.meshing_decimation_quadric_edge_collapse(
        targetperc=factor,
        optimalplacement=False,
        preservetopology=True,
        preserveboundary=False,
        preservenormal=True,
    )
    m_dec = meshlab_ms.current_mesh()
    new_verts = m_dec.vertex_matrix()
    new_faces = m_dec.face_matrix().astype(np.int64)

    n_orig = mesh.vertices.shape[0]
    n_new = new_verts.shape[0]

    # Build a mapping from each new vertex to the original vertex.
    # New vertices should be a subset of original vertices, we use
    # a KDTree to be robust to potential numerical differences.
    tree = KDTree(mesh.vertices)
    _, closest_idx = tree.query(new_verts)  # closest_idx: (n_new,)

    # Build sparse transform: row = new vertex index, col = original vertex index.
    row = np.arange(n_new, dtype=np.int64)
    col = closest_idx.ravel().astype(np.int64)
    data = np.ones(n_new, dtype=np.float64)

    mtx = csc_matrix((data, (row, col)), shape=(n_new, n_orig))

    return new_faces, mtx


def generate_transform_matrices(
    mesh: trimesh.Trimesh,
    factors: Sequence[float],
) -> tuple[
    list[trimesh.Trimesh],
    list[coo_matrix],
    list[coo_matrix],
    list[coo_matrix],
    Sequence[np.ndarray],
    Sequence[np.ndarray],
]:
    """Generate multiresolution meshes and transforms for COMA-style models.

    For each factor f in ``factors``, we:
      - Downsample the previous mesh by roughly a factor of f.
      - Build downsampling matrix D_l (from level l to l+1).
      - Build upsampling matrix U_l (from level l+1 to l) via deformation
        transfer.
      - Build adjacency matrix A_l for each level.

    Args:
        mesh: Finest-resolution template mesh.
        factors: Downsampling factors (e.g. [4, 4, 4, 4]).

    Returns:
        M: List of Trimesh meshes [M_0, M_1, ..., M_L].
        A: List of adjacency matrices (COO), one per level.
        D: List of downsampling transforms D_l (COO), l = 0..L-1.
        U: List of upsampling transforms U_l (COO), l = 0..L-1.
    """
    inv_factors = [1.0 / float(f) for f in factors]

    M: list[trimesh.Trimesh] = []
    A: list[coo_matrix] = []
    D: list[coo_matrix] = []
    U: list[coo_matrix] = []
    F = [mesh.faces]
    V = [mesh.vertices]

    # Level 0 (finest).
    A.append(coo_matrix(get_vert_connectivity(mesh.vertices, mesh.faces)))
    M.append(mesh)

    # Subsequent levels.
    for factor in inv_factors:
        ds_faces, ds_D = quadric_decimator_transformer(M[-1], factor=factor)
        D.append(coo_matrix(ds_D))

        new_mesh_v = ds_D.dot(M[-1].vertices)  # (N_new, 3)

        # Rebuild Trimesh mesh from decimated faces.
        new_mesh = trimesh.Trimesh(
            vertices=new_mesh_v,
            faces=ds_faces,
            process=False,
        )

        F.append(new_mesh.faces)
        V.append(new_mesh.vertices)
        M.append(new_mesh)
        A.append(coo_matrix(get_vert_connectivity(new_mesh_v, ds_faces)))

        # Upsampling from new_mesh (coarse) to previous mesh (fine).
        up_mat = setup_deformation_transfer(
            source_mesh=new_mesh,
            target_mesh=M[-2],
        )
        U.append(coo_matrix(up_mat))

    return M, A, D, U, F, V


def _build_ordered_one_ring(vertices: np.ndarray, faces: np.ndarray) -> list[list[int]]:
    """Return ordered_vv[v] = one-ring neighbor indices in cyclic order.

    Uses directed-edge → face mapping (half-edge style) to traverse the
    one-ring of each vertex consistently, replicating OpenMesh's vv iterator.
    """
    faces = np.asarray(faces, dtype=np.int64)
    _require_triangular_faces(faces)

    # directed edge (u, v) -> face index that contains the half-edge u->v
    edge_to_face = {}
    for fi, f in enumerate(faces):
        for j in range(3):
            edge_to_face[(int(f[j]), int(f[(j + 1) % 3]))] = fi

    # vertex -> list of adjacent face indices
    n_verts = len(vertices)
    vertex_faces = [[] for _ in range(n_verts)]
    for fi, f in enumerate(faces):
        for v in f:
            vertex_faces[int(v)].append(fi)

    ordered_vv = []
    for v in range(n_verts):
        adj = vertex_faces[v]
        if not adj:
            ordered_vv.append([])
            continue

        f0 = faces[adj[-1]]
        v_pos = int(np.where(f0 == v)[0][0])
        first_neighbor = int(f0[(v_pos + 1) % 3])

        ring = []
        current = first_neighbor
        for _ in range(len(adj) + 1):
            ring.append(current)
            next_face_key = (current, v)
            if next_face_key not in edge_to_face:
                break  # boundary vertex
            nf = faces[edge_to_face[next_face_key]]
            vp = int(np.where(nf == v)[0][0])
            nxt = int(nf[(vp + 1) % 3])
            if nxt == first_neighbor:
                break
            current = nxt

        ordered_vv.append(ring)

    return ordered_vv


def _next_ring(
    ordered_vv: list[list[int]], last_ring: list[int], other: list[int]
) -> list[int]:
    res = []

    def is_new_vertex(idx: int) -> bool:
        return idx not in last_ring and idx not in other and idx not in res

    for vh1 in last_ring:
        neighbors = ordered_vv[vh1]
        after_last_ring = False
        for vh2 in neighbors:
            if after_last_ring:
                if is_new_vertex(vh2):
                    res.append(vh2)
            if vh2 in last_ring:
                after_last_ring = True
        for vh2 in neighbors:
            if vh2 in last_ring:
                break
            if is_new_vertex(vh2):
                res.append(vh2)
    return res


def extract_spirals(
    vertices: np.ndarray,
    faces: np.ndarray,
    seq_length: int,
    dilation: int = 1,
) -> list[list[int]]:
    """Extract fixed-length spiral neighborhoods for each vertex."""
    _validate_seq_params(seq_length, dilation)

    ordered_vv = _build_ordered_one_ring(vertices, faces)
    spirals = []
    fallback_tree: KDTree | None = None
    for v0 in range(len(vertices)):
        one_ring = list(ordered_vv[v0])
        spiral = [v0]
        last_ring = one_ring
        next_ring = _next_ring(ordered_vv, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(ordered_vv, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            if fallback_tree is None:
                fallback_tree = KDTree(vertices, metric="euclidean")
            query_result = cast(
                np.ndarray,
                fallback_tree.query(
                np.expand_dims(vertices[spiral[0]], axis=0),
                k=seq_length * dilation,
                return_distance=False,
                ),
            )
            spiral = query_result.tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[: seq_length * dilation][::dilation])
    return spirals


def preprocess_spiral(
    face: np.ndarray,
    seq_length: int,
    vertices: np.ndarray | None = None,
    dilation: int = 1,
) -> torch.Tensor:
    face = np.asarray(face, dtype=np.int64)
    _require_triangular_faces(face)
    _validate_seq_params(seq_length, dilation)

    if vertices is not None:
        verts = np.asarray(vertices, dtype=np.float64)
    else:
        n_vertices = int(face.max()) + 1
        verts = np.ones((n_vertices, 3), dtype=np.float64)

    spirals = torch.tensor(
        extract_spirals(verts, face, seq_length=seq_length, dilation=dilation),
        dtype=torch.long,
    )
    return spirals
