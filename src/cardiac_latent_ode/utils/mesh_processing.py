import io
import math
from importlib.resources import files
from typing import Any, cast

import numpy as np
import pyvista as pv
import torch

from cardiac_latent_ode.utils.pylogger import RichLogger

log = RichLogger(__name__)

_ASSETS = files("cardiac_latent_ode.assets")


def _optional_index_array(z, key):
    if key not in z:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(z[key], dtype=np.int64).reshape(-1)


def _decode_annulus_loops(flat, offsets):
    """Decode flat annulus arrays back into list[list[int]]."""
    flat_arr = np.asarray(flat, dtype=np.int64).reshape(-1)
    off = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if off.size == 0:
        return []

    loops = []
    for i in range(off.size - 1):
        s = int(off[i])
        e = int(off[i + 1])
        loops.append(flat_arr[s:e].tolist())
    return loops


def load_landmark_bundle():
    """Load clinical landmark indices and annulus loops from NPZ."""
    data = _ASSETS.joinpath("landmark_indices_bundle.npz").read_bytes()
    z = np.load(io.BytesIO(data), allow_pickle=False)
    return {
        "format_version": int(z["format_version"]) if "format_version" in z else 0,
        "lv_endo_base_idx": np.asarray(z["lv_endo_base_idx"], dtype=np.int64),
        "lv_endo_apex_idx": int(z["lv_endo_apex_idx"]),
        "lv_endo_annulus_indices": _decode_annulus_loops(
            z["lv_endo_annulus_flat"], z["lv_endo_annulus_offsets"]
        ),
        "lv_epi_annulus_indices": _decode_annulus_loops(
            z["lv_epi_annulus_flat"], z["lv_epi_annulus_offsets"]
        ),
        "rv_endo_annulus_indices": _decode_annulus_loops(
            z["rv_endo_annulus_flat"], z["rv_endo_annulus_offsets"]
        ),
        "rv_epi_annulus_indices": _decode_annulus_loops(
            z["rv_epi_annulus_flat"], z["rv_epi_annulus_offsets"]
        ),
        "lv_gls_2ch_idx": _optional_index_array(z, "lv_gls_2ch_idx"),
        "lv_gls_4ch_idx": _optional_index_array(z, "lv_gls_4ch_idx"),
        "rvfw_gls_4ch_idx": _optional_index_array(z, "rvfw_gls_4ch_idx"),
    }


def load_remap_bundle():
    """Load remapping bundle and return metadata + arrays."""
    data = _ASSETS.joinpath("mesh_remap_bundle.npz").read_bytes()
    z = np.load(io.BytesIO(data), allow_pickle=False)

    meta = {
        "combined_n_vertices": int(z["combined_n_vertices"]),
        "precision_dtype": str(z["precision_dtype"]),
    }

    component_to_combined = {}
    local_faces = {}
    for key in z.files:
        if key.startswith("map_"):
            component_to_combined[key.replace("map_", "")] = z[key]
        elif key.startswith("faces_"):
            local_faces[key.replace("faces_", "")] = z[key]

    return meta, component_to_combined, local_faces


def as_pv_polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """Create a PyVista PolyData from (V,F) triangles.

    Parameters
    - vertices: (N, 3)
    - faces: (F, 3) triangle indices into vertices
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3); got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3); got {faces.shape}")

    # PyVista expects faces encoded as: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    faces_pv = np.concatenate(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces], axis=1
    ).reshape(-1)
    return pv.PolyData(vertices, faces_pv)


def cap_mesh_with_annulus_indices(
    vertices: np.ndarray,
    faces: np.ndarray,
    annulus_indices: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Cap valve annuli using provided ordered boundary vertex indices.

    Each inner list is one annulus loop (in mesh vertex indexing). A centroid
    fan is added so every provided annulus vertex participates in the cap.
    """
    new_vertices = [vertices]
    new_faces = [faces]
    next_vid = int(vertices.shape[0])

    for loop in annulus_indices:
        ids = np.asarray(loop, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            continue

        # Keep only valid vertex ids, preserve order.
        ids = ids[(ids >= 0) & (ids < int(vertices.shape[0]))]
        if ids.size < 3:
            continue

        # Remove repeated consecutive ids and optional closing duplicate.
        cleaned = [int(ids[0])]
        for vid in ids[1:]:
            vid_i = int(vid)
            if vid_i != cleaned[-1]:
                cleaned.append(vid_i)
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
            cleaned = cleaned[:-1]
        if len(cleaned) < 3:
            continue

        loop_ids = np.asarray(cleaned, dtype=np.int64)
        center = np.mean(vertices[loop_ids], axis=0, dtype=np.float64)
        new_vertices.append(center.reshape(1, 3))
        c = next_vid
        next_vid += 1

        fan = np.column_stack(
            [
                loop_ids,
                np.roll(loop_ids, -1),
                np.full(loop_ids.shape[0], c, dtype=np.int64),
            ]
        )
        new_faces.append(fan)

    if len(new_faces) == 1:
        return vertices, faces

    v_out = np.concatenate(new_vertices, axis=0)
    f_out = np.concatenate(new_faces, axis=0).astype(np.int64, copy=False)
    return v_out, f_out


def signed_volume_ml(
    vertices: np.ndarray,
    faces: np.ndarray,
    annulus_indices: list[list[int]],
) -> float:
    vertices, faces = cap_mesh_with_annulus_indices(vertices, faces, annulus_indices)

    mesh = cast(
        pv.PolyData,
        as_pv_polydata(vertices, faces)
        .extract_surface(algorithm=None)
        .triangulate()
        .clean(),
    )

    # Enforce consistent orientation on the closed triangulated surface.
    # This helps keep downstream signed-volume computations stable.
    mesh = cast(
        pv.PolyData,
        mesh.compute_normals(
            point_normals=False,
            cell_normals=True,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=False,
        ),
    )
    return abs(float(mesh.volume)) / 1000.0


def compute_volumes_from_remapped(remapped: dict[str, object]) -> dict[str, np.ndarray]:
    """Close remapped component meshes and compute volumes over batch/time.

    Expected entries in ``remapped``:
    - ``*_v_bvtc`` with shape [B, V, T, 3]
    - ``*_f_f3`` with shape [F, 3]
    - ``*_annulus_indices`` as list[list[int]] for epi/endo closure

    Returns a dict of volume arrays with shape [B, T].
    """
    lv_endo_v_bvtc = np.asarray(remapped["lv_endo_v_bvtc"], dtype=np.float64)
    batch_size, _, num_frames, _ = lv_endo_v_bvtc.shape

    faces_np = {}
    for key in ("lv_endo_f_f3", "lv_epi_f_f3", "rv_endo_f_f3", "rv_epi_f_f3"):
        faces_np[key] = np.asarray(remapped[key], dtype=np.int64)

    verts_np = {}
    for key in ("lv_endo_v_bvtc", "lv_epi_v_bvtc", "rv_endo_v_bvtc", "rv_epi_v_bvtc"):
        verts_np[key] = np.asarray(remapped[key], dtype=np.float64)

    out = {
        "lv_endo_volume": np.empty((batch_size, num_frames), dtype=np.float32),
        "lv_epi_volume": np.empty((batch_size, num_frames), dtype=np.float32),
        "rv_endo_volume": np.empty((batch_size, num_frames), dtype=np.float32),
        "rv_epi_volume": np.empty((batch_size, num_frames), dtype=np.float32),
    }

    lv_endo_annulus_indices = cast(list[list[int]], remapped["lv_endo_annulus_indices"])
    lv_epi_annulus_indices = cast(list[list[int]], remapped["lv_epi_annulus_indices"])
    rv_endo_annulus_indices = cast(list[list[int]], remapped["rv_endo_annulus_indices"])
    rv_epi_annulus_indices = cast(list[list[int]], remapped["rv_epi_annulus_indices"])

    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            out["lv_endo_volume"][batch_idx, frame_idx] = signed_volume_ml(
                verts_np["lv_endo_v_bvtc"][batch_idx, :, frame_idx, :],
                faces_np["lv_endo_f_f3"],
                annulus_indices=lv_endo_annulus_indices,
            )
            out["lv_epi_volume"][batch_idx, frame_idx] = signed_volume_ml(
                verts_np["lv_epi_v_bvtc"][batch_idx, :, frame_idx, :],
                faces_np["lv_epi_f_f3"],
                annulus_indices=lv_epi_annulus_indices,
            )
            out["rv_endo_volume"][batch_idx, frame_idx] = signed_volume_ml(
                verts_np["rv_endo_v_bvtc"][batch_idx, :, frame_idx, :],
                faces_np["rv_endo_f_f3"],
                annulus_indices=rv_endo_annulus_indices,
            )
            out["rv_epi_volume"][batch_idx, frame_idx] = signed_volume_ml(
                verts_np["rv_epi_v_bvtc"][batch_idx, :, frame_idx, :],
                faces_np["rv_epi_f_f3"],
                annulus_indices=rv_epi_annulus_indices,
            )

    out["lv_myo_volume"] = out["lv_epi_volume"] - out["lv_endo_volume"]
    out["rv_myo_volume"] = out["rv_epi_volume"] - out["rv_endo_volume"]
    return out


def unit_vector(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 0.0:
        return None
    return v.astype(np.float64) / n


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return R (3x3) such that R @ a_hat ~= b_hat."""
    a_hat = unit_vector(a)
    b_hat = unit_vector(b)
    if a_hat is None or b_hat is None:
        return np.eye(3, dtype=np.float64)

    c = float(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0))
    if np.isclose(c, 1.0):
        return np.eye(3, dtype=np.float64)
    if np.isclose(c, -1.0):
        trial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a_hat, trial))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = unit_vector(np.cross(a_hat, trial))
        if axis is None:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    v = np.cross(a_hat, b_hat)
    s = float(np.linalg.norm(v))
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def rotate_vertices(
    vertices: np.ndarray, R: np.ndarray, pivot: np.ndarray
) -> np.ndarray:
    p = pivot.reshape(1, 3)
    return ((vertices - p) @ R.T + p).astype(vertices.dtype, copy=False)


def lv_wall_thickness(
    lv_endo_vertices: np.ndarray,
    lv_endo_faces: np.ndarray,
    lv_epi_vertices: np.ndarray,
    lv_epi_faces: np.ndarray,
    lv_base_idx: np.ndarray,
    lv_apex_idx: int,
) -> dict[str, float]:
    """Compute LV wall thickness and internal diameter in mm from endo/epi meshes.

    Resulting wall thickness values are quite low, but consistent with:
    https://biobank.ndph.ox.ac.uk/ukb/field.cgi?id=24140
    """

    lv_endo_mesh = as_pv_polydata(lv_endo_vertices, lv_endo_faces)
    lv_epi_mesh = as_pv_polydata(lv_epi_vertices, lv_epi_faces)

    z_base = np.mean([lv_endo_mesh.points[i][2] for i in lv_base_idx])
    z_apex = lv_endo_mesh.points[lv_apex_idx][2]

    # We only consider points in the LV mid-cavity segment
    z_min = z_apex + 0.35 * (z_base - z_apex)
    z_max = z_apex + 0.65 * (z_base - z_apex)

    # Left ventricle
    filtered_lv_endo_points = lv_endo_mesh.points[
        (lv_endo_mesh.points[:, 2] >= z_min) & (lv_endo_mesh.points[:, 2] <= z_max)
    ]
    _, closest_points_lv = cast(
        tuple[np.ndarray, np.ndarray],
        lv_epi_mesh.find_closest_cell(
            filtered_lv_endo_points, return_closest_point=True
        ),
    )
    d_exact_lv = np.linalg.norm(filtered_lv_endo_points - closest_points_lv, axis=1)
    mean_lv_thickness = np.mean(d_exact_lv).item()

    lv_diameters = []
    for z in np.linspace(z_min, z_max, 10):
        lv_slice = lv_endo_mesh.slice(normal="z", origin=(0, 0, z)).delaunay_2d()
        lv_diameters.append(2 * (lv_slice.area / math.pi) ** 0.5)
    mean_lv_diameter = np.mean(lv_diameters).item()

    return {
        "lv_thickness": mean_lv_thickness,
        "lv_diameter": mean_lv_diameter,
    }


def compute_wall_thickness_from_remapped(
    remapped: dict[str, object],
) -> dict[str, np.ndarray]:
    """Rotate, and compute wall thickness/diameter over batch and time."""
    lv_endo_v_bvtc = np.asarray(remapped["lv_endo_v_bvtc"], dtype=np.float64)
    lv_epi_v_bvtc = np.asarray(remapped["lv_epi_v_bvtc"], dtype=np.float64)

    batch_size, _, num_frames, _ = lv_endo_v_bvtc.shape

    lv_endo_f = np.asarray(remapped["lv_endo_f_f3"], dtype=np.int64)
    lv_epi_f = np.asarray(remapped["lv_epi_f_f3"], dtype=np.int64)

    lv_endo_base_idx = np.asarray(remapped["lv_endo_base_idx"], dtype=np.int64)
    lv_endo_apex_idx = cast(int, remapped["lv_endo_apex_idx"])

    out = {
        "lv_thickness": np.empty((batch_size, num_frames), dtype=np.float32),
        "lv_diameter": np.empty((batch_size, num_frames), dtype=np.float32),
    }

    for batch_idx in range(batch_size):
        for frame_idx in range(num_frames):
            lv_endo_v = lv_endo_v_bvtc[batch_idx, :, frame_idx, :]
            lv_epi_v = lv_epi_v_bvtc[batch_idx, :, frame_idx, :]

            base_center = np.mean(lv_endo_v[lv_endo_base_idx], axis=0)
            apex_pt = lv_endo_v[lv_endo_apex_idx]
            long_axis_vec = np.asarray(apex_pt - base_center, dtype=np.float64)
            target_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            if np.linalg.norm(long_axis_vec) > 0.0:
                R = rotation_matrix_from_vectors(long_axis_vec, target_dir)
                pivot = 0.5 * (
                    np.asarray(apex_pt, dtype=np.float64)
                    + np.asarray(base_center, dtype=np.float64)
                )
                lv_endo_v = rotate_vertices(lv_endo_v, R, pivot)
                lv_epi_v = rotate_vertices(lv_epi_v, R, pivot)

            th = lv_wall_thickness(
                lv_endo_vertices=lv_endo_v,
                lv_endo_faces=lv_endo_f,
                lv_epi_vertices=lv_epi_v,
                lv_epi_faces=lv_epi_f,
                lv_base_idx=lv_endo_base_idx,
                lv_apex_idx=lv_endo_apex_idx,
            )

            out["lv_thickness"][batch_idx, frame_idx] = th["lv_thickness"]
            out["lv_diameter"][batch_idx, frame_idx] = th["lv_diameter"]

    return out


def _polyline_length_bvtc(
    vertices_bvtc: np.ndarray,
    path_indices: np.ndarray,
) -> np.ndarray:
    """Compute polyline lengths for one indexed path over batched frames."""
    path = np.asarray(path_indices, dtype=np.int64).reshape(-1)
    if path.size < 2:
        batch_size = int(vertices_bvtc.shape[0])
        num_frames = int(vertices_bvtc.shape[2])
        return np.full((batch_size, num_frames), np.nan, dtype=np.float32)

    points = vertices_bvtc[:, path, :, :]
    segments = points[:, 1:, :, :] - points[:, :-1, :, :]
    lengths = np.linalg.norm(segments, axis=-1).sum(axis=1)
    return np.asarray(lengths, dtype=np.float32)


def compute_longitudinal_strain_from_remapped(
    remapped: dict[str, object],
) -> dict[str, np.ndarray]:
    """Compute legacy longitudinal-strain path lengths over batch and time.

    The returned arrays have shape [B, T].
    """
    lv_endo_v_bvtc = np.asarray(remapped["lv_endo_v_bvtc"], dtype=np.float64)
    rv_endo_v_bvtc = np.asarray(remapped["rv_endo_v_bvtc"], dtype=np.float64)

    return {
        "lv_gls_2ch": _polyline_length_bvtc(
            lv_endo_v_bvtc,
            np.asarray(remapped["lv_gls_2ch_idx"], dtype=np.int64),
        ),
        "lv_gls_4ch": _polyline_length_bvtc(
            lv_endo_v_bvtc,
            np.asarray(remapped["lv_gls_4ch_idx"], dtype=np.int64),
        ),
        "rvfw_gls_4ch": _polyline_length_bvtc(
            rv_endo_v_bvtc,
            np.asarray(remapped["rvfw_gls_4ch_idx"], dtype=np.int64),
        ),
    }


def _strain_percent(length_ed: np.ndarray, length_es: np.ndarray) -> np.ndarray:
    out = np.full(length_ed.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(length_ed) & np.isfinite(length_es) & (length_ed > 0)
    out[valid] = ((length_es[valid] - length_ed[valid]) / length_ed[valid]) * 100.0
    return out


def extract_clinical_markers(
    x_bvtc: torch.Tensor,
    bsa: np.ndarray,
    myocardial_density_g_per_ml: float = 1.05,
) -> dict[str, Any]:
    """Remap combined temporal meshes [B,V,T,C] to component meshes and compute clinical markers."""

    if x_bvtc.ndim != 4:
        raise ValueError(
            f"Expected x_bvtc with shape [B, V, T, C], got {tuple(x_bvtc.shape)}"
        )

    # Step 1. Remap the vertices from the standardized combined space to individual component meshes
    meta, c2c_reload, faces_reload = load_remap_bundle()
    if x_bvtc.shape[1] != meta["combined_n_vertices"]:
        raise ValueError(
            "Bundle/topology mismatch: different combined vertex count. "
            f"x_bvtc V={x_bvtc.shape[1]}, bundle V={meta['combined_n_vertices']}"
        )

    # Remap along vertex axis: local component vertices = x_bvtc[:, map_idx, :, :]
    remapped: dict[str, Any] = {}
    for name, idx_np in c2c_reload.items():
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=x_bvtc.device)
        remapped[f"{name}_v_bvtc"] = x_bvtc.index_select(1, idx).detach().cpu().numpy().astype(np.float64)
        remapped[f"{name}_f_f3"] = np.asarray(faces_reload[name], dtype=np.int64)

    # Step 2. Load reusable landmark indices (same local component indexing).
    landmark_bundle = load_landmark_bundle()
    remapped["lv_endo_base_idx"] = np.asarray(landmark_bundle["lv_endo_base_idx"], dtype=np.int64)
    remapped["lv_endo_apex_idx"] = int(landmark_bundle["lv_endo_apex_idx"])
    remapped["lv_endo_annulus_indices"] = landmark_bundle["lv_endo_annulus_indices"]
    remapped["lv_epi_annulus_indices"] = landmark_bundle["lv_epi_annulus_indices"]
    remapped["rv_endo_annulus_indices"] = landmark_bundle["rv_endo_annulus_indices"]
    remapped["rv_epi_annulus_indices"] = landmark_bundle["rv_epi_annulus_indices"]
    for key in (
        "lv_gls_2ch_idx",
        "lv_gls_4ch_idx",
        "rvfw_gls_4ch_idx",
    ):
        remapped[key] = np.asarray(landmark_bundle[key], dtype=np.int64)

    # Step 3. Compute volumes for each component and time step on a closed version of the remapped mesh
    volumes = compute_volumes_from_remapped(remapped)

    # Step 4. Compute wall thickness on a rotated version of the remapped mesh
    thicknesses = compute_wall_thickness_from_remapped(remapped)

    # Step 5. Compute strains on the remapped mesh
    strains = compute_longitudinal_strain_from_remapped(remapped)

    # Compute the different CMR markers
    # ED is the first frame, ES is where LV volume is minimal
    lv_endo_volume = volumes["lv_endo_volume"]
    rv_endo_volume = volumes["rv_endo_volume"]
    lv_myo_volume = volumes["lv_myo_volume"]
    lv_thickness = thicknesses["lv_thickness"]
    lv_diameter = thicknesses["lv_diameter"]
    lv_gls_2ch = strains["lv_gls_2ch"]
    lv_gls_4ch = strains["lv_gls_4ch"]
    rvfw_gls_4ch = strains["rvfw_gls_4ch"]

    es_frame = np.argmin(lv_endo_volume, axis=1)
    batch_idx = np.arange(lv_endo_volume.shape[0], dtype=np.int64)

    lv_edv = lv_endo_volume[:, 0]
    lv_esv = lv_endo_volume[batch_idx, es_frame]
    rv_edv = rv_endo_volume[:, 0]
    rv_esv = rv_endo_volume[batch_idx, es_frame]

    # Left ventricular ejection fraction
    lvef = ((lv_edv - lv_esv) / lv_edv) * 100.0

    # Right ventricular ejection fraction
    rvef = ((rv_edv - rv_esv) / rv_edv) * 100.0

    # Left ventricular global longitudinal strain
    lv_len_ed = 0.5 * (lv_gls_2ch[:, 0] + lv_gls_4ch[:, 0])
    lv_len_es = 0.5 * (
        lv_gls_2ch[batch_idx, es_frame] + lv_gls_4ch[batch_idx, es_frame]
    )
    lv_gls = _strain_percent(lv_len_ed, lv_len_es)

    # Right ventricular free wall longitudinal strain
    rv_fwls = _strain_percent(rvfw_gls_4ch[:, 0], rvfw_gls_4ch[batch_idx, es_frame])

    # Left ventricular wall thickness (at ED)
    lv_wall_thickness = lv_thickness[:, 0]

    # Relative wall thickness (at ED)
    lvid_ed = lv_diameter[:, 0]
    rwt = (2.0 * lv_wall_thickness) / lvid_ed

    # Left ventricular mass index (at ED)
    lv_mass_g = lv_myo_volume[:, 0] * myocardial_density_g_per_ml
    lvmi = lv_mass_g / bsa

    return {
        "LVEF": lvef,
        "RVEF": rvef,
        "LV-GLS": lv_gls,
        "RV-FWLS": rv_fwls,
        "LVMi": lvmi,
        "LVWT": lv_wall_thickness,
        "RWT": rwt,
    }
