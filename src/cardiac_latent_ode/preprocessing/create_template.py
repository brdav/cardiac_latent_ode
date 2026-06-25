from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

from cardiac_latent_ode.utils.utils import (
    apply_merge_map,
    compute_merge_map,
    load_vtk_as_trimesh,
    save_trimesh_as_vtk,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STRUCTURES = ["epi", "lv_endo", "rv_endo"]
TYPE_MAP = {
    "epi": "EPICARDIAL",
    "lv_endo": "LV_ENDOCARDIAL",
    "rv_endo": "RV_ENDOCARDIAL",
}
ED_FRAME = "000"  # end-diastolic frame used for template construction


def _load_case_meshes(case_dir: Path, case_id: str) -> dict[str, trimesh.Trimesh] | None:
    """Load the ED-frame mesh for each structure of a case.

    Returns None if any structure file is missing.
    """
    meshes: dict[str, trimesh.Trimesh] = {}
    for s in STRUCTURES:
        path = case_dir / "vtk_smoothed" / f"{case_id}_{TYPE_MAP[s]}_{ED_FRAME}.vtk"
        if not path.exists():
            log.warning("Missing %s for case %s; skipping case.", path.name, case_id)
            return None
        meshes[s] = load_vtk_as_trimesh(path)
    return meshes


def create_mean_template(
    bivme_output_dir: Path,
    cohort_file: Path,
    output_dir: Path,
    max_cases: int = 1000,
) -> None:
    """Create a mean template mesh jointly across epi, lv_endo, rv_endo.

    For each training case, loads the ED-frame meshes, concatenates their
    vertices, and rigidly aligns them (Procrustes) to a common reference.
    The mean of the aligned vertices is back-rotated, centered at origin,
    and saved as a merged VTK template with duplicate boundary vertices
    deduplicated.

    Args:
        bivme_fitted_models_dir: Directory containing one subdirectory per case.
        cohort_file: CSV with 'case_id' and 'split' columns.
        output_dir: Where to write template_mesh.vtk and mesh_repair.npz.
        max_cases: Maximum number of training cases to use.
    """
    cohort = pd.read_csv(cohort_file)
    train_cases = set(cohort[cohort["split"] == "train"]["case_id"].astype(str))
    log.info("Building template from up to %d of %d train cases.", max_cases, len(train_cases))

    output_dir.mkdir(parents=True, exist_ok=True)

    accumulated: np.ndarray | None = None
    rotation_sum: np.ndarray | None = None
    reference_vertices: np.ndarray | None = None
    reference_meshes: dict[str, trimesh.Trimesh] | None = None
    reference_splits: list[int] | None = None
    count = 0

    for case_dir in tqdm(sorted((bivme_output_dir / "fitted-models").iterdir())):
        if not case_dir.is_dir() or case_dir.name not in train_cases:
            continue
        if count >= max_cases:
            break

        try:
            meshes = _load_case_meshes(case_dir, case_dir.name)
            if meshes is None:
                continue

            vert_list = [np.asarray(meshes[s].vertices, dtype=np.float64) for s in STRUCTURES]
            splits = [v.shape[0] for v in vert_list]
            concatenated = np.vstack(vert_list)

            if reference_meshes is None:
                reference_meshes = {s: meshes[s].copy() for s in STRUCTURES}
                reference_vertices = concatenated.copy()
                reference_splits = list(splits)
                accumulated = np.zeros_like(reference_vertices)
                rotation_sum = np.zeros((3, 3), dtype=np.float64)

            matrix, transformed, _ = trimesh.registration.procrustes(
                concatenated,
                reference_vertices,
                reflection=False,
                translation=True,
                scale=False,
                return_cost=True,
            )
            accumulated += np.asarray(transformed, dtype=np.float64)
            rotation_sum += matrix[:3, :3]
            count += 1

        except Exception as e:
            log.warning("Error processing %s: %s; skipping.", case_dir.name, e)

    if count == 0 or accumulated is None or rotation_sum is None or reference_splits is None:
        raise RuntimeError("No valid meshes processed; cannot create template.")

    log.info("Computed mean from %d rigidly aligned meshes.", count)

    mean_points = accumulated / count

    # Project average rotation to SO(3) and apply its inverse to undo the
    # mean alignment bias introduced by Procrustes registration.
    U, _, Vt = np.linalg.svd(rotation_sum / count)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0.0:
        raise ValueError("Unexpected reflection in average rotation.")
    mean_points = (R_avg.T @ mean_points.T).T
    mean_points -= mean_points.mean(axis=0, keepdims=True)

    # Split mean vertices back into per-structure arrays.
    per_struct: dict[str, np.ndarray] = {}
    start = 0
    for s, n in zip(STRUCTURES, reference_splits):
        per_struct[s] = mean_points[start : start + n]
        start += n

    # Fix RV endo: merge near-duplicate vertices at open boundaries.
    rv_verts = per_struct["rv_endo"]
    rv_faces = reference_meshes["rv_endo"].faces
    rv_merge_labels, n_unique_rv = compute_merge_map(rv_verts, threshold_percent=0.01)
    new_rv_verts, canonical_faces = apply_merge_map(rv_verts, rv_faces, rv_merge_labels, n_unique_rv)
    tmp = trimesh.Trimesh(vertices=np.zeros((n_unique_rv, 3)), faces=canonical_faces, process=False)
    trimesh.repair.fix_normals(tmp)
    per_struct["rv_endo"] = new_rv_verts
    reference_meshes["rv_endo"] = trimesh.Trimesh(
        vertices=new_rv_verts, faces=tmp.faces, process=False
    )

    # Build combined mesh (concatenated structures, face indices offset per structure).
    verts_list, faces_list, offset = [], [], 0
    for s in STRUCTURES:
        v = np.asarray(reference_meshes[s].vertices, dtype=np.float64)
        f = np.asarray(reference_meshes[s].faces, dtype=np.int64)
        verts_list.append(per_struct[s])
        faces_list.append(f + offset)
        offset += v.shape[0]

    combined_verts = np.vstack(verts_list)
    combined_faces = np.vstack(faces_list)

    # Deduplicate vertices at valve boundaries.
    combined_merge_labels, n_unique_combined = compute_merge_map(combined_verts, threshold_percent=0.01)
    merged_verts, merged_faces = apply_merge_map(
        combined_verts, combined_faces, combined_merge_labels, n_unique_combined
    )
    merged_mesh = trimesh.Trimesh(
        vertices=merged_verts.astype(np.float64),
        faces=merged_faces.astype(np.int64),
        process=False,
    )
    trimesh.repair.fix_normals(merged_mesh)

    removed = combined_verts.shape[0] - merged_verts.shape[0]
    log.info(
        "Merged template: %d → %d vertices (%d duplicates removed).",
        combined_verts.shape[0], merged_verts.shape[0], removed,
    )

    template_path = output_dir / "template_mesh.vtk"
    save_trimesh_as_vtk(merged_mesh, template_path)
    log.info("Saved template mesh to %s.", template_path)

    # Compose the two repair maps into a single map from the original unrepaired
    # combined vertices (shape: n_epi + n_lv + n_rv_orig) to the final merged
    # vertices. This lets later callers apply one operation instead of two.
    n_non_rv = reference_splits[0] + reference_splits[1]
    n_rv_orig = reference_splits[2]
    intermediate = np.empty(n_non_rv + n_rv_orig, dtype=np.int64)
    intermediate[:n_non_rv] = np.arange(n_non_rv, dtype=np.int64)
    intermediate[n_non_rv:] = n_non_rv + rv_merge_labels
    merge_labels = combined_merge_labels[intermediate]

    repair_path = output_dir / "mesh_repair.npz"
    np.savez_compressed(
        repair_path,
        merge_labels=merge_labels,
        n_unique=n_unique_combined,
        structure_splits=np.array(reference_splits, dtype=np.int64),
    )
    log.info("Saved merge map to %s.", repair_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a mean template mesh from training cases."
    )
    parser.add_argument("--bivme-output-dir", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=1000)
    args = parser.parse_args()

    output_dir = args.output_dir or args.cohort_file.parent
    create_mean_template(
        bivme_output_dir=args.bivme_output_dir,
        cohort_file=args.cohort_file,
        output_dir=output_dir,
        max_cases=args.max_cases,
    )
