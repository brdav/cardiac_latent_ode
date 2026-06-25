from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pydicom as dicom
import trimesh
from tqdm import tqdm

from cardiac_latent_ode.utils.utils import apply_merge_map, load_vtk_as_trimesh

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

STRUCTURES = ["epi", "lv_endo", "rv_endo"]
TYPE_MAP = {
    "epi": "EPICARDIAL",
    "lv_endo": "LV_ENDOCARDIAL",
    "rv_endo": "RV_ENDOCARDIAL",
}


def _load_frame_verts(case_dir: Path, case_id: str, frame: int) -> np.ndarray | None:
    """Load and concatenate [epi | lv_endo | rv_endo] vertices for one frame.

    Returns (N_orig, 3) float64, or None if any structure file is missing.
    """
    verts_list = []
    for s in STRUCTURES:
        path = case_dir / "vtk_smoothed" / f"{case_id}_{TYPE_MAP[s]}_{frame:03d}.vtk"
        if not path.exists():
            log.warning("Missing %s frame %03d for %s.", s, frame, case_id)
            return None
        mesh = load_vtk_as_trimesh(path)
        verts_list.append(np.asarray(mesh.vertices, dtype=np.float64))
    return np.vstack(verts_list)


def _count_frames(case_dir: Path, case_id: str) -> int:
    return len(
        sorted((case_dir / "vtk_smoothed").glob(f"{case_id}_{TYPE_MAP['epi']}_*.vtk"))
    )


def _merge_verts(
    verts: np.ndarray, merge_labels: np.ndarray, n_unique: int
) -> np.ndarray:
    """Apply vertex merge map: (N_orig, 3) → (V, 3).

    Thin wrapper around apply_merge_map; faces are not needed here.
    """
    merged, _ = apply_merge_map(
        verts, np.empty((0, 3), dtype=np.int64), merge_labels, n_unique
    )
    return merged


def _get_heart_rate(dicom_dir: Path, slice_info_dir: Path, case_id: str) -> float:
    slice_info_path = slice_info_dir / "SliceInfoFile.txt"
    with open(slice_info_path) as f:
        dicom_names = [line.split("\t")[0] for line in f if line.strip()]

    series = None
    for dicom_n in dicom_names:
        try:
            img = dicom.dcmread(dicom_dir / dicom_n)
            series = img.SeriesInstanceUID
            break
        except:
            pass
    if series is None:
        log.info(f"Could not find series for case {case_id}, trying fallback.")

        # Go for best guess: Load last present series, and read HR from there
        files = sorted(
            [p for p in dicom_dir.iterdir() if p.suffix == ".dcm"], reverse=True
        )
        try:
            img = dicom.dcmread(files[0])
            series = img.SeriesInstanceUID
        except:
            log.info(f"Fallback failed for case {case_id}")
            return np.nan

    files_time = []
    for f in dicom_dir.iterdir():
        if f.suffix != ".dcm":
            continue
        d = dicom.dcmread(f)
        s = d.SeriesInstanceUID
        if s == series:
            t = d.TriggerTime
            files_time += [[f.name, t]]

    if len(files_time) != 50:
        log.info(f"case {case_id} has {len(files_time)} files")
        return np.nan

    files_time = sorted(files_time, key=lambda x: x[1])
    files_time_t = np.array([ft[1] for ft in files_time])
    heart_rate = 60 / (
        np.median(files_time_t[1:] - files_time_t[:-1]) * 1e-3 * 50
    )  # in BPM
    return heart_rate


def process_case(
    case_dir: Path,
    case_id: str,
    n_frames: int,
    template_verts: np.ndarray,
    merge_labels: np.ndarray,
    n_unique: int,
) -> np.ndarray | None:
    """Process all frames for one case: apply merge map, align to template.

    The rigid alignment transform is estimated from frame 0 (Procrustes to
    template) and applied uniformly to all frames to preserve temporal
    consistency.

    Returns (T, V, 3) float32, or None if any frame is missing.
    """
    merged_frames = []
    for t in range(n_frames):
        raw = _load_frame_verts(case_dir, case_id, t)
        if raw is None:
            return None
        merged_frames.append(_merge_verts(raw, merge_labels, n_unique))

    matrix, _, _ = trimesh.registration.procrustes(
        merged_frames[0],
        template_verts,
        reflection=False,
        translation=True,
        scale=False,
        return_cost=True,
    )

    def _apply(verts: np.ndarray) -> np.ndarray:
        h = np.column_stack([verts, np.ones(len(verts))])
        return (matrix @ h.T).T[:, :3]

    aligned = np.stack([_apply(f) for f in merged_frames], axis=0)
    return aligned.astype(np.float32)


def process_meshes(
    bivme_output_dir: Path,
    bulk_dir: Path,
    cohort_file: Path,
    demographics_file: Path,
    template_file: Path,
    mesh_repair_file: Path,
    output_file: Path,
) -> None:
    """Process all cohort meshes and write to HDF5.

    Output datasets:
        pos:     (N, T, V, 3) float32 — aligned merged vertex positions
        case_id: (N,)         str
        sex:     (N,)         float32
        age:     (N,)         float32
        bsa:     (N,)         float32
    """
    cohort = pd.read_csv(cohort_file)
    demo_df = pd.read_csv(demographics_file).set_index("case_id")
    demo_df.index = demo_df.index.astype(str)

    repair = np.load(mesh_repair_file)
    merge_labels = repair["merge_labels"]
    n_unique = int(repair["n_unique"])

    template_verts = np.asarray(
        load_vtk_as_trimesh(template_file).vertices, dtype=np.float64
    )
    if template_verts.shape[0] != n_unique:
        raise ValueError(
            f"Template has {template_verts.shape[0]} vertices but mesh_repair expects {n_unique}."
        )

    case_ids = cohort["case_id"].astype(str).tolist()

    n_frames = next(
        (
            _count_frames(bivme_output_dir / "fitted-models" / case_id, case_id)
            for case_id in case_ids
            if (bivme_output_dir / "fitted-models" / case_id).is_dir()
        ),
        None,
    )
    if not n_frames:
        raise RuntimeError("Could not detect frame count from any cohort case.")
    log.info("Detected T=%d frames per case.", n_frames)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, "w") as h5f:
        pos_ds = case_id_ds = sex_ds = age_ds = bsa_ds = None
        saved = 0

        for case_id in tqdm(case_ids, desc="Processing cases"):
            case_dir = bivme_output_dir / "fitted-models" / case_id
            raw_case_dir = bulk_dir / case_id / "lax"
            if not case_dir.is_dir():
                log.warning("No directory for %s; skipping.", case_id)
                continue

            if case_id not in demo_df.index:
                log.warning("No demographics for %s; skipping.", case_id)
                continue

            pos = process_case(
                case_dir, case_id, n_frames, template_verts, merge_labels, n_unique
            )
            if pos is None:
                continue

            heart_rate = _get_heart_rate(
                raw_case_dir, bivme_output_dir / "guidepoints" / case_id, case_id
            )

            # Initialise resizable datasets on first valid case.
            if pos_ds is None:
                T, V, _ = pos.shape
                pos_ds = h5f.create_dataset(
                    "pos",
                    shape=(0, T, V, 3),
                    maxshape=(None, T, V, 3),
                    dtype=np.float32,
                    chunks=(1, T, V, 3),
                )
                case_id_ds = h5f.create_dataset(
                    "case_id",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(),
                    chunks=True,
                )
                sex_ds = h5f.create_dataset(
                    "sex", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True
                )
                age_ds = h5f.create_dataset(
                    "age", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True
                )
                bsa_ds = h5f.create_dataset(
                    "bsa", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True
                )
                heart_rate_ds = h5f.create_dataset(
                    "heart_rate",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float32,
                    chunks=True,
                )

            demo = demo_df.loc[case_id]
            pos_ds.resize((saved + 1, *pos_ds.shape[1:]))
            pos_ds[saved] = pos
            case_id_ds.resize((saved + 1,))
            case_id_ds[saved] = case_id
            sex_ds.resize((saved + 1,))
            _sex = demo["sex"]
            if isinstance(_sex, str):
                _sex = {"male": 0, "female": 1}[_sex.lower()]
            sex_ds[saved] = float(_sex)
            age_ds.resize((saved + 1,))
            age_ds[saved] = float(demo["age"])
            bsa_ds.resize((saved + 1,))
            bsa_ds[saved] = float(demo["bsa"])
            heart_rate_ds.resize((saved + 1,))
            heart_rate_ds[saved] = float(heart_rate)
            saved += 1

        h5f.attrs["n_cases"] = saved
        h5f.attrs["n_frames"] = n_frames
        h5f.attrs["n_vertices"] = n_unique

    log.info("Saved %d cases to %s.", saved, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process cardiac meshes into HDF5 dataset."
    )
    parser.add_argument("--cohort-file", type=Path, required=True)
    parser.add_argument("--bivme-output-dir", type=Path, required=True)
    parser.add_argument("--bulk-dir", type=Path, required=True)
    parser.add_argument("--demographics-file", type=Path, default=None)
    parser.add_argument("--template-file", type=Path, default=None)
    parser.add_argument("--mesh-repair-file", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    args = parser.parse_args()

    data_dir = args.cohort_file.parent
    process_meshes(
        bivme_output_dir=args.bivme_output_dir,
        bulk_dir=args.bulk_dir,
        cohort_file=args.cohort_file,
        demographics_file=args.demographics_file or data_dir / "demographics.csv",
        template_file=args.template_file or data_dir / "template_mesh.vtk",
        mesh_repair_file=args.mesh_repair_file or data_dir / "mesh_repair.npz",
        output_file=args.output_file or data_dir / "meshes.h5",
    )
