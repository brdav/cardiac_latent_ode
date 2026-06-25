from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from cardiac_latent_ode.utils.pylogger import RichLogger
from cardiac_latent_ode.utils.utils import load_vtk_as_trimesh, save_trimesh_as_vtk

log = RichLogger(__name__)


def load_control_mesh_sequence(
    case_folder: Path,
    case_name: str,
    structure: str,
    target_frame_count: int = 50,
    max_gap_size: int = 3,
    outlier_threshold: float = 3.0,
) -> tuple[np.ndarray | None, list[int]]:
    """Load a sequence of meshes from saved model .vtk files.

    Missing frames (<= max_gap_size) are interpolated via CubicSpline.
    Returns (None, []) if gaps are too large or multiple outliers are detected.
    """
    do_interpolation = False
    requested_frames: np.ndarray | None = None

    detected_frames = []
    for f in (case_folder / "vtk").glob(f"{case_name}_{structure}_*.vtk"):
        try:
            if not f.is_file() or f.stat().st_size == 0:
                log.debug("Skipping empty or non-file: %s", f)
                continue
        except OSError:
            log.debug("Could not stat file, skipping: %s", f)
            continue
        m = re.search(rf"{case_name}_{structure}_(\d+)\.vtk", f.name)
        if m:
            detected_frames.append(int(m.group(1)))
    detected_frames = sorted(set(detected_frames))

    if not detected_frames:
        raise ValueError(f"No valid {structure} mesh frames found in {case_folder}/vtk")

    missing_frames = [
        i for i in range(target_frame_count) if i not in set(detected_frames)
    ]
    if missing_frames:
        extended = detected_frames + [detected_frames[0] + target_frame_count]
        max_gap_found = int(np.max(np.diff(extended))) - 1
        if max_gap_found > max_gap_size:
            log.warning(
                "Gap of size %d exceeds max %d. Skipping.",
                max_gap_found,
                max_gap_size,
            )
            return None, []
        log.info("Found %d missing frames. Will interpolate.", len(missing_frames))
        do_interpolation = True

    frame_numbers = detected_frames
    requested_frames = np.arange(target_frame_count)

    loaded_data: dict[int, np.ndarray] = {}
    loaded_frames: list[int] = []
    for frame_num in frame_numbers:
        model_file = case_folder / "vtk" / f"{case_name}_{structure}_{frame_num:03}.vtk"
        loaded_data[frame_num] = load_vtk_as_trimesh(model_file).vertices
        loaded_frames.append(frame_num)

    # Outlier detection via modified Z-score (MAD)
    meshes_array = np.array([loaded_data[f] for f in loaded_frames])
    median_mesh = np.median(meshes_array, axis=0)
    deviations = np.mean(np.linalg.norm(meshes_array - median_mesh, axis=2), axis=1)
    median_dev = np.median(deviations)
    mad_dev = np.median(np.abs(deviations - median_dev))

    if mad_dev < 1e-9:
        log.warning("MAD of deviations is zero; cannot detect outliers reliably.")
        return None, []

    z_scores = 0.6745 * (deviations - median_dev) / mad_dev
    outliers = np.where(np.abs(z_scores) > outlier_threshold)[0]

    if len(outliers) > 1:
        log.warning(
            "Multiple outlier frames %s (max score %.2f). Skipping.",
            [loaded_frames[i] for i in outliers],
            float(np.max(np.abs(z_scores[outliers]))),
        )
        return None, []
    elif len(outliers) == 1:
        bad_frame = loaded_frames[outliers[0]]
        log.warning(
            "Single outlier frame %d (score %.2f). Interpolating.",
            bad_frame,
            float(np.abs(z_scores[outliers[0]])),
        )
        del loaded_data[bad_frame]
        loaded_frames.pop(outliers[0])
        do_interpolation = True

    if do_interpolation:
        sorted_frames = np.array(sorted(loaded_frames))
        sorted_meshes = np.array([loaded_data[f] for f in sorted_frames])
        frames_ext = np.concatenate(
            [
                sorted_frames - target_frame_count,
                sorted_frames,
                sorted_frames + target_frame_count,
            ]
        )
        meshes_ext = np.concatenate(
            [sorted_meshes, sorted_meshes, sorted_meshes], axis=0
        )
        target_frames = (
            requested_frames
            if requested_frames is not None
            else np.arange(target_frame_count)
        )
        mesh_sequence: np.ndarray = CubicSpline(frames_ext, meshes_ext, axis=0)(
            target_frames
        )
        # Restore loaded frames exactly (no interpolation artefacts)
        if np.array_equal(target_frames, np.arange(target_frame_count)):
            mesh_sequence[sorted_frames] = sorted_meshes
        else:
            frame_to_idx = {int(f): i for i, f in enumerate(target_frames)}
            for f, mesh in loaded_data.items():
                if int(f) in frame_to_idx:
                    mesh_sequence[frame_to_idx[int(f)]] = mesh
        loaded_frames = list(map(int, target_frames))
        log.info("Interpolated to %d frames.", len(loaded_frames))
    else:
        mesh_sequence = np.array([loaded_data[f] for f in loaded_frames])

    log.info(
        "Loaded %d frames with %d control points each.",
        len(loaded_frames),
        mesh_sequence.shape[1],
    )
    return mesh_sequence, loaded_frames


def apply_temporal_smoothing(
    verts: np.ndarray,
    window_size: int = 7,
    poly_order: int = 3,
) -> np.ndarray:
    """Apply Savitzky-Golay temporal smoothing to (n_frames, n_verts, 3) vertex array."""
    if window_size % 2 == 0:
        window_size += 1
    poly_order = min(poly_order, window_size - 1)
    return savgol_filter(verts, window_size, poly_order, axis=0, mode="wrap")


def export_smoothed_vtk_meshes(
    smoothed_verts: np.ndarray,
    frame_indices: list[int],
    case_folder: Path,
    case_name: str,
    structure: str,
) -> None:
    """Save smoothed meshes by replacing vertices in original VTK files."""
    vtk_dir = case_folder / "vtk"
    vtk_smoothed_dir = case_folder / "vtk_smoothed"
    vtk_smoothed_dir.mkdir(parents=True, exist_ok=True)
    for i, frame_idx in enumerate(frame_indices):
        src = vtk_dir / f"{case_name}_{structure}_{frame_idx:03d}.vtk"
        template = load_vtk_as_trimesh(src)
        out_mesh = trimesh.Trimesh(
            vertices=smoothed_verts[i], faces=template.faces, process=False
        )
        save_trimesh_as_vtk(
            out_mesh,
            vtk_smoothed_dir / f"{case_name}_{structure}_{frame_idx:03d}.vtk",
        )


def main(
    bivme_output_dir: str,
    window_size: int = 7,
    poly_order: int = 3,
    outlier_threshold: float = 3.0,
) -> None:
    """Apply temporal smoothing to a mesh sequence and write smoothed VTK files."""
    case_dir = Path(bivme_output_dir) / "fitted-models"

    for case_path in case_dir.iterdir():
        if not case_path.is_dir():
            continue
        case_name = case_path.name

        for structure in ["EPICARDIAL", "LV_ENDOCARDIAL", "RV_ENDOCARDIAL"]:

            log.info("Loading %s mesh sequence for %s...", structure, case_name)
            mesh_sequence, loaded_frames = load_control_mesh_sequence(
                case_path,
                case_name,
                structure=structure,
                outlier_threshold=outlier_threshold,
            )

            if mesh_sequence is None:
                log.warning("Skipping smoothing due to gaps or outliers.")
                return

            smoothed = apply_temporal_smoothing(
                mesh_sequence, window_size=window_size, poly_order=poly_order
            )

            displacement = np.linalg.norm(smoothed - mesh_sequence, axis=2)
            log.info(
                "Mean displacement: %.4f mm, Max: %.4f mm",
                np.mean(displacement),
                np.max(displacement),
            )

            export_smoothed_vtk_meshes(
                smoothed, loaded_frames, case_path, case_name, structure
            )
            log.info("Saved %d smoothed frames to %s.", len(loaded_frames), case_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply temporal smoothing to biventricular meshes"
    )
    parser.add_argument("--bivme-output-dir", type=str, required=True)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--poly-order", type=int, default=3)
    args = parser.parse_args()
    main(
        args.bivme_output_dir,
        window_size=args.window_size,
        poly_order=args.poly_order,
    )
