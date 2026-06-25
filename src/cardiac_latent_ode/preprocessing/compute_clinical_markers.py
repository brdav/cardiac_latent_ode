#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cardiac_latent_ode.utils.mesh_processing import extract_clinical_markers
from cardiac_latent_ode.utils.pylogger import RichLogger

log = RichLogger(__name__)

MARKER_COLUMNS = ["LVEF", "RVEF", "LV-GLS", "RV-FWLS", "LVMi", "LVWT", "RWT"]


def _decode_case_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _iter_batches(start: int, end: int, batch_size: int):
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        yield batch_start, batch_end


def _compute_records_for_index_range(
    h5_data_path: str,
    start: int,
    end: int,
    batch_size: int,
    myocardial_density_g_per_ml: float,
    show_progress: bool = False,
    progress_desc: str = "Computing markers",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with h5py.File(h5_data_path, "r") as h5f:
        pos_ds = h5f["pos"]
        bsa_ds = h5f["bsa"]
        case_id_ds = h5f["case_id"]

        iterator = _iter_batches(start, end, batch_size)
        if show_progress:
            total_batches = math.ceil((end - start) / batch_size)
            iterator = tqdm(
                iterator,
                total=total_batches,
                desc=progress_desc,
                unit="batch",
            )

        for batch_start, batch_end in iterator:
            pos_btv3 = np.asarray(pos_ds[batch_start:batch_end, ...], dtype=np.float32)
            bsa_b = np.asarray(bsa_ds[batch_start:batch_end], dtype=np.float32)
            case_ids = [_decode_case_id(v) for v in case_id_ds[batch_start:batch_end]]

            # HDF5 stores meshes as [B, T, V, 3]; markers expect [B, V, T, 3].
            x_bvtc = torch.from_numpy(pos_btv3).permute(0, 2, 1, 3).contiguous()
            markers = extract_clinical_markers(
                x_bvtc=x_bvtc,
                bsa=bsa_b,
                myocardial_density_g_per_ml=myocardial_density_g_per_ml,
            )

            for local_idx, case_id in enumerate(case_ids):
                rec: dict[str, Any] = {
                    "_row_index": batch_start + local_idx,
                    "case_id": case_id,
                }
                for marker_name in MARKER_COLUMNS:
                    value = float(markers[marker_name][local_idx])
                    rec[marker_name] = value
                records.append(rec)

    return records


def _build_ranges(total_cases: int, num_workers: int) -> list[tuple[int, int]]:
    bounds = np.linspace(0, total_cases, num_workers + 1, dtype=np.int64)
    ranges: list[tuple[int, int]] = []
    for i in range(num_workers):
        start = int(bounds[i])
        end = int(bounds[i + 1])
        if end > start:
            ranges.append((start, end))
    return ranges


def _infer_total_cases(h5_data_path: str) -> int:
    with h5py.File(h5_data_path, "r") as h5f:
        pos_ds = h5f["pos"]
        return pos_ds.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute seven clinical markers for all meshes in an HDF5 file."
    )
    parser.add_argument(
        "--h5-data-path",
        type=str,
        required=True,
        help=("Path to HDF5 mesh file."),
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help=("Output CSV path. If omitted, writes to <h5-dir>/clinical_markers.csv."),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of meshes to process per batch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of worker processes. 0/1 disables multiprocessing; "
            "use >1 to parallelize over index ranges."
        ),
    )
    parser.add_argument(
        "--myocardial-density",
        type=float,
        default=1.05,
        help="Myocardial density in g/mL for LVMi.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    h5_data_path = Path(args.h5_data_path)
    if not h5_data_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_data_path}")

    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be > 0, got {args.batch_size}")

    total_cases = _infer_total_cases(str(h5_data_path))

    out_csv = (
        Path(args.out_csv)
        if args.out_csv is not None
        else h5_data_path.parent / "clinical_markers.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    num_workers = int(args.num_workers)
    if num_workers < 0:
        raise ValueError(f"num-workers must be >= 0, got {num_workers}")
    if num_workers == 0:
        num_workers = 1
    num_workers = min(num_workers, total_cases) if total_cases > 0 else 1

    all_records: list[dict[str, Any]] = []

    if num_workers == 1:
        all_records.extend(
            _compute_records_for_index_range(
                str(h5_data_path),
                0,
                total_cases,
                args.batch_size,
                float(args.myocardial_density),
                show_progress=True,
            )
        )
    else:
        ranges = _build_ranges(total_cases, num_workers)
        max_workers = min(len(ranges), os.cpu_count() or len(ranges))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _compute_records_for_index_range,
                    str(h5_data_path),
                    start,
                    end,
                    int(args.batch_size),
                    float(args.myocardial_density),
                )
                for start, end in ranges
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Computing markers",
                unit="worker",
            ):
                all_records.extend(future.result())

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.sort_values("_row_index", kind="mergesort").reset_index(drop=True)

    front_cols = ["case_id", *MARKER_COLUMNS]
    ordered_cols = [c for c in front_cols if c in df.columns] + [
        c for c in df.columns if c not in front_cols and c != "_row_index"
    ]
    df = df[ordered_cols] if ordered_cols else df

    df.to_csv(out_csv, index=False)
    log.info(
        f"Wrote markers for {len(df)} meshes to {out_csv} "
        f"(batch_size={args.batch_size}, num_workers={num_workers})."
    )


if __name__ == "__main__":
    main()
