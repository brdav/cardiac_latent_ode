import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CLINICAL_COLS = ["lv_ef", "lv_edv", "lv_esv", "rv_ef", "rv_edv", "rv_esv"]

DEFAULT_CLINICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "lv_ef": (10.0, 90.0),
    "rv_ef": (10.0, 85.0),
    "lv_edv": (10.0, 600.0),
    "lv_esv": (0.0, 600.0),
    "rv_edv": (10.0, 600.0),
    "rv_esv": (0.0, 600.0),
}


def _compute_summary_stats(volume_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-subject EDV, ESV, EF from per-frame volumes.

    Args:
        volume_df: DataFrame with columns 'name', 'lv_vol', 'rv_vol' (one row per frame).

    Returns:
        One row per subject with columns: case_id, lv_edv, lv_esv, lv_ef, rv_edv, rv_esv, rv_ef.
    """
    for col in ("name", "lv_vol", "rv_vol"):
        if col not in volume_df.columns:
            raise KeyError(f"Missing required column '{col}' in volume file.")

    grp = volume_df.groupby("name")
    stats = pd.DataFrame({
        "lv_edv": grp["lv_vol"].max(),
        "lv_esv": grp["lv_vol"].min(),
        "rv_edv": grp["rv_vol"].max(),
        "rv_esv": grp["rv_vol"].min(),
    }).reset_index().rename(columns={"name": "case_id"})
    stats["lv_ef"] = 100.0 * (stats["lv_edv"] - stats["lv_esv"]) / stats["lv_edv"]
    stats["rv_ef"] = 100.0 * (stats["rv_edv"] - stats["rv_esv"]) / stats["rv_edv"]
    return stats


def _clinical_qc(
    summary_df: pd.DataFrame,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
) -> pd.DataFrame:
    """Keep only rows with physiologically plausible LV/RV volumes and EF."""
    missing = [c for c in CLINICAL_COLS if c not in summary_df.columns]
    if missing:
        raise KeyError(f"Missing clinical columns: {', '.join(missing)}")

    df = summary_df.copy()
    bounds = DEFAULT_CLINICAL_BOUNDS if bounds is None else bounds

    mask = df[CLINICAL_COLS].notna().all(axis=1)
    mask &= df["lv_edv"] > df["lv_esv"]
    mask &= df["rv_edv"] > df["rv_esv"]

    for col in CLINICAL_COLS:
        lower, upper = bounds[col]
        if lower is not None:
            mask &= df[col] >= lower
        if upper is not None:
            mask &= df[col] <= upper

    return df.loc[mask].reset_index(drop=True)


def define_cohort(
    bivme_fitted_models_dir: Path,
    volume_file: Path,
    outcomes_file: Path,
) -> tuple[list[str], pd.DataFrame]:
    """Define the analysis cohort.

    Subjects must have a fitted model directory, pass clinical QC, and appear
    in the outcomes file.

    Returns:
        (cohort_ids, summary_df): list of case IDs and a DataFrame with one
        row per case containing clinical summary stats.
    """
    cases = {p.name for p in bivme_fitted_models_dir.iterdir() if p.is_dir()}
    log.info("Found %d fitted model directories.", len(cases))

    volume_df = pd.read_csv(volume_file)
    summary_df = _compute_summary_stats(volume_df)
    summary_df = summary_df[summary_df["case_id"].isin(cases)].reset_index(drop=True)
    log.info("%d cases have both fitted models and volume data.", len(summary_df))

    before = len(summary_df)
    summary_df = _clinical_qc(summary_df)
    log.info("Clinical QC: removed %d; %d remain.", before - len(summary_df), len(summary_df))

    outcomes = pd.read_csv(outcomes_file)
    if "case_id" not in outcomes.columns:
        raise KeyError("Expected a 'case_id' column in the outcomes CSV.")
    outcome_ids = set(outcomes["case_id"].astype(str))
    before = len(summary_df)
    summary_df = summary_df[summary_df["case_id"].isin(outcome_ids)].reset_index(drop=True)
    log.info("Outcomes intersection: removed %d; %d remain.", before - len(summary_df), len(summary_df))

    return summary_df["case_id"].tolist(), summary_df


def define_split(
    cohort_ids: list[str],
    outcomes_file: Path,
    seed: int = 0,
    test_frac: float = 0.5,
) -> tuple[list[str], list[str], list[str]]:
    """Create a stratified train/test split by incident HF status.

    Args:
        cohort_ids: Case IDs to split.
        outcomes_file: CSV with 'case_id' and 'incident_hf' columns (numeric days to
            event, or NaN for controls).
        seed: Random seed for reproducibility.
        test_frac: Fraction of subjects to assign to the test set.

    Returns:
        (train_ids, test_ids, hf_ids): split lists plus cohort-intersected HF cases.
    """
    if not (0.0 < test_frac < 1.0):
        raise ValueError("test_frac must be between 0 and 1 (exclusive).")

    outcomes = pd.read_csv(outcomes_file)
    if "incident_hf" not in outcomes.columns:
        raise KeyError("Expected an 'incident_hf' column in the outcomes CSV.")

    hf_ids = set(outcomes.loc[outcomes["incident_hf"].notna(), "case_id"].astype(str))

    cohort_unique = sorted(set(cohort_ids))
    y = [1 if c in hf_ids else 0 for c in cohort_unique]
    hf_count = sum(y)
    log.info(
        "Cohort: %d total, %d incident HF, %d controls.",
        len(cohort_unique), hf_count, len(cohort_unique) - hf_count,
    )

    if len(set(y)) < 2:
        raise ValueError("Cannot stratify split: only one class present in cohort.")

    train_ids, test_ids = train_test_split(
        cohort_unique,
        test_size=test_frac,
        random_state=seed,
        stratify=y,
    )
    return list(train_ids), list(test_ids), list(hf_ids & set(cohort_unique))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Define the cardiac imaging cohort and train/test split."
    )
    parser.add_argument("--bivme-fitted-models-dir", type=Path, required=True)
    parser.add_argument("--volume-file", type=Path, required=True)
    parser.add_argument("--outcomes-file", type=Path, required=True)
    parser.add_argument("--cohort-file", type=Path, default=None)
    args = parser.parse_args()

    cohort_ids, summary_df = define_cohort(
        bivme_fitted_models_dir=args.bivme_fitted_models_dir,
        volume_file=args.volume_file,
        outcomes_file=args.outcomes_file,
    )

    train_ids, test_ids, hf_ids = define_split(cohort_ids, args.outcomes_file)

    cohort_file = args.cohort_file or args.outcomes_file.parent / "cohort.csv"
    cohort_file.parent.mkdir(parents=True, exist_ok=True)

    split_map = {case_id: "train" for case_id in train_ids}
    split_map.update({case_id: "test" for case_id in test_ids})
    summary_df["split"] = summary_df["case_id"].map(split_map)

    if summary_df["split"].isna().any():
        n = int(summary_df["split"].isna().sum())
        raise RuntimeError(f"Could not assign split for {n} rows (case_id mismatch).")

    summary_df.to_csv(cohort_file, index=False)
    log.info("Saved cohort to: %s", cohort_file)
