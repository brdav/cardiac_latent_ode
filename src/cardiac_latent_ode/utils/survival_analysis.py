from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils.concordance import (
    _concordance_ratio,
    _concordance_summary_statistics,
)
from patsy import build_design_matrices, dmatrix
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Feature-set definitions
# ---------------------------------------------------------------------------

# Covariates: the 15 engineered terms used in the PCP-HF linear predictor.
# Each term mirrors the corresponding coefficient entry in compute_pcp_hf.py.
COVARIATE_COLS = [
    "ln_age",
    "ln_age_sq",
    "ln_bp_treated",
    "ln_age_x_ln_bp_treated",
    "ln_bp_untreated",
    "ln_age_x_ln_bp_untreated",
    "current_smoker",
    "ln_age_x_current_smoker",
    "ln_glucose_treated",
    "ln_glucose_untreated",
    "ln_total_chol",
    "ln_hdl",
    "ln_bmi",
    "ln_age_x_ln_bmi",
    "ln_qrs",
]

# These are collected from Appendix 3 of "2022 AHA/ACC/HFSA Guideline for the
# Management of Heart Failure: A Report of the American College of
# Cardiology/American Heart Association Joint Committee on Clinical Practice
# Guidelines"
# We additionally added two RV metrics (RVEF and RV-FWLS).
MESH_METRIC_COLS = ["LVEF", "LV-GLS", "LVMi", "LVWT", "RWT", "RVEF", "RV-FWLS"]


# ---------------------------------------------------------------------------
# Spline expansion utilities
# ---------------------------------------------------------------------------


def spline_expand_train(df, columns, df_spline=3):
    spline_terms = []
    design_infos = {}

    for col in columns:
        formula = f'cr(Q("{col}"), df={df_spline}) - 1'

        transformed = dmatrix(formula, df, return_type="dataframe")

        design_infos[col] = transformed.design_info

        transformed.columns = [f"{col}_s{i}" for i in range(transformed.shape[1])]

        spline_terms.append(transformed)

    return pd.concat(spline_terms, axis=1), design_infos


def spline_expand_test(df, columns, design_infos):
    spline_terms = []

    for col in columns:
        design_info = design_infos[col]

        transformed = build_design_matrices([design_info], df)[0]

        transformed = pd.DataFrame(
            transformed,
            columns=[f"{col}_s{i}" for i in range(transformed.shape[1])],
            index=df.index,
        )

        spline_terms.append(transformed)

    return pd.concat(spline_terms, axis=1)


# ---------------------------------------------------------------------------
# Feature loaders
# ---------------------------------------------------------------------------


def load_pcp_hf_features(demographics_path: str) -> pd.DataFrame:
    """Load and engineer demographic covariates from demographics.csv.

    Produces the 15 transformed terms that enter the PCP-HF linear predictor
    (see compute_pcp_hf._ind_x).  Each column in the returned DataFrame
    corresponds directly to one entry in COVARIATE_COLS:

        ln_age, ln_age_sq,
        ln_bp_treated,            ln_age_x_ln_bp_treated,
        ln_bp_untreated,          ln_age_x_ln_bp_untreated,
        current_smoker,           ln_age_x_current_smoker,
        ln_glucose_treated,       ln_glucose_untreated,
        ln_total_chol, ln_hdl, ln_bmi, ln_age_x_ln_bmi, ln_qrs

    Unit conversions match compute_pcp_hf:
        casual_glucose      × 18.0  → mg/dL
        total_cholesterol   × 38.67 → mg/dL
        hdl_cholesterol     × 38.67 → mg/dL

    Returns a DataFrame with columns: case_id, <COVARIATE_COLS>
    """
    df = pd.read_csv(demographics_path)

    # --- Raw inputs --------------------------------------------------------
    age = pd.to_numeric(df["age"], errors="coerce")
    systolic_bp = pd.to_numeric(df["systolic_bp"], errors="coerce")
    bmi = pd.to_numeric(df["body_mass_index"], errors="coerce")
    glucose = pd.to_numeric(df["casual_glucose"], errors="coerce") * 18.0  # → mg/dL
    total_chol = (
        pd.to_numeric(df["total_cholesterol"], errors="coerce") * 38.67
    )  # → mg/dL
    hdl = pd.to_numeric(df["hdl_cholesterol"], errors="coerce") * 38.67  # → mg/dL
    qrs = pd.to_numeric(df["qrs_duration"], errors="coerce")
    smoker = pd.to_numeric(df["current_smoker"], errors="coerce")
    on_htn = pd.to_numeric(df["hypertension_treatment"], errors="coerce")
    on_dm = pd.to_numeric(df["diabetes_treatment"], errors="coerce")

    off_htn = 1.0 - on_htn
    off_dm = 1.0 - on_dm

    # --- Natural log transforms --------------------------------------------
    ln_age = np.log(age)
    ln_bp = np.log(systolic_bp)
    ln_gluc = np.log(glucose)
    ln_tc = np.log(total_chol)
    ln_hdl = np.log(hdl)
    ln_bmi = np.log(bmi)
    ln_qrs = np.log(qrs)

    # --- Assemble feature DataFrame ----------------------------------------
    out = df[["case_id"]].copy()
    out["ln_age"] = ln_age
    out["ln_age_sq"] = ln_age**2
    out["ln_bp_treated"] = ln_bp * on_htn
    out["ln_age_x_ln_bp_treated"] = ln_age * ln_bp * on_htn
    out["ln_bp_untreated"] = ln_bp * off_htn
    out["ln_age_x_ln_bp_untreated"] = ln_age * ln_bp * off_htn
    out["current_smoker"] = smoker
    out["ln_age_x_current_smoker"] = ln_age * smoker
    out["ln_glucose_treated"] = ln_gluc * on_dm
    out["ln_glucose_untreated"] = ln_gluc * off_dm
    out["ln_total_chol"] = ln_tc
    out["ln_hdl"] = ln_hdl
    out["ln_bmi"] = ln_bmi
    out["ln_age_x_ln_bmi"] = ln_age * ln_bmi
    out["ln_qrs"] = ln_qrs
    # sex as a stratification column (binary: 1=male, 0=female); not in COVARIATE_COLS
    s = df["sex"].astype(str).str.lower().str.strip()
    out["sex"] = s.map({"male": 1, "m": 1, "female": 0, "f": 0}).astype(float)
    return out


def load_auxiliary_features(demographics_path: str) -> pd.DataFrame:
    """Load sex, age, and bsa from demographics.csv"""
    df = pd.read_csv(demographics_path)
    out = df[["case_id"]].copy()
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["bsa"] = pd.to_numeric(df["bsa"], errors="coerce")
    s = df["sex"].astype(str).str.lower().str.strip()
    out["sex"] = s.map({"male": 1, "m": 1, "female": 0, "f": 0}).astype(float)
    return out


def load_cmr_marker_features(mesh_metrics_path: str) -> pd.DataFrame:
    """Load pre-computed clinical image metrics from mesh_metrics.csv.

    Returns a DataFrame with columns: case_id, LVEF, LVEDVi, LVMi, LV-GLS, LV-GCS, RVEF, RV-FWLS.
    """
    df = pd.read_csv(mesh_metrics_path)
    missing = [c for c in MESH_METRIC_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{mesh_metrics_path!r} is missing columns: {missing}")
    for col in MESH_METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["case_id"] + MESH_METRIC_COLS].copy()


def load_latent_features(path: str) -> pd.DataFrame:
    npz = np.load(path, allow_pickle=True)

    Z = np.asarray(npz["z"])
    case_ids = npz["case_id"].astype(int)

    df = pd.DataFrame(Z, columns=[f"x{i}" for i in range(Z.shape[1])])
    df.insert(0, "case_id", case_ids)
    return df


# ---------------------------------------------------------------------------
# Outcomes / survival targets
# ---------------------------------------------------------------------------


def prepare_survival_targets(
    df: pd.DataFrame,
    endpoint: str,
) -> pd.DataFrame:
    """Attach 'time' (days) and 'event' (0/1) columns to df."""
    b = pd.to_datetime(df["baseline_date"])
    ev_date = pd.to_datetime(df[endpoint])

    # Censoring: use all_cause_death date where available, else administrative
    # censor date of 2025-08-01 (end of UK Biobank follow-up window).
    _ADMIN_CENSOR = pd.Timestamp("2025-08-01")
    censor_col = "all_cause_death"
    censor = pd.to_datetime(df[censor_col]).fillna(_ADMIN_CENSOR)

    incident = (~b.isna()) & (~ev_date.isna()) & (ev_date > b)
    event = incident.astype(int)
    time_event = (ev_date - b).dt.total_seconds() / 86_400.0
    time_censor = (censor - b).dt.total_seconds() / 86_400.0
    time = time_event.where(event == 1, time_censor)

    out = df.copy()
    out["time"] = pd.to_numeric(time, errors="coerce")
    out["event"] = event.fillna(0).astype(int)
    return out.loc[out["time"].notna() & (out["time"] >= 0)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _expected_observed_at_time(
    event_prob: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    target_time: float,
) -> tuple[float | None, float | None]:
    """Compute expected and observed events at a specific time."""
    event_at_time = (event == 1) & (time <= target_time)
    known_mask = event_at_time | (time > target_time)
    if int(known_mask.sum()) == 0:
        return None, None

    # E/O for survival data should account for censoring at the target horizon.
    # Use expected events from all baseline predictions and observed events from
    # the Kaplan-Meier event probability at target_time.
    expected = float(np.sum(np.clip(event_prob.astype(np.float64), 1e-8, 1 - 1e-8)))

    durations = np.asarray(pd.to_numeric(time, errors="coerce"), dtype=np.float64)
    events = np.asarray(pd.to_numeric(event, errors="coerce"), dtype=np.float64)
    events = np.nan_to_num(events, nan=0.0).astype(int)

    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=durations,
        event_observed=events,
    )
    surv_pred = kmf.predict(target_time)
    surv_t = (
        float(surv_pred.iloc[0])
        if isinstance(surv_pred, pd.Series)
        else float(surv_pred)
    )
    obs_prob_t = float(1.0 - np.clip(surv_t, 0.0, 1.0))
    observed = obs_prob_t * float(len(event_prob))

    return expected, observed


def expected_observed_ratio_at_time_stratified(
    event_prob: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    strata: np.ndarray,
    target_time: float,
) -> float | None:
    """Compute expected/observed ratio within strata and aggregate."""
    strata_series = pd.Series(strata)
    stratum_values = list(strata_series.dropna().unique())

    if not stratum_values:
        return None

    total_expected = 0.0
    total_observed = 0.0

    for stratum in stratum_values:
        mask = (strata_series == stratum).to_numpy()
        n_s = int(mask.sum())
        if n_s == 0:
            continue

        expected_s, observed_s = _expected_observed_at_time(
            event_prob[mask],
            time[mask],
            event[mask],
            target_time,
        )
        if expected_s is not None:
            total_expected += float(expected_s)
        if observed_s is not None:
            total_observed += float(observed_s)

    eo_ratio = None if total_observed <= 0 else float(total_expected / total_observed)
    return eo_ratio


def compute_auroc_at_time_stratified(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray,
    target_time: float,
) -> float | None:
    """Compute stratified AUROC at a specific time point using pair-weighted aggregation.

    AUROC is computed independently per stratum and then aggregated by pooling
    the pair-wise comparisons across strata.
    """
    total_correct = 0
    total_pairs = 0

    strata_series = pd.Series(strata)
    for stratum in strata_series.dropna().unique():
        mask = strata_series == stratum
        if int(mask.sum()) <= 1:
            continue

        time_s = time[mask]
        event_s = event[mask]
        risk_s = risk[mask]

        event_at_time = (event_s == 1) & (time_s <= target_time)
        no_event_by_time = (event_s == 0) | (time_s > target_time)

        if int(event_at_time.sum()) == 0 or int(no_event_by_time.sum()) == 0:
            continue

        # Count pair-wise comparisons within stratum.
        n_events = int(event_at_time.sum())
        n_non_events = int(no_event_by_time.sum())
        n_pairs = n_events * n_non_events

        # For AUROC, ties contribute 0.5 by convention.
        event_risks = risk_s[event_at_time]
        non_event_risks = risk_s[no_event_by_time]

        diffs = event_risks[:, np.newaxis] - non_event_risks[np.newaxis, :]
        concordant = (diffs > 0).sum()
        ties = (diffs == 0).sum()
        total_correct += concordant + 0.5 * ties
        total_pairs += n_pairs

    if total_pairs == 0:
        return None

    return float(total_correct / total_pairs)


def concordance_stratified(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    strata: np.ndarray,
) -> float | None:
    """Compute stratified C-index.

    Concordance summary statistics are computed independently per stratum and
    then aggregated before taking the concordance ratio.
    """
    total_correct, total_tied, total_pairs = 0, 0, 0

    strata_series = pd.Series(strata)
    for stratum in strata_series.dropna().unique():
        mask = strata_series == stratum
        if int(mask.sum()) <= 1:
            continue
        n_correct, n_tied, n_pairs = _concordance_summary_statistics(
            time[mask], -risk[mask], event[mask]
        )
        total_correct += n_correct
        total_tied += n_tied
        total_pairs += n_pairs

    if total_pairs == 0:
        return None
    return float(
        _concordance_ratio(
            int(total_correct),
            int(total_tied),
            int(total_pairs),
        )
    )


def tune_cox_penalizer(
    X_tr: np.ndarray,
    t_tr: np.ndarray,
    e_tr: np.ndarray,
    feature_cols: list[str],
    penalizer_grid: list[float],
    n_splits: int,
    seed: int,
    strata_arr: np.ndarray,
) -> float:
    """Select the best stratified Cox penalizer via inner stratified K-fold CV.

    Returns the penalizer with the highest mean inner-fold C-index.
    """
    if len(penalizer_grid) == 1:
        return penalizer_grid[0]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    inner_folds = list(skf.split(X_tr, e_tr))
    scores = {p: [] for p in penalizer_grid}

    for p in penalizer_grid:
        converged = True
        for inner_tr_idx, inner_val_idx in inner_folds:
            X_itr, X_ival = X_tr[inner_tr_idx], X_tr[inner_val_idx]
            t_itr, e_itr = t_tr[inner_tr_idx], e_tr[inner_tr_idx]
            t_ival, e_ival = t_tr[inner_val_idx], e_tr[inner_val_idx]

            df_inner = pd.DataFrame(X_itr, columns=feature_cols)
            df_inner["time"] = t_itr
            df_inner["event"] = e_itr
            df_inner["_strata"] = strata_arr[inner_tr_idx]

            pred_df = pd.DataFrame(X_ival, columns=feature_cols)
            pred_df["_strata"] = strata_arr[inner_val_idx]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    cph = CoxPHFitter(penalizer=p)
                    cph.fit(
                        df_inner,
                        duration_col="time",
                        event_col="event",
                        strata="_strata",
                    )
            except ConvergenceWarning:
                converged = False
                break

            risk_val = cph.predict_partial_hazard(pred_df).to_numpy(np.float64)
            ci = concordance_stratified(
                t_ival, e_ival, risk_val, strata_arr[inner_val_idx]
            )
            if ci is not None:
                scores[p].append(ci)

        if not converged:
            scores[p] = []  # discard this penalizer

    mean_scores = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
    best = max(mean_scores, key=mean_scores.__getitem__)
    return best
