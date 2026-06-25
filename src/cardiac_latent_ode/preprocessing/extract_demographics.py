import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from cardiac_latent_ode.utils.pylogger import RichLogger

log = RichLogger(__name__)


def _encode_sex(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.lower().str.strip()
    return s.map({"female": 0.0, "f": 0.0, "male": 1.0, "m": 1.0})


def _encode_bool(series: pd.Series) -> pd.Series:
    # Accept True/False/pd.NA, and common string forms.
    if series.dtype == bool:
        return series.astype(float)
    s = series.astype("object")
    s = s.replace(
        {
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        }
    )
    return s.map({True: 1.0, False: 0.0})


def _decode_bool(values: np.ndarray) -> pd.Series:
    # Threshold at 0.5 after clipping.
    v = np.asarray(values, dtype=float)
    v = np.clip(v, 0.0, 1.0)
    return pd.Series(v >= 0.5, dtype=bool)


def _mice_impute(
    df: pd.DataFrame,
    *,
    train_mask: pd.Series,
    random_state: int = 0,
) -> pd.DataFrame:
    """Impute missing demographics using MICE fit on training rows only."""

    # Columns the user asked to impute.
    impute_cols = [
        "bsa",
        "age",
        "systolic_bp",
        "hypertension_treatment",
        "casual_glucose",
        "diabetes_treatment",
        "total_cholesterol",
        "hdl_cholesterol",
        "current_smoker",
        "qrs_duration",
        "body_mass_index",
    ]

    for c in impute_cols:
        if c not in df.columns:
            raise KeyError(
                f"Expected column '{c}' in demographics table for imputation."
            )

    if train_mask.sum() == 0:
        raise ValueError("Training split is empty; cannot fit imputer.")

    work = df.copy()

    # Encode non-numeric columns into numeric for IterativeImputer.
    work["sex_code"] = _encode_sex(work["sex"]).astype(float)
    work["hypertension_treatment_code"] = _encode_bool(
        work["hypertension_treatment"]
    ).astype(float)
    work["diabetes_treatment_code"] = _encode_bool(work["diabetes_treatment"]).astype(
        float
    )
    work["current_smoker_code"] = _encode_bool(work["current_smoker"]).astype(float)

    numeric_cols = [
        "sex_code",
        "body_mass_index",
        "bsa",
        "age",
        "systolic_bp",
        "hypertension_treatment_code",
        "casual_glucose",
        "diabetes_treatment_code",
        "total_cholesterol",
        "hdl_cholesterol",
        "current_smoker_code",
        "qrs_duration",
    ]

    # Ensure floats and keep NaNs.
    X = work[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(float)

    # Fit only on train, transform all.
    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=20,
        skip_complete=True,
    )
    imputer.fit(X.loc[train_mask])
    X_imp = imputer.transform(X)
    X_imp = pd.DataFrame(X_imp, columns=numeric_cols, index=work.index)

    # Decode back to original columns.
    work["age"] = X_imp["age"].round().astype(float)
    work["bsa"] = X_imp["bsa"].astype(float)
    work["systolic_bp"] = X_imp["systolic_bp"].round().astype(float)
    work["casual_glucose"] = X_imp["casual_glucose"].astype(float)
    work["total_cholesterol"] = X_imp["total_cholesterol"].astype(float)
    work["hdl_cholesterol"] = X_imp["hdl_cholesterol"].astype(float)
    work["qrs_duration"] = X_imp["qrs_duration"].round().astype(float)
    work["body_mass_index"] = X_imp["body_mass_index"].astype(float)

    work["hypertension_treatment"] = _decode_bool(
        X_imp["hypertension_treatment_code"].to_numpy()
    )
    work["diabetes_treatment"] = _decode_bool(
        X_imp["diabetes_treatment_code"].to_numpy()
    )
    work["current_smoker"] = _decode_bool(X_imp["current_smoker_code"].to_numpy())

    # Drop helper columns.
    work = work.drop(
        columns=[
            "sex_code",
            "hypertension_treatment_code",
            "diabetes_treatment_code",
            "current_smoker_code",
        ],
        errors="ignore",
    )
    return work


def _compute_bsa(height_cm: float, weight_kg: float) -> float:
    """Compute body surface area (BSA) in m^2 using the Dubois & Dubois formula.

    Returns NaN for missing or non-positive inputs.
    """
    h = float(height_cm) / 100.0  # convert cm to m
    w = float(weight_kg)
    if not (h > 0.0 and w > 0.0):  # catches NaN, zero, and negative values
        return float("nan")
    return 0.20247 * (w**0.425) * (h**0.725)


def _iter_filtered_chunks(
    csv_file_path: str,
    *,
    cohort_eids: set[str],
    usecols: list[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(csv_file_path, usecols=usecols, chunksize=chunksize):
        # Keep IDs stable as strings for matching.
        chunk_eid = chunk["eid"].astype(str)
        filtered = chunk.loc[chunk_eid.isin(cohort_eids)].copy()
        if not filtered.empty:
            filtered["eid"] = filtered["eid"].astype(str)
            yield filtered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a demographics table for subjects present in the mesh cohort."
        )
    )
    parser.add_argument(
        "--cohort-file",
        type=str,
        required=True,
        help="Path to text file listing cohort subject IDs.",
    )
    parser.add_argument(
        "--ukbb-csv",
        type=str,
        default="/cluster/work/grlab/projects/projects2025-dataspectrum4cvd/ukbb/raw/Main/ukb679928.csv",
        help="Path to the UKBB main CSV export.",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default=None,
        help="Where to write the cohort demographics table (CSV).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Chunk size for streaming the UKBB CSV.",
    )

    args = parser.parse_args()

    cohort_df = pd.read_csv(args.cohort_file, dtype=str)
    if "case_id" not in cohort_df.columns:
        raise KeyError("Cohort file must contain an 'case_id' column.")
    if "split" not in cohort_df.columns:
        raise KeyError("Cohort file must contain a 'split' column (train/test).")

    cohort_case_ids = cohort_df["case_id"].astype(str).tolist()
    cohort_eids = {c.split("_")[0] for c in cohort_case_ids}
    if not cohort_eids:
        raise RuntimeError(
            "define_cohort returned an empty cohort; cannot extract demographics."
        )

    # ------------------------------
    # AGE
    # ------------------------------

    fields_age = ["21003-2.0"]

    # ------------------------------
    # SEX
    # ------------------------------

    fields_sex = ["31-0.0"]

    # ------------------------------
    # HEIGHT
    # ------------------------------

    fields_height = ["12144-2.0"]

    # ------------------------------
    # WEIGHT
    # ------------------------------

    fields_weight = ["21002-2.0", "23098-2.0", "12143-2.0"]

    # ------------------------------
    # SYSTOLIC BP
    # ------------------------------

    fields_systolic_bp = [
        "4080-2.0",
        "4080-2.1",
        "93-2.0",
        "93-2.1",
        "4080-1.0",
        "4080-1.1",
        "93-1.0",
        "93-1.1",
        "4080-0.0",
        "4080-0.1",
        "93-0.0",
        "93-0.1",
    ]

    # ------------------------------
    # HYPERTENSION TREATMENT
    # ------------------------------

    fields_hypertension_treatment = [
        "6153-2.0",
        "6153-2.1",
        "6153-2.2",
        "6153-2.3",
        "6177-2.0",
        "6177-2.1",
        "6177-2.2",
        "6153-1.0",
        "6153-1.1",
        "6153-1.2",
        "6153-1.3",
        "6177-1.0",
        "6177-1.1",
        "6177-1.2",
        "6153-0.0",
        "6153-0.1",
        "6153-0.2",
        "6153-0.3",
        "6177-0.0",
        "6177-0.1",
        "6177-0.2",
    ]
    # include verbal interview medication fields
    fields_hypertension_treatment_2 = []
    for instance in [0, 1]:
        for array in range(48):
            fields_hypertension_treatment_2.append(f"20003-{instance}.{array}")

    # ------------------------------
    # GLUCOSE
    # ------------------------------

    # casual glucose
    fields_glucose = ["30740-1.0", "30740-0.0"]

    # ------------------------------
    # DIABETES TREATMENT
    # ------------------------------

    # should match glucose instance --> instances 1 and 0
    fields_diabetes_treatment = [
        "6153-1.0",
        "6153-1.1",
        "6153-1.2",
        "6153-1.3",
        "6177-1.0",
        "6177-1.1",
        "6177-1.2",
        "6153-0.0",
        "6153-0.1",
        "6153-0.2",
        "6153-0.3",
        "6177-0.0",
        "6177-0.1",
        "6177-0.2",
    ]
    fields_diabetes_treatment_2 = ["2986-1.0", "2986-0.0"]
    # include verbal interview medication fields
    fields_diabetes_treatment_3 = []
    for instance in [0, 1]:
        for array in range(48):
            fields_diabetes_treatment_3.append(f"20003-{instance}.{array}")

    # ------------------------------
    # CURRENT SMOKER
    # ------------------------------

    fields_current_smoker = ["20116-2.0", "20116-1.0", "20116-0.0"]

    # ------------------------------
    # TOTAL CHOLESTEROL
    # ------------------------------

    fields_total_cholesterol = ["30690-1.0", "30690-0.0"]

    # ------------------------------
    # HDL CHOLESTEROL
    # ------------------------------

    fields_hdl_cholesterol = ["30760-1.0", "30760-0.0"]

    # ------------------------------
    # QRS DURATION
    # ------------------------------

    fields_qrs_duration = ["12340-2.0"]

    usecols = list(
        set(
            [
                "eid",
                *fields_age,
                *fields_height,
                *fields_weight,
                *fields_sex,
                *fields_systolic_bp,
                *fields_hypertension_treatment,
                *fields_hypertension_treatment_2,
                *fields_glucose,
                *fields_diabetes_treatment,
                *fields_diabetes_treatment_2,
                *fields_diabetes_treatment_3,
                *fields_current_smoker,
                *fields_total_cholesterol,
                *fields_hdl_cholesterol,
                *fields_qrs_duration,
            ]
        )
    )

    chunks = list(
        _iter_filtered_chunks(
            args.ukbb_csv,
            cohort_eids=cohort_eids,
            usecols=usecols,
            chunksize=args.chunksize,
        )
    )

    if not chunks:
        raise RuntimeError(
            "No cohort case_ids were found in the UKBB CSV. "
        )

    df = pd.concat(chunks, axis=0, ignore_index=True)

    # De-duplicate on eid (some exports can contain duplicates after merges).
    df = df.drop_duplicates(subset=["eid"], keep="first").copy()

    df["age"] = pd.to_numeric(df[fields_age].bfill(axis=1).iloc[:, 0], errors="coerce")
    df["sex"] = pd.to_numeric(
        df[fields_sex].bfill(axis=1).iloc[:, 0], errors="coerce"
    ).map({0: "female", 1: "male"})
    df["height_cm"] = pd.to_numeric(
        df[fields_height].bfill(axis=1).iloc[:, 0], errors="coerce"
    )
    df["weight_kg"] = pd.to_numeric(
        df[fields_weight].bfill(axis=1).iloc[:, 0], errors="coerce"
    )
    df["bsa"] = df.apply(
        lambda row: _compute_bsa(row["height_cm"], row["weight_kg"]), axis=1
    )

    visit_2_bp_fields = [
        f for f in fields_systolic_bp if f.split("-")[1].startswith("2.")
    ]
    visit_1_bp_fields = [
        f for f in fields_systolic_bp if f.split("-")[1].startswith("1.")
    ]
    visit_0_bp_fields = [
        f for f in fields_systolic_bp if f.split("-")[1].startswith("0.")
    ]
    df["visit_2_systolic_bp"] = pd.to_numeric(
        df[visit_2_bp_fields].mean(axis=1), errors="coerce"
    )
    df["visit_1_systolic_bp"] = pd.to_numeric(
        df[visit_1_bp_fields].mean(axis=1), errors="coerce"
    )
    df["visit_0_systolic_bp"] = pd.to_numeric(
        df[visit_0_bp_fields].mean(axis=1), errors="coerce"
    )
    df["systolic_bp"] = pd.to_numeric(
        df[["visit_2_systolic_bp", "visit_1_systolic_bp", "visit_0_systolic_bp"]]
        .bfill(axis=1)
        .iloc[:, 0],
        errors="coerce",
    )

    # As long as there is a single treatment marked as 'Blood pressure medication' (2), we consider
    # the subject to be under hypertension treatment.
    df[fields_hypertension_treatment] = df[fields_hypertension_treatment].replace(
        {
            1: False,  # Cholesterol lowering medication
            2: True,  # Blood pressure medication
            3: False,  # Insulin
            4: False,  # Hormone replacement therapy
            5: False,  # Oral contraceptive pill or minipill
            -7: False,  # None of the above
            -1: pd.NA,  # Do not know
            -3: pd.NA,  # Prefer not to answer
        }
    )
    mask_hypertension_treatment = (df[fields_hypertension_treatment] == True).any(
        axis=1
    )
    mask_no_hypertension_treatment = (df[fields_hypertension_treatment] == False).any(
        axis=1
    )
    df["hypertension_treatment"] = pd.NA
    df.loc[mask_no_hypertension_treatment, "hypertension_treatment"] = False
    df.loc[mask_hypertension_treatment, "hypertension_treatment"] = True

    # Include verbal interview medication fields for hypertension treatment.
    hypertension_medication_codes = [
        # --- ACE Inhibitors ---
        1140860750,  # Captopril
        1140860758,  # Capoten (Captopril)
        1140860764,  # Captopril + Hydrochlorothiazide
        1140851692,  # Capozide (Captopril combination)
        1140881714,  # Capozide tablet
        1141181186,  # Co-zidocapt (Captopril + Hydrochlorothiazide)
        1140888552,  # Enalapril
        1140860776,  # Innovace (Enalapril)
        1140881712,  # Renitec (Enalapril)
        1140860790,  # Enalapril + Hydrochlorothiazide
        1140860696,  # Lisinopril
        1140860706,  # Carace (Lisinopril)
        1140864910,  # Carace 10 Plus (Lisinopril + Hydrochlorothiazide)
        1140860714,  # Zestril (Lisinopril)
        1140864952,  # Lisinopril + Hydrochlorothiazide
        1140864618,  # Zestoretic (Lisinopril + Hydrochlorothiazide)
        1140860806,  # Ramipril
        1141188408,  # Tritace (Ramipril)
        1141165470,  # Felodipine + Ramipril
        1140888560,  # Perindopril
        1140860802,  # Coversyl (Perindopril)
        1141180592,  # Perindopril + Indapamide
        1141180598,  # Coversyl Plus (Perindopril + Indapamide)
        1140860728,  # Quinapril
        1140881706,  # Accupro (Quinapril)
        1140860738,  # Quinapril + Hydrochlorothiazide
        1140860736,  # Accuretic (Quinapril + Hydrochlorothiazide)
        1140888556,  # Fosinopril
        1140860878,  # Staril (Fosinopril)
        1140860904,  # Trandolapril
        1140860912,  # Gopten (Trandolapril)
        1141153328,  # Trandolapril + Verapamil
        1140860882,  # Cilazapril
        1140860892,  # Vascace (Cilazapril)
        1141164148,  # Imidapril
        1141164154,  # Tanatril (Imidapril)
        1140923712,  # Moexipril
        1140923718,  # Perdix (Moexipril)
        # --- Angiotensin Receptor Blockers (ARBs) ---
        1140916356,  # Losartan
        1141179974,  # Cozaar (Losartan)
        1140916362,  # Cozaar half strength (Losartan)
        1141151016,  # Losartan + Hydrochlorothiazide
        1141151018,  # Cozaar-Comp (Losartan + Hydrochlorothiazide)
        1141145660,  # Valsartan
        1141145668,  # Diovan (Valsartan)
        1141201038,  # Valsartan + Hydrochlorothiazide
        1141201040,  # Co-Diovan (Valsartan + Hydrochlorothiazide)
        1141152998,  # Irbesartan
        1141153006,  # Aprovel (Irbesartan)
        1141172682,  # Irbesartan + Hydrochlorothiazide
        1141172686,  # Coaprovel (Irbesartan + Hydrochlorothiazide)
        1141156836,  # Candesartan
        1141156846,  # Amias (Candesartan)
        1141193282,  # Olmesartan
        1141193346,  # Olmetec (Olmesartan)
        1141166006,  # Telmisartan
        1141172492,  # Micardis (Telmisartan)
        1141187788,  # Telmisartan + Hydrochlorothiazide
        1141187790,  # Micardisplus (Telmisartan + Hydrochlorothiazide)
        1141171336,  # Eprosartan
        1141171344,  # Teveten (Eprosartan)
        # --- Beta-Blockers ---
        1140866738,  # Atenolol
        1140866756,  # Tenormin (Atenolol)
        1141146126,  # Atenolol + Bendrofluazide
        1141194810,  # Atenolol + Bendroflumethiazide
        1141180778,  # Atenolol + Chlortalidone
        1141146124,  # Atenolol + Chlorthalidone
        1141146128,  # Atenolol + Co-amilozide
        1140860426,  # Atenolol + Nifedipine
        1140860356,  # Beta-Adalat (Atenolol + Nifedipine)
        1140860398,  # Kalten (Atenolol + Amiloride + Hydrochlorothiazide)
        1140923336,  # Co-tenidone (Atenolol + Chlortalidone)
        1140860328,  # Tenoretic (Atenolol + Chlortalidone)
        1140879760,  # Bisoprolol
        1141171152,  # Cardicor (Bisoprolol)
        1140860492,  # Emcor (Bisoprolol)
        1140864950,  # Bisoprolol + Hydrochlorothiazide
        1140879818,  # Metoprolol
        1140860266,  # Betaloc (Metoprolol)
        1140860274,  # Lopresor (Metoprolol)
        1140860308,  # Metoprolol + Chlorthalidone
        1140860404,  # Metoprolol + Hydrochlorothiazide
        1140860386,  # Co-Betaloc (Metoprolol + Hydrochlorothiazide)
        1140860402,  # Lopresoretic (Metoprolol + Chlorthalidone)
        1140879842,  # Propranolol
        1140866804,  # Inderal (Propranolol)
        1140866800,  # Half-Inderal (Propranolol)
        1140860418,  # Propranolol + Bendrofluazide
        1140909368,  # Carvedilol
        1141168498,  # Eucardic (Carvedilol)
        1141164276,  # Nebivolol
        1141164280,  # Nebilet (Nebivolol)
        1140879824,  # Labetalol
        1140860250,  # Trandate (Labetalol)
        1140866724,  # Acebutolol
        1140866726,  # Sectral (Acebutolol)
        1140860422,  # Acebutolol + Hydrochlorothiazide
        1140879762,  # Celiprolol
        1140860498,  # Celectol (Celiprolol)
        1140879830,  # Oxprenolol
        1140860220,  # Slow-Trasicor (Oxprenolol)
        1140860222,  # Trasicor (Oxprenolol)
        1140860292,  # Pindolol
        1140860294,  # Visken (Pindolol)
        1140860322,  # Pindolol + Clopamide
        1140879854,  # Sotalol
        1140860332,  # Sotalol + Hydrochlorothiazide
        1140879866,  # Timolol
        1140860340,  # Timolol + Bendrofluazide
        1140860342,  # Timolol + Bendrofluazide (higher dose)
        1141194808,  # Timolol + Bendroflumethiazide
        1140860336,  # Timolol + Co-amilozide
        1140860192,  # Nadolol
        1140860312,  # Nadolol + Bendrofluazide 40mg
        1140860316,  # Nadolol + Bendrofluazide 80mg
        1141194804,  # Nadolol + Bendroflumethiazide
        1140879758,  # Betaxolol
        1140860320,  # Penbutolol + Furosemide
        # --- Calcium Channel Blockers ---
        1140879802,  # Amlodipine
        1140861202,  # Istin (Amlodipine)
        1140861088,  # Nifedipine
        1140861090,  # Adalat (Nifedipine)
        1140881702,  # Adalate (Nifedipine)
        1140861120,  # Coracten (Nifedipine)
        1140888646,  # Felodipine
        1140928212,  # Plendil (Felodipine)
        1141153026,  # Lercanidipine
        1141153032,  # Zanidip (Lercanidipine)
        1140861276,  # Lacidipine
        1140861282,  # Motens (Lacidipine)
        1140879810,  # Nicardipine
        1140861176,  # Cardene (Nicardipine)
        1140872568,  # Nimodipine
        1140872472,  # Nimotop (Nimodipine)
        1140928226,  # Nisoldipine
        1140879806,  # Diltiazem
        1140861128,  # Tildiem (Diltiazem)
        1140861138,  # Adizem (Diltiazem)
        1140926778,  # Diltiazem + Hydrochlorothiazide
        1140926780,  # Adizem-XL Plus (Diltiazem + Hydrochlorothiazide)
        1140888510,  # Verapamil
        1140866554,  # Cordilox (Verapamil)
        1140866460,  # Half Securon (Verapamil)
        1140866466,  # Securon (Verapamil)
        # --- Thiazide Diuretics ---
        1140866122,  # Bendrofluazide
        1141194794,  # Bendroflumethiazide
        1140866450,  # Bendrofluazide + Potassium
        1141194800,  # Bendroflumethiazide + Potassium
        1140910442,  # BZT - Bendrofluazide
        1140866162,  # Hydrochlorothiazide
        1140909706,  # Chlortalidone
        1140866146,  # Hygroton (Chlortalidone)
        1140851364,  # Hygroton K (Chlortalidone + Potassium)
        1140866078,  # Indapamide
        1141146378,  # Natrilix SR (Indapamide)
        1140866092,  # Metolazone
        1140866156,  # Cyclopenthiazide
        1140866138,  # Chlorothiazide
        1140866102,  # Polythiazide
        # --- Loop Diuretics ---
        1140866116,  # Frusemide
        1140866506,  # Frusemide + Potassium
        1140909708,  # Furosemide
        1141195258,  # Furosemide + Potassium
        1140866248,  # Lasix (Furosemide)
        1140881728,  # Lasix+K (Furosemide + Potassium)
        1140866280,  # Bumetanide
        1140866282,  # Burinex (Bumetanide)
        1140866356,  # Burinex A (Bumetanide + Amiloride)
        1140866438,  # Burinex K (Bumetanide + Potassium)
        1140866448,  # Bumetanide + Potassium
        1140888496,  # Torasemide
        1140866200,  # Ethacrynic Acid
        1140866206,  # Ethacrynic Acid 50mg tablet
        1141157184,  # Ethacrynic Acid product
        # --- Potassium-sparing Diuretics / Aldosterone Antagonists ---
        1140888512,  # Amiloride
        1140866220,  # Midamor (Amiloride)
        1140866422,  # Amiloride + Cyclopenthiazide
        1140866426,  # Amiloride + Bumetanide
        1140866236,  # Spironolactone
        1140866244,  # Aldactone (Spironolactone)
        1141201244,  # Eplerenone
        1141201250,  # Inspra (Eplerenone)
        1140866388,  # Triamterene
        1140866324,  # Triamterene + Benzthiazide
        1141180772,  # Triamterene + Chlortalidone
        1140866330,  # Triamterene + Chlorthalidone
        1140866332,  # Triamterene + Frusemide
        1141195254,  # Triamterene + Furosemide
        # --- Combination Diuretics ---
        1140923402,  # Co-amilofruse (Amiloride + Furosemide)
        1140923276,  # Co-amilozide (Amiloride + Hydrochlorothiazide)
        1140851436,  # Vasetic Co-amilozide
        # --- Alpha-Blockers ---
        1140879778,  # Doxazosin
        1140860690,  # Cardura (Doxazosin)
        1140879794,  # Prazosin
        1140860580,  # Hypovase (Prazosin)
        1140879798,  # Terazosin
        1140860610,  # Hytrin (Terazosin)
        1140879782,  # Indoramin
        1140860654,  # Baratol (Indoramin)
        1140860658,  # Doralese (Indoramin)
        1141157490,  # Indoramin product
        # --- Centrally Acting Antihypertensives ---
        1140860470,  # Methyldopa
        1140860478,  # Aldomet (Methyldopa)
        1140910606,  # Alpha Methyldopa
        1140860562,  # Methyldopa + Hydrochlorothiazide
        1140883468,  # Clonidine
        1140860454,  # Catapres (Clonidine)
        1140871986,  # Clonidine hydrochloride tablet
        1140871984,  # Dixarit (Clonidine)
        1140928284,  # Moxonidine
        1140928290,  # Physiotens (Moxonidine)
        # --- Direct Vasodilators ---
        1140888686,  # Hydralazine
        1140860520,  # Apresoline (Hydralazine)
        1140860532,  # Minoxidil
        1140860534,  # Loniten (Minoxidil)
        1140888684,  # Diazoxide
        # --- Other ---
        1140888578,  # Antihypertensive (generic)
    ]
    mask_hypertension_treatment_2 = (
        df[fields_hypertension_treatment_2].isin(hypertension_medication_codes)
    ).any(axis=1)
    df.loc[mask_hypertension_treatment_2, "hypertension_treatment"] = True

    df["casual_glucose"] = pd.to_numeric(
        df[fields_glucose].bfill(axis=1).iloc[:, 0], errors="coerce"
    )

    # As long as there is a single treatment marked as 'Insulin' (3), we consider
    # the subject to be under diabetes treatment.
    df[fields_diabetes_treatment] = df[fields_diabetes_treatment].replace(
        {
            1: False,  # Cholesterol lowering medication
            2: False,  # Blood pressure medication
            3: True,  # Insulin
            4: False,  # Hormone replacement therapy
            5: False,  # Oral contraceptive pill or minipill
            -7: False,  # None of the above
            -1: pd.NA,  # Do not know
            -3: pd.NA,  # Prefer not to answer
        }
    )
    mask_diabetes_treatment = (df[fields_diabetes_treatment] == True).any(axis=1)
    mask_no_diabetes_treatment = (df[fields_diabetes_treatment] == False).any(axis=1)
    df["diabetes_treatment"] = pd.NA
    df.loc[mask_no_diabetes_treatment, "diabetes_treatment"] = False
    df.loc[mask_diabetes_treatment, "diabetes_treatment"] = True

    # Check additional diabetes treatment fields (2986) for any indication of treatment.
    mask_diabetes_treatment_2 = (df[fields_diabetes_treatment_2] == 1).any(axis=1)
    df.loc[mask_diabetes_treatment_2, "diabetes_treatment"] = True

    # Include verbal interview medication fields for diabetes treatment.
    diabetes_medication_codes = [
        1140884600,  # Metformin
        1140921964,  # Glucamet (Metformin)
        1140874744,  # Gliclazide
        1140874746,  # Diamicron (Gliclazide)
        1141169504,  # Diaglyk (Gliclazide)
        1141171508,  # Vivazide (Gliclazide)
        1140874718,  # Glibenclamide
        1140874724,  # Daonil (Glibenclamide)
        1140874726,  # Semi-Daonil (Glibenclamide)
        1140874728,  # Euglucon (Glibenclamide)
        1140874732,  # Malix (Glibenclamide)
        1140874736,  # Diabetamide (Glibenclamide)
        1140874740,  # Calabren (Glibenclamide)
        1140857590,  # Libanil (Glibenclamide)
        1140874646,  # Glipizide
        1140874652,  # Minodiab (Glipizide)
        1140874650,  # Glibenese (Glipizide)
        1141157284,  # Glipizide product
        1141152590,  # Glimepiride
        1141156984,  # Amaryl (Glimepiride)
        1140874674,  # Tolbutamide
        1140874678,  # Glyconon (Tolbutamide)
        1140874680,  # Rastinon (Tolbutamide)
        1140874686,  # Glucophage (Metformin)
        1140874690,  # Orabet (Tolbutamide)
        1140874706,  # Chlorpropamide
        1140874712,  # Diabinese (Chlorpropamide)
        1140874716,  # Glymese (Chlorpropamide)
        1140857584,  # Acetohexamide
        1140857494,  # Glibornuride
        1140874664,  # Tolazamide
        1140874666,  # Tolinase (Tolazamide)
        1140874658,  # Gliquidone
        1140874660,  # Glurenorm (Gliquidone)
        1141168660,  # Repaglinide
        1141168668,  # Novonorm (Repaglinide)
        1141173882,  # Nateglinide
        1141173786,  # Starlix (Nateglinide)
        1141171646,  # Pioglitazone
        1141171652,  # Actos (Pioglitazone)
        1141177600,  # Rosiglitazone
        1141177606,  # Avandia (Rosiglitazone)
        1141153254,  # Troglitazone
        1140868902,  # Acarbose
        1140868908,  # Glucobay (Acarbose)
        1140883066,  # Insulin product
        1141189090,  # Rosiglitazone + Metformin
    ]
    mask_diabetes_treatment_3 = (
        df[fields_diabetes_treatment_3].isin(diabetes_medication_codes)
    ).any(axis=1)
    df.loc[mask_diabetes_treatment_3, "diabetes_treatment"] = True

    df[fields_current_smoker] = df[fields_current_smoker].replace(
        {
            0: False,  # Never
            1: False,  # Previous
            2: True,  # Current
            -3: pd.NA,  # Prefer not to answer
        }
    )
    df["current_smoker"] = (
        df[fields_current_smoker].infer_objects(copy=False).bfill(axis=1).iloc[:, 0]
    )

    df["total_cholesterol"] = pd.to_numeric(
        df[fields_total_cholesterol].bfill(axis=1).iloc[:, 0], errors="coerce"
    )
    df["hdl_cholesterol"] = pd.to_numeric(
        df[fields_hdl_cholesterol].bfill(axis=1).iloc[:, 0], errors="coerce"
    )
    df["qrs_duration"] = pd.to_numeric(
        df[fields_qrs_duration].bfill(axis=1).iloc[:, 0], errors="coerce"
    )

    # Outlier removal for QRS duration (in ms).
    qrs_outlier_mask = (df["qrs_duration"] < 40) | (df["qrs_duration"] > 200)
    df.loc[qrs_outlier_mask, "qrs_duration"] = pd.NA

    # ------------------------------
    # BMI
    # ------------------------------

    df["body_mass_index"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    # Build an output table starting from the cohort file so that missing UKBB rows are kept.
    # eid strips the visit suffix (e.g. "_2") to match plain UKBB eids.
    cohort_cases_df = cohort_df[["case_id", "split"]].copy()
    cohort_cases_df["eid"] = (
        cohort_cases_df["case_id"].astype(str).str.split("_").str[0]
    )

    out = cohort_cases_df.merge(
        df,
        on="eid",
        how="left",
        suffixes=("", "_ukbb"),
    )

    # Sort for stable output.
    out = out.sort_values(by=["case_id"], na_position="last").reset_index(drop=True)

    # ------------------------------
    # DATA IMPUTATION
    # ------------------------------

    train_case_ids = (
        cohort_df.loc[cohort_df["split"] == "train", "case_id"].astype(str).tolist()
    )
    train_mask = out["case_id"].astype(str).isin(set(train_case_ids))

    out_complete = _mice_impute(out, train_mask=train_mask, random_state=0)

    # Keep a stable, explicit column ordering (extra columns are okay downstream).
    ordered_cols = [
        "case_id",
        "split",
        "bsa",
        "age",
        "sex",
        "casual_glucose",
        "diabetes_treatment",
        "current_smoker",
        "systolic_bp",
        "hypertension_treatment",
        "total_cholesterol",
        "hdl_cholesterol",
        "body_mass_index",
        "qrs_duration",
    ]
    out_complete = out_complete[ordered_cols].copy()

    if args.out_file is None:
        out_file = Path(args.cohort_file).parent / "demographics.csv"
    else:
        out_file = Path(args.out_file)
    out_complete.to_csv(out_file, index=False)

    n_total = len(cohort_df)
    n_found = int(out_complete["case_id"].nunique())
    log.info("Wrote demographics for " f"{n_found}/{n_total} cohort cases to: {out_file}")


if __name__ == "__main__":
    main()
