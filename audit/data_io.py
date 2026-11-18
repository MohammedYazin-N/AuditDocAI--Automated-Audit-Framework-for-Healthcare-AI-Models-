"""FEATURE 1 - dataset loading, profiling and suitability checks.

The uploaded dataframe is never modified: `load_table` returns a fresh frame and
the app keeps it as the immutable "original". Cleaning (data_quality.py) always
works on a copy.
"""
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd

# Tokens commonly used for "missing" in healthcare exports. The diabetes notebook used "?".
# They are only *converted* to NaN if the user keeps them ticked - a visible, logged choice.
DEFAULT_MISSING_TOKENS = ["?", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", ""]

ID_NAME_HINTS = re.compile(r"(^|_)(id|ids|nbr|number|num|mrn|uid|uuid|key|index|patient|encounter|record)($|_)", re.I)
DEMOGRAPHIC_HINTS = ["age", "gender", "sex", "race", "ethnic", "ethnicity", "religion", "marital",
                     "insurance", "payer", "language", "nationality"]


def load_table(file_obj, filename: str, missing_tokens=None) -> tuple[pd.DataFrame, dict]:
    """Read CSV / XLSX into a DataFrame. Returns (df, load_info)."""
    missing_tokens = DEFAULT_MISSING_TOKENS if missing_tokens is None else missing_tokens
    name = filename.lower()
    raw_bytes = file_obj.read() if hasattr(file_obj, "read") else file_obj
    info = {"filename": filename, "size_bytes": len(raw_bytes), "missing_tokens_applied": list(missing_tokens)}
    # keep_default_na=False: pandas would otherwise silently turn strings like "None"/"NA" into NaN,
    # which in healthcare data can be a real category (e.g. lab "None" = test not performed).
    na_vals = [t for t in missing_tokens]
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw_bytes), na_values=na_vals, keep_default_na=False, low_memory=False)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes), na_values=na_vals, keep_default_na=False)
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")
    # strip header whitespace only (values untouched)
    df.columns = [str(c).strip() for c in df.columns]
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        raise ValueError(f"Duplicate column names found: {dup_cols}. Please rename them and re-upload.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The file is empty.")
    return df, info


def column_roles(df: pd.DataFrame) -> dict:
    """Numeric vs categorical split (numeric = numeric dtype with > 10 distinct values or float)."""
    num, cat = [], []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            cat.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            num.append(c)
        else:
            cat.append(c)
    return {"numeric": num, "categorical": cat}


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing, unique."""
    n = len(df)
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({"column": c, "dtype": str(s.dtype), "missing_n": int(s.isna().sum()),
                     "missing_pct": round(s.isna().mean() * 100, 2), "n_unique": int(s.nunique(dropna=True)),
                     "example": "" if s.dropna().empty else str(s.dropna().iloc[0])[:30]})
    return pd.DataFrame(rows)


def detect_identifier_columns(df: pd.DataFrame, target: str | None = None) -> list[dict]:
    """Heuristic identifier detection. Returns candidates with a reason; the USER confirms.

    Criteria (documented, heuristic - not a guarantee):
      * name looks like an id (id / nbr / mrn / key ...), OR
      * (almost) every row has a unique value in an integer/string column.
    """
    out = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        uniq_ratio = s.nunique(dropna=True) / max(n, 1)
        name_hit = bool(ID_NAME_HINTS.search(c))
        is_int_like = pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s)
        reasons = []
        if name_hit:
            reasons.append("name looks like an identifier")
        if uniq_ratio >= 0.95 and is_int_like and n > 20:
            reasons.append(f"{uniq_ratio:.0%} of values are unique")
        # patient-level ids repeat (several encounters per patient): name-hit + moderate uniqueness
        if name_hit and 0.2 <= uniq_ratio < 0.95:
            reasons.append(f"{uniq_ratio:.0%} unique - may be a repeated (patient-level) identifier")
        if reasons:
            out.append({"column": c, "unique_ratio": round(uniq_ratio, 4), "reason": "; ".join(reasons)})
    return out


def suggest_demographic_columns(df: pd.DataFrame, exclude=()) -> list[str]:
    """Columns whose NAME suggests a demographic attribute (user can add/remove)."""
    found = []
    for c in df.columns:
        if c in exclude:
            continue
        low = c.lower()
        if any(h in low for h in DEMOGRAPHIC_HINTS):
            found.append(c)
    return found


def suggest_target_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look like a classification label (2-10 distinct values). Not a decision - a hint."""
    hints = []
    for c in df.columns:
        k = df[c].nunique(dropna=True)
        if 2 <= k <= 10:
            hints.append((0 if re.search(r"target|label|outcome|class|diagnos|readmit|death|status|y$", c, re.I) else 1, c))
    return [c for _, c in sorted(hints)]


def audit_availability(df: pd.DataFrame, target: str | None, model_loaded: bool, model_compatible: bool,
                       demographics: list[str], has_predictions_possible: bool) -> list[dict]:
    """FEATURE 'dynamic compatibility': which audits can run, and why / why not."""
    rows = []
    rows.append({"audit": "Data Quality (Feature 3)", "available": df is not None,
                 "reason": "Dataset uploaded." if df is not None else "Upload a dataset first.",
                 "needs": "dataset (target optional - imbalance/outcome checks need it)"})
    leak_ok = df is not None and target is not None
    rows.append({"audit": "Leakage - data-only checks (Feature 4A)", "available": leak_ok,
                 "reason": "Dataset and target selected." if leak_ok else "Select the target column.",
                 "needs": "dataset + target"})
    perf_ok = leak_ok and model_loaded and model_compatible and has_predictions_possible
    if not model_loaded:
        r = "No model uploaded - performance, fairness and explainability cannot be evaluated."
    elif not model_compatible:
        r = "Model is not compatible with this dataset - see Model Upload page for the reason."
    elif not leak_ok:
        r = "Select the target column."
    else:
        r = "Dataset, target and compatible model available."
    rows.append({"audit": "Model performance (Feature 4B)", "available": perf_ok, "reason": r,
                 "needs": "dataset + target + compatible model + evaluation data with true labels"})
    fair_ok = perf_ok and len(demographics) > 0
    rows.append({"audit": "Fairness (Feature 5)", "available": fair_ok,
                 "reason": r if not perf_ok else ("Demographic column(s) selected." if fair_ok else "Select at least one demographic column."),
                 "needs": "performance prerequisites + demographic column(s)"})
    rows.append({"audit": "Explainability (Feature 6)", "available": perf_ok,
                 "reason": r if not perf_ok else "Model loaded; method availability depends on model type (checked on the page).",
                 "needs": "compatible model + feature data"})
    return rows
