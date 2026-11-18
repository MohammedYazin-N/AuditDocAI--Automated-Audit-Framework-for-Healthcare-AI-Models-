"""FEATURE 3 - Data Quality Audit.

Generalised from Person 1's Module 1 notebook. The statistical methods and the thresholds
(CRITICAL_MISSING=0.90, HIGH_MISSING=0.50, NEAR_CONSTANT=0.99, IQR 1.5/3, |z|>3, rare level
<0.1 %, high cardinality >=50, score grades 90/75/50) are taken UNCHANGED from the notebook.
What changed: nothing is hardcoded to the diabetes columns - every column name comes in as
an argument chosen by the user.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .common import Finding, PASSED, WARNING, FAILED, NOT_EVALUATED, FROM_NOTEBOOK, NEW

# --- thresholds copied from the notebook (Module 1, config cell) -------------------------
CRITICAL_MISSING = 0.90
HIGH_MISSING = 0.50
NEAR_CONSTANT = 0.99
Z_THRESHOLD = 3.0
IQR_K, IQR_K_EXTREME = 1.5, 3.0
RARE_LEVEL_SHARE = 0.001
HIGH_CARDINALITY = 50
MIN_GROUP_SIZE = 500          # notebook default for "low representation"


def missing_severity(p: float) -> str:
    return "Critical" if p >= CRITICAL_MISSING else "High" if p >= HIGH_MISSING else "Medium" if p >= 0.05 else "Low"


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    t = pd.DataFrame({"missing_n": df.isna().sum(), "missing_pct": df.isna().mean() * 100})
    t = t[t.missing_n > 0].sort_values("missing_pct", ascending=False)
    t["severity"] = (t.missing_pct / 100).apply(missing_severity)
    return t


def missing_by_group(df: pd.DataFrame, group_cols: list[str], min_group: int = 100, top_n: int = 10):
    """Notebook 2.1(a): spread (percentage points) of missing rate between subgroups."""
    rows, tables = [], {}
    miss_cols = [c for c in df.columns if df[c].isna().sum() >= 100]
    for g in group_cols:
        if g not in df.columns:
            continue
        gs = df[g]
        sizes = gs.value_counts()
        keep = sizes[sizes >= min_group].index
        cols = [c for c in miss_cols if c != g]
        if not cols or len(keep) < 2:
            continue
        tbl = df[cols].isna().groupby(gs).mean() * 100
        tbl = tbl[tbl.index.isin(keep)]
        tables[g] = tbl
        for c in tbl.columns:
            rows.append({"attribute": g, "column": c, "spread_pp": float(tbl[c].max() - tbl[c].min()),
                         "highest_group": str(tbl[c].idxmax()), "lowest_group": str(tbl[c].idxmin())})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("spread_pp", ascending=False).reset_index(drop=True)
    return out, tables


def missing_vs_outcome(df: pd.DataFrame, y_bin: pd.Series) -> pd.DataFrame:
    """Notebook 2.1(b): outcome rate when a value is missing vs present (two-proportion z-test)."""
    rows = []
    for c in df.columns:
        m = df[c].isna()
        n1, n0 = int(m.sum()), int((~m).sum())
        if n1 < 100 or n0 < 100 or n1 / len(df) >= CRITICAL_MISSING:
            continue
        x1, x0 = int(y_bin[m].sum()), int(y_bin[~m].sum())
        p1, p0 = x1 / n1, x0 / n0
        pooled = (x1 + x0) / (n1 + n0)
        se = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / n1 + 1 / n0))
        z = (p1 - p0) / se
        rows.append({"column": c, "n_missing": n1, "rate_missing_pct": p1 * 100, "rate_present_pct": p0 * 100,
                     "diff_pp": (p1 - p0) * 100, "p_value": math.erfc(abs(z) / math.sqrt(2))})
    out = pd.DataFrame(rows)
    return out.sort_values("p_value").reset_index(drop=True) if not out.empty else out


def duplicate_summary(df: pd.DataFrame, id_col: str | None = None, patient_col: str | None = None) -> dict:
    """Exact duplicate rows; duplicate record-ID values; repeat-patient structure."""
    out = {"rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum())}
    # same features, ignoring the identifier columns -> "near-duplicates"
    non_id = [c for c in df.columns if c not in {id_col, patient_col}]
    out["duplicate_rows_ignoring_ids"] = int(df.duplicated(subset=non_id).sum()) if non_id else 0
    out["duplicate_record_id"] = int(df[id_col].duplicated().sum()) if id_col else None
    if patient_col:
        vc = df[patient_col].value_counts()
        out.update({"unique_patients": int(vc.size), "patients_with_multiple_rows": int((vc > 1).sum()),
                    "rows_of_repeat_patients": int(vc[vc > 1].sum()), "max_rows_per_patient": int(vc.max())})
    return out


def outlier_table(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    """Notebook section 4. Outliers are FLAGGED ONLY - never removed (clinically meaningful)."""
    rows = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        lox, hix = q1 - IQR_K_EXTREME * iqr, q3 + IQR_K_EXTREME * iqr
        sd = s.std(ddof=0)
        z = (s - s.mean()) / sd if sd > 0 else s * 0
        rows.append({"column": c, "min": s.min(), "q1": q1, "median": s.median(), "q3": q3, "p99": s.quantile(.99),
                     "max": s.max(), "iqr_outliers_n": int(((s < lo) | (s > hi)).sum()),
                     "iqr_outliers_pct": float(((s < lo) | (s > hi)).mean() * 100),
                     "extreme_outliers_n": int(((s < lox) | (s > hix)).sum()),
                     "z_gt_3_n": int((z.abs() > Z_THRESHOLD).sum()),
                     "degenerate_iqr": bool(iqr == 0)})
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlation_pairs(df: pd.DataFrame, num_cols: list[str], y_bin: pd.Series | None, top_n: int = 10):
    """Notebook 4.1: Spearman (robust to skewed counts). Returns (pairs, with_target)."""
    num = [c for c in num_cols if df[c].nunique() > 1]
    if len(num) < 2:
        return pd.DataFrame(), pd.DataFrame()
    corr = df[num].corr(method="spearman")
    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"feature_a": a, "feature_b": b, "spearman_rho": v, "abs_rho": abs(v)})
    pairs = pd.DataFrame(pairs).sort_values("abs_rho", ascending=False).head(top_n).reset_index(drop=True)
    wt = pd.DataFrame()
    if y_bin is not None:
        wt = pd.DataFrame({"spearman_with_target": df[num].apply(lambda s: s.corr(y_bin, method="spearman"))})
        wt = wt.assign(abs=wt.spearman_with_target.abs()).sort_values("abs", ascending=False).drop(columns="abs")
    return pairs, corr


def invalid_value_checks(df: pd.DataFrame, num_cols: list[str], range_rules: dict | None = None) -> pd.DataFrame:
    """Generic validity rules. Returns one row per rule with violation counts.

    Built-in rules (dataset-independent): infinite values; numbers stored as text mixed with text;
    columns with a single constant value. Optional user rules: {column: (min, max)}.
    """
    rows = []
    n = len(df)
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        k = int(np.isinf(s).sum())
        if k:
            rows.append({"rule": f"{c}: infinite values", "type": "Validity", "violations_n": k, "column": c})
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            s = df[c].dropna().astype(str)
            if len(s) == 0:
                continue
            numlike = pd.to_numeric(s, errors="coerce").notna().mean()
            if 0.05 < numlike < 0.95:
                rows.append({"rule": f"{c}: mixed numeric and text values ({numlike:.0%} numeric)", "type": "Validity",
                             "violations_n": int(round(min(numlike, 1 - numlike) * len(s))), "column": c})
    for c, (lo, hi) in (range_rules or {}).items():
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            k = int(((s < lo) | (s > hi)).sum())
            rows.append({"rule": f"{c}: outside user range [{lo}, {hi}]", "type": "Validity", "violations_n": k, "column": c})
    out = pd.DataFrame(rows, columns=["rule", "type", "violations_n", "column"])
    if not out.empty:
        out["violations_pct"] = out.violations_n / n * 100
    return out


def constant_columns(df: pd.DataFrame):
    const, near = [], {}
    for c in df.columns:
        vc = df[c].value_counts(dropna=True, normalize=True)
        if vc.empty:
            continue
        if df[c].nunique(dropna=True) <= 1:
            const.append(c)
        elif vc.iloc[0] >= NEAR_CONSTANT:
            near[c] = float(vc.iloc[0])
    return const, near


def cardinality_table(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cat_cols:
        vc = df[c].value_counts(dropna=True)
        if vc.empty:
            continue
        share = vc / vc.sum()
        rare = share < RARE_LEVEL_SHARE
        rows.append({"column": c, "n_levels": int(vc.size), "top_level": str(vc.index[0]),
                     "top_level_share_pct": float(share.iloc[0] * 100), "rare_levels_n": int(rare.sum()),
                     "rows_in_rare_levels_pct": float(share[rare].sum() * 100)})
    out = pd.DataFrame(rows)
    return out.sort_values("n_levels", ascending=False).reset_index(drop=True) if not out.empty else out


def class_balance(y: pd.Series) -> dict:
    """Notebook section 5. Works for binary and multiclass."""
    vc = y.value_counts(dropna=False)
    n = int(vc.sum())
    maj, mino = int(vc.max()), int(vc[vc > 0].min())
    ratio = maj / mino if mino else float("inf")
    label = "Balanced" if ratio < 1.5 else "Mild" if ratio < 3 else "Moderate" if ratio < 10 else "Severe"
    return {"counts": vc.to_dict(), "n": n, "n_classes": int(vc.size), "majority_share": maj / n,
            "minority_share": mino / n, "imbalance_ratio": ratio, "label": label,
            "majority_baseline_accuracy": maj / n}


def subgroup_representation(df, y_bin, group_cols, min_group=MIN_GROUP_SIZE):
    rows = []
    for g in group_cols:
        if g not in df.columns:
            continue
        t = pd.DataFrame({"g": df[g].astype("object").where(df[g].notna(), "Missing"), "y": y_bin.values})
        agg = t.groupby("g")["y"].agg(["size", "mean"]).reset_index()
        for _, r in agg.iterrows():
            rows.append({"attribute": g, "group": r["g"], "n": int(r["size"]), "outcome_rate_pct": r["mean"] * 100,
                         "share_of_data_pct": r["size"] / len(df) * 100, "low_representation": bool(r["size"] < min_group)})
    return pd.DataFrame(rows)


def patient_demographic_consistency(df, patient_col, demo_cols):
    """Notebook 6: demographics should not change between one patient's rows."""
    rows = []
    for c in demo_cols:
        if c == patient_col or c not in df.columns:
            continue
        k = int((df.groupby(patient_col)[c].nunique(dropna=True) > 1).sum())
        rows.append({"column": c, "patients_with_changing_value": k})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Scorecard (notebook section 7). Dimension definitions unchanged; dimensions that cannot be
# computed from the given inputs are marked "Not evaluated" and left OUT of the average.
# --------------------------------------------------------------------------------------
def grade(s: float) -> str:
    return "Good" if s >= 90 else "Acceptable" if s >= 75 else "Poor" if s >= 50 else "Critical"


def scorecard(df, dup: dict, validity_df: pd.DataFrame, balance: dict | None, consistency_rows: int | None,
              consistency_violations: int | None) -> pd.DataFrame:
    sc = {}
    sc["Completeness"] = (1 - df.isna().sum().sum() / df.size) * 100
    dup_n = dup["exact_duplicate_rows"] + (dup.get("duplicate_record_id") or 0)
    sc["Uniqueness"] = (1 - dup_n / len(df)) * 100
    sc["Validity"] = (1 - min(validity_df.violations_n.sum(), len(df)) / len(df)) * 100 if len(validity_df) else 100.0
    if consistency_rows:
        sc["Consistency"] = (1 - consistency_violations / consistency_rows) * 100
    if balance and balance["n_classes"] == 2:
        sc["Balance"] = balance["minority_share"] / 0.5 * 100
    t = pd.Series(sc, name="score").to_frame()
    core = t.drop(index="Balance", errors="ignore")
    overall = t.score.mean()
    t.loc["OVERALL (mean of evaluated dimensions)"] = overall
    if "Balance" in t.index:
        t.loc["OVERALL (excl. Balance)"] = core.score.mean()
    t["grade"] = t.score.apply(grade)
    t["note"] = ""
    t.loc["Validity", "note"] = "Built-in generic rules + any user-defined ranges only; domain-specific validity needs user rules."
    if "Consistency" not in t.index:
        t.loc["Consistency (not evaluated)"] = [np.nan, "Not Evaluated", "Needs a patient-ID column and demographic columns."]
    return t


SCORE_METHOD = (
    "Heuristic 0-100 score per dimension (higher = better), carried over unchanged from the Module 1 notebook. "
    "Completeness = 1 - missing cells / all cells. Uniqueness = 1 - (exact duplicate rows + duplicate record-ID values) / rows. "
    "Validity = 1 - rows violating a validity rule / rows. Consistency = 1 - patients whose demographics change between "
    "records / patients (only if a patient ID is chosen). Balance = 100 x minority-class share / 50% (binary targets only). "
    "The overall score is the plain average of the dimensions that could be evaluated; class imbalance is a modelling risk, so the "
    "score is also shown without Balance. Grades (>=90 Good, >=75 Acceptable, >=50 Poor, else Critical) are the notebook's own "
    "heuristic. This is a project-specific summary, NOT an official or regulatory data-quality score."
)


# --------------------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------------------
def build_findings(df, target, y_bin_or_none, miss, dup, outliers, validity, const, near, card, balance,
                   miss_outcome, miss_group, patient_col):
    F = []
    A = "Data quality"
    # 1 missingness (criteria = notebook severity bands)
    crit = miss[miss.severity == "Critical"] if len(miss) else miss
    high = miss[miss.severity == "High"] if len(miss) else miss
    if len(crit):
        F.append(Finding(A, "Missing values", FAILED,
                         f"{len(crit)} column(s) are >= 90% missing: " + ", ".join(f"{c} ({v:.1f}%)" for c, v in crit.missing_pct.items()),
                         "Failed if any column >= 90% missing; Warning if >= 50% (notebook severity bands).",
                         "Consider excluding these columns, or confirm with clinicians whether 'missing' means 'not measured'.", FROM_NOTEBOOK))
    elif len(high):
        F.append(Finding(A, "Missing values", WARNING,
                         f"{len(high)} column(s) are 50-90% missing: " + ", ".join(f"{c} ({v:.1f}%)" for c, v in high.missing_pct.items()),
                         "Warning if any column >= 50% missing.", "Review imputation strategy and whether missingness is informative.", FROM_NOTEBOOK))
    else:
        tot = df.isna().mean().mean() * 100
        F.append(Finding(A, "Missing values", PASSED, f"No column is >= 50% missing (overall {tot:.2f}% of cells).",
                         "Passed if no column >= 50% missing.", "", FROM_NOTEBOOK))
    # 2 informative missingness
    if miss_outcome is not None and len(miss_outcome):
        sig = miss_outcome[(miss_outcome.p_value < 0.001) & (miss_outcome.diff_pp.abs() >= 1)]
        if len(sig):
            F.append(Finding(A, "Missingness linked to outcome", WARNING,
                             f"{len(sig)} column(s) show a >=1 pp outcome-rate difference (p<0.001) when missing: " + ", ".join(sig.column.head(5)),
                             "Warning if p<0.001 and |difference| >= 1 percentage point (p-value alone is not used; large samples make tiny effects significant).",
                             "Missingness may carry information; keep a 'missing' indicator rather than silently imputing.", FROM_NOTEBOOK))
        else:
            F.append(Finding(A, "Missingness linked to outcome", PASSED, "No column's missingness shifts the outcome rate by >=1 pp at p<0.001.",
                             "As above.", "", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Missingness linked to outcome", NOT_EVALUATED, "Needs a target column and columns with >=100 missing values.", "", "", FROM_NOTEBOOK))
    if miss_group is not None and len(miss_group):
        top = miss_group.iloc[0]
        st = WARNING if top.spread_pp >= 20 else PASSED
        F.append(Finding(A, "Missingness across subgroups", st,
                         f"Largest subgroup gap: {top.column} is missing {top.spread_pp:.1f} pp more in '{top.highest_group}' than '{top.lowest_group}' ({top.attribute}).",
                         "Warning if any gap >= 20 percentage points (an audit design choice, adjustable in code).",
                         "Unequal data completeness can cause unequal model performance." if st == WARNING else "", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Missingness across subgroups", NOT_EVALUATED, "Select demographic columns (and need columns with >=100 missing values).", "", "", FROM_NOTEBOOK))
    # 3 duplicates
    if dup["exact_duplicate_rows"] > 0 or (dup.get("duplicate_record_id") or 0) > 0:
        F.append(Finding(A, "Duplicate records", FAILED,
                         f"{dup['exact_duplicate_rows']} exact duplicate rows; {dup.get('duplicate_record_id') or 0} repeated record IDs.",
                         "Failed if any exact duplicate row or repeated record ID.", "Review and remove true duplicates (the Cleaning tab can do this, logged).", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Duplicate records", PASSED, "No exact duplicate rows or repeated record IDs.", "Passed if none found.", "", FROM_NOTEBOOK))
    if dup.get("duplicate_rows_ignoring_ids", 0) > dup["exact_duplicate_rows"]:
        F.append(Finding(A, "Near-duplicates (identical features, different ID)", WARNING,
                         f"{dup['duplicate_rows_ignoring_ids']} rows repeat another row's feature values once identifier columns are ignored.",
                         "Warning if any such rows exist.", "May be legitimate (similar patients) or re-entered records; investigate.", FROM_NOTEBOOK))
    if patient_col:
        share = dup.get("rows_of_repeat_patients", 0) / len(df)
        F.append(Finding(A, "Repeated patients", WARNING if share > 0 else PASSED,
                         f"{dup.get('patients_with_multiple_rows', 0)} patients have several rows ({share:.1%} of rows).",
                         "Warning if any patient has several rows.",
                         "Split train/test by patient (not by row) to avoid leakage." if share > 0 else "", FROM_NOTEBOOK))
    # 4 outliers (flag only)
    if len(outliers):
        ext = outliers[outliers.extreme_outliers_n > 0]
        F.append(Finding(A, "Outliers (flagged, not removed)", WARNING if len(ext) else PASSED,
                         (f"{len(ext)} numeric column(s) have extreme values beyond 3xIQR: " + ", ".join(ext.index[:6])) if len(ext) else "No extreme (3xIQR) outliers.",
                         "Warning if any value lies beyond 3xIQR fences. Outliers are never auto-removed.",
                         "Have a clinician check whether extremes are real (e.g. long stays) or entry errors." if len(ext) else "", FROM_NOTEBOOK))
    # 5 validity
    if len(validity) and validity.violations_n.sum() > 0:
        F.append(Finding(A, "Invalid values", FAILED, f"{int(validity.violations_n.sum())} violations across {len(validity)} rule(s): " + "; ".join(validity.rule.head(4)),
                         "Failed if any validity rule is violated.", "Correct at source or define handling rules.", NEW))
    else:
        F.append(Finding(A, "Invalid values", PASSED, "No violations of the built-in generic rules (infinite values, mixed text/numbers) or user ranges.",
                         "Passed if none. Domain-specific rules (e.g. plausible ages) must be added by the user.", "", NEW))
    if const:
        F.append(Finding(A, "Constant columns", WARNING, f"Zero-information columns: {', '.join(const)}.", "Warning if any column has one distinct value.", "Exclude from modelling.", FROM_NOTEBOOK))
    if near:
        F.append(Finding(A, "Near-constant columns", WARNING, f"{len(near)} column(s) have >=99% one value: " + ", ".join(list(near)[:6]), "Warning if >=99% one value.", "May add noise or unstable subgroup estimates.", FROM_NOTEBOOK))
    if len(card):
        hc = card[card.n_levels >= HIGH_CARDINALITY]
        if len(hc):
            F.append(Finding(A, "High-cardinality categoricals", WARNING, f"{len(hc)} column(s) have >=50 levels: " + ", ".join(hc.column.head(5)),
                             "Warning if >= 50 levels.", "Group rare levels (e.g. ICD codes into chapters) before modelling.", FROM_NOTEBOOK))
    # 6 balance
    if balance:
        st = PASSED if balance["label"] in ("Balanced", "Mild") else WARNING if balance["label"] == "Moderate" else FAILED
        F.append(Finding(A, "Class imbalance", st,
                         f"{balance['n_classes']} classes; majority:minority = {balance['imbalance_ratio']:.1f}:1 ({balance['label']}). A model that always predicts the majority class scores {balance['majority_baseline_accuracy']:.1%} accuracy.",
                         "Balanced <1.5:1, Mild <3, Moderate <10 (Warning), Severe >=10 (Failed) - notebook bands.",
                         "Use AUPRC / recall / calibration, not accuracy alone." if st != PASSED else "", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Class imbalance", NOT_EVALUATED, "No target column selected.", "", "", FROM_NOTEBOOK))
    return F


# --------------------------------------------------------------------------------------
# Cleaning - every step is opt-in and logged. The original frame is never modified.
# --------------------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame, *, drop_exact_duplicates=False, drop_constant_columns=False,
                  drop_missing_above: float | None = None, fill_categorical_unknown=False,
                  impute_numeric_median=False, protect_columns=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (cleaned_copy, log). `protect_columns` (e.g. target, ids) are never dropped/imputed.
    Outliers are deliberately NOT touched."""
    out = df.copy(deep=True)
    log = [{"step": "0. Original data", "rows": len(out), "columns": out.shape[1], "detail": "Unmodified upload", "rows_removed": 0, "columns_removed": 0}]
    protect = set(protect_columns)

    def add(step, detail, before_rows, before_cols):
        log.append({"step": step, "rows": len(out), "columns": out.shape[1], "detail": detail,
                    "rows_removed": before_rows - len(out), "columns_removed": before_cols - out.shape[1]})

    if drop_exact_duplicates:
        r, c = len(out), out.shape[1]
        out = out.drop_duplicates()
        add("Drop exact duplicate rows", f"kept first of each duplicate group; removed {r - len(out)} rows", r, c)
    if drop_constant_columns:
        r, c = len(out), out.shape[1]
        const = [x for x in out.columns if out[x].nunique(dropna=True) <= 1 and x not in protect]
        out = out.drop(columns=const)
        add("Drop constant columns", f"dropped: {const}", r, c)
    if drop_missing_above is not None:
        r, c = len(out), out.shape[1]
        drop = [x for x in out.columns if out[x].isna().mean() >= drop_missing_above and x not in protect]
        out = out.drop(columns=drop)
        add(f"Drop columns >= {drop_missing_above:.0%} missing", f"dropped: {drop}", r, c)
    if fill_categorical_unknown:
        r, c = len(out), out.shape[1]
        filled = {}
        for x in out.columns:
            if x in protect or pd.api.types.is_numeric_dtype(out[x]) or not out[x].isna().any():
                continue
            filled[x] = int(out[x].isna().sum())
            out[x] = out[x].astype("object").where(out[x].notna(), "Unknown")
        add("Fill missing categoricals with 'Unknown'", f"cells filled: {filled}", r, c)
    if impute_numeric_median:
        r, c = len(out), out.shape[1]
        filled = {}
        for x in out.columns:
            if x in protect or not pd.api.types.is_numeric_dtype(out[x]) or not out[x].isna().any():
                continue
            filled[x] = int(out[x].isna().sum())
            out[x] = out[x].fillna(out[x].median())
        add("Impute numeric missing with median", f"cells filled: {filled}. NOTE: for model training, fit imputers on the training split only.", r, c)
    return out, pd.DataFrame(log)
