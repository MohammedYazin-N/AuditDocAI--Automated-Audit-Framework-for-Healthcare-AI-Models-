"""FEATURE 5 - subgroup (fairness) audit.

Generalised from Person 3's Module 3: representation table with MIN_GROUP_SIZE=100 /
MIN_POSITIVE_CASES=20, subgroup accuracy / precision / recall / F1 / FPR / FNR / selection rate,
Wilson 95% CIs, subgroup calibration (mean predicted vs observed, Brier), per-group calibration
curves (>=200 rows, 5 quantile bins). Added: specificity, CIs for more metrics, automatic binning of
numeric demographics, multiclass fallback, and a *statistical* (CI non-overlap) rule for flagging gaps
instead of an invented numeric threshold.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, recall_score
from .common import Finding, PASSED, WARNING, FAILED, NOT_EVALUATED, FROM_NOTEBOOK, NEW
from .stats_utils import wilson_ci, safe_divide

MIN_GROUP_SIZE = 100          # Module 3
MIN_POSITIVE_CASES = 20       # Module 3
MIN_ROWS_FOR_CURVE = 200      # Module 3 (MIN_PATIENTS_FOR_CURVE)
CURVE_BINS = 5                # Module 3

LIMITATIONS = (
    "Group metrics are computed at ONE decision threshold on ONE evaluation sample. Different fairness definitions (equal recall, equal "
    "false-positive rate, equal selection rate, calibration within groups) can conflict mathematically, so no single number defines 'fair'. "
    "An observed difference shows that performance varies between groups; it does not establish discrimination, nor explain why (differences in "
    "case mix, data quality or label definition are possible causes). Small groups give wide, unreliable intervals. Demographic columns are "
    "used as recorded (they may be incomplete or coarse), and intersections (e.g. age x race) are not analysed."
)


def make_groups(s: pd.Series, max_levels: int = 12, n_bins: int = 5) -> tuple[pd.Series, str]:
    """Return a categorical grouping of a demographic column. Numeric columns with many values are binned
    into quantile groups (documented in the note). Missing values become their own 'Missing' group."""
    note = ""
    if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > max_levels:
        g = pd.qcut(s, n_bins, duplicates="drop").astype(str)
        note = f"numeric column binned into {g.nunique()} quantile groups"
    else:
        g = s.astype("object").astype(str)
    g = g.where(s.notna(), "Missing")
    return g, note


def subgroup_table(groups: pd.Series, y_bin, yhat, p=None) -> pd.DataFrame:
    rows = []
    for gname, idx in groups.groupby(groups).groups.items():
        pos = groups.index.get_indexer(idx)
        a, h = y_bin[pos], yhat[pos]
        tp = int(((a == 1) & (h == 1)).sum()); tn = int(((a == 0) & (h == 0)).sum())
        fp = int(((a == 0) & (h == 1)).sum()); fn = int(((a == 1) & (h == 0)).sum())
        n = len(a)
        prec, rec = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
        r = {"group": gname, "n": n, "positives": tp + fn,
             "accuracy": safe_divide(tp + tn, n), "precision": prec, "recall_sensitivity": rec,
             "specificity": safe_divide(tn, tn + fp), "false_positive_rate": safe_divide(fp, fp + tn),
             "false_negative_rate": safe_divide(fn, fn + tp),
             "f1": safe_divide(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else np.nan,
             "selection_rate": float(h.mean())}
        r["recall_ci_low"], r["recall_ci_high"] = wilson_ci(tp, tp + fn)
        r["fpr_ci_low"], r["fpr_ci_high"] = wilson_ci(fp, fp + tn)
        r["accuracy_ci_low"], r["accuracy_ci_high"] = wilson_ci(tp + tn, n)
        if p is not None:
            pg = p[pos]
            r["mean_predicted_prob"] = float(pg.mean()); r["observed_rate"] = float(a.mean())
            r["calibration_gap"] = r["mean_predicted_prob"] - r["observed_rate"]   # + = over-prediction
            r["brier"] = float(np.mean((pg - a) ** 2))
        r["data_adequacy"] = "Limited - interpret cautiously" if (n < MIN_GROUP_SIZE or (tp + fn) < MIN_POSITIVE_CASES) else "Adequate for exploratory analysis"
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def multiclass_subgroup_table(groups: pd.Series, y_idx, pred_idx, n_classes) -> pd.DataFrame:
    rows = []
    for gname, idx in groups.groupby(groups).groups.items():
        pos = groups.index.get_indexer(idx)
        a, h = y_idx[pos], pred_idx[pos]
        rows.append({"group": gname, "n": len(a), "accuracy": accuracy_score(a, h),
                     "recall_macro": recall_score(a, h, average="macro", labels=list(range(n_classes)), zero_division=0),
                     "data_adequacy": "Limited - interpret cautiously" if len(a) < MIN_GROUP_SIZE else "Adequate for exploratory analysis"})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def disparity_summary(tbl: pd.DataFrame) -> pd.DataFrame:
    """Max-min gaps among ADEQUATELY sized groups + whether any pair's 95% CIs fail to overlap."""
    ad = tbl[tbl.data_adequacy.str.startswith("Adequate")]
    rows = []
    if len(ad) < 2:
        return pd.DataFrame(rows)
    for metric, lo, hi in (("recall_sensitivity", "recall_ci_low", "recall_ci_high"), ("false_positive_rate", "fpr_ci_low", "fpr_ci_high"),
                           ("selection_rate", None, None), ("accuracy", "accuracy_ci_low", "accuracy_ci_high")):
        v = ad[metric]
        hi_g, lo_g = ad.loc[v.idxmax()], ad.loc[v.idxmin()]
        sep = None
        if lo:
            sep = bool(lo_g[hi] < hi_g[lo])      # intervals do not overlap
        rows.append({"metric": metric, "highest_group": hi_g.group, "highest": v.max(), "lowest_group": lo_g.group, "lowest": v.min(),
                     "gap": v.max() - v.min(), "ratio_low_to_high": safe_divide(v.min(), v.max()), "ci_non_overlap": sep})
    return pd.DataFrame(rows)


def calibration_curves(groups, y_bin, p):
    out = {}
    for g, idx in groups.groupby(groups).groups.items():
        pos = groups.index.get_indexer(idx)
        if len(pos) < MIN_ROWS_FOR_CURVE or y_bin[pos].sum() < 5:
            continue
        try:
            o, pr = calibration_curve(y_bin[pos], p[pos], n_bins=CURVE_BINS, strategy="quantile")
            out[g] = pd.DataFrame({"mean_predicted": pr, "observed_rate": o, "n": len(pos)})
        except Exception:
            continue
    return out


def fairness_findings(attr: str, tbl: pd.DataFrame, disp: pd.DataFrame, note: str = "") -> list[Finding]:
    A = f"Fairness: {attr}"
    F = []
    lim = tbl[tbl.data_adequacy.str.startswith("Limited")]
    if len(lim):
        F.append(Finding(A, "Subgroup sample size", WARNING, f"{len(lim)} group(s) are too small for reliable estimates: " + ", ".join(f"{r.group} (n={r.n}, positives={r.positives})" for r in lim.head(6).itertuples()),
                         f"Warning if a group has < {MIN_GROUP_SIZE} rows or < {MIN_POSITIVE_CASES} positive cases (Module 3 rule).", "Collect more data for these groups; do not draw conclusions about them.", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Subgroup sample size", PASSED, "All groups meet the minimum size rules.", f">= {MIN_GROUP_SIZE} rows and >= {MIN_POSITIVE_CASES} positives per group.", "", FROM_NOTEBOOK))
    if disp.empty:
        F.append(Finding(A, "Performance differences between groups", NOT_EVALUATED, "Fewer than two adequately sized groups.", "Needs >= 2 groups meeting the size rules.", "", NEW))
    else:
        flagged = disp[disp.ci_non_overlap == True]
        if len(flagged):
            F.append(Finding(A, "Performance differences between groups", WARNING,
                             "; ".join(f"{r.metric}: {r.highest_group} {r.highest:.1%} vs {r.lowest_group} {r.lowest:.1%} (95% CIs do not overlap)" for r in flagged.itertuples()),
                             "Warning if, among adequately sized groups, the highest and lowest group have non-overlapping 95% Wilson CIs for recall, false-positive rate or accuracy. (A statistical criterion; no arbitrary gap threshold is used.)",
                             "Investigate causes (case mix, data quality, threshold) before any deployment. This is an observed disparity, not proof of discrimination.", FROM_NOTEBOOK))
        else:
            F.append(Finding(A, "Performance differences between groups", PASSED,
                             "No statistically distinguishable difference in recall, false-positive rate or accuracy between the highest and lowest adequately sized groups at this threshold.",
                             "Passed if all highest-vs-lowest 95% CIs overlap. This does NOT show the model is fair.", "", FROM_NOTEBOOK))
        sr = disp[disp.metric == "selection_rate"]
        if len(sr):
            r = sr.iloc[0]
            F.append(Finding(A, "Selection-rate difference (informational)", PASSED if r.gap < 0.05 else WARNING,
                             f"Flag rate ranges from {r.lowest:.1%} ({r.lowest_group}) to {r.highest:.1%} ({r.highest_group}); ratio {r.ratio_low_to_high:.2f}.",
                             "Informational: Warning when the absolute gap is >= 5 percentage points (a presentation choice, not a legal or clinical standard).",
                             "Different flag rates may reflect different true risk; compare with observed outcome rates.", NEW))
    return F
