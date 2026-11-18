"""FEATURE 4A - Data-leakage detection.

Adapted from Person 2's Module 2 (identifier exclusion, exact-target-copy check, statistical
association screen, prediction-time availability review, patient-level split/overlap check,
derived-feature lineage). Differences from the notebook:
  * no hard-coded column names - everything is passed in;
  * the association screen uses an EFFECT SIZE (AUC / Cramer's V). The notebook flagged p<0.001, but with
    ~100k rows nearly every feature is "significant", so p-values alone cannot separate leakage from signal;
    the p-value is still reported;
  * every result carries one of the four required categories:
      Confirmed / Potential / Suspicious / Cannot be performed.
Thresholds below are AUDIT HEURISTICS (adjustable), not standards. Passing these checks does NOT prove
the absence of leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score
from .common import Finding, PASSED, WARNING, FAILED, NOT_EVALUATED, FROM_NOTEBOOK, NEW

CONFIRMED, POTENTIAL, SUSPICIOUS, CANNOT = "Confirmed", "Potential", "Suspicious", "Cannot be performed"

DEFAULTS = {"auc_suspicious": 0.90, "auc_likely": 0.99, "cramers_suspicious": 0.50, "max_cat_levels": 100, "sample_cap": 60000}


def cramers_v(a: pd.Series, b: pd.Series):
    """Bias-corrected Cramer's V and chi-square p-value for two categorical series."""
    tbl = pd.crosstab(a.astype(str), b.astype(str))
    if tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(tbl)
    n = tbl.values.sum()
    phi2 = chi2 / n
    r, k = tbl.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return (np.sqrt(phi2c / denom) if denom > 0 else np.nan), p


def feature_target_strength(df: pd.DataFrame, target: str, features: list[str], positive_label=None, cfg=DEFAULTS) -> pd.DataFrame:
    """Single-feature predictive strength vs the target. Returns one row per feature."""
    d = df
    if len(d) > cfg["sample_cap"]:
        d = d.sample(cfg["sample_cap"], random_state=42)
    y = d[target]
    binary = y.nunique(dropna=True) == 2
    if binary:
        pos = positive_label if positive_label is not None else sorted(y.dropna().unique(), key=str)[-1]
        yb = (y.astype(str) == str(pos)).astype(int)
    rows = []
    for c in features:
        if c == target:
            continue
        s = d[c]
        try:
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 1:
                m = s.notna()
                if binary and yb[m].nunique() == 2:
                    auc = roc_auc_score(yb[m], s[m])
                    rows.append({"feature": c, "kind": "numeric", "metric": "single-feature AUC (max of AUC, 1-AUC)", "strength": max(auc, 1 - auc), "p_value": np.nan})
                    continue
                binned = pd.qcut(s[m], 10, duplicates="drop")
                v, p = cramers_v(binned, y[m])
                rows.append({"feature": c, "kind": "numeric", "metric": "Cramer's V (10 quantile bins)", "strength": v, "p_value": p})
            elif s.nunique(dropna=True) > 1 and s.nunique(dropna=True) <= cfg["max_cat_levels"]:
                v, p = cramers_v(s.astype("object").where(s.notna(), "Missing"), y)
                rows.append({"feature": c, "kind": "categorical", "metric": "Cramer's V", "strength": v, "p_value": p})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    return out.sort_values("strength", ascending=False).reset_index(drop=True) if len(out) else out


def deterministic_target_columns(df, target, features, max_card_ratio=0.2):
    """Features that are an exact copy or a one-to-one/many-to-one deterministic recoding of the target."""
    hits = []
    n = len(df)
    if n < 50:
        return hits
    for c in features:
        if c == target:
            continue
        s = df[c]
        if s.nunique(dropna=True) > max_card_ratio * n or s.nunique(dropna=True) < 2:
            continue
        g = pd.DataFrame({"f": s.astype("object").where(s.notna(), "__NA__"), "t": df[target].astype(str)})
        # target is a function of the feature in EVERY row -> the feature reveals the label
        if g.groupby("f")["t"].nunique().max() == 1:
            exact = s.astype(str).equals(df[target].astype(str))
            hits.append({"feature": c, "exact_copy": bool(exact)})
    return hits


def row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.util.hash_pandas_object(df[cols].astype("object").where(df[cols].notna(), "__NA__").astype(str), index=False)


def run_leakage_checks(df: pd.DataFrame, target: str, feature_cols: list[str], id_cols: list[str], patient_col: str | None,
                       *, positive_label=None, train_df: pd.DataFrame | None = None, split_col: str | None = None,
                       train_value=None, test_value=None, availability: dict | None = None,
                       prediction_context: str = "", cfg=DEFAULTS):
    """Returns (findings, tables) where tables holds DataFrames for display."""
    F, T = [], {}
    A = "Leakage"
    # ---------------- 1. identifier-based leakage ------------------------------------------------
    ids_in_feat = [c for c in id_cols if c in feature_cols]
    if ids_in_feat:
        F.append(Finding(A, "Identifier-based leakage", FAILED, f"Identifier column(s) are used as model inputs: {ids_in_feat}. Row/patient IDs can encode ordering, site or time and act as shortcuts.",
                         "Failed if a user-confirmed identifier column is among the model's input features.", "Remove identifiers from the feature set and retrain.", FROM_NOTEBOOK, CONFIRMED))
    elif id_cols:
        F.append(Finding(A, "Identifier-based leakage", PASSED, f"Identifier column(s) {id_cols} are not among the model's input features.",
                         "Passed if no confirmed identifier is a model input.", "", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Identifier-based leakage", NOT_EVALUATED, "No identifier columns were declared; select them on the Dataset page.", "", "", FROM_NOTEBOOK, CANNOT))
    # identifier values that predict the target (ordering effects)
    if id_cols:
        st = feature_target_strength(df, target, [c for c in id_cols if c != patient_col], positive_label, cfg)
        if len(st):
            T["identifier_strength"] = st
            bad = st[st.strength >= cfg["auc_suspicious"]]
            if len(bad):
                F.append(Finding(A, "Identifier predicts target", WARNING, "Identifier value alone predicts the target strongly: " + ", ".join(bad.feature) + ".",
                                 f"Warning if single-feature strength >= {cfg['auc_suspicious']}.", "Check whether rows are sorted by outcome or IDs encode time/site.", NEW, SUSPICIOUS))
    # ---------------- 2. target copies / deterministic recodes ------------------------------------
    feats_no_id = [c for c in feature_cols if c not in id_cols]
    det = deterministic_target_columns(df, target, feats_no_id)
    if det:
        names = [h["feature"] for h in det]
        F.append(Finding(A, "Target leakage: exact copy / deterministic recode", FAILED,
                         f"{len(det)} model input(s) fully determine the target in this data: {names[:8]}" + (" (exact copies: " + str([h['feature'] for h in det if h['exact_copy']]) + ")" if any(h['exact_copy'] for h in det) else ""),
                         "Failed if, for every value of the feature, the target takes exactly one value (and the feature is not a near-unique ID).",
                         "Remove or justify these features (clinically legitimate only if known before the prediction time).", FROM_NOTEBOOK, CONFIRMED))
    else:
        F.append(Finding(A, "Target leakage: exact copy / deterministic recode", PASSED, "No model input is an exact copy or deterministic recoding of the target.",
                         "Passed if no feature fully determines the target.", "", FROM_NOTEBOOK))
    # name-based
    tl = str(target).lower()
    name_hits = [c for c in feature_cols if c != target and len(tl) >= 3 and len(c) >= 3 and (tl in c.lower() or c.lower() in tl)]
    if name_hits:
        F.append(Finding(A, "Target-related column names", WARNING, f"Column name(s) resemble the target name: {name_hits}.", "Warning if a model input's name contains / is contained in the target name.",
                         "Confirm these are not derived from the outcome.", NEW, SUSPICIOUS))
    # ---------------- 3. suspiciously strong relationships -------------------------------------------
    st = feature_target_strength(df, target, feats_no_id, positive_label, cfg)
    if len(st):
        T["feature_strength"] = st
        sus = st[((st.metric.str.contains("AUC")) & (st.strength >= cfg["auc_suspicious"])) | ((~st.metric.str.contains("AUC")) & (st.strength >= cfg["cramers_suspicious"]))]
        if len(sus):
            F.append(Finding(A, "Suspiciously strong feature-target relationship", WARNING,
                             "Unusually strong single-feature signal: " + ", ".join(f"{r.feature} ({r.strength:.2f})" for r in sus.head(6).itertuples()),
                             f"Warning if single-feature AUC >= {cfg['auc_suspicious']} or Cramer's V >= {cfg['cramers_suspicious']} (audit heuristics, adjustable). Strong signal is not proof of leakage.",
                             "Ask: would this value be known at the prediction time? Review each flagged feature.", FROM_NOTEBOOK, SUSPICIOUS))
        else:
            top = st.iloc[0]
            F.append(Finding(A, "Suspiciously strong feature-target relationship", PASSED,
                             f"Strongest single feature: {top.feature} ({top.strength:.2f}, {top.metric}); none above the heuristic thresholds.",
                             f"Passed if all single-feature AUC < {cfg['auc_suspicious']} and Cramer's V < {cfg['cramers_suspicious']}.", "", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Suspiciously strong feature-target relationship", NOT_EVALUATED, "No analysable feature (need >=2 distinct values).", "", "", FROM_NOTEBOOK, CANNOT))
    # ---------------- 4. duplicates / near-duplicates within the evaluation data ------------------------------
    if feats_no_id:
        h = row_hashes(df, feats_no_id)
        dup_rows = int(h.duplicated().sum())
        tmp = pd.DataFrame({"h": h, "t": df[target].astype(str)})
        conflict = int((tmp.groupby("h")["t"].nunique() > 1).sum())
        T["dup_info"] = pd.DataFrame([{"rows_identical_to_an_earlier_row_(features_only)": dup_rows, "feature_vectors_with_conflicting_labels": conflict}])
        if dup_rows:
            F.append(Finding(A, "Duplicate / near-duplicate records", WARNING, f"{dup_rows} rows repeat another row's feature values (identifiers ignored); {conflict} such feature vectors carry conflicting labels.",
                             "Warning if any feature-identical rows exist.", "With a random row split, copies can land in both train and test and inflate scores. Deduplicate or group the split.", FROM_NOTEBOOK, POTENTIAL))
        else:
            F.append(Finding(A, "Duplicate / near-duplicate records", PASSED, "No feature-identical rows.", "Passed if none.", "", FROM_NOTEBOOK))
    # ---------------- 5. train-test contamination & patient overlap ----------------------------------------
    tr, te = None, None
    if train_df is not None:
        tr, te = train_df, df
        src = "uploaded training dataset"
    elif split_col and train_value is not None and test_value is not None and split_col in df.columns:
        tr = df[df[split_col].astype(str) == str(train_value)]
        te = df[df[split_col].astype(str) == str(test_value)]
        src = f"'{split_col}' column ({train_value} vs {test_value})"
    if tr is not None and len(tr) and len(te):
        common = [c for c in feats_no_id if c in tr.columns and c in te.columns]
        if patient_col and patient_col in tr.columns and patient_col in te.columns:
            ov = set(tr[patient_col].dropna()) & set(te[patient_col].dropna())
            if ov:
                F.append(Finding(A, "Patient-level overlap (train vs test)", FAILED, f"{len(ov)} patients appear in BOTH training and test data ({src}).",
                                 "Failed if any patient identifier occurs in both sets.", "Re-split by patient (e.g. StratifiedGroupKFold, as in Module 2).", FROM_NOTEBOOK, CONFIRMED))
            else:
                F.append(Finding(A, "Patient-level overlap (train vs test)", PASSED, f"No patient identifier occurs in both sets ({src}).", "Passed if the intersection of patient IDs is empty.", "", FROM_NOTEBOOK))
        else:
            F.append(Finding(A, "Patient-level overlap (train vs test)", NOT_EVALUATED, "No patient-ID column available in both sets.", "", "Provide a patient identifier to enable this check.", FROM_NOTEBOOK, CANNOT))
        if common:
            ov_rows = int(row_hashes(te, common).isin(set(row_hashes(tr, common))).sum())
            F.append(Finding(A, "Train-test contamination (identical rows)", WARNING if ov_rows else PASSED,
                             f"{ov_rows} of {len(te)} test rows have feature values identical to a training row ({src}).",
                             "Warning if any test row's full feature vector appears in the training set.",
                             "Remove duplicated rows from the test set or investigate how the split was made." if ov_rows else "", NEW, POTENTIAL if ov_rows else ""))
    else:
        F.append(Finding(A, "Train-test contamination / patient overlap", NOT_EVALUATED,
                         "No training dataset or split column was provided, so overlap between training and evaluation rows cannot be tested.",
                         "Needs a training file or a split column.", "Upload the training data (Model Upload page) or choose a split column.", NEW, CANNOT))
        if patient_col and patient_col in df.columns:
            vc = df[patient_col].value_counts()
            share = vc[vc > 1].sum() / len(df)
            if share > 0:
                F.append(Finding(A, "Repeated patients (split risk)", WARNING, f"{share:.1%} of rows belong to patients with several rows. A random row-level split would put the same patient in train and test.",
                                 "Warning if any patient has several rows and the split cannot be verified.", "Confirm the split was done by patient.", FROM_NOTEBOOK, POTENTIAL))
    # ---------------- 6. prediction-time availability -------------------------------------------------------
    if availability:
        after = [f for f, v in availability.items() if v == "After prediction time" and f in feature_cols]
        unk = [f for f in feature_cols if availability.get(f, "Not reviewed") in ("Not reviewed", "Unsure")]
        T["availability"] = pd.DataFrame({"feature": feature_cols, "availability": [availability.get(f, "Not reviewed") for f in feature_cols]})
        if after:
            F.append(Finding(A, "Prediction-time availability", FAILED, f"{len(after)} model input(s) were marked as only known AFTER the prediction time ({prediction_context or 'context not described'}): {after[:8]}",
                             "Failed if the reviewer marks a model input as recorded after the prediction time.", "Remove them or move the prediction time later.", FROM_NOTEBOOK, CONFIRMED))
        elif unk:
            F.append(Finding(A, "Prediction-time availability", WARNING if len(unk) < len(feature_cols) else NOT_EVALUATED,
                             f"{len(unk)} of {len(feature_cols)} features are not yet reviewed for availability at the prediction time.",
                             "Warning/Not Evaluated while features remain 'Not reviewed' or 'Unsure' (needs clinical/data-owner knowledge).",
                             "Review each feature with a clinician / data owner (the Module 2 notebook did this manually and noted exact timestamps were unavailable).", FROM_NOTEBOOK, SUSPICIOUS if len(unk) < len(feature_cols) else CANNOT))
        else:
            F.append(Finding(A, "Prediction-time availability", PASSED, "All model inputs were reviewed and marked as available at/before the prediction time (reviewer's judgement; not verifiable from the data).",
                             "Passed if every input is marked 'Before/at prediction time'.", "", FROM_NOTEBOOK))
    else:
        F.append(Finding(A, "Prediction-time availability", NOT_EVALUATED, "Prediction time / feature availability has not been defined.", "Needs the user to describe the prediction context and tag features.",
                         "Use the table on this page to tag each feature.", FROM_NOTEBOOK, CANNOT))
    # ---------------- overall caveat -----------------------------------------------------------------------
    F.append(Finding(A, "Overall leakage conclusion", WARNING,
                     "Automated checks can find some leakage patterns but cannot prove its absence (e.g. subtle temporal leakage, leakage through preprocessing fitted on all data, or features derived from the outcome).",
                     "Always Warning by design.", "Confirm preprocessing was fit on training data only and review feature timing with domain experts.", NEW, POTENTIAL))
    return F, T
