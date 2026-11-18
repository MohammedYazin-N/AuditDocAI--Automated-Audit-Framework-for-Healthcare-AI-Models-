"""Glue layer: one function per audit section, used by the Streamlit pages, the tests and the report.

Each function takes plain Python/pandas inputs and returns a dict of results, so the same code runs
in the app and headless. Nothing here touches Streamlit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import data_io, data_quality as dq, leakage as lk, performance as perf, fairness as fr, explain as ex, model_io
from .common import Finding, NOT_EVALUATED, NEW


def run_data_quality(df, target, positive_label, id_cols, patient_col, record_id_col, demographics, range_rules=None) -> dict:
    roles = data_io.column_roles(df)
    num = [c for c in roles["numeric"] if c not in id_cols and c != target]
    cat = [c for c in roles["categorical"] if c not in id_cols and c != target]
    y = df[target] if target else None
    y_bin = None
    balance = None
    if target:
        balance = dq.class_balance(y)
        if y.nunique(dropna=True) == 2:
            pos = positive_label if positive_label is not None else sorted(y.dropna().unique(), key=str)[-1]
            y_bin = (y.astype(str) == str(pos)).astype(int)
    miss = dq.missing_table(df)
    dup = dq.duplicate_summary(df, record_id_col, patient_col)
    out = dq.outlier_table(df, num)
    val = dq.invalid_value_checks(df, num, range_rules)
    const, near = dq.constant_columns(df)
    card = dq.cardinality_table(df, cat)
    miss_out = dq.missing_vs_outcome(df, y_bin) if y_bin is not None else None
    miss_grp, miss_grp_tables = dq.missing_by_group(df, demographics) if demographics else (pd.DataFrame(), {})
    pairs, corr = dq.correlation_pairs(df, num, y_bin)
    rep = dq.subgroup_representation(df, y_bin, demographics) if (y_bin is not None and demographics) else pd.DataFrame()
    cons_rows = cons_viol = None
    cons_tbl = pd.DataFrame()
    if patient_col and demographics:
        cons_tbl = dq.patient_demographic_consistency(df, patient_col, demographics)
        if len(cons_tbl):
            cons_rows = df[patient_col].nunique()
            cons_viol = int(cons_tbl.patients_with_changing_value.max())
    score = dq.scorecard(df, dup, val, balance, cons_rows, cons_viol)
    findings = dq.build_findings(df, target, y_bin, miss, dup, out, val, const, near, card, balance, miss_out, miss_grp, patient_col)
    if len(cons_tbl) and cons_viol:
        findings.append(Finding("Data quality", "Demographic consistency across a patient's records", "Warning",
                                f"{cons_viol} patients have a changing value in at least one demographic column.", "Warning if any patient's demographic value differs between records.",
                                "Verify demographic recording; affects subgroup (fairness) analysis.", dq.FROM_NOTEBOOK))
    return {"findings": findings, "missing": miss, "duplicates": dup, "outliers": out, "validity": val, "constant": const, "near_constant": near,
            "cardinality": card, "balance": balance, "missing_vs_outcome": miss_out, "missing_by_group": miss_grp, "missing_by_group_tables": miss_grp_tables,
            "corr_pairs": pairs, "corr_matrix": corr, "representation": rep, "scorecard": score, "score_method": dq.SCORE_METHOD,
            "roles": roles, "consistency": cons_tbl}


def run_leakage(df_eval, target, feature_cols, id_cols, patient_col, positive_label, train_df=None, split_col=None, train_value=None,
                test_value=None, availability=None, prediction_context="") -> dict:
    F, T = lk.run_leakage_checks(df_eval, target, feature_cols, id_cols, patient_col, positive_label=positive_label, train_df=train_df,
                                 split_col=split_col, train_value=train_value, test_value=test_value, availability=availability,
                                 prediction_context=prediction_context)
    return {"findings": F, "tables": T}


def run_performance(model, mi, df_eval, target, feat_cols, positive_label, provenance, threshold) -> dict:
    proba, pred = model_io.predict_all(model, mi, df_eval, feat_cols)
    classes = mi.classes or sorted(pd.unique(pd.Series(pred)))
    lab = perf.prepare_labels(df_eval[target], classes, positive_label)
    res = {"kind": lab["kind"], "classes": lab["classes"], "pos_idx": lab["pos_idx"], "y_idx": lab["y_idx"], "provenance": provenance, "threshold": threshold}
    cls_str = lab["classes"]
    pred_idx = pd.Series(pred.astype(str)).map({c: i for i, c in enumerate(cls_str)}).to_numpy()
    if np.isnan(pred_idx.astype(float)).any():
        raise ValueError("The model predicted labels that are not among its classes.")
    pred_idx = pred_idx.astype(int)
    res["proba"] = proba
    if lab["kind"] == "binary":
        res["metrics"] = perf.binary_metrics(lab["y_idx"], cls_str, proba, lab["pos_idx"], threshold, has_proba=proba is not None, pred_labels=pred_idx)
        if proba is not None and "probabilities" in res["metrics"]:
            res["threshold_table"] = perf.threshold_table(res["metrics"]["y_binary"], res["metrics"]["probabilities"])
            yhat = (proba[:, lab["pos_idx"]] >= threshold).astype(int)
        else:
            yhat = (pred_idx == lab["pos_idx"]).astype(int)
        res["yhat_binary"] = yhat
    else:
        res["metrics"] = perf.multiclass_metrics(lab["y_idx"], cls_str, proba, pred_idx)
        res["pred_idx"] = pred_idx
    res["findings"] = perf.performance_findings(lab["kind"], res["metrics"], provenance, threshold, proba is not None)
    return res


def run_fairness(perf_res, df_eval, demographics) -> dict:
    out = {"attributes": {}, "findings": [], "limitations": fr.LIMITATIONS}
    kind = perf_res["kind"]
    for attr in demographics:
        if attr not in df_eval.columns:
            out["findings"].append(Finding(f"Fairness: {attr}", "Demographic column", NOT_EVALUATED, f"Column '{attr}' not found.", "", "", NEW))
            continue
        g, note = fr.make_groups(df_eval[attr].reset_index(drop=True))
        if g.nunique() < 2:
            out["findings"].append(Finding(f"Fairness: {attr}", "Subgroups", NOT_EVALUATED, "Only one group present.", "Needs >= 2 groups.", "", NEW))
            continue
        if kind == "binary":
            m = perf_res["metrics"]
            yb = (perf_res["y_idx"] == perf_res["pos_idx"]).astype(int)
            p = perf_res["proba"][:, perf_res["pos_idx"]] if perf_res["proba"] is not None else None
            tbl = fr.subgroup_table(g, yb, perf_res["yhat_binary"], p)
            disp = fr.disparity_summary(tbl)
            curves = fr.calibration_curves(g, yb, p) if p is not None else {}
            n_flag, n_pos = int(perf_res["yhat_binary"].sum()), int(yb.sum())
            if n_flag == 0 or n_pos == 0 or n_flag == len(yb):
                why = ("The model flags NO rows at the chosen threshold" if n_flag == 0 else
                       "The model flags EVERY row at the chosen threshold" if n_flag == len(yb) else "The evaluation rows contain no positive cases")
                F = [Finding(f"Fairness: {attr}", "Group error-rate comparison", NOT_EVALUATED,
                             f"{why}, so recall, false-positive rate and selection rate are degenerate and cannot be compared across groups.",
                             "Not Evaluated if the threshold flags none/all rows or there are no positives.",
                             "Choose a clinically justified threshold on a validation split (Performance page) and re-run.", NEW)]
                disp = pd.DataFrame()
            else:
                F = fr.fairness_findings(attr, tbl, disp, note)
            if note:
                F[0].detail += f" (Note: {note}.)"
        else:
            tbl = fr.multiclass_subgroup_table(g, perf_res["y_idx"], perf_res["pred_idx"], len(perf_res["classes"]))
            disp, curves = pd.DataFrame(), {}
            F = [Finding(f"Fairness: {attr}", "Multiclass subgroup performance", NOT_EVALUATED if False else "Warning",
                         "Accuracy and macro-recall by group computed; FPR/FNR, CIs and calibration are binary-only and were not evaluated.", "Descriptive only.", "Inspect gaps manually.", NEW)]
        out["attributes"][attr] = {"table": tbl, "disparity": disp, "curves": curves, "note": note}
        out["findings"] += F
    return out


def run_explainability(model, mi, df_eval, feat_cols, y_idx, classes, pos_idx, do_permutation=True, n_shap=200) -> dict:
    X = df_eval[feat_cols]
    res = {"findings": [], "notes": []}
    native, why = ex.native_importance(model, list(feat_cols))
    res["native"] = native
    if native is None:
        res["notes"].append(why)
    shap_res = ex.shap_explain(model, X, pos_idx, n_rows=n_shap)
    res["shap"] = shap_res
    if "error" in shap_res:
        res["notes"].append(shap_res["error"])
    if do_permutation:
        try:
            res["permutation"] = ex.permutation_importance_table(model, X, y_idx, classes, pos_idx)
        except Exception as e:
            res["permutation"] = None
            res["notes"].append(f"Permutation importance failed: {type(e).__name__}: {str(e)[:200]}")
    has = native is not None or "error" not in shap_res or res.get("permutation") is not None
    A = "Explainability"
    if "error" not in shap_res:
        top = ", ".join(shap_res["importance"].feature.head(3))
        res["findings"].append(Finding(A, "SHAP", "Passed", f"SHAP computed ({shap_res['method']}). Top features: {top}.", "Passed if SHAP values were computed.", "", "Reused/generalised from notebook"))
    else:
        res["findings"].append(Finding(A, "SHAP", NOT_EVALUATED, shap_res["error"], "", "Alternative methods shown where available.", "Reused/generalised from notebook"))
    if res.get("permutation") is not None:
        p = res["permutation"]
        res["findings"].append(Finding(A, "Permutation importance", "Passed", f"Top inputs: {', '.join(p.feature.head(3))} (score = {p.attrs.get('scoring')}, {p.attrs.get('n_rows')} rows).", "Passed if computed.", "", NEW))
    if not has:
        res["findings"].append(Finding(A, "Explainability", NOT_EVALUATED, "No supported explanation method for this model.", "", "", NEW))
    return res
