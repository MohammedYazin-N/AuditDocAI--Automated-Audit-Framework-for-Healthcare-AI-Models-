"""FEATURE 4B - model performance + calibration.

Metric definitions follow Module 2 (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier,
log-loss, quantile-bin calibration curve, "mean calibration error", prevalence-baseline Brier,
validation-style threshold table). Added: specificity, Wilson CIs, expected calibration error,
multiclass support, and the provenance labelling the brief requires.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss, confusion_matrix,
                             roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from .common import Finding, PASSED, WARNING, FAILED, NOT_EVALUATED, FROM_NOTEBOOK, NEW
from .stats_utils import wilson_ci, safe_divide

PROVENANCE = {
    "independent": "Independent test data (never used for training, tuning or threshold selection)",
    "training": "Training data (the model has seen these rows)",
    "unknown": "Unknown provenance (user cannot confirm how these rows relate to training)",
}
# Audit design choice (adjustable): AUROC at/above this on real clinical data deserves a leakage review.
SUSPICIOUS_AUC = 0.95


def provenance_banner(kind: str) -> tuple[str, str]:
    """(level, text) for st.success/warning/error."""
    if kind == "independent":
        return "success", ("Metrics below are reported as **independent test performance**, based on your statement that these rows were never "
                           "used for training/tuning. The app cannot verify this claim.")
    if kind == "training":
        return "error", ("These rows were used for **training**. The metrics below are **training performance, NOT independent test performance** "
                         "and will usually look better than real-world performance. Do not use them to judge the model's readiness.")
    return "warning", ("Provenance is **unknown**. Metrics are descriptive only; they must not be presented as independent validation until you can "
                       "confirm the rows were held out from training.")


def prepare_labels(y: pd.Series, classes: list, positive_label=None):
    """Align the target with model.classes_. Returns dict(kind, y_idx, classes_str, pos_idx) or raises ValueError."""
    cls_str = [str(c) for c in classes]
    ys = y.astype(str)
    unknown = sorted(set(ys.dropna().unique()) - set(cls_str))
    if unknown:
        raise ValueError(f"Target values {unknown[:6]} do not match the model's classes {cls_str}.")
    idx_map = {c: i for i, c in enumerate(cls_str)}
    y_idx = ys.map(idx_map)
    if y_idx.isna().any():
        raise ValueError("The target column contains missing values in the evaluation rows; remove or fix them first.")
    kind = "binary" if len(cls_str) == 2 else "multiclass"
    pos_idx = None
    if kind == "binary":
        pos = str(positive_label) if positive_label is not None else cls_str[1]
        if pos not in idx_map:
            raise ValueError(f"Positive label '{pos}' is not one of the model classes {cls_str}.")
        pos_idx = idx_map[pos]
    return {"kind": kind, "y_idx": y_idx.astype(int).to_numpy(), "classes": cls_str, "pos_idx": pos_idx}


def _cal_curve(y, p, n_bins=10):
    a, b = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": b, "observed_rate": a})


def expected_calibration_error(y, p, n_bins=10):
    """ECE with equal-width bins, weighted by bin size."""
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def binary_metrics(y_true_idx, labels, proba, pos_idx, threshold=0.5, has_proba=True, pred_labels=None):
    """All binary metrics. `y_true_idx` are class indices; positive = pos_idx."""
    yb = (y_true_idx == pos_idx).astype(int)
    if has_proba:
        p = proba[:, pos_idx]
        yhat = (p >= threshold).astype(int)
    else:
        yhat = (pred_labels == pos_idx).astype(int)
        p = None
    tn, fp, fn, tp = confusion_matrix(yb, yhat, labels=[0, 1]).ravel()
    res = {"n": int(len(yb)), "positives": int(yb.sum()), "prevalence": float(yb.mean()), "threshold": threshold if has_proba else None}
    res["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    rows = []

    def add(name, val, ci=None, note=""):
        rows.append({"metric": name, "value": val, "ci_low": ci[0] if ci else np.nan, "ci_high": ci[1] if ci else np.nan, "note": note})

    add("Accuracy", accuracy_score(yb, yhat), wilson_ci(tp + tn, len(yb)), "Misleading when classes are imbalanced")
    add("Precision (PPV)", safe_divide(tp, tp + fp), wilson_ci(tp, tp + fp))
    add("Recall / Sensitivity", safe_divide(tp, tp + fn), wilson_ci(tp, tp + fn))
    add("Specificity", safe_divide(tn, tn + fp), wilson_ci(tn, tn + fp))
    prec, rec = safe_divide(tp, tp + fp), safe_divide(tp, tp + fn)
    add("F1-score", safe_divide(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else np.nan)
    if has_proba and 0 < yb.sum() < len(yb):
        res["roc_auc"] = float(roc_auc_score(yb, p))
        res["pr_auc"] = float(average_precision_score(yb, p))
        res["brier"] = float(brier_score_loss(yb, p))
        res["brier_baseline"] = float(brier_score_loss(yb, np.full(len(yb), yb.mean())))
        try:
            res["log_loss"] = float(log_loss(yb, np.clip(p, 1e-15, 1 - 1e-15)))
        except Exception:
            res["log_loss"] = np.nan
        add("ROC-AUC", res["roc_auc"], note="Threshold-free discrimination")
        add("PR-AUC (average precision)", res["pr_auc"], note=f"Chance level = prevalence = {yb.mean():.3f}")
        add("Brier score", res["brier"], note=f"Lower is better; predicting the prevalence for everyone scores {res['brier_baseline']:.4f}")
        cc = _cal_curve(yb, p)
        res["calibration_curve"] = cc
        res["mean_calibration_error"] = float((cc.observed_rate - cc.mean_predicted).abs().mean())   # notebook definition
        res["ece"] = expected_calibration_error(yb, p)
        add("Mean calibration error (quantile bins)", res["mean_calibration_error"], note="Module 2 definition: mean |observed - predicted| over 10 quantile bins")
        add("Expected calibration error (ECE)", res["ece"], note="Equal-width 10 bins, weighted by bin size")
        fpr, tpr, _ = roc_curve(yb, p)
        pr, rc, _ = precision_recall_curve(yb, p)
        res["roc_curve"] = pd.DataFrame({"fpr": fpr, "tpr": tpr})
        res["pr_curve"] = pd.DataFrame({"recall": rc, "precision": pr})
        res["probabilities"] = p
        res["y_binary"] = yb
    elif has_proba:
        res["note_single_class"] = "Only one class present in the evaluation rows - AUC/Brier-based metrics are undefined."
    res["table"] = pd.DataFrame(rows)
    return res


def threshold_table(yb, p, lo=0.10, hi=0.80, step=0.05):
    """Module 2 cell 'THRESHOLD OPTIMISATION' style table. DESCRIPTIVE ONLY here - picking a threshold on the
    same rows you report would be optimistic; the notebook chose it on a separate validation split."""
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        yhat = (p >= t).astype(int)
        rows.append({"threshold": round(float(t), 2), "precision": precision_score(yb, yhat, zero_division=0),
                     "recall": recall_score(yb, yhat, zero_division=0), "f1": f1_score(yb, yhat, zero_division=0),
                     "flagged_pct": float(yhat.mean() * 100)})
    return pd.DataFrame(rows)


def multiclass_metrics(y_true_idx, classes, proba, pred_idx):
    n = len(y_true_idx)
    labels = list(range(len(classes)))
    res = {"n": int(n), "classes": classes}
    rows = [{"metric": "Accuracy", "value": accuracy_score(y_true_idx, pred_idx), "note": ""}]
    for avg in ("macro", "weighted"):
        rows.append({"metric": f"Precision ({avg})", "value": precision_score(y_true_idx, pred_idx, average=avg, labels=labels, zero_division=0), "note": ""})
        rows.append({"metric": f"Recall ({avg})", "value": recall_score(y_true_idx, pred_idx, average=avg, labels=labels, zero_division=0), "note": ""})
        rows.append({"metric": f"F1 ({avg})", "value": f1_score(y_true_idx, pred_idx, average=avg, labels=labels, zero_division=0), "note": ""})
    if proba is not None and proba.shape[1] == len(classes):
        try:
            rows.append({"metric": "ROC-AUC (one-vs-rest, macro)", "value": roc_auc_score(y_true_idx, proba, multi_class="ovr", average="macro", labels=labels), "note": ""})
            rows.append({"metric": "Log loss", "value": log_loss(y_true_idx, proba, labels=labels), "note": ""})
        except Exception as e:
            rows.append({"metric": "ROC-AUC (OvR macro)", "value": np.nan, "note": f"not computable: {e}"})
    cm = confusion_matrix(y_true_idx, pred_idx, labels=labels)
    res["confusion_matrix"] = pd.DataFrame(cm, index=[f"true {c}" for c in classes], columns=[f"pred {c}" for c in classes])
    # per-class sensitivity / specificity
    per = []
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        per.append({"class": c, "support": int(cm[i].sum()), "sensitivity": safe_divide(tp, tp + fn), "specificity": safe_divide(tn, tn + fp)})
    res["per_class"] = pd.DataFrame(per)
    res["table"] = pd.DataFrame(rows)
    return res


def performance_findings(kind, res, provenance, threshold, has_proba) -> list[Finding]:
    A = "Performance"
    F = []
    label = {"independent": "independent test data", "training": "TRAINING data (not independent)", "unknown": "data of unknown provenance"}[provenance]
    if provenance == "training":
        F.append(Finding(A, "Evaluation data provenance", FAILED, "Evaluation was run on training data; results are not independent test performance.",
                         "Failed if the user states the rows were used for training.", "Evaluate on a patient-level held-out test set.", NEW))
    elif provenance == "unknown":
        F.append(Finding(A, "Evaluation data provenance", WARNING, "Provenance of evaluation rows is unknown.", "Warning if the user cannot confirm the rows were held out.",
                         "Document how the evaluation rows were separated from training data.", NEW))
    else:
        F.append(Finding(A, "Evaluation data provenance", PASSED, "User states the rows are independent of training (not verifiable by the app).", "Passed if user declares independent test data.", "", NEW))
    if kind == "binary":
        t = res["table"].set_index("metric").value
        F.append(Finding(A, "Discrimination (ROC-AUC / PR-AUC)", NOT_EVALUATED if "roc_auc" not in res else (WARNING if res["roc_auc"] < 0.7 else PASSED),
                         "Requires predict_proba and both classes present." if "roc_auc" not in res else
                         f"ROC-AUC {res['roc_auc']:.3f}, PR-AUC {res['pr_auc']:.3f} vs chance {res['prevalence']:.3f} on {label}.",
                         "Warning if ROC-AUC < 0.70 (a conventional 'poor-to-modest discrimination' rule of thumb, not a clinical standard).",
                         "Moderate discrimination is not enough for clinical use without further validation." if "roc_auc" in res and res["roc_auc"] < 0.7 else "", FROM_NOTEBOOK))
        if "roc_auc" in res:
            pa = res["roc_auc"] >= SUSPICIOUS_AUC
            F.append(Finding("Leakage", "Suspiciously high performance", WARNING if pa else PASSED,
                             f"ROC-AUC {res['roc_auc']:.3f} " + ("is unusually high for a clinical prediction task." if pa else "is below the suspicion level."),
                             f"Warning if ROC-AUC >= {SUSPICIOUS_AUC} (audit heuristic - adjustable; a high score is a reason to look for leakage, not proof of it).",
                             "Inspect top features and the split strategy for leakage." if pa else "", NEW, "Suspicious" if pa else ""))
            better = res["brier"] < res["brier_baseline"]
            F.append(Finding(A, "Calibration (Brier / calibration error)", PASSED if (better and res["mean_calibration_error"] < 0.05) else WARNING,
                             f"Brier {res['brier']:.4f} (prevalence baseline {res['brier_baseline']:.4f}); mean calibration error {res['mean_calibration_error']:.3f}; ECE {res['ece']:.3f}.",
                             "Passed if Brier beats the prevalence baseline AND mean calibration error < 0.05 (audit design choice, adjustable).",
                             "Recalibrate (e.g. sigmoid/isotonic on a validation split) if probabilities will inform decisions." if not better or res['mean_calibration_error'] >= .05 else "", FROM_NOTEBOOK))
        else:
            F.append(Finding(A, "Calibration", NOT_EVALUATED, "Needs predict_proba.", "", "", FROM_NOTEBOOK))
        if res["confusion"]["tp"] + res["confusion"]["fp"] == 0:
            F.append(Finding(A, f"Operating point (threshold {threshold if has_proba else 'n/a'})", WARNING,
                             "The model flags no cases at this threshold (recall 0%). Predicted probabilities may all be below it - common for well-calibrated models of rare outcomes.",
                             "Warning if no row is predicted positive.", "Select a threshold on a separate validation split, based on the clinical cost of misses vs false alarms.", FROM_NOTEBOOK))
            return F
        rec = t["Recall / Sensitivity"]; pr = t["Precision (PPV)"]
        F.append(Finding(A, f"Operating point (threshold {threshold if has_proba else 'n/a'})", WARNING if (np.isnan(pr) or pr < 0.5 or np.isnan(rec) or rec < 0.5) else PASSED,
                         f"Recall {rec:.1%}, precision {pr:.1%}, accuracy {t['Accuracy']:.1%} (majority-class accuracy would be {1 - res['prevalence']:.1%}).",
                         "Warning if recall or precision < 50% at the chosen threshold (rule of thumb for 'more misses/false alarms than hits').",
                         "Choose the threshold on a separate validation split according to clinical cost of misses vs false alarms.", FROM_NOTEBOOK))
    else:
        t = res["table"].set_index("metric").value
        F.append(Finding(A, "Multiclass performance", PASSED, f"Accuracy {t['Accuracy']:.3f}; macro-F1 {t['F1 (macro)']:.3f} on {label}.", "Descriptive; no pass/fail threshold is invented for multiclass.", "", NEW))
    return F
