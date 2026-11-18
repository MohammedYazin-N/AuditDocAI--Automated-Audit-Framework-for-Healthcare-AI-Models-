"""FEATURE 6 - explainability.

Mirrors Module 3's SHAP approach: for a calibrated model, SHAP is computed on the underlying fitted
estimator of the FIRST calibration fold (calibrated_classifiers_[0]) using a random sample of rows
(default 200), and this is stated in the output (it explains the raw forest score, not the final
calibrated probability). Added: automatic support for plain Pipelines / linear models, and a
model-agnostic permutation-importance fallback. If no method applies, the function says so - nothing
is invented.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from scipy import sparse


def _parts(model):
    """Return (preprocessor_or_None, final_estimator, note)."""
    note = ""
    est = model
    if type(est).__name__ == "CalibratedClassifierCV":
        try:
            est = est.calibrated_classifiers_[0].estimator
            note = ("The model is a CalibratedClassifierCV. Explanations are computed on the underlying estimator from the FIRST calibration fold, "
                    "so they approximate (but are not identical to) what drives the final calibrated probabilities.")
        except Exception:
            return None, None, "Could not unwrap the calibrated model."
    if isinstance(est, Pipeline):
        pre = Pipeline(est.steps[:-1]) if len(est.steps) > 1 else None
        return pre, est.steps[-1][1], note
    return None, est, note


def _feature_names(pre, n, raw_cols):
    if pre is not None:
        try:
            return [str(x) for x in pre.get_feature_names_out()]
        except Exception:
            pass
    if raw_cols is not None and len(raw_cols) == n:
        return list(raw_cols)
    return [f"feature_{i}" for i in range(n)]


def native_importance(model, raw_cols):
    """Model-specific importance (tree feature_importances_ / linear |coef|) on the transformed feature space."""
    pre, final, note = _parts(model)
    if final is None:
        return None, "No inspectable final estimator."
    if hasattr(final, "feature_importances_"):
        imp, kind = np.asarray(final.feature_importances_), "Tree impurity-based feature importance"
    elif hasattr(final, "coef_"):
        c = np.asarray(final.coef_)
        imp, kind = (np.abs(c).mean(axis=0) if c.ndim > 1 else np.abs(c)), "Absolute linear coefficient (features are on their own scale unless standardised)"
    else:
        return None, f"{type(final).__name__} exposes no native importance."
    names = _feature_names(pre, len(imp), raw_cols)
    df = pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
    df.attrs["kind"], df.attrs["note"] = kind, note
    return df, ""


def permutation_importance_table(model, X: pd.DataFrame, y_idx, classes, pos_idx, n_rows=1500, n_repeats=3, seed=42):
    """Model-agnostic: drop in score when one RAW input column is shuffled. Works for any model with predict()."""
    rng = np.random.default_rng(seed)
    take = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    Xs, ys = X.iloc[take], np.asarray(y_idx)[take]
    scoring = "roc_auc" if (len(classes) == 2 and hasattr(model, "predict_proba")) else "accuracy"
    # sklearn scorers need y in the model's own label space
    ylab = np.asarray(classes, dtype=object)[ys]
    try:
        ylab = ylab.astype(np.asarray(model.classes_).dtype)
    except Exception:
        pass
    if scoring == "roc_auc":
        # roc_auc scorer treats classes_[1] as positive -> only valid if pos_idx==1; otherwise use accuracy
        if pos_idx != 1 or len(np.unique(ys)) < 2:
            scoring = "accuracy"
    r = permutation_importance(model, Xs, ylab, scoring=scoring, n_repeats=n_repeats, random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std}).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.attrs["scoring"], out.attrs["n_rows"] = scoring, len(Xs)
    return out


def shap_explain(model, X: pd.DataFrame, pos_idx: int | None, n_rows=200, seed=42):
    """Returns dict with shap values for the positive class, or {'error': reason}. Never fabricates output."""
    try:
        import shap
    except ImportError:
        return {"error": "The 'shap' package is not installed."}
    pre, final, note = _parts(model)
    if final is None:
        return {"error": "No inspectable final estimator."}
    fam = type(final).__name__
    sample = X.sample(n=min(n_rows, len(X)), random_state=seed)
    try:
        data = pre.transform(sample) if pre is not None else sample
        if sparse.issparse(data):
            data = data.toarray()
        data = np.asarray(data, dtype=float)
    except Exception as e:
        return {"error": f"Could not transform the sample for SHAP ({type(e).__name__}: {e})."}
    names = _feature_names(pre, data.shape[1], list(sample.columns))
    try:
        if any(t in fam for t in ("RandomForest", "GradientBoosting", "DecisionTree", "ExtraTrees", "XGB", "LGBM", "CatBoost", "HistGradientBoosting")):
            expl = shap.TreeExplainer(final)
            sv = expl.shap_values(data)
            method = "SHAP TreeExplainer"
        elif hasattr(final, "coef_"):
            bg = data[: min(100, len(data))]
            expl = shap.LinearExplainer(final, bg)
            sv = expl.shap_values(data)
            method = "SHAP LinearExplainer"
        else:
            return {"error": f"SHAP is not set up for {fam} here (tree and linear models are supported). Permutation importance is used instead."}
    except Exception as e:
        return {"error": f"SHAP failed for {fam} ({type(e).__name__}: {str(e)[:200]})."}
    cls = 1 if pos_idx is None else pos_idx
    if isinstance(sv, list):
        arr = np.asarray(sv[cls] if len(sv) > cls else sv[-1])
    else:
        arr = np.asarray(sv)
        if arr.ndim == 3:
            arr = arr[:, :, cls if arr.shape[2] > cls else -1]
    if arr.shape != data.shape:
        return {"error": f"Unexpected SHAP output shape {arr.shape} vs data {data.shape}."}
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(arr).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return {"method": method, "values": arr, "data": data, "feature_names": names, "importance": imp, "sample_index": sample.index.to_numpy(),
            "note": (note + " " if note else "") + "SHAP shows which inputs influence the model's output; it does not show what causes the outcome."}


def local_explanation(shap_res: dict, position: int, top_n=12) -> pd.DataFrame:
    v, d = shap_res["values"][position], shap_res["data"][position]
    df = pd.DataFrame({"feature": shap_res["feature_names"], "feature_value": d, "shap_value": v})
    df["abs"] = df.shap_value.abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").head(top_n).reset_index(drop=True)


def summary_plot_figure(shap_res: dict, max_display=20, dark=True):
    """SHAP beeswarm as a matplotlib figure. dark=True gives a transparent background with light text for the dashboard theme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    rc = {"text.color": "#E8EEFA", "axes.labelcolor": "#E8EEFA", "xtick.color": "#E8EEFA", "ytick.color": "#E8EEFA",
          "axes.edgecolor": "#24324F", "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none"} if dark else {}
    with plt.rc_context(rc):
        plt.figure()
        shap.summary_plot(shap_res["values"], shap_res["data"], feature_names=shap_res["feature_names"], max_display=max_display, show=False)
        fig = plt.gcf()
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.patch.set_alpha(0)
        fig.tight_layout()
    return fig
