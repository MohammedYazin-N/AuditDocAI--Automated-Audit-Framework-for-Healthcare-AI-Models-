"""FEATURE 2 - model loading and compatibility validation.

SECURITY: pickle/joblib files can run arbitrary code when loaded. `load_model` must only be
called after the user has explicitly confirmed that they trust the file (the UI enforces this).
There is NO reliable automatic scan for malicious pickles, so none is claimed.
"""
from __future__ import annotations
import io
import pickle
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

SECURITY_WARNING = (
    "Pickle and joblib files can execute arbitrary code when they are loaded. Only upload model files "
    "that you created yourself or that come from a source you fully trust. This app cannot verify that a file is safe."
)


def load_model(raw_bytes: bytes, filename: str):
    """Load a model. Returns (model, load_warnings). Raises ValueError with a readable message."""
    name = filename.lower()
    if not name.endswith((".pkl", ".pickle", ".joblib")):
        raise ValueError("Unsupported model file. Please upload a .pkl or .joblib file.")
    caught = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            if name.endswith(".joblib"):
                import joblib
                model = joblib.load(io.BytesIO(raw_bytes))
            else:
                try:
                    model = pickle.load(io.BytesIO(raw_bytes))
                except Exception:
                    import joblib  # some .pkl files are written with joblib.dump
                    model = joblib.load(io.BytesIO(raw_bytes))
            caught = [str(x.message) for x in w]
    except ModuleNotFoundError as e:
        raise ValueError(f"The model needs a Python package that is not installed here: '{e.name}'. "
                         f"Install it (see requirements.txt) and retry. Original error: {e}")
    except Exception as e:
        hint = ""
        if "sklearn" in str(e) or isinstance(e, (AttributeError, ImportError, ModuleNotFoundError)):
            try:
                import sklearn
                here = sklearn.__version__
            except Exception:
                here = "unknown"
            hint = (f"\n\n**Most likely a library-version mismatch.** This app runs scikit-learn {here}. Install the version the model was trained with "
                    "(in the training notebook run `import sklearn; print(sklearn.__version__)`, then in the app environment run "
                    "`python -m pip install scikit-learn==<that version>`), or re-save the model from an environment with the same version. "
                    "Demo models can be rebuilt with `python scripts/make_demo_assets.py`.")
        raise ValueError(f"The model file could not be loaded ({type(e).__name__}: {e}). "
                         "It may be corrupted, saved with an incompatible library version, or not a model." + hint)
    if isinstance(model, (pd.DataFrame, np.ndarray, dict, list, str)):
        raise ValueError(f"The file contains a {type(model).__name__}, not a trained model.")
    return model, caught


@dataclass
class ModelInfo:
    type_name: str = ""
    has_predict: bool = False
    has_predict_proba: bool = False
    classes: list = field(default_factory=list)
    feature_names: list | None = None
    n_features: int | None = None
    is_pipeline: bool = False
    has_preprocessing: bool | None = None       # None = cannot tell
    preprocessing_steps: list = field(default_factory=list)
    calibrated_wrapper: bool = False
    explain_family: str = "unknown"             # tree | linear | other
    sklearn_version: str = ""
    notes: list = field(default_factory=list)


_PREPROC_NAMES = ("ColumnTransformer", "OneHotEncoder", "OrdinalEncoder", "StandardScaler", "MinMaxScaler",
                  "RobustScaler", "SimpleImputer", "KNNImputer", "FunctionTransformer", "TargetEncoder",
                  "PolynomialFeatures", "QuantileTransformer", "PowerTransformer", "Normalizer")
_TREE_NAMES = ("RandomForest", "GradientBoosting", "DecisionTree", "ExtraTrees", "XGB", "LGBM", "CatBoost",
               "HistGradientBoosting", "AdaBoost")
_LINEAR_NAMES = ("LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDClassifier",
                 "LinearSVC", "LinearDiscriminantAnalysis", "RidgeClassifier", "Perceptron")


def _unwrap(model):
    """Return (final_estimator, preprocessing_step_names, wrapper_flag, base_pipeline_or_none)."""
    steps, calibrated, est, pipe = [], False, model, None
    # CalibratedClassifierCV (as used in Module 2): fitted base pipeline lives in calibrated_classifiers_
    if type(est).__name__ == "CalibratedClassifierCV":
        calibrated = True
        try:
            est = est.calibrated_classifiers_[0].estimator
        except Exception:
            try:
                est = est.estimator
            except Exception:
                pass
    if hasattr(est, "named_steps"):
        pipe = est
        for n, s in est.steps:
            if type(s).__name__ in _PREPROC_NAMES:
                steps.append(f"{n} ({type(s).__name__})")
        est = est.steps[-1][1]
    return est, steps, calibrated, pipe


def inspect_model(model) -> ModelInfo:
    mi = ModelInfo(type_name=type(model).__name__)
    mi.has_predict = callable(getattr(model, "predict", None))
    mi.has_predict_proba = callable(getattr(model, "predict_proba", None))
    _cl = getattr(model, "classes_", None)
    mi.classes = list(_cl) if _cl is not None else []
    fn = getattr(model, "feature_names_in_", None)
    mi.feature_names = [str(c) for c in fn] if fn is not None else None
    mi.n_features = getattr(model, "n_features_in_", None)
    final, steps, mi.calibrated_wrapper, pipe = _unwrap(model)
    mi.is_pipeline = hasattr(model, "named_steps") or pipe is not None
    mi.preprocessing_steps = steps
    mi.has_preprocessing = True if steps else (False if mi.is_pipeline or mi.has_predict else None)
    fam = type(final).__name__
    mi.explain_family = "tree" if any(t in fam for t in _TREE_NAMES) else "linear" if any(t in fam for t in _LINEAR_NAMES) else "other"
    try:
        import sklearn
        mi.sklearn_version = sklearn.__version__
    except Exception:
        pass
    if not mi.has_predict:
        mi.notes.append("Object has no predict() method - it cannot be audited.")
    if mi.has_predict and not mi.has_predict_proba:
        mi.notes.append("No predict_proba(): ROC-AUC, PR-AUC, Brier score and calibration cannot be computed (class predictions only).")
    if mi.has_preprocessing is False and not mi.is_pipeline:
        mi.notes.append("Not a Pipeline - the model expects already-numeric, already-encoded input. Text/categorical columns will fail unless you provide encoded data.")
    if mi.feature_names is None:
        mi.notes.append("Model does not store feature names - column names/order cannot be verified automatically; you must select the feature columns in the right order.")
    return mi


@dataclass
class Compatibility:
    ok: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    feature_cols: list = field(default_factory=list)
    class_labels: list = field(default_factory=list)   # model.classes_ as given
    can_proba: bool = False


def check_compatibility(model, mi: ModelInfo, df: pd.DataFrame, target: str | None,
                        user_feature_cols: list[str] | None = None, sample_n: int = 200) -> Compatibility:
    """Validate model vs dataset WITHOUT silently fixing anything. Collect clear errors/warnings."""
    c = Compatibility()
    if not mi.has_predict:
        c.errors.append("The uploaded object has no predict() method.")
        return c
    # ---- which columns will be fed to the model -----------------------------------
    if mi.feature_names is not None:
        missing = [f for f in mi.feature_names if f not in df.columns]
        if missing:
            c.errors.append(f"The dataset is missing {len(missing)} column(s) the model was trained on: {missing[:15]}"
                            + (" ..." if len(missing) > 15 else "")
                            + ". Upload the data in the same format used for training (same preprocessing / derived columns).")
            return c
        feat = list(mi.feature_names)
        if target in feat:
            c.errors.append(f"The target column '{target}' is one of the model's input features - this is target leakage, or the wrong target was selected.")
            return c
        extra = [x for x in df.columns if x not in feat and x != target]
        if extra:
            c.warnings.append(f"{len(extra)} dataset column(s) are not used by the model (ignored): {extra[:10]}")
        if user_feature_cols and list(user_feature_cols) != feat:
            c.warnings.append("Your selected feature columns differ from the model's stored feature list; the model's own list/order is used.")
    else:
        feat = list(user_feature_cols or [])
        if not feat:
            c.errors.append("The model does not store its feature names - please select the feature columns (in training order).")
            return c
        if mi.n_features is not None and len(feat) != mi.n_features:
            c.errors.append(f"Shape mismatch: the model expects {mi.n_features} input features but {len(feat)} columns were selected.")
            return c
        c.warnings.append("Feature names cannot be verified for this model; correct column ORDER is the user's responsibility.")
    c.feature_cols = feat
    # ---- dtype sanity for models without preprocessing ----------------------------
    if mi.has_preprocessing is False:
        bad = [f for f in feat if not pd.api.types.is_numeric_dtype(df[f]) and not pd.api.types.is_bool_dtype(df[f])]
        if bad:
            c.errors.append(f"The model has no preprocessing step but {len(bad)} selected column(s) are non-numeric (e.g. {bad[:5]}). "
                            "Encode them exactly as during training, or upload a full Pipeline that includes the encoder.")
            return c
    elif mi.has_preprocessing is None:
        c.warnings.append("Cannot tell whether preprocessing is included in the uploaded model; the trial prediction below tests this.")
    # ---- trial prediction on a small sample ---------------------------------------
    sample = df[feat].head(sample_n)
    X_in = sample if mi.feature_names is not None else sample.to_numpy()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pred = model.predict(X_in)
            if mi.has_predict_proba:
                proba = np.asarray(model.predict_proba(X_in))
                c.can_proba = True
                if proba.ndim != 2 or proba.shape[0] != len(sample):
                    c.errors.append(f"predict_proba returned unexpected shape {proba.shape}.")
                    return c
            for x in w:
                c.warnings.append(f"During trial prediction: {str(x.message)[:200]}")
    except Exception as e:
        c.errors.append(f"The model could not predict on this dataset ({type(e).__name__}: {str(e)[:300]}). "
                        "Typical causes: missing encoders/scalers, unseen category formats, missing values the model cannot handle, or a different training schema.")
        return c
    if len(pred) != len(sample):
        c.errors.append("Model output length does not match the number of input rows.")
        return c
    c.class_labels = list(mi.classes) if mi.classes else sorted(pd.unique(pd.Series(pred)))
    # ---- target vs model classes -------------------------------------------------------
    if target is not None and target in df.columns:
        tv = {str(v) for v in df[target].dropna().unique()}
        cl = {str(v) for v in c.class_labels}
        if len(tv) > 1 and not tv.issubset(cl):
            c.warnings.append(f"Target values {sorted(tv - cl)[:6]} are not among the model's classes {sorted(cl)}. "
                              "For a binary model, pick the positive label below; otherwise the target column is probably not the one this model predicts.")
    if not mi.has_predict_proba:
        c.warnings.append("No probability output: probability-based metrics will be 'Not Evaluated'.")
    c.ok = True
    return c


def predict_all(model, mi: ModelInfo, df: pd.DataFrame, feat: list[str]):
    """Predict for every row. Returns (proba_or_None, labels_pred). Raises on failure (caller shows message)."""
    X = df[feat] if mi.feature_names is not None else df[feat].to_numpy()
    pred = np.asarray(model.predict(X))
    proba = np.asarray(model.predict_proba(X)) if mi.has_predict_proba else None
    return proba, pred
