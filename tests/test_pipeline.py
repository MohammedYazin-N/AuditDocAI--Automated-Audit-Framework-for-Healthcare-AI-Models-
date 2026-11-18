"""Headless end-to-end tests of every audit module on three different datasets/models.
Run:  python -m pytest -q tests
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, pytest, joblib, io
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from audit import pipeline as pl, model_io, data_io
from audit.common import STATUSES

D = ROOT / "demo_assets"
CASES = {
    "readmission": dict(test="synthetic_readmission_test.csv", train="synthetic_readmission_train.csv", model="synthetic_readmission_model.joblib",
                        target="target", pos=1, ids=["encounter_id", "patient_nbr"], patient="patient_nbr", demo=["race", "gender", "age"]),
    "heart": dict(test="synthetic_heart_test.csv", train="synthetic_heart_train.csv", model="synthetic_heart_model.joblib",
                  target="heart_disease", pos=1, ids=["patient_id"], patient=None, demo=["sex", "ethnicity", "age"]),
    "cancer": dict(test="breast_cancer_test.csv", train="breast_cancer_train.csv", model="breast_cancer_model.joblib",
                   target="diagnosis", pos=1, ids=[], patient=None, demo=[]),
}


@pytest.fixture(params=list(CASES))
def case(request):
    c = CASES[request.param]
    df = pd.read_csv(D / c["test"])
    tr = pd.read_csv(D / c["train"])
    model, _ = model_io.load_model((D / c["model"]).read_bytes(), c["model"])
    mi = model_io.inspect_model(model)
    return request.param, c, df, tr, model, mi


def test_full_pipeline(case):
    name, c, df, tr, model, mi = case
    comp = model_io.check_compatibility(model, mi, df, c["target"])
    assert comp.ok, comp.errors
    dqr = pl.run_data_quality(df, c["target"], c["pos"], c["ids"], c["patient"], None, c["demo"])
    assert all(f.status in STATUSES for f in dqr["findings"])
    lk = pl.run_leakage(df, c["target"], comp.feature_cols, c["ids"], c["patient"], c["pos"], train_df=tr)
    assert all(f.status in STATUSES for f in lk["findings"])
    # model features must never contain identifiers / target for these models
    assert not set(comp.feature_cols) & set(c["ids"] + [c["target"]])
    pr = pl.run_performance(model, mi, df, c["target"], comp.feature_cols, c["pos"], "independent", 0.5)
    m = pr["metrics"]
    # independent re-computation of a few numbers (not through our code)
    from sklearn.metrics import roc_auc_score
    p = model.predict_proba(df[comp.feature_cols])[:, 1]
    assert abs(m["roc_auc"] - roc_auc_score(df[c["target"]], p)) < 1e-9
    yhat = (p >= .5).astype(int); y = df[c["target"]].to_numpy()
    assert m["confusion"]["tp"] == int(((y == 1) & (yhat == 1)).sum())
    fa = pl.run_fairness(pr, df, c["demo"])
    if c["demo"]:
        assert set(fa["attributes"]) == set(c["demo"])
        for a in fa["attributes"].values():
            assert a["table"].n.sum() == len(df)       # every row lands in exactly one group
    ex = pl.run_explainability(model, mi, df, comp.feature_cols, pr["y_idx"], pr["classes"], pr["pos_idx"], do_permutation=False, n_shap=50)
    assert "error" not in ex["shap"], ex["shap"]
    assert ex["shap"]["importance"].mean_abs_shap.sum() > 0


def test_incompatible_dataset_gives_clear_error():
    df = pd.read_csv(D / "synthetic_heart_test.csv")
    model, _ = model_io.load_model((D / "synthetic_readmission_model.joblib").read_bytes(), "x.joblib")
    comp = model_io.check_compatibility(model, model_io.inspect_model(model), df, "heart_disease")
    assert not comp.ok and "missing" in comp.errors[0].lower()


def test_training_provenance_is_flagged_failed():
    c = CASES["heart"]; df = pd.read_csv(D / c["train"])
    model, _ = model_io.load_model((D / c["model"]).read_bytes(), "m.joblib"); mi = model_io.inspect_model(model)
    comp = model_io.check_compatibility(model, mi, df, c["target"])
    pr = pl.run_performance(model, mi, df, c["target"], comp.feature_cols, 1, "training", 0.5)
    assert any(f.check == "Evaluation data provenance" and f.status == "Failed" for f in pr["findings"])


def test_original_dataframe_is_never_modified():
    df = pd.read_csv(D / "synthetic_readmission_test.csv"); before = df.copy(deep=True)
    from audit import data_quality as dq
    cleaned, log = dq.clean_dataset(df, drop_exact_duplicates=True, drop_constant_columns=True, drop_missing_above=.5, fill_categorical_unknown=True, impute_numeric_median=True)
    pd.testing.assert_frame_equal(df, before)
    assert len(log) >= 2


def test_leakage_detection_catches_planted_leaks():
    from audit import leakage
    rng = np.random.default_rng(0); n = 2000
    df = pd.DataFrame({"pid": rng.integers(0, 900, n), "a": rng.normal(size=n)})
    df["y"] = (df.a + rng.normal(size=n) > 0).astype(int); df["y_copy"] = df.y
    F, _ = leakage.run_leakage_checks(df, "y", ["pid", "a", "y_copy"], ["pid"], "pid")
    byname = {f.check: f for f in F}
    assert byname["Identifier-based leakage"].status == "Failed"
    assert byname["Target leakage: exact copy / deterministic recode"].status == "Failed"
    assert byname["Train-test contamination / patient overlap"].status == "Not Evaluated"
    assert byname["Overall leakage conclusion"].status == "Warning"


def test_non_model_file_rejected():
    import pickle
    with pytest.raises(ValueError):
        model_io.load_model(pickle.dumps({"a": 1}), "x.pkl")
    with pytest.raises(ValueError):
        model_io.load_model(b"garbage", "x.joblib")
