"""Create demo datasets + trained models so the dashboard can be tried immediately.

    python scripts/make_demo_assets.py                      # writes demo_assets/
    python scripts/make_demo_assets.py --cleaned-csv diabetes_cleaned_all_encounters.csv
         -> ALSO re-creates Person 2's real Module-2 model from the notebook's cleaned export.

Demos written (all clearly labelled in the file names):
  A  synthetic_readmission_*   SYNTHETIC data shaped like the diabetes-readmission project
                               (patient_nbr, race, gender, age ...), calibrated Random Forest pipeline.
  B  synthetic_heart_*         SYNTHETIC cardiology-style data, Gradient Boosting pipeline (different task/columns).
  C  breast_cancer_*           REAL public scikit-learn Wisconsin breast-cancer data, scaler + logistic regression.
                               (no demographic columns -> fairness is correctly 'Not Evaluated')
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.datasets import load_breast_cancer

OUT = Path(__file__).resolve().parent.parent / "demo_assets"


def preproc(X: pd.DataFrame):
    num = X.select_dtypes(include=["number"]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num),
                              ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                                ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])


def demo_a(n=12000, seed=0):
    rng = np.random.default_rng(seed)
    pat = rng.integers(0, int(n * 0.7), n)
    age = rng.choice(["[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)"], n, p=[.05, .1, .2, .25, .25, .15])
    df = pd.DataFrame({"encounter_id": np.arange(n), "patient_nbr": pat,
                       "race": rng.choice(["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"], n, p=[.72, .19, .04, .02, .03]),
                       "gender": rng.choice(["Female", "Male"], n), "age": age,
                       "time_in_hospital": rng.integers(1, 15, n), "num_medications": rng.poisson(15, n),
                       "number_inpatient": rng.poisson(.6, n), "number_emergency": rng.poisson(.2, n),
                       "num_lab_procedures": rng.poisson(43, n),
                       "insulin": rng.choice(["No", "Steady", "Up", "Down"], n), "A1Cresult": rng.choice(["None", ">7", "Norm", ">8"], n, p=[.8, .08, .05, .07])})
    logit = -3.0 + .5 * df.number_inpatient + .08 * df.time_in_hospital + .02 * (df.num_medications - 15) + .4 * (df.age.isin(["[70-80)", "[80-90)"])) + rng.normal(0, .8, n)
    df["target"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.random(n) < .04, "race"] = np.nan
    feats = [c for c in df.columns if c not in ("encounter_id", "patient_nbr", "target")]
    # patient-level split, exactly as Module 2 (StratifiedGroupKFold)
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(df[feats], df.target, groups=df.patient_nbr))
    Xtr, ytr = df.iloc[tr][feats], df.iloc[tr].target
    base = Pipeline([("preprocessor", preproc(Xtr)), ("classifier", RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))])
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xtr, ytr, groups=df.iloc[tr].patient_nbr))
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=cv).fit(Xtr, ytr)
    joblib.dump(model, OUT / "synthetic_readmission_model.joblib", compress=3)
    df.iloc[te].to_csv(OUT / "synthetic_readmission_test.csv", index=False)
    df.iloc[tr].to_csv(OUT / "synthetic_readmission_train.csv", index=False)


def demo_b(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"patient_id": np.arange(n), "age": rng.integers(30, 85, n), "sex": rng.choice(["F", "M"], n),
                       "ethnicity": rng.choice(["GroupA", "GroupB", "GroupC"], n, p=[.6, .3, .1]),
                       "systolic_bp": rng.normal(130, 18, n).round(), "cholesterol": rng.normal(215, 40, n).round(),
                       "smoker": rng.choice(["yes", "no"], n, p=[.25, .75]), "bmi": rng.normal(27.5, 4.5, n).round(1)})
    z = -9 + .05 * df.age + .02 * df.systolic_bp + .008 * df.cholesterol + .6 * (df.smoker == "yes") + .5 * (df.sex == "M") + rng.normal(0, .7, n)
    df["heart_disease"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    feats = [c for c in df.columns if c not in ("patient_id", "heart_disease")]
    tr, te = train_test_split(df.index, test_size=.3, random_state=0, stratify=df.heart_disease)
    model = Pipeline([("pre", preproc(df.loc[tr, feats])), ("clf", GradientBoostingClassifier(random_state=0))]).fit(df.loc[tr, feats], df.loc[tr, "heart_disease"])
    joblib.dump(model, OUT / "synthetic_heart_model.joblib", compress=3)
    df.loc[te].to_csv(OUT / "synthetic_heart_test.csv", index=False)
    df.loc[tr].to_csv(OUT / "synthetic_heart_train.csv", index=False)


def demo_c():
    d = load_breast_cancer(as_frame=True)
    df = d.frame.rename(columns={"target": "diagnosis"})      # 0 = malignant, 1 = benign (scikit-learn coding)
    tr, te = train_test_split(df.index, test_size=.3, random_state=0, stratify=df.diagnosis)
    feats = [c for c in df.columns if c != "diagnosis"]
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]).fit(df.loc[tr, feats], df.loc[tr, "diagnosis"])
    joblib.dump(model, OUT / "breast_cancer_model.joblib", compress=3)
    df.loc[te].to_csv(OUT / "breast_cancer_test.csv", index=False)
    df.loc[tr].to_csv(OUT / "breast_cancer_train.csv", index=False)


def real_diabetes(cleaned_csv: str):
    """Re-create Module 2's model from Module 1's export (same split, features, RF settings, calibration)."""
    df = pd.read_csv(cleaned_csv, low_memory=False)
    excl = [c for c in ["encounter_id", "patient_nbr", "readmitted", "readmitted_30d", "target"] if c in df.columns]
    feats = [c for c in df.columns if c not in excl]
    X, y, g = df[feats], df["target"], df["patient_nbr"]
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups=g))
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    rf = Pipeline([("preprocessor", preproc(Xtr)), ("classifier", RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1))])
    cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(Xtr, ytr, groups=g.iloc[tr]))
    model = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=cv).fit(Xtr, ytr)
    joblib.dump(model, OUT / "diabetes_final_calibrated_model.joblib", compress=3)
    df.iloc[te].to_csv(OUT / "diabetes_test_set.csv", index=False)
    df.iloc[tr].to_csv(OUT / "diabetes_train_set.csv", index=False)
    print("Real diabetes assets written. Recommended decision threshold from Module 2: 0.13")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned-csv", help="Module 1 export diabetes_cleaned_all_encounters.csv")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    demo_a(); demo_b(); demo_c()
    if a.cleaned_csv:
        real_diabetes(a.cleaned_csv)
    print("Wrote:", *sorted(p.name for p in OUT.iterdir()), sep="\n  ")
