# AuditDoc AI (Person 4 - Application Development & Integration)

Streamlit app that audits a **healthcare dataset + trained classification model**: data quality, data leakage, performance &
calibration, subgroup fairness, explainability, and a downloadable PDF/JSON/CSV report. Nothing is hard-coded to the diabetes data:
you choose the target, identifier and demographic columns in the UI.

> Exploratory academic tool. Not a clinical validation, regulatory assessment or certification.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_demo_assets.py                      # (re)creates demo_assets/ (already included)
streamlit run app.py
```
Try it immediately: **Home → pick a demo → Load demo dataset + model**.

## Folder layout
```
app.py                 Streamlit pages (UI only)
ui_helpers.py          badges, CSS, Plotly figures
audit/
  common.py            statuses (Passed / Warning / Failed / Not Evaluated) + Finding record
  data_io.py           Feature 1  upload, profiling, identifier/demographic hints, audit availability
  model_io.py          Feature 2  safe-ish loading, compatibility checks
  data_quality.py      Feature 3  <- Module 1 (Person 1) generalised
  leakage.py           Feature 4A <- Module 2 (Person 2) generalised
  performance.py       Feature 4B <- Module 2 metrics/calibration
  fairness.py          Feature 5  <- Module 3 (Person 3)
  explain.py           Feature 6  <- Module 3 SHAP
  report.py            Feature 7  PDF (ReportLab) + JSON/CSV
  pipeline.py          glue used by app, tests and report
scripts/make_demo_assets.py   builds demo data/models (+ real diabetes model, see below)
tests/                 pytest: backend + headless UI tests
demo_assets/           3 demo datasets + models
```

## Reused vs new (for your viva)
| Notebook source | Reused in | What changed |
|---|---|---|
| Module 1: missing/duplicate/outlier/imbalance/correlation/cardinality checks, severity bands, scorecard, cleaning log | `data_quality.py` | thresholds kept unchanged; column names became arguments; cleaning is opt-in & logged; scorecard drops dimensions it cannot compute |
| Module 2: ID exclusion, target-copy check, association screen, availability review, patient-level split, metrics, calibration, threshold table | `leakage.py`, `performance.py` | association screen uses effect size (AUC / Cramér's V) instead of p<0.001 (almost everything is significant at n≈100k); four leakage categories added |
| Module 3: subgroup table, Wilson CI, min group size 100 / 20 positives, subgroup calibration, SHAP on calibrated model's base estimator | `fairness.py`, `explain.py` | any demographic column; numeric ones auto-binned; disparity flagged by non-overlapping CIs (no invented gap threshold) |
| **New** | `model_io.py`, `data_io.py`, `report.py`, `app.py`, permutation importance, multiclass, provenance labelling | - |

## Using YOUR real diabetes model
The notebook never saved the model, so export it from the notebook (add at the end of Module 2):
```python
import joblib
joblib.dump(final_calibrated_model, "diabetes_final_calibrated_model.joblib")
test_df = <cleaned dataframe>.loc[X_test.index]          # the held-out patient-level test rows, WITH patient_nbr and target
test_df.to_csv("diabetes_test_set.csv", index=False)
train_df.to_csv("diabetes_train_set.csv", index=False)    # optional: enables train/test overlap checks
```
(or run `python scripts/make_demo_assets.py --cleaned-csv diabetes_cleaned_all_encounters.csv` to rebuild it with the notebook's settings.)
In the app: target = `target`, positive class = `1`, identifiers = `encounter_id, patient_nbr`, patient identifier = `patient_nbr`,
demographics = `race, gender, age`, provenance = *Independent test data*, threshold = the value chosen in Module 2 (~0.13).
Upload the same preprocessed columns the model was trained on.

## Tests
```bash
python -m pytest -q tests
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (do **not** commit real patient data).
2. share.streamlit.io → New app → pick the repo, branch, main file `app.py`.
3. Python 3.10-3.12 recommended. Model files must be saved with a scikit-learn version close to the one in `requirements.txt` (pin the version you trained with, e.g. `scikit-learn==1.5.2`, if you see version warnings).
4. Cloud uploads are limited (200-300 MB, see `.streamlit/config.toml`).

## Security
`.pkl`/`.joblib` files execute code when loaded. The app requires an explicit "I trust this file" tick and cannot detect malicious files.
A public deployment should only be used with models you created yourself.

## Known limitations
* Classification models from scikit-learn-compatible APIs (predict / predict_proba). Regression and deep-learning models are out of scope.
* SHAP supports tree and linear models; others fall back to permutation importance (or "not available").
* Leakage and fairness checks are heuristics; passing them never proves absence of leakage or bias.
* Whether evaluation rows are truly independent of training is declared by the user and cannot be verified.
