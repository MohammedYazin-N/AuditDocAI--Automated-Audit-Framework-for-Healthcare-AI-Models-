# AuditDoc AI — Automated Audit Framework for Healthcare AI Models

AuditDoc AI is a Streamlit web app that audits a **healthcare dataset together with a trained classification model**. You upload a dataset (CSV/XLSX) and, optionally, a trained model. The app checks the data, looks for leakage, measures performance and calibration, compares subgroups for fairness, explains the model, and produces a downloadable report.

> **Important:** AuditDoc AI is an exploratory academic tool. It is **not** a clinical validation, regulatory assessment or certification of any model. A "Passed" result means no problem was found by these specific checks; it never proves a model is safe, unbiased or free of leakage.

**Live app:**(https://auditdocai.streamlit.app/) · **Project:** AuditDocAI — *Automated Audit Framework for Healthcare AI Models* -Healthcare Analytics

---

## Why this project exists

Healthcare AI models are often reported with a single accuracy or AUC number. That number can be misleading when the data is dirty, when information from the outcome has leaked into the inputs, when the model is poorly calibrated, or when it works much worse for some patient groups. This project turns those checks into one reusable tool, so any compatible dataset and model can be audited the same way and the results are written down in a report.

The project grew out of an audit of the **Diabetes 130-US Hospitals** readmission dataset (Strack et al., 2014). The notebook logic from the team's three modules (data quality, leakage/performance, fairness/explainability) was generalised so that nothing is hard-coded to that dataset.

---

## What the app does

| Page | What it checks | Possible results |
|---|---|---|
| **1. Dataset** | Upload CSV/XLSX, profile columns, choose the target, positive class, identifier and demographic columns (the app suggests candidates, you decide). | Shows which audits are available for what you uploaded. |
| **2. Model** | Upload a trained model, run a compatibility check against the dataset (features, probabilities, preprocessing, classes). | Compatible / incompatible, with the reason. |
| **3. Data Quality** | Missing values, duplicates, outliers, invalid values, high-cardinality columns, class balance, a quality scorecard. Optional cleaning, with every change logged. **The original data is never overwritten.** | Passed / Warning / Failed |
| **4. Leakage & Performance** | Leakage: identifier columns, target copies, suspicious column names, single-feature strength, duplicates, train/test overlap, whether each feature would be available at prediction time. Performance: AUC, accuracy, precision/recall, confusion matrix, calibration, Brier score, threshold table. | Leakage: Confirmed / Potential / Suspicious / Cannot be performed. Performance: Passed / Warning / Failed |
| **5. Fairness** | Subgroup performance for the demographic columns you choose, with confidence intervals; gaps are flagged only when the intervals do not overlap. Groups that are too small are reported as not evaluated. | Passed / Warning / Failed / Not Evaluated |
| **6. Explainability** | SHAP (tree and linear models), permutation importance as fallback, plus local explanations for single rows. Never fabricated: if it cannot be computed, it says so. | Available / Not Evaluated |
| **7. Report** | One-click **PDF**, **JSON** and **CSV** report of every finding, with the criteria used and recommended actions. | Downloadable files |

### Result statuses
Every finding uses one of four statuses: **Passed**, **Warning**, **Failed**, **Not Evaluated**. Audits that cannot run (for example, fairness with no demographic column) are shown as unavailable or Not Evaluated with the reason, never as a made-up result.

### Honest labelling of results
Performance numbers are labelled with where the data came from: *independent test data*, *training data* or *unknown*. The app asks you to declare this and cannot verify it, so the report states it as "declared by the user".

---

## How to use it (quick start)

1. Open the app and go to **Home**. To try it straight away, pick a demo and click **Load demo dataset + model**.
2. For your own data: **Dataset** → upload a CSV/XLSX → choose the target column and the positive class.
3. **Model** → tick the trust box → upload a `.pkl` or `.joblib` model. The compatibility check runs automatically.
4. Run the audits (Data Quality, Leakage & Performance, Fairness, Explainability). Pages with missing prerequisites explain what is needed.
5. **Report** → download the PDF, JSON or CSV.

Demo data included: a synthetic hospital-readmission dataset, a synthetic heart-disease dataset, and the public breast-cancer dataset from scikit-learn, each with a pre-trained model.

---

## Supported models and limits

**Supported now**
- Classification models saved as `.pkl` or `.joblib` from **scikit-learn or scikit-learn-compatible** APIs (they need `predict` and, for most audits, `predict_proba`): pipelines, calibrated models, random forests, gradient boosting, logistic regression, and similar.
- Best results when the model is a Pipeline that **includes its own preprocessing**, so it accepts the same columns as the uploaded dataset.
- **Version rule:** pickle/joblib files only load reliably in the same scikit-learn version they were saved with. This deployment pins `scikit-learn==1.6.1`. A model saved with a different version may fail to load with an `AttributeError`. Install or pin the matching version, or re-save the model.

**Planned ** (not part of this release)
- ONNX model upload (independent of library versions).
- A "predictions only" mode that audits a file of true labels and predicted probabilities from any framework (explainability would remain unavailable in that mode).
- Evaluation across several datasets (train / validation / test / external), with drift checks and bootstrap confidence intervals.

**Out of scope**
- Regression and deep-learning models; text, image or time-series models.
- Proving clinical safety, regulatory compliance or absence of bias.

---

## Security

`.pkl` and `.joblib` files **run code when they are loaded**. The app asks you to confirm that you created the file or fully trust its source, but it **cannot detect a malicious file**. On a public deployment, upload only models you created yourself.

Do not upload real, identifiable patient data to a public deployment. Use de-identified or synthetic data.

---

## How the audits work (method notes)

- **Data quality:** thresholds from the team's Module 1 (for example, missing-value severity bands) are kept unchanged. The scorecard covers Completeness, Uniqueness, Validity, Consistency and Balance, and drops any dimension it cannot compute.
- **Leakage:** a single-feature screen uses effect size (AUC ≥ 0.90 or Cramér's V ≥ 0.5 as heuristic flags) instead of p-values, because almost any feature is "significant" with ~100,000 rows. A patient-level split (StratifiedGroupKFold) is used when a patient identifier is given.
- **Performance:** AUC, accuracy, precision, recall, specificity, F1, confusion matrix, Brier score against the prevalence baseline, calibration bins and mean calibration error.
- **Fairness:** per-group true-positive rate, false-positive rate and selection rate with Wilson 95% intervals. Groups need a minimum size (100 rows, 20 positives); a gap is flagged only when the intervals do not overlap, so no arbitrary gap threshold is invented. Numeric demographics (such as age) are binned automatically.
- **Explainability:** SHAP on the base estimator of calibrated models; permutation importance as a fallback.
- **Limits of heuristics:** leakage and fairness checks are heuristics. Passing them does not prove the absence of leakage or bias.

---

## Where the logic came from

| Notebook source | Used in | What changed |
|---|---|---|
| Module 1 — data quality (missing, duplicates, outliers, imbalance, correlation, cardinality, scorecard, cleaning log) | `audit/data_quality.py` | Thresholds kept; column names became arguments; cleaning is opt-in and logged. |
| Module 2 — leakage screen, patient-level split, metrics, calibration, threshold table | `audit/leakage.py`, `audit/performance.py` | Effect sizes instead of p-values; four leakage categories added. |
| Module 3 — subgroup table, Wilson intervals, minimum group size, SHAP | `audit/fairness.py`, `audit/explain.py` | Works with any demographic column; numeric columns auto-binned. |
| New for the app | `audit/model_io.py`, `audit/data_io.py`, `audit/report.py`, `app.py` | Upload and validation, compatibility checks, provenance labelling, PDF/JSON/CSV report, UI. |

---

## Project structure

```
app.py                 Streamlit pages (UI)
ui_helpers.py          theme, components and Plotly charts
.streamlit/config.toml dark theme and upload size
audit/
  common.py            statuses and the Finding record
  data_io.py           upload, profiling, column suggestions, audit availability
  model_io.py          model loading and compatibility checks
  data_quality.py      data quality audit and logged cleaning
  leakage.py           leakage checks
  performance.py       performance and calibration
  fairness.py          subgroup fairness
  explain.py           SHAP / permutation importance
  report.py            PDF, JSON and CSV report
  pipeline.py          glue between the pages, tests and report
scripts/make_demo_assets.py   rebuilds the demo datasets and models
tests/                 automated tests (backend and headless UI)
demo_assets/           demo datasets and models
requirements.txt
```

---

## Run it locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Run the tests with `python -m pytest -q tests`.

To rebuild the demo files: `python scripts/make_demo_assets.py`.

## Deploy on Streamlit Community Cloud

1. Push this folder to a public GitHub repository (never commit real patient data).
2. Go to share.streamlit.io → **Create app** → choose the repository, branch `main` and main file `app.py`.
3. In **Advanced settings**, choose Python 3.12.
4. Dependencies are installed from `requirements.txt` (scikit-learn is pinned to match the trained model).

---

## Known limitations

- The tool audits the model and data you give it. It cannot verify that the evaluation data was truly independent of training; that is declared by the user.
- Results depend on the columns and choices you make (target, positive class, demographics, threshold).
- Subgroup results for small groups are unreliable and are shown as Not Evaluated.
- Free-tier hosting has limited memory; use a sample for very large datasets.


