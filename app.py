"""AuditDoc AI - Streamlit entry point.

Run:   streamlit run app.py
Pages (sidebar): Home, Dataset Upload, Model Upload, Data Quality, Leakage & Performance, Fairness,
Explainability, Final Report.  All audit logic lives in the `audit/` package (reused from the notebook).
"""
from __future__ import annotations
import hashlib
import io
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from audit import data_io, data_quality as dq, model_io, pipeline as pl, report, explain as ex_mod, fairness as fr_mod, performance as perf_mod
from audit.common import STATUSES, PASSED, WARNING, FAILED, NOT_EVALUATED, counts
import ui_helpers as ui

st.set_page_config(page_title="AuditDoc AI", page_icon="🩺", layout="wide", initial_sidebar_state="collapsed")
ui.inject_css()
DEMO_DIR = Path(__file__).parent / "demo_assets"
TOKENS_ON = ["?", "NA", "N/A", "NaN", "nan", "NULL", "null", ""]          # "None" is OFF by default: in lab data it can be a real category
PAGES = ["Home", "Dataset Upload", "Model Upload", "Data Quality Audit", "Leakage & Performance", "Fairness Audit", "Model Explainability", "Final Audit Report"]

DEMOS = {
    "Synthetic readmission (calibrated Random Forest, patient IDs)": dict(
        test="synthetic_readmission_test.csv", train="synthetic_readmission_train.csv", model="synthetic_readmission_model.joblib",
        target="target", pos="1", ids=["encounter_id", "patient_nbr"], patient="patient_nbr", demo=["race", "gender", "age"], threshold=0.12),
    "Synthetic heart disease (Gradient Boosting pipeline)": dict(
        test="synthetic_heart_test.csv", train="synthetic_heart_train.csv", model="synthetic_heart_model.joblib",
        target="heart_disease", pos="1", ids=["patient_id"], patient=None, demo=["sex", "ethnicity", "age"], threshold=0.5),
    "Breast cancer - real public data (scaler + logistic regression, no demographics)": dict(
        test="breast_cancer_test.csv", train="breast_cancer_train.csv", model="breast_cancer_model.joblib",
        target="diagnosis", pos="1", ids=[], patient=None, demo=[], threshold=0.5),
}

# ------------------------------------------------------------------ state ----
DEFAULTS = dict(df=None, filename=None, file_key=None, raw_bytes=None, target=None, pos=None, ids=[], patient=None, record_id=None, demo=[],
                model=None, mi=None, model_name=None, model_key=None, load_warnings=[], compat=None, user_feats=[], train_df=None, train_name=None,
                provenance="unknown", threshold=0.5, split_col=None, split_val=None, res_dq=None, dq_sig=None, res_leak=None, res_perf=None,
                res_fair=None, res_expl=None, cleaned=None, cleaning_log=None, availability={}, prediction_context="", nav="Home", demo_loaded=None)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
S = st.session_state


def clear_results(from_stage="all"):
    for k in ("res_dq", "dq_sig", "res_leak", "res_perf", "res_fair", "res_expl", "cleaned", "cleaning_log"):
        S[k] = None
    if from_stage == "model":     # dataset results still valid
        pass


def goto(page):
    S.nav = page


def eval_df():
    """Rows used for model evaluation = uploaded dataset (never modified), optionally restricted to a split-column value."""
    df = S.df
    if df is None:
        return None
    if S.split_col and S.split_val is not None and S.split_col in df.columns:
        return df[df[S.split_col].astype(str) == str(S.split_val)]
    return df


def model_ready():
    return S.model is not None and S.compat is not None and S.compat.ok


def target_ready():
    return S.df is not None and S.target is not None


def refresh_compat():
    if S.model is None or S.df is None:
        S.compat = None
        return
    S.compat = model_io.check_compatibility(S.model, S.mi, S.df, S.target, S.user_feats)


def load_demo(name):
    d = DEMOS[name]
    S.df = pd.read_csv(DEMO_DIR / d["test"], keep_default_na=False, na_values=TOKENS_ON, low_memory=False)
    S.filename, S.file_key, S.raw_bytes = d["test"], f"demo-{name}", None
    S.target, S.pos, S.ids, S.patient, S.record_id, S.demo = d["target"], d["pos"], d["ids"], d["patient"], None, d["demo"]
    S.train_df = pd.read_csv(DEMO_DIR / d["train"], keep_default_na=False, na_values=TOKENS_ON, low_memory=False)
    S.train_name = d["train"]
    S.model, S.load_warnings = model_io.load_model((DEMO_DIR / d["model"]).read_bytes(), d["model"])
    S.mi, S.model_name, S.model_key = model_io.inspect_model(S.model), d["model"], f"demo-{name}"
    S.provenance, S.threshold, S.split_col, S.split_val, S.user_feats = "independent", d["threshold"], None, None, []
    S.demo_loaded = name
    clear_results()
    refresh_compat()


def pick_index(options, value, default=0):
    return options.index(value) if value in options else default


# --------------------------------------------------------------- top bar + nav ---
ui.topbar(S.filename if S.df is not None else None, S.target,
          "ok" if model_ready() else ("bad" if S.model is not None else "none"))
NAV_ICONS = {"Home": "🏠", "Dataset Upload": "📁", "Model Upload": "🧠", "Data Quality Audit": "🧪", "Leakage & Performance": "🎯",
             "Fairness Audit": "⚖️", "Model Explainability": "🔍", "Final Audit Report": "📄"}
NAV_SHORT = {"Home": "Home", "Dataset Upload": "Dataset", "Model Upload": "Model", "Data Quality Audit": "Data Quality", "Leakage & Performance": "Leakage & Performance",
             "Fairness Audit": "Fairness", "Model Explainability": "Explainability", "Final Audit Report": "Report"}
st.radio("Navigate", PAGES, key="nav", horizontal=True, label_visibility="collapsed", format_func=lambda p: f"{NAV_ICONS[p]} {NAV_SHORT[p]}")
page = S.nav

# =================================================================== HOME =====
def page_home():
    st.markdown("<div class='hero'><h1>Is your healthcare AI model <span>safe to trust?</span></h1>"
                "<p>Upload a dataset and a trained model. AuditDoc AI checks data quality, leakage, performance, fairness and explainability, "
                "tells you which audits can run and why, and exports a report of risks and limitations.</p></div>", unsafe_allow_html=True)
    # live pipeline strip
    done = lambda ok: "done" if ok else "todo"
    ui.pipeline_strip([("① Dataset", done(S.df is not None)), ("② Target set", done(S.target is not None)), ("③ Model checked", done(model_ready())),
                       ("④ Data quality", done(S.res_dq is not None)), ("⑤ Leakage", done(S.res_leak is not None)), ("⑥ Performance", done(S.res_perf is not None)),
                       ("⑦ Fairness", done(S.res_fair is not None)), ("⑧ Explainability", done(S.res_expl is not None))])
    left, right = st.columns([1.55, 1])
    with right:
        with st.container(border=True):
            st.markdown("#### ⚡ Quick start")
            st.caption("Demo files are in `demo_assets/`. Two are synthetic (not real patients); one is real public breast-cancer data.")
            name = st.selectbox("Demo", list(DEMOS), key="demo_pick", label_visibility="collapsed")
            st.button("Load demo dataset + model", on_click=load_demo, args=(name,), type="primary", width="stretch")
            if S.demo_loaded:
                st.success("Demo loaded. Open the next tabs above.")
            st.button("Or upload your own data →", on_click=goto, args=("Dataset Upload",), width="stretch")
        allf = report.all_findings({"dq": S.res_dq, "leak": S.res_leak, "perf": S.res_perf, "fair": S.res_fair, "expl": S.res_expl})
        if allf:
            st.plotly_chart(ui.fig_status_donut(allf), width="stretch")
    with left:
        st.markdown("#### What can run right now?")
        av = data_io.audit_availability(S.df, S.target, S.model is not None, model_ready(), S.demo, model_ready())
        cols = st.columns(2)
        for i, a in enumerate(av):
            colr = ui.GREEN if a["available"] else ui.CORAL
            reason = a["reason"] if (S.df is not None or a["available"]) else "Upload a dataset first."
            cols[i % 2].markdown(f"<div class='tile' style='margin-bottom:.6rem'><div class='t'><span class='dot' style='--c:{colr}'></span>{a['audit']}</div>"
                                 f"<div class='d'>{reason}</div></div>", unsafe_allow_html=True)
    st.markdown("#### Three situations AuditDoc AI handles")
    c1, c2, c3 = st.columns(3)
    for col, (t, d, colr) in zip((c1, c2, c3), [
            ("A · Dataset only", "Data-quality audit and data-only leakage checks. Model-based audits are marked Not Evaluated.", ui.SKY),
            ("B · Dataset + compatible model", "Everything: leakage, performance, calibration, fairness and explainability where the model supports it.", ui.TEAL),
            ("C · Incompatible model", "The exact reason is shown (missing columns, preprocessing, shape). No misleading results are produced.", ui.CORAL)]):
        col.markdown(f"<div class='kpi' style='--c:{colr}'><div class='l'>Scenario</div><div class='v' style='font-size:1.1rem'>{t}</div><div class='s'>{d}</div></div>", unsafe_allow_html=True)
    st.caption("Exploratory academic tool. Not a clinical validation, regulatory assessment or certification.")


# ============================================================ DATASET UPLOAD ===
def page_dataset():
    ui.page_head("01", "Dataset Upload & Validation", "Your original file is never modified. Cleaning later creates a separate copy.")
    tokens = st.multiselect("Text values to treat as **missing**", ["?", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", ""], default=TOKENS_ON,
                            help="'None' is OFF by default because in some medical columns (e.g. a lab result of 'None') it is a real category, not missing data.")
    up = st.file_uploader("Upload a healthcare dataset (.csv or .xlsx)", type=["csv", "xlsx"])
    if up is not None:
        raw = up.getvalue()
        key = hashlib.sha256(raw + "|".join(tokens).encode()).hexdigest()
        if key != S.file_key:
            try:
                with st.spinner("Reading file..."):
                    df, info = data_io.load_table(io.BytesIO(raw), up.name, tokens)
                S.df, S.filename, S.file_key, S.raw_bytes = df, up.name, key, raw
                S.target, S.pos, S.demo, S.record_id, S.patient = None, None, [], None, None
                S.ids = [d["column"] for d in data_io.detect_identifier_columns(df)]
                S.demo = data_io.suggest_demographic_columns(df, exclude=S.ids)
                S.compat = None; S.demo_loaded = None
                clear_results()
            except Exception as e:
                st.error(f"Could not read the file: {e}")
                return
    df = S.df
    if df is None:
        st.info("Upload a file, or load a demo on the Home page.")
        return
    sha = hashlib.sha256(S.raw_bytes).hexdigest()[:16] if S.raw_bytes else "(demo file)"
    roles = data_io.column_roles(df)
    ui.kpi_row([("Rows", f"{len(df):,}", "records", ui.TEAL), ("Columns", df.shape[1], f"{len(roles['numeric'])} numeric · {len(roles['categorical'])} categorical", ui.SKY),
                ("Missing cells", f"{df.isna().mean().mean():.1%}", "of all cells", ui.AMBER), ("Duplicate rows", f"{int(df.duplicated().sum()):,}", "exact copies", ui.CORAL)])
    st.caption(f"File: `{S.filename}` · SHA-256 (first 16): `{sha}` — recorded in the report for traceability.")
    st.write("")
    t3, _right = st.columns([1.05, 1.3], gap="large")      # left: configure · right: preview
    t1, t2 = _right.tabs(["Preview", "Columns & types"])
    with t1:
        st.dataframe(df.head(100), width="stretch")
    with t2:
        prof = data_io.profile(df)
        st.dataframe(prof, width="stretch", height=420)
        st.caption(f"{int((prof.missing_n > 0).sum())} of {len(prof)} columns have missing values. Overall missing cells: {df.isna().mean().mean():.2%}.")
    with t3:
        st.markdown("### ⚙️ Configure the audit")
        st.markdown("#### 1. Target (outcome) column")
        sugg = data_io.suggest_target_columns(df)
        opts = ["— select —"] + sugg + [c for c in df.columns if c not in sugg]
        t = st.selectbox("Which column does the model predict?", opts, index=pick_index(opts, S.target), help="No column name is assumed. Suggestions list columns with 2-10 distinct values.")
        new_t = None if t == "— select —" else t
        if new_t != S.target:
            S.target = new_t; S.pos = None; S.compat = None; clear_results()
        if S.target:
            vals = sorted(df[S.target].dropna().astype(str).unique())
            if len(vals) > 20:
                st.error(f"'{S.target}' has {len(vals)} distinct values - it looks continuous. This app audits **classification** models.")
            elif len(vals) == 2 or len(vals) > 2:
                if len(vals) == 2:
                    pos = st.selectbox("Positive class (the event being predicted, e.g. 'readmitted')", vals, index=pick_index(vals, str(S.pos), len(vals) - 1))
                    if pos != S.pos:
                        S.pos = pos; clear_results()
                else:
                    st.info(f"Multiclass target with {len(vals)} classes: {vals[:8]}. Multiclass metrics are used; binary-only checks are marked Not Evaluated.")
            st.dataframe(df[S.target].value_counts(dropna=False).rename("count").to_frame().T, width="stretch")
        st.markdown("#### 2. Identifier columns")
        det = data_io.detect_identifier_columns(df, S.target)
        if det:
            st.dataframe(pd.DataFrame(det), width="stretch", hide_index=True)
        ids = st.multiselect("Confirm identifier columns (should NOT be model inputs)", [c for c in df.columns if c != S.target], default=[c for c in S.ids if c in df.columns])
        if ids != S.ids:
            S.ids = ids; S.compat = None; clear_results()
        c1, c2 = st.columns(2)
        pat = c1.selectbox("Patient identifier (rows of one patient share it)", ["— none —"] + S.ids, index=pick_index(["— none —"] + S.ids, S.patient))
        rec = c2.selectbox("Record identifier (should be unique per row)", ["— none —"] + S.ids, index=pick_index(["— none —"] + S.ids, S.record_id))
        S.patient = None if pat == "— none —" else pat
        S.record_id = None if rec == "— none —" else rec
        st.markdown("#### 3. Demographic columns (for fairness and subgroup missingness)")
        cand = [c for c in df.columns if c != S.target and c not in S.ids]
        demo = st.multiselect("Select the demographic / subgroup columns that exist in your data", cand, default=[c for c in S.demo if c in cand],
                              help="Not every dataset has all of age, sex, race, ethnicity. Choose what is available; numeric columns such as age in years are binned automatically.")
        if demo != S.demo:
            S.demo = demo; clear_results()
        st.markdown("#### 4. Suitability for each audit")
        checks = []
        checks.append(("At least 100 rows", len(df) >= 100, f"{len(df):,} rows"))
        checks.append(("Target selected", S.target is not None, S.target or "choose one above"))
        if S.target:
            vc = df[S.target].value_counts()
            checks.append(("Target has >= 2 classes", len(vc) >= 2, f"{len(vc)} classes"))
            checks.append(("Smallest class has >= 20 rows (needed for stable metrics)", vc.min() >= 20, f"{int(vc.min())} rows"))
            checks.append(("No missing target values", df[S.target].isna().sum() == 0, f"{int(df[S.target].isna().sum())} missing"))
        checks.append(("Identifier column(s) declared (leakage checks)", len(S.ids) > 0, f"{len(S.ids)} declared"))
        checks.append(("Demographic column(s) declared (fairness)", len(S.demo) > 0, f"{len(S.demo)} declared"))
        st.dataframe(pd.DataFrame([{"check": a, "result": "✅" if b else "⚠️", "detail": c} for a, b, c in checks]), hide_index=True, width="stretch")
        st.button("Continue → Model Upload", on_click=goto, args=("Model Upload",))


# ============================================================== MODEL UPLOAD ===
def page_model():
    ui.page_head("02", "Model Upload & Compatibility", "The model is checked against your dataset before any audit uses it.")
    if S.df is None:
        st.info("Upload a dataset first (the model is checked against it)."); return
    st.error("**Security warning.** " + model_io.SECURITY_WARNING)
    trust = st.checkbox("I created this model file or fully trust its source.", key="trust")
    up = st.file_uploader("Upload a trained model (.pkl / .joblib)", type=["pkl", "pickle", "joblib"], disabled=not trust)
    if up is not None and trust:
        raw = up.getvalue()
        key = hashlib.sha256(raw).hexdigest()
        if key != S.model_key:
            try:
                with st.spinner("Loading model..."):
                    S.model, S.load_warnings = model_io.load_model(raw, up.name)
                S.mi, S.model_name, S.model_key = model_io.inspect_model(S.model), up.name, key
                S.compat = None; S.res_perf = S.res_fair = S.res_expl = S.res_leak = None; S.demo_loaded = None
            except ValueError as e:
                S.model, S.mi, S.compat = None, None, None
                st.error(str(e)); return
    if S.model is None:
        st.info("No model loaded. You can still run the dataset-only audits (Scenario A)."); return
    mi: model_io.ModelInfo = S.mi
    ui.kpi_row([("Model", mi.type_name, S.model_name, ui.VIOLET), ("Probabilities", "Yes" if mi.has_predict_proba else "No", "predict_proba()", ui.TEAL if mi.has_predict_proba else ui.CORAL),
                ("Preprocessing", {True: "Included", False: "Missing", None: "Unknown"}[mi.has_preprocessing], f"{len(mi.preprocessing_steps)} step(s)", ui.SKY),
                ("Family", mi.explain_family, "for explainability", ui.AMBER)])
    st.write("")
    with st.expander("Full model properties"):
      st.dataframe(pd.DataFrame([
        ("File", S.model_name), ("Python type", mi.type_name), ("Has predict()", mi.has_predict), ("Has predict_proba()", mi.has_predict_proba),
        ("Classes", [str(c) for c in mi.classes]), ("Stores feature names", mi.feature_names is not None), ("Number of input features", mi.n_features),
        ("Is a scikit-learn Pipeline / wrapper", mi.is_pipeline), ("Preprocessing included", {True: "Yes", False: "No", None: "Unknown"}[mi.has_preprocessing]),
        ("Preprocessing steps", ", ".join(mi.preprocessing_steps) or "—"), ("Calibration wrapper", mi.calibrated_wrapper), ("Estimator family", mi.explain_family),
        ("scikit-learn here", mi.sklearn_version)], columns=["property", "value"]).astype(str), hide_index=True, width="stretch")
    for n in mi.notes:
        st.warning(n)
    for w in S.load_warnings:
        st.warning(f"While loading: {w[:300]}")
    if mi.feature_names is None:
        cand = [c for c in S.df.columns if c != S.target and c not in S.ids]
        S.user_feats = st.multiselect("This model does not store feature names - select its input columns **in training order**", [c for c in S.df.columns if c != S.target],
                                      default=S.user_feats or cand)
    refresh_compat()
    comp = S.compat
    st.markdown("### Compatibility with the uploaded dataset")
    if comp is None:
        st.info("Select the target column on the Dataset page first.")
    elif comp.ok:
        st.success(f"Compatible: the model produced predictions for a trial sample ({len(comp.feature_cols)} input columns).")
        for w in comp.warnings:
            st.warning(w)
    else:
        for e in comp.errors:
            st.error(e)
        st.caption("Audits that need the model are disabled until this is resolved. Dataset-only audits remain available.")
    if not (comp and comp.ok):
        return
    st.subheader("Evaluation setup")
    c1, c2 = st.columns(2)
    with c1:
        labels = {k: v for k, v in perf_mod.PROVENANCE.items()}
        S.provenance = st.radio("What are the rows you uploaded?", list(labels), format_func=lambda k: labels[k], index=list(labels).index(S.provenance),
                                help="This controls how metrics are labelled. The app cannot verify your answer.")
        lvl, txt = perf_mod.provenance_banner(S.provenance)
        getattr(st, lvl)(txt)
    with c2:
        S.threshold = st.slider("Decision threshold for the positive class", 0.01, 0.99, float(S.threshold), 0.01,
                                help="Predicted probability at/above which a case is flagged. 0.5 is only a default: for rare outcomes well-calibrated models often never reach it. "
                                     "Select it on a separate validation split, not on the rows you report.")
        others = [c for c in S.df.columns if c != S.target and S.df[c].nunique() <= 10]
        sc = st.selectbox("Optional: evaluate only rows where this 'split' column equals…", ["— use all rows —"] + others, index=pick_index(["— use all rows —"] + others, S.split_col))
        S.split_col = None if sc == "— use all rows —" else sc
        if S.split_col:
            vals = sorted(S.df[S.split_col].dropna().astype(str).unique())
            S.split_val = st.selectbox("…this value", vals, index=pick_index(vals, S.split_val))
        else:
            S.split_val = None
    st.subheader("Optional: training dataset (for train-test contamination checks)")
    tr = st.file_uploader("Upload the data the model was trained on (.csv / .xlsx)", type=["csv", "xlsx"], key="train_up")
    if tr is not None and tr.name != S.train_name:
        try:
            S.train_df, _ = data_io.load_table(io.BytesIO(tr.getvalue()), tr.name, TOKENS_ON)
            S.train_name = tr.name; S.res_leak = None
        except Exception as e:
            st.error(f"Could not read training file: {e}")
    if S.train_df is not None:
        st.success(f"Training data loaded: `{S.train_name}` ({len(S.train_df):,} rows).")
    else:
        st.caption("Without it (or a split column) overlap between training and evaluation rows cannot be tested.")
    st.button("Continue → Data Quality", on_click=goto, args=("Data Quality Audit",))


# ============================================================ DATA QUALITY ====
def page_dq():
    ui.page_head("03", "Data Quality Audit", "Missingness, duplicates, outliers, balance and a transparent 0-100 scorecard.")
    if S.df is None:
        st.info("Upload a dataset first."); return
    df = S.df
    sig = (S.file_key, S.target, str(S.pos), tuple(S.ids), S.patient, S.record_id, tuple(S.demo))
    if S.dq_sig != sig or S.res_dq is None:
        with st.spinner("Running data-quality checks..."):
            S.res_dq = pl.run_data_quality(df, S.target, S.pos, S.ids, S.patient, S.record_id, S.demo)
            S.dq_sig = sig
    r = S.res_dq
    if not S.target:
        st.warning("No target column selected: class-imbalance and outcome-linked checks are Not Evaluated.")
    sc = r["scorecard"]
    overall = sc.loc["OVERALL (mean of evaluated dimensions)"]
    g1, g2, g3 = st.columns([1, 1, 1.1])
    g1.plotly_chart(ui.fig_gauge(float(overall.score), overall.grade), width="stretch")
    g2.plotly_chart(ui.fig_status_donut(r["findings"], "Data-quality checks"), width="stretch")
    with g3:
        cn = counts(r["findings"])
        for s_ in (FAILED, WARNING, PASSED, NOT_EVALUATED):
            st.markdown(f"<div class='tile' style='margin-bottom:.45rem'><span class='dot' style='--c:{ui.STATUS_UI[s_]}'></span><b>{cn[s_]}</b> &nbsp;{s_}</div>", unsafe_allow_html=True)
    tabs = st.tabs(["Scorecard", "Missing values", "Duplicates", "Outliers", "Class balance", "Correlation", "Validity & structure", "All findings", "Cleaning"])
    with tabs[0]:
        st.plotly_chart(ui.fig_scorecard(r["scorecard"]), width="stretch")
        st.dataframe(ui.fmt_df(r["scorecard"][["score", "grade", "note"]], 1), width="stretch")
        with st.expander("How is this score calculated?"):
            st.write(r["score_method"])
    with tabs[1]:
        if len(r["missing"]):
            st.plotly_chart(ui.fig_missing(r["missing"]), width="stretch")
            st.dataframe(ui.fmt_df(r["missing"], 2), width="stretch")
        else:
            st.success("No missing values.")
        mg = r["missing_by_group"]
        st.markdown("##### Does missingness differ between demographic groups?")
        if len(mg):
            st.dataframe(ui.fmt_df(mg.head(15), 1), width="stretch", hide_index=True)
            for attr, tbl in r["missing_by_group_tables"].items():
                st.plotly_chart(__import__("plotly.express", fromlist=["x"]).imshow(tbl, text_auto=".0f", color_continuous_scale="Blues", aspect="auto", title=f"% missing by {attr}"), width="stretch")
        else:
            st.info("Select demographic columns on the Dataset page (and need columns with >=100 missing values).")
        mo = r["missing_vs_outcome"]
        st.markdown("##### Is missingness linked to the outcome?")
        if mo is not None and len(mo):
            st.dataframe(ui.fmt_df(mo, 3), width="stretch", hide_index=True)
        else:
            st.info("Needs a binary target and columns with >=100 missing values.")
    with tabs[2]:
        d = r["duplicates"]
        st.dataframe(pd.DataFrame([(k.replace("_", " "), v) for k, v in d.items() if v is not None], columns=["measure", "value"]).astype(str), hide_index=True, width="stretch")
    with tabs[3]:
        st.caption("Outliers are flagged only - never removed. In healthcare, extreme values are often real.")
        if len(r["outliers"]):
            st.dataframe(ui.fmt_df(r["outliers"], 2), width="stretch")
            import plotly.express as px
            col = st.selectbox("Show distribution of", list(r["outliers"].index))
            st.plotly_chart(px.box(df, x=col, points="outliers", color_discrete_sequence=[ui.TEAL]), width="stretch")
        else:
            st.info("No numeric feature columns.")
    with tabs[4]:
        b = r["balance"]
        if b:
            st.plotly_chart(ui.fig_class(b["counts"], b["label"]), width="stretch")
            st.markdown(f"Majority : minority = **{b['imbalance_ratio']:.1f} : 1** ({b['label']}). Always predicting the majority class gives **{b['majority_baseline_accuracy']:.1%}** accuracy, so accuracy alone is misleading.")
            if len(r["representation"]):
                st.markdown("##### Subgroup representation and outcome rate")
                st.dataframe(ui.fmt_df(r["representation"], 2), width="stretch", hide_index=True)
        else:
            st.info("Select a target column.")
    with tabs[5]:
        if len(r["corr_pairs"]):
            st.plotly_chart(ui.fig_corr(r["corr_matrix"]), width="stretch")
            st.markdown("Strongest pairs (Spearman)")
            st.dataframe(ui.fmt_df(r["corr_pairs"], 3), width="stretch", hide_index=True)
        else:
            st.info("Fewer than two usable numeric columns.")
    with tabs[6]:
        st.markdown("**Invalid-value rules**")
        st.dataframe(r["validity"] if len(r["validity"]) else pd.DataFrame({"result": ["No violations of built-in generic rules."]}), hide_index=True, width="stretch")
        st.markdown(f"**Constant columns:** {', '.join(r['constant']) or 'none'}")
        st.markdown(f"**Near-constant (>=99% one value):** {', '.join(r['near_constant']) or 'none'}")
        if len(r["cardinality"]):
            st.markdown("**Cardinality / rare levels**"); st.dataframe(ui.fmt_df(r["cardinality"], 2), hide_index=True, width="stretch")
        st.caption("Domain-specific validity rules (plausible age range, etc.) are not assumed; the checks above are generic.")
    with tabs[7]:
        ui.show_findings(r["findings"], "dq")
    with tabs[8]:
        st.markdown("Every step is **optional and logged**; the original upload is never changed, and outliers are never removed automatically.")
        c1, c2 = st.columns(2)
        o_dup = c1.checkbox("Remove exact duplicate rows"); o_const = c1.checkbox("Drop constant columns")
        o_miss = c1.checkbox("Drop columns with missing share ≥ threshold"); thr = c1.slider("Missing-share threshold", .3, 1.0, .9, .05, disabled=not o_miss)
        o_cat = c2.checkbox("Fill missing categorical values with 'Unknown'"); o_num = c2.checkbox("Impute missing numeric values with the median")
        if o_num:
            c2.caption("⚠ For model training, fit imputers on the training split only; this is a convenience copy.")
        if st.button("Create cleaned copy", type="primary"):
            prot = [c for c in [S.target] + S.ids if c]
            cl, log = dq.clean_dataset(df, drop_exact_duplicates=o_dup, drop_constant_columns=o_const, drop_missing_above=thr if o_miss else None,
                                       fill_categorical_unknown=o_cat, impute_numeric_median=o_num, protect_columns=prot)
            S.cleaned, S.cleaning_log = cl, log
        if S.cleaned is not None:
            st.success(f"Cleaned copy: {S.cleaned.shape[0]:,} rows × {S.cleaned.shape[1]} columns (original: {len(df):,} × {df.shape[1]}).")
            st.dataframe(S.cleaning_log, hide_index=True, width="stretch")
            st.download_button("⬇ Download cleaned dataset (CSV)", S.cleaned.to_csv(index=False).encode(), "cleaned_dataset.csv", "text/csv")
            st.download_button("⬇ Download cleaning log (CSV)", S.cleaning_log.to_csv(index=False).encode(), "cleaning_log.csv", "text/csv")


# ================================================== LEAKAGE & PERFORMANCE =====
def page_leak_perf():
    ui.page_head("04", "Leakage & Performance", "Could the model be cheating, and how well does it really perform?")
    if not target_ready():
        ui.requirement_box(["a dataset" if S.df is None else "a target column (Dataset page)"], "Leakage and performance audits"); return
    tab_l, tab_p = st.tabs(["A. Data leakage", "B. Model performance & calibration"])
    ed = eval_df()
    comp = S.compat
    feats = comp.feature_cols if model_ready() else [c for c in S.df.columns if c != S.target]
    # ------------------------------------------------------------ leakage ----
    with tab_l:
        if not model_ready():
            st.info("No compatible model: checking **all non-target columns as potential model inputs**. Checks that need the model's real feature list or training data are marked accordingly.")
        st.markdown("**Prediction context** - when will the model be used, and what is known at that moment? (e.g. *at discharge*, *at admission*)")
        S.prediction_context = st.text_input("Prediction time / context", S.prediction_context, placeholder="e.g. at hospital discharge, predicting 30-day readmission")
        st.caption("Tag each feature. 'After prediction time' = confirmed leakage. Leave 'Not reviewed' if unknown - the app then reports the check as not evaluated.")
        opts = ["Not reviewed", "Before/at prediction time", "After prediction time", "Unsure"]
        cur = S.availability or {}
        base = pd.DataFrame({"feature": feats, "availability": [cur.get(f, "Not reviewed") for f in feats]})
        edited = st.data_editor(base, hide_index=True, width="stretch", height=min(420, 38 * len(base) + 40), key=f"avail_{len(feats)}_{S.target}",
                                column_config={"feature": st.column_config.TextColumn(disabled=True), "availability": st.column_config.SelectboxColumn(options=opts, required=True)})
        S.availability = dict(zip(edited.feature, edited.availability))
        split_ok = S.split_col is not None and S.split_val is not None
        if S.train_df is None and not split_ok:
            st.caption("To test train/test contamination and patient overlap, upload the training data on the Model Upload page, or choose a split column there.")
        if st.button("Run leakage audit", type="primary"):
            with st.spinner("Running leakage checks..."):
                other = None                                   # the 'training' value of the split column, if it can be inferred
                if split_ok:
                    vals = [v for v in S.df[S.split_col].astype(str).unique() if v != str(S.split_val)]
                    other = vals[0] if len(vals) == 1 else None
                use_split = S.train_df is None and other is not None
                S.res_leak = pl.run_leakage(
                    S.df if use_split or S.train_df is None else ed, S.target, feats, S.ids, S.patient, S.pos,
                    train_df=S.train_df, split_col=S.split_col if use_split else None, train_value=other, test_value=S.split_val,
                    availability=S.availability if any(v != "Not reviewed" for v in S.availability.values()) else None,
                    prediction_context=S.prediction_context)
        r = S.res_leak
        if r:
            ui.status_metrics(r["findings"])
            st.markdown("Categories: **Confirmed** · **Potential** · **Suspicious** · **Cannot be performed** (missing information)")
            ui.show_findings(r["findings"], "leak")
            fs = r["tables"].get("feature_strength")
            if fs is not None and len(fs):
                with st.expander("Single-feature strength vs target (top 15)"):
                    st.dataframe(ui.fmt_df(fs.head(15), 3), hide_index=True, width="stretch")
            st.info("Passing these checks does **not** prove leakage is absent; subtle temporal leakage and preprocessing fitted on all data cannot be detected automatically.")
    # -------------------------------------------------------- performance ----
    with tab_p:
        if not model_ready():
            reasons = ["no model uploaded (Model Upload page)"] if S.model is None else (S.compat.errors if S.compat else ["compatibility not checked"])
            ui.requirement_box(reasons, "Model performance evaluation"); return
        lvl, txt = perf_mod.provenance_banner(S.provenance)
        getattr(st, lvl)(txt)
        st.caption(f"Evaluating {len(ed):,} rows · threshold {S.threshold:.2f} · settings on the Model Upload page.")
        if st.button("Run performance evaluation", type="primary"):
            try:
                with st.spinner("Predicting and computing metrics..."):
                    S.res_perf = pl.run_performance(S.model, S.mi, ed, S.target, comp.feature_cols, S.pos, S.provenance, S.threshold)
                    S.res_fair = S.res_expl = None
            except Exception as e:
                S.res_perf = None
                st.error(f"Evaluation failed: {e}")
        r = S.res_perf
        if not r:
            return
        m = r["metrics"]
        if r["kind"] == "binary":
            tv = m["table"].set_index("metric").value
            ui.kpi_row([("ROC-AUC", f"{m['roc_auc']:.3f}" if "roc_auc" in m else "n/a", "discrimination", ui.TEAL), ("PR-AUC", f"{m['pr_auc']:.3f}" if "pr_auc" in m else "n/a", f"chance {m['prevalence']:.3f}", ui.VIOLET),
                        ("Recall", f"{tv['Recall / Sensitivity']:.1%}", "cases caught", ui.SKY), ("Precision", f"{tv['Precision (PPV)']:.1%}", "flags that are right", ui.AMBER),
                        ("Brier", f"{m['brier']:.3f}" if "brier" in m else "n/a", f"baseline {m['brier_baseline']:.3f}" if "brier_baseline" in m else "", ui.CORAL)])
            st.write("")
        ui.status_tiles(r["findings"])
        if r["kind"] == "binary":
            t = m["table"].copy()
            t["95% CI"] = [("" if pd.isna(a) else f"{a:.3f} – {b:.3f}") for a, b in zip(t.ci_low, t.ci_high)]
            st.dataframe(t[["metric", "value", "95% CI", "note"]].style.format({"value": "{:.4f}"}), hide_index=True, width="stretch")
            if "roc_curve" in m:
                st.plotly_chart(ui.fig_roc_pr_cal(m), width="stretch")
            c = m["confusion"]
            cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("##### Confusion matrix")
                st.plotly_chart(ui.fig_confusion(cm, ["negative", "positive"]), width="stretch")
            with col2:
                if "threshold_table" in r:
                    st.markdown("##### Threshold table (descriptive only)")
                    st.dataframe(ui.fmt_df(r["threshold_table"], 3), hide_index=True, width="stretch", height=330)
                    st.caption("Do not pick a threshold from this table and then report the same rows as independent performance - choose it on a separate validation split.")
        else:
            st.dataframe(m["table"].style.format({"value": "{:.4f}"}), hide_index=True, width="stretch")
            st.plotly_chart(ui.fig_confusion(m["confusion_matrix"].values, r["classes"]), width="stretch")
            st.dataframe(ui.fmt_df(m["per_class"], 3), hide_index=True, width="stretch")
        ui.show_findings(r["findings"], "perf")


# ================================================================ FAIRNESS ====
def page_fairness():
    ui.page_head("05", "Fairness Audit", "Does the model perform differently across patient subgroups?")
    if S.res_perf is None:
        miss = []
        if not target_ready(): miss.append("dataset and target column")
        if S.model is None: miss.append("a trained model")
        elif not model_ready(): miss.append("a compatible model (see Model Upload)")
        if not miss: miss.append("run the performance evaluation first (Leakage & Performance → B)")
        ui.requirement_box(miss, "Fairness audit"); return
    if not S.demo:
        ui.requirement_box(["at least one demographic column (Dataset page → Configure the audit)"], "Fairness audit"); return
    st.info("Subgroup differences are **observed disparities**, not proof of discrimination. See the limitations box below.")
    ed = eval_df()
    lvl, txt = perf_mod.provenance_banner(S.provenance)
    getattr(st, lvl)(txt)
    demo = st.multiselect("Demographic columns to audit", [c for c in ed.columns if c != S.target], default=[c for c in S.demo if c in ed.columns])
    if st.button("Run fairness audit", type="primary"):
        with st.spinner("Computing subgroup metrics..."):
            S.res_fair = pl.run_fairness(S.res_perf, ed.reset_index(drop=True), demo)
    r = S.res_fair
    if not r:
        return
    ui.status_tiles(r["findings"])
    st.write("")
    sel = st.radio("Attribute", list(r["attributes"]), horizontal=True, label_visibility="collapsed") if r["attributes"] else None
    for attr, a in ([(sel, r["attributes"][sel])] if sel else []):
        st.markdown(f"### {attr}")
        if a["note"]:
            st.caption(a["note"])
        t = a["table"]
        lim = t[t.data_adequacy.str.startswith("Limited")]
        if len(lim):
            st.warning(f"Small-sample warning: {', '.join(f'{g} (n={n})' for g, n in zip(lim.group, lim.n))}. Results for these groups are unreliable.")
        if S.res_perf["kind"] == "binary":
            st.plotly_chart(ui.fig_group_heatmap(t), width="stretch")
            show = [c for c in ["group", "n", "positives", "accuracy", "precision", "recall_sensitivity", "specificity", "false_positive_rate", "false_negative_rate", "f1", "selection_rate", "data_adequacy"] if c in t.columns]
            st.dataframe(ui.fmt_df(t[show], 3), hide_index=True, width="stretch")
            c1, c2 = st.columns(2)
            c1.plotly_chart(ui.fig_subgroup_bars(t, "recall_sensitivity", "recall_ci_low", "recall_ci_high", "Recall / sensitivity (95% CI)"), width="stretch")
            c2.plotly_chart(ui.fig_subgroup_bars(t, "false_positive_rate", "fpr_ci_low", "fpr_ci_high", "False-positive rate (95% CI)"), width="stretch")
            c3, c4 = st.columns(2)
            c3.plotly_chart(ui.fig_subgroup_bars(t, "selection_rate", title="Share flagged (selection rate)"), width="stretch")
            c4.plotly_chart(ui.fig_subgroup_bars(t, "accuracy", "accuracy_ci_low", "accuracy_ci_high", "Accuracy (95% CI)"), width="stretch")
            if not a["disparity"].empty:
                st.markdown("**Largest differences between adequately sized groups**")
                st.dataframe(ui.fmt_df(a["disparity"], 3), hide_index=True, width="stretch")
            if "mean_predicted_prob" in t.columns:
                with st.expander("Subgroup calibration"):
                    st.dataframe(ui.fmt_df(t[["group", "n", "mean_predicted_prob", "observed_rate", "calibration_gap", "brier"]], 4), hide_index=True, width="stretch")
                    st.caption("Positive gap = the model over-predicts risk for that group on average.")
                    if a["curves"]:
                        st.plotly_chart(ui.fig_group_calibration(a["curves"], f"Calibration by {attr} (groups with >=200 rows)"), width="stretch")
        else:
            st.dataframe(ui.fmt_df(t, 3), hide_index=True, width="stretch")
    st.markdown("### Findings")
    ui.show_findings(r["findings"], "fair")
    with st.expander("Limitations of these fairness metrics", expanded=True):
        st.write(r["limitations"])


# ============================================================ EXPLAINABILITY ==
def page_explain():
    ui.page_head("06", "Model Explainability", "Which inputs drive the model's predictions (SHAP, importance, per-patient view).")
    if not model_ready() or S.res_perf is None:
        miss = ["a compatible model (Model Upload)"] if not model_ready() else ["run the performance evaluation first (needed for true labels)"]
        ui.requirement_box(miss, "Explainability"); return
    comp, ed, rp = S.compat, eval_df(), S.res_perf
    fam = S.mi.explain_family
    st.caption(f"Estimator family: **{fam}**. SHAP supports tree and linear models; permutation importance works for any model; unsupported methods are reported, never faked.")
    c1, c2 = st.columns(2)
    n_shap = c1.slider("Rows used for SHAP", 50, 500, 200, 50, help="Module 3 used 200 rows.")
    do_perm = c2.checkbox("Also compute permutation importance (slower, model-agnostic)", value=True)
    if st.button("Run explainability", type="primary"):
        with st.spinner("Computing explanations (this can take a minute for large models)..."):
            S.res_expl = pl.run_explainability(S.model, S.mi, ed, comp.feature_cols, rp["y_idx"], rp["classes"], rp["pos_idx"], do_perm, n_shap)
    r = S.res_expl
    if not r:
        return
    for n in r["notes"]:
        st.warning(n)
    tabs = st.tabs(["SHAP", "Feature importance", "Individual prediction", "Findings"])
    sh = r["shap"]
    with tabs[0]:
        if "error" in sh:
            st.info("SHAP unavailable: " + sh["error"])
        else:
            st.caption(sh["method"] + ". " + sh["note"])
            st.plotly_chart(ui.fig_importance(sh["importance"], "mean_abs_shap", "Mean |SHAP| - top 20 (transformed features)"), width="stretch")
            fig = ex_mod.summary_plot_figure(sh)
            st.pyplot(fig, clear_figure=True)
    with tabs[1]:
        if r["native"] is not None:
            st.markdown(f"**Model-specific:** {r['native'].attrs.get('kind')}")
            st.plotly_chart(ui.fig_importance(r["native"], "importance", "Native importance (top 20)"), width="stretch")
        else:
            st.info("This estimator has no native importance attribute.")
        if r.get("permutation") is not None:
            p = r["permutation"]
            st.markdown(f"**Permutation importance** ({p.attrs.get('scoring')} drop, {p.attrs.get('n_rows')} rows; on original input columns)")
            st.plotly_chart(ui.fig_importance(p, "importance_mean", "Permutation importance (top 20)"), width="stretch")
    with tabs[2]:
        if "error" in sh:
            st.info("Individual explanations need SHAP, which is unavailable for this model.")
        else:
            pos = st.number_input("Sample row number", 0, len(sh["values"]) - 1, 0)
            row_label = sh["sample_index"][pos]
            pr = rp["proba"]
            st.caption(f"Row index in your evaluation data: {row_label}")
            loc = ex_mod.local_explanation(sh, int(pos))
            import plotly.express as px
            fig = px.bar(loc.iloc[::-1], x="shap_value", y="feature", orientation="h", color=loc.iloc[::-1].shap_value > 0, color_discrete_map={True: "#C0392B", False: "#3B7DD8"})
            fig.update_layout(showlegend=False, height=420, title="Contribution to the model output (red = pushes towards the positive class)", margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(ui.fmt_df(loc, 4), hide_index=True, width="stretch")
    with tabs[3]:
        ui.show_findings(r["findings"], "expl")


# ================================================================= REPORT =====
def build_ctx():
    df = S.df
    ctx = {"project": {"title": S.get("rep_title") or "Automated Audit Framework for Healthcare AI Models", "auditor": S.get("rep_auditor", ""), "context": S.prediction_context or S.get("rep_context", "")}}
    ctx["dataset"] = {"File": S.filename, "Rows": f"{len(df):,}", "Columns": df.shape[1], "SHA-256 (first 16)": hashlib.sha256(S.raw_bytes).hexdigest()[:16] if S.raw_bytes else "(demo file)",
                      "Target column": S.target, "Positive class": S.pos, "Identifier columns": ", ".join(S.ids) or "none declared", "Demographic columns": ", ".join(S.demo) or "none declared",
                      "Missing cells": f"{df.isna().mean().mean():.2%}"}
    if S.model is not None and S.mi is not None:
        mi = S.mi
        ctx["model"] = {"File": S.model_name, "Type": mi.type_name, "Compatible with dataset": "Yes" if model_ready() else "No - " + "; ".join(S.compat.errors if S.compat else ["not checked"]),
                        "Probability output": mi.has_predict_proba, "Preprocessing included": {True: "Yes", False: "No", None: "Unknown"}[mi.has_preprocessing],
                        "Preprocessing steps": ", ".join(mi.preprocessing_steps) or "-", "Classes": [str(c) for c in mi.classes], "Input features": len(S.compat.feature_cols) if model_ready() else "-"}
    ctx["dq"], ctx["leak"], ctx["perf"], ctx["fair"], ctx["expl"] = S.res_dq, S.res_leak, S.res_perf, S.res_fair, S.res_expl
    if S.res_perf:
        ctx["provenance_text"] = perf_mod.PROVENANCE[S.provenance]; ctx["threshold"] = S.threshold
    ctx["cleaning_log"] = S.cleaning_log
    return ctx


def page_report():
    ui.page_head("07", "Final Audit Report", "One downloadable report: findings, risks, limitations and recommendations.")
    if S.df is None:
        st.info("Upload a dataset first."); return
    st.markdown("The report includes only audits that were **actually run**; the others appear as *Not Evaluated* with the reason.")
    c1, c2 = st.columns(2)
    S["rep_title"] = c1.text_input("Report title", S.get("rep_title", "Automated Audit Framework for Healthcare AI Models"))
    S["rep_auditor"] = c2.text_input("Prepared by (group / names)", S.get("rep_auditor", ""))
    S["rep_context"] = st.text_area("Purpose and intended use of the model being audited", S.get("rep_context", S.prediction_context), height=70)
    status = [("Data quality", S.res_dq), ("Leakage", S.res_leak), ("Performance", S.res_perf), ("Fairness", S.res_fair), ("Explainability", S.res_expl)]
    st.dataframe(pd.DataFrame([{"section": n, "status": "✅ run" if r is not None else "⏸ not run"} for n, r in status]), hide_index=True, width="stretch")
    if S.res_dq is None:
        st.caption("Open the Data Quality page once to run that audit automatically.")
    if st.button("Generate report", type="primary"):
        with st.spinner("Building PDF..."):
            ctx = build_ctx()
            S["pdf_bytes"] = report.build_pdf(ctx)
            S["json_text"] = report.build_json(ctx)
            S["csv_bytes"] = report.findings_dataframe(ctx).to_csv(index=False).encode()
            S["rep_counts"] = counts(report.all_findings(ctx))
    if S.get("pdf_bytes"):
        cn = S["rep_counts"]
        cols = st.columns(4)
        for col, s in zip(cols, STATUSES):
            col.metric(f"{ui.ICON[s]} {s}", cn[s])
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ PDF report", S["pdf_bytes"], "healthcare_ai_audit_report.pdf", "application/pdf", type="primary")
        d2.download_button("⬇ Structured results (JSON)", S["json_text"], "audit_results.json", "application/json")
        d3.download_button("⬇ Findings table (CSV)", S["csv_bytes"], "audit_findings.csv", "text/csv")
        st.caption("Regenerate after running more audits. The report is not a certification or clinical validation.")


# ----------------------------------------------------------------- router ----
ROUTES = {"Home": page_home, "Dataset Upload": page_dataset, "Model Upload": page_model, "Data Quality Audit": page_dq,
          "Leakage & Performance": page_leak_perf, "Fairness Audit": page_fairness, "Model Explainability": page_explain, "Final Audit Report": page_report}
try:
    ROUTES[page]()
except Exception as e:                    # last-resort guard: show a readable error instead of a stack trace
    st.error(f"Something went wrong on this page: {type(e).__name__}: {e}")
    with st.expander("Technical details"):
        st.exception(e)
