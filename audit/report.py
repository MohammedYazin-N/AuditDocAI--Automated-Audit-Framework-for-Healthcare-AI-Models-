"""FEATURE 7 - downloadable audit report (PDF via ReportLab) + structured JSON / CSV export.

The report only contains sections that were actually run; sections without results are printed as
"Not Evaluated" with the reason. It never states that a model is fair, safe, validated, compliant or certified.
"""
from __future__ import annotations
import io
import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from .common import STATUSES, STATUS_COLOURS, counts, PASSED, WARNING, FAILED, NOT_EVALUATED

NAVY = colors.HexColor("#12355B")
DISCLAIMER = ("This report is the output of an automated, exploratory audit tool built for an academic capstone project. It is not a clinical validation, "
              "a regulatory submission, a safety assessment or a certification, and a passed check does not mean the model is safe, fair or fit for clinical use. "
              "All thresholds are documented audit heuristics, not official standards.")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontSize=22, textColor=NAVY, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, textColor=NAVY, spaceBefore=10, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, textColor=NAVY, spaceBefore=6, spaceAfter=3),
        "b": ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12),
        "s": ParagraphStyle("s", parent=ss["BodyText"], fontSize=7.5, leading=9.5),
        "note": ParagraphStyle("n", parent=ss["BodyText"], fontSize=8, leading=10.5, textColor=colors.HexColor("#444444")),
    }


def _esc(x) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tbl(rows, widths, st, header=True, font=7.5):
    data = [[Paragraph(_esc(c), st["s"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#B8C2CC")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2"))]
    t.setStyle(TableStyle(style))
    return t


def _df_table(df: pd.DataFrame, st, widths=None, max_rows=25, fmt=3):
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{fmt}f}")
    rows = [list(map(str, d.columns))] + d.astype(str).values.tolist()
    w = widths or [(A4[0] - 30 * mm) / len(d.columns)] * len(d.columns)
    return _tbl(rows, w, st)


def _findings_table(findings, st):
    W = A4[0] - 30 * mm
    rows = [["Check", "Status", "Finding", "Criterion"]]
    for f in findings:
        rows.append([f.check + (f" [{f.category}]" if f.category else ""), f.status, f.detail, f.criterion])
    t = _tbl(rows, [W * .21, W * .09, W * .42, W * .28], st)
    extra = []
    for i, f in enumerate(findings, start=1):
        extra.append(("TEXTCOLOR", (1, i), (1, i), colors.HexColor(STATUS_COLOURS[f.status])))
        extra.append(("BACKGROUND", (1, i), (1, i), colors.HexColor("#F4F6F8")))
    t.setStyle(TableStyle(extra))
    return t


def _fig_to_img(fig, width_mm=165):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    b.seek(0)
    from PIL import Image as PILImage
    w, h = PILImage.open(io.BytesIO(b.getvalue())).size
    b.seek(0)
    return Image(b, width=width_mm * mm, height=width_mm * mm * h / w)


def _charts(ctx):
    """matplotlib versions of the key charts for the PDF."""
    out = {}
    dqr, perf, fair = ctx.get("dq"), ctx.get("perf"), ctx.get("fair")
    if dqr is not None:
        sc = dqr["scorecard"].dropna(subset=["score"])
        fig, ax = plt.subplots(figsize=(7, 2.8))
        cmap = {"Good": "#1B873F", "Acceptable": "#E0B400", "Poor": "#E67E22", "Critical": "#C0392B"}
        ax.barh(sc.index[::-1], sc.score[::-1], color=[cmap[g] for g in sc.grade[::-1]])
        for i, v in enumerate(sc.score[::-1]):
            ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8)
        ax.set_xlim(0, 112); ax.set_title("Data-quality scorecard (project heuristic, 0-100)", fontsize=10)
        out["scorecard"] = fig
        m = dqr["missing"]
        if len(m):
            fig, ax = plt.subplots(figsize=(7, max(2, .28 * min(len(m), 15) + .8)))
            mm_ = m.head(15)
            ax.barh(mm_.index[::-1], mm_.missing_pct[::-1], color="#3B7DD8")
            ax.set_xlabel("% missing"); ax.set_title("Columns with missing values", fontsize=10)
            out["missing"] = fig
        bal = dqr.get("balance")
        if bal:
            fig, ax = plt.subplots(figsize=(5, 2.4))
            ax.bar([str(k) for k in bal["counts"]], list(bal["counts"].values()), color="#3B7DD8")
            ax.set_title(f"Class distribution ({bal['label']}, {bal['imbalance_ratio']:.1f}:1)", fontsize=10)
            out["balance"] = fig
    if perf is not None and perf["kind"] == "binary" and "roc_curve" in perf["metrics"]:
        mt = perf["metrics"]
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
        axs[0].plot(mt["roc_curve"].fpr, mt["roc_curve"].tpr, label=f"AUC={mt['roc_auc']:.3f}"); axs[0].plot([0, 1], [0, 1], "--", c="grey")
        axs[0].set_title("ROC", fontsize=9); axs[0].legend(fontsize=7); axs[0].set_xlabel("FPR"); axs[0].set_ylabel("TPR")
        axs[1].plot(mt["pr_curve"].recall, mt["pr_curve"].precision, label=f"AP={mt['pr_auc']:.3f}"); axs[1].axhline(mt["prevalence"], ls="--", c="grey")
        axs[1].set_title("Precision-Recall", fontsize=9); axs[1].legend(fontsize=7); axs[1].set_xlabel("Recall")
        cc = mt["calibration_curve"]
        axs[2].plot(cc.mean_predicted, cc.observed_rate, "o-"); axs[2].plot([0, 1], [0, 1], "--", c="grey")
        axs[2].set_title("Calibration (quantile bins)", fontsize=9); axs[2].set_xlabel("Mean predicted"); axs[2].set_ylabel("Observed")
        fig.tight_layout(); out["perf"] = fig
    if fair is not None:
        for attr, a in list(fair["attributes"].items())[:3]:
            t = a["table"]
            if "recall_sensitivity" in t:
                fig, ax = plt.subplots(figsize=(7, 2.6))
                x = np.arange(len(t))
                lo = (t.recall_sensitivity - t.recall_ci_low).clip(lower=0).fillna(0); hi = (t.recall_ci_high - t.recall_sensitivity).clip(lower=0).fillna(0)
                cols = ["#3B7DD8" if s.startswith("Adequate") else "#B0B7C0" for s in t.data_adequacy]
                ax.bar(x, t.recall_sensitivity.fillna(0), yerr=[lo, hi], color=cols, capsize=2)
                ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(t.group, t.n)], fontsize=6, rotation=0)
                ax.set_ylim(0, 1); ax.set_title(f"Recall by {attr} (95% Wilson CI; grey = limited data)", fontsize=9)
                out[f"fair_{attr}"] = fig
    ex = ctx.get("expl")
    if ex is not None and ex.get("shap") is not None and "error" not in ex["shap"]:
        imp = ex["shap"]["importance"].head(15)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(imp.feature[::-1], imp.mean_abs_shap[::-1], color="#3B7DD8"); ax.set_title("Mean |SHAP| (top 15)", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        out["shap"] = fig
    elif ex is not None and ex.get("permutation") is not None:
        imp = ex["permutation"].head(15)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.barh(imp.feature[::-1], imp.importance_mean[::-1], color="#3B7DD8"); ax.set_title("Permutation importance (top 15)", fontsize=10)
        out["perm"] = fig
    return out


def all_findings(ctx):
    F = []
    for k in ("dq", "leak", "perf", "fair", "expl"):
        if ctx.get(k) is not None:
            F += ctx[k]["findings"]
    return F


def recommendations(findings):
    seen, rec = set(), []
    for f in findings:
        if f.status in (FAILED, WARNING) and f.action and f.action not in seen:
            seen.add(f.action)
            rec.append((f.status, f.check, f.action))
    rec.sort(key=lambda r: 0 if r[0] == FAILED else 1)
    return rec


def _section_findings(story, st, title, findings, empty_msg):
    story.append(Paragraph(title, st["h1"]))
    if not findings:
        story.append(Paragraph(f"<b>Not Evaluated.</b> {_esc(empty_msg)}", st["b"]))
        return
    story.append(_findings_table(findings, st))


def build_pdf(ctx: dict) -> bytes:
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="AuditDoc AI - Audit Report", author=ctx.get("auditor", ""))
    W = A4[0] - 30 * mm
    S = []
    charts = _charts(ctx)
    info = ctx.get("project", {})
    ds, md = ctx.get("dataset", {}), ctx.get("model")
    F_all = all_findings(ctx)
    cn = counts(F_all)

    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
        c.drawString(15 * mm, 8 * mm, "AuditDoc AI - automated exploratory audit, not a certification")
        c.drawRightString(A4[0] - 15 * mm, 8 * mm, f"Page {d.page}")
        c.restoreState()

    # ---- 1 project and audit information
    S.append(Paragraph("AuditDoc AI - Audit Report", st["title"]))
    S.append(Paragraph(_esc(info.get("title", "Automated Audit Framework for Healthcare AI Models")), st["h2"]))
    S.append(_tbl([["Item", "Value"],
                   ["Report generated", datetime.now().strftime("%d %b %Y %H:%M")],
                   ["Prepared by / group", info.get("auditor") or "(not entered)"],
                   ["Audit purpose / prediction context", info.get("context") or "(not entered)"],
                   ["Evaluation data provenance", ctx.get("provenance_text", "(no model evaluation run)")],
                   ["Decision threshold", str(ctx.get("threshold", "n/a"))],
                   ["Software", "AuditDoc AI (Streamlit); audit logic reused from Modules 1-3 of the project notebook"]], [W * .3, W * .7], st))
    S.append(Spacer(1, 6))
    # ---- 11 (placed first for readers) overall summary
    S.append(Paragraph("Overall audit summary", st["h1"]))
    S.append(Paragraph(f"<b>{cn[FAILED]}</b> check(s) failed, <b>{cn[WARNING]}</b> raised a warning, <b>{cn[PASSED]}</b> passed and <b>{cn[NOT_EVALUATED]}</b> could not be evaluated "
                       "(out of the checks that were run). No composite pass/fail grade or certification is given.", st["b"]))
    areas = {}
    for f in F_all:
        areas.setdefault(f.area.split(":")[0], []).append(f)
    rows = [["Area"] + list(STATUSES)]
    for a, fs in areas.items():
        c = counts(fs)
        rows.append([a] + [str(c[s]) for s in STATUSES])
    S.append(_tbl(rows, [W * .4] + [W * .15] * 4, st))
    S.append(Spacer(1, 4))
    S.append(Paragraph(_esc(DISCLAIMER), st["note"]))
    S.append(Paragraph("Status meanings - <b>Passed</b>: the defined check was completed and met its documented criterion. <b>Warning</b>: a potential concern or limitation was identified. "
                       "<b>Failed</b>: a defined check detected an issue according to its documented criterion. <b>Not Evaluated</b>: required information or functionality was unavailable.", st["note"]))
    # ---- 2 dataset
    S.append(Paragraph("1. Uploaded dataset", st["h1"]))
    S.append(_tbl([["Item", "Value"]] + [[k, str(v)] for k, v in ds.items()], [W * .3, W * .7], st))
    # ---- 3 model
    S.append(Paragraph("2. Uploaded model", st["h1"]))
    if md:
        S.append(_tbl([["Item", "Value"]] + [[k, str(v)] for k, v in md.items()], [W * .3, W * .7], st))
    else:
        S.append(Paragraph("<b>Not Evaluated.</b> No compatible model was supplied; performance, fairness and explainability sections are unavailable.", st["b"]))
    # ---- 4 data quality
    dq = ctx.get("dq")
    _section_findings(S, st, "3. Data quality audit", dq["findings"] if dq else [], "No dataset audit was run.")
    if dq:
        S.append(Paragraph("Scorecard methodology", st["h2"])); S.append(Paragraph(_esc(dq["score_method"]), st["note"]))
        sc = dq["scorecard"].reset_index().rename(columns={"index": "dimension"})[["dimension", "score", "grade"]]
        S.append(_df_table(sc, st, [W * .5, W * .25, W * .25], fmt=1))
        for k in ("scorecard", "missing", "balance"):
            if k in charts:
                S.append(_fig_to_img(charts[k], 150))
        if ctx.get("cleaning_log") is not None and len(ctx["cleaning_log"]) > 1:
            S.append(Paragraph("Cleaning steps applied (original data is kept unchanged)", st["h2"]))
            S.append(_df_table(ctx["cleaning_log"], st, [W * .22, W * .09, W * .09, W * .4, W * .1, W * .1]))
    # ---- 5 leakage
    lk = ctx.get("leak")
    _section_findings(S, st, "4. Data leakage findings", lk["findings"] if lk else [], "Leakage checks need a dataset with a selected target.")
    if lk:
        S.append(Paragraph("Categories: <b>Confirmed</b> = a defined check detected leakage; <b>Potential</b> = a risk that needs confirming; "
                           "<b>Suspicious</b> = needs investigation; <b>Cannot be performed</b> = required information missing. Passing these automated checks does not prove "
                           "that leakage has been eliminated.", st["note"]))
        fs = lk["tables"].get("feature_strength")
        if fs is not None and len(fs):
            S.append(Paragraph("Strongest single-feature relationships with the target", st["h2"]))
            S.append(_df_table(fs[["feature", "metric", "strength"]].head(10), st, [W * .25, W * .5, W * .25]))
    # ---- 6 performance
    pf = ctx.get("perf")
    _section_findings(S, st, "5. Model performance", pf["findings"] if pf else [], "Needs a compatible model and evaluation data with true labels.")
    if pf:
        S.append(Paragraph(_esc(ctx.get("provenance_text", "")), st["note"]))
        m = pf["metrics"]
        t = m["table"].copy()
        if pf["kind"] == "binary":
            t["95% CI"] = [("" if pd.isna(a) else f"{a:.3f}-{b:.3f}") for a, b in zip(t.ci_low, t.ci_high)]
            t = t[["metric", "value", "95% CI", "note"]]
        else:
            t = t[["metric", "value", "note"]]
        S.append(_df_table(t, st, None if pf["kind"] != "binary" else [W * .28, W * .12, W * .15, W * .45], fmt=3))
        if pf["kind"] == "binary":
            c = m["confusion"]
            S.append(Spacer(1, 3))
            S.append(_tbl([["", "Predicted negative", "Predicted positive"], ["Actual negative", str(c["tn"]), f"{c['fp']} (false positives)"], ["Actual positive", f"{c['fn']} (false negatives)", str(c["tp"])]],
                          [W * .3, W * .35, W * .35], st))
        else:
            S.append(_df_table(m["confusion_matrix"].reset_index(), st))
        if "perf" in charts:
            S.append(_fig_to_img(charts["perf"], 175))
    # ---- 7 fairness
    fa = ctx.get("fair")
    _section_findings(S, st, "6. Fairness audit", fa["findings"] if fa else [], "Needs a compatible model, true outcomes and at least one demographic column.")
    if fa:
        S.append(Paragraph("Limitations of these fairness metrics", st["h2"])); S.append(Paragraph(_esc(fa["limitations"]), st["note"]))
        for attr, a in fa["attributes"].items():
            S.append(Paragraph(f"Subgroup results: {_esc(attr)}", st["h2"]))
            t = a["table"]
            cols = [c for c in ["group", "n", "positives", "accuracy", "precision", "recall_sensitivity", "specificity", "false_positive_rate", "false_negative_rate", "selection_rate", "data_adequacy"] if c in t.columns]
            S.append(_df_table(t[cols], st, None, max_rows=14, fmt=3))
            if f"fair_{attr}" in charts:
                S.append(_fig_to_img(charts[f"fair_{attr}"], 150))
    # ---- 8 explainability
    ex = ctx.get("expl")
    _section_findings(S, st, "7. Explainability", ex["findings"] if ex else [], "Needs a compatible model and feature data.")
    if ex:
        for n in ex.get("notes", []):
            S.append(Paragraph("Note: " + _esc(n), st["note"]))
        if ex.get("shap") and "error" not in ex["shap"]:
            S.append(Paragraph(_esc(ex["shap"]["note"]), st["note"]))
        for k in ("shap", "perm"):
            if k in charts:
                S.append(_fig_to_img(charts[k], 150))
    # ---- 9 warnings & limitations
    S.append(Paragraph("8. Critical warnings and limitations", st["h1"]))
    crit = [f for f in F_all if f.status in (FAILED,)] 
    lim = ["Automated checks cannot prove the absence of leakage or bias.",
           "Fairness metrics are computed at one threshold on one evaluation sample; observed disparities are not evidence of discrimination, and the absence of detected disparity is not evidence of fairness.",
           "Provenance of evaluation data (independent vs training) is declared by the user and cannot be verified by the app.",
           "Explanations (SHAP, permutation importance) describe model behaviour, not clinical causation.",
           "Thresholds (e.g. missingness bands, AUC suspicion level, CI-overlap rule) are documented heuristics, not standards."]
    for f in crit:
        S.append(Paragraph(f"<b>FAILED - {_esc(f.area)} / {_esc(f.check)}:</b> {_esc(f.detail)}", st["b"]))
    for l in lim:
        S.append(Paragraph("&bull; " + _esc(l), st["b"]))
    # ---- 10 recommendations
    S.append(Paragraph("9. Recommendations for improvement", st["h1"]))
    recs = recommendations(F_all)
    if recs:
        S.append(_tbl([["Priority", "Related check", "Recommended action"]] + [["High" if s == FAILED else "Medium", c, a] for s, c, a in recs], [W * .1, W * .27, W * .63], st))
    else:
        S.append(Paragraph("No corrective actions were generated by the checks that were run. Further expert review is still recommended before any clinical use.", st["b"]))
    S.append(Spacer(1, 8)); S.append(Paragraph(_esc(DISCLAIMER), st["note"]))
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ------------------------------- structured exports -------------------------------------
def _clean(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="records"))
    if isinstance(o, pd.Series):
        return o.to_dict()
    return str(o)


def findings_dataframe(ctx) -> pd.DataFrame:
    return pd.DataFrame([f.as_dict() for f in all_findings(ctx)])


def build_json(ctx: dict) -> str:
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "disclaimer": DISCLAIMER, "project": ctx.get("project", {}),
           "dataset": ctx.get("dataset", {}), "model": ctx.get("model"), "provenance": ctx.get("provenance_text"), "threshold": ctx.get("threshold"),
           "summary_counts": counts(all_findings(ctx)), "findings": [f.as_dict() for f in all_findings(ctx)]}
    if ctx.get("dq"):
        out["data_quality"] = {"scorecard": ctx["dq"]["scorecard"].reset_index().to_dict(orient="records"), "score_method": ctx["dq"]["score_method"],
                               "missing": ctx["dq"]["missing"].reset_index().to_dict(orient="records"), "duplicates": ctx["dq"]["duplicates"]}
    if ctx.get("perf"):
        out["performance"] = {"kind": ctx["perf"]["kind"], "metrics": ctx["perf"]["metrics"]["table"].to_dict(orient="records"),
                              "confusion": ctx["perf"]["metrics"].get("confusion")}
    if ctx.get("fair"):
        out["fairness"] = {a: v["table"].to_dict(orient="records") for a, v in ctx["fair"]["attributes"].items()}
    if ctx.get("expl") and ctx["expl"].get("shap") and "error" not in ctx["expl"]["shap"]:
        out["explainability"] = {"shap_importance": ctx["expl"]["shap"]["importance"].head(30).to_dict(orient="records"), "method": ctx["expl"]["shap"]["method"]}
    return json.dumps(out, indent=2, default=_clean)
