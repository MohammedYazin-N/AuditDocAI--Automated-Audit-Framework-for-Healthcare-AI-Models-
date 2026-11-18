"""UI helpers for the 'command centre' theme: palette, CSS, top bar, KPI tiles, gauges, Plotly figures."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from audit.common import STATUSES, counts, PASSED, WARNING, FAILED, NOT_EVALUATED

# ------------------------------------------------------------------ palette ---
BG, PANEL, PANEL2, BORDER, TEXT, MUTED = "#0A0F1E", "#111A2E", "#16213A", "#24324F", "#E8EEFA", "#8FA0BF"
TEAL, CORAL, AMBER, VIOLET, SKY, GREEN, SLATE = "#2DD4BF", "#FB7185", "#FBBF24", "#A78BFA", "#38BDF8", "#34D399", "#94A3B8"
NAVY = SKY
STATUS_UI = {PASSED: GREEN, WARNING: AMBER, FAILED: CORAL, NOT_EVALUATED: SLATE}
ICON = {PASSED: "✅", WARNING: "⚠️", FAILED: "❌", NOT_EVALUATED: "⏸️"}
GRADE_COL = {"Good": GREEN, "Acceptable": SKY, "Poor": AMBER, "Critical": CORAL}

CSS = f"""
<style>
/* ---- hide default chrome & sidebar: navigation lives in the top bar ---- */
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], header[data-testid="stHeader"] {{display:none !important;}}
.block-container {{padding-top: 1rem; max-width: 1320px;}}
.stApp {{background: radial-gradient(1200px 500px at 85% -10%, rgba(45,212,191,.13), transparent 60%),
                    radial-gradient(900px 500px at -5% 10%, rgba(167,139,250,.13), transparent 60%), {BG};}}
h1, h2, h3, h4 {{color: {TEXT}; letter-spacing: -.01em;}}
/* ---- top bar ---- */
.topbar {{display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.7rem 1.1rem; border:1px solid {BORDER};
          background: linear-gradient(90deg, rgba(17,26,46,.95), rgba(22,33,58,.85)); border-radius: 16px; margin-bottom:.6rem; backdrop-filter: blur(6px);}}
.brand {{display:flex; align-items:center; gap:.65rem; font-weight:800; font-size:1.15rem; color:{TEXT};}}
.brand .logo {{width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size:1.1rem;
              background: linear-gradient(135deg,{TEAL},{VIOLET}); box-shadow: 0 0 18px rgba(45,212,191,.45);}}
.brand small {{display:block; font-weight:500; font-size:.68rem; color:{MUTED}; letter-spacing:.08em; text-transform:uppercase;}}
.chips {{display:flex; gap:.45rem; flex-wrap:wrap; justify-content:flex-end;}}
.chip {{font-size:.74rem; padding:3px 11px; border-radius:999px; border:1px solid {BORDER}; color:{MUTED}; background:{PANEL};}}
.chip.on {{color:{TEXT}; border-color:rgba(45,212,191,.55); box-shadow: inset 0 0 12px rgba(45,212,191,.12);}}
.chip.bad {{color:{CORAL}; border-color:rgba(251,113,133,.55);}}
/* ---- nav: horizontal radio rendered as pills ---- */
div[role="radiogroup"][aria-label="nav"] {{gap:.35rem; flex-wrap:wrap;}}
div[data-testid="stRadio"] > div[role="radiogroup"] {{display:flex; flex-direction:row; gap:.4rem; flex-wrap:wrap;}}
div[data-testid="stRadio"] label {{background:{PANEL}; border:1px solid {BORDER}; border-radius:999px; padding:.25rem .8rem; cursor:pointer; transition:.15s;}}
div[data-testid="stRadio"] label:hover {{border-color:{TEAL}; transform: translateY(-1px);}}
div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div:first-child {{display:none !important;}}
div[data-testid="stRadio"] label[data-selected="true"] {{background: linear-gradient(90deg,{TEAL},{VIOLET}); border-color:transparent; box-shadow:0 4px 18px rgba(45,212,191,.35);}}
div[data-testid="stRadio"] label[data-selected="true"] p {{color:#06101C !important; font-weight:700;}}
div[data-testid="stRadio"] label p {{font-size:.82rem; margin:0; white-space:nowrap;}}
/* ---- hero ---- */
.hero {{position:relative; overflow:hidden; border-radius:22px; padding:2.2rem 2.2rem 1.8rem; margin:.4rem 0 1rem; border:1px solid {BORDER};
        background: linear-gradient(120deg, #0E2A44 0%, #14325A 40%, #2B2466 100%);}}
.hero:before {{content:""; position:absolute; right:-80px; top:-80px; width:340px; height:340px; border-radius:50%;
               background: radial-gradient(circle, rgba(45,212,191,.55), transparent 65%);}}
.hero:after {{content:""; position:absolute; left:35%; bottom:-140px; width:380px; height:380px; border-radius:50%;
              background: radial-gradient(circle, rgba(251,113,133,.30), transparent 65%);}}
.hero h1 {{position:relative; z-index:1; font-size:2.35rem; margin:0 0 .35rem; line-height:1.12; color:#fff;}}
.hero h1 span {{background: linear-gradient(90deg,{TEAL},{AMBER}); -webkit-background-clip:text; background-clip:text; color:transparent;}}
.hero p {{position:relative; z-index:1; margin:0; max-width:760px; color:#C9D6EE; font-size:1.02rem;}}
.pagehead {{display:flex; align-items:center; gap:.8rem; margin:.3rem 0 .9rem;}}
.pagehead .num {{width:42px; height:42px; border-radius:12px; display:grid; place-items:center; font-weight:800; color:#06101C;
                 background: linear-gradient(135deg,{TEAL},{SKY});}}
.pagehead h1 {{margin:0; font-size:1.75rem;}} .pagehead p {{margin:0; color:{MUTED}; font-size:.9rem;}}
/* ---- tiles / cards ---- */
.kpi {{background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:.85rem 1rem; position:relative; overflow:hidden; height:100%;}}
.kpi:before {{content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--c);}}
.kpi:after {{content:""; position:absolute; right:-30px; top:-30px; width:90px; height:90px; border-radius:50%; background:var(--c); opacity:.12;}}
.kpi .l {{font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; color:{MUTED};}}
.kpi .v {{font-size:1.75rem; font-weight:800; color:{TEXT}; line-height:1.15;}}
.kpi .s {{font-size:.76rem; color:{MUTED};}}
.stepcard {{background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:1rem; height:100%; position:relative;}}
.stepcard .n {{position:absolute; right:.8rem; top:.5rem; font-size:2.4rem; font-weight:900; color:rgba(255,255,255,.05);}}
.stepcard.done {{border-color:rgba(52,211,153,.6); box-shadow: inset 0 0 22px rgba(52,211,153,.08);}}
.stepcard h4 {{margin:.1rem 0 .3rem; font-size:1rem;}} .stepcard p {{margin:0; color:{MUTED}; font-size:.82rem;}}
.tile {{background:{PANEL}; border:1px solid {BORDER}; border-radius:16px; padding:.9rem 1rem; height:100%;}}
.tile .t {{font-weight:700; font-size:.92rem;}} .tile .d {{color:{MUTED}; font-size:.78rem; margin-top:.2rem;}}
.dot {{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.45rem; background:var(--c); box-shadow:0 0 10px var(--c);}}
.badge {{display:inline-block; padding:2px 10px; border-radius:999px; font-size:.76rem; font-weight:700; color:#06101C; background:var(--c);}}
.pipeline {{display:flex; gap:.35rem; align-items:center; margin:.2rem 0 1rem; flex-wrap:wrap;}}
.pipeline .p {{flex:1; min-width:110px; text-align:center; padding:.45rem .3rem; border-radius:12px; font-size:.78rem; border:1px solid {BORDER}; color:{MUTED}; background:{PANEL};}}
.pipeline .p.done {{color:#06101C; font-weight:700; background:linear-gradient(90deg,{TEAL},{GREEN}); border-color:transparent;}}
.pipeline .p.warn {{color:#06101C; font-weight:700; background:{AMBER}; border-color:transparent;}}
.small {{font-size:.85rem; color:{MUTED};}}
/* ---- widgets ---- */
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{background: linear-gradient(90deg,{TEAL},{SKY}); color:#06101C; font-weight:700; border:0; border-radius:12px; box-shadow:0 6px 20px rgba(45,212,191,.3);}}
.stButton > button, .stDownloadButton > button {{border-radius:12px;}}
div[data-testid="stExpander"] {{border:1px solid {BORDER}; border-radius:14px; background:{PANEL};}}
button[data-baseweb="tab"] {{font-weight:600;}}
div[data-testid="stMetric"] {{background:{PANEL}; border:1px solid {BORDER}; border-radius:14px; padding:.6rem .9rem;}}
div[data-testid="stVerticalBlockBorderWrapper"] {{border-radius:16px;}}
</style>
"""


# --------------------------------------------------------------- components ---
def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def topbar(dataset: str | None, target: str | None, model_state: str):
    """model_state: 'ok' | 'bad' | 'none'."""
    def chip(txt, cls=""):
        return f"<span class='chip {cls}'>{txt}</span>"
    chips = chip(f"📁 {dataset}" if dataset else "📁 no dataset", "on" if dataset else "")
    chips += chip(f"🎯 {target}" if target else "🎯 no target", "on" if target else "")
    chips += chip({"ok": "🧠 model compatible", "bad": "🧠 model incompatible", "none": "🧠 no model"}[model_state], {"ok": "on", "bad": "bad", "none": ""}[model_state])
    st.markdown(f"<div class='topbar'><div class='brand'><div class='logo'>🩺</div><div>AuditDoc AI<small>Healthcare model audit studio</small></div></div>"
                f"<div class='chips'>{chips}</div></div>", unsafe_allow_html=True)


def page_head(num: str, title: str, sub: str):
    st.markdown(f"<div class='pagehead'><div class='num'>{num}</div><div><h1>{title}</h1><p>{sub}</p></div></div>", unsafe_allow_html=True)


def kpi(col, label, value, sub="", colour=TEAL):
    col.markdown(f"<div class='kpi' style='--c:{colour}'><div class='l'>{label}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, sub, colour)."""
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        kpi(c, *it)


def pipeline_strip(steps):
    """steps: list of (label, state) with state in done|warn|todo."""
    html = "".join(f"<div class='p {s if s != 'todo' else ''}'>{l}</div>" for l, s in steps)
    st.markdown(f"<div class='pipeline'>{html}</div>", unsafe_allow_html=True)


def status_tiles(findings):
    c = counts(findings)
    kpi_row([(f"{ICON[s]} {s}", c[s], "checks", STATUS_UI[s]) for s in STATUSES])


status_metrics = status_tiles       # backwards-compatible name used by older pages


def badge(status: str) -> str:
    return f"<span class='badge' style='--c:{STATUS_UI[status]}'>{status}</span>"


def show_findings(findings, key: str):
    """Findings as an expandable list, worst first."""
    order = {FAILED: 0, WARNING: 1, NOT_EVALUATED: 2, PASSED: 3}
    for f in sorted(findings, key=lambda x: order[x.status]):
        cat = f" · {f.category}" if f.category else ""
        with st.expander(f"{ICON[f.status]} {f.check}{cat} — {f.status}", expanded=f.status == FAILED):
            st.markdown(f"**Finding.** {f.detail}")
            if f.criterion:
                st.markdown(f"**Criterion.** {f.criterion}")
            if f.action:
                st.markdown(f"**Recommended action.** {f.action}")
            st.caption(f"Implementation: {f.origin}")


def fmt_df(df: pd.DataFrame, digits=3):
    return df.style.format(precision=digits, na_rep="—")


def requirement_box(missing: list[str], what: str):
    st.warning(f"**{what} cannot run yet.** Missing: " + "; ".join(missing))


# ------------------------------------------------------------ plotly figures ---
def _style(fig, h=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT), margin=dict(l=0, r=10, t=40, b=0), **({"height": h} if h else {}))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER); fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def fig_gauge(score: float, grade: str, title="Overall data-quality score"):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score, number=dict(font=dict(size=46, color=TEXT), suffix=""),
                                 title=dict(text=f"{title}<br><span style='font-size:14px;color:{GRADE_COL[grade]}'>{grade}</span>", font=dict(size=14, color=MUTED)),
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=GRADE_COL[grade], thickness=.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                                            steps=[dict(range=[0, 50], color="rgba(251,113,133,.22)"), dict(range=[50, 75], color="rgba(251,191,36,.22)"),
                                                   dict(range=[75, 90], color="rgba(56,189,248,.22)"), dict(range=[90, 100], color="rgba(52,211,153,.25)")])))
    return _style(fig, 300)


def fig_status_donut(findings, title="Audit status mix"):
    c = counts(findings)
    fig = go.Figure(go.Pie(labels=list(STATUSES), values=[c[s] for s in STATUSES], hole=.68, sort=False, marker=dict(colors=[STATUS_UI[s] for s in STATUSES], line=dict(color=BG, width=3)),
                           textinfo="value", hovertemplate="%{label}: %{value}<extra></extra>"))
    fig.update_layout(title=title, annotations=[dict(text=f"<b>{sum(c.values())}</b><br>checks", x=.5, y=.5, showarrow=False, font=dict(size=18, color=TEXT))], legend=dict(orientation="h", y=-.08))
    return _style(fig, 320)


def fig_missing(miss: pd.DataFrame):
    m = miss.reset_index().rename(columns={"index": "column"})
    m.columns = ["column"] + list(m.columns[1:])
    fig = px.bar(m.head(25), x="missing_pct", y="column", orientation="h", color="severity", text="missing_pct",
                 color_discrete_map={"Critical": CORAL, "High": AMBER, "Medium": SKY, "Low": VIOLET}, labels={"missing_pct": "% missing"})
    fig.update_traces(texttemplate="%{text:.1f}%", marker_line_width=0)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return _style(fig, max(300, 26 * min(len(m), 25) + 120))


def fig_class(counts_dict, label):
    d = pd.DataFrame({"class": [str(k) for k in counts_dict], "count": list(counts_dict.values())})
    fig = px.bar(d, x="class", y="count", text="count", color="class", color_discrete_sequence=[TEAL, CORAL, AMBER, VIOLET, SKY])
    fig.update_layout(title=f"Class distribution ({label})", showlegend=False)
    return _style(fig, 330)


def fig_scorecard(sc: pd.DataFrame):
    s = sc.dropna(subset=["score"]).reset_index().rename(columns={"index": "dimension"})
    fig = px.bar(s, x="score", y="dimension", orientation="h", color="grade", color_discrete_map=GRADE_COL, text="score", range_x=[0, 112])
    fig.update_traces(texttemplate="%{text:.1f}")
    fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": list(s.dimension[::-1])}, showlegend=True)
    return _style(fig, 330)


def fig_corr(corr: pd.DataFrame):
    fig = px.imshow(corr, color_continuous_scale=[[0, SKY], [.5, PANEL2], [1, CORAL]], zmin=-1, zmax=1, aspect="auto")
    return _style(fig, 520)


def fig_roc_pr_cal(mt):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=("ROC curve", "Precision-Recall curve", "Calibration (quantile bins)"))
    fig.add_trace(go.Scatter(x=mt["roc_curve"].fpr, y=mt["roc_curve"].tpr, name=f"AUC {mt['roc_auc']:.3f}", line=dict(color=TEAL, width=3), fill="tozeroy", fillcolor="rgba(45,212,191,.12)"), 1, 1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=SLATE), showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=mt["pr_curve"].recall, y=mt["pr_curve"].precision, name=f"AP {mt['pr_auc']:.3f}", line=dict(color=VIOLET, width=3)), 1, 2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[mt["prevalence"]] * 2, line=dict(dash="dash", color=SLATE), name="prevalence"), 1, 2)
    cc = mt["calibration_curve"]
    fig.add_trace(go.Scatter(x=cc.mean_predicted, y=cc.observed_rate, mode="lines+markers", name="model", line=dict(color=AMBER, width=3), marker=dict(size=8)), 1, 3)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=SLATE), showlegend=False), 1, 3)
    fig.update_xaxes(title_text="False positive rate", row=1, col=1); fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2); fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_xaxes(title_text="Mean predicted probability", row=1, col=3); fig.update_yaxes(title_text="Observed rate", row=1, col=3)
    fig.update_layout(legend=dict(orientation="h", y=-0.25))
    return _style(fig, 380)


def fig_confusion(cm: np.ndarray, labels):
    fig = px.imshow(cm, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels], text_auto=True, color_continuous_scale=[[0, PANEL2], [1, TEAL]], aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    return _style(fig, 340)


def fig_subgroup_bars(tbl: pd.DataFrame, metric: str, lo=None, hi=None, title=""):
    t = tbl.copy()
    colour = np.array(["Adequate data" if a.startswith("Adequate") else "Limited data" for a in t.data_adequacy])
    err = dict(array=(t[hi] - t[metric]).clip(lower=0).fillna(0).to_numpy(), arrayminus=(t[metric] - t[lo]).clip(lower=0).fillna(0).to_numpy()) if lo and hi else None
    fig = go.Figure()
    for lab, colr in (("Adequate data", TEAL), ("Limited data", "#64748B")):
        m = colour == lab
        if m.any():
            kw = dict(error_y=dict(type="data", symmetric=False, array=err["array"][m], arrayminus=err["arrayminus"][m], color=TEXT)) if err else {}
            fig.add_trace(go.Bar(x=[f"{g} (n={n})" for g, n in zip(t.group[m], t.n[m])], y=t[metric][m], name=lab, marker_color=colr, **kw))
    fig.update_layout(title=title, yaxis=dict(range=[0, 1], tickformat=".0%"), legend=dict(orientation="h", y=-0.25))
    return _style(fig, 330)


def fig_group_heatmap(tbl: pd.DataFrame):
    cols = [c for c in ["recall_sensitivity", "specificity", "accuracy", "selection_rate"] if c in tbl.columns]
    z = tbl.set_index(tbl.group.astype(str) + np.where(tbl.data_adequacy.str.startswith("Limited"), " ⚠", ""))[cols]
    fig = px.imshow(z, text_auto=".0%", aspect="auto", color_continuous_scale=[[0, "#2B1B4F"], [.5, VIOLET], [1, TEAL]], zmin=0, zmax=1,
                    labels=dict(color="rate"))
    fig.update_layout(title="Group × metric heatmap (⚠ = limited data)")
    return _style(fig, max(260, 42 * len(z) + 120))


def fig_group_calibration(curves: dict, title: str):
    fig = go.Figure()
    for g, c in curves.items():
        fig.add_trace(go.Scatter(x=c.mean_predicted, y=c.observed_rate, mode="lines+markers", name=f"{g} (n={int(c.n.iloc[0])})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=SLATE), name="perfect"))
    fig.update_layout(title=title, xaxis_title="Mean predicted probability", yaxis_title="Observed rate")
    return _style(fig, 380)


def fig_importance(df, value_col, title, n=20):
    d = df.head(n).iloc[::-1]
    fig = px.bar(d, x=value_col, y="feature", orientation="h", color=value_col, color_continuous_scale=[[0, VIOLET], [1, TEAL]])
    fig.update_layout(title=title, coloraxis_showscale=False)
    return _style(fig, max(320, 22 * len(d) + 100))
