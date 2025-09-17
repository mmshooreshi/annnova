# app.py
# -----------------------------------------------------------------------------
# Locomotion Analysis — FINAL RESULTS & DIAGRAMS (Streamlit)
# -----------------------------------------------------------------------------
# What this focused app does:
#   • Upload CSV → map columns (Group, Rat, bins)
#   • Auto long-reshape with derived factors:
#       Prenatal = first token of Group (e.g., LPS/NS)
#       Drug     = second token of Group (e.g., NS/DZP/BSP)
#   • Time course (mean±SEM) in *classic error-bar* style (only)
#   • PRE / POST windows + Δ (POST−PRE)
#   • Two-way ANOVA (Prenatal × Drug) on totals + Tukey HSD across 6 groups
#   • Diagnostics (Levene; Shapiro per group)
#   • Publication-style grayscale bar plot with dashed sig brackets + CLD letters
#   • One-click downloads for tables & figures
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf
from matplotlib.ticker import MultipleLocator
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import t as tdist

import streamlit as st

# ------------------------------- PAGE CONFIG ---------------------------------

st.set_page_config(page_title="Locomotion — Final Figures & Stats", layout="wide")
st.markdown("<style>div.block-container{padding-top:1rem;padding-bottom:2rem;}</style>", unsafe_allow_html=True)

# ------------------------------- DATA MODEL ----------------------------------

@dataclass
class ColumnMap:
    group: str
    rat: str
    bins: List[str]
    time_start: int
    bin_width: int

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def infer_time_labels(bin_cols: Sequence[str], start: int, step: int) -> List[int]:
    return [int(start + i * step) for i in range(len(bin_cols))]

def derive_factors_from_group(g: str) -> Tuple[str, str]:
    parts = re.split(r"[_\-\s]+", str(g).strip())
    return (parts[0], parts[1]) if len(parts) >= 2 else ("UNK", "UNK")

def to_long(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    long = df.melt(
        id_vars=[cmap.group, cmap.rat],
        value_vars=cmap.bins,
        var_name="TimeBin",
        value_name="Distance",
    ).rename(columns={cmap.group: "Group", cmap.rat: "Rat"})
    long["Rat"] = long["Rat"].astype(str)

    time_labels = infer_time_labels(cmap.bins, cmap.time_start, cmap.bin_width)
    long["TimeMin"] = long["TimeBin"].map({b: t for b, t in zip(cmap.bins, time_labels)})

    pren, drug = zip(*[derive_factors_from_group(g) for g in long["Group"]])
    long["Prenatal"] = pren
    long["Drug"] = drug

    long["Distance"] = pd.to_numeric(long["Distance"], errors="coerce")
    long = long.dropna(subset=["Distance"])
    return long

def summarize_timecourse(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["Group", "TimeMin"])["Distance"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )

def totals_by_rat(long: pd.DataFrame) -> pd.DataFrame:
    tot = long.groupby(["Group", "Rat"], as_index=False)["Distance"].sum()
    pren, drug = zip(*[derive_factors_from_group(g) for g in tot["Group"]])
    tot["Prenatal"] = pren
    tot["Drug"] = drug
    return tot

# ------------------------------- PLOTTING ------------------------------------

def style_axes(ax, y_major: Optional[int] = None) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(direction="out", length=4, width=1.0)
    if y_major:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))

def fig_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def _marker_jitter_positions(t: np.ndarray, idx: int, total: int) -> np.ndarray:
    if t.size < 2:
        step = 1.0
    else:
        diffs = np.diff(np.unique(t))
        step = np.median(diffs) if diffs.size else 1.0
    spread = 0.12 * step
    if total == 1:
        return t
    offs = np.linspace(-spread, spread, total)
    return t + offs[idx]

def plot_timecourse_errorbar(summary: pd.DataFrame, legend_order: Sequence[str]) -> Tuple[plt.Figure, plt.Axes]:
    """Classic black-only error-bar style with slight x-jitter to avoid overlap."""
    drug_marker = {"NS": "o", "DZP": "^", "BSP": "s"}      # shape by Drug
    prenatal_fill = {"NS": ("white", "black"), "LPS": ("black", "black")}  # (mfc, mec)

    s = summary.copy()
    s["TimeMin"] = pd.to_numeric(s["TimeMin"], errors="coerce")
    s = s.dropna(subset=["TimeMin", "mean"]).sort_values(["Group", "TimeMin"])
    groups = [g for g in legend_order if g in s["Group"].unique().tolist()]

    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=240)
    for gi, g in enumerate(groups):
        sub = s.loc[s["Group"] == g, ["TimeMin", "mean", "sem"]]
        if sub.empty:
            continue
        t = sub["TimeMin"].to_numpy(); m = sub["mean"].to_numpy(); se = sub["sem"].to_numpy()
        mask = np.isfinite(t) & np.isfinite(m) & np.isfinite(se)
        t, m, se = t[mask], m[mask], se[mask]
        if t.size == 0:
            continue

        pren, drug = derive_factors_from_group(g)
        marker = drug_marker.get(drug, "o")
        mfc, mec = prenatal_fill.get(pren, ("white", "black"))
        tj = _marker_jitter_positions(t, gi, len(groups))

        ax.errorbar(tj, m, yerr=se, color="black", lw=1.4, capsize=2, elinewidth=1.0, zorder=3)
        ax.plot(tj, m, ls="-", color="black", lw=0.8, zorder=2)
        ax.plot(tj, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.5, mew=1.1, zorder=4, label=g)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total Movement Distance (cm)")
    style_axes(ax)
    ax.yaxis.grid(True, linewidth=0.4, color="0.9")

    leg = ax.legend(frameon=False, ncol=3, columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0, loc="upper left")
    for h in leg.legend_handles:
        h.set_linestyle("none")

    fig.tight_layout()
    return fig, ax

def plot_totals_bar(
    totals: pd.DataFrame, group_order: Sequence[str], tukey_summary: Optional[pd.DataFrame], cld_letters: Optional[Dict[str, str]]
) -> Tuple[plt.Figure, plt.Axes]:
    means = [totals.loc[totals["Group"] == g, "Distance"].mean() for g in group_order]
    sems  = [totals.loc[totals["Group"] == g, "Distance"].sem()  for g in group_order]
    fills = ["white", "0.85", "0.70", "0.55", "0.40", "0.25"]  # up to 6 groups

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    x = np.arange(len(group_order))
    ax.bar(x, means, yerr=sems, capsize=3, edgecolor="black", linewidth=1.0, color=fills[:len(group_order)], width=0.65)
    ax.set_xticks(x, group_order, rotation=25, ha="right")
    ax.set_ylabel("Total Movement Distance (cm)")
    style_axes(ax)

    # Significance brackets
    if tukey_summary is not None and not tukey_summary.empty:
        sig = tukey_summary.query("reject == True").copy()
        y0 = max(np.array(means) + np.array(sems))
        step = 0.07 * y0
        pairs, heights = [], []
        for idx, row in sig.reset_index(drop=True).iterrows():
            g1, g2 = row["group1"], row["group2"]
            if g1 in group_order and g2 in group_order:
                i, j = group_order.index(g1), group_order.index(g2)
                stars = ("***" if row["p-adj"] < 0.001 else "**" if row["p-adj"] < 0.01 else "*" if row["p-adj"] < 0.05 else "ns")
                pairs.append((i, j, stars))
                heights.append(y0 + (idx + 1) * step)

        for (i, j, txt), h in zip(pairs, heights):
            xi, xj = x[i], x[j]
            ax.plot([xi, xi, xj, xj], [h, h + step * 0.2, h + step * 0.2, h], lw=1.0, color="black", linestyle=(0, (4, 3)))
            ax.text((xi + xj) / 2, h + step * 0.25, txt, ha="center", va="bottom", fontsize=11)

    # CLD letters
    if cld_letters:
        ymax = (np.array(means) + np.array(sems)).max()
        ypad = 0.06 * ymax
        for idx, g in enumerate(group_order):
            lab = cld_letters.get(g, "")
            if lab:
                ax.text(x[idx], means[idx] + sems[idx] + ypad, lab, ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    return fig, ax

# ------------------------------- STATISTICS ----------------------------------

def group_descriptives(totals: pd.DataFrame, group_order: Sequence[str]) -> pd.DataFrame:
    desc = (
        totals.groupby("Group")["Distance"]
        .agg(mean="mean", sd="std", n="count")
        .reindex(group_order)
        .reset_index()
    )
    desc["sem"] = desc["sd"] / np.sqrt(desc["n"])
    crit = desc["n"].apply(lambda k: tdist.ppf(0.975, max(k - 1, 1)))
    desc["ci95"] = crit * desc["sem"]
    return desc

def add_effect_sizes(aov: pd.DataFrame) -> pd.DataFrame:
    aov = aov.copy()
    if "sum_sq" not in aov.columns or "Residual" not in aov.index:
        return aov
    ss_total = aov["sum_sq"].sum()
    ss_error = aov.loc["Residual", "sum_sq"]
    df_error = aov.loc["Residual", "df"]
    ms_error = ss_error / df_error
    effects = [ix for ix in aov.index if ix != "Residual"]
    aov.loc[effects, "eta2"] = aov.loc[effects, "sum_sq"] / ss_total
    aov.loc[effects, "partial_eta2"] = aov.loc[effects, "sum_sq"] / (aov.loc[effects, "sum_sq"] + ss_error)
    aov.loc[effects, "omega2"] = ((aov.loc[effects, "sum_sq"] - aov.loc[effects, "df"] * ms_error) /
                                  (ss_total + ms_error)).clip(lower=0)
    return aov

def compact_letter_display(tukey_df: pd.DataFrame, means: pd.Series) -> Dict[str, str]:
    groups = list(means.sort_values(ascending=False).index)
    sig = set(tuple(sorted([r["group1"], r["group2"]])) for _, r in tukey_df.iterrows() if r["reject"])
    letter_sets: List[set] = []
    for g in groups:
        placed = False
        for s in letter_sets:
            if any(tuple(sorted([g, h])) in sig for h in s):
                continue
            s.add(g); placed = True; break
        if not placed:
            letter_sets.append({g})
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cld: Dict[str, str] = {}
    for i, s in enumerate(letter_sets):
        lab = alphabet[i] if i < len(alphabet) else f"L{i+1}"
        for g in s:
            cld[g] = cld.get(g, "") + lab
    return cld

def twoway_anova_totals(totals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = smf.ols("Distance ~ C(Prenatal)*C(Drug)", data=totals).fit()
    aov = anova_lm(model, typ=2)
    tukey_res = pairwise_tukeyhsd(endog=totals["Distance"], groups=totals["Group"], alpha=0.05)
    tukey_df = pd.DataFrame(tukey_res.summary().data[1:], columns=tukey_res.summary().data[0])
    tukey_df.columns = [c.lower().replace(" ", "_") for c in tukey_df.columns]
    for col in ["meandiff", "p-adj", "lower", "upper"]:
        tukey_df[col] = pd.to_numeric(tukey_df[col], errors="coerce")
    tukey_df["reject"] = tukey_df["reject"].astype(bool)
    tukey_df = tukey_df.rename(columns={"group1": "group1", "group2": "group2"})
    return aov, tukey_df

def variance_overview(totals: pd.DataFrame) -> pd.DataFrame:
    v = (totals.groupby("Group")["Distance"]
         .agg(var="var", sd=lambda s: s.std(ddof=1), n="count")
         .reset_index())
    if not v["var"].isna().all():
        vmax, vmin = v["var"].max(), v["var"].min()
        v["var_ratio_max_min"] = vmax / vmin if vmin not in (0, np.nan) else np.nan
    return v

def normality_by_group(totals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, sub in totals.groupby("Group"):
        x = pd.to_numeric(sub["Distance"], errors="coerce").dropna()
        try:
            W, p = stats.shapiro(x) if 3 <= len(x) <= 5000 else (np.nan, np.nan)
        except Exception:
            W, p = (np.nan, np.nan)
        rows.append(dict(Group=g, n=len(x), mean=x.mean(), sd=x.std(ddof=1), W=W, p=p))
    return pd.DataFrame(rows).sort_values("Group")

def totals_by_window(long: pd.DataFrame, t_from: int, t_to: int) -> pd.DataFrame:
    dat = long[(long["TimeMin"] >= t_from) & (long["TimeMin"] <= t_to)].copy()
    out = dat.groupby(["Group", "Rat"], as_index=False)["Distance"].sum()
    pren, drug = zip(*[derive_factors_from_group(g) for g in out["Group"]])
    out["Prenatal"] = pren; out["Drug"] = drug
    return out

def totals_delta(pre_totals: pd.DataFrame, post_totals: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(post_totals, pre_totals, on=["Group", "Rat"], suffixes=("_post", "_pre"))
    m["Distance"] = m["Distance_post"] - m["Distance_pre"]
    m["Prenatal"] = m["Group"].map(lambda g: derive_factors_from_group(g)[0])
    m["Drug"]     = m["Group"].map(lambda g: derive_factors_from_group(g)[1])
    return m[["Group", "Rat", "Prenatal", "Drug", "Distance"]]

def make_totals_package(totals: pd.DataFrame, group_order: Sequence[str]) -> dict:
    aov, tukey = twoway_anova_totals(totals)
    aov_es = add_effect_sizes(aov)
    desc = group_descriptives(totals, group_order)
    cld = compact_letter_display(tukey, desc.set_index("Group")["mean"]) if not tukey.empty else {}
    vtab = variance_overview(totals)
    W, p_lev = stats.levene(*[g["Distance"].values for _, g in totals.groupby("Group")], center="median")
    norm = normality_by_group(totals)
    fig_bars, ax_bars = plot_totals_bar(totals, group_order, tukey, cld_letters=cld)
    return dict(aov=aov_es, tukey=tukey, desc=desc, cld=cld, variances=vtab, levene=(W, p_lev), normality=norm, fig=fig_bars)

# ------------------------------- SIDEBAR -------------------------------------

st.title("Locomotion — Final Figures & Stats")
st.caption("Upload your CSV, map columns, and get publication-ready error-bar time courses and totals with ANOVA/Tukey.")

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("CSV file", type=["csv"], help="Table with Group, Rat, and time-bin columns.")
    if up:
        df = load_csv(up)
    else:
        st.info("Upload a CSV to begin.")
        df = pd.DataFrame()

    if df.empty:
        st.stop()

    st.header("2) Column mapping")
    cols = list(df.columns)
    group_col = st.selectbox("Group column", cols, index=cols.index("Group") if "Group" in cols else 0)
    rat_col   = st.selectbox("Rat ID column", cols, index=cols.index("Rat") if "Rat" in cols else 1)
    candidate_bins = [c for c in cols if c not in (group_col, rat_col)]
    bin_cols = st.multiselect("Time-bin columns (ordered)", candidate_bins, default=candidate_bins)
    time_start = st.number_input("Start time (min) of first selected bin", value=10, step=5)
    bin_width  = st.number_input("Bin width (min)", value=10, step=5)

    labels = infer_time_labels(bin_cols, time_start, bin_width) if bin_cols else []
    if labels:
        tmin, tmax = min(labels), max(labels)
        mid_idx = len(labels) // 2
        pre_default  = (labels[0], labels[mid_idx-1] if mid_idx > 0 else labels[0])
        post_default = (labels[mid_idx], labels[-1])
        st.header("3) Windows")
        pre_range = st.slider("PRE window (min)", min_value=tmin, max_value=tmax, value=pre_default, step=bin_width)
        post_range = st.slider("POST window (min)", min_value=tmin, max_value=tmax, value=post_default, step=bin_width)
    else:
        pre_range = post_range = None

    cmap = ColumnMap(group=group_col, rat=rat_col, bins=bin_cols, time_start=time_start, bin_width=bin_width)

# ------------------------------- CORE PIPELINE --------------------------------

if len(cmap.bins) == 0:
    st.error("Select at least one time-bin column in the sidebar.")
    st.stop()

long = to_long(df, cmap)
summary = summarize_timecourse(long)
totals  = totals_by_rat(long)

group_order = sorted(totals["Group"].unique().tolist())

# ------------------------------- TIME COURSE ----------------------------------

st.subheader("Time Course (mean ± SEM) — Classic Error Bars")
fig_tc, ax_tc = plot_timecourse_errorbar(summary, group_order)
st.pyplot(fig_tc, use_container_width=True)

# ------------------------------- TOTALS (PRE/POST/Δ) --------------------------

if pre_range is None or post_range is None:
    st.warning("Define PRE/POST windows in the sidebar to compute totals and Δ.")
    st.stop()

pre_totals  = totals_by_window(long, pre_range[0],  pre_range[1])
post_totals = totals_by_window(long, post_range[0], post_range[1])
delta_tot   = totals_delta(pre_totals, post_totals)

tab_pre, tab_post, tab_delta, tab_dl = st.tabs([
    f"PRE [{pre_range[0]}–{pre_range[1]} min]",
    f"POST [{post_range[0]}–{post_range[1]} min]",
    "Δ (POST − PRE)",
    "Downloads"
])

def render_totals_panel(totals_df: pd.DataFrame, label: str):
    st.markdown(f"**{label} totals per group (ANOVA, Tukey, Diagnostics, Figure)**")
    pkg = make_totals_package(totals_df, group_order)

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg["aov"].round(4), use_container_width=True)
        st.markdown("**Tukey HSD (6 groups)**")
        st.dataframe(pkg["tukey"], use_container_width=True)
        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg["desc"].copy()
        if pkg["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)
    with c2:
        st.markdown("**Diagnostics**")
        W, p_lev = pkg["levene"]
        st.write(f"Levene’s test: W = {W:.3f}, p = {p_lev:.4f}")
        st.dataframe(pkg["variances"].round(4), use_container_width=True)
        st.caption("Shapiro–Wilk per group")
        st.dataframe(pkg["normality"].round(4), use_container_width=True)

    st.subheader(f"{label} — Grayscale bars with significance")
    st.pyplot(pkg["fig"], use_container_width=True)

    # Return tables/fig for downloads
    return pkg, desc_disp

with tab_pre:
    pre_pkg, pre_desc = render_totals_panel(pre_totals, "PRE")

with tab_post:
    post_pkg, post_desc = render_totals_panel(post_totals, "POST")

with tab_delta:
    delta_pkg, delta_desc = render_totals_panel(delta_tot, "Δ (POST − PRE)")

# ------------------------------- DOWNLOADS ------------------------------------

with tab_dl:
    st.subheader("Exports")

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Long data (tidy)")
        st.download_button("Download CSV", long.to_csv(index=False).encode(), "long_data.csv", "text/csv")
    with cB:
        st.caption("Time-course summary")
        st.download_button("Download CSV", summary.to_csv(index=False).encode(), "timecourse_summary.csv", "text/csv")
    with cC:
        st.caption("Totals (per rat, all bins)")
        st.download_button("Download CSV", totals.to_csv(index=False).encode(), "totals_all_bins.csv", "text/csv")

    cF, cG = st.columns(2)
    with cF:
        st.caption("Time-course figure (PNG)")
        st.download_button("Download PNG", fig_bytes(fig_tc), "figure_timecourse_errorbars.png", "image/png")

    # PRE
    st.markdown("### PRE")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.download_button("ANOVA (CSV)", pre_pkg["aov"].to_csv().encode(), "pre_anova_effectsizes.csv", "text/csv")
    with col2: st.download_button("Tukey (CSV)", pre_pkg["tukey"].to_csv(index=False).encode(), "pre_tukey.csv", "text/csv")
    with col3: st.download_button("Descriptives (CSV)", pre_desc.to_csv(index=False).encode(), "pre_descriptives.csv", "text/csv")
    with col4: st.download_button("Diagnostics (CSV)", pre_pkg["variances"].to_csv(index=False).encode(), "pre_variances.csv", "text/csv")
    with col5: st.download_button("Figure (PNG)", fig_bytes(pre_pkg["fig"]), "pre_totals.png", "image/png")

    # POST
    st.markdown("### POST")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.download_button("ANOVA (CSV)", post_pkg["aov"].to_csv().encode(), "post_anova_effectsizes.csv", "text/csv")
    with col2: st.download_button("Tukey (CSV)", post_pkg["tukey"].to_csv(index=False).encode(), "post_tukey.csv", "text/csv")
    with col3: st.download_button("Descriptives (CSV)", post_desc.to_csv(index=False).encode(), "post_descriptives.csv", "text/csv")
    with col4: st.download_button("Diagnostics (CSV)", post_pkg["variances"].to_csv(index=False).encode(), "post_variances.csv", "text/csv")
    with col5: st.download_button("Figure (PNG)", fig_bytes(post_pkg["fig"]), "post_totals.png", "image/png")

    # DELTA
    st.markdown("### Δ (POST − PRE)")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.download_button("ANOVA (CSV)", delta_pkg["aov"].to_csv().encode(), "delta_anova_effectsizes.csv", "text/csv")
    with col2: st.download_button("Tukey (CSV)", delta_pkg["tukey"].to_csv(index=False).encode(), "delta_tukey.csv", "text/csv")
    with col3: st.download_button("Descriptives (CSV)", delta_desc.to_csv(index=False).encode(), "delta_descriptives.csv", "text/csv")
    with col4: st.download_button("Diagnostics (CSV)", delta_pkg["variances"].to_csv(index=False).encode(), "delta_variances.csv", "text/csv")
    with col5: st.download_button("Figure (PNG)", fig_bytes(delta_pkg["fig"]), "delta_totals.png", "image/png")

st.markdown("---")
st.caption("Tip: adjust selected bin columns or start/width in the sidebar to reshape the time axis and PRE/POST windows.")
