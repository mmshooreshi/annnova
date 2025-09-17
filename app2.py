# app.py — EXHAUSTIVE Locomotion Analysis & Journal-Style Figures
# =============================================================================
# This app:
#   1) Reads your CSV and maps columns (Group, Rat, time-bins)
#   2) QC: missingness, Ns, descriptive stats, outlier flags
#   3) Visuals: per-rat spaghetti; per-bin box+strip; totals violin+box+jitter
#   4) Metrics: time-course mean±SEM; AUC; peak; time-to-peak
#   5) Stats:
#       • Mixed ANOVA (within: Time; between: Group) + sphericity (Mauchly) & GG/HF
#       • Per-bin 2×3 ANOVAs (Prenatal × Drug) with partial η² + Holm-FDR
#       • Totals 2×3 ANOVA with partial η² + Tukey + planned contrasts
#       • Robust sensitivity: Welch one-way across 6 groups (totals)
#       • Linear Mixed-Effects: Distance ~ Time*Prenatal*Drug + (1|Rat)
#   6) Interpretation: auto-written plain-English summary of major findings
#   7) Downloads: all tables & figures (PNG/CSV)
# =============================================================================

from __future__ import annotations

import io
import itertools
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.ticker import MultipleLocator
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

import streamlit as st

# Optional: mixed/repeated utilities (sphericity, GG/HF)
try:
    import pingouin as pg  # pip install pingouin
    HAVE_PG = True
except Exception:
    HAVE_PG = False

# ------------------------------------ UI -------------------------------------

st.set_page_config(page_title="Locomotion Figure Lab — Exhaustive", layout="wide")
st.markdown(
    "<style>div.block-container{padding-top:0.75rem;padding-bottom:1.25rem;}</style>",
    unsafe_allow_html=True,
)

# ------------------------------- Data mapping --------------------------------

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


def parse_group(g: str) -> Tuple[str, str]:
    parts = re.split(r"[_\-\s]+", str(g).strip())
    if len(parts) >= 2:
        return parts[0], parts[1]
    return ("UNK", "UNK")


def to_long(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    long = df.melt(
        id_vars=[cmap.group, cmap.rat],
        value_vars=cmap.bins,
        var_name="TimeBin",
        value_name="Distance",
    ).rename(columns={cmap.group: "Group", cmap.rat: "Rat"})

    # IDs
    long["Rat"] = long["Rat"].astype(str)
    long["Subject"] = (long["Group"].astype(str) + "_" + long["Rat"])  # UNIQUE per group

    # numeric DV
    long["Distance"] = pd.to_numeric(long["Distance"], errors="coerce")

    # deterministic time labels (don’t parse column text)
    times = infer_time_labels(cmap.bins, cmap.time_start, cmap.bin_width)
    mapper = {b: t for b, t in zip(cmap.bins, times)}
    long["TimeMin"] = long["TimeBin"].map(mapper)

    # factors
    pren, drug = zip(*[parse_group(g) for g in long["Group"]])
    long["Prenatal"] = pren
    long["Drug"] = drug

    long = long.dropna(subset=["Distance"])
    return long

# --------------------------------- Helpers -----------------------------------


def mixed_anova_timecourse(long: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Between = Group (6 levels), Within = Time (TimeMin), Subject = unique per group.
    """
    if not HAVE_PG:
        return None
    dat = long.rename(columns={"TimeMin": "Time"})
    dat["Subject"] = dat["Subject"].astype(str)
    # Pingouin expects within-factor categorical; numeric is fine, but ensure dtype
    dat["Time"] = dat["Time"].astype(str)

    try:
        aov = pg.mixed_anova(
            dv="Distance",
            within="Time",
            between="Group",
            subject="Subject",
            data=dat
        )
        return aov
    except Exception:
        return None

def style_axes(ax, y_major: Optional[int] = None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
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

def partial_eta_squared(ss_effect: float, ss_error: float) -> float:
    # ηp² = SS_effect / (SS_effect + SS_error)
    try:
        return float(ss_effect) / float(ss_effect + ss_error)
    except Exception:
        return np.nan

def iqr_outlier_flags(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x < lo) | (x > hi)

# --------------------------------- Sidebar -----------------------------------

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.info("Upload your CSV (e.g., input.csv).")
        st.stop()
    df = load_csv(up)

    st.header("2) Column mapping")
    cols = list(df.columns)
    group_col = st.selectbox("Group column", cols, index=cols.index("Group") if "Group" in cols else 0)
    rat_col = st.selectbox("Rat ID column", cols, index=cols.index("Rat") if "Rat" in cols else 1)
    candidate_bins = [c for c in cols if c not in (group_col, rat_col)]
    bin_cols = st.multiselect("Time-bin columns (ordered)", candidate_bins, default=candidate_bins)
    time_start = st.number_input("Start time (min) of first selected bin", value=10, step=5)
    bin_width = st.number_input("Bin width (min)", value=10, step=5)
    cmap = ColumnMap(group=group_col, rat=rat_col, bins=bin_cols, time_start=time_start, bin_width=bin_width)

    if len(cmap.bins) == 0:
        st.error("Select at least one time-bin column.")
        st.stop()

# ------------------------------- Transform -----------------------------------

long = to_long(df, cmap)

# ------------------------------ Tabs layout ----------------------------------

tab_qc, tab_tc, tab_tot, tab_bin, tab_adv, tab_dl = st.tabs([
    "QC & Descriptives",
    "Time course (Mixed ANOVA)",
    "Totals & AUC (ANOVA + Tukey)",
    "Per-bin ANOVAs",
    "Advanced (Robust + LME)",
    "Downloads",
])

# =========================== QC & DESCRIPTIVES ===============================

with tab_qc:
    st.subheader("Data preview (long/‘tidy’)")
    st.dataframe(long.head(20), use_container_width=True)

    st.markdown("### Missingness")
    miss = long.isna().mean().to_frame("missing_rate")
    st.dataframe(miss.style.format("{:.2%}"), use_container_width=True)

    st.markdown("### Group sizes (rats per group)")
    Ns = long.groupby(["Group", "Rat"]).size().reset_index().groupby("Group")["Rat"].nunique()
    st.dataframe(Ns.to_frame("N"), use_container_width=True)

    st.markdown("### Per-bin descriptives (mean ± SEM)")
    desc = (
        long.groupby(["Group", "TimeMin"])["Distance"]
        .agg(mean="mean", sem="sem", sd="std", n="count")
        .reset_index()
    )
    st.dataframe(desc, use_container_width=True)

    st.markdown("### Outlier counts (IQR rule, by Group × Time)")
    out_rows = []
    for (g, t), sub in long.groupby(["Group", "TimeMin"]):
        vals = sub["Distance"].values
        flags = iqr_outlier_flags(vals)
        out_rows.append(dict(Group=g, TimeMin=t, outliers=int(flags.sum()), n=len(vals)))
    out_tbl = pd.DataFrame(out_rows)
    st.dataframe(out_tbl, use_container_width=True)

    st.markdown("### Per-rat spaghetti")
    fig_sp, ax_sp = plt.subplots(figsize=(8, 4), dpi=200)
    for (g, r), sub in long.groupby(["Group", "Rat"]):
        ax_sp.plot(sub["TimeMin"], sub["Distance"], lw=0.8, alpha=0.4)
    ax_sp.set_xlabel("Time (min)")
    ax_sp.set_ylabel("Distance (cm)")
    style_axes(ax_sp)
    st.pyplot(fig_sp, use_container_width=True)

    st.markdown("### Per-bin distribution (box + strip) by Group")
    bin_pick = st.selectbox("Pick a time bin (min)", sorted(long["TimeMin"].unique()))
    sub = long[long["TimeMin"] == bin_pick]
    fig_box, ax_box = plt.subplots(figsize=(8, 3.8), dpi=200)
    order = sorted(sub["Group"].unique())
    positions = np.arange(len(order))
    # box
    data_by_group = [sub.loc[sub["Group"] == g, "Distance"].values for g in order]
    ax_box.boxplot(data_by_group, positions=positions, widths=0.6, showfliers=False)
    # jitter
    for i, g in enumerate(order):
        y = sub.loc[sub["Group"] == g, "Distance"].values
        x = np.random.normal(loc=i, scale=0.05, size=len(y))
        ax_box.plot(x, y, "o", ms=3, alpha=0.6, color="black")
    ax_box.set_xticks(positions, order, rotation=20, ha="right")
    ax_box.set_ylabel("Distance (cm)")
    style_axes(ax_box)
    st.pyplot(fig_box, use_container_width=True)

# =============================== TIME COURSE =================================

def timecourse_summary(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["Group", "TimeMin"])["Distance"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

def plot_timecourse(summary: pd.DataFrame, legend_order: Sequence[str]):
    marker_map = {
        "NS_NS": dict(marker="o", mfc="white", mec="black"),
        "NS_DZP": dict(marker="o", mfc="white", mec="black"),
        "NS_BSP": dict(marker="s", mfc="white", mec="black"),
        "LPS_NS": dict(marker="o", mfc="black", mec="black"),
        "LPS_DZP": dict(marker="o", mfc="black", mec="black"),
        "LPS_BSP": dict(marker="s", mfc="black", mec="black"),
    }
    fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
    for g in legend_order:
        sub = summary[summary["Group"] == g]
        mk = marker_map.get(g, dict(marker="o", mfc="white", mec="black"))
        ax.errorbar(
            sub["TimeMin"], sub["mean"], yerr=sub["sem"],
            marker=mk["marker"], mfc=mk["mfc"], mec=mk["mec"],
            ms=5, lw=1.5, ls="-", color="black", capsize=2, label=g
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total Movement Distance (cm)")
    style_axes(ax)
    ax.legend(frameon=False, ncol=3)
    return fig

with tab_tc:
    st.subheader("Mean ± SEM over time (journal style)")
    summary_tc = timecourse_summary(long)

    # consistent order
    group_order = sorted(long["Group"].unique().tolist())
    fig_tc = plot_timecourse(summary_tc, group_order)
    st.pyplot(fig_tc, use_container_width=True)

    st.markdown("### Mixed ANOVA (within = Time, between = Group)")
    if HAVE_PG:
        dat = long.rename(columns={"Rat": "Subject", "TimeMin": "Time"})
        dat["Subject"] = dat["Subject"].astype(str)
        try:
            aov = pg.mixed_anova(dv="Distance", within="Time", between="Group", subject="Subject", data=dat)
            st.dataframe(aov, use_container_width=True)
        except Exception as e:
            st.error(f"mixed_anova failed: {e}")
    else:
        st.info("Install `pingouin` for mixed ANOVA and sphericity checks (`pip install pingouin`).")

    st.markdown("### Sphericity & Corrections (Time factor only)")
    st.caption("We test sphericity on the within-subject factor (Time). If violated (p<.05), use GG-corrected results.")

    if HAVE_PG:
        try:
            # Wide matrix: rows = unique Subject, cols = Time
            wide = long.pivot_table(index="Subject", columns="TimeMin", values="Distance", aggfunc="mean")
            # Require ≥ 3 time points and no all-missing rows
            wide = wide.dropna(how="any")
            if wide.shape[1] < 3 or wide.shape[0] < 2:
                st.info("Need ≥ 3 time levels and ≥ 2 subjects with complete data.")
            else:
                res = pg.sphericity(wide, method="mauchly")
                # Pingouin versions differ in return length
                if isinstance(res, tuple):
                    if len(res) == 4:
                        W, chi2, dof, pval = res
                    elif len(res) == 2:
                        W, pval = res
                        chi2 = np.nan; dof = np.nan
                    else:
                        W = pval = chi2 = dof = np.nan
                else:
                    W = pval = chi2 = dof = np.nan

                st.write(f"**Mauchly’s W = {W:.3f}**, χ² = {chi2 if not np.isnan(chi2) else 'NA'}, "
                        f"df = {dof if not np.isnan(dof) else 'NA'}, **p = {pval:.4f}**")

                eps = pg.epsilon(wide)
                eps_renamed = eps.rename(index={"gg": "Greenhouse–Geisser", "hf": "Huynh–Feldt"})
                st.dataframe(eps_renamed)
        except Exception as e:
            st.warning(f"Sphericity check failed: {e}")
    else:
        st.info("Install `pingouin` to run sphericity diagnostics.")

# =========================== TOTALS & AUC ANALYSES ===========================

def totals_auc_peak(long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-rat totals (sum), AUC (trapezoid), peak value, time-to-peak."""
    def metrics(group):
        t = group["TimeMin"].values
        y = group["Distance"].values
        # ensure time sorted
        idx = np.argsort(t)
        t, y = t[idx], y[idx]
        auc = np.trapz(y, t)
        peak = float(y.max())
        tpeak = float(t[y.argmax()])
        total = float(y.sum())
        return pd.Series({"Total": total, "AUC": auc, "Peak": peak, "TimeToPeak": tpeak})

    per_rat = long.groupby(["Group", "Prenatal", "Drug", "Rat"]).apply(metrics).reset_index()
    tall = per_rat.melt(id_vars=["Group", "Prenatal", "Drug", "Rat"], var_name="Metric", value_name="Value")
    return per_rat, tall

def anova_totals(df_tot: pd.DataFrame, dv: str = "Total"):
    model = smf.ols(f"{dv} ~ C(Prenatal)*C(Drug)", data=df_tot).fit()
    aov = anova_lm(model, typ=2)
    # effect sizes
    ss_resid = aov.loc["Residual", "sum_sq"]
    eff = {}
    for eff_name in ["C(Prenatal)", "C(Drug)", "C(Prenatal):C(Drug)"]:
        if eff_name in aov.index:
            eff[eff_name] = partial_eta_squared(aov.loc[eff_name, "sum_sq"], ss_resid)
    aov_es = aov.copy()
    for k, v in eff.items():
        aov_es.loc[k, "partial_eta2"] = v
    # Post-hoc Tukey on 6 groups (only for Total)
    tuk_df = None
    if dv == "Total":
        tuk = pairwise_tukeyhsd(df_tot["Total"], df_tot["Group"], alpha=0.05)
        tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
        tuk_df.columns = [c.lower().replace(" ", "_") for c in tuk_df.columns]
    return aov_es, tuk_df, model

def plot_totals_violin(df_tot: pd.DataFrame, dv: str, order: Sequence[str]):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    x = np.arange(len(order))
    data = [df_tot.loc[df_tot["Group"] == g, dv].values for g in order]
    parts = ax.violinplot(data, positions=x, showmeans=False, showmedians=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor("0.8"); pc.set_edgecolor("black"); pc.set_alpha(0.6)
    # box + jitter
    ax.boxplot(data, positions=x, widths=0.38, showcaps=True, showfliers=False)
    for i, y in enumerate(data):
        xx = np.random.normal(loc=x[i], scale=0.05, size=y.size)
        ax.plot(xx, y, "o", ms=3, color="black", alpha=0.5)
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylabel(dv)
    style_axes(ax)
    return fig

def plot_totals_bars(df_tot: pd.DataFrame, order: Sequence[str], tukey_df: Optional[pd.DataFrame]):
    means = [df_tot.loc[df_tot["Group"] == g, "Total"].mean() for g in order]
    sems  = [df_tot.loc[df_tot["Group"] == g, "Total"].sem()  for g in order]
    fills = ["white", "0.85", "0.70", "0.55", "0.40", "0.25"]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
    x = np.arange(len(order))
    ax.bar(x, means, yerr=sems, capsize=3, edgecolor="black", linewidth=1.0,
           color=fills[:len(order)], width=0.65)
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylabel("Total after amphetamine (cm)")
    style_axes(ax)
    # Tukey brackets
    if tukey_df is not None and not tukey_df.empty:
        sig = tukey_df.query("reject == True").copy()
        base = max(np.array(means) + np.array(sems))
        step = 0.07 * base
        for idx, row in sig.reset_index(drop=True).iterrows():
            g1, g2 = row["group1"], row["group2"]
            if g1 in order and g2 in order:
                i, j = order.index(g1), order.index(g2)
                xi, xj = x[i], x[j]
                y = base + (idx + 1) * step
                ax.plot([xi, xi, xj, xj], [y, y+0.015*base, y+0.015*base, y],
                        color="black", lw=1.0, linestyle=(0,(4,3)))
                star = "***" if row["p-adj"] < 0.001 else "**" if row["p-adj"] < 0.01 else "*" if row["p-adj"] < 0.05 else "ns"
                ax.text((xi+xj)/2, y+0.02*base, star, ha="center", va="bottom", fontsize=11)
    return fig

def planned_contrasts_total(df_tot: pd.DataFrame) -> pd.DataFrame:
    """
    Example planned contrasts (two-sample t-tests):
      - Within LPS: LPS_NS vs LPS_DZP, LPS_NS vs LPS_BSP, LPS_DZP vs LPS_BSP
      - Between prenatal at each drug: LPS_? vs NS_?
    Adjust p-values with Holm.
    """
    pairs = [
        ("LPS_NS", "LPS_DZP"), ("LPS_NS", "LPS_BSP"), ("LPS_DZP", "LPS_BSP"),
        ("LPS_NS", "NS_NS"), ("LPS_DZP", "NS_DZP"), ("LPS_BSP", "NS_BSP"),
    ]
    rows = []
    for a, b in pairs:
        A = df_tot.loc[df_tot["Group"] == a, "Total"].values
        B = df_tot.loc[df_tot["Group"] == b, "Total"].values
        t, p = stats.ttest_ind(A, B, equal_var=False)  # Welch t
        diff = A.mean() - B.mean()
        rows.append(dict(contrast=f"{a} - {b}", t=t, p=p, mean_diff=diff))
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out

with tab_tot:
    st.subheader("Per-rat metrics: totals, AUC, peak, time-to-peak")
    per_rat, tall = totals_auc_peak(long)
    st.dataframe(per_rat.head(20), use_container_width=True)

    # choose DV
    dv = st.selectbox("Choose metric for plots/ANOVA", ["Total", "AUC", "Peak", "TimeToPeak"])
    order6 = sorted(per_rat["Group"].unique())
    st.markdown("#### Distribution (violin + box + jitter)")
    fig_v = plot_totals_violin(per_rat, dv, order6)
    st.pyplot(fig_v, use_container_width=True)

    st.markdown("#### Two-way ANOVA (Prenatal × Drug) + effect sizes")
    aov_es, tuk_df, model = anova_totals(per_rat, dv=dv)
    st.dataframe(aov_es, use_container_width=True)

    st.markdown("#### Tukey HSD (6 groups) — only for Total")
    if dv == "Total" and tuk_df is not None:
        st.dataframe(tuk_df, use_container_width=True)
        st.markdown("#### Bar plot (journal-style, with Tukey brackets)")
        fig_bars = plot_totals_bars(per_rat, order6, tuk_df)
        st.pyplot(fig_bars, use_container_width=True)
    else:
        st.info("Tukey is computed for Total only (to mirror the paper’s Panel b).")

    st.markdown("#### Planned contrasts (Holm-adjusted; Welch t-tests)")
    pc = planned_contrasts_total(per_rat)
    st.dataframe(pc, use_container_width=True)

    st.markdown("#### Assumption checks (for Total model)")
    if dv == "Total":
        resid = model.resid
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Residual QQ")
            figq, axq = plt.subplots(figsize=(3.2, 3.2), dpi=200)
            sm.ProbPlot(resid).qqplot(line="s", ax=axq)
            axq.set_title("QQ plot")
            st.pyplot(figq)
        with col2:
            st.caption("Levene across 6 groups")
            W, pL = stats.levene(*[per_rat.loc[per_rat["Group"]==g, "Total"] for g in order6], center="median")
            st.write(f"W = {W:.3f}, p = {pL:.4f}")

# =============================== PER-BIN ANOVAS ==============================

def per_bin_anovas(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t, dsub in long.groupby("TimeMin"):
        model = smf.ols("Distance ~ C(Prenatal)*C(Drug)", data=dsub).fit()
        aov = anova_lm(model, typ=2)
        ss_resid = aov.loc["Residual", "sum_sq"]
        for eff in ["C(Prenatal)", "C(Drug)", "C(Prenatal):C(Drug)"]:
            if eff in aov.index:
                F = aov.loc[eff, "F"]; p = aov.loc[eff, "PR(>F)"]
                eta_p = partial_eta_squared(aov.loc[eff, "sum_sq"], ss_resid)
                rows.append(dict(TimeMin=t, Effect=eff, F=F, p=p, partial_eta2=eta_p))
    out = pd.DataFrame(rows)
    # Holm-FDR per effect across time
    out["p_holm"] = np.nan
    for eff in out["Effect"].unique():
        idx = out["Effect"] == eff
        out.loc[idx, "p_holm"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    return out

with tab_bin:
    st.subheader("Two-way ANOVAs per time bin (Prenatal × Drug)")
    bin_tbl = per_bin_anovas(long)
    st.dataframe(bin_tbl.round(4), use_container_width=True)

# ============================ ADVANCED (ROBUST/LME) ==========================

with tab_adv:
    st.subheader("Robust sensitivity: Welch one-way across 6 groups (Total)")
    per_rat_total = per_rat.copy()
    welch = pg.welch_anova(dv="Total", between="Group", data=per_rat_total) if HAVE_PG else None
    if welch is not None:
        st.dataframe(welch, use_container_width=True)
    else:
        st.info("Install `pingouin` for Welch ANOVA.")

    st.subheader("Linear Mixed-Effects (random intercept for Rat)")
    st.caption("Model: Distance ~ C(TimeMin)*C(Prenatal)*C(Drug) + (1|Rat). Useful if you add baseline or have missing bins.")
    try:
        # Reduce factor levels for stability on small samples
        mdl = smf.mixedlm("Distance ~ C(TimeMin)*C(Prenatal)*C(Drug)", data=long, groups=long["Rat"])
        mfit = mdl.fit(method="lbfgs", maxiter=200, disp=False)
        st.text(mfit.summary().as_text())
        st.write({"AIC": mfit.aic, "BIC": mfit.bic})
    except Exception as e:
        st.warning(f"MixedLM failed: {e}")

    st.subheader("Interpretation (auto-generated)")
    # Simple narrative from totals + mixed anova if available
    narrative = []
    if HAVE_PG:
        try:
            aov_tc = pg.mixed_anova(dv="Distance", within="Time", between="Group",
                                    subject="Rat", data=long.rename(columns={"TimeMin":"Time"}))
            sig_tc = aov_tc[aov_tc["p-unc"] < 0.05]
            if not sig_tc.empty:
                narrative.append("Time course shows significant effects:")
                for _, r in sig_tc.iterrows():
                    narrative.append(f"  • {r['Source']}: F={r['F']:.2f}, p={r['p-unc']:.4g}")
        except Exception:
            pass
    aov_totals, tuk, _ = anova_totals(per_rat, dv="Total")
    sig_rows = aov_totals[(aov_totals.index != "Residual") & (aov_totals["PR(>F)"] < 0.05)]
    if not sig_rows.empty:
        narrative.append("Totals ANOVA indicates:")
        for idx, r in sig_rows.iterrows():
            narrative.append(f"  • {idx}: F={r['F']:.2f}, p={r['PR(>F)']:.4g}, ηp²={r['partial_eta2']:.3f}")
    if tuk is not None:
        sig = tuk[tuk["reject"] == True]
        if not sig.empty:
            best = sig.sort_values("p-adj").head(5)
            narrative.append("Key post-hoc differences (Tukey):")
            for _, r in best.iterrows():
                star = "***" if r["p-adj"] < 0.001 else "**" if r["p-adj"] < 0.01 else "*"
                narrative.append(f"  • {r['group1']} vs {r['group2']}: Δ={r['meandiff']:.1f}, p={r['p-adj']:.4g} {star}")
    if narrative:
        st.code("\n".join(narrative))
    else:
        st.info("Run analyses first or provide larger sample sizes for stable summaries.")

# ================================ DOWNLOADS ==================================

with tab_dl:
    st.subheader("Download all tables & figures")

    # Tables
    colA, colB, colC = st.columns(3)
    with colA:
        st.caption("Long (tidy) data")
        st.download_button("CSV", long.to_csv(index=False).encode(), "long_data.csv", "text/csv")
        st.caption("Per-rat metrics (Total/AUC/Peak/TTP)")
        st.download_button("CSV", per_rat.to_csv(index=False).encode(), "per_rat_metrics.csv", "text/csv")
    with colB:
        st.caption("Time-course summary (mean±SEM)")
        st.download_button("CSV", timecourse_summary(long).to_csv(index=False).encode(), "timecourse_summary.csv", "text/csv")
        st.caption("Per-bin ANOVAs")
        st.download_button("CSV", bin_tbl.to_csv(index=False).encode(), "per_bin_anovas.csv", "text/csv")
    with colC:
        st.caption("Totals ANOVA (for Total)")
        aov_total_only, tuk_only, _ = anova_totals(per_rat, "Total")
        st.download_button("CSV", aov_total_only.to_csv().encode(), "totals_anova_total.csv", "text/csv")
        if tuk_only is not None:
            st.caption("Tukey HSD (6 groups)")
            st.download_button("CSV", tuk_only.to_csv(index=False).encode(), "tukey_total.csv", "text/csv")

    # Figures
    colF, colG, colH = st.columns(3)
    with colF:
        st.caption("Time course (PNG)")
        st.download_button("Download", fig_bytes(fig_tc), "figure_timecourse.png", "image/png")
    with colG:
        st.caption("Totals violin (PNG)")
        st.download_button("Download", fig_bytes(fig_v), "figure_totals_violin.png", "image/png")
    if "fig_bars" in globals():
        with colH:
            st.caption("Totals bars (PNG)")
            st.download_button("Download", fig_bytes(fig_bars), "figure_totals_bars.png", "image/png")

# =============================================================================
# End of app
# =============================================================================
