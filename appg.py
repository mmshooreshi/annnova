# app.py — Locomotion Figure Lab (EXHAUSTIVE, merged)
# =============================================================================
# Features unified from both versions:
#   1) CSV mapping (Group, Rat, time-bins) → long format with derived factors.
#   2) QC: missingness, group Ns, descriptives, outlier flags; spaghetti & box/strip.
#   3) Time-course: five journal styles (A–E), mean±SEM; Mixed ANOVA; sphericity (Mauchly) + GG/HF.
#   4) Per-rat METRICS: Total, AUC, Peak, Time-to-Peak; violin/box/jitter; 2×3 ANOVA + partial η²;
#      Tukey on 6 groups (Total) + planned contrasts; assumptions.
#   5) WINDOWED totals: PRE / POST / Δ (POST−PRE) with ANOVA+effect sizes, Tukey, CLD letters,
#      diagnostics (Levene, Shapiro), PRE baseline one-way ANOVA + CI figure, insight narratives;
#      Outlier & Prenatal effect exploration tools.
#   6) Per-bin 2×3 ANOVAs with partial η² + Holm-adjusted p.
#   7) Advanced: Welch one-way (Totals) if pingouin available; LME Distance ~ Time*Prenatal*Drug + (1|Rat).
#   8) Downloads: all tables & figures (PNG/CSV/TXT).
# =============================================================================

from __future__ import annotations

import io
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

# Optional: pingouin (mixed ANOVA, sphericity, Welch)
try:
    import pingouin as pg  # pip install pingouin
    HAVE_PG = True
except Exception:
    HAVE_PG = False

# ------------------------------- PAGE SETUP ----------------------------------

st.set_page_config(page_title="Locomotion Figure Lab — Exhaustive", layout="wide")
st.markdown(
    "<style>div.block-container{padding-top:0.75rem;padding-bottom:1.25rem;}</style>",
    unsafe_allow_html=True,
)

# --------------------------------- MODELS ------------------------------------

@dataclass
class ColumnMap:
    group: str
    rat: str
    bins: List[str]
    time_start: int
    bin_width: int

# --------------------------------- HELPERS -----------------------------------

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
    # Subject unique per Group to be safe for repeated measures across groups
    long["Subject"] = (long["Group"].astype(str) + "_" + long["Rat"].astype(str))

    # Numeric DV & deterministic time
    long["Distance"] = pd.to_numeric(long["Distance"], errors="coerce")
    times = infer_time_labels(cmap.bins, cmap.time_start, cmap.bin_width)
    mapper = {b: t for b, t in zip(cmap.bins, times)}
    long["TimeMin"] = long["TimeBin"].map(mapper)

    pren, drug = zip(*[derive_factors_from_group(g) for g in long["Group"]])
    long["Prenatal"] = pren
    long["Drug"] = drug

    long = long.dropna(subset=["Distance"])
    return long

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
    try:
        return float(ss_effect) / float(ss_effect + ss_error)
    except Exception:
        return np.nan

def iqr_outlier_flags(x: np.ndarray, k: float = 1.5) -> np.ndarray:
    q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (x < lo) | (x > hi)

# --------------------------- TIME COURSE SUMMARY -----------------------------

def summarize_timecourse(long: pd.DataFrame) -> pd.DataFrame:
    return (
        long.groupby(["Group", "TimeMin"])["Distance"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )

# -------------------- TIME COURSE: FIVE PUBLICATION STYLES -------------------

from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

DRUG_MARKER = {"NS": "o", "DZP": "^", "BSP": "s"}
PREN_FILL   = {"NS": ("white", "black"), "LPS": ("black", "black")}
PREN_SHADE  = {"NS": "0.88", "LPS": "0.75"}
GROUP_LS    = {
    "NS_NS": "-", "NS_DZP": (0, (4, 3)), "NS_BSP": (0, (1, 2)),
    "LPS_NS": (0, (6, 3)), "LPS_DZP": (0, (2, 2)), "LPS_BSP": (0, (3, 1, 1, 1)),
}

def _prep_summary(summary: pd.DataFrame, legend_order: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    s = summary.copy()
    s["TimeMin"] = pd.to_numeric(s["TimeMin"], errors="coerce")
    s = s.dropna(subset=["TimeMin", "mean"]).sort_values(["Group", "TimeMin"])
    present = [g for g in legend_order if g in s["Group"].unique().tolist()]
    return s, present

def _tick_grid(ax, y_major=None):
    style_axes(ax, y_major=y_major)
    ax.yaxis.grid(True, linewidth=0.4, color="0.9")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total Beam Breaks (n)")

def normalize_epsilon(eps_raw):
    """
    Normalize pingouin.epsilon output into a DataFrame with
    index ['Greenhouse–Geisser','Huynh–Feldt','Lower-bound'] (if available).
    """
    import pandas as pd
    import numpy as np

    mapping = {"gg": "Greenhouse–Geisser", "hf": "Huynh–Feldt", "lb": "Lower-bound"}

    # If already a DataFrame
    if isinstance(eps_raw, pd.DataFrame):
        eps_df = eps_raw.copy()
        if eps_df.shape[0] == 1 and set(["gg","hf","lb"]).issubset(eps_df.columns):
            eps_df = eps_df.T
            eps_df.columns = ["epsilon"]
        return eps_df.rename(index=mapping)

    # If Series
    if isinstance(eps_raw, pd.Series):
        return eps_raw.rename(index=mapping).to_frame("epsilon")

    # If dict
    if isinstance(eps_raw, dict):
        return pd.DataFrame.from_dict(eps_raw, orient="index", columns=["epsilon"]).rename(index=mapping)

    # If numpy array, list, or tuple
    if isinstance(eps_raw, (np.ndarray, list, tuple)):
        keys = ["gg","hf","lb"][:len(eps_raw)]
        eps_df = pd.DataFrame({"epsilon": list(eps_raw)}, index=keys)
        return eps_df.rename(index=mapping)

    # If single float
    if np.isscalar(eps_raw):
        return pd.DataFrame({"epsilon": [float(eps_raw)]}, index=[mapping.get("gg","gg")])

    # If Index or other weird type: just wrap values as strings
    try:
        vals = list(eps_raw)
        keys = ["gg","hf","lb"][:len(vals)]
        eps_df = pd.DataFrame({"epsilon": vals}, index=keys)
        return eps_df.rename(index=mapping)
    except Exception:
        return pd.DataFrame({"epsilon": [np.nan]}, index=["Unknown"])


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

def _build_legend_handles(groups: List[str]) -> List[Line2D]:
    handles = []
    for g in groups:
        pren, drug = derive_factors_from_group(g)
        marker = DRUG_MARKER.get(drug, "o")
        mfc, mec = PREN_FILL.get(pren, ("white", "black"))
        handles.append(Line2D([0], [0],
                              marker=marker, linestyle="none",
                              mfc=mfc, mec=mec, mew=1.1, ms=6, color="black", label=g))
    return handles

def _right_edge_labels(ax, end_points: List[Tuple[float, float, str]], min_sep_ratio=0.04):
    if not end_points:
        return
    end_points = sorted(end_points, key=lambda x: x[1])
    yvals = [y for _, y, _ in end_points]
    ymin, ymax = min(yvals), max(yvals)
    span = max(ymax - ymin, 1e-9)
    min_sep = span * min_sep_ratio
    placed_y = []
    for _, y, _ in end_points:
        y_target = y
        for py in placed_y:
            if abs(y_target - py) < min_sep:
                y_target = py + min_sep
        placed_y.append(y_target)
    xlims = ax.get_xlim()
    x_right = xlims[1]
    for (_, y, lab), ydraw in zip(end_points, placed_y):
        ax.text(x_right, ydraw, f" {lab}",
                ha="left", va="center", fontsize=10, color="black",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.plot([x_right - 0.02*(x_right - xlims[0]), x_right],
                [ydraw, ydraw], color="black", linewidth=0.8)

def plot_timecourse5(summary: pd.DataFrame,
                     legend_order: Sequence[str],
                     variant: str = "A") -> Tuple[plt.Figure, plt.Axes]:
    s, groups = _prep_summary(summary, legend_order)
    fig, ax = plt.subplots(figsize=(7.4, 4.3), dpi=240)
    end_pts = []

    for gi, g in enumerate(groups):
        sub = s.loc[s["Group"] == g, ["TimeMin", "mean", "sem"]]
        if sub.empty:
            continue
        t = sub["TimeMin"].to_numpy()
        m = sub["mean"].to_numpy()
        se = sub["sem"].to_numpy()
        mask = np.isfinite(t) & np.isfinite(m) & np.isfinite(se)
        t, m, se = t[mask], m[mask], se[mask]
        if t.size == 0:
            continue

        pren, drug = derive_factors_from_group(g)
        marker = DRUG_MARKER.get(drug, "o")
        mfc, mec = PREN_FILL.get(pren, ("white", "black"))
        shade = PREN_SHADE.get(pren, "0.85")

        V = variant.upper()
        if V == "A":  # Ribbon Classic
            ax.fill_between(t, m - se, m + se, color=shade, alpha=1.0, linewidth=0)
            ax.plot(t, m, color="black", lw=1.8)
            ax.plot(t, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.5, mew=1.1, label=g)
        elif V == "B":  # Errorbar Classic with jitter
            tj = _marker_jitter_positions(t, gi, len(groups))
            ax.errorbar(tj, m, yerr=se, color="black", lw=1.4, capsize=2, elinewidth=1.0)
            ax.plot(tj, m, ls="-", color="black", lw=0.8)
            ax.plot(tj, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.5, mew=1.1, label=g)
        elif V == "C":  # Marker-only
            ax.errorbar(t, m, yerr=se, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
            ax.plot(t, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=6.0, mew=1.2, label=g)
        elif V == "D":  # Linestyle suite
            ls = GROUP_LS.get(g, "-")
            ax.fill_between(t, m - se, m + se, color="0.9", alpha=1.0, linewidth=0)
            ax.plot(t, m, color="black", lw=1.6, ls=ls)
            ax.plot(t, m, ls="none", marker="o", mfc=mfc, mec=mec, ms=5.2, mew=1.0, label=g)
        elif V == "E":  # Direct-labeled
            ax.plot(t, m, color="black", lw=1.8)
            ax.fill_between(t, m - se, m + se, color="0.92", alpha=1.0, linewidth=0)
            ax.plot(t, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.2, mew=1.0)
            end_pts.append((t[-1], m[-1], g))
        else:
            raise ValueError("variant must be one of A, B, C, D, E")

    _tick_grid(ax)
    ax.set_xticks(np.arange(min(s["TimeMin"].min(), 0)+10, s["TimeMin"].max()+10, 10))

    if variant.upper() == "E":
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax * 1.05 if xmax > 0 else xmax + 0.5)
        _right_edge_labels(ax, end_pts, min_sep_ratio=0.05)
    else:
        handles = _build_legend_handles(groups)
        leg = ax.legend(handles=handles, frameon=False, ncol=3, columnspacing=1.1,
                        handletextpad=0.5, borderaxespad=0.0, loc="upper left")
        for h in leg.legend_handles:
            h.set_linestyle("none")

    fig.tight_layout()
    return fig, ax

# ------------------------------ MIXED ANOVA ----------------------------------

def mixed_anova_timecourse(long: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not HAVE_PG:
        return None
    dat = long.rename(columns={"TimeMin": "Time"})
    dat["Subject"] = long["Subject"].astype(str)
    dat["Time"] = dat["Time"].astype(str)
    try:
        return pg.mixed_anova(dv="Distance", within="Time", between="Group", subject="Subject", data=dat)
    except Exception:
        return None

# ------------------------- TOTALS/AUC/PEAK/TTP (ALL) -------------------------

def totals_auc_peak(long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def metrics(group):
        t = group["TimeMin"].values
        y = group["Distance"].values
        idx = np.argsort(t)
        t, y = t[idx], y[idx]
        auc = np.trapz(y, t)
        peak = float(y.max())
        tpeak = float(t[y.argmax()])
        total = float(y.sum())
        return pd.Series({"Total": total, "AUC": auc, "Peak": peak, "TimeToPeak": tpeak})

    per_rat = long.groupby(["Group", "Prenatal", "Drug", "Rat"], as_index=False).apply(metrics).reset_index(drop=True)
    tall = per_rat.melt(id_vars=["Group", "Prenatal", "Drug", "Rat"], var_name="Metric", value_name="Value")
    return per_rat, tall

def anova_totals(df_tot: pd.DataFrame, dv: str = "Total"):
    model = smf.ols(f"{dv} ~ C(Prenatal)*C(Drug)", data=df_tot).fit()
    aov = anova_lm(model, typ=2)
    ss_resid = aov.loc["Residual", "sum_sq"]
    eff = {}
    for eff_name in ["C(Prenatal)", "C(Drug)", "C(Prenatal):C(Drug)"]:
        if eff_name in aov.index:
            eff[eff_name] = partial_eta_squared(aov.loc[eff_name, "sum_sq"], ss_resid)
    aov_es = aov.copy()
    for k, v in eff.items():
        aov_es.loc[k, "partial_eta2"] = v
    tuk_df = None
    if dv == "Total":
        tuk = pairwise_tukeyhsd(df_tot["Total"], df_tot["Group"], alpha=0.05)
        tuk_df = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
        tuk_df.columns = [c.lower().replace(" ", "_") for c in tuk_df.columns]
        tuk_df["reject"] = tuk_df["reject"].astype(bool)
    return aov_es, tuk_df, model

def planned_contrasts_total(df_tot: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("LPS_NS", "LPS_DZP"), ("LPS_NS", "LPS_BSP"), ("LPS_DZP", "LPS_BSP"),
        ("LPS_NS", "NS_NS"), ("LPS_DZP", "NS_DZP"), ("LPS_BSP", "NS_BSP"),
    ]
    rows = []
    for a, b in pairs:
        A = df_tot.loc[df_tot["Group"] == a, "Total"].values
        B = df_tot.loc[df_tot["Group"] == b, "Total"].values
        if len(A) >= 2 and len(B) >= 2:
            t, p = stats.ttest_ind(A, B, equal_var=False)
        else:
            t, p = (np.nan, np.nan)
        diff = (A.mean() - B.mean()) if len(A) and len(B) else np.nan
        rows.append(dict(contrast=f"{a} - {b}", t=t, p=p, mean_diff=diff))
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1] if out["p"].notna().any() else np.nan
    return out

def plot_totals_violin(df_tot: pd.DataFrame, dv: str, order: Sequence[str]):
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    x = np.arange(len(order))
    data = [df_tot.loc[df_tot["Group"] == g, dv].values for g in order]
    parts = ax.violinplot(data, positions=x, showmeans=False, showmedians=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor("0.8"); pc.set_edgecolor("black"); pc.set_alpha(0.6)
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
    ax.set_ylabel("Total Beam Breaks (n)")
    style_axes(ax)
    if tukey_df is not None and not tukey_df.empty:
        sig = tukey_df.query("reject == True").copy()
        base = max(np.array(means) + np.array(sems)) if len(means) else 0
        step = 0.07 * base if base else 1.0
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

# -------------------------- WINDOWED TOTALS TOOLING --------------------------

from scipy.stats import t as tdist

def group_descriptives(totals: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    desc = (
        totals.groupby("Group")["Distance"]
        .agg(mean="mean", sd="std", n="count")
        .reindex(order)
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
    sig_pairs = set(tuple(sorted([r["group1"], r["group2"]])) for _, r in tukey_df.iterrows() if r["reject"])
    letter_sets: List[set] = []
    for g in groups:
        placed = False
        for s in letter_sets:
            if any(tuple(sorted([g, h])) in sig_pairs for h in s):
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


import numpy as np
import pandas as pd
from scipy.stats import chi2

def mauchly_test(df, subject_col="Rat", within_cols=None):
    """
    Compute Mauchly’s Test of Sphericity for wide-format data.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe (subjects × timepoints).
    subject_col : str
        Column for subject IDs (ignored in computation).
    within_cols : list of str
        Columns for repeated measures (e.g., time points).

    Returns
    -------
    dict with W, chi2, df, p
    """
    if within_cols is None:
        within_cols = df.columns.drop([subject_col, "Group"])

    X = df[within_cols].to_numpy(dtype=float)

    # covariance matrix across time points
    S = np.cov(X, rowvar=False)

    k = S.shape[0]   # number of timepoints
    n = X.shape[0]   # number of subjects

    # compute W
    detS = np.linalg.det(S)
    trS = np.trace(S)
    W = detS / ((trS / k) ** k)

    # chi-square statistic
    df_chi = int((k - 1) * (k + 2) / 2)
    chi2_stat = -(n - 1) * (2*k + 1) / 6 * np.log(W)
    p_val = 1 - chi2.cdf(chi2_stat, df_chi)

    return {"W": W, "chi2": chi2_stat, "df": df_chi, "p": p_val}

def format_totals_narrative(desc: pd.DataFrame, aov_es: pd.DataFrame, tukey_df: pd.DataFrame) -> str:
    lines = []
    labels = {
        "C(Prenatal)": "Prenatal",
        "C(Drug)": "Drug",
        "C(Prenatal):C(Drug)": "Prenatal × Drug"
    }
    for eff, label in labels.items():
        if eff in aov_es.index:
            F = aov_es.loc[eff, "F"]; p = aov_es.loc[eff, "PR(>F)"]
            pet = aov_es.loc[eff].get("partial_eta2", np.nan)
            sig = "✅ significant" if p < 0.05 else "❌ not significant"
            lines.append(f"• {label} effect: F={F:.2f}, p={p:.3g}, partial η²={pet:.3f} → {sig}.")
    top = desc.sort_values("mean", ascending=False)
    if not top.empty:
        hi = top.iloc[0]; lo = top.iloc[-1]
        lines.append(f"• Highest mean: {hi['Group']} ({hi['mean']:.1f} ± {hi['sem']:.1f}); "
                     f"lowest: {lo['Group']} ({lo['mean']:.1f} ± {lo['sem']:.1f}).")
    sig = tukey_df.query("reject == True").sort_values("p-adj") if not tukey_df.empty else pd.DataFrame()
    if not sig.empty:
        first = sig.iloc[0]
        direction = ">" if first["meandiff"] > 0 else "<"
        lines.append(f"• Strongest pairwise separation (Tukey): {first['group1']} {direction} {first['group2']} "
                     f"(Δ={abs(first['meandiff']):.1f}, p_adj={first['p-adj']:.3g}).")
        picks = []
        for _, r in sig.head(6).iterrows():
            arrow = ">" if r["meandiff"] > 0 else "<"
            picks.append(f"{r['group1']} {arrow} {r['group2']}")
        if picks:
            lines.append("• Notable significant pairs: " + "; ".join(picks) + ".")
    else:
        lines.append("• Tukey: no pairwise differences reached p<.05 after correction.")
    return "\n".join(lines)

def totals_by_window(long: pd.DataFrame, t_from: int, t_to: int) -> pd.DataFrame:
    dat = long[(long["TimeMin"] >= t_from) & (long["TimeMin"] <= t_to)].copy()
    out = dat.groupby(["Group", "Rat"], as_index=False)["Distance"].sum()
    pren, drug = zip(*[derive_factors_from_group(g) for g in out["Group"]])
    out["Prenatal"] = pren
    out["Drug"] = drug
    return out

def totals_delta(pre_totals: pd.DataFrame, post_totals: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(post_totals, pre_totals, on=["Group", "Rat"], suffixes=("_post", "_pre"))
    m["Distance"] = m["Distance_post"] - m["Distance_pre"]
    m["Prenatal"] = m["Group"].map(lambda g: derive_factors_from_group(g)[0])
    m["Drug"]     = m["Group"].map(lambda g: derive_factors_from_group(g)[1])
    return m[["Group", "Rat", "Prenatal", "Drug", "Distance"]]

def twoway_anova_totals(totals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = smf.ols("Distance ~ C(Prenatal)*C(Drug)", data=totals).fit()
    aov = anova_lm(model, typ=2)
    tukey_res = pairwise_tukeyhsd(endog=totals["Distance"], groups=totals["Group"], alpha=0.05)
    tukey_df = pd.DataFrame(tukey_res.summary().data[1:], columns=tukey_res.summary().data[0])
    tukey_df.columns = [c.lower().replace(" ", "_") for c in tukey_df.columns]
    tukey_df["reject"] = tukey_df["reject"].astype(bool)
    tukey_df["meandiff"] = pd.to_numeric(tukey_df["meandiff"], errors="coerce")
    tukey_df["p-adj"] = pd.to_numeric(tukey_df["p-adj"], errors="coerce")
    tukey_df["lower"] = pd.to_numeric(tukey_df["lower"], errors="coerce")
    tukey_df["upper"] = pd.to_numeric(tukey_df["upper"], errors="coerce")
    return aov, tukey_df

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

def variance_overview(totals: pd.DataFrame) -> pd.DataFrame:
    v = (totals.groupby("Group")["Distance"]
         .agg(var="var", sd=lambda s: s.std(ddof=1), n="count")
         .reset_index())
    if not v["var"].isna().all():
        vmax, vmin = v["var"].max(), v["var"].min()
        v["var_ratio_max_min"] = (vmax / vmin) if vmin not in (0, np.nan) else np.nan
    return v

def plot_baseline_bar(pre_totals: pd.DataFrame, desc: pd.DataFrame, order: Sequence[str]) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=200)
    means = desc.set_index("Group").loc[order, "mean"].values
    ci95s = desc.set_index("Group").loc[order, "ci95"].values
    x = np.arange(len(order))
    ax.bar(x, means, yerr=ci95s, capsize=3, color="0.85", edgecolor="black", linewidth=1.0, width=0.65)
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylabel("Baseline total beam breaks (n)")
    style_axes(ax)
    return fig, ax

def make_totals_package(totals: pd.DataFrame, group_order: Sequence[str]) -> dict:
    aov_tot, tukey_df = twoway_anova_totals(totals)
    aov_es = add_effect_sizes(aov_tot)
    desc = group_descriptives(totals, group_order)
    means_series = desc.set_index("Group")["mean"]
    cld = compact_letter_display(tukey_df, means_series) if not tukey_df.empty else {}
    vtab = variance_overview(totals)
    W, p_lev = stats.levene(*[g["Distance"].values for _, g in totals.groupby("Group")], center="median")
    norm = normality_by_group(totals)
    # Figure with CLD letters
    fig_bars = plot_totals_bar_CLD(totals, group_order, tukey_df, cld)
    narrative = format_totals_narrative(desc, aov_es, tukey_df)
    return dict(
        aov=aov_es, tukey=tukey_df, desc=desc, cld=cld,
        variances=vtab, levene=(W, p_lev), normality=norm,
        fig=fig_bars, narrative=narrative
    )

def plot_totals_bar_CLD(
    totals: pd.DataFrame, group_order: Sequence[str], tukey_summary: Optional[pd.DataFrame], cld_letters: Optional[Dict[str, str]]
) -> plt.Figure:
    means = [totals.loc[totals["Group"] == g, "Distance"].mean() for g in group_order]
    sems  = [totals.loc[totals["Group"] == g, "Distance"].sem() for g in group_order]
    fills = ["white", "0.85", "0.70", "0.55", "0.40", "0.25"]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    x = np.arange(len(group_order))
    ax.bar(x, means, yerr=sems, capsize=3, edgecolor="black", linewidth=1.0,
           color=fills[: len(group_order)], width=0.65)
    ax.set_xticks(x, group_order, rotation=25, ha="right")
    ax.set_ylabel("Total Beam Breaks (after amphetamine) (n)")
    style_axes(ax)
    if tukey_summary is not None and not tukey_summary.empty:
        sig = tukey_summary.query("reject == True").copy()
        y0 = max(np.array(means) + np.array(sems)) if len(means) else 0
        step = 0.07 * y0 if y0 else 1.0
        pairs, heights = [], []
        for idx, row in sig.reset_index(drop=True).iterrows():
            g1, g2 = row["group1"], row["group2"]
            if g1 in group_order and g2 in group_order:
                i, j = group_order.index(g1), group_order.index(g2)
                stars = ("***" if row["p-adj"] < 0.001 else
                         "**" if row["p-adj"] < 0.01 else
                         "*" if row["p-adj"] < 0.05 else "ns")
                pairs.append((i, j, stars))
                heights.append(y0 + (idx + 1) * step)
        for (i, j, txt), h in zip(pairs, heights):
            xi, xj = x[i], x[j]
            ax.plot([xi, xi, xj, xj], [h, h + step * 0.2, h + step * 0.2, h],
                    lw=1.0, color="black", linestyle=(0, (4, 3)))
            ax.text((xi + xj) / 2, h + step * 0.25, txt, ha="center", va="bottom", fontsize=11)
    if cld_letters:
        ymax = (np.array(means) + np.array(sems)).max() if len(means) else 0
        ypad = 0.06 * ymax if ymax else 1.0
        for idx, g in enumerate(group_order):
            lab = cld_letters.get(g, "")
            # if lab:
            #     ax.text(x[idx], means[idx] + sems[idx] + ypad, lab,
            #             ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    return fig

# ------------------------------- PER-BIN ANOVAS ------------------------------

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
    out["p_holm"] = np.nan
    for eff in out["Effect"].unique():
        idx = out["Effect"] == eff
        out.loc[idx, "p_holm"] = multipletests(out.loc[idx, "p"], method="holm")[1]
    return out

# ----------------------------------- UI --------------------------------------

st.title("Locomotion Figure Lab — Exhaustive")
st.caption("Upload your CSV, map columns, and generate publication-style figures with full statistics & downloads.")

with st.sidebar:
    st.header("1) Data")
    # Try loading default CSV if nothing uploaded
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = load_csv(up)
        st.success("✅ Using your uploaded file")
    else:
        try:
            df = pd.read_csv("AIH_simple.csv")
            st.info("📂 No file uploaded — using default sample: AIH_simple.csv")
        except Exception:
            st.error("⚠️ No file uploaded and default sample not found. Please upload a CSV to proceed.")
            st.stop()


    st.header("2) Column mapping")
    cols = list(df.columns)
    group_col = st.selectbox("Group column", cols, index=cols.index("Group") if "Group" in cols else 0)
    rat_col = st.selectbox("Rat ID column", cols, index=cols.index("Rat") if "Rat" in cols else 1)
    candidate_bins = [c for c in cols if c not in (group_col, rat_col)]
    bin_cols = st.multiselect("Time-bin columns (ordered)", candidate_bins, default=candidate_bins)
    time_start = st.number_input("Start time (min) of first selected bin", value=10, step=5)
    bin_width = st.number_input("Bin width (min)", value=10, step=5)
    cmap = ColumnMap(group=group_col, rat=rat_col, bins=bin_cols, time_start=time_start, bin_width=bin_width)

    # Windows defaults
    labels = infer_time_labels(bin_cols, time_start, bin_width) if bin_cols else []
    if labels:
        tmin, tmax = min(labels), max(labels)
        pre_default  = (labels[0], 30 if 30 in labels else labels[min(2, len(labels)-1)])
        post_default = (40 if 40 in labels else labels[max(len(labels)//2, 0)], labels[-1])

        st.header("3) Windows (PRE/POST for Δ)")
        pre_range = st.slider("PRE window (min)", min_value=tmin, max_value=tmax,
                              value=pre_default, step=bin_width)
        post_range = st.slider("POST window (min)", min_value=tmin, max_value=tmax,
                               value=post_default, step=bin_width)
    else:
        pre_range = post_range = None

    if len(cmap.bins) == 0:
        st.error("Select at least one time-bin column.")
        st.stop()

# Transform
long = to_long(df, cmap)
summary_tc = summarize_timecourse(long)

# Consistent group order
group_order = sorted(long["Group"].unique().tolist())

# ------------------------------ Tabs layout ----------------------------------

tab_qc, tab_tc, tab_tot, tab_win, tab_bin, tab_adv, tab_dl = st.tabs([
    "QC & Descriptives",
    "Time Course",
    "Per-rat Metrics (Totals/AUC/Peak/TTP)",
    "Windows: PRE / POST / Δ",
    "Per-bin ANOVAs",
    "Advanced (Robust & LME)",
    "Downloads",
])

# =========================== QC & DESCRIPTIVES ===============================

with tab_qc:
    st.subheader("Long (tidy) preview")
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
        sub_sorted = sub.sort_values("TimeMin")
        ax_sp.plot(sub_sorted["TimeMin"], sub_sorted["Distance"], lw=0.8, alpha=0.4)
    ax_sp.set_xlabel("Time (min)")
    ax_sp.set_ylabel("Beam Breaks Count (n)")
    style_axes(ax_sp)
    st.pyplot(fig_sp, use_container_width=True)

    st.markdown("### Per-bin distribution (box + strip) by Group")
    bin_pick = st.selectbox("Pick a time bin (min)", sorted(long["TimeMin"].unique()))
    sub = long[long["TimeMin"] == bin_pick]
    fig_box, ax_box = plt.subplots(figsize=(8, 3.8), dpi=200)
    order = sorted(sub["Group"].unique())
    positions = np.arange(len(order))
    data_by_group = [sub.loc[sub["Group"] == g, "Distance"].values for g in order]
    ax_box.boxplot(data_by_group, positions=positions, widths=0.6, showfliers=False)
    for i, g in enumerate(order):
        y = sub.loc[sub["Group"] == g, "Distance"].values
        xj = np.random.normal(loc=i, scale=0.05, size=len(y))
        ax_box.plot(xj, y, "o", ms=3, alpha=0.6, color="black")
    ax_box.set_xticks(positions, order, rotation=20, ha="right")
    ax_box.set_ylabel("Beam Breaks Count (n)")
    style_axes(ax_box)
    st.pyplot(fig_box, use_container_width=True)

# =============================== TIME COURSE =================================

# =============================== TIME COURSE =================================
# =============================== TIME COURSE (FINAL PAPER) ===============================
# =============================== TIME COURSE =================================
with tab_tc:
    st.header("تحلیل مسیر زمانی (AIH locomotion)")

    # --- Time course plot ---
    st.subheader("📈 نمودار مسیر زمانی")
    fig_tc, ax_tc = plot_timecourse5(summary_tc, group_order, variant="B")  # Errorbars only
    st.pyplot(fig_tc, use_container_width=True)
    st.download_button("⬇️ دانلود نمودار مسیر زمانی", fig_bytes(fig_tc),
                       "figure_timecourse.png", "image/png")

    # --- Mixed ANOVA ---
    st.subheader("🔎 آنالیز واریانس با اندازه‌گیری مکرر (Mixed ANOVA)")
    aov, eps_df = None, None
    if HAVE_PG:
        try:
            aov = mixed_anova_timecourse(long)
            if aov is not None:
                st.dataframe(aov.round(4), use_container_width=True)
                st.download_button("⬇️ دانلود ANOVA",
                                   aov.to_csv(index=False).encode(),
                                   "mixed_anova.csv", "text/csv")
        except Exception as e:
            st.error(f"Mixed ANOVA failed: {e}")
    else:
        st.warning("Install `pingouin` for Mixed ANOVA.")

    # --- Mauchly’s Test + Epsilon ---
    st.subheader("⚖️ آزمون کرویت موچلی + تصحیحات GG/HF")
    if HAVE_PG and aov is not None:
        wide = long.pivot_table(index="Subject", columns="TimeMin",
                                values="Distance", aggfunc="mean")
        wide = wide.dropna(axis=0, how="any")
        wide = wide.loc[:, wide.var(axis=0) > 0]

        if wide.shape[1] >= 3 and wide.shape[0] >= 2:
            try:
                sph = pg.sphericity(wide, method="mauchly")
                if isinstance(sph, tuple) and len(sph) >= 4:
                    W, pval, chi2, dof = sph[0], sph[1], sph[2], sph[3]
                    st.write(f"χ²({dof}) = {chi2:.2f}, p = {pval:.4g}, W = {W:.4f}")
            except Exception as e:
                st.warning(f"Sphericity test failed: {e}")

            try:
                eps_raw = pg.epsilon(wide)
                eps_df = normalize_epsilon(eps_raw)
                st.subheader("Epsilon Corrections")
                st.dataframe(eps_df.round(4), use_container_width=True)
                st.download_button("⬇️ دانلود ضرایب اپسیلون",
                                   eps_df.to_csv().encode(),
                                   "epsilon_corrections.csv", "text/csv")
            except Exception as e:
                st.warning(f"Epsilon computation failed: {e}")

    # --- Post hoc tests ---
    st.subheader("🔬 تست‌های تعقیبی (Post hoc)")
    if aov is not None:
        sig_effects = aov.query("`p-unc` < 0.05")["Source"].tolist()

        if "Time" in sig_effects:
            st.markdown("**مقایسه زمان‌ها (Pairwise, Holm correction)**")
            pw = pg.pairwise_ttests(dv="Distance", within="Time", subject="Subject",
                                    data=long.rename(columns={"TimeMin":"Time"}),
                                    padjust="holm")
            st.dataframe(pw.round(4), use_container_width=True)
            st.download_button("⬇️ دانلود Post hoc زمان",
                               pw.to_csv(index=False).encode(),
                               "posthoc_time.csv", "text/csv")

        if "Time * Group" in sig_effects or "Interaction" in sig_effects:
            st.markdown("**مقایسه زمان × گروه**")
            pw_int = pg.pairwise_ttests(dv="Distance", within="Time", between="Group",
                                        subject="Subject",
                                        data=long.rename(columns={"TimeMin":"Time"}),
                                        padjust="holm")
            st.dataframe(pw_int.round(4), use_container_width=True)
            st.download_button("⬇️ دانلود Post hoc زمان×گروه",
                               pw_int.to_csv(index=False).encode(),
                               "posthoc_time_group.csv", "text/csv")

    # --- Summary & Narrative ---
    if aov is not None:
        st.subheader("📑 Summary Tables & Reporting")

        # ANOVA summary
        aov_summary = aov[["Source","DF1","DF2","F","p-unc","np2"]].round(4)
        st.dataframe(aov_summary, use_container_width=True)
        st.download_button("⬇️ Download ANOVA Summary",
                           aov_summary.to_csv(index=False).encode(),
                           "anova_summary.csv","text/csv")

        # Narrative
        st.subheader("Narrative Interpretation")
        text_lines = []
        for _, row in aov_summary.iterrows():
            src, df1, df2, Fv, p, eta = row
            sig = "معنادار ✅" if p < 0.05 else "غیرمعنادار ❌"
            text_lines.append(
                f"• اثر {src}: F({int(df1)},{int(df2)})={Fv:.2f}, p={p:.4g}, η²={eta:.3f} → {sig}"
            )
        st.markdown("\n".join(text_lines))
        st.download_button("⬇️ Download Narrative",
                           "\n".join(text_lines).encode(),
                           "anova_narrative.txt","text/plain")

with tab_tot:
    st.subheader("Per-rat metrics: totals, AUC, peak, time-to-peak")
    per_rat, tall = totals_auc_peak(long)
    st.dataframe(per_rat.head(20), use_container_width=True)

    dv = st.selectbox("Choose metric for plots/ANOVA", ["Total", "AUC", "Peak", "TimeToPeak"])
    order6 = sorted(per_rat["Group"].unique())
    st.markdown("#### Distribution (violin + box + jitter)")
    fig_v = plot_totals_violin(per_rat, dv, order6)
    st.pyplot(fig_v, use_container_width=True)

    st.markdown("#### Two-way ANOVA (Prenatal × Drug) + partial η²")
    aov_es, tuk_df, model = anova_totals(per_rat, dv=dv)
    st.dataframe(aov_es, use_container_width=True)

    st.markdown("#### Tukey HSD (6 groups) — only for Total")
    if dv == "Total" and tuk_df is not None:
        st.dataframe(tuk_df, use_container_width=True)
        st.markdown("#### Bar plot (journal-style, with Tukey brackets)")
        fig_bars_total = plot_totals_bars(per_rat, order6, tuk_df)
        st.pyplot(fig_bars_total, use_container_width=True)
    else:
        fig_bars_total = None
        st.info("Tukey is computed for Total only (to mirror a classic 'panel b').")

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

# ================================ WINDOWS ====================================

with tab_win:
    st.subheader("Totals (PRE / POST / Δ)")
    if pre_range is None or post_range is None:
        st.warning("Define PRE/POST windows in the sidebar to continue.")
        st.stop()

    pre_totals  = totals_by_window(long, pre_range[0],  pre_range[1])
    post_totals = totals_by_window(long, post_range[0], post_range[1])
    delta_tot   = totals_delta(pre_totals, post_totals)

    t_pre, t_post, t_delta = st.tabs([
        f"PRE  [{pre_range[0]}–{pre_range[1]} min]",
        f"POST [{post_range[0]}–{post_range[1]} min]",
        "Δ  (POST − PRE)"
    ])

    # PRE
    with t_pre:
        st.caption("Summed distance per rat over the PRE window.")
        st.markdown("**Per-rat totals (PRE)**")
        st.dataframe(pre_totals.sort_values(["Group","Rat"]).reset_index(drop=True), use_container_width=True)
        st.download_button("Download PRE per-rat totals (CSV)",
                           pre_totals.to_csv(index=False).encode(),
                           "pre_per_rat_totals.csv", "text/csv")
        pkg_pre = make_totals_package(pre_totals, group_order)
        st.session_state["totals_fig_pre"] = pkg_pre["fig"]

        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg_pre["aov"].round(4), use_container_width=True)

        st.markdown("**Tukey HSD (6 groups) + CLD letters**")
        st.dataframe(pkg_pre["tukey"], use_container_width=True)

        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg_pre["desc"].copy()
        if pkg_pre["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg_pre["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)

        st.markdown("**Diagnostics**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Levene’s test (variance homogeneity)")
            W, p_lev = pkg_pre["levene"]
            st.write(f"W = {W:.3f}, p = {p_lev:.4f}")
            st.caption("Group variances / SDs")
            st.dataframe(pkg_pre["variances"].round(4), use_container_width=True)
        with col2:
            st.caption("Shapiro–Wilk per group")
            st.dataframe(pkg_pre["normality"].round(4), use_container_width=True)

        st.subheader("PRE totals (grayscale bars with CLD)")
        st.pyplot(pkg_pre["fig"], use_container_width=True)

        st.subheader("PRE: Insights & conclusions")
        st.markdown(pkg_pre["narrative"])

        st.markdown("**Baseline (PRE) one-way ANOVA across 6 groups**")
        model_pre = smf.ols("Distance ~ C(Group)", data=pre_totals).fit()
        aov_pre = anova_lm(model_pre, typ=2)
        desc_pre = (
            pre_totals.groupby("Group")["Distance"]
            .agg(mean="mean", sd="std", n="count").reset_index()
        )
        desc_pre["sem"] = desc_pre["sd"] / np.sqrt(desc_pre["n"])
        crit = desc_pre["n"].apply(lambda k: stats.t.ppf(0.975, max(k - 1, 1)))
        desc_pre["ci95"] = crit * desc_pre["sem"]
        st.dataframe(aov_pre.round(4), use_container_width=True)
        st.dataframe(desc_pre.round(3), use_container_width=True)
        pval = aov_pre.loc["C(Group)", "PR(>F)"]
        if pval >= 0.05:
            st.success(f"Baseline groups did not differ (p = {pval:.3g}); post-amphetamine ANOVA can be interpreted without correction.")
        else:
            st.warning(f"Baseline groups differ (p = {pval:.3g}); consider baseline correction (ANCOVA or Δ analysis).")
        st.subheader("Baseline (PRE) totals: mean ± 95% CI")
        fig_pre, _ = plot_baseline_bar(pre_totals, desc_pre, group_order)
        st.pyplot(fig_pre, use_container_width=True)

    # POST
    with t_post:
        st.caption("Summed distance per rat over the POST window.")
        st.markdown("**Per-rat totals (POST)**")
        st.dataframe(post_totals.sort_values(["Group","Rat"]).reset_index(drop=True), use_container_width=True)
        st.download_button("Download POST per-rat totals (CSV)",
                           post_totals.to_csv(index=False).encode(),
                           "post_per_rat_totals.csv", "text/csv")
        pkg_post = make_totals_package(post_totals, group_order)
        st.session_state["totals_fig_post"] = pkg_post["fig"]

        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg_post["aov"].round(4), use_container_width=True)

        st.markdown("**Tukey HSD (6 groups) + CLD letters**")
        st.dataframe(pkg_post["tukey"], use_container_width=True)

        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg_post["desc"].copy()
        if pkg_post["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg_post["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)

        st.markdown("**Diagnostics**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Levene’s test (variance homogeneity)")
            W, p_lev = pkg_post["levene"]
            st.write(f"W = {W:.3f}, p = {p_lev:.4f}")
            st.caption("Group variances / SDs")
            st.dataframe(pkg_post["variances"].round(4), use_container_width=True)
        with col2:
            st.caption("Shapiro–Wilk per group")
            st.dataframe(pkg_post["normality"].round(4), use_container_width=True)

        st.subheader("POST totals (grayscale bars with CLD)")
        st.pyplot(pkg_post["fig"], use_container_width=True)

        st.subheader("POST: Insights & conclusions")
        st.markdown(pkg_post["narrative"])

        # Collapse across Drug to show Prenatal main effect only
        pren_post = post_totals.groupby("Prenatal")["Distance"].agg(["mean","sem"]).reset_index()
        # pren_post["CLD"] = ["a","b"]  # because Prenatal effect was significant

        fig, ax = plt.subplots(figsize=(4,6), dpi=200)
        x = np.arange(len(pren_post))

        # Bars
        ax.bar(x, pren_post["mean"], yerr=pren_post["sem"],
            color=["0.8","0.4"], edgecolor="black", capsize=5, width=0.6)

        # CLD letters (optional, you can drop them if you want only the star)
        # for i,row in pren_post.iterrows():
        #     ax.text(i, row["mean"] + row["sem"] + 20, row["CLD"],
        #             ha="center", va="bottom", fontsize=14, fontweight="bold")

        # --- Add significance bar with asterisk ---
        # Coordinates: start at bar 0 and bar 1
        x1, x2 = 0, 1
        y, h, col = pren_post["mean"].max() + pren_post["sem"].max() + 40, 20, "black"
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        ax.text((x1+x2)*.5, y+h+5, "*", ha="center", va="bottom", color=col, fontsize=16)

        # Axis formatting
        ax.set_xticks(x, pren_post["Prenatal"])
        ax.set_ylabel("POST total (mean ± SEM)")
        style_axes(ax)
        st.pyplot(fig, use_container_width=True)
                

    # DELTA
    with t_delta:
        st.caption("Per-rat change: POST − PRE. Positive = increase after amphetamine.")
        st.markdown("**Per-rat Δ (POST − PRE)**")
        st.dataframe(delta_tot.sort_values(["Group","Rat"]).reset_index(drop=True), use_container_width=True)
        st.download_button("Download Δ per-rat values (CSV)",
                           delta_tot.to_csv(index=False).encode(),
                           "delta_per_rat.csv", "text/csv")
        pkg_delta = make_totals_package(delta_tot, group_order)
        st.session_state["totals_fig_delta"] = pkg_delta["fig"]

        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg_delta["aov"].round(4), use_container_width=True)

        st.markdown("**Tukey HSD (6 groups) + CLD letters**")
        st.dataframe(pkg_delta["tukey"], use_container_width=True)

        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg_delta["desc"].copy()
        if pkg_delta["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg_delta["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)

        st.markdown("**Diagnostics**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Levene’s test (variance homogeneity)")
            W, p_lev = pkg_delta["levene"]
            st.write(f"W = {W:.3f}, p = {p_lev:.4f}")
            st.caption("Group variances / SDs")
            st.dataframe(pkg_delta["variances"].round(4), use_container_width=True)
        with col2:
            st.caption("Shapiro–Wilk per group")
            st.dataframe(pkg_delta["normality"].round(4), use_container_width=True)

        st.subheader("Δ totals (grayscale bars with CLD)")
        st.pyplot(pkg_delta["fig"], use_container_width=True)

        st.subheader("Δ: Insights & conclusions")
        st.markdown("Interpretation: Δ reflects change **after amphetamine** relative to the chosen PRE window. If baseline differs, Δ helps isolate treatment-related shifts.")
        st.markdown(pkg_delta["narrative"])

        with st.expander("Prenatal Effect Explorer (Δ)", expanded=False):
            st.markdown("Removes symmetric extremes within each group and re-tests Prenatal effect.")
            results = []
            max_remove = 3
            for n in range(max_remove + 1):
                dfp = delta_tot.copy()
                pruned = []
                for g, subg in dfp.groupby("Group"):
                    sub_sorted = subg.sort_values("Distance")
                    if n == 0:
                        pruned.append(sub_sorted)
                    elif len(sub_sorted) > 2*n:
                        pruned.append(sub_sorted.iloc[n:-n])
                    else:
                        pruned.append(sub_sorted)
                df_pruned = pd.concat(pruned)
                pkgp = make_totals_package(df_pruned, group_order)
                pval = pkgp["aov"].loc["C(Prenatal)", "PR(>F)"] if "C(Prenatal)" in pkgp["aov"].index else np.nan
                results.append({"Removed per group": n, "Prenatal p-value": pval})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df)
            fig, ax = plt.subplots()
            ax.plot(res_df["Removed per group"], res_df["Prenatal p-value"], marker="o")
            ax.axhline(0.05, color="red", linestyle="--")
            ax.set_xlabel("Rats removed per group (high & low Δ)")
            ax.set_ylabel("Prenatal p-value")
            style_axes(ax)
            st.pyplot(fig)

        with st.expander("Outlier Exploration (Δ)", expanded=False):
            q1 = delta_tot["Distance"].quantile(0.25)
            q3 = delta_tot["Distance"].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
            st.write(f"Default IQR range: [{lower:.1f}, {upper:.1f}]")
            method = st.radio("Outlier rule", ["IQR (±1.5)", "Z > 3"], horizontal=True)
            if method == "IQR (±1.5)":
                mask = (delta_tot["Distance"] < lower) | (delta_tot["Distance"] > upper)
            else:
                zscores = (delta_tot["Distance"] - delta_tot["Distance"].mean()) / delta_tot["Distance"].std(ddof=1)
                mask = zscores.abs() > 3
            flagged = delta_tot.loc[mask]
            st.write("### Flagged outliers", flagged)
            exclude_ids = st.multiselect(
                "Select rats to exclude",
                options=[f"{r.Group}_{r.Rat}" for r in flagged.itertuples()],
            )
            if exclude_ids:
                filt = delta_tot[~delta_tot.apply(lambda r: f"{r.Group}_{r.Rat}" in exclude_ids, axis=1)]
            else:
                filt = delta_tot
            pkg_filt = make_totals_package(filt, group_order)
            st.subheader("Recomputed ANOVA/Tukey (after exclusions)")
            st.dataframe(pkg_filt["aov"].round(4), use_container_width=True)
            st.dataframe(pkg_filt["tukey"], use_container_width=True)
            st.pyplot(pkg_filt["fig"], use_container_width=True)
            st.markdown(pkg_filt["narrative"])

# =============================== PER-BIN ANOVAS ==============================

with tab_bin:
    st.subheader("Two-way ANOVAs per time bin (Prenatal × Drug)")
    bin_tbl = per_bin_anovas(long).round(4)
    st.dataframe(bin_tbl, use_container_width=True)

# ============================ ADVANCED (ROBUST/LME) ==========================

with tab_adv:
    st.subheader("Robust sensitivity: Welch one-way across 6 groups (Totals over all bins)")
    totals_all = long.groupby(["Group", "Rat"], as_index=False)["Distance"].sum()
    totals_all["Prenatal"] = totals_all["Group"].map(lambda g: derive_factors_from_group(g)[0])
    totals_all["Drug"] = totals_all["Group"].map(lambda g: derive_factors_from_group(g)[1])
    if HAVE_PG:
        welch = pg.welch_anova(dv="Distance", between="Group", data=totals_all)
        st.dataframe(welch, use_container_width=True)
    else:
        st.info("Install `pingouin` for Welch ANOVA.")

    st.subheader("Linear Mixed-Effects (random intercept for Rat)")
    st.caption("Model: Distance ~ C(TimeMin)*C(Prenatal)*C(Drug) + (1|Rat). Useful with missing bins / baseline.")
    try:
        mdl = smf.mixedlm("Distance ~ C(TimeMin)*C(Prenatal)*C(Drug)", data=long, groups=long["Rat"])
        mfit = mdl.fit(method="lbfgs", maxiter=200, disp=False)
        st.text(mfit.summary().as_text())
        st.write({"AIC": mfit.aic, "BIC": mfit.bic})
    except Exception as e:
        st.warning(f"MixedLM failed: {e}")

    st.subheader("Interpretation (auto-generated)")
    narrative_lines = []
    if HAVE_PG:
        try:
            aov_tc = pg.mixed_anova(dv="Distance", within="Time", between="Group",
                                    subject="Subject", data=long.rename(columns={"TimeMin":"Time"}))
            sig_tc = aov_tc[aov_tc["p-unc"] < 0.05]
            if not sig_tc.empty:
                narrative_lines.append("Time course shows significant effects:")
                for _, r in sig_tc.iterrows():
                    narrative_lines.append(f"  • {r['Source']}: F={r['F']:.2f}, p={r['p-unc']:.4g}")
        except Exception:
            pass
    aov_totals_all, tuk_all = twoway_anova_totals(totals_all)
    aov_totals_es = add_effect_sizes(aov_totals_all)
    sig_rows = aov_totals_es[(aov_totals_es.index != "Residual") & (aov_totals_es["PR(>F)"] < 0.05)]
    if not sig_rows.empty:
        narrative_lines.append("Totals ANOVA indicates:")
        for idx, r in sig_rows.iterrows():
            pet = r.get("partial_eta2", np.nan)
            narrative_lines.append(f"  • {idx}: F={r['F']:.2f}, p={r['PR(>F)']:.4g}, ηp²={pet:.3f}")
    if tuk_all is not None and not tuk_all.empty:
        sig = tuk_all[tuk_all["reject"] == True]
        if not sig.empty:
            best = sig.sort_values("p-adj").head(5)
            narrative_lines.append("Key post-hoc differences (Tukey):")
            for _, r in best.iterrows():
                star = "***" if r["p-adj"] < 0.001 else "**" if r["p-adj"] < 0.01 else "*"
                narrative_lines.append(f"  • {r['group1']} vs {r['group2']}: Δ={r['meandiff']:.1f}, p={r['p-adj']:.4g} {star}")
    if narrative_lines:
        st.code("\n".join(narrative_lines))
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
        per_rat_all, _ = totals_auc_peak(long)
        st.download_button("CSV", per_rat_all.to_csv(index=False).encode(), "per_rat_metrics.csv", "text/csv")
    with colB:
        st.caption("Time-course summary (mean±SEM)")
        st.download_button("CSV", summarize_timecourse(long).to_csv(index=False).encode(), "timecourse_summary.csv", "text/csv")
        st.caption("Per-bin ANOVAs")
        st.download_button("CSV", per_bin_anovas(long).to_csv(index=False).encode(), "per_bin_anovas.csv", "text/csv")
    with colC:
        st.caption("Totals ANOVA (for Total)")
        aov_total_only, tuk_only, _ = anova_totals(per_rat_all, "Total")
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
    with colH:
        st.caption("Totals bars (PNG)")
        if 'fig_bars_total' in locals() and fig_bars_total is not None:
            st.download_button("Download", fig_bytes(fig_bars_total), "figure_totals_bars.png", "image/png")
        else:
            st.info("Totals bars will appear after running Tukey on Total in the Per-rat Metrics tab.")

    st.markdown("### PRE/POST/Δ Figures")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_pre_dl = st.session_state.get("totals_fig_pre", None)
        if fig_pre_dl is not None:
            st.download_button("PRE totals (PNG)", fig_bytes(fig_pre_dl), "pre_totals.png", "image/png")
    with c2:
        fig_post_dl = st.session_state.get("totals_fig_post", None)
        if fig_post_dl is not None:
            st.download_button("POST totals (PNG)", fig_bytes(fig_post_dl), "post_totals.png", "image/png")
    with c3:
        fig_delta_dl = st.session_state.get("totals_fig_delta", None)
        if fig_delta_dl is not None:
            st.download_button("Δ totals (PNG)", fig_bytes(fig_delta_dl), "delta_totals.png", "image/png")

st.markdown("---")
st.caption(
    "Tip: Adjust the selected bin columns or start/width in the sidebar to redefine analysis windows (e.g., to include baseline bins or restrict to 40–120 min)."
)
