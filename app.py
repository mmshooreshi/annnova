# app.py
# -----------------------------------------------------------------------------
# Locomotion Analysis & NPP-style Figures (Streamlit)
# -----------------------------------------------------------------------------
# What you get:
#   • Upload CSV → map columns (Group, Rat, bins).
#   • Auto-long reshape with derived factors:
#         Prenatal = first token of Group (e.g., LPS vs NS)
#         Drug     = second token of Group (e.g., NS/DZP/BSP)
#   • Time-course plot (mean±SEM), black-only markers/lines (journal style).
#   • Mixed (repeated) ANOVA on the time course (between = 6-group "Group";
#     within = Time). (Uses pingouin if present; graceful fallback if absent.)
#   • Per-bin two-way ANOVAs: Prenatal (2) × Drug (3), table per bin.
#   • Totals (sum of selected bins) two-way ANOVA + Tukey HSD on 6 groups.
#   • Diagnostics (QQ, Levene) for totals model.
#   • Publication-style bar plot (grayscale) with dashed significance brackets.
#   • One-click downloads (CSV tables + PNG figures).
#
# Expected CSV shape (flexible; you map columns in-app):
#   Group,Rat,1 10',2 10',...,9 10'
#
# Best practices used:
#   • Clear separation of I/O, transform, analyze, visualize.
#   • Robust parsing/mapping, caching for heavy steps.
#   • Defensive fallbacks when optional libs are missing.
#   • Reusable plotting/stats helpers with docstrings & type hints.
# -----------------------------------------------------------------------------

from __future__ import annotations

import io
import itertools
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

import streamlit as st


# ---------- DESCRIPTIVES, EFFECT SIZES, CLD, DIAGNOSTICS, NARRATIVE ----------

from scipy.stats import t as tdist

def group_descriptives(totals: pd.DataFrame, group_order: Sequence[str]) -> pd.DataFrame:
    desc = (
        totals.groupby("Group")["Distance"]
        .agg(mean="mean", sd="std", n="count")
        .reindex(group_order)
        .reset_index()
    )
    desc["sem"] = desc["sd"] / np.sqrt(desc["n"])
    # 95% CI using t critical per group
    crit = desc["n"].apply(lambda k: tdist.ppf(0.975, max(k - 1, 1)))
    desc["ci95"] = crit * desc["sem"]
    return desc

def add_effect_sizes(aov: pd.DataFrame) -> pd.DataFrame:
    """Add eta², partial eta², and omega² to ANOVA (Type II)."""
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


# ========================= TIME COURSE: FIVE VARIATIONS =========================
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

def set_pub_fonts(base=10):
    """Classic, legible typography."""
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 2,
        "axes.labelsize": base + 1,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 1,
        "figure.dpi": 240,
        "savefig.dpi": 300,
        "mathtext.fontset": "stix",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlepad": 8,
    })

def _split_group(g: str) -> Tuple[str, str]:
    parts = re.split(r"[_\-\s]+", str(g).strip())
    return (parts[0], parts[1]) if len(parts) >= 2 else ("UNK", "UNK")

# Encodings (black-only, classic)
DRUG_MARKER = {"NS": "o", "DZP": "^", "BSP": "s"}
PREN_FILL   = {"NS": ("white", "black"), "LPS": ("black", "black")}
PREN_SHADE  = {"NS": "0.88", "LPS": "0.75"}   # SEM ribbons
GROUP_LS    = {  # distinct line styles for variant D
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

def _marker_jitter_positions(t: np.ndarray, idx: int, total: int) -> np.ndarray:
    """Small x-offsets to reduce overlap on markers/errorbars (variant B)."""
    if t.size < 2:
        step = 1.0
    else:
        diffs = np.diff(np.unique(t))
        step = np.median(diffs) if diffs.size else 1.0
    # symmetric offsets across groups, <= ~12% of step
    spread = 0.12 * step
    if total == 1:
        return t
    offs = np.linspace(-spread, spread, total)
    return t + offs[idx]

def _build_legend_handles(groups: List[str]) -> List[Line2D]:
    handles = []
    for g in groups:
        pren, drug = _split_group(g)
        marker = DRUG_MARKER.get(drug, "o")
        mfc, mec = PREN_FILL.get(pren, ("white", "black"))
        handles.append(Line2D([0], [0],
                              marker=marker, linestyle="none",
                              mfc=mfc, mec=mec, mew=1.1, ms=6, color="black", label=g))
    return handles

def _right_edge_labels(ax, end_points: List[Tuple[float, float, str]], min_sep_ratio=0.04):
    """
    Place group labels at the right edge with anti-overlap.
    end_points: list of (x_last, y_last, label).
    """
    if not end_points:
        return
    # sort by y
    end_points = sorted(end_points, key=lambda x: x[1])
    yvals = [y for _, y, _ in end_points]
    ymin, ymax = min(yvals), max(yvals)
    span = max(ymax - ymin, 1e-9)
    min_sep = span * min_sep_ratio
    placed_y = []
    for _, y, lab in end_points:
        y_target = y
        # bump minimally to keep separation
        for py in placed_y:
            if abs(y_target - py) < min_sep:
                y_target = py + min_sep
        placed_y.append(y_target)
    # draw
    xlims = ax.get_xlim()
    x_right = xlims[1]
    for (_, y, lab), ydraw in zip(end_points, placed_y):
        txt = ax.text(x_right, ydraw, f" {lab}",
                      ha="left", va="center", fontsize=10, color="black",
                      path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        # guiding short tick
        ax.plot([x_right - 0.02*(x_right - xlims[0]), x_right],
                [ydraw, ydraw], color="black", linewidth=0.8)

# at the top of the file you already have:
# from matplotlib.ticker import MultipleLocator
# and _marker_jitter_positions() helper

def plot_timecourse5(summary: pd.DataFrame,
                     legend_order: Sequence[str],
                     variant: str = "A") -> Tuple[plt.Figure, plt.Axes]:
    set_pub_fonts(base=10)
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

        pren, drug = _split_group(g)
        marker = DRUG_MARKER.get(drug, "o")
        mfc, mec = PREN_FILL.get(pren, ("white", "black"))
        shade = PREN_SHADE.get(pren, "0.85")

        if variant.upper() == "A":
            ax.fill_between(t, m - se, m + se, color=shade, alpha=1.0, linewidth=0, zorder=1)
            ax.plot(t, m, color="black", lw=1.8, zorder=3)
            ax.plot(t, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.5, mew=1.1, zorder=4, label=g)

        elif variant.upper() == "B":
            tj = _marker_jitter_positions(t, gi, len(groups))
            ax.errorbar(tj, m, yerr=se, color="black", lw=1.4, capsize=2, elinewidth=1.0, zorder=3)
            ax.plot(tj, m, ls="-", color="black", lw=0.8, zorder=2)
            ax.plot(tj, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.5, mew=1.1, zorder=4, label=g)

        # >>> NEW: tiny-symbol classic error bars
        elif variant.upper() == "B_TINY":
            tj = _marker_jitter_positions(t, gi, len(groups))
            # thinner everything + smaller markers, classic look
            ax.errorbar(
                tj, m, yerr=se,
                fmt="none",
                ecolor="black",
                elinewidth=0.9, capsize=1.8, capthick=0.9,  # tiny caps
                zorder=2
            )
            ax.plot(
                tj, m,
                ls="-", color="black", lw=0.9, alpha=0.95,  # thin mean line
                zorder=3
            )
            ax.plot(
                tj, m,
                ls="none", marker=marker,
                mfc=mfc, mec=mec,
                ms=4.0, mew=0.9,  # <<< tinier symbols
                zorder=4, label=g
            )

        elif variant.upper() == "C":
            ax.errorbar(t, m, yerr=se, fmt="none", ecolor="black", elinewidth=1.2, capsize=3, zorder=2)
            ax.plot(t, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=6.0, mew=1.2, zorder=4, label=g)

        elif variant.upper() == "D":
            ls = GROUP_LS.get(g, "-")
            ax.fill_between(t, m - se, m + se, color="0.9", alpha=1.0, linewidth=0, zorder=1)
            ax.plot(t, m, color="black", lw=1.6, ls=ls, zorder=3)
            ax.plot(t, m, ls="none", marker="o", mfc=mfc, mec=mec, ms=5.2, mew=1.0, zorder=4, label=g)

        elif variant.upper() == "E":
            ax.plot(t, m, color="black", lw=1.8, zorder=3)
            ax.fill_between(t, m - se, m + se, color="0.92", alpha=1.0, linewidth=0, zorder=1)
            ax.plot(t, m, ls="none", marker=marker, mfc=mfc, mec=mec, ms=5.2, mew=1.0, zorder=4)
            end_pts.append((t[-1], m[-1], g))

        else:
            raise ValueError("variant must be one of A, B, B_TINY, C, D, E")

    # Axes, grid, and ticks
    _tick_grid(ax)
    ax.xaxis.set_major_locator(MultipleLocator(10))     # 10-min ticks
    ax.set_xticks(np.arange(10, 100, 10))               # 10..90 labels
    # ax.set_xlim(10, 90)                                # uncomment to clip strictly to 10..90
    ax.yaxis.grid(True, linewidth=0.4, color="0.90")

    if variant.upper() == "E":
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax * 1.04 if xmax > 0 else xmax + 0.5)
        _right_edge_labels(ax, end_pts, min_sep_ratio=0.05)
    else:
        handles = _build_legend_handles(groups)
        leg = ax.legend(title=None, frameon=False, ncol=3, columnspacing=1.1,
                        handletextpad=0.5, borderaxespad=0.0, loc="upper left")
        for h in leg.legend_handles:
            h.set_linestyle("none")

    fig.tight_layout()
    return fig, ax


def normality_by_group(totals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, sub in totals.groupby("Group"):
        x = pd.to_numeric(sub["Distance"], errors="coerce").dropna()
        try:
            W, p = stats.shapiro(x) if len(x) >= 3 and len(x) <= 5000 else (np.nan, np.nan)
        except Exception:
            W, p = (np.nan, np.nan)
        rows.append(dict(Group=g, n=len(x), mean=x.mean(), sd=x.std(ddof=1), W=W, p=p))
    out = pd.DataFrame(rows).sort_values("Group")
    return out

def variance_overview(totals: pd.DataFrame) -> pd.DataFrame:
    v = (totals.groupby("Group")["Distance"]
         .agg(var="var", sd=lambda s: s.std(ddof=1), n="count")
         .reset_index())
    if not v["var"].isna().all():
        vmax, vmin = v["var"].max(), v["var"].min()
        v["var_ratio_max_min"] = vmax / vmin if vmin not in (0, np.nan) else np.nan
    return v

def compact_letter_display(tukey_df: pd.DataFrame, means: pd.Series) -> Dict[str, str]:
    """
    Greedy CLD (groups sharing a letter are NOT significantly different by Tukey).
    `tukey_df` expects columns: group1, group2, reject (bool).
    """
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

def format_totals_narrative(desc: pd.DataFrame, aov_es: pd.DataFrame, tukey_df: pd.DataFrame) -> str:
    lines = []
    # ANOVA effects (rename for readability)
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
    # Top/bottom groups
    top = desc.sort_values("mean", ascending=False)
    hi = top.iloc[0]; lo = top.iloc[-1]
    lines.append(f"• Highest mean: {hi['Group']} ({hi['mean']:.1f} ± {hi['sem']:.1f}); "
                 f"lowest: {lo['Group']} ({lo['mean']:.1f} ± {lo['sem']:.1f}).")
    # Tukey highlights
    sig = tukey_df.query("reject == True").sort_values("p-adj")
    if not sig.empty:
        first = sig.iloc[0]
        direction = ">" if first["meandiff"] > 0 else "<"
        lines.append(f"• Strongest pairwise separation (Tukey): {first['group1']} {direction} {first['group2']} "
                     f"(Δ={abs(first['meandiff']):.1f}, p_adj={first['p-adj']:.3g}).")
        # Summarize a few pairs
        picks = []
        for _, r in sig.head(6).iterrows():
            arrow = ">" if r["meandiff"] > 0 else "<"
            picks.append(f"{r['group1']} {arrow} {r['group2']}")
        if picks:
            lines.append("• Notable significant pairs: " + "; ".join(picks) + ".")
    else:
        lines.append("• Tukey: no pairwise differences reached p<.05 after correction.")
    return "\n".join(lines)

# Optional: mixed ANOVA convenience
try:
    import pingouin as pg  # pip install pingouin
    HAVE_PG = True
except Exception:
    HAVE_PG = False

# ------------------------------- UI CONFIG -----------------------------------

st.set_page_config(page_title="Locomotion Figure Lab", layout="wide")
st.markdown(
    "<style>div.block-container{padding-top:1rem;padding-bottom:2rem;}</style>",
    unsafe_allow_html=True,
)

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
    """
    Produce monotonically increasing time (min) labels for the bin columns.
    """
    return [int(start + i * step) for i in range(len(bin_cols))]


def derive_factors_from_group(g: str) -> Tuple[str, str]:
    """
    Split group like 'LPS_DZP' → ('LPS','DZP'); robust to separators.
    """
    parts = re.split(r"[_\-\s]+", str(g).strip())
    if len(parts) >= 2:
        return parts[0], parts[1]
    return ("UNK", "UNK")


def to_long(df: pd.DataFrame, cmap: ColumnMap) -> pd.DataFrame:
    """
    Wide → long; add TimeMin, Prenatal, Drug, and sanitize types.
    """
    long = df.melt(
        id_vars=[cmap.group, cmap.rat],
        value_vars=cmap.bins,
        var_name="TimeBin",
        value_name="Distance",
    )
    long = long.rename(columns={cmap.group: "Group", cmap.rat: "Rat"})
    long["Rat"] = long["Rat"].astype(str)

    # time labels (deterministic; don't parse column names)
    time_labels = infer_time_labels(cmap.bins, cmap.time_start, cmap.bin_width)
    mapper = {b: t for b, t in zip(cmap.bins, time_labels)}
    long["TimeMin"] = long["TimeBin"].map(mapper)

    pren, drug = zip(*[derive_factors_from_group(g) for g in long["Group"]])
    long["Prenatal"] = pren
    long["Drug"] = drug

    # numeric DV
    long["Distance"] = pd.to_numeric(long["Distance"], errors="coerce")
    long = long.dropna(subset=["Distance"])
    return long


def summarize_timecourse(long: pd.DataFrame) -> pd.DataFrame:
    """
    Mean/SEM per group × time.
    """
    return (
        long.groupby(["Group", "TimeMin"])["Distance"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )


def totals_by_rat(long: pd.DataFrame) -> pd.DataFrame:
    """
    Sum across selected bins for each rat.
    """
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


def plot_timecourse(
    summary: pd.DataFrame, legend_order: Sequence[str]
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Publication-style time course (black-only):
      - Solid black lines.
      - SEM as light grayscale ribbons (no caps).
      - Marker shape encodes Drug (o=NS, ^=DZP, s=BSP).
      - Marker fill encodes Prenatal (open=NS, filled=LPS).
      - Minimalist axes, tight layout, robust to NaNs.
    Expects columns: Group, TimeMin, mean, sem.
    """

    # --- mappings (black-only aesthetics) ---
    def _split(g: str) -> Tuple[str, str]:
        # 'LPS_DZP' -> ('LPS', 'DZP'); fallback to ('UNK','UNK')
        parts = re.split(r"[_\-\s]+", str(g).strip())
        return (parts[0], parts[1]) if len(parts) >= 2 else ("UNK", "UNK")

    drug_marker = {"NS": "o", "DZP": "^", "BSP": "s"}  # shape by Drug
    prenatal_fill = {"NS": ("white", "black"), "LPS": ("black", "black")}  # (mfc, mec)
    sem_shade = {"NS": "0.88", "LPS": "0.75"}  # subtle grayscale for ribbons

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=240)

    # ensure numeric & sorted time
    summary = summary.copy()
    summary["TimeMin"] = pd.to_numeric(summary["TimeMin"], errors="coerce")
    summary = summary.dropna(subset=["TimeMin", "mean"]).sort_values(["Group", "TimeMin"])

    # keep only groups we’ll plot (and preserve requested order)
    present = [g for g in legend_order if g in summary["Group"].unique().tolist()]

    # plot each group
    for g in present:
        sub = summary.loc[summary["Group"] == g, ["TimeMin", "mean", "sem"]].copy()
        if sub.empty:
            continue
        t = sub["TimeMin"].to_numpy()
        m = sub["mean"].to_numpy()
        s = sub["sem"].to_numpy()

        # mask finite for robustness
        mask = np.isfinite(t) & np.isfinite(m) & np.isfinite(s)
        t, m, s = t[mask], m[mask], s[mask]
        if t.size == 0:
            continue

        pren, drug = _split(g)
        marker = drug_marker.get(drug, "o")
        mfc, mec = prenatal_fill.get(pren, ("white", "black"))
        shade = sem_shade.get(pren, "0.85")

        # SEM ribbon (draw first so line/markers sit on top)
        ax.fill_between(t, m - s, m + s, color=shade, alpha=1.0, linewidth=0)

        # mean line + markers
        ax.plot(
            t,
            m,
            color="black",
            linewidth=1.8,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            t,
            m,
            linestyle="none",
            marker=marker,
            mfc=mfc,
            mec=mec,
            ms=5.5,
            mew=1.1,
            zorder=4,
            label=g,
        )

    # labels & axes style
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Total Beam Breaks (n)")
    style_axes(ax)  # uses your existing helper
    ax.yaxis.grid(True, linewidth=0.4, color="0.9")  # subtle y-grid only

    # legend: compact, 2 rows fits 6 groups nicely
    leg = ax.legend(
        frameon=False,
        ncol=3,
        handlelength=1.6,
        columnspacing=1.2,
        handletextpad=0.6,
        borderaxespad=0.0,
        loc="upper left",
    )
    for lh in leg.legend_handles:
        # keep line-less handles compact (markers only)
        lh.set_linestyle("none")

    # tight layout for export
    fig.tight_layout()
    return fig, ax

def plot_totals_bar(
    totals: pd.DataFrame, group_order: Sequence[str], tukey_summary: Optional[pd.DataFrame],cld_letters: Optional[Dict[str, str]]
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Grayscale bars with SEM caps and dashed significance brackets (from Tukey).
    """
    means = [totals.loc[totals["Group"] == g, "Distance"].mean() for g in group_order]
    sems = [totals.loc[totals["Group"] == g, "Distance"].sem() for g in group_order]
    fills = ["white", "0.85", "0.70", "0.55", "0.40", "0.25"]  # 6 groups
    
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    x = np.arange(len(group_order))
    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=3,
        edgecolor="black",
        linewidth=1.0,
        color=fills[: len(group_order)],
        width=0.65,
    )
    ax.set_xticks(x, group_order, rotation=25, ha="right")
    ax.set_ylabel("Total Beam Breaks\n(after amphetamine) (n)")
    style_axes(ax)

    # significance brackets from Tukey (reject==True)
    if tukey_summary is not None and not tukey_summary.empty:
        sig = tukey_summary.query("reject == True").copy()
        # compute y-positions above the max bar
        y0 = max(np.array(means) + np.array(sems))
        step = 0.07 * y0
        heights = []
        pairs = []
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
            ax.plot(
                [xi, xi, xj, xj],
                [h, h + step * 0.2, h + step * 0.2, h],
                lw=1.0,
                color="black",
                linestyle=(0, (4, 3)),  # dashed
            )
            ax.text((xi + xj) / 2, h + step * 0.25, txt, ha="center", va="bottom", fontsize=11)
    
    if cld_letters:
        ymax = (np.array(means) + np.array(sems)).max()
        ypad = 0.06 * ymax
        for idx, g in enumerate(group_order):
            lab = cld_letters.get(g, "")
            if lab:
                ax.text(x[idx], means[idx] + sems[idx] + ypad, lab,
                        ha="center", va="bottom", fontsize=11)


    return fig, ax

# ------------------------------- STATISTICS ----------------------------------



# ---------------------- WINDOWED TOTALS + ANALYTICS --------------------------

def totals_by_window(long: pd.DataFrame, t_from: int, t_to: int) -> pd.DataFrame:
    """Sum Distance for each Rat across TimeMin in [t_from, t_to]."""
    dat = long[(long["TimeMin"] >= t_from) & (long["TimeMin"] <= t_to)].copy()
    out = dat.groupby(["Group", "Rat"], as_index=False)["Distance"].sum()
    pren, drug = zip(*[derive_factors_from_group(g) for g in out["Group"]])
    out["Prenatal"] = pren
    out["Drug"] = drug
    return out

def totals_delta(pre_totals: pd.DataFrame, post_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Δ per Rat = POST - PRE (inner merge on Group & Rat).
    Keeps factor columns and recomputes Distance as delta.
    """
    m = pd.merge(post_totals, pre_totals, on=["Group", "Rat"], suffixes=("_post", "_pre"))
    m["Distance"] = m["Distance_post"] - m["Distance_pre"]
    # copy factors (same across merge)
    m["Prenatal"] = m["Group"].map(lambda g: derive_factors_from_group(g)[0])
    m["Drug"]     = m["Group"].map(lambda g: derive_factors_from_group(g)[1])
    return m[["Group", "Rat", "Prenatal", "Drug", "Distance"]]

def make_totals_package(totals: pd.DataFrame, group_order: Sequence[str]) -> dict:
    """
    Run full stack: ANOVA (+effect sizes), Tukey, CLD, diagnostics, descriptives, figure.
    Returns dict of everything needed.
    """
    # ANOVA + Tukey
    aov_tot, tukey_df = twoway_anova_totals(totals)
    aov_es = add_effect_sizes(aov_tot)
    # Descriptives + CLD
    desc = group_descriptives(totals, group_order)
    means_series = desc.set_index("Group")["mean"]
    cld = compact_letter_display(tukey_df, means_series) if not tukey_df.empty else {}
    # Diagnostics
    vtab = variance_overview(totals)
    W, p_lev = stats.levene(*[g["Distance"].values for _, g in totals.groupby("Group")], center="median")
    # Normality (per group)
    norm = normality_by_group(totals)
    # Figure
    fig_bars, ax_bars = plot_totals_bar(totals, group_order, tukey_df, cld_letters=cld)
    # Narrative
    narrative = format_totals_narrative(desc, aov_es, tukey_df)
    return dict(
        aov=aov_es, tukey=tukey_df, desc=desc, cld=cld,
        variances=vtab, levene=(W, p_lev), normality=norm,
        fig=fig_bars, ax=ax_bars, narrative=narrative
    )

def mixed_anova_timecourse(
    long: pd.DataFrame,
    time_col: str = "TimeMin",
    group_col: str = "Group",
    subj_col: str = "Rat",
    dv_col: str = "Distance",
    require_complete: bool = True,
) -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Try Pingouin mixed ANOVA: within=Time (categorical), between=Group, subject=Rat.
    Returns (anova_table, info), where info contains diagnostics and any fallback results.
    If Pingouin fails, falls back to statsmodels MixedLM and returns Wald tests for terms.
    """
    info = {"reason": None, "n_subjects": None, "n_groups": None, "times": None,
            "coverage_table": None, "fallback": None}

    df = long[[subj_col, group_col, time_col, dv_col]].dropna().copy()

    # 1) Coerce types and factor-ize
    df[subj_col] = df[subj_col].astype(str)
    df[group_col] = df[group_col].astype(str)

    # treat time as categorical with nice labels (e.g., "10", "20", ...)
    df[time_col] = pd.Categorical(df[time_col].astype(int).astype(str), ordered=True)

    # 2) Coverage diagnostics: each subject should have 1 row per time level
    cov = (df.groupby([subj_col, time_col]).size()
             .unstack(time_col)
             .fillna(0)
             .astype(int))
    info["coverage_table"] = cov.copy()

    # Keep only time levels shared by everyone if require_complete
    if require_complete:
        common_times = [t for t, ok in (cov.sum(axis=0) == cov.shape[0]).items() if ok]
        if len(common_times) < 2:
            info["reason"] = "Not enough common time levels across subjects."
            return None, info
        df = df[df[time_col].isin(common_times)]
        # recompute coverage after trimming
        cov = (df.groupby([subj_col, time_col]).size()
                 .unstack(time_col)
                 .fillna(0)
                 .astype(int))
        # drop subjects missing any common time
        complete_subjects = cov.index[(cov.min(axis=1) >= 1)]
        df = df[df[subj_col].isin(complete_subjects)]
        # update coverage once more for reporting
        cov = (df.groupby([subj_col, time_col]).size()
                 .unstack(time_col)
                 .fillna(0)
                 .astype(int))

    info["n_subjects"] = df[subj_col].nunique()
    info["n_groups"] = df[group_col].nunique()
    info["times"] = list(df[time_col].cat.categories)

    if info["n_subjects"] < 4 or len(info["times"]) < 2:
        info["reason"] = "Too few subjects or time levels after cleaning."
        return None, info

    # 3) Try Pingouin mixed ANOVA
    if HAVE_PG:
        dat = df.rename(columns={subj_col: "Subject", time_col: "Time",
                                 group_col: "Group", dv_col: "Distance"})
        dat["Subject"] = dat["Subject"].astype(str)
        dat["Group"] = dat["Group"].astype(str)
        try:
            aov = pg.mixed_anova(
                dv="Distance", within="Time", between="Group",
                subject="Subject", data=dat
            )
            # pretty index
            aov = aov.rename(columns={"Source": "Effect"}).set_index("Effect")
            return aov, info
        except Exception as e:
            info["reason"] = f"Pingouin failed: {type(e).__name__}: {e}"

    else:
        info["reason"] = "pingouin not installed"

    # 4) Fallback: Linear Mixed Model with random intercept for subject
    # Fixed: C(Time)*C(Group); Random: 1|Subject
    try:
        dff = df.copy()
        dff["Time"] = dff[time_col].astype(str)
        md = smf.mixedlm("Distance ~ C(Time)*C(Group)", data=dff,
                         groups=dff[subj_col])
        mres = md.fit(method="lbfgs", reml=False)

        # Wald tests for main effects and interaction
        # Build contrast matrices for each term
        terms = {
            "C(Time)": [name for name in mres.params.index if name.startswith("C(Time)[T.")],
            "C(Group)": [name for name in mres.params.index if name.startswith("C(Group)[T.")],
            "C(Time):C(Group)": [name for name in mres.params.index if "C(Time)[T." in name and "C(Group)[T." in name],
        }

        rows = []
        for eff, coeffs in terms.items():
            if not coeffs:
                continue
            L = np.zeros((len(coeffs), len(mres.params)))
            for i, cname in enumerate(coeffs):
                j = list(mres.params.index).index(cname)
                L[i, j] = 1.0
            wtest = mres.wald_test(L)
            stat = float(wtest.statistic) if np.ndim(wtest.statistic) == 0 else float(np.squeeze(wtest.statistic))
            df_w = int(L.shape[0])
            pval = float(wtest.pvalue)
            rows.append(dict(Effect=eff, W=stat, df=df_w, p=pval))
        fallback_tbl = pd.DataFrame(rows).set_index("Effect")
        info["fallback"] = fallback_tbl
        return None, info
    except Exception as e:
        info["reason"] = (info["reason"] or "") + f" | MixedLM fallback failed: {type(e).__name__}: {e}"
        return None, info

def oneway_anova_baseline(pre_totals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-way ANOVA across the 6 groups on PRE totals.
    Returns ANOVA table + descriptives (mean, SD, SEM, 95% CI).
    """
    # OLS one-way ANOVA
    model = smf.ols("Distance ~ C(Group)", data=pre_totals).fit()
    aov = anova_lm(model, typ=2)

    # Descriptives with 95% CI
    desc = (
        pre_totals.groupby("Group")["Distance"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    desc["sem"] = desc["sd"] / np.sqrt(desc["n"])
    crit = desc["n"].apply(lambda k: stats.t.ppf(0.975, max(k - 1, 1)))
    desc["ci95"] = crit * desc["sem"]

    return aov, desc

def twoway_anova_totals(totals: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    2×3 between-subjects ANOVA on summed totals; Tukey on 6 groups.
    """
    model = smf.ols("Distance ~ C(Prenatal)*C(Drug)", data=totals).fit()
    aov = anova_lm(model, typ=2)

    tukey_res = pairwise_tukeyhsd(endog=totals["Distance"], groups=totals["Group"], alpha=0.05)
    # Convert Tukey summary to DataFrame
    tukey_df = pd.DataFrame(
        tukey_res.summary().data[1:], columns=tukey_res.summary().data[0]
    )
    # Normalize column names
    tukey_df.columns = [c.lower().replace(" ", "_") for c in tukey_df.columns]
    tukey_df["meandiff"] = pd.to_numeric(tukey_df["meandiff"], errors="coerce")
    tukey_df["p-adj"] = pd.to_numeric(tukey_df["p-adj"], errors="coerce")
    tukey_df["lower"] = pd.to_numeric(tukey_df["lower"], errors="coerce")
    tukey_df["upper"] = pd.to_numeric(tukey_df["upper"], errors="coerce")
    tukey_df["reject"] = tukey_df["reject"].astype(bool)
    tukey_df = tukey_df.rename(columns={"group1": "group1", "group2": "group2"})
    return aov, tukey_df


def per_bin_anovas(long: pd.DataFrame) -> pd.DataFrame:
    """
    For each time bin, run 2×3 ANOVA: Distance ~ C(Prenatal)*C(Drug).
    Returns tidy table with F, df, p for each effect per time.
    """
    rows = []
    for t, dsub in long.groupby("TimeMin"):
        model = smf.ols("Distance ~ C(Prenatal)*C(Drug)", data=dsub).fit()
        aov = anova_lm(model, typ=2)
        for effect in aov.index:
            rows.append(
                dict(
                    TimeMin=t,
                    Effect=effect,
                    df=float(aov.loc[effect, "df"]),
                    F=float(aov.loc[effect, "F"]),
                    p=float(aov.loc[effect, "PR(>F)"]),
                )
            )
    return pd.DataFrame(rows)

def plot_baseline_bar(pre_totals: pd.DataFrame, desc: pd.DataFrame, group_order: Sequence[str]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of PRE totals (baseline), mean ± 95% CI for each group.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=200)

    means = desc.set_index("Group").loc[group_order, "mean"].values
    ci95s = desc.set_index("Group").loc[group_order, "ci95"].values
    sems  = desc.set_index("Group").loc[group_order, "sem"].values

    x = np.arange(len(group_order))
    ax.bar(
        x,
        means,
        yerr=ci95s,
        capsize=3,
        color="0.85",
        edgecolor="black",
        linewidth=1.0,
        width=0.65,
    )

    ax.set_xticks(x, group_order, rotation=25, ha="right")
    ax.set_ylabel("Baseline total movement (n)")
    style_axes(ax)

    return fig, ax

def totals_diagnostics(totals: pd.DataFrame) -> Dict[str, object]:
    """
    Residual QQ and Levene test across 6 groups.
    """
    model = smf.ols("Distance ~ C(Prenatal)*C(Drug)", data=totals).fit()
    resid = model.resid

    # Levene across 6 groups
    groups = [g["Distance"].values for _, g in totals.groupby("Group")]
    W, p_lev = stats.levene(*groups, center="median")
    return {"residuals": resid, "levene_W": W, "levene_p": p_lev}

# ------------------------------- UI LAYOUT -----------------------------------

st.title("Locomotion Figure Lab")
st.caption("Upload your CSV, map columns, and generate publication-style figures + full stats.")

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("CSV file", type=["csv"], help="Your table with Group, Rat, and time bins.")
    if up:
        df = load_csv(up)
    else:
        st.info("No file uploaded yet. Paste or upload when ready.")
        df = pd.DataFrame()

    if df.empty:
        st.stop()

    st.header("2) Column mapping")
    cols = list(df.columns)
    group_col = st.selectbox("Group column", cols, index=cols.index("Group") if "Group" in cols else 0)
    rat_col = st.selectbox("Rat ID column", cols, index=cols.index("Rat") if "Rat" in cols else 1)
    candidate_bins = [c for c in cols if c not in (group_col, rat_col)]
    default_bins = candidate_bins
    bin_cols = st.multiselect("Time-bin columns (ordered)", candidate_bins, default=default_bins)
    time_start = st.number_input("Start time (min) of first selected bin", value=10, step=5)
    bin_width = st.number_input("Bin width (min)", value=10, step=5)


    # --- 3) Windows (pre/post) ----------------------------------------------
    # derive minute labels from the selected bin columns
    _labels = infer_time_labels(bin_cols, time_start, bin_width) if bin_cols else []
    if _labels:
        tmin, tmax = min(_labels), max(_labels)
        # sensible defaults: split the series in half
        # mid_idx = len(_labels) // 2
        # pre_default  = (_labels[0], _labels[mid_idx-1] if mid_idx > 0 else _labels[0])
        # post_default = (_labels[mid_idx], _labels[-1])

        pre_default  = (_labels[0], 30)
        post_default = (40, _labels[-1])

        st.header("3) Windows")
        pre_range = st.slider(
            "Pre-amphetamine window (min)",
            min_value=tmin, max_value=tmax,
            value=pre_default, step=bin_width,
            help="Choose the time range to sum for the PRE totals."
        )
        post_range = st.slider(
            "Post-amphetamine window (min)",
            min_value=tmin, max_value=tmax,
            value=post_default, step=bin_width,
            help="Choose the time range to sum for the POST totals."
        )
    else:
        pre_range = None
        post_range = None

    cmap = ColumnMap(group=group_col, rat=rat_col, bins=bin_cols, time_start=time_start, bin_width=bin_width)



# Safety checks
if len(cmap.bins) == 0:
    st.error("Select at least one time-bin column in the sidebar.")
    st.stop()

# Transform
long = to_long(df, cmap)
summary = summarize_timecourse(long)
totals = totals_by_rat(long)

# Order groups consistently (alphabetical is fine; you can drag below)
default_order = sorted(totals["Group"].unique().tolist())
with st.expander("Group order for plots", expanded=True):
    st.write("Drag to reorder:")
    group_order = st.sortable(default_order, key="grp_order") if hasattr(st, "sortable") else default_order  # Streamlit may not have drag widget; fallback.
    st.write("Order used:", group_order)


# --- Default totals bar figure for the Downloads tab (uses current analysis window) ---
_aov_tmp, _tukey_tmp = twoway_anova_totals(totals)
fig_bars, _ax_bars = plot_totals_bar(totals, group_order, _tukey_tmp,None)  # <- defines fig_bars


# Tabs for outputs
tab_tc, tab_tot, tab_bin, tab_dl = st.tabs(
    ["Time course", "Totals (ANOVA + Tukey)", "Per-bin ANOVAs", "Downloads"]
)

# ------------------------------- TIME COURSE ---------------------------------

with tab_tc:
    col_plot, col_stats = st.columns([2, 1.2], gap="large")

    with col_plot:
        st.subheader("Mean ± SEM over time (choose a style)")
        style_options = [
            "A) Ribbon Classic",
            "B) Errorbar Classic",
            "B-tiny) Errorbar Classic (tiny symbols)",   # <<< new
            "C) Marker-Only",
            "D) Linestyle Suite",
            "E) Direct-Labeled",
        ]
        style_choice = st.selectbox("Style", style_options, index=2)  # default to the new tiny style if you like

        # robust mapping instead of relying on split(')')
        choice_to_variant = {
            "A) Ribbon Classic": "A",
            "B) Errorbar Classic": "B",
            "B-tiny) Errorbar Classic (tiny symbols)": "B_TINY",
            "C) Marker-Only": "C",
            "D) Linestyle Suite": "D",
            "E) Direct-Labeled": "E",
        }
        variant = choice_to_variant[style_choice]

        fig_tc, ax_tc = plot_timecourse5(summary, group_order, variant=variant)
        st.pyplot(fig_tc, use_container_width=True)

    with col_stats:
        st.subheader("Mixed ANOVA (within = Time, between = Group)")
        aov_tc, info = mixed_anova_timecourse(long)

        # Diagnostics panel
        with st.expander("Design diagnostics", expanded=False):
            st.write(f"Subjects: {info.get('n_subjects')}, Groups: {info.get('n_groups')}")
            st.write(f"Time levels used: {info.get('times')}")
            cov_tab = info.get("coverage_table")
            if cov_tab is not None:
                st.markdown("**Subject × Time coverage (counts per cell):**")
                st.dataframe(cov_tab, use_container_width=True)

        if aov_tc is not None:
            st.success("Mixed ANOVA succeeded (Pingouin).")
            st.dataframe(aov_tc.round(4), use_container_width=True)
            # Download
            st.download_button("Download ANOVA (CSV)",
                            aov_tc.to_csv().encode(),
                            "timecourse_mixed_anova.csv", "text/csv")
        else:
            reason = info.get("reason", "Unknown error")
            st.warning(f"Mixed ANOVA (Pingouin) unavailable: {reason}")

            fb = info.get("fallback")
            if fb is not None and not fb.empty:
                st.info("Showing fallback: Linear Mixed Model (random intercept per subject), Wald tests.")
                st.dataframe(fb.round(4), use_container_width=True)
                st.download_button("Download MixedLM Wald tests (CSV)",
                                fb.to_csv().encode(),
                                "timecourse_mixedlm_wald.csv", "text/csv")
            else:
                st.error("No fallback results available. Check coverage and time bins.")

# ------------------------------- TOTALS --------------------------------------

with tab_tot:
    st.subheader("Totals (PRE / POST / Δ)")

    # Guard if windows aren’t set yet
    if pre_range is None or post_range is None:
        st.warning("Define PRE/POST windows in the sidebar to continue.")
        st.stop()

    # Compute per-rat totals
    pre_totals  = totals_by_window(long, pre_range[0],  pre_range[1])
    post_totals = totals_by_window(long, post_range[0], post_range[1])
    delta_tot   = totals_delta(pre_totals, post_totals)

    # Tabs
    t_pre, t_post, t_delta = st.tabs([
        f"PRE  [{pre_range[0]}–{pre_range[1]} min]",
        f"POST [{post_range[0]}–{post_range[1]} min]",
        "Δ  (POST − PRE)"
    ])

    # ============================== PRE TAB ==============================
    with t_pre:
        st.caption("Summed distance per rat over the PRE window.")
        # Show raw per-rat totals table for transparency
        st.markdown("**Per-rat totals (PRE)**")
        st.dataframe(pre_totals.sort_values(["Group","Rat"]).reset_index(drop=True),
                     use_container_width=True)
        st.download_button("Download PRE per-rat totals (CSV)",
                           pre_totals.to_csv(index=False).encode(),
                           "pre_per_rat_totals.csv", "text/csv")

        # Full analytics package
        pkg = make_totals_package(pre_totals, group_order)
        st.session_state["totals_fig_pre"] = pkg["fig"]

        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg["aov"].round(4), use_container_width=True)

        st.markdown("**Tukey HSD (6 groups) + CLD letters**")
        st.dataframe(pkg["tukey"], use_container_width=True)

        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg["desc"].copy()
        if pkg["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)

        st.markdown("**Diagnostics**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Levene’s test (variance homogeneity)")
            W, p_lev = pkg["levene"]
            st.write(f"W = {W:.3f}, p = {p_lev:.4f}")
            st.caption("Group variances / SDs")
            st.dataframe(pkg["variances"].round(4), use_container_width=True)
        with col2:
            st.caption("Shapiro–Wilk per group")
            st.dataframe(pkg["normality"].round(4), use_container_width=True)

        st.subheader("PRE totals (grayscale bars with CLD)")
        st.pyplot(pkg["fig"], use_container_width=True)

        st.subheader("PRE: Insights & conclusions")
        st.markdown(pkg["narrative"])

        st.markdown("**Downloads (PRE)**")
        cA, cB, cC, cD, cE, cF = st.columns(6)
        with cA: st.download_button("ANOVA (CSV)", pkg["aov"].to_csv().encode(), "pre_anova_effectsizes.csv", "text/csv")
        with cB: st.download_button("Tukey (CSV)", pkg["tukey"].to_csv(index=False).encode(), "pre_tukey.csv", "text/csv")
        with cC: st.download_button("Descriptives (CSV)", desc_disp.to_csv(index=False).encode(), "pre_descriptives.csv", "text/csv")
        with cD: st.download_button("Diagnostics (CSV)", pkg["variances"].to_csv(index=False).encode(), "pre_variances.csv", "text/csv")
        with cE: st.download_button("Figure (PNG)", fig_bytes(pkg["fig"]), "pre_totals.png", "image/png")
        with cF: st.download_button("Insights (TXT)", pkg["narrative"].encode("utf-8"), "pre_insights.txt", "text/plain")


        # ================= Baseline (PRE) sanity check =================
        st.markdown("### Baseline (PRE) one-way ANOVA across 6 groups")

        aov_pre, desc_pre = oneway_anova_baseline(pre_totals)
        st.markdown("**ANOVA (PRE)**")
        st.dataframe(aov_pre.round(4), use_container_width=True)

        st.markdown("**Descriptives with 95% CI (PRE)**")
        st.dataframe(desc_pre.round(3), use_container_width=True)

        pval = aov_pre.loc["C(Group)", "PR(>F)"]
        if pval >= 0.05:
            st.success(f"Baseline groups did not differ (p = {pval:.3g}); "
                    "post-amphetamine ANOVA can be interpreted without correction.")
        else:
            st.warning(f"Baseline groups differ (p = {pval:.3g}); "
                    "consider baseline correction (ANCOVA or Δ analysis).")
        # Visualization of PRE totals with 95% CI
        st.subheader("Baseline (PRE) totals: mean ± 95% CI")
        fig_pre, ax_pre = plot_baseline_bar(pre_totals, desc_pre, group_order)
        st.pyplot(fig_pre, use_container_width=True)
        st.markdown("**Downloads (Baseline PRE)**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "ANOVA (CSV)",
                aov_pre.to_csv().encode(),
                "baseline_anova.csv",
                "text/csv",
            )
        with c2:
            st.download_button(
                "Descriptives (CSV)",
                desc_pre.to_csv(index=False).encode(),
                "baseline_descriptives.csv",
                "text/csv",
            )
        with c3:
            st.download_button(
                "Figure (PNG)",
                fig_bytes(fig_pre),
                "baseline_totals.png",
                "image/png",
            )




    # ============================== POST TAB =============================
    with t_post:
        st.caption("Summed distance per rat over the POST window.")
        st.markdown("**Per-rat totals (POST)**")
        st.dataframe(post_totals.sort_values(["Group","Rat"]).reset_index(drop=True),
                     use_container_width=True)
        st.download_button("Download POST per-rat totals (CSV)",
                           post_totals.to_csv(index=False).encode(),
                           "post_per_rat_totals.csv", "text/csv")

        pkg = make_totals_package(post_totals, group_order)
        st.session_state["totals_fig_post"] = pkg["fig"]

        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg["aov"].round(4), use_container_width=True)

        st.markdown("**Tukey HSD (6 groups) + CLD letters**")
        st.dataframe(pkg["tukey"], use_container_width=True)

        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg["desc"].copy()
        if pkg["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)

        st.markdown("**Diagnostics**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Levene’s test (variance homogeneity)")
            W, p_lev = pkg["levene"]
            st.write(f"W = {W:.3f}, p = {p_lev:.4f}")
            st.caption("Group variances / SDs")
            st.dataframe(pkg["variances"].round(4), use_container_width=True)
        with col2:
            st.caption("Shapiro–Wilk per group")
            st.dataframe(pkg["normality"].round(4), use_container_width=True)

        st.subheader("POST totals (grayscale bars with CLD)")
        st.pyplot(pkg["fig"], use_container_width=True)

        st.subheader("POST: Insights & conclusions")
        st.markdown(pkg["narrative"])

        st.markdown("**Downloads (POST)**")
        cA, cB, cC, cD, cE, cF = st.columns(6)
        with cA: st.download_button("ANOVA (CSV)", pkg["aov"].to_csv().encode(), "post_anova_effectsizes.csv", "text/csv")
        with cB: st.download_button("Tukey (CSV)", pkg["tukey"].to_csv(index=False).encode(), "post_tukey.csv", "text/csv")
        with cC: st.download_button("Descriptives (CSV)", desc_disp.to_csv(index=False).encode(), "post_descriptives.csv", "text/csv")
        with cD: st.download_button("Diagnostics (CSV)", pkg["variances"].to_csv(index=False).encode(), "post_variances.csv", "text/csv")
        with cE: st.download_button("Figure (PNG)", fig_bytes(pkg["fig"]), "post_totals.png", "image/png")
        with cF: st.download_button("Insights (TXT)", pkg["narrative"].encode("utf-8"), "post_insights.txt", "text/plain")

    # ============================== DELTA TAB ============================
    with t_delta:
        st.caption("Per-rat change: POST − PRE. Positive = increase after amphetamine.")
        st.markdown("**Per-rat Δ (POST − PRE)**")
        st.dataframe(delta_tot.sort_values(["Group","Rat"]).reset_index(drop=True),
                     use_container_width=True)
        st.download_button("Download Δ per-rat values (CSV)",
                           delta_tot.to_csv(index=False).encode(),
                           "delta_per_rat.csv", "text/csv")

        pkg = make_totals_package(delta_tot, group_order)
        st.session_state["totals_fig_delta"] = pkg["fig"]

        st.markdown("**ANOVA (Prenatal × Drug) with effect sizes**")
        st.dataframe(pkg["aov"].round(4), use_container_width=True)

        st.markdown("**Tukey HSD (6 groups) + CLD letters**")
        st.dataframe(pkg["tukey"], use_container_width=True)

        st.markdown("**Descriptives (mean ± SEM, 95% CI)**")
        desc_disp = pkg["desc"].copy()
        if pkg["cld"]:
            desc_disp["CLD"] = desc_disp["Group"].map(pkg["cld"])
        st.dataframe(desc_disp.round(3), use_container_width=True)

        st.markdown("**Diagnostics**")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Levene’s test (variance homogeneity)")
            W, p_lev = pkg["levene"]
            st.write(f"W = {W:.3f}, p = {p_lev:.4f}")
            st.caption("Group variances / SDs")
            st.dataframe(pkg["variances"].round(4), use_container_width=True)
        with col2:
            st.caption("Shapiro–Wilk per group")
            st.dataframe(pkg["normality"].round(4), use_container_width=True)

        st.subheader("Δ totals (grayscale bars with CLD)")
        st.pyplot(pkg["fig"], use_container_width=True)

        st.subheader("Δ: Insights & conclusions")
        st.markdown(
            "Interpretation note: Δ reflects **change after amphetamine** relative to the chosen PRE window. "
            "If baseline differs across groups, Δ helps isolate treatment-related shifts."
        )
        st.markdown(pkg["narrative"])
        # ----------------- NEW: OUTLIER EXPLORATION TAB -----------------
        with st.expander("Prenatal Effect Explorer (Δ)", expanded=False):
            st.markdown("This tool checks under what outlier-removal scenarios the Prenatal effect in Δ becomes significant.")

            # Sort rats within each group by Δ
            results = []
            max_remove = 3  # test removing up to 3 rats per group

            for n in range(max_remove + 1):
                df = delta_tot.copy()
                pruned = []
                for g, sub in df.groupby("Group"):
                    sub_sorted = sub.sort_values("Distance")
                    if n == 0:
                        # keep full group when no removals
                        pruned.append(sub_sorted)
                    elif len(sub_sorted) > 2*n:
                        # remove n lowest and n highest Δ values
                        pruned.append(sub_sorted.iloc[n:-n])
                    else:
                        # if too few animals, just keep as-is
                        pruned.append(sub_sorted)

                df_pruned = pd.concat(pruned)

                # Re-run ANOVA
                pkg = make_totals_package(df_pruned, group_order)
                pval = pkg["aov"].loc["C(Prenatal)", "PR(>F)"]

                results.append({"Removed per group": n, "Prenatal p-value": pval})

            res_df = pd.DataFrame(results)

            # Display results
            st.dataframe(res_df)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(res_df["Removed per group"], res_df["Prenatal p-value"], marker="o")
            ax.axhline(0.05, color="red", linestyle="--")
            ax.set_xlabel("Rats removed per group (high & low Δ)")
            ax.set_ylabel("Prenatal p-value")
            st.pyplot(fig)

        with st.expander("Outlier Exploration (Δ)", expanded=False):
            st.markdown("Use this tool to flag and remove rats with extreme Δ values, then recompute ANOVA/Tukey.")

            # Show boxplot-style thresholds (IQR)
            q1 = delta_tot["Distance"].quantile(0.25)
            q3 = delta_tot["Distance"].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr

            st.write(f"Default IQR range: [{lower:.1f}, {upper:.1f}]")

            # Let user pick rule
            method = st.radio("Outlier rule", ["IQR (±1.5)", "Z > 3"], horizontal=True)

            if method == "IQR (±1.5)":
                mask = (delta_tot["Distance"] < lower) | (delta_tot["Distance"] > upper)
            else:
                zscores = (delta_tot["Distance"] - delta_tot["Distance"].mean()) / delta_tot["Distance"].std(ddof=1)
                mask = zscores.abs() > 3

            flagged = delta_tot.loc[mask]
            st.write("### Flagged outliers", flagged)

            # Let user select which to exclude
            exclude_ids = st.multiselect(
                "Select rats to exclude",
                options=[f"{r.Group}_{r.Rat}" for r in flagged.itertuples()],
            )

            # Build filtered dataset
            if exclude_ids:
                filt = delta_tot[~delta_tot.apply(lambda r: f"{r.Group}_{r.Rat}" in exclude_ids, axis=1)]
            else:
                filt = delta_tot

            # Rerun stats
            pkg_filt = make_totals_package(filt, group_order)

            st.subheader("Recomputed ANOVA/Tukey (after exclusions)")
            st.dataframe(pkg_filt["aov"].round(4), use_container_width=True)
            st.dataframe(pkg_filt["tukey"], use_container_width=True)
            st.pyplot(pkg_filt["fig"], use_container_width=True)
            st.markdown(pkg_filt["narrative"])

        st.markdown("**Downloads (Δ)**")
        cA, cB, cC, cD, cE, cF = st.columns(6)
        with cA: st.download_button("ANOVA (CSV)", pkg["aov"].to_csv().encode(), "delta_anova_effectsizes.csv", "text/csv")
        with cB: st.download_button("Tukey (CSV)", pkg["tukey"].to_csv(index=False).encode(), "delta_tukey.csv", "text/csv")
        with cC: st.download_button("Descriptives (CSV)", desc_disp.to_csv(index=False).encode(), "delta_descriptives.csv", "text/csv")
        with cD: st.download_button("Diagnostics (CSV)", pkg["variances"].to_csv(index=False).encode(), "delta_variances.csv", "text/csv")
        with cE: st.download_button("Figure (PNG)", fig_bytes(pkg["fig"]), "delta_totals.png", "image/png")
        with cF: st.download_button("Insights (TXT)", pkg["narrative"].encode("utf-8"), "delta_insights.txt", "text/plain")

# ------------------------------- PER-BIN ANOVAS ------------------------------

with tab_bin:
    st.subheader("Two-way ANOVAs per time bin (Prenatal × Drug)")
    aov_bins = per_bin_anovas(long)
    # Pretty print: pivot by Effect
    show = aov_bins.pivot_table(index=["TimeMin"], columns="Effect", values=["F", "p"], aggfunc="first")
    st.dataframe(show.round(4), use_container_width=True)

# ------------------------------- DOWNLOADS -----------------------------------

with tab_dl:
    st.subheader("Export everything")

    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.caption("Long data (tidy)")
        st.download_button("Download CSV", long.to_csv(index=False).encode(), "long_data.csv", "text/csv")
    with colB:
        st.caption("Time-course summary")
        st.download_button("Download CSV", summary.to_csv(index=False).encode(), "timecourse_summary.csv", "text/csv")
    with colC:
        st.caption("Totals (per rat)")
        st.download_button("Download CSV", totals.to_csv(index=False).encode(), "totals.csv", "text/csv")
    with colD:
        st.caption("Per-bin ANOVA table")
        st.download_button("Download CSV", aov_bins.to_csv(index=False).encode(), "per_bin_anovas.csv", "text/csv")

    colF, colG = st.columns(2)
    with colF:
        st.caption("Time-course figure (PNG)")
        st.download_button("Download PNG", fig_bytes(fig_tc), "figure_timecourse.png", "image/png")
    with colG:
        st.caption("Totals bar figures (PNG)")
        # Fallback: build a generic fig if session figs aren’t set yet
        if "totals_fig_pre" not in st.session_state or st.session_state["totals_fig_pre"] is None:
            _aov_tmp, _tukey_tmp = twoway_anova_totals(totals)
            _fallback_fig, _ = plot_totals_bar(totals, group_order, _tukey_tmp)
        else:
            _fallback_fig = st.session_state["totals_fig_post"] or st.session_state["totals_fig_pre"]

        c1, c2, c3 = st.columns(3)
        with c1:
            fig_pre = st.session_state.get("totals_fig_pre", _fallback_fig)
            st.download_button("Download PRE", fig_bytes(fig_pre), "pre_totals.png", "image/png")
        with c2:
            fig_post = st.session_state.get("totals_fig_post", _fallback_fig)
            st.download_button("Download POST", fig_bytes(fig_post), "post_totals.png", "image/png")
        with c3:
            fig_delta = st.session_state.get("totals_fig_delta", _fallback_fig)
            st.download_button("Download Δ", fig_bytes(fig_delta), "delta_totals.png", "image/png")

st.markdown("---")
st.caption(
    "Tip: Change the selected bin columns or the start/width in the sidebar to redefine the analysis window "
    "(e.g., if you later add baseline bins or want to restrict to 40–120 min)."
)
