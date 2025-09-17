import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from io import BytesIO

# Helper to save figures
def fig_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()

st.set_page_config(layout="wide")
st.title("📊 AIH Locomotion Analysis")

# --- File upload ---
st.sidebar.header("Upload Data")
file = st.sidebar.file_uploader("Upload long-format CSV (Subject, Group, Time, Distance)", type="csv")

if file:
    long = pd.read_csv(file)
    st.success(f"Loaded dataset: {long.shape[0]} rows")

    # Ensure factors
    long["Group"] = long["Group"].astype("category")
    long["Time"] = long["Time"].astype("category")
    st.write("Preview:", long.head())

    # ==============================================================
    # 1. Pre-injection one-way ANOVA (baseline differences)
    # ==============================================================
    st.header("۱-۳-۲- مقایسه بین‌گروهی در خط پایه (One-way ANOVA)")
    baseline = long[long["Time"].isin([1,2,3])].groupby(["Subject","Group"])["Distance"].mean().reset_index()
    aov_pre = pg.anova(dv="Distance", between="Group", data=baseline, detailed=True)
    st.dataframe(aov_pre.round(4))
    st.download_button("⬇️ دانلود ANOVA خط پایه", aov_pre.to_csv(index=False).encode(), "anova_baseline.csv","text/csv")

    fig_pre, ax = plt.subplots()
    sns.boxplot(data=baseline, x="Group", y="Distance", ax=ax)
    sns.swarmplot(data=baseline, x="Group", y="Distance", color="black", size=3, ax=ax)
    st.pyplot(fig_pre)
    st.download_button("⬇️ دانلود نمودار خط پایه", fig_bytes(fig_pre), "baseline_boxplot.png","image/png")

    # ==============================================================
    # 2. Post-injection two-way repeated-measures ANOVA
    # ==============================================================
    st.header("۲-۳-۲- آنالیز واریانس دوطرفه با اندازه‌های مکرر (Time × Group)")
    aov = pg.mixed_anova(dv="Distance", within="Time", subject="Subject", between="Group", data=long, detailed=True)
    st.dataframe(aov.round(4))
    st.download_button("⬇️ دانلود ANOVA دوطرفه", aov.to_csv(index=False).encode(), "anova_post.csv","text/csv")

    # Plot time course
    st.subheader("📈 نمودار مسیر زمانی")
    fig_tc, ax_tc = plt.subplots()
    sns.lineplot(data=long, x="Time", y="Distance", hue="Group", estimator="mean", ci="se", ax=ax_tc)
    st.pyplot(fig_tc)
    st.download_button("⬇️ دانلود نمودار مسیر زمانی", fig_bytes(fig_tc),"timecourse.png","image/png")

    # ==============================================================
    # 3. Mauchly’s test + epsilon
    # ==============================================================
    st.header("۳-۳-۲- آزمون کرویت موچلی و ضرایب اپسیلون")
    wide = long.pivot(index="Subject", columns="Time", values="Distance")
    try:
        sph = pg.sphericity(wide, method="mauchly")
        st.write("Mauchly:", sph)
    except Exception as e:
        st.warning(f"Sphericity test failed: {e}")

    try:
        eps = pg.epsilon(wide)
        st.dataframe(eps.round(4))
    except Exception as e:
        st.warning(f"Epsilon computation failed: {e}")

    # ==============================================================
    # 4. Post-hoc tests
    # ==============================================================
    st.header("۴-۳-۲- تست‌های تعقیبی")
    pw_time = pg.pairwise_ttests(dv="Distance", within="Time", subject="Subject", data=long, padjust="holm")
    st.subheader("زمان‌ها")
    st.dataframe(pw_time.round(4))
    pw_group = pg.pairwise_ttests(dv="Distance", between="Group", data=long, padjust="holm")
    st.subheader("گروه‌ها")
    st.dataframe(pw_group.round(4))

    # ==============================================================
    # 5. Baseline-corrected analysis
    # ==============================================================
    st.header("۵-۳-۲- تحلیل اصلاح‌شده خط پایه")
    baseline_means = long[long["Time"].isin([1,2,3])].groupby("Subject")["Distance"].mean()
    long_corr = long.copy()
    long_corr["Delta"] = long_corr["Distance"] - long_corr["Subject"].map(baseline_means)
    aov_corr = pg.mixed_anova(dv="Delta", within="Time", subject="Subject", between="Group", data=long_corr, detailed=True)
    st.dataframe(aov_corr.round(4))

    # ==============================================================
    # 6. Linear Mixed Model (LMM)
    # ==============================================================
    st.header("۶-۳-۲- مدل خطی مختلط (LMM)")
    md = smf.mixedlm("Distance ~ Group*Time", long, groups=long["Subject"])
    mdf = md.fit()
    st.text(mdf.summary())

    # ==============================================================
    # 7. Factor effect (MIA vs NS)
    # ==============================================================
    st.header("۷-۳-۲- اثر عامل مدل (MIA vs NS)")
    if "Model" in long.columns:
        aov_model = pg.anova(dv="Distance", between="Model", data=long, detailed=True)
        st.dataframe(aov_model.round(4))

    # ==============================================================
    # 8. High responders
    # ==============================================================
    st.header("۸-۳-۲- High Responders (ΔAUC)")
    auc = long_corr.groupby("Subject")["Delta"].sum()
    thr = auc.quantile(0.75)
    high = (auc > thr).astype(int)
    long_high = pd.DataFrame({"Subject":auc.index,"AUC":auc.values,"HighResponder":high.values})
    summary_high = long_high.groupby(long.set_index("Subject")["Group"])["HighResponder"].mean()*100
    st.write(summary_high)

    # ==============================================================
    # 9. Variance comparison (Levene)
    # ==============================================================
    st.header("۹-۳-۲- مقایسه واریانس (Levene)")
    groups = [auc[long.set_index("Subject")["Group"]==g].values for g in long["Group"].unique()]
    lev = pg.homoscedasticity(long_corr, dv="Delta", group="Group", method="levene")
    st.dataframe(lev.round(4))

    # ==============================================================
    # 10. Clustering (K-means)
    # ==============================================================
    st.header("۱۰-۳-۲- خوشه‌بندی K-means")
    mat = long_corr.pivot(index="Subject", columns="Time", values="Delta").fillna(0)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(mat)
    long_corr["Cluster"] = long_corr["Subject"].map(dict(zip(mat.index,kmeans.labels_)))
    st.write("Cluster counts:", pd.Series(kmeans.labels_).value_counts())

    fig_clust, axc = plt.subplots()
    for c in np.unique(kmeans.labels_):
        axc.plot(mat.columns, mat[kmeans.labels_==c].mean(), label=f"Cluster {c}")
    axc.legend()
    st.pyplot(fig_clust)
