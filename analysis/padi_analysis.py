"""
02_padi_analysis.py
===================
PADI (Produktivitas) — Full Descriptive Analysis
-------------------------------------------------
PRIMARY outputs (for journal):
  1.  Descriptive statistics table         → padi_1_descriptive_stats.csv
  2.  Frequency distribution histogram     → padi_2_histogram.png
  3.  IQR outlier identification table     → padi_3_iqr_outliers.csv
  4.  CV per wilayah table + bar chart     → padi_4_cv_per_wilayah.csv / .png
  5.  Boxplot per wilayah                  → padi_5_boxplot_per_wilayah.png

BACKUP outputs (additional charts):
  B1. Histogram + KDE overlay             → padi_B1_histogram_kde.png
  B2. Frequency distribution (binned)     → padi_B2_freq_distribution.png
  B3. Line plot — all 38 wilayah         → padi_B3_lineplot_all_wilayah.png
  B4. Boxplot per bulan                   → padi_B4_boxplot_per_bulan.png

All outputs → __output__/padi/
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = "__data__/ch_padi_training_dataset.csv"
OUT_DIR   = "__output__/padi"

COL_NAME  = "nama_wilayah"
COL_JENIS = "jenis_wilayah"
COL_BULAN = "bulan"
COL_PADI  = "produktivitas"          # ton/ha

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

ACCENT  = "#00b894"
ACCENT2 = "#55efc4"

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.30,
    "grid.linestyle"   : "--",
    "figure.dpi"       : 150,
})

os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD & PREPARE ────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)
df_raw["wilayah"] = (
    df_raw[COL_NAME] + " "
    + df_raw[COL_JENIS].map({0: "Kab.", 1: "Kota"})
)

# Keep all rows; zero produktivitas is valid (no harvest that month)
df   = df_raw.dropna(subset=[COL_PADI]).copy()

# For CV/statistics, exclude zero-productivity months (no planting)
df_nz   = df[df[COL_PADI] > 0].copy()
padi_nz = df_nz[COL_PADI].to_numpy(dtype=float)   # numpy — no Pylance complaints

print("=" * 65)
print("  02_PADI_ANALYSIS.PY")
print("=" * 65)
print(f"  Total rows          : {len(df)}")
print(f"  Rows produktivitas>0: {len(df_nz)}  (used for stats & CV)")
print(f"  Unique wilayah      : {df['wilayah'].nunique()}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS (non-zero rows)
# ═══════════════════════════════════════════════════════════════════════════════
mean_p   = float(np.mean(padi_nz))
median_p = float(np.median(padi_nz))
std_p    = float(np.std(padi_nz, ddof=1))
min_p    = float(np.min(padi_nz))
max_p    = float(np.max(padi_nz))
range_p  = max_p - min_p
q1       = float(np.percentile(padi_nz, 25))
q3       = float(np.percentile(padi_nz, 75))
iqr      = q3 - q1
cv_all   = std_p / mean_p * 100

stats_dict = {
    "Mean (Rata-rata)"          : mean_p,
    "Median"                    : median_p,
    "Simpangan Baku (Std Dev)"  : std_p,
    "Minimum"                   : min_p,
    "Maximum"                   : max_p,
    "Rentang (Range)"           : range_p,
    "Q1 (25th Percentile)"      : q1,
    "Q3 (75th Percentile)"      : q3,
    "IQR"                       : iqr,
    "CV (Koefisien Variasi, %)" : cv_all,
}

print("── 1. Descriptive Statistics (Produktivitas Padi) ──")
for k, v in stats_dict.items():
    unit = "%" if "CV" in k else ""
    print(f"   {k:<35}: {v:>10.4f}{unit}")
print()

pd.DataFrame({
    "Statistik": list(stats_dict.keys()),
    "Nilai"    : [f"{v:.4f}" for v in stats_dict.values()],
}).to_csv(f"{OUT_DIR}/padi_1_descriptive_stats.csv", index=False)
print(f"   → Saved: {OUT_DIR}/padi_1_descriptive_stats.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FREQUENCY DISTRIBUTION HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(padi_nz, bins=25, color=ACCENT, edgecolor="white",
        linewidth=0.7, alpha=0.88)
ax.axvline(mean_p,   color="#d63031", linewidth=2,
           linestyle="--", label=f"Mean: {mean_p:.2f}")
ax.axvline(median_p, color="#0984e3", linewidth=2,
           linestyle="-.", label=f"Median: {median_p:.2f}")
ax.set_title(
    "Distribusi Frekuensi Produktivitas Padi\n(observasi produktivitas > 0)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Produktivitas (ton/ha)", fontsize=11)
ax.set_ylabel("Frekuensi", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_2_histogram.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_2_histogram.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. IQR OUTLIER IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

padi_nz_series = df_nz[COL_PADI]
outliers = df_nz[
    (padi_nz_series < lower_fence) | (padi_nz_series > upper_fence)
][["wilayah", COL_BULAN, COL_PADI]].copy()
outliers["bulan_nama"] = outliers[COL_BULAN].apply(
    lambda m: MONTH_LABELS[int(m) - 1]
)
outliers = outliers.sort_values(COL_PADI, ascending=False).reset_index(drop=True)

print("── 3. IQR Outlier Identification (Produktivitas) ──")
print(f"   Q1          : {q1:.4f}")
print(f"   Q3          : {q3:.4f}")
print(f"   IQR         : {iqr:.4f}")
print(f"   Lower fence : {lower_fence:.4f}  (Q1 − 1.5×IQR)")
print(f"   Upper fence : {upper_fence:.4f}  (Q3 + 1.5×IQR)")
print(f"   Outliers    : {len(outliers)} observations")
print()
if not outliers.empty:
    print(outliers[["wilayah", "bulan_nama", COL_PADI]].to_string(index=False))
print()

outliers.to_csv(f"{OUT_DIR}/padi_3_iqr_outliers.csv", index=False)
print(f"   → Saved: {OUT_DIR}/padi_3_iqr_outliers.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CV PER WILAYAH (resilience / stability indicator)
# ═══════════════════════════════════════════════════════════════════════════════
cv_per_wilayah = (
    df_nz.groupby("wilayah")[COL_PADI]
    .agg(
        n_obs="count",
        mean_prod="mean",
        std_prod=lambda x: float(np.std(x.to_numpy(dtype=float), ddof=1)),
    )
    .reset_index()
)
cv_per_wilayah["CV (%)"] = (
    cv_per_wilayah["std_prod"] / cv_per_wilayah["mean_prod"] * 100
).round(2)
cv_per_wilayah = cv_per_wilayah.sort_values("CV (%)", ascending=False).reset_index(drop=True)

# Stability label: at or below median CV = Stabil
median_cv = float(cv_per_wilayah["CV (%)"].median())
cv_per_wilayah["Stabilitas"] = cv_per_wilayah["CV (%)"].apply(
    lambda x: "Stabil" if x <= median_cv else "Tidak Stabil"
)

print("── 4. CV per Wilayah ──")
print(f"   Median CV (threshold) : {median_cv:.2f}%")
print()
print(cv_per_wilayah[
    ["wilayah", "mean_prod", "std_prod", "CV (%)", "Stabilitas"]
].to_string(index=False))
print()

cv_per_wilayah.to_csv(f"{OUT_DIR}/padi_4_cv_per_wilayah.csv", index=False)
print(f"   → Saved: {OUT_DIR}/padi_4_cv_per_wilayah.csv")

# ── CV bar chart ──────────────────────────────────────────────────────────────
colors_cv = [
    "#d63031" if s == "Tidak Stabil" else ACCENT
    for s in cv_per_wilayah["Stabilitas"]
]
cv_vals = cv_per_wilayah["CV (%)"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(cv_per_wilayah["wilayah"].tolist(), cv_vals,
               color=colors_cv, edgecolor="white", linewidth=0.5, height=0.7)
ax.axvline(median_cv, color="#2d3436", linewidth=1.8,
           linestyle="--", label=f"Median CV: {median_cv:.1f}%")
for bar, val in zip(bars, cv_vals):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=7.5)
legend_patches = [
    mpatches.Patch(color=ACCENT,    label="Stabil (CV ≤ median)"),
    mpatches.Patch(color="#d63031", label="Tidak Stabil (CV > median)"),
]
ax.legend(handles=legend_patches + [ax.lines[0]], fontsize=9, loc="lower right")
ax.set_title(
    "Koefisien Variasi (CV) Produktivitas Padi per Wilayah\n"
    "(indikator stabilitas produksi — ketahanan pangan)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("CV (%)", fontsize=11)
ax.set_ylabel("")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_4_cv_per_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_4_cv_per_wilayah.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BOXPLOT PER WILAYAH (journal primary)
# ═══════════════════════════════════════════════════════════════════════════════
wilayah_order = (
    df_nz.groupby("wilayah")[COL_PADI].median()
    .sort_values(ascending=False).index.tolist()
)
groups_w = [
    df_nz.loc[df_nz["wilayah"] == w, COL_PADI].to_numpy(dtype=float)
    for w in wilayah_order
]

fig, ax = plt.subplots(figsize=(7, 14))
bp = ax.boxplot(
    groups_w,
    patch_artist=True,
    vert=False,
    notch=False,
    medianprops=dict(color="#d63031", linewidth=2),
    whiskerprops=dict(linewidth=1.1),
    capprops=dict(linewidth=1.1),
    flierprops=dict(marker="o", markersize=3,
                    alpha=0.45, markerfacecolor="#636e72", markeredgewidth=0),
)
cmap = plt.colormaps["Greens"]
n_w  = len(wilayah_order)
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(0.30 + 0.50 * (i / n_w)))
    patch.set_alpha(0.85)

ax.set_yticks(range(1, n_w + 1))
ax.set_yticklabels(wilayah_order, fontsize=8.5)
ax.set_title(
    "Distribusi Produktivitas Padi per Wilayah\n(diurutkan berdasarkan median)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlabel("Produktivitas (ton/ha)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_5_boxplot_per_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_5_boxplot_per_wilayah.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("── BACKUP CHARTS ──")

# ── B1. HISTOGRAM + KDE ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(padi_nz, bins=25, density=True,
        color=ACCENT, edgecolor="white", linewidth=0.7, alpha=0.75,
        label="Histogram (density)")
kde   = gaussian_kde(padi_nz)
x_kde = np.linspace(padi_nz.min(), padi_nz.max(), 300)
ax.plot(x_kde, kde(x_kde), color="#2d3436", linewidth=2.2, label="KDE")
ax.axvline(mean_p,   color="#d63031", linewidth=1.8,
           linestyle="--", label=f"Mean: {mean_p:.2f}")
ax.axvline(median_p, color="#0984e3", linewidth=1.8,
           linestyle="-.", label=f"Median: {median_p:.2f}")
ax.set_title("Histogram + KDE Produktivitas Padi",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Produktivitas (ton/ha)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B1_histogram_kde.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B1_histogram_kde.png")

# ── B2. FREQUENCY DISTRIBUTION (equal-width bins) ─────────────────────────────
pmin         = float(np.floor(padi_nz.min() * 2) / 2)
pmax         = float(np.ceil(padi_nz.max() * 2) / 2)
bin_edges_p  = list(np.arange(pmin, pmax + 0.5, 0.5))   # list → valid bins arg
bin_labels_p = [f"{b:.1f}–{b+0.5:.1f}" for b in bin_edges_p[:-1]]
fd_padi = pd.cut(
    df_nz[COL_PADI],
    bins=bin_edges_p,
    labels=bin_labels_p,
    right=False,
).value_counts().reindex(bin_labels_p, fill_value=0)
fd_arr = fd_padi.to_numpy(dtype=int)

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(bin_labels_p, fd_arr,
              color=ACCENT, edgecolor="white", linewidth=0.6, width=0.8, alpha=0.88)
for bar, cnt in zip(bars, fd_arr):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(cnt), ha="center", va="bottom", fontsize=7.5)
ax.set_title("Distribusi Frekuensi Produktivitas Padi (Interval 0.5 ton/ha)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Kelas Interval (ton/ha)", fontsize=11)
ax.set_ylabel("Frekuensi", fontsize=11)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B2_freq_distribution.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B2_freq_distribution.png")

# ── B3. LINE PLOT — all 38 wilayah overlaid ───────────────────────────────────
pivot_p        = df.pivot_table(index=COL_BULAN, columns="wilayah",
                                 values=COL_PADI, aggfunc="mean")
overall_mean_p = pivot_p.mean(axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
for col in pivot_p.columns:
    ax.plot(pivot_p.index, pivot_p[col].to_numpy(dtype=float),
            linewidth=0.9, alpha=0.40, color=ACCENT)
ax.plot(pivot_p.index, overall_mean_p.to_numpy(dtype=float),
        color="black", linewidth=2.5, linestyle="--",
        label="Rata-rata Seluruh Wilayah", zorder=5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Produktivitas Padi Bulanan — 38 Wilayah Jawa Timur\n"
    "(garis hitam putus-putus = rata-rata keseluruhan)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Produktivitas (ton/ha)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B3_lineplot_all_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B3_lineplot_all_wilayah.png")

# ── B4. BOXPLOT PER BULAN ─────────────────────────────────────────────────────
groups_b = [
    df_nz.loc[df_nz[COL_BULAN] == m, COL_PADI].dropna().to_numpy(dtype=float)
    for m in range(1, 13)
]

fig, ax = plt.subplots(figsize=(12, 6))
bp2 = ax.boxplot(
    groups_b,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="#d63031", linewidth=2.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3,
                    alpha=0.45, markerfacecolor="#636e72", markeredgewidth=0),
)
cmap2 = plt.colormaps["Greens"]
for i, patch in enumerate(bp2["boxes"]):
    patch.set_facecolor(cmap2(0.30 + 0.45 * (i / 12)))
    patch.set_alpha(0.85)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Boxplot Produktivitas Padi per Bulan — 38 Wilayah Jawa Timur",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Produktivitas (ton/ha)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B4_boxplot_per_bulan.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B4_boxplot_per_bulan.png")

print()
print("=" * 65)
print("  02_padi_analysis.py  — COMPLETE")
print(f"  All outputs in: {OUT_DIR}/")
print("=" * 65)