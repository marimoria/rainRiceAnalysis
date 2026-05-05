"""
02_padi_analysis.py
===================
PADI (Produktivitas) — Analisis Deskriptif Komprehensif
-------------------------------------------------
PRIMARY outputs (for journal):
  1.  Descriptive statistics table         → padi_1_descriptive_stats.csv
  2.  Frequency distribution histogram     → padi_2_histogram.png
  3.  Monthly median line trend (NEW)      → padi_3_monthly_median_lineplot.png
  4.  IQR outlier identification table     → padi_4_iqr_outliers.csv
  5.  CV per wilayah table + bar chart     → padi_5_cv_per_wilayah.csv / .png
  6.  Boxplot per wilayah                  → padi_6_boxplot_per_wilayah.png

BACKUP outputs (additional charts):
  B1. Histogram + KDE overlay             → padi_B1_histogram_kde.png
  B2. Frequency distribution (binned)     → padi_B2_freq_distribution.png
  B3. Line plot — all 38 wilayah          → padi_B3_lineplot_all_wilayah.png
  B4. Boxplot per bulan                   → padi_B4_boxplot_per_bulan.png
  B5. Boxplot keseluruhan                 → padi_B5_boxplot_overall.png

All outputs → __output__/padi/
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
padi_nz = df_nz[COL_PADI].astype(float).to_numpy()   

print("=" * 65)
print("  02_PADI_ANALYSIS.PY (ROBUST FOCUS)")
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
    "Median (Nilai Tengah)"     : median_p,
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
ax.axvline(median_p, color="#0984e3", linewidth=2.5,
           linestyle="-.", label=f"Median: {median_p:.2f}")
ax.axvline(mean_p,   color="#d63031", linewidth=1.5,
           linestyle="--", label=f"Mean: {mean_p:.2f}")

ax.set_title(
    "Analisis Distribusi Frekuensi Produktivitas Padi\n(Observasi > 0 ton/ha)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Produktivitas (ton/ha)", fontsize=11)
ax.set_ylabel("Frekuensi Observasi", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_2_histogram.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_2_histogram.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MONTHLY MEDIAN LINE PLOT (NEW ROBUST TREND)
# ═══════════════════════════════════════════════════════════════════════════════
monthly_median_p = df_nz.groupby(COL_BULAN)[COL_PADI].median().reindex(range(1, 13))
monthly_q1_p     = df_nz.groupby(COL_BULAN)[COL_PADI].quantile(0.25).reindex(range(1, 13))
monthly_q3_p     = df_nz.groupby(COL_BULAN)[COL_PADI].quantile(0.75).reindex(range(1, 13))

md_arr_p = monthly_median_p.to_numpy(dtype=float)
q1_arr_p = monthly_q1_p.to_numpy(dtype=float)
q3_arr_p = monthly_q3_p.to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(11, 5))
x = list(range(1, 13))

ax.fill_between(x, q1_arr_p, q3_arr_p, alpha=0.15, color=ACCENT, label="Rentang Interkuartil (Q1 - Q3)")
ax.plot(x, md_arr_p, marker="s", color=ACCENT, linewidth=2.5, markersize=8, zorder=5, label="Median Produktivitas Bulanan")

for xi, yi in zip(x, md_arr_p):
    ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8.5, color="#2d3436")

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_ylim(max(0, min(q1_arr_p) - 1), max(q3_arr_p) * 1.1)

ax.set_title(
    "Dinamika Tren Nilai Tengah (Median) Produktivitas Padi Bulanan\nProvinsi Jawa Timur",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Produktivitas (ton/ha)", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_3_monthly_median_lineplot.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_3_monthly_median_lineplot.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IQR OUTLIER IDENTIFICATION
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

print("── 4. IQR Outlier Identification (Produktivitas) ──")
print(f"   Q1          : {q1:.4f}")
print(f"   Q3          : {q3:.4f}")
print(f"   IQR         : {iqr:.4f}")
print(f"   Lower fence : {lower_fence:.4f}  (Q1 − 1.5×IQR)")
print(f"   Upper fence : {upper_fence:.4f}  (Q3 + 1.5×IQR)")
print(f"   Outliers    : {len(outliers)} observations")
print()

outliers.to_csv(f"{OUT_DIR}/padi_4_iqr_outliers.csv", index=False)
print(f"   → Saved: {OUT_DIR}/padi_4_iqr_outliers.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CV PER WILAYAH (resilience / stability indicator)
# ═══════════════════════════════════════════════════════════════════════════════
cv_per_wilayah = (
    df_nz.groupby("wilayah")[COL_PADI]
    .agg(
        n_obs="count",
        mean_prod="mean",
        std_prod="std", # Clean string alias for standard deviation
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

print("── 5. CV per Wilayah ──")
print(f"   Median CV (threshold) : {median_cv:.2f}%")
print()

cv_per_wilayah.to_csv(f"{OUT_DIR}/padi_5_cv_per_wilayah.csv", index=False)
print(f"   → Saved: {OUT_DIR}/padi_5_cv_per_wilayah.csv")

# ── CV bar chart ──────────────────────────────────────────────────────────────
colors_cv = [
    "#d63031" if s == "Tidak Stabil" else ACCENT
    for s in cv_per_wilayah["Stabilitas"]
]
cv_vals = cv_per_wilayah["CV (%)"].astype(float).to_numpy()

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(cv_per_wilayah["wilayah"].tolist(), cv_vals,
               color=colors_cv, edgecolor="white", linewidth=0.5, height=0.7)
ax.axvline(median_cv, color="#2d3436", linewidth=1.8,
           linestyle="--", label=f"Median CV: {median_cv:.1f}%")

for bar, val in zip(bars, cv_vals):
    x_pos = float(val) + 0.3
    y_pos = float(bar.get_y() + bar.get_height() / 2)
    ax.text(x_pos, y_pos, f"{val:.1f}%", va="center", fontsize=7.5)

legend_patches = [
    mpatches.Patch(color=ACCENT,    label="Stabil (CV ≤ median)"),
    mpatches.Patch(color="#d63031", label="Tidak Stabil (CV > median)"),
]
ax.legend(handles=legend_patches + [ax.lines[0]], fontsize=9, loc="lower right")
ax.set_title(
    "Analisis Koefisien Variasi (CV) Produktivitas Padi per Wilayah\n"
    "(Indikator Stabilitas Produksi Pangan Tingkat Wilayah)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Koefisien Variasi / CV (%)", fontsize=11)
ax.set_ylabel("")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_5_cv_per_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_5_cv_per_wilayah.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BOXPLOT PER WILAYAH (journal primary)
# ═══════════════════════════════════════════════════════════════════════════════
wilayah_order = (
    df_nz.groupby("wilayah")[COL_PADI].median()
    .sort_values(ascending=False).index.tolist()
)
groups_w = [
    df_nz.loc[df_nz["wilayah"] == w, COL_PADI].astype(float).to_numpy() # type: ignore
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
    "Variabilitas dan Distribusi Produktivitas Padi Antar Wilayah\n(Berdasarkan Pemeringkatan Nilai Tengah)",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_xlabel("Produktivitas (ton/ha)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_6_boxplot_per_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_6_boxplot_per_wilayah.png\n")


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
ax.axvline(median_p, color="#0984e3", linewidth=2.5,
           linestyle="-.", label=f"Median: {median_p:.2f}")
ax.axvline(mean_p,   color="#d63031", linewidth=1.5,
           linestyle="--", label=f"Mean: {mean_p:.2f}")

ax.set_title("Analisis Histogram dan KDE Produktivitas Padi",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Produktivitas (ton/ha)", fontsize=11)
ax.set_ylabel("Densitas Frekuensi", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B1_histogram_kde.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B1_histogram_kde.png")

# ── B2. FREQUENCY DISTRIBUTION (equal-width bins) ─────────────────────────────
pmin         = float(np.floor(padi_nz.min() * 2) / 2)
pmax         = float(np.ceil(padi_nz.max() * 2) / 2)
bin_edges_p  = list(np.arange(pmin, pmax + 0.5, 0.5))   
bin_labels_p = [f"{b:.1f}–{b+0.5:.1f}" for b in bin_edges_p[:-1]]
fd_padi = pd.cut(
    df_nz[COL_PADI],
    bins=bin_edges_p,
    labels=bin_labels_p,
    right=False,
).value_counts().reindex(bin_labels_p, fill_value=0)
fd_arr = fd_padi.astype(int).to_numpy()

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(bin_labels_p, fd_arr,
              color=ACCENT, edgecolor="white", linewidth=0.6, width=0.8, alpha=0.88)

for bar, cnt in zip(bars, fd_arr):
    if cnt > 0:
        x_pos = float(bar.get_x() + bar.get_width() / 2)
        y_pos = float(bar.get_height() + 0.3)
        ax.text(x_pos, y_pos, str(cnt), ha="center", va="bottom", fontsize=7.5)

ax.set_title("Distribusi Frekuensi Berdasarkan Interval Produktivitas Padi (0.5 ton/ha)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Kelas Interval Produktivitas (ton/ha)", fontsize=11)
ax.set_ylabel("Frekuensi Observasi", fontsize=11)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B2_freq_distribution.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B2_freq_distribution.png")

# ── B3. LINE PLOT — all 38 wilayah overlaid (MEDIAN AGGREGATE) ─────────────────
pivot_p          = df_nz.pivot_table(index=COL_BULAN, columns="wilayah", values=COL_PADI, aggfunc="median")
overall_median_p = pivot_p.median(axis=1)

fig, ax = plt.subplots(figsize=(14, 7))
cmap = cm.get_cmap("tab20")

for i, col in enumerate(pivot_p.columns):
    ax.plot(pivot_p.index, pivot_p[col].astype(float).to_numpy(),
            linewidth=1.2, alpha=0.6, color=cmap(i % 20), label=col)
    
ax.plot(pivot_p.index, overall_median_p.astype(float).to_numpy(),
        color="black", linewidth=3.5, linestyle="--",
        label="MEDIAN KESELURUHAN WILAYAH", zorder=5)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Perbandingan Tren Produktivitas Padi Bulanan — 38 Wilayah Jawa Timur",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Produktivitas (ton/ha)", fontsize=11)

ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=7)
plt.subplots_adjust(right=0.75) 

plt.savefig(f"{OUT_DIR}/padi_B3_lineplot_all_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B3_lineplot_all_wilayah.png")

# ── B4. BOXPLOT PER BULAN ─────────────────────────────────────────────────────
groups_b = [
    df_nz.loc[df_nz[COL_BULAN] == m, COL_PADI].dropna().astype(float).to_numpy()
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
    "Sebaran dan Variabilitas Produktivitas Padi Berdasarkan Periode Bulanan",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Produktivitas (ton/ha)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B4_boxplot_per_bulan.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B4_boxplot_per_bulan.png")


# ── B5. BOXPLOT KESELURUHAN (Seluruh Dataset) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 8))
bp3 = ax.boxplot(
    [padi_nz],
    patch_artist=True,
    notch=False,
    widths=0.4,
    medianprops=dict(color="#d63031", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=4,
                    alpha=0.5, markerfacecolor="#636e72", markeredgewidth=0),
)

bp3["boxes"][0].set_facecolor(ACCENT)
bp3["boxes"][0].set_alpha(0.85)

ax.set_xticks([1])
ax.set_xticklabels(["Tingkat Provinsi\n(Seluruh Wilayah & Periode)"], fontsize=11)
ax.set_title(
    "Sebaran Keseluruhan Observasi Produktivitas Padi",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_ylabel("Produktivitas (ton/ha)", fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B5_boxplot_overall.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/padi_B5_boxplot_overall.png")

print()
print("=" * 65)
print("  02_padi_analysis.py  — COMPLETE")
print(f"  All outputs in: {OUT_DIR}/")
print("=" * 65)