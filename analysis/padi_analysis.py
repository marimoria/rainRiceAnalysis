"""
padi_analysis.py
===================
PADI (Produktivitas) Full Descriptive Analysis
"""

import os
import warnings
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from plot_style import apply_academic_style

apply_academic_style()
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "__data__/ch_padi_training_dataset.csv"
OUT_DIR   = "__output__/padi"

COL_NAME  = "nama_wilayah"
COL_JENIS = "jenis_wilayah"
COL_BULAN = "bulan"
COL_PADI  = "produktivitas"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

ACCENT = "#00b894"

os.makedirs(OUT_DIR, exist_ok=True)

# LOAD & PREPARE
df_raw = pd.read_csv(DATA_PATH)
df_raw["wilayah"] = (
    df_raw[COL_NAME] + " "
    + df_raw[COL_JENIS].map({0: "Kab.", 1: "Kota"})
)

df    = df_raw.dropna(subset=[COL_PADI]).copy()
df_nz = df[df[COL_PADI] > 0].copy()  # exclude zero-productivity months (no planting)
padi_nz = df_nz[COL_PADI].to_numpy(dtype=float)

print("=" * 65)
print("PADI_ANALYSIS.PY")
print("=" * 65)
print(f"  Total rows          : {len(df)}")
print(f"  Rows produktivitas>0: {len(df_nz)}  (used for stats & CV)")
print(f"  Unique wilayah      : {df['wilayah'].nunique()}\n")


# 1. DESCRIPTIVE STATISTICS
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
    "Mean (Rata-rata)"      : mean_p,
    "Median (Nilai Tengah)" : median_p,
    "Simpangan Baku"        : std_p,
    "Minimum"               : min_p,
    "Maksimum"              : max_p,
    "Rentang"               : range_p,
    "Q1 (Persentil ke-25)"  : q1,
    "Q3 (Persentil ke-75)"  : q3,
    "IQR"                   : iqr,
    "Koefisien Variasi (%)" : cv_all,
}

print("── 1. Statistika Deskriptif ──")
for k, v in stats_dict.items():
    print(f"   {k:<30}: {v:>10.2f}")
print()

pd.DataFrame({
    "Statistik"     : list(stats_dict.keys()),
    "Nilai (ton/ha)": [f"{v:.4f}" for v in stats_dict.values()],
}).to_csv(f"{OUT_DIR}/padi_1_descriptive_stats.csv", index=False)


# 2. FREQUENCY DISTRIBUTION HISTOGRAM
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(padi_nz, bins=25, color=ACCENT, edgecolor="white", alpha=0.88)
ax.axvline(median_p, color="#0984e3", linestyle="-.", linewidth=2.5, label=f"Median: {median_p:.2f} ton/ha")
ax.axvline(mean_p,   color="#d63031", linestyle="--", linewidth=1.5, label=f"Rata-rata: {mean_p:.2f} ton/ha")

ax.set_title("Distribusi Frekuensi Produktivitas Padi di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Produktivitas (ton/ha)")
ax.set_ylabel("Frekuensi Observasi")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_2_histogram.png", bbox_inches="tight")
plt.close()


# 3. MONTHLY MEDIAN LINE PLOT
monthly_median_p = df_nz.groupby(COL_BULAN)[COL_PADI].median().reindex(range(1, 13))
monthly_q1_p     = df_nz.groupby(COL_BULAN)[COL_PADI].quantile(0.25).reindex(range(1, 13))
monthly_q3_p     = df_nz.groupby(COL_BULAN)[COL_PADI].quantile(0.75).reindex(range(1, 13))

md_arr_p = monthly_median_p.to_numpy(dtype=float)
q1_arr_p = monthly_q1_p.to_numpy(dtype=float)
q3_arr_p = monthly_q3_p.to_numpy(dtype=float)

x = list(range(1, 13))
fig, ax = plt.subplots(figsize=(11, 5))

ax.fill_between(x, q1_arr_p, q3_arr_p, alpha=0.15, color=ACCENT, label="Rentang Interkuartil (Q1 hingga Q3)")
ax.plot(x, md_arr_p, marker="s", color=ACCENT, linewidth=2.5, markersize=8, zorder=5, label="Median Produktivitas Bulanan")

for xi, yi in zip(x, md_arr_p):
    ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 10), textcoords="offset points", ha="center", color="#2d3436")

ax.axvspan(4.5, 10.5, alpha=0.15, color="#e17055", label="Musim Kemarau (Mei hingga Oktober)")
ax.axvspan(10.5, 12.5, alpha=0.15, color="#0062b8")
ax.axvspan(0.5,  4.5,  alpha=0.15, color="#0062b8", label="Musim Hujan (November hingga April)")

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylim(max(0, min(q1_arr_p) - 1), max(q3_arr_p) * 1.1)
ax.set_title("Pola Produktivitas Padi Median Bulanan di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Produktivitas Padi (ton/ha)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_3_monthly_median_lineplot.png", bbox_inches="tight")
plt.close()


# 4. IQR OUTLIER IDENTIFICATION
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

padi_nz_series = df_nz[COL_PADI]
outliers = df_nz[
    (padi_nz_series < lower_fence) | (padi_nz_series > upper_fence)
][["wilayah", COL_BULAN, COL_PADI]].copy()
outliers["bulan_nama"] = outliers[COL_BULAN].apply(lambda m: MONTH_LABELS[int(m) - 1])
outliers = outliers.sort_values(COL_PADI, ascending=False).reset_index(drop=True)

print("── 4. IQR Outlier Identification ──")
print(f"   Lower fence : {lower_fence:.4f}  (Q1 − 1.5×IQR)")
print(f"   Upper fence : {upper_fence:.4f}  (Q3 + 1.5×IQR)")
print(f"   Outliers    : {len(outliers)} observations\n")

outliers.to_csv(f"{OUT_DIR}/padi_4_iqr_outliers.csv", index=False)


# 5. CV PER WILAYAH
cv_per_wilayah = (
    df_nz.groupby("wilayah")[COL_PADI]
    .agg(n_obs="count", mean_prod="mean", std_prod="std")
    .reset_index()
)
cv_per_wilayah["CV (%)"] = (cv_per_wilayah["std_prod"] / cv_per_wilayah["mean_prod"] * 100).round(2)
cv_per_wilayah = cv_per_wilayah.sort_values("CV (%)", ascending=False).reset_index(drop=True)

median_cv = float(cv_per_wilayah["CV (%)"].median())
cv_per_wilayah["Stabilitas"] = cv_per_wilayah["CV (%)"].apply(
    lambda x: "Stabil" if x <= median_cv else "Tidak Stabil"
)

print("── 5. CV per Wilayah ──")
print(f"   Median CV (threshold) : {median_cv:.2f}%\n")

cv_per_wilayah.to_csv(f"{OUT_DIR}/padi_5_cv_per_wilayah.csv", index=False)

colors_cv = ["#d63031" if s == "Tidak Stabil" else ACCENT for s in cv_per_wilayah["Stabilitas"]]
cv_vals   = cv_per_wilayah["CV (%)"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(cv_per_wilayah["wilayah"].tolist(), cv_vals, color=colors_cv, edgecolor="white", height=0.7)
ax.axvline(median_cv, color="#2d3436", linewidth=1.8, linestyle="--", label=f"Median CV: {median_cv:.1f}%")

for bar, val in zip(bars, cv_vals):
    ax.text(float(val) + 0.3, float(bar.get_y() + bar.get_height() / 2), f"{val:.1f}%", va="center")

legend_patches = [
    mpatches.Patch(color=ACCENT,    label="Stabil (CV ≤ median)"),
    mpatches.Patch(color="#d63031", label="Tidak Stabil (CV > median)"),
]
ax.legend(handles=legend_patches + [ax.lines[0]], loc="lower right")
ax.set_title("Koefisien Variasi (CV) Produktivitas Padi per Wilayah di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Koefisien Variasi (%)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_5_cv_per_wilayah.png", bbox_inches="tight")
plt.close()


# 6. BOXPLOT PER WILAYAH
wilayah_order = (
    df_nz.groupby("wilayah")[COL_PADI].median()
    .sort_values(ascending=False).index.tolist()
)
groups_w = [df_nz.loc[df_nz["wilayah"] == w, COL_PADI].to_numpy(dtype=float) for w in wilayah_order] # type: ignore

fig, ax = plt.subplots(figsize=(7, 14))
bp = ax.boxplot(
    groups_w, patch_artist=True, vert=False, notch=False,
    medianprops=dict(color="#d63031", linewidth=2),
    whiskerprops=dict(linewidth=1.1),
    capprops=dict(linewidth=1.1),
    flierprops=dict(marker="o", markersize=3, alpha=0.45, markerfacecolor="#636e72", markeredgewidth=0),
)
cmap = plt.colormaps["Greens"]
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(0.30 + 0.50 * (i / len(wilayah_order))))
    patch.set_alpha(0.85)

ax.set_yticks(range(1, len(wilayah_order) + 1))
ax.set_yticklabels(wilayah_order, fontsize=8.5)
ax.set_title("Variabilitas Produktivitas Padi Antarwilayah di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Produktivitas (ton/ha)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_6_boxplot_per_wilayah.png", bbox_inches="tight")
plt.close()


# B1. HISTOGRAM + KDE
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(padi_nz, bins=25, density=True, color=ACCENT, edgecolor="white", alpha=0.75, label="Histogram (density)")
kde   = gaussian_kde(padi_nz)
x_kde = np.linspace(padi_nz.min(), padi_nz.max(), 300)
ax.plot(x_kde, kde(x_kde), color="#2d3436", linewidth=2.2, label="KDE")
ax.axvline(median_p, color="#0984e3", linestyle="-.", linewidth=2.5, label=f"Median: {median_p:.2f} ton/ha")
ax.axvline(mean_p,   color="#d63031", linestyle="--", linewidth=1.5, label=f"Rata-rata: {mean_p:.2f} ton/ha")

ax.set_title("Histogram dan KDE Produktivitas Padi di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Produktivitas (ton/ha)")
ax.set_ylabel("Densitas Frekuensi")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B1_histogram_kde.png", bbox_inches="tight")
plt.close()


# B2. FREQUENCY DISTRIBUTION
pmin         = float(np.floor(padi_nz.min() * 2) / 2)
pmax         = float(np.ceil(padi_nz.max() * 2) / 2)
bin_edges_p  = list(np.arange(pmin, pmax + 0.5, 0.5))
bin_labels_p = [f"{b:.1f}–{b+0.5:.1f}" for b in bin_edges_p[:-1]]
fd_padi = pd.cut(df_nz[COL_PADI], bins=bin_edges_p, labels=bin_labels_p, right=False).value_counts().reindex(bin_labels_p, fill_value=0)
fd_arr  = fd_padi.to_numpy(dtype=int)

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(bin_labels_p, fd_arr, color=ACCENT, edgecolor="white", width=0.8, alpha=0.88)
for bar, cnt in zip(bars, fd_arr):
    if cnt > 0:
        ax.text(float(bar.get_x() + bar.get_width() / 2), float(bar.get_height()) + 0.3, str(cnt), ha="center", va="bottom", fontsize=7.5)

ax.set_title("Distribusi Frekuensi Berdasarkan Interval Produktivitas Padi di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Kelas Interval Produktivitas (ton/ha)")
ax.set_ylabel("Frekuensi Observasi")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B2_freq_distribution.png", bbox_inches="tight")
plt.close()


# B3. LINE PLOT ALL WILAYAH
pivot_p          = df_nz.pivot_table(index=COL_BULAN, columns="wilayah", values=COL_PADI, aggfunc="median")
overall_median_p = pivot_p.median(axis=1)

fig, ax = plt.subplots(figsize=(14, 7))
cmap_tab = cm.get_cmap("tab20", len(pivot_p.columns))
for i, col in enumerate(pivot_p.columns):
    ax.plot(pivot_p.index, pivot_p[col].to_numpy(dtype=float), alpha=0.6, color=cmap_tab(i), label=col)
ax.plot(pivot_p.index, overall_median_p.to_numpy(dtype=float), color="black", linestyle="--", linewidth=3.5, label="Median Seluruh Wilayah", zorder=5)

ax.axvspan(4.5, 10.5, alpha=0.07, color="#e17055", label="Musim Kemarau (Mei hingga Oktober)")
ax.axvspan(10.5, 12.5, alpha=0.05, color=ACCENT)
ax.axvspan(0.5,  4.5,  alpha=0.05, color=ACCENT, label="Musim Hujan (November hingga April)")

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title("Produktivitas Padi Median Bulanan per Wilayah di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Produktivitas (ton/ha)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=7)
plt.subplots_adjust(right=0.75)
plt.savefig(f"{OUT_DIR}/padi_B3_lineplot_all_wilayah.png", bbox_inches="tight")
plt.close()


# B4. BOXPLOT PER BULAN
groups_b = [df_nz.loc[df_nz[COL_BULAN] == m, COL_PADI].to_numpy(dtype=float) for m in range(1, 13)]

fig, ax = plt.subplots(figsize=(12, 6))
bp2 = ax.boxplot(
    groups_b, patch_artist=True, notch=False,
    medianprops=dict(color="#d63031", linewidth=2.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3, alpha=0.45, markerfacecolor="#636e72", markeredgewidth=0),
)
cmap2 = plt.colormaps["Greens"]
for i, patch in enumerate(bp2["boxes"]):
    patch.set_facecolor(cmap2(0.30 + 0.45 * (i / 12)))
    patch.set_alpha(0.85)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title("Sebaran Produktivitas Padi Bulanan Antarwilayah di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Produktivitas (ton/ha)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B4_boxplot_per_bulan.png", bbox_inches="tight")
plt.close()


# B5. BOXPLOT OVERALL
q1_agg  = float(np.percentile(padi_nz, 25))
q3_agg  = float(np.percentile(padi_nz, 75))
med_agg = float(np.median(padi_nz))
iqr_agg = q3_agg - q1_agg
pda_agg = q3_agg + 1.5 * iqr_agg
pdb_agg = q1_agg - 1.5 * iqr_agg

fig, ax = plt.subplots(figsize=(12, 4))
bp3 = ax.boxplot(
    [padi_nz], positions=[1], patch_artist=True, notch=False, widths=0.4,
    vert=False,
    medianprops=dict(color="#d63031", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=4, alpha=0.5, markerfacecolor="#636e72", markeredgewidth=0),
)
bp3["boxes"][0].set_facecolor(ACCENT)
bp3["boxes"][0].set_alpha(0.85)

ax.set_ylim(0.6, 1.4)
ax.set_yticks([1])
ax.set_yticklabels([""])
ax.set_ylabel("Seluruh Kabupaten dan Kota")
ax.set_xlabel("Produktivitas Padi (ton/ha)")
ax.set_title("Boxplot Produktivitas Padi di Jawa Timur pada Tahun 2024", fontweight="bold", pad=12)

legend_text = "\n".join([
    f"PDA = {pda_agg:.2f}",
    f"Q3 = {q3_agg:.2f}",
    f"Q2 = {med_agg:.2f}",
    f"Q1 = {q1_agg:.2f}",
    f"PDB = {pdb_agg:.2f}",
])
legend_patch = Patch(color="none", label=legend_text)
ax.legend(
    handles=[legend_patch],
    loc="upper right",
    handlelength=0,
    handleheight=0,
    handletextpad=0,
    frameon=True,
    edgecolor="black",
    fancybox=False,
    prop={"weight": "medium", "size": 10},
)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/padi_B5_boxplot_overall.png", bbox_inches="tight")
plt.close()