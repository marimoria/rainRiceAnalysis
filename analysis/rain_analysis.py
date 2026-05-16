"""
rain_analysis.py
===================
RAIN (Curah Hujan) Full Descriptive Analysis
"""

import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from plot_style import apply_academic_style

apply_academic_style()
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "__data__/ch_padi_training_dataset.csv"
OUT_DIR   = "__output__/rain"

COL_NAME  = "nama_wilayah"
COL_JENIS = "jenis_wilayah"
COL_BULAN = "bulan"
COL_CH    = "curah_hujan_per_bulan"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

BMKG_BINS    = [-np.inf, 100, 300, 500, np.inf]
BMKG_KEYS    = ["Rendah", "Menengah", "Tinggi", "Sangat Tinggi"]
BMKG_XLABELS = ["Rendah\n(<=100 mm)", "Menengah\n(101-300 mm)",
                "Tinggi\n(301-500 mm)", "Sangat Tinggi\n(>500 mm)"]
BMKG_COLORS  = ["#d63031", "#e17055", "#74b9ff", "#0984e3"]

ACCENT = "#0009b8"

os.makedirs(OUT_DIR, exist_ok=True)

# LOAD & PREPARE
df_raw = pd.read_csv(DATA_PATH)
df_raw["wilayah"] = (
    df_raw[COL_NAME] + " "
    + df_raw[COL_JENIS].map({0: "Kab.", 1: "Kota"})
)
df = df_raw.dropna(subset=[COL_CH]).copy()
ch = df[COL_CH].to_numpy(dtype=float)

df_agg = (
    df.groupby([COL_BULAN, "wilayah"])[COL_CH]
    .median()
    .reset_index()
    .rename(columns={COL_CH: "ch_median"})
)

n_wilayah = df["wilayah"].nunique()

print("=" * 65)
print("RAIN_ANALYSIS.PY")
print("=" * 65)
print(f"  Raw rows        : {len(df)}")
print(f"  Wilayah         : {n_wilayah}")
print(f"  Aggregated rows : {len(df_agg)}\n")

# 1. DESCRIPTIVE STATISTICS
mean_ch   = float(np.mean(ch))
median_ch = float(np.median(ch))
std_ch    = float(np.std(ch, ddof=1))
min_ch    = float(np.min(ch))
max_ch    = float(np.max(ch))
range_ch  = max_ch - min_ch
q1_raw    = float(np.percentile(ch, 25))
q3_raw    = float(np.percentile(ch, 75))
iqr_raw   = q3_raw - q1_raw
cv_all    = std_ch / mean_ch * 100

stats_dict = {
    "Mean (Rata-rata)"      : mean_ch,
    "Median (Nilai Tengah)" : median_ch,
    "Simpangan Baku"        : std_ch,
    "Minimum"               : min_ch,
    "Maksimum"              : max_ch,
    "Rentang"               : range_ch,
    "Q1 (Persentil ke-25)"  : q1_raw,
    "Q3 (Persentil ke-75)"  : q3_raw,
    "IQR"                   : iqr_raw,
    "Koefisien Variasi (%)" : cv_all,
}

print("── 1. Statistika Deskriptif ──")
for k, v in stats_dict.items():
    print(f"   {k:<30}: {v:>10.2f}")
print()

pd.DataFrame({
    "Statistik"       : list(stats_dict.keys()),
    "Nilai (mm/bulan)": [f"{v:.2f}" for v in stats_dict.values()],
}).to_csv(f"{OUT_DIR}/rain_1_descriptive_stats.csv", index=False)


# 2. BMKG FREQUENCY DISTRIBUTION
df["bmkg_cat"] = pd.cut(df[COL_CH], bins=BMKG_BINS, labels=BMKG_KEYS)
freq_bmkg = df["bmkg_cat"].value_counts().reindex(BMKG_KEYS, fill_value=0)
pct_bmkg  = (freq_bmkg / len(df) * 100).round(1)
freq_arr  = freq_bmkg.to_numpy(dtype=int)
pct_arr   = pct_bmkg.to_numpy(dtype=float)

pd.DataFrame({
    "Kategori BMKG" : BMKG_KEYS,
    "Frekuensi"     : freq_arr,
    "Persentase (%)": pct_arr,
}).to_csv(f"{OUT_DIR}/rain_2_bmkg_frequency.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(BMKG_XLABELS, freq_arr, color=BMKG_COLORS, edgecolor="white", width=0.6)

for bar, cnt, pct in zip(bars, freq_arr, pct_arr):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{cnt}\n({pct:.1f}%)", ha="center", va="bottom")

ax.set_title("Distribusi Frekuensi Curah Hujan Bulanan Berdasarkan Klasifikasi BMKG di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Klasifikasi Intensitas Curah Hujan (BMKG)")
ax.set_ylabel("Jumlah Observasi")
ax.set_ylim(0, float(freq_arr.max()) * 1.20)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_2_bmkg_frequency.png", bbox_inches="tight")
plt.close()


# 3. MONTHLY MEDIAN LINE PLOT
monthly_stats = (
    df_agg.groupby(COL_BULAN)["ch_median"]
    .agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
    .reindex(range(1, 13))
)

md_arr = monthly_stats["median"].to_numpy(dtype=float)
q1_arr = monthly_stats["q1"].to_numpy(dtype=float)
q3_arr = monthly_stats["q3"].to_numpy(dtype=float)

x = list(range(1, 13))
fig, ax = plt.subplots(figsize=(11, 5))

ax.fill_between(x, q1_arr, q3_arr, alpha=0.15, color=ACCENT, label="Rentang Interkuartil (Q1 hingga Q3)")
ax.plot(x, md_arr, marker="s", color=ACCENT, linewidth=2.5, markersize=8, zorder=5, label="Median curah hujan bulanan")

for xi, yi in zip(x, md_arr):
    ax.annotate(f"{yi:.0f}", xy=(xi, yi), xytext=(0, 10), textcoords="offset points", ha="center", color="#2d3436")

ax.axvspan(4.5, 10.5, alpha=0.15, color="#e17055", label="Musim Kemarau (Mei hingga Oktober)")
ax.axvspan(10.5, 12.5, alpha=0.15, color="#0062b8")
ax.axvspan(0.5,  4.5,  alpha=0.15, color="#0062b8", label="Musim Hujan (November hingga April)")

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS)
ax.set_ylim(0, max(q3_arr) * 1.5)
ax.set_title("Pola Curah Hujan Median Bulanan di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Curah Hujan Median (mm/bulan)")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_3_monthly_median_lineplot.png", bbox_inches="tight")
plt.close()


# 4. IQR OUTLIER IDENTIFICATION
outlier_list = []
for bulan_val, group in df_agg.groupby(COL_BULAN):
    q1 = group["ch_median"].quantile(0.25)
    q3 = group["ch_median"].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    mask = (group["ch_median"] < lower_fence) | (group["ch_median"] > upper_fence)
    outliers_bulan_ini = group[mask]
    
    if not outliers_bulan_ini.empty:
        outlier_list.append(outliers_bulan_ini)

if len(outlier_list) > 0:
    outliers = pd.concat(outlier_list, ignore_index=True)
    outliers["bulan_nama"] = outliers[COL_BULAN].apply(lambda m: MONTH_LABELS[int(m) - 1])
    outliers = outliers.sort_values(by=[COL_BULAN, "ch_median"], ascending=[True, False]).reset_index(drop=True)
    outliers = outliers[["wilayah", COL_BULAN, "bulan_nama", "ch_median"]]
else:
    outliers = pd.DataFrame(columns=["wilayah", COL_BULAN, "bulan_nama", "ch_median"])

outliers.to_csv(f"{OUT_DIR}/rain_4_iqr_outliers.csv", index=False)


# HELPER
def _style_bp(bp, n_boxes, cmap_name="Blues"):
    cmap_obj = plt.colormaps[cmap_name]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap_obj(0.30 + 0.45 * (i / max(n_boxes - 1, 1))))
        patch.set_alpha(0.85)


# B1. HISTOGRAM
CAP_HIST  = 1500
ch_capped = ch[ch <= CAP_HIST]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ch_capped, bins=30, color="#74b9ff", edgecolor="white", alpha=0.85)
ax.axvline(median_ch, color=ACCENT, linestyle="-", label=f"Median: {median_ch:.0f} mm")
ax.axvline(mean_ch, color="#d63031", linestyle="--", alpha=0.7, label=f"Rata-rata: {mean_ch:.0f} mm")

ax.set_title("Distribusi Frekuensi Curah Hujan Bulanan di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Curah Hujan (mm/bulan)")
ax.set_ylabel("Frekuensi Observasi")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B1_histogram.png", bbox_inches="tight")
plt.close()


# B2. FREQUENCY DISTRIBUTION
bin_edges_fd  = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000]
bin_labels_fd = ["0-50", "50-100", "100-150", "150-200", "200-300", "300-400", "400-500", "500-700", "700-1000"]
ch_series = df.loc[df[COL_CH] <= 1000, COL_CH]
fd_series = pd.cut(ch_series, bins=bin_edges_fd, labels=bin_labels_fd, right=False).value_counts().reindex(bin_labels_fd, fill_value=0)
fd_arr = fd_series.to_numpy(dtype=int)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(bin_labels_fd, fd_arr, color="#0984e3", edgecolor="white", width=0.7, alpha=0.88)
for bar, cnt in zip(bars, fd_arr):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(cnt), ha="center", va="bottom")

ax.set_title("Distribusi Frekuensi Curah Hujan Berdasarkan Kelas Interval di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Kelas Interval Curah Hujan (mm/bulan)")
ax.set_ylabel("Frekuensi Observasi")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B2_freq_distribution.png", bbox_inches="tight")
plt.close()


# B3. LINE PLOT ALL WILAYAH
pivot = df_agg.pivot_table(index=COL_BULAN, columns="wilayah", values="ch_median")
overall_median_agg = pivot.median(axis=1)

fig, ax = plt.subplots(figsize=(14, 7))
cmap_tab = cm.get_cmap("tab20", len(pivot.columns))
for i, col in enumerate(pivot.columns):
    ax.plot(pivot.index, pivot[col].to_numpy(dtype=float), alpha=0.6, color=cmap_tab(i), label=col)
ax.plot(pivot.index, overall_median_agg.to_numpy(dtype=float), color="black", linestyle="--", linewidth=3.5, label="Median Seluruh Wilayah", zorder=5)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title("Curah Hujan Median Bulanan per Wilayah di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Curah Hujan (mm/bulan)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2)
plt.subplots_adjust(right=0.75)
plt.savefig(f"{OUT_DIR}/rain_B3_lineplot_all_wilayah.png", bbox_inches="tight")
plt.close()


# B4. BOXPLOT PER BULAN
groups_agg = [df_agg.loc[df_agg[COL_BULAN] == m, "ch_median"].to_numpy(dtype=float) for m in range(1, 13)]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(
    groups_agg, patch_artist=True, notch=False,
    medianprops=dict(color="#d63031", linewidth=2.2), 
    flierprops=dict(marker="o", alpha=0.6, markerfacecolor="#636e72", markeredgewidth=0)
)
_style_bp(bp, 12)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title("Sebaran Curah Hujan Median Bulanan Antarwilayah di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Bulan")
ax.set_ylabel("Curah Hujan (mm/bulan)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B4_boxplot_per_bulan.png", bbox_inches="tight")
plt.close()


# B5. BOXPLOT OVERALL
ch_agg_all = df_agg["ch_median"].to_numpy(dtype=float)
q1_agg  = float(np.percentile(ch_agg_all, 25))
q3_agg  = float(np.percentile(ch_agg_all, 75))
med_agg = float(np.median(ch_agg_all))
iqr_agg = q3_agg - q1_agg
pda_agg = q3_agg + 1.5 * iqr_agg
pdb_agg = q1_agg - 1.5 * iqr_agg

fig, ax = plt.subplots(figsize=(12, 4))
bp = ax.boxplot(
    [ch_agg_all], positions=[1], patch_artist=True, notch=False, widths=0.4,
    vert=False,
    medianprops=dict(color="#d63031", linewidth=2.5),
    flierprops=dict(marker="o", alpha=0.6, markerfacecolor="#636e72", markeredgewidth=0)
)
bp["boxes"][0].set_facecolor("#74b9ff")
bp["boxes"][0].set_alpha(0.85)

ax.set_ylim(0.6, 1.4)
ax.set_yticks([1])
ax.set_yticklabels([""])
ax.set_ylabel("Seluruh Kabupaten dan Kota")
ax.set_xlabel("Curah Hujan Median per Wilayah (mm/bulan)")
ax.set_title("Boxplot Curah Hujan di Jawa Timur pada Tahun 2024", fontweight="bold", pad=12)

legend_text = "\n".join([
    f"PDA = {pda_agg:.1f}",
    f"Q3 = {q3_agg:.0f}",
    f"Q2 = {med_agg:.0f}",
    f"Q1 = {q1_agg:.0f}",
    f"PDB = {pdb_agg:.1f}",
])
from matplotlib.patches import Patch
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
plt.savefig(f"{OUT_DIR}/rain_B5_boxplot_overall.png", bbox_inches="tight")
plt.close()