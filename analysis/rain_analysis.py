"""
01_rain_analysis.py
===================
RAIN (Curah Hujan) — Full Descriptive Analysis

SATU SUMBER DATA UNTUK SEMUA CHART PER BULAN:
  df_agg = median curah hujan per wilayah per bulan
  Semua chart per bulan (line plot, boxplot per bulan) pakai df_agg.
  IQR, median, dan sebaran IDENTIK di semua chart.

  Raw data (ch) hanya untuk:
  - Statistika deskriptif (Section 1)
  - Distribusi frekuensi BMKG (Section 2)
  - Histogram (B1)
  - IQR outlier identification (Section 4)

PRIMARY outputs:
  1. rain_1_descriptive_stats.csv
  2. rain_2_bmkg_frequency.png / .csv
  3. rain_3_monthly_median_lineplot.png
  4. rain_4_iqr_outliers.csv

BACKUP outputs:
  B1. rain_B1_histogram.png
  B2. rain_B2_freq_distribution.png
  B3. rain_B3_lineplot_all_wilayah.png
  B4. rain_B4_boxplot_per_bulan.png
  B5. rain_B5_boxplot_overall.png

All outputs → __output__/rain/
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
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

ACCENT = "#00b894"

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
df = df_raw.dropna(subset=[COL_CH]).copy()
ch = df[COL_CH].to_numpy(dtype=float)

# ── SUMBER DATA TUNGGAL untuk semua chart per bulan ───────────────────────────
# Agregat: satu nilai per wilayah per bulan (median).
# Ini menghilangkan inkonsistensi antara line plot dan boxplot.
df_agg = (
    df.groupby([COL_BULAN, "wilayah"])[COL_CH]
    .median()
    .reset_index()
    .rename(columns={COL_CH: "ch_median"})
)

n_wilayah = df["wilayah"].nunique()

print("=" * 65)
print("  01_RAIN_ANALYSIS.PY")
print("=" * 65)
print(f"  Raw rows        : {len(df)}")
print(f"  Wilayah         : {n_wilayah}")
print(f"  Aggregated rows : {len(df_agg)}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS  (raw data)
# ═══════════════════════════════════════════════════════════════════════════════
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
print(f"   -> Saved: {OUT_DIR}/rain_1_descriptive_stats.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BMKG FREQUENCY DISTRIBUTION  (raw data)
# ═══════════════════════════════════════════════════════════════════════════════
df["bmkg_cat"] = pd.cut(df[COL_CH], bins=BMKG_BINS, labels=BMKG_KEYS)
freq_bmkg = df["bmkg_cat"].value_counts().reindex(BMKG_KEYS, fill_value=0)
pct_bmkg  = (freq_bmkg / len(df) * 100).round(1)
freq_arr  = freq_bmkg.to_numpy(dtype=int)
pct_arr   = pct_bmkg.to_numpy(dtype=float)

print("── 2. Distribusi Frekuensi BMKG ──")
for cat, cnt, pct in zip(BMKG_KEYS, freq_arr, pct_arr):
    print(f"   {cat:<15}: {cnt:>4} obs  ({pct:.1f}%)")
print()

pd.DataFrame({
    "Kategori BMKG" : BMKG_KEYS,
    "Frekuensi"     : freq_arr,
    "Persentase (%)": pct_arr,
}).to_csv(f"{OUT_DIR}/rain_2_bmkg_frequency.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(BMKG_XLABELS, freq_arr, color=BMKG_COLORS,
              edgecolor="white", linewidth=0.8, width=0.6)
for bar, cnt, pct in zip(bars, freq_arr, pct_arr):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{cnt}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
ax.set_title(
    "Distribusi Frekuensi Curah Hujan Bulanan\nBerdasarkan Klasifikasi BMKG, Jawa Timur 2024",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Klasifikasi Intensitas Curah Hujan (BMKG)", fontsize=11)
ax.set_ylabel("Jumlah Observasi", fontsize=11)
ax.set_ylim(0, float(freq_arr.max()) * 1.20)
ax.grid(axis="x", alpha=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_2_bmkg_frequency.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_2_bmkg_frequency.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MONTHLY MEDIAN LINE PLOT  (df_agg — konsisten dengan B4 boxplot)
# ═══════════════════════════════════════════════════════════════════════════════
monthly_stats = (
    df_agg.groupby(COL_BULAN)["ch_median"]
    .agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75)
    )
    .reindex(range(1, 13))
)

md_arr = monthly_stats["median"].to_numpy(dtype=float)
q1_arr = monthly_stats["q1"].to_numpy(dtype=float)
q3_arr = monthly_stats["q3"].to_numpy(dtype=float)

print("── 3. Pola Temporal (dari df_agg) ──")
for m, md_v, q1_v, q3_v in zip(range(1, 13), md_arr, q1_arr, q3_arr):
    print(f"   {MONTH_LABELS[m-1]:<4}: Median={md_v:>6.1f}  Q1={q1_v:>6.1f}  Q3={q3_v:>6.1f}")
print()

x = list(range(1, 13))

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(x, q1_arr, q3_arr, alpha=0.15, color=ACCENT,
                label="Rentang Interkuartil (Q1 hingga Q3)")
ax.plot(x, md_arr, marker="s", color=ACCENT,
        linewidth=2.5, markersize=8, zorder=5, label="Median curah hujan bulanan")

for xi, yi in zip(x, md_arr):
    ax.annotate(f"{yi:.0f}", xy=(xi, yi), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8.5, color="#2d3436")

ax.axvspan(4.5, 10.5, alpha=0.07, color="#e17055",
           label="Musim Kemarau (Mei hingga Oktober)")
ax.axvspan(10.5, 12.5, alpha=0.05, color=ACCENT)
ax.axvspan(0.5,  4.5,  alpha=0.05, color=ACCENT,
           label="Musim Hujan (November hingga April)")

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_ylim(0, max(q3_arr) * 1.5)
ax.set_title(
    f"Pola Curah Hujan Bulanan di Jawa Timur, Tahun 2024\n(n={n_wilayah} Kabupaten/Kota)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Curah Hujan Median (mm/bulan)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_3_monthly_median_lineplot.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_3_monthly_median_lineplot.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IQR OUTLIER IDENTIFICATION  (df_agg per bulan)
# ═══════════════════════════════════════════════════════════════════════════════
outlier_list = []
for bulan_val, group in df_agg.groupby(COL_BULAN):
    q1 = group["ch_median"].quantile(0.25)
    q3 = group["ch_median"].quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    # Filter pencilan di bulan tersebut
    mask = (group["ch_median"] < lower_fence) | (group["ch_median"] > upper_fence)
    outliers_bulan_ini = group[mask]
    
    # Hanya masukkan ke daftar jika memang ada outlier (mencegah list isi DataFrame kosong)
    if not outliers_bulan_ini.empty:
        outlier_list.append(outliers_bulan_ini)

# Gabungkan semua pencilan jika list tidak kosong
if len(outlier_list) > 0:
    outliers = pd.concat(outlier_list, ignore_index=True)
    outliers["bulan_nama"] = outliers[COL_BULAN].apply(
        lambda m: MONTH_LABELS[int(m) - 1]
    )
    # Urutkan berdasarkan bulan, lalu curah hujan tertinggi
    outliers = outliers.sort_values(
        by=[COL_BULAN, "ch_median"], ascending=[True, False]
    ).reset_index(drop=True)
    
    # Rapikan urutan kolom
    outliers = outliers[["wilayah", COL_BULAN, "bulan_nama", "ch_median"]]
else:
    # Jika tidak ada pencilan sama sekali sepanjang tahun
    outliers = pd.DataFrame(columns=["wilayah", COL_BULAN, "bulan_nama", "ch_median"])

print("── 4. Identifikasi Pencilan (Per Bulan) ──")
print(f"   Pencilan     : {len(outliers)} observasi (Berdasarkan sebaran per bulan)")
print()

outliers.to_csv(f"{OUT_DIR}/rain_4_iqr_outliers.csv", index=False)
print(f"   -> Saved: {OUT_DIR}/rain_4_iqr_outliers.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _style_bp(bp, n_boxes, cmap_name="Blues"):
    cmap_obj = plt.colormaps[cmap_name]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap_obj(0.30 + 0.45 * (i / max(n_boxes - 1, 1))))
        patch.set_alpha(0.85)


# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("── BACKUP CHARTS ──")

# ── B1. HISTOGRAM (raw data, linear) ─────────────────────────────────────────
CAP_HIST  = 1500
ch_capped = ch[ch <= CAP_HIST]
pct_shown = len(ch_capped) / len(ch) * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ch_capped, bins=30, color="#74b9ff",
        edgecolor="white", linewidth=0.6, alpha=0.85)
ax.axvline(median_ch, color=ACCENT, linewidth=2.5, linestyle="-",
           label=f"Median: {median_ch:.0f} mm")
ax.axvline(mean_ch, color="#d63031", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"Rata-rata: {mean_ch:.0f} mm")
ax.set_title(
    f"Distribusi Frekuensi Curah Hujan Bulanan, Jawa Timur 2024\n"
    f"({pct_shown:.1f}% observasi ditampilkan, nilai >1.500 mm tidak termasuk)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Curah Hujan (mm/bulan)", fontsize=11)
ax.set_ylabel("Frekuensi Observasi", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B1_histogram.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_B1_histogram.png")

# ── B2. FREQUENCY DISTRIBUTION (binned) ──────────────────────────────────────
bin_edges_fd  = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000]
bin_labels_fd = ["0-50", "50-100", "100-150", "150-200",
                 "200-300", "300-400", "400-500", "500-700", "700-1000"]
ch_series = df.loc[df[COL_CH] <= 1000, COL_CH]
fd_series = pd.cut(
    ch_series, bins=bin_edges_fd, labels=bin_labels_fd, right=False
).value_counts().reindex(bin_labels_fd, fill_value=0)
fd_arr = fd_series.to_numpy(dtype=int)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(bin_labels_fd, fd_arr, color="#0984e3",
              edgecolor="white", linewidth=0.7, width=0.7, alpha=0.88)
for bar, cnt in zip(bars, fd_arr):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=9)
ax.set_title(
    "Distribusi Frekuensi Curah Hujan Berdasarkan Kelas Interval\nJawa Timur 2024 (0 hingga 1.000 mm/bulan)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Kelas Interval (mm/bulan)", fontsize=11)
ax.set_ylabel("Frekuensi Observasi", fontsize=11)
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B2_freq_distribution.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_B2_freq_distribution.png")

# ── B3. LINE PLOT semua wilayah (df_agg) ─────────────────────────────────────
pivot = df_agg.pivot_table(index=COL_BULAN, columns="wilayah", values="ch_median")
overall_median_agg = pivot.median(axis=1)

fig, ax = plt.subplots(figsize=(14, 7))
cmap_tab = cm.get_cmap("tab20", len(pivot.columns))
for i, col in enumerate(pivot.columns):
    ax.plot(pivot.index, pivot[col].to_numpy(dtype=float),
            linewidth=1.2, alpha=0.6, color=cmap_tab(i), label=col)
ax.plot(pivot.index, overall_median_agg.to_numpy(dtype=float),
        color="black", linewidth=3.5, linestyle="--",
        label="Median Seluruh Wilayah", zorder=5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Curah Hujan Bulanan per Wilayah, Jawa Timur 2024",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Curah Hujan Median (mm/bulan)", fontsize=11)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=7)
plt.subplots_adjust(right=0.75)
plt.savefig(f"{OUT_DIR}/rain_B3_lineplot_all_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_B3_lineplot_all_wilayah.png")

# ── B4. BOXPLOT PER BULAN (df_agg) ────────────────
groups_agg = [
    df_agg.loc[df_agg[COL_BULAN] == m, "ch_median"].to_numpy(dtype=float)
    for m in range(1, 13)
]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(
    groups_agg,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="#d63031", linewidth=2.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=4,
                    alpha=0.6, markerfacecolor="#636e72", markeredgewidth=0),
)
_style_bp(bp, 12)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Sebaran Curah Hujan Bulanan Antarwilayah, Jawa Timur 2024",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Curah Hujan Median per Wilayah (mm/bulan)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B4_boxplot_per_bulan.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_B4_boxplot_per_bulan.png")

# ── B5. BOXPLOT OVERALL (df_agg) ─────────────────────────────────────────────
ch_agg_all = df_agg["ch_median"].to_numpy(dtype=float)
q1_agg  = float(np.percentile(ch_agg_all, 25))
q3_agg  = float(np.percentile(ch_agg_all, 75))
med_agg = float(np.median(ch_agg_all))

fig, ax = plt.subplots(figsize=(6, 8))
bp = ax.boxplot(
    [ch_agg_all],
    patch_artist=True,
    notch=False,
    widths=0.4,
    medianprops=dict(color="#d63031", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=5,
                    alpha=0.6, markerfacecolor="#636e72", markeredgewidth=0),
)
bp["boxes"][0].set_facecolor("#74b9ff")
bp["boxes"][0].set_alpha(0.85)

for label, val in {
    f"Q3: {q3_agg:.0f} mm" : q3_agg,
    f"Median: {med_agg:.0f} mm": med_agg,
    f"Q1: {q1_agg:.0f} mm" : q1_agg,
}.items():
    ax.annotate(label, xy=(1.15, val),
                xycoords=("axes fraction", "data"),
                fontsize=8, va="center", color="#2d3436")

ax.set_xticks([1])
ax.set_xticklabels([f"Seluruh Wilayah\n(n={n_wilayah}, Semua Bulan)"], fontsize=10)
ax.set_title(
    "Sebaran Keseluruhan Curah Hujan\nJawa Timur 2024",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_ylabel("Curah Hujan Median per Wilayah (mm/bulan)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B5_boxplot_overall.png", bbox_inches="tight")
plt.close()
print(f"   -> Saved: {OUT_DIR}/rain_B5_boxplot_overall.png")

print()
print("=" * 65)
print("  01_rain_analysis.py — COMPLETE")
print(f"  Semua output: {OUT_DIR}/")
print("=" * 65)