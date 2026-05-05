"""
01_rain_analysis.py
===================
RAIN (Curah Hujan) — Full Descriptive Analysis
-----------------------------------------------
PRIMARY outputs (for journal):
  1.  Descriptive statistics table        → rain_1_descriptive_stats.csv
  2.  BMKG category frequency bar chart   → rain_2_bmkg_frequency.png / .csv
  3.  Monthly mean line plot (wet/dry)    → rain_3_monthly_mean_lineplot.png
  4.  IQR outlier identification table    → rain_4_iqr_outliers.csv

BACKUP outputs (additional charts):
  B1. Histogram                           → rain_B1_histogram.png
  B2. Frequency distribution (binned)     → rain_B2_freq_distribution.png
  B3. Line plot — all 38 wilayah          → rain_B3_lineplot_all_wilayah.png
  B4. Boxplot per bulan                   → rain_B4_boxplot_per_bulan.png
  B5. Boxplot keseluruhan                 → rain_B5_boxplot_overall.png

All outputs → __output__/rain/
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = "__data__/ch_padi_training_dataset.csv"
OUT_DIR   = "__output__/rain"

COL_NAME  = "nama_wilayah"
COL_JENIS = "jenis_wilayah"          # 0 = Kabupaten, 1 = Kota
COL_BULAN = "bulan"
COL_CH    = "curah_hujan_per_bulan"

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# BMKG monthly rainfall thresholds (mm/bulan)
BMKG_BINS    = [-np.inf, 100, 300, 500, np.inf]
BMKG_KEYS    = ["Rendah", "Menengah", "Tinggi", "Sangat Tinggi"]
BMKG_XLABELS = ["Rendah\n(≤100 mm)", "Menengah\n(101–300 mm)",
                "Tinggi\n(301–500 mm)", "Sangat Tinggi\n(>500 mm)"]
BMKG_COLORS  = ["#74b9ff", "#0984e3", "#e17055", "#d63031"]

ACCENT = "#0984e3"

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.30,
    "grid.linestyle"  : "--",
    "figure.dpi"      : 150,
})

os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD & PREPARE ────────────────────────────────────────────────────────────
df_raw = pd.read_csv(DATA_PATH)
df_raw["wilayah"] = (
    df_raw[COL_NAME] + " "
    + df_raw[COL_JENIS].map({0: "Kab.", 1: "Kota"})
)
df = df_raw.dropna(subset=[COL_CH]).copy()
ch = df[COL_CH].to_numpy(dtype=float)          # numpy array — no Pylance complaints

print("=" * 65)
print("  01_RAIN_ANALYSIS.PY")
print("=" * 65)
print(f"  Rows (after dropping NaN CH) : {len(df)}")
print(f"  Unique wilayah               : {df['wilayah'].nunique()}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════
mean_ch   = float(np.mean(ch))
median_ch = float(np.median(ch))
std_ch    = float(np.std(ch, ddof=1))
min_ch    = float(np.min(ch))
max_ch    = float(np.max(ch))
range_ch  = max_ch - min_ch
q1        = float(np.percentile(ch, 25))
q3        = float(np.percentile(ch, 75))
iqr       = q3 - q1

stats_dict = {
    "Mean (Rata-rata)"         : mean_ch,
    "Median"                   : median_ch,
    "Simpangan Baku (Std Dev)" : std_ch,
    "Minimum"                  : min_ch,
    "Maximum"                  : max_ch,
    "Rentang (Range)"          : range_ch,
    "Q1 (25th Percentile)"     : q1,
    "Q3 (75th Percentile)"     : q3,
    "IQR"                      : iqr,
}

print("── 1. Descriptive Statistics (Curah Hujan, mm/bulan) ──")
for k, v in stats_dict.items():
    print(f"   {k:<30}: {v:>12.2f} mm")
print()

pd.DataFrame({
    "Statistik"       : list(stats_dict.keys()),
    "Nilai (mm/bulan)": [f"{v:.2f}" for v in stats_dict.values()],
}).to_csv(f"{OUT_DIR}/rain_1_descriptive_stats.csv", index=False)
print(f"   → Saved: {OUT_DIR}/rain_1_descriptive_stats.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BMKG CATEGORY FREQUENCY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
df["bmkg_cat"] = pd.cut(df[COL_CH], bins=BMKG_BINS, labels=BMKG_KEYS)
freq_bmkg      = df["bmkg_cat"].value_counts().reindex(BMKG_KEYS, fill_value=0)
pct_bmkg       = (freq_bmkg / len(df) * 100).round(1)

freq_arr = freq_bmkg.to_numpy(dtype=int)
pct_arr  = pct_bmkg.to_numpy(dtype=float)

print("── 2. BMKG Category Frequency Distribution ──")
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
    "Distribusi Frekuensi Kategori Curah Hujan (BMKG)\nJawa Timur — 38 Wilayah",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Kategori BMKG", fontsize=11)
ax.set_ylabel("Frekuensi (jumlah observasi)", fontsize=11)
ax.set_ylim(0, float(freq_arr.max()) * 1.20)
ax.grid(axis="x", alpha=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_2_bmkg_frequency.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_2_bmkg_frequency.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MONTHLY MEAN LINE PLOT — seasonal wet/dry pattern
# ═══════════════════════════════════════════════════════════════════════════════
monthly_mean = df.groupby(COL_BULAN)[COL_CH].mean().reindex(range(1, 13))
monthly_std  = df.groupby(COL_BULAN)[COL_CH].std().reindex(range(1, 13))

mm_arr  = monthly_mean.to_numpy(dtype=float)
msd_arr = monthly_std.to_numpy(dtype=float)

print("── 3. Monthly Mean Curah Hujan ──")
for m, mean_v, sd_v in zip(range(1, 13), mm_arr, msd_arr):
    print(f"   {MONTH_LABELS[m-1]:<4}: {mean_v:>8.1f} mm  (±{sd_v:.1f})")
print()

x = list(range(1, 13))
lower_band = np.clip(mm_arr - msd_arr, 0, None)
upper_band = mm_arr + msd_arr

fig, ax = plt.subplots(figsize=(11, 5))
ax.fill_between(x, lower_band, upper_band, alpha=0.13, color=ACCENT)
ax.plot(x, mm_arr, marker="o", color=ACCENT,
        linewidth=2.5, markersize=8, zorder=5, label="Rata-rata CH bulanan")
for xi, yi in zip(x, mm_arr):
    ax.annotate(f"{yi:.0f}", xy=(xi, yi), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=8.5, color="#2d3436")

ax.axvspan(5.5, 9.5,  alpha=0.07, color="#e17055", label="Musim Kemarau (est. Jun–Sep)")
ax.axvspan(9.5, 12.5, alpha=0.05, color=ACCENT)
ax.axvspan(0.5, 3.5,  alpha=0.05, color=ACCENT, label="Musim Hujan (est.)")

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Rata-rata Curah Hujan Bulanan — Seluruh Wilayah Jawa Timur\n(Pola Musim Hujan / Kemarau)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Rata-rata Curah Hujan (mm)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_3_monthly_mean_lineplot.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_3_monthly_mean_lineplot.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. IQR OUTLIER IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = df[
    (df[COL_CH] < lower_fence) | (df[COL_CH] > upper_fence)
][["wilayah", COL_BULAN, COL_CH]].copy()
outliers["bulan_nama"] = outliers[COL_BULAN].apply(
    lambda m: MONTH_LABELS[int(m) - 1]
)
outliers = outliers.sort_values(COL_CH, ascending=False).reset_index(drop=True)

print("── 4. IQR Outlier Identification ──")
print(f"   Q1          : {q1:.2f} mm")
print(f"   Q3          : {q3:.2f} mm")
print(f"   IQR         : {iqr:.2f} mm")
print(f"   Lower fence : {lower_fence:.2f} mm  (Q1 − 1.5×IQR)")
print(f"   Upper fence : {upper_fence:.2f} mm  (Q3 + 1.5×IQR)")
print(f"   Outliers    : {len(outliers)} observations")
print()
if not outliers.empty:
    print(outliers[["wilayah", "bulan_nama", COL_CH]].to_string(index=False))
print()

outliers.to_csv(f"{OUT_DIR}/rain_4_iqr_outliers.csv", index=False)
print(f"   → Saved: {OUT_DIR}/rain_4_iqr_outliers.csv\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKUP CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("── BACKUP CHARTS ──")

# ── B1. HISTOGRAM ─────────────────────────────────────────────────────────────
p99    = float(np.percentile(ch, 99))
ch_clip = ch[ch <= p99]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ch_clip, bins=35, color=ACCENT, edgecolor="white", linewidth=0.6, alpha=0.85)
ax.axvline(mean_ch,   color="#d63031", linewidth=2,
           linestyle="--", label=f"Mean: {mean_ch:.0f} mm")
ax.axvline(median_ch, color="#00b894", linewidth=2,
           linestyle="-.", label=f"Median: {median_ch:.0f} mm")
ax.set_title(
    "Histogram Curah Hujan Bulanan\n(nilai >P99 dipotong untuk keterbacaan)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Curah Hujan (mm/bulan)", fontsize=11)
ax.set_ylabel("Frekuensi", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B1_histogram.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_B1_histogram.png")

# ── B2. FREQUENCY DISTRIBUTION (equal-width bins, labeled) ───────────────────
bin_edges_fd  = [0, 50, 100, 150, 200, 300, 400, 500, 700, 1000]
bin_labels_fd = ["0–50", "50–100", "100–150", "150–200",
                 "200–300", "300–400", "400–500", "500–700", "700–1000"]
ch_series = df.loc[df[COL_CH] <= 1000, COL_CH]
fd_series = pd.cut(
    ch_series, bins=bin_edges_fd, labels=bin_labels_fd, right=False
).value_counts().reindex(bin_labels_fd, fill_value=0)
fd_arr = fd_series.to_numpy(dtype=int)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(bin_labels_fd, fd_arr, color=ACCENT,
              edgecolor="white", linewidth=0.7, width=0.7, alpha=0.88)
for bar, cnt in zip(bars, fd_arr):
    if cnt > 0:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=9)
ax.set_title("Distribusi Frekuensi Curah Hujan per Kelas Interval",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Kelas Interval (mm/bulan)", fontsize=11)
ax.set_ylabel("Frekuensi", fontsize=11)
ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B2_freq_distribution.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_B2_freq_distribution.png")

# ── B3. LINE PLOT — all 38 wilayah overlaid ───────────────────────────────────
pivot        = df.pivot_table(index=COL_BULAN, columns="wilayah",
                               values=COL_CH, aggfunc="mean")
overall_mean = pivot.mean(axis=1)

fig, ax = plt.subplots(figsize=(12, 6))
for col in pivot.columns:
    ax.plot(pivot.index, pivot[col].to_numpy(dtype=float),
            linewidth=0.9, alpha=0.45, color=ACCENT)
ax.plot(pivot.index, overall_mean.to_numpy(dtype=float),
        color="black", linewidth=2.5, linestyle="--",
        label="Rata-rata Seluruh Wilayah", zorder=5)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Curah Hujan Bulanan — 38 Wilayah Jawa Timur\n"
    "(garis hitam putus-putus = rata-rata keseluruhan)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Curah Hujan (mm/bulan)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B3_lineplot_all_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_B3_lineplot_all_wilayah.png")

# ── B4. BOXPLOT PER BULAN ─────────────────────────────────────────────────────
df_b   = df[df[COL_CH] <= float(np.percentile(ch, 99))].copy()
groups = [
    df_b.loc[df_b[COL_BULAN] == m, COL_CH].dropna().to_numpy(dtype=float)
    for m in range(1, 13)
]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(
    groups,
    patch_artist=True,
    notch=False,
    medianprops=dict(color="#d63031", linewidth=2.2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3,
                    alpha=0.45, markerfacecolor="#636e72", markeredgewidth=0),
)
cmap = plt.colormaps["Blues"]
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(0.30 + 0.45 * (i / 12)))
    patch.set_alpha(0.85)

ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_title(
    "Boxplot Curah Hujan per Bulan — 38 Wilayah Jawa Timur\n"
    "(nilai >P99 dipotong untuk keterbacaan)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Bulan", fontsize=11)
ax.set_ylabel("Curah Hujan (mm/bulan)", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B4_boxplot_per_bulan.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_B4_boxplot_per_bulan.png")


# ── B5. BOXPLOT KESELURUHAN (Seluruh Dataset) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 8))
bp = ax.boxplot(
    [ch],
    patch_artist=True,
    notch=False,
    widths=0.4,
    medianprops=dict(color="#d63031", linewidth=2.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=4,
                    alpha=0.5, markerfacecolor="#636e72", markeredgewidth=0),
)

bp["boxes"][0].set_facecolor(ACCENT)
bp["boxes"][0].set_alpha(0.85)

ax.set_xticks([1])
ax.set_xticklabels(["Seluruh Observasi\n(Jawa Timur, Semua Bulan)"], fontsize=11)
ax.set_title(
    "Distribusi Keseluruhan Curah Hujan",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_ylabel("Curah Hujan (mm/bulan)", fontsize=11)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rain_B5_boxplot_overall.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/rain_B5_boxplot_overall.png")

print()
print("=" * 65)
print("  01_rain_analysis.py  — COMPLETE")
print(f"  All outputs in: {OUT_DIR}/")
print("=" * 65)