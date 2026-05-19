"""
combined_analysis.py
======================
COMBINED - Rain and Padi Relationship Analysis
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from plot_style import apply_academic_style

apply_academic_style()
warnings.filterwarnings("ignore")

# CONFIG
DATA_PATH = "__data__/ch_padi_training_dataset.csv"
OUT_DIR   = "__output__/combined"
os.makedirs(OUT_DIR, exist_ok=True)

COL_NAME  = "nama_wilayah"
COL_JENIS = "jenis_wilayah"
COL_BULAN = "bulan"
COL_CH    = "curah_hujan_per_bulan"
COL_PADI  = "produktivitas"

SPEARMAN_THRESHOLD: float = 0.0
ACCENT = "#6c5ce7"

# LOAD & PREPARE
df_raw = pd.read_csv(DATA_PATH)
df_raw["wilayah"] = (
    df_raw[COL_NAME] + " "
    + df_raw[COL_JENIS].map({0: "Kab.", 1: "Kota"})
)

df = df_raw.dropna(subset=[COL_CH, COL_PADI]).copy()
df_pair = df.copy()
df_clean = df_pair.copy()

print("03_COMBINED_ANALYSIS.PY")
print(f"  Rows (both vars present, produktivitas>0) : {len(df_pair)}")
print(f"  Unique wilayah                            : {df_pair['wilayah'].nunique()}\n")

# 1. LOWESS SCATTER PLOT
x_raw = df_clean[COL_CH].to_numpy(dtype=float)
y_raw = df_clean[COL_PADI].to_numpy(dtype=float)

lw = lowess(y_raw, x_raw, frac=0.35, it=3, return_sorted=True)

max_idx = np.argmax(lw[:, 1])
peak_x = lw[max_idx, 0]
peak_y = lw[max_idx, 1]

q25 = np.percentile(x_raw, 25)
q75 = np.percentile(x_raw, 75)

print("1. LOWESS Scatter Plot")
print(f"   LOWESS frac = 0.35  |  n = {len(x_raw)} observations (all valid data)\n")

fig, ax = plt.subplots(figsize=(10, 6))

ax.axvspan(q25, q75, color='gray', alpha=0.15, label='Mayoritas Data (Q1-Q3)')

ax.scatter(x_raw, y_raw, alpha=0.28, s=20, color=ACCENT, edgecolors="none", label="Observasi")
ax.plot(lw[:, 0], lw[:, 1], color="#d63031", linewidth=2.8, label="LOWESS (frac=0.35)", zorder=5)

ax.scatter([peak_x], [peak_y], color='black', s=60, zorder=6, label=f'Puncak Tren ({peak_x:.0f} mm, {peak_y:.2f} ton/ha)')
ax.vlines(x=peak_x, ymin=-1, ymax=peak_y, color='black', linestyle='--', alpha=0.6, linewidth=1.5, zorder=4)
ax.hlines(y=peak_y, xmin=-100, xmax=peak_x, color='black', linestyle='--', alpha=0.6, linewidth=1.5, zorder=4)

ax.set_xscale('symlog', linthresh=10)
ax.set_xticks([0, 10, 100, 1000])
ax.set_xticklabels(['0', '10', '100', '1000'])

ax.set_xlim(left=-1, right=np.max(x_raw) * 1.5)
ax.set_ylim(bottom=-0.5, top=np.max(y_raw) * 1.05)

ax.set_title("Scatterplot LOWESS Curah Hujan per Bulan terhadap Produktivitas Padi di Jawa Timur pada Tahun 2024", fontweight="bold")
ax.set_xlabel("Curah Hujan (mm/bulan)")
ax.set_ylabel("Produktivitas Padi (ton/ha)")

ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_1_lowess_scatter.png")
plt.close()


# 2. SPEARMAN CORRELATION OVERALL
ch_pair   = df_pair[COL_CH].to_numpy(dtype=float)
padi_pair = df_pair[COL_PADI].to_numpy(dtype=float)

rho_result = stats.spearmanr(ch_pair, padi_pair)
rho_all    = float(rho_result.statistic) # type: ignore[attr-defined]
p_all      = float(rho_result.pvalue)    # type: ignore[attr-defined]

abs_rho = abs(rho_all)
if abs_rho >= 0.80:
    strength = "Sangat Kuat"
elif abs_rho >= 0.60:
    strength = "Kuat"
elif abs_rho >= 0.40:
    strength = "Sedang"
elif abs_rho >= 0.20:
    strength = "Lemah"
else:
    strength = "Sangat Lemah"

direction = "Positif" if rho_all > 0 else "Negatif"
sig_label = "Signifikan (p<0.05)" if p_all < 0.05 else "Tidak Signifikan"
n_pairs   = len(df_pair)

print("2. Spearman Correlation — Overall")
print(f"   n pairs      : {n_pairs}")
print(f"   Spearman rho : {rho_all:.4f}")
print(f"   p-value      : {p_all:.6f}")
print(f"   Significance : {sig_label}")
print(f"   Direction    : {direction}")
print(f"   Strength     : {strength}\n")

overall_df = pd.DataFrame([{
    "n_pairs"     : n_pairs,
    "Spearman_rho": round(rho_all, 4),
    "p_value"     : round(p_all, 6),
    "Signifikansi": sig_label,
    "Arah"        : direction,
    "Kekuatan"    : strength,
}])
overall_df.to_csv(f"{OUT_DIR}/combined_2_spearman_overall.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")
card_text = (
    f"Korelasi Spearman Keseluruhan\n\n"
    f"n pasangan observasi  :  {n_pairs}\n"
    f"Koefisien rho         :  {rho_all:.4f}\n"
    f"p-value               :  {p_all:.6f}\n"
    f"Signifikansi          :  {sig_label}\n"
    f"Arah                  :  {direction}\n"
    f"Kekuatan Korelasi     :  {strength}"
)
ax.text(0.05, 0.95, card_text, transform=ax.transAxes, verticalalignment="top",
        bbox=dict(boxstyle="square,pad=0.7", facecolor="white", edgecolor="black"))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_2_spearman_overall.png")
plt.close()


# 3. SPEARMAN CORRELATION PER WILAYAH
results_per_w: list[dict] = []
for wilayah, grp in df_pair.groupby("wilayah"):
    grp_clean = grp[[COL_CH, COL_PADI]].dropna()
    if len(grp_clean) < 4:
        continue
    x_w = grp_clean[COL_CH].to_numpy(dtype=float)
    y_w = grp_clean[COL_PADI].to_numpy(dtype=float)
    res_w  = stats.spearmanr(x_w, y_w)
    rho_w  = float(res_w.statistic) # type: ignore[attr-defined]
    p_w    = float(res_w.pvalue)    # type: ignore[attr-defined]
    results_per_w.append({
        "wilayah"     : wilayah,
        "n_bulan"     : len(grp_clean),
        "Spearman_rho": round(rho_w, 4),
        "p_value"     : round(p_w, 4),
        "Signifikan"  : "Ya" if p_w < 0.05 else "Tidak",
        "Arah"        : "Positif" if rho_w > 0 else "Negatif",
    })

spearman_per_w = (
    pd.DataFrame(results_per_w)
    .sort_values("Spearman_rho", ascending=False)
    .reset_index(drop=True)
)

print("3. Spearman Correlation per Wilayah")
print(spearman_per_w.to_string(index=False))
print()

spearman_per_w.to_csv(f"{OUT_DIR}/combined_3_spearman_per_wilayah.csv", index=False)

rho_vals   = spearman_per_w["Spearman_rho"].to_numpy(dtype=float)
colors_rho = ["#00b894" if r > 0 else "#d63031" for r in rho_vals]

fig, ax = plt.subplots(figsize=(9, 13))
bars = ax.barh(spearman_per_w["wilayah"].tolist(), rho_vals, color=colors_rho, edgecolor="white", height=0.72)
ax.axvline(0, color="#2d3436", linewidth=1.5, linestyle="-")
ax.axvline(rho_all, color="#6c5ce7", linewidth=1.5, linestyle="--", label=f"Rata-rata overall ρ = {rho_all:.3f}")

for bar, rho_v, sig in zip(bars, rho_vals, spearman_per_w["Signifikan"]):
    offset = 0.012 if rho_v >= 0 else -0.012
    ha     = "left"  if rho_v >= 0 else "right"
    marker = "*"     if sig == "Ya" else ""
    ax.text(rho_v + offset, bar.get_y() + bar.get_height() / 2, f"{rho_v:.3f}{marker}", va="center", ha=ha)

legend_patches = [
    mpatches.Patch(color="#00b894", label="Positif (CH ↑ → Produktivitas ↑)"),
    mpatches.Patch(color="#d63031", label="Negatif (CH ↑ → Produktivitas ↓)"),
]
ax.legend(handles=legend_patches + [ax.lines[1]], loc="lower right")
ax.set_title("Korelasi Spearman per Wilayah (* = signifikan p<0.05)", fontweight="bold")
ax.set_xlabel("Spearman ρ")
ax.set_xlim(-1.1, 1.1)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_3_spearman_per_wilayah.png")
plt.close()


# 4. 2x2 RESILIENCE CLASSIFICATION
df_nz = df_raw[df_raw[COL_PADI] > 0].dropna(subset=[COL_PADI]).copy()
df_nz["wilayah"] = (
    df_nz[COL_NAME] + " "
    + df_nz[COL_JENIS].map({0: "Kab.", 1: "Kota"})
)

cv_table = (
    df_nz.groupby("wilayah")[COL_PADI]
    .agg(
        mean_p="mean",
        std_p=lambda x: float(np.std(x.to_numpy(dtype=float), ddof=1)),
    )
    .reset_index()
)

cv_table["CV_ratio"] = cv_table["std_p"] / cv_table["mean_p"]
cv_table["CV (%)"]   = cv_table["CV_ratio"] * 100
cv_table["Resilience_Index"] = 1 / (cv_table["CV_ratio"] ** 2)

median_resilience = float(cv_table["Resilience_Index"].median())
median_cv_pct = float(cv_table["CV (%)"].median())

cv_table["Stabilitas"] = cv_table["Resilience_Index"].apply(
    lambda x: "Stabil" if x >= median_resilience else "Tidak Stabil"
)

resilience = spearman_per_w.merge(
    cv_table[["wilayah", "CV (%)", "Resilience_Index", "Stabilitas"]], 
    on="wilayah", 
    how="left"
)

def classify(row: pd.Series) -> str:
    stabil = row["Stabilitas"] == "Stabil"
    rho_positif = float(row["Spearman_rho"]) > 0.0
    is_significant = float(row["p_value"]) < 0.05
    responsif = rho_positif and is_significant
    
    if stabil and responsif:
        return "Tangguh"
    if (not stabil) and responsif:
        return "Rentan Produktif"
    if stabil and (not responsif):
        return "Tangguh Mandiri"
    return "Rentan"

resilience["Klasifikasi Ketahanan"] = resilience.apply(classify, axis=1)
resilience = resilience.sort_values(
    ["Klasifikasi Ketahanan", "Resilience_Index"], ascending=[True, False]
).reset_index(drop=True)

print("4. 2x2 Resilience Classification Table")
print(f"   CV Median Threshold         : {median_cv_pct:.2f}%")
print(f"   Resilience Index Threshold  : {median_resilience:.2f}")
print(f"   Responsiveness Rule         : Spearman rho > 0.0 AND p_value < 0.05\n")
print(resilience[["wilayah", "CV (%)", "Resilience_Index", "Stabilitas", "Spearman_rho", "p_value", "Klasifikasi Ketahanan"]].to_string(index=False))
print()

count_class = resilience["Klasifikasi Ketahanan"].value_counts()
print("   Summary:")
for cat in ["Tangguh", "Tangguh Mandiri", "Rentan Produktif", "Rentan"]:
    print(f"   {cat:<20}: {count_class.get(cat, 0)} wilayah")
print()

resilience.to_csv(f"{OUT_DIR}/combined_4_resilience_table.csv", index=False)

q_tangguh     = resilience.loc[resilience["Klasifikasi Ketahanan"] == "Tangguh",          "wilayah"].tolist()
q_rentan_prod = resilience.loc[resilience["Klasifikasi Ketahanan"] == "Rentan Produktif", "wilayah"].tolist()
q_tangguh_m   = resilience.loc[resilience["Klasifikasi Ketahanan"] == "Tangguh Mandiri",  "wilayah"].tolist()
q_rentan      = resilience.loc[resilience["Klasifikasi Ketahanan"] == "Rentan",           "wilayah"].tolist()

max_rows = max(len(q_tangguh), len(q_rentan_prod), len(q_tangguh_m), len(q_rentan))
fig_h = max(11, 3.0 + max_rows * 0.38)

fig = plt.figure(figsize=(12, fig_h))

LEFT   = 0.30
RIGHT  = 0.97
BOTTOM = 0.04
TOP    = 0.82

MID_X = (LEFT + RIGHT) / 2
MID_Y = (BOTTOM + TOP)  / 2

outer = mpatches.Rectangle(
    (LEFT, BOTTOM), RIGHT - LEFT, TOP - BOTTOM,
    transform=fig.transFigure, figure=fig,
    facecolor="none", edgecolor="black", linewidth=1.2, clip_on=False,
)
fig.add_artist(outer)

fig.add_artist(Line2D([MID_X, MID_X], [BOTTOM, TOP], transform=fig.transFigure, figure=fig, color="black", linewidth=1.2))
fig.add_artist(Line2D([LEFT, RIGHT], [MID_Y, MID_Y], transform=fig.transFigure, figure=fig, color="black", linewidth=1.2))

def _draw_matrix_text(fig_obj, x, y, items):
    text = "\n".join(items) if items else "(tidak ada)"
    fig_obj.text(x, y, text, ha="center", va="center", transform=fig_obj.transFigure, linespacing=1.6)

cx_left  = (LEFT  + MID_X) / 2
cx_right = (MID_X + RIGHT) / 2
cy_top   = (MID_Y + TOP)   / 2
cy_bot   = (BOTTOM + MID_Y) / 2

_draw_matrix_text(fig, cx_left,  cy_top, q_tangguh)
_draw_matrix_text(fig, cx_right, cy_top, q_rentan_prod)
_draw_matrix_text(fig, cx_left,  cy_bot, q_tangguh_m)
_draw_matrix_text(fig, cx_right, cy_bot, q_rentan)

col_hdr_y = TOP + 0.025
fig.text(cx_left,  col_hdr_y, f"CV ≤ {median_cv_pct:.2f}%\n(Stabil)", ha="center", va="bottom")
fig.text(cx_right, col_hdr_y, f"CV > {median_cv_pct:.2f}%\n(Tidak Stabil)", ha="center", va="bottom")

fig.text(MID_X, TOP + 0.13, "Koefisien Variasi (CV)", ha="center", va="bottom", fontweight="bold")

row_label_x = LEFT - 0.02
fig.text(row_label_x, cy_top, "ρ > 0,\np < 0,05", ha="right", va="center")
fig.text(row_label_x, cy_bot, "ρ ≤ 0 atau\np ≥ 0,05", ha="right", va="center")

fig.text(0.04, MID_Y, "Korelasi Spearman (ρ)", ha="center", va="center", fontweight="bold", rotation=90)

plt.savefig(f"{OUT_DIR}/combined_4_resilience_matrix.png")
plt.close()


# 4B. CLASSIFICATION TABLE VISUAL
table_rows = []
for cls in ["Tangguh", "Tangguh Mandiri", "Rentan Produktif", "Rentan"]:
    subset = resilience[resilience["Klasifikasi Ketahanan"] == cls].copy()
    subset = subset.sort_values("Resilience_Index", ascending=False)
    for _, row in subset.iterrows():
        table_rows.append([
            row["wilayah"],
            cls,
            f"{float(row['CV (%)']):.2f}%",
            f"{float(row['Spearman_rho']):.4f}",
            f"{float(row['p_value']):.4f}",
        ])

col_labels  = ["Wilayah", "Klasifikasi Ketahanan", "CV (%)", "Spearman ρ", "p-value"]
n_rows      = len(table_rows)
row_h       = 0.38
header_h    = 0.55
fig_height  = header_h + n_rows * row_h + 1.0

fig, ax = plt.subplots(figsize=(11, fig_height))
ax.axis("off")

col_x     = [0.0, 0.28, 0.54, 0.68, 0.82]
col_w     = [0.28, 0.26, 0.14, 0.14, 0.18]
col_align = ["left", "left", "center", "center", "center"]

y_top  = 0.97
y_curr = y_top
cell_h = (1.0 - 0.06) / (n_rows + 1)

ax.axhline(y_top, color="black", linewidth=1.2, xmin=0, xmax=1)
ax.axhline(y_top - cell_h, color="black", linewidth=1.2, xmin=0, xmax=1)

for i, (lbl, xp, align) in enumerate(zip(col_labels, col_x, col_align)):
    ha = align
    x_pos = xp + (col_w[i] / 2 if ha == "center" else 0.01)
    ax.text(x_pos, y_top - cell_h / 2, lbl, ha=ha, va="center", fontweight="bold", transform=ax.transAxes)

for r_idx, row_data in enumerate(table_rows):
    y_row = y_top - cell_h * (r_idx + 1)
    y_mid = y_row - cell_h / 2
    for i, (val, xp, align) in enumerate(zip(row_data, col_x, col_align)):
        ha = align
        x_pos = xp + (col_w[i] / 2 if ha == "center" else 0.01)
        ax.text(x_pos, y_mid, val, ha=ha, va="center", transform=ax.transAxes)

y_bottom = y_top - cell_h * (n_rows + 1)
ax.axhline(y_bottom, color="black", linewidth=1.2, xmin=0, xmax=1)

ax.set_title("Klasifikasi Ketahanan Pangan per Wilayah di Jawa Timur pada Tahun 2024", fontweight="bold")

plt.savefig(f"{OUT_DIR}/combined_4_resilience_wilayah_table.png")
plt.close()


# 5. DUAL-AXIS MONTHLY MEDIAN LINE PLOT
MONTH_LABELS_5 = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                   "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
COLOR_CH   = "#0009b8"
COLOR_PADI = "#00b894"

monthly_ch5 = (
    df_raw.dropna(subset=[COL_CH])
    .assign(wilayah=lambda d: d[COL_NAME] + " " + d[COL_JENIS].map({0: "Kab.", 1: "Kota"}))
    .groupby([COL_BULAN, "wilayah"])[COL_CH]
    .median()
    .reset_index()
    .rename(columns={COL_CH: "ch_median"})
    .groupby(COL_BULAN)["ch_median"]
    .agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
    .reindex(range(1, 13))
)

df_nz5 = df_raw[df_raw[COL_PADI] > 0].dropna(subset=[COL_PADI]).copy()
monthly_padi5 = (
    df_nz5.groupby(COL_BULAN)[COL_PADI]
    .agg(median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
    .reindex(range(1, 13))
)

x5    = list(range(1, 13))
ch_md = monthly_ch5["median"].to_numpy(dtype=float)
ch_q1 = monthly_ch5["q1"].to_numpy(dtype=float)
ch_q3 = monthly_ch5["q3"].to_numpy(dtype=float)
pd_md = monthly_padi5["median"].to_numpy(dtype=float)
pd_q1 = monthly_padi5["q1"].to_numpy(dtype=float)
pd_q3 = monthly_padi5["q3"].to_numpy(dtype=float)

fig, ax1 = plt.subplots(figsize=(11, 5))

ax1.axvspan(4.5,  10.5, alpha=0.15, color="#e17055")
ax1.axvspan(10.5, 12.5, alpha=0.15, color="#0062b8")
ax1.axvspan(0.5,   4.5, alpha=0.15, color="#0062b8")

ax1.fill_between(x5, ch_q1, ch_q3, alpha=0.15, color=COLOR_CH, label="Rentang Interkuartil (Q1 hingga Q3)")
ax1.plot(x5, ch_md, marker="s", color=COLOR_CH, linewidth=2.5, markersize=8, zorder=5,
         label="Median Curah Hujan Bulanan")
for xi, yi in zip(x5, ch_md):
    ax1.annotate(f"{yi:.0f}", xy=(xi, yi), xytext=(0, 10),
                 textcoords="offset points", ha="center", color="#2d3436")

ax1.set_xlabel("Bulan")
ax1.set_ylabel("Curah Hujan Median (mm/bulan)")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_xticks(x5)
ax1.set_xticklabels(MONTH_LABELS_5)
ax1.set_ylim(0, float(np.nanmax(ch_q3)) * 2.0)
ax1.set_xlim(0.5, 12.5)

ax2 = ax1.twinx()
ax2.fill_between(x5, pd_q1, pd_q3, alpha=0.15, color=COLOR_PADI, label="Rentang Interkuartil (Q1 hingga Q3)")
ax2.plot(x5, pd_md, marker="s", color=COLOR_PADI, linewidth=2.5, markersize=8, zorder=5,
         label="Median Produktivitas Padi Bulanan")
for xi, yi in zip(x5, pd_md):
    ax2.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 10),
                 textcoords="offset points", ha="center", color="#2d3436")

ax2.set_ylabel("Produktivitas Padi (ton/ha)")
ax2.tick_params(axis="y", labelcolor="black")
ax2.set_ylim(float(np.nanmin(pd_q1)) - 1.0, float(np.nanmax(pd_q3)) + 1.5)

handles_ch = [
    mpatches.Patch(color=COLOR_CH,   alpha=0.3, label="Rentang Interkuartil Curah Hujan"),
    Line2D([0], [0], color=COLOR_CH,   marker="s", linewidth=2.5, markersize=7, label="Median Curah Hujan Bulanan"),
]
handles_padi = [
    mpatches.Patch(color=COLOR_PADI, alpha=0.3, label="Rentang Interkuartil Produktivitas"),
    Line2D([0], [0], color=COLOR_PADI, marker="s", linewidth=2.5, markersize=7, label="Median Produktivitas Padi Bulanan"),
]
handles_musim = [
    mpatches.Patch(color="#e17055", alpha=0.3, label="Musim Kemarau (Mei hingga Oktober)"),
    mpatches.Patch(color="#0062b8", alpha=0.3, label="Musim Hujan (November hingga April)"),
]
ax1.legend(handles=handles_ch + handles_padi + handles_musim, loc="upper right")

ax1.set_title(
    "Pola Curah Hujan dan Produktivitas Padi Median Bulanan di Jawa Timur pada Tahun 2024",
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_5_dual_axis_monthly_median.png")
plt.close()