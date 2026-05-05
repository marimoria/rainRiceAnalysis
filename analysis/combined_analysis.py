"""
03_combined_analysis.py
=======================
COMBINED — Rain × Padi Relationship Analysis
---------------------------------------------
PRIMARY outputs (for journal):
  1.  LOWESS scatter plot                  → combined_1_lowess_scatter.png
  2.  Spearman correlation overall         → combined_2_spearman_overall.csv / .png
  3.  Spearman correlation per wilayah     → combined_3_spearman_per_wilayah.csv / .png
  4.  2×2 Resilience classification table → combined_4_resilience_table.csv / .png
                                            combined_4_resilience_matrix.png
                                            combined_4_resilience_wilayah_table.png

All outputs → __output__/combined/
"""

import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH = "__data__/ch_padi_training_dataset.csv"
OUT_DIR   = "__output__/combined"

COL_NAME  = "nama_wilayah"
COL_JENIS = "jenis_wilayah"
COL_BULAN = "bulan"
COL_CH    = "curah_hujan_per_bulan"
COL_PADI  = "produktivitas"

# rho > 0 → Positif/Responsif
SPEARMAN_THRESHOLD: float = 0.0

# Colors for resilience quadrants
Q_COLORS = {
    "Tangguh"          : "#00b894",   # low CV  + positive rho
    "Rentan Produktif" : "#fdcb6e",   # high CV + positive rho
    "Tangguh Mandiri"  : "#74b9ff",   # low CV  + negative/weak rho
    "Rentan"           : "#d63031",   # high CV + negative/weak rho
}

ACCENT = "#6c5ce7"

# ── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.28,
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

# Drop rows missing either variable
df = df_raw.dropna(subset=[COL_CH, COL_PADI]).copy()
# For correlation: keep only rows where both variables are meaningful
df_pair = df[df[COL_PADI] > 0].copy()

# Clip extreme CH outliers (>P99) to keep LOWESS / scatter readable
p99_ch   = float(df_pair[COL_CH].quantile(0.99))
df_clean = df_pair[df_pair[COL_CH] <= p99_ch].copy()

print("=" * 65)
print("  03_COMBINED_ANALYSIS.PY")
print("=" * 65)
print(f"  Rows (both vars present, produktivitas>0) : {len(df_pair)}")
print(f"  Rows after clipping CH >P99               : {len(df_clean)}")
print(f"  Unique wilayah                            : {df_pair['wilayah'].nunique()}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOWESS SCATTER PLOT
# ═══════════════════════════════════════════════════════════════════════════════
# .to_numpy(dtype=float) → pure ndarray, eliminates ExtensionArray Pylance errors
x_raw = df_clean[COL_CH].to_numpy(dtype=float)
y_raw = df_clean[COL_PADI].to_numpy(dtype=float)

lw = lowess(y_raw, x_raw, frac=0.35, it=3, return_sorted=True)

print("── 1. LOWESS Scatter Plot ──")
print(f"   LOWESS frac = 0.35  |  n = {len(x_raw)} observations")
print()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_raw, y_raw,
           alpha=0.28, s=20, color=ACCENT,
           edgecolors="none", label="Observasi")
ax.plot(lw[:, 0], lw[:, 1],
        color="#d63031", linewidth=2.8, label="LOWESS (frac=0.35)", zorder=5)
ax.set_title(
    "Hubungan Curah Hujan dan Produktivitas Padi\n(LOWESS Regression Line)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.set_xlabel("Curah Hujan (mm/bulan)", fontsize=11)
ax.set_ylabel("Produktivitas Padi (ton/ha)", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_1_lowess_scatter.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/combined_1_lowess_scatter.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SPEARMAN CORRELATION — OVERALL
# ═══════════════════════════════════════════════════════════════════════════════
ch_pair   = df_pair[COL_CH].to_numpy(dtype=float)
padi_pair = df_pair[COL_PADI].to_numpy(dtype=float)
rho_result = stats.spearmanr(ch_pair, padi_pair)
rho_all    = float(rho_result.statistic)   # type: ignore[attr-defined]
p_all      = float(rho_result.pvalue)      # type: ignore[attr-defined]

# Interpretation
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

print("── 2. Spearman Correlation — Overall ──")
print(f"   n pairs      : {n_pairs}")
print(f"   Spearman rho : {rho_all:.4f}")
print(f"   p-value      : {p_all:.6f}")
print(f"   Significance : {sig_label}")
print(f"   Direction    : {direction}")
print(f"   Strength     : {strength}")
print()

overall_df = pd.DataFrame([{
    "n_pairs"     : n_pairs,
    "Spearman_rho": round(rho_all, 4),
    "p_value"     : round(p_all, 6),
    "Signifikansi": sig_label,
    "Arah"        : direction,
    "Kekuatan"    : strength,
}])
overall_df.to_csv(f"{OUT_DIR}/combined_2_spearman_overall.csv", index=False)
print(f"   → Saved: {OUT_DIR}/combined_2_spearman_overall.csv")

# Visual summary card
fig, ax = plt.subplots(figsize=(7, 4))
ax.axis("off")
card_text = (
    f"Korelasi Spearman — Keseluruhan\n\n"
    f"n pasangan observasi  :  {n_pairs}\n"
    f"Koefisien rho (ρ)    :  {rho_all:.4f}\n"
    f"p-value               :  {p_all:.6f}\n"
    f"Signifikansi          :  {sig_label}\n"
    f"Arah                  :  {direction}\n"
    f"Kekuatan Korelasi     :  {strength}"
)
ax.text(0.05, 0.95, card_text,
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7",
                  facecolor="#dfe6e9", edgecolor="#b2bec3", linewidth=1.5))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_2_spearman_overall.png",
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"   → Saved: {OUT_DIR}/combined_2_spearman_overall.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPEARMAN CORRELATION PER WILAYAH
# ═══════════════════════════════════════════════════════════════════════════════
results_per_w: list[dict] = []
for wilayah, grp in df_pair.groupby("wilayah"):
    grp_clean = grp[[COL_CH, COL_PADI]].dropna()
    if len(grp_clean) < 4:
        continue
    x_w = grp_clean[COL_CH].to_numpy(dtype=float)
    y_w = grp_clean[COL_PADI].to_numpy(dtype=float)
    res_w  = stats.spearmanr(x_w, y_w)
    rho_w  = float(res_w.statistic)   # type: ignore[attr-defined]
    p_w    = float(res_w.pvalue)      # type: ignore[attr-defined]
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

print("── 3. Spearman Correlation per Wilayah ──")
print(spearman_per_w.to_string(index=False))
print()

spearman_per_w.to_csv(f"{OUT_DIR}/combined_3_spearman_per_wilayah.csv", index=False)
print(f"   → Saved: {OUT_DIR}/combined_3_spearman_per_wilayah.csv")

# Horizontal bar chart
rho_vals   = spearman_per_w["Spearman_rho"].to_numpy(dtype=float)
colors_rho = ["#00b894" if r > 0 else "#d63031" for r in rho_vals]

fig, ax = plt.subplots(figsize=(9, 13))
bars = ax.barh(
    spearman_per_w["wilayah"].tolist(),
    rho_vals,
    color=colors_rho,
    edgecolor="white",
    linewidth=0.5,
    height=0.72,
)
ax.axvline(0,       color="#2d3436", linewidth=1.5, linestyle="-")
ax.axvline(rho_all, color="#6c5ce7", linewidth=1.5,
           linestyle="--", label=f"Rata-rata overall ρ = {rho_all:.3f}")

for bar, rho_v, sig in zip(bars, rho_vals, spearman_per_w["Signifikan"]):
    offset = 0.012 if rho_v >= 0 else -0.012
    ha     = "left"  if rho_v >= 0 else "right"
    marker = "★"     if sig == "Ya" else ""
    ax.text(rho_v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{rho_v:.3f}{marker}", va="center", fontsize=7.5, ha=ha)

legend_patches = [
    mpatches.Patch(color="#00b894", label="Positif (CH ↑ → Produktivitas ↑)"),
    mpatches.Patch(color="#d63031", label="Negatif (CH ↑ → Produktivitas ↓)"),
]
ax.legend(handles=legend_patches + [ax.lines[1]], fontsize=8.5, loc="lower right")
ax.set_title("Korelasi Spearman per Wilayah\n(★ = signifikan p<0.05)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Spearman ρ", fontsize=11)
ax.set_xlim(-1.1, 1.1)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_3_spearman_per_wilayah.png", bbox_inches="tight")
plt.close()
print(f"   → Saved: {OUT_DIR}/combined_3_spearman_per_wilayah.png\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 2×2 RESILIENCE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Recompute CV per wilayah ──────────────────────────────────────────────────
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
cv_table["CV (%)"]    = cv_table["std_p"] / cv_table["mean_p"] * 100
median_cv              = float(cv_table["CV (%)"].median())
cv_table["Stabilitas"] = cv_table["CV (%)"].apply(
    lambda x: "Stabil" if x <= median_cv else "Tidak Stabil"
)

# ── Merge CV + Spearman ───────────────────────────────────────────────────────
resilience = spearman_per_w.merge(
    cv_table[["wilayah", "CV (%)", "Stabilitas"]], on="wilayah", how="left"
)

# ── 2×2 Classification ───────────────────────────────────────────────────────
def classify(row: pd.Series) -> str:
    stabil  = row["Stabilitas"] == "Stabil"
    positif = float(row["Spearman_rho"]) > SPEARMAN_THRESHOLD
    if stabil and positif:
        return "Tangguh"
    if (not stabil) and positif:
        return "Rentan Produktif"
    if stabil and (not positif):
        return "Tangguh Mandiri"
    return "Rentan"

resilience["Klasifikasi Ketahanan"] = resilience.apply(classify, axis=1)
resilience = resilience.sort_values(
    ["Klasifikasi Ketahanan", "CV (%)"]
).reset_index(drop=True)

print("── 4. 2×2 Resilience Classification Table ──")
print(f"   CV median threshold : {median_cv:.2f}%")
print(f"   Spearman threshold  : rho > {SPEARMAN_THRESHOLD} = Responsif")
print()
print(resilience[[
    "wilayah", "CV (%)", "Stabilitas", "Spearman_rho", "Arah",
    "Klasifikasi Ketahanan",
]].to_string(index=False))
print()

count_class = resilience["Klasifikasi Ketahanan"].value_counts()
print("   Summary:")
for cat in ["Tangguh", "Tangguh Mandiri", "Rentan Produktif", "Rentan"]:
    print(f"   {cat:<20}: {count_class.get(cat, 0)} wilayah")
print()

resilience.to_csv(f"{OUT_DIR}/combined_4_resilience_table.csv", index=False)
print(f"   → Saved: {OUT_DIR}/combined_4_resilience_table.csv")

# ── 2×2 Matrix Visual ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.axis("off")

quadrant_defs = [
    (0.5, 1.5, "TANGGUH MANDIRI",
     "Stabil + Korelasi Negatif\n(Produktivitas stabil,\ntidak bergantung CH)",
     "#74b9ff"),
    (1.5, 1.5, "TANGGUH",
     "Stabil + Korelasi Positif\n(Produktivitas stabil,\nresponsif terhadap CH)",
     "#00b894"),
    (0.5, 0.5, "RENTAN",
     "Tidak Stabil + Korelasi Negatif\n(Produktivitas fluktuatif,\ntidak responsif CH)",
     "#d63031"),
    (1.5, 0.5, "RENTAN PRODUKTIF",
     "Tidak Stabil + Korelasi Positif\n(Produktivitas fluktuatif,\nresponsif terhadap CH)",
     "#fdcb6e"),
]

for xc, yc, lbl, desc, color in quadrant_defs:
    rect = mpatches.FancyBboxPatch(
        (xc - 0.47, yc - 0.47), 0.94, 0.94,
        boxstyle="round,pad=0.03",
        facecolor=color, edgecolor="white", linewidth=2.5, alpha=0.85,
    )
    ax.add_patch(rect)
    txt_color = "white" if color in ("#d63031", "#00b894") else "#2d3436"
    ax.text(xc, yc + 0.18, lbl, ha="center", va="center",
            fontsize=11, fontweight="bold", color=txt_color)
    ax.text(xc, yc - 0.05, desc, ha="center", va="center",
            fontsize=8.5, color=txt_color)

    wil_in_q = resilience.loc[
        resilience["Klasifikasi Ketahanan"].str.upper() == lbl.upper(), "wilayah"
    ].tolist()
    wil_text = "\n".join(wil_in_q) if wil_in_q else "(tidak ada)"
    list_color = "white" if color in ("#d63031", "#00b894") else "#555"
    ax.text(xc, yc - 0.33, wil_text, ha="center", va="center",
            fontsize=6.5, color=list_color, style="italic")

ax.text(0.5, -0.04, "CV Rendah\n(Stabil)", ha="center", va="top",
        fontsize=11, fontweight="bold", color="#2d3436")
ax.text(1.5, -0.04, "CV Tinggi\n(Tidak Stabil)", ha="center", va="top",
        fontsize=11, fontweight="bold", color="#2d3436")
ax.text(-0.04, 1.5, "Korelasi\nPositif\n(Responsif)", ha="right", va="center",
        fontsize=11, fontweight="bold", color="#2d3436", rotation=90)
ax.text(-0.04, 0.5, "Korelasi\nNegatif/Nol\n(Tidak Responsif)", ha="right", va="center",
        fontsize=11, fontweight="bold", color="#2d3436", rotation=90)

ax.plot([1, 1], [0, 2], color="#b2bec3", linewidth=2)
ax.plot([0, 2], [1, 1], color="#b2bec3", linewidth=2)

ax.text(1.0, -0.10, "← Koefisien Variasi (CV) →",
        ha="center", va="top", fontsize=10, color="#636e72")
ax.text(-0.12, 1.0, "← Korelasi Spearman (ρ) →",
        ha="center", va="center", fontsize=10, color="#636e72", rotation=90)

ax.set_title(
    "Matriks Klasifikasi Ketahanan Pangan 2×2\n"
    "Berdasarkan Stabilitas Produktivitas (CV) × "
    "Sensitivitas Terhadap Curah Hujan (Spearman ρ)",
    fontsize=13, fontweight="bold", pad=16,
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_4_resilience_matrix.png",
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"   → Saved: {OUT_DIR}/combined_4_resilience_matrix.png")

# ── Wilayah list per class (table visual) ─────────────────────────────────────
class_order = ["Tangguh", "Tangguh Mandiri", "Rentan Produktif", "Rentan"]
class_colors_hex = {
    "Tangguh"          : "#00b894",
    "Tangguh Mandiri"  : "#74b9ff",
    "Rentan Produktif" : "#fdcb6e",
    "Rentan"           : "#d63031",
}

fig, axes = plt.subplots(1, 4, figsize=(16, 9))
for ax_i, cls in zip(axes, class_order):
    wil_cls = resilience[resilience["Klasifikasi Ketahanan"] == cls].copy()
    cell_data = [
        [w, f"{cv:.1f}", f"{rh:.3f}"]
        for w, cv, rh in zip(
            wil_cls["wilayah"].tolist(),
            wil_cls["CV (%)"].to_numpy(dtype=float),
            wil_cls["Spearman_rho"].to_numpy(dtype=float),
        )
    ]
    ax_i.axis("off")
    hdr_color = "white" if cls in ("Tangguh", "Rentan") else "#2d3436"
    ax_i.text(0.5, 1.01, cls,
              transform=ax_i.transAxes, ha="center", va="bottom",
              fontsize=11, fontweight="bold", color=hdr_color,
              bbox=dict(facecolor=class_colors_hex[cls], edgecolor="none",
                        boxstyle="round,pad=0.4"))
    ax_i.text(0.5, 0.97, f"n = {len(wil_cls)} wilayah",
              transform=ax_i.transAxes, ha="center", va="top",
              fontsize=9, color="#636e72")

    if cell_data:
        tbl = ax_i.table(
            cellText=cell_data,
            colLabels=["Wilayah", "CV (%)", "ρ"],
            cellLoc="center", loc="center",
            bbox=[0, 0, 1, 0.92],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        cell_hdr_color = class_colors_hex[cls]
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(cell_hdr_color)
                cell.set_text_props(color=hdr_color, fontweight="bold")
            else:
                cell.set_facecolor("#f8f9fa" if r % 2 == 0 else "white")
            cell.set_edgecolor("#dfe6e9")

fig.suptitle(
    "Klasifikasi Ketahanan Pangan per Wilayah — Jawa Timur\n"
    "(CV = stabilitas produksi  |  ρ = Spearman per wilayah)",
    fontsize=13, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/combined_4_resilience_wilayah_table.png",
            bbox_inches="tight", facecolor="white")
plt.close()
print(f"   → Saved: {OUT_DIR}/combined_4_resilience_wilayah_table.png")

print()
print("=" * 65)
print("  03_combined_analysis.py  — COMPLETE")
print(f"  All outputs in: {OUT_DIR}/")
print("=" * 65)