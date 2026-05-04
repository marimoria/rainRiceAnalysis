"""
data_processor.py
─────────────────
Combines raw Curah Hujan (rainfall) and Padi (rice) Excel files
from East Java (Jawa Timur) into a single tidy CSV and/or plain
Excel file ready for analysis or upload.

Usage
─────
  python data_processor.py                          # interactive (uses __data__/ defaults)
  python data_processor.py --ch <file> --padi <file>
  python data_processor.py --ch <file> --padi <file> --out-dir __data__/out/
  python data_processor.py --ch <file> --padi <file> --format csv
  python data_processor.py --ch <file> --padi <file> --format xlsx
  python data_processor.py --ch <file> --padi <file> --format both
  python data_processor.py --preview                # dry-run, print table

Default paths (when run interactively or omitted):
  CH   : __data__/curah_hujan_dataset.xlsx
  Padi : __data__/padi_dataset.xlsx
  Out  : __data__/out/ch_padi_training_dataset.{csv,xlsx}
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ─── constants ────────────────────────────────────────────────────────────────

BULAN_MAP = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12,
}

OUTPUT_COLUMNS = [
    "nama_wilayah",
    "bulan",
    "jenis_wilayah",
    "curah_hujan_per_bulan",
    "hari_hujan_per_bulan",
    "luas_panen",
    "produksi_gkg",
    "produktivitas",
]

BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "__data__"
DEFAULT_CH      = DATA_DIR / "curah_hujan_dataset.xlsx"
DEFAULT_PADI    = DATA_DIR / "padi_dataset.xlsx"
DEFAULT_OUT_DIR = DATA_DIR / "out"
DEFAULT_STEM    = "ch_padi_training_dataset"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sep(title: str = "", width: int = 56):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _parse_sheet_name(sheet_name: str) -> tuple[str, int]:
    """Return (nama_wilayah, jenis_wilayah) from the sheet name.

    'Kabupaten Bangkalan' → ('Bangkalan', 0)
    'Kota Batu'           → ('Batu', 1)
    """
    name = sheet_name.strip()
    if name.lower().startswith("kabupaten "):
        area = name[len("kabupaten "):].strip().title()
        return area, 0
    elif name.lower().startswith("kota "):
        area = name[len("kota "):].strip().title()
        return area, 1
    # Fallback: keep as-is, jenis unknown → 0
    return name.title(), 0


def _find_header_row(df_raw: pd.DataFrame) -> int | None:
    """Find the first row where column 0 contains 'bulan' (case-insensitive)."""
    for i, val in enumerate(df_raw.iloc[:, 0]):
        if "bulan" in str(val).strip().lower():
            return i
    return None


def _read_ch_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse a Curah Hujan sheet; returns [bulan, curah_hujan, hari_hujan]."""
    hr = _find_header_row(df_raw)
    if hr is None:
        return pd.DataFrame(columns=["bulan", "curah_hujan_per_bulan", "hari_hujan_per_bulan"])

    df = df_raw.iloc[hr + 1:].copy().reset_index(drop=True)
    df.columns = ["bulan_raw", "curah_hujan_per_bulan", "hari_hujan_per_bulan",
                  *[f"_extra{i}" for i in range(len(df.columns) - 3)]]
    df = df[["bulan_raw", "curah_hujan_per_bulan", "hari_hujan_per_bulan"]].dropna(subset=["bulan_raw"])
    df["bulan"] = df["bulan_raw"].astype(str).str.strip().map(BULAN_MAP)
    df = df.dropna(subset=["bulan"])
    df["bulan"] = df["bulan"].astype(int)
    df["curah_hujan_per_bulan"] = pd.to_numeric(df["curah_hujan_per_bulan"], errors="coerce")
    df["hari_hujan_per_bulan"]  = pd.to_numeric(df["hari_hujan_per_bulan"],  errors="coerce")
    return df[["bulan", "curah_hujan_per_bulan", "hari_hujan_per_bulan"]].reset_index(drop=True)


def _read_padi_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse a Padi sheet; returns [bulan, luas_panen, produksi_gkg, produktivitas]."""
    if df_raw.empty:
        return pd.DataFrame(columns=["bulan", "luas_panen", "produksi_gkg", "produktivitas"])

    hr = _find_header_row(df_raw)
    if hr is None:
        return pd.DataFrame(columns=["bulan", "luas_panen", "produksi_gkg", "produktivitas"])

    df = df_raw.iloc[hr + 1:].copy().reset_index(drop=True)
    df.columns = ["bulan_raw", "luas_panen", "produksi_gkg", "produktivitas",
                  *[f"_extra{i}" for i in range(len(df.columns) - 4)]]
    df = df[["bulan_raw", "luas_panen", "produksi_gkg", "produktivitas"]].dropna(subset=["bulan_raw"])
    df["bulan"] = df["bulan_raw"].astype(str).str.strip().map(BULAN_MAP)
    df = df.dropna(subset=["bulan"])
    df["bulan"] = df["bulan"].astype(int)
    df["luas_panen"]   = pd.to_numeric(df["luas_panen"],   errors="coerce")
    df["produksi_gkg"] = pd.to_numeric(df["produksi_gkg"], errors="coerce")
    df["produktivitas"] = pd.to_numeric(df["produktivitas"], errors="coerce")
    return df[["bulan", "luas_panen", "produksi_gkg", "produktivitas"]].reset_index(drop=True)


# ─── core ─────────────────────────────────────────────────────────────────────

def build_combined_df(ch_file: Path, padi_file: Path, verbose: bool = True) -> pd.DataFrame:
    """Read both Excel files and return one tidy DataFrame."""
    if verbose:
        print(f"  Reading: {ch_file.name}")
    ch_sheets = {k.strip(): v
                 for k, v in pd.read_excel(ch_file, sheet_name=None, header=None).items()}

    if verbose:
        print(f"  Reading: {padi_file.name}")
    padi_sheets = {k.strip(): v
                   for k, v in pd.read_excel(padi_file, sheet_name=None, header=None).items()}

    if verbose:
        print(f"\n  Found {len(ch_sheets)} CH sheets / {len(padi_sheets)} Padi sheets\n")

    rows: list[pd.DataFrame] = []
    skipped = []

    for sheet in sorted(ch_sheets.keys()):
        nama, jenis = _parse_sheet_name(sheet)
        ch_df   = _read_ch_sheet(ch_sheets[sheet])
        padi_df = _read_padi_sheet(padi_sheets.get(sheet, pd.DataFrame()))

        merged = pd.merge(ch_df, padi_df, on="bulan", how="outer").sort_values("bulan")
        if merged.empty:
            skipped.append(sheet)
            continue

        merged.insert(0, "nama_wilayah", nama)
        merged.insert(1, "jenis_wilayah", jenis)
        rows.append(merged)

        if verbose:
            print(f"  ✓  {sheet:<35}  {len(merged):>2} baris")

    if skipped and verbose:
        print(f"\n  ⚠  Skipped (empty): {skipped}")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined[OUTPUT_COLUMNS]
    return combined.sort_values(["nama_wilayah", "jenis_wilayah", "bulan"]).reset_index(drop=True)


def save_outputs(df: pd.DataFrame, out_dir: Path, fmt: str, stem: str = DEFAULT_STEM):
    """Save combined df to csv and/or plain xlsx (no styling)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    if fmt in ("csv", "both"):
        path = out_dir / f"{stem}.csv"
        df.to_csv(path, index=False)
        saved.append(path)
        print(f"  ✓  CSV  → {path}")

    if fmt in ("xlsx", "both"):
        path = out_dir / f"{stem}.xlsx"
        df.to_excel(path, index=False)
        saved.append(path)
        print(f"  ✓  XLSX → {path}")

    return saved


def print_summary(df: pd.DataFrame):
    _sep("Summary")
    total = len(df)
    # Fix: count unique wilayah by (nama_wilayah, jenis_wilayah) pair, not nama alone
    n_wilayah = df[["nama_wilayah", "jenis_wilayah"]].drop_duplicates().shape[0]
    kabupaten = (df["jenis_wilayah"] == 0).sum()
    kota      = (df["jenis_wilayah"] == 1).sum()
    print(f"  Total rows      : {total:,}")
    print(f"  Unique wilayah  : {n_wilayah}")
    print(f"  Rows kabupaten  : {kabupaten:,}  (jenis_wilayah=0)")
    print(f"  Rows kota       : {kota:,}  (jenis_wilayah=1)")
    missing = df[OUTPUT_COLUMNS[3:]].isnull().sum()
    if missing.any():
        print(f"\n  Missing values:")
        for col, n in missing[missing > 0].items():
            print(f"    {col}: {n}")
    _sep()


# ─── interactive menu ─────────────────────────────────────────────────────────

def _prompt_file(prompt: str, must_exist: bool = True) -> Path:
    while True:
        raw = input(f"  {prompt}: ").strip()
        if not raw:
            print("  ✗ Path cannot be empty.")
            continue
        p = Path(raw)
        if must_exist and not p.exists():
            print(f"  ✗ File not found: {p}")
            continue
        return p


def _prompt_choice(prompt: str, options: list[str]) -> str:
    opts_str = " / ".join(f"[{o}]" for o in options)
    while True:
        val = input(f"  {prompt} {opts_str}: ").strip().lower()
        if val in options:
            return val
        print(f"  ✗ Please enter one of: {', '.join(options)}")


def interactive_menu():
    _sep("Jatim Curah Hujan & Padi — Data Processor")
    print("  Combines Curah Hujan + Padi Excel files into CSV / XLSX.\n")

    ch_raw = input(f"  Path to Curah Hujan Excel  (default: {DEFAULT_CH}): ").strip()
    ch_file = Path(ch_raw) if ch_raw else DEFAULT_CH
    if not ch_file.exists():
        print(f"  ✗ File not found: {ch_file}"); return

    padi_raw = input(f"  Path to Padi Excel         (default: {DEFAULT_PADI}): ").strip()
    padi_file = Path(padi_raw) if padi_raw else DEFAULT_PADI
    if not padi_file.exists():
        print(f"  ✗ File not found: {padi_file}"); return

    out_dir_raw = input(f"  Output directory           (default: {DEFAULT_OUT_DIR}): ").strip()
    out_dir = Path(out_dir_raw) if out_dir_raw else DEFAULT_OUT_DIR

    fmt = _prompt_choice("Output format", ["csv", "xlsx", "both"])

    _sep("Processing")
    df = build_combined_df(ch_file, padi_file, verbose=True)
    save_outputs(df, out_dir, fmt)
    print_summary(df)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="data_processor",
        description="Combine Curah Hujan + Padi Excel files into a tidy dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ch",      metavar="FILE", help="Path to Curah Hujan Excel file")
    p.add_argument("--padi",    metavar="FILE", help="Path to Padi Excel file")
    p.add_argument("--out-dir", metavar="DIR",  default=str(DEFAULT_OUT_DIR),
                   help="Output directory (default: output/)")
    p.add_argument("--format",  choices=["csv", "xlsx", "both"], default="both",
                   help="Output format (default: both)")
    p.add_argument("--stem",    default=DEFAULT_STEM,
                   help="Output filename stem (default: Data_Gabungan_Curah_Hujan_Padi)")
    p.add_argument("--preview", action="store_true",
                   help="Print first 20 rows then exit without saving")
    p.add_argument("--quiet",   action="store_true",
                   help="Suppress per-sheet progress output")
    return p.parse_args()


def main():
    args = parse_args()

    # No arguments → interactive
    if not args.ch and not args.padi and not args.preview:
        interactive_menu()
        return

    # Both files required unless just --help
    if not args.ch or not args.padi:
        print("✗  Both --ch and --padi are required when running non-interactively.")
        print("   Run without arguments for the interactive menu.")
        sys.exit(1)

    ch_file   = Path(args.ch)
    padi_file = Path(args.padi)

    for f in (ch_file, padi_file):
        if not f.exists():
            print(f"✗  File not found: {f}")
            sys.exit(1)

    _sep("Processing")
    df = build_combined_df(ch_file, padi_file, verbose=not args.quiet)

    if args.preview:
        print(df.head(20).to_string(index=False))
        print_summary(df)
        return

    save_outputs(df, Path(args.out_dir), args.format, args.stem)
    print_summary(df)


if __name__ == "__main__":
    main()