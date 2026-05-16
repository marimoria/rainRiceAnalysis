"""
data_processor.py
-------------------
Combines Curah Hujan (rainfall) and Padi (rice) CSV files
into a single combined CSV dataset.

Usage
-------------------
  python data_processor.py                          # uses __data__/ defaults
  python data_processor.py --ch my_ch.csv --padi my_padi.csv --out combined.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "__data__"

def parse_args():
    p = argparse.ArgumentParser(
        prog="data_processor",
        description="Combine Curah Hujan + Padi CSV files into a tidy dataset.",
    )
    p.add_argument("--ch", default=str(DATA_DIR / "curah_hujan_dataset.csv"), 
                   help="Path to Curah Hujan CSV file")
    p.add_argument("--padi", default=str(DATA_DIR / "padi_dataset.csv"), 
                   help="Path to Padi CSV file")
    p.add_argument("--out", default=str(DATA_DIR / "ch_padi_training_dataset.csv"), 
                   help="Output combined CSV filename")
    return p.parse_args()

def _sep(title: str = "", width: int = 56):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)

def main():
    args = parse_args()
    
    ch_file = Path(args.ch)
    padi_file = Path(args.padi)
    
    for f in (ch_file, padi_file):
        if not f.exists():
            print(f"✗ File not found: {f}")
            print(f"  Make sure you've downloaded it to the {DATA_DIR.name}/ folder.")
            sys.exit(1)

    _sep("Processing")
    print(f"  Reading: {ch_file.name}")
    ch_df = pd.read_csv(ch_file)
    
    print(f"  Reading: {padi_file.name}")
    padi_df = pd.read_csv(padi_file)

    # Merge the two datasets on the common columns
    merged = pd.merge(
        ch_df, 
        padi_df, 
        on=["nama_wilayah", "jenis_wilayah", "bulan"], 
        how="outer"
    )
    
    # Sort for tidiness
    merged = merged.sort_values(["nama_wilayah", "jenis_wilayah", "bulan"]).reset_index(drop=True)

    # Save to output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True) # Ensure __data__ exists
    merged.to_csv(out_path, index=False)
    print(f"  ✓ Saved combined data to → {out_path}")

    _sep("Summary")
    n_wilayah = merged[["nama_wilayah", "jenis_wilayah"]].drop_duplicates().shape[0]
    kabupaten = (merged["jenis_wilayah"] == 0).sum()
    kota      = (merged["jenis_wilayah"] == 1).sum()
    
    print(f"  Total rows      : {len(merged):,}")
    print(f"  Unique wilayah  : {n_wilayah}")
    print(f"  Rows kabupaten  : {kabupaten:,}  (jenis_wilayah=0)")
    print(f"  Rows kota       : {kota:,}  (jenis_wilayah=1)")
    
    missing = merged.isnull().sum()
    if missing.any():
        print(f"\n  Missing values:")
        for col, n in missing[missing > 0].items():
            print(f"    {col}: {n}")
    _sep()

if __name__ == "__main__":
    main()