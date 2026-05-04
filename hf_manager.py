"""
hf_manager.py
─────────────
Manages HuggingFace dataset uploads and downloads for the
Jatim Curah Hujan & Padi dataset.

Usage
─────
  python hf_manager.py                  # interactive menu
  python hf_manager.py upload <files…>  # upload specific files
  python hf_manager.py download raw     # download raw Excel files
  python hf_manager.py download combined# download combined CSV + XLSX
  python hf_manager.py list             # list files in the repo

Environment
───────────
  HF_TOKEN   — HuggingFace access token (read from .env or shell env)
  DATA_DIR   — local data directory override (default: __data__/)
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

load_dotenv()

# ─── config ───────────────────────────────────────────────────────────────────

HF_TOKEN     = os.getenv("HF_TOKEN")
HF_REPO_ID   = "mariaamandadevina/jatim-curah-hujan-padi-2024"
HF_REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "__data__")))

CH_FILENAME       = "Data_Curah_Hujan.xlsx"
PADI_FILENAME     = "Data_Padi.xlsx"
COMBINED_CSV      = "Data_Gabungan_Curah_Hujan_Padi.csv"
COMBINED_XLSX     = "Data_Gabungan_Curah_Hujan_Padi.xlsx"

RAW_FILES      = [CH_FILENAME, PADI_FILENAME]
COMBINED_FILES = [COMBINED_CSV, COMBINED_XLSX]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _sep(title: str = "", width: int = 56):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def _require_token():
    if not HF_TOKEN:
        print("  ✗  HF_TOKEN not set. Add it to a .env file or export it as an env var.")
        sys.exit(1)


def _ensure_repo(api: HfApi):
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
        print(f"  ✓  Repo: https://huggingface.co/datasets/{HF_REPO_ID}")
    except RepositoryNotFoundError:
        print(f"  ℹ  Repo not found — creating {HF_REPO_ID} …")
        create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
                    private=False, token=HF_TOKEN)
        print(f"  ✓  Created: https://huggingface.co/datasets/{HF_REPO_ID}")


# ─── actions ──────────────────────────────────────────────────────────────────

def cmd_upload(files: list[Path]):
    """Upload a list of local files to the HuggingFace repo."""
    _require_token()
    api = HfApi()
    _ensure_repo(api)

    any_ok = False
    for local in files:
        if not local.exists():
            print(f"  ⚠  Not found (skipped): {local}")
            continue
        size_kb = local.stat().st_size / 1024
        print(f"  ↑  {local.name}  ({size_kb:.1f} KB) …", end="", flush=True)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=local.name,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=HF_TOKEN,
        )
        print(" ✓")
        any_ok = True

    if any_ok:
        print(f"\n  🔗  https://huggingface.co/datasets/{HF_REPO_ID}")


def cmd_download(filenames: list[str]):
    """Download a list of files from the HuggingFace repo into DATA_DIR."""
    _require_token()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        print(f"  ↓  {filename} …", end="", flush=True)
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN,
                local_dir=str(DATA_DIR),
            )
            print(f" ✓  → {DATA_DIR / filename}")
        except Exception as exc:
            print(f" ✗  {exc}")


def cmd_list():
    """List all files currently in the HuggingFace repo."""
    _require_token()
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID,
                                    repo_type=HF_REPO_TYPE, token=HF_TOKEN)
        file_list = list(files)
        if not file_list:
            print("  (repo is empty)")
        else:
            print(f"  Files in {HF_REPO_ID}:")
            for f in file_list:
                print(f"    • {f}")
    except RepositoryNotFoundError:
        print(f"  ✗  Repo not found: {HF_REPO_ID}")


# ─── interactive menu ─────────────────────────────────────────────────────────

def _pick_files_from_dir() -> list[Path]:
    """Let the user choose files from DATA_DIR interactively."""
    if not DATA_DIR.exists():
        print(f"  ✗  DATA_DIR not found: {DATA_DIR}")
        return []

    all_files = sorted(f for f in DATA_DIR.iterdir() if f.is_file())
    if not all_files:
        print(f"  ✗  No files in {DATA_DIR}")
        return []

    print(f"\n  Files in {DATA_DIR}:")
    for i, f in enumerate(all_files, 1):
        print(f"    [{i}] {f.name}  ({f.stat().st_size / 1024:.1f} KB)")

    raw = input("\n  Enter numbers to upload (e.g. 1 3), or 'all': ").strip().lower()
    if raw == "all":
        return all_files
    try:
        indices = [int(x) - 1 for x in raw.split()]
        return [all_files[i] for i in indices]
    except (ValueError, IndexError):
        print("  ✗  Invalid selection.")
        return []


def interactive_menu():
    _sep("Jatim Curah Hujan & Padi — HF Manager")
    print(f"  Repo : {HF_REPO_ID}")
    print(f"  Local: {DATA_DIR}\n")
    _sep()
    print("  1  Upload files from local DATA_DIR to HuggingFace")
    print("  2  Download raw Excel files  (CH + Padi)")
    print("  3  Download combined files   (CSV + XLSX)")
    print("  4  List files in repo")
    print("  0  Exit")
    _sep()

    choice = input("  Option [0–4]: ").strip()

    if choice == "0":
        print("  Goodbye.")
        sys.exit(0)

    elif choice == "1":
        _sep("Upload")
        files = _pick_files_from_dir()
        if files:
            cmd_upload(files)

    elif choice == "2":
        _sep("Download raw")
        cmd_download(RAW_FILES)

    elif choice == "3":
        _sep("Download combined")
        cmd_download(COMBINED_FILES)

    elif choice == "4":
        _sep("Repo listing")
        cmd_list()

    else:
        print("  ✗  Invalid option.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="hf_manager",
        description="Upload/download dataset files to/from HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("upload", help="Upload file(s) to HuggingFace")
    up.add_argument("files", nargs="+", metavar="FILE",
                    help="Local file paths to upload")

    dl = sub.add_parser("download", help="Download files from HuggingFace")
    dl.add_argument("target", choices=["raw", "combined", "all"],
                    help="'raw' = source Excels, 'combined' = merged CSV+XLSX, 'all' = everything")

    sub.add_parser("list", help="List files in the HuggingFace repo")

    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd is None:
        interactive_menu()
        return

    if args.cmd == "upload":
        _sep("Upload")
        cmd_upload([Path(f) for f in args.files])

    elif args.cmd == "download":
        _sep("Download")
        target_map = {
            "raw":      RAW_FILES,
            "combined": COMBINED_FILES,
            "all":      RAW_FILES + COMBINED_FILES,
        }
        cmd_download(target_map[args.target])

    elif args.cmd == "list":
        _sep("Repo listing")
        cmd_list()


if __name__ == "__main__":
    main()