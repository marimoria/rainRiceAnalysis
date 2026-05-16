"""
hf_manager.py
─────────────
Manages HuggingFace dataset uploads, downloads, and deletions.

Usage
─────
  python hf_manager.py                      # Interactive menu
  python hf_manager.py upload <files…>      # CLI: upload specific files
  python hf_manager.py download <files…>    # CLI: download specific files
  python hf_manager.py delete <filenames…>  # CLI: delete files from repo

Environment
───────────
  HF_TOKEN   — HuggingFace access token (read from .env or shell env)
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
DATA_DIR = BASE_DIR / "__data__"

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
    except RepositoryNotFoundError:
        print(f"  ℹ  Repo not found — creating {HF_REPO_ID} …")
        create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, private=False, token=HF_TOKEN)
        print(f"  ✓  Created: https://huggingface.co/datasets/{HF_REPO_ID}")

def _repo_filenames(api: HfApi) -> list[str]:
    """Return list of filenames currently in the repo."""
    try:
        return list(api.list_repo_files(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN
        ))
    except RepositoryNotFoundError:
        return []

# ─── actions ──────────────────────────────────────────────────────────────────

def cmd_upload(files: list[Path]):
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

def cmd_delete(filenames: list[str]):
    _require_token()
    api = HfApi()

    existing = _repo_filenames(api)
    if not existing:
        print("  ✗  Repo is empty or not found.")
        return

    for name in filenames:
        if name not in existing:
            print(f"  ⚠  Not in repo (skipped): {name}")
            continue
            
        print(f"  🗑  {name} …", end="", flush=True)
        try:
            api.delete_file(
                path_in_repo=name,
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN,
            )
            print(" ✓")
        except Exception as exc:
            print(f" ✗  {exc}")

# ─── interactive menu ─────────────────────────────────────────────────────────

def _pick_files_from_dir() -> list[Path]:
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

def _pick_files_from_repo(action: str) -> list[str]:
    _require_token()
    api = HfApi()
    existing = _repo_filenames(api)

    if not existing:
        print("  ✗  Repo is empty or not found.")
        return []

    print(f"\n  Files in repo:")
    for i, name in enumerate(existing, 1):
        print(f"    [{i}] {name}")

    raw = input(f"\n  Enter numbers to {action} (e.g. 1 3), or 'all': ").strip().lower()
    if raw == "all":
        return existing
    try:
        indices = [int(x) - 1 for x in raw.split()]
        return [existing[i] for i in indices]
    except (ValueError, IndexError):
        print("  ✗  Invalid selection.")
        return []

def interactive_menu():
    _sep("Jatim Curah Hujan & Padi — HF Manager")
    print(f"  Repo : {HF_REPO_ID}")
    print(f"  Local: {DATA_DIR}\n")
    _sep()
    print("  1  Upload files from local to HuggingFace")
    print("  2  Download files from HuggingFace to local")
    print("  3  Delete files from repo")
    print("  0  Exit")
    _sep()

    choice = input("  Option [0–3]: ").strip()

    if choice == "0":
        print("  Goodbye.")
        sys.exit(0)
    elif choice == "1":
        _sep("Upload")
        files = _pick_files_from_dir()
        if files: cmd_upload(files)
    elif choice == "2":
        _sep("Download")
        names = _pick_files_from_repo("download")
        if names: cmd_download(names)
    elif choice == "3":
        _sep("Delete")
        names = _pick_files_from_repo("delete")
        if names: cmd_delete(names)
    else:
        print("  ✗  Invalid option.")

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        prog="hf_manager",
        description="Upload/download/delete dataset files to/from HuggingFace.",
    )
    sub = p.add_subparsers(dest="cmd")

    up = sub.add_parser("upload", help="Upload file(s) to HuggingFace")
    up.add_argument("files", nargs="+", help="Local file paths to upload")

    dl = sub.add_parser("download", help="Download files from HuggingFace")
    dl.add_argument("filenames", nargs="+", help="Filenames in the repo to download")

    de = sub.add_parser("delete", help="Delete file(s) from the HuggingFace repo")
    de.add_argument("filenames", nargs="+", help="Filenames in the repo to delete")

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
        cmd_download(args.filenames)
    elif args.cmd == "delete":
        _sep("Delete")
        cmd_delete(args.filenames)

if __name__ == "__main__":
    main()