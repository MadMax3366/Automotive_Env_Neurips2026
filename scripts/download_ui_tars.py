#!/usr/bin/env python3
"""Download ByteDance-Seed/UI-TARS-1.5-7B into ./models/UI-TARS-1.5-7B/.

Thin wrapper around download_qwen_vl.py (same snapshot_download flow,
verify, hf-transfer toggle) with UI-TARS-specific defaults.

Usage:
    python3 scripts/download_ui_tars.py
    python3 scripts/download_ui_tars.py --repo X    # custom HF repo
    python3 scripts/download_ui_tars.py --dest Y    # custom local dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_qwen_vl as _qw  # noqa: E402

DEFAULT_REPO = "ByteDance-Seed/UI-TARS-1.5-7B"
DEFAULT_DEST = "models/UI-TARS-1.5-7B"
# UI-TARS-1.5-7B is qwen2_5_vl arch, 8.3B params -> ~17GB BF16 on disk.
DEFAULT_APPROX_GB = 18


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"Local directory (default: {DEFAULT_DEST})")
    ap.add_argument("--hf-home", default=None, help="Override HF_HOME (cache root)")
    ap.add_argument("--force", action="store_true", help="Redownload even if verify passes")
    ap.add_argument(
        "--approx-gb", type=int, default=DEFAULT_APPROX_GB,
        help=f"Disk-space pre-check (GB, default {DEFAULT_APPROX_GB}).",
    )
    args = ap.parse_args()

    dest = Path(args.dest).expanduser()
    if not dest.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        dest = (project_root / dest).resolve()

    if dest.exists() and not args.force and _qw._verify(dest):
        print(f"[skip] {dest} already looks complete — pass --force to redownload.")
        return 0

    _qw.download(args.repo, dest, args.hf_home, args.approx_gb)
    if not _qw._verify(dest):
        print(
            "ERROR: verify step failed after download — directory exists but "
            "looks incomplete. Re-run with --force or inspect the path.",
            file=sys.stderr,
        )
        return 1
    print(f"[done] {args.repo} ready at {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
