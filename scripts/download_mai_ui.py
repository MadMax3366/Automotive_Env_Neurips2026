#!/usr/bin/env python3
"""Download Tongyi-MAI/MAI-UI-{2B,8B} into ./models/MAI-UI-{2B,8B}/.

Thin wrapper around download_qwen_vl.py (same snapshot_download flow,
same verify, same hf-transfer toggle) with MAI-UI-specific defaults.

Usage:
    python3 scripts/download_mai_ui.py            # 2B (default)
    python3 scripts/download_mai_ui.py --8b       # 8B variant
    python3 scripts/download_mai_ui.py --repo X   # custom repo id
    python3 scripts/download_mai_ui.py --dest Y   # custom local dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse helpers from download_qwen_vl.py (same scripts/ dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_qwen_vl as _qw  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--8b", dest="eight_b", action="store_true",
        help="Use Tongyi-MAI/MAI-UI-8B instead of 2B.",
    )
    ap.add_argument("--repo", default=None, help="Override HF repo id")
    ap.add_argument("--dest", default=None, help="Override local destination dir")
    ap.add_argument("--hf-home", default=None, help="Override HF_HOME (cache root)")
    ap.add_argument(
        "--force", action="store_true", help="Redownload even if verify passes",
    )
    ap.add_argument(
        "--approx-gb", type=int, default=None,
        help="Override disk-space pre-check (GB); default 8 for 2B, 18 for 8B.",
    )
    args = ap.parse_args()

    if args.repo is None:
        args.repo = "Tongyi-MAI/MAI-UI-8B" if args.eight_b else "Tongyi-MAI/MAI-UI-2B"
    if args.dest is None:
        args.dest = "models/MAI-UI-8B" if args.eight_b else "models/MAI-UI-2B"
    if args.approx_gb is None:
        args.approx_gb = 18 if args.eight_b else 8

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
