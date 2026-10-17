#!/usr/bin/env python3
"""Download XiaomiMiMo/MiMo-VL-7B-{RL,SFT} into ./models/MiMo-VL-7B-{RL,SFT}/.

Thin wrapper around download_qwen_vl.py (same snapshot_download flow,
same verify, same hf-transfer toggle) with MiMo-VL-specific defaults.

Usage:
    python3 scripts/download_mimo_vl.py            # 7B-RL (default)
    python3 scripts/download_mimo_vl.py --sft      # 7B-SFT variant
    python3 scripts/download_mimo_vl.py --repo X   # custom repo id
    python3 scripts/download_mimo_vl.py --dest Y   # custom local dir
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
        "--sft", action="store_true",
        help="Use XiaomiMiMo/MiMo-VL-7B-SFT instead of the RL variant.",
    )
    ap.add_argument("--repo", default=None, help="Override HF repo id")
    ap.add_argument("--dest", default=None, help="Override local destination dir")
    ap.add_argument("--hf-home", default=None, help="Override HF_HOME (cache root)")
    ap.add_argument(
        "--force", action="store_true", help="Redownload even if verify passes",
    )
    ap.add_argument(
        "--approx-gb", type=int, default=None,
        help="Override disk-space pre-check (GB); default 16 for 7B BF16.",
    )
    args = ap.parse_args()

    if args.repo is None:
        args.repo = "XiaomiMiMo/MiMo-VL-7B-SFT" if args.sft else "XiaomiMiMo/MiMo-VL-7B-RL"
    if args.dest is None:
        args.dest = "models/MiMo-VL-7B-SFT" if args.sft else "models/MiMo-VL-7B-RL"
    if args.approx_gb is None:
        args.approx_gb = 16

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
