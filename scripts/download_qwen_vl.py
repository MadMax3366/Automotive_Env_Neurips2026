#!/usr/bin/env python3
"""Download Qwen2.5-VL-3B-Instruct to ./models/Qwen2.5-VL-3B-Instruct/.

Why this script instead of `hf download` CLI:
- runs verify step (config.json + >=1 model shard present) inside Python
- prints clear errors for network / disk-full / permission issues
- honors HF_HOME (mirror/cache override) and HF_HUB_ENABLE_HF_TRANSFER

Usage:
    python3 scripts/download_qwen_vl.py                    # to ./models/Qwen2.5-VL-3B-Instruct/
    python3 scripts/download_qwen_vl.py --dest /some/path  # custom dest
    python3 scripts/download_qwen_vl.py --repo Qwen/Qwen2.5-VL-7B-Instruct  # other sibling
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_DEST = "models/Qwen2.5-VL-3B-Instruct"

# Rough BF16 on-disk footprint (params × 2 bytes + processor/config overhead).
# Used for the pre-flight disk-space check; override with --approx-gb if wrong.
_SIZE_GB_BY_MARKER = {
    "72b": 150,
    "7b": 16,
    "3b": 7,
}
_FALLBACK_SIZE_GB = 10


def _approx_size_gb(repo_id: str) -> int:
    """Best-effort size lookup by substring in the repo id."""
    r = repo_id.lower()
    for marker, gb in _SIZE_GB_BY_MARKER.items():
        if marker in r:
            return gb
    return _FALLBACK_SIZE_GB


def _check_space(dest: Path, need_gb: int) -> None:
    """Raise if <need_gb GB free where dest will live."""
    parent = dest.parent if not dest.exists() else dest
    free_bytes = shutil.disk_usage(parent).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < need_gb + 1:  # +1 GB safety margin for metadata
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free at {parent} — need ~{need_gb + 1} GB. "
            f"Move the repo or set --dest to a larger volume."
        )


def _verify(dest: Path) -> bool:
    """Confirm essential files are present. Returns True iff download looks complete."""
    config = dest / "config.json"
    if not config.is_file():
        print(f"[verify] MISSING: {config}", file=sys.stderr)
        return False
    # Qwen2.5-VL ships as either model.safetensors (single) or sharded
    # model-00001-of-000NN.safetensors + model.safetensors.index.json
    shards = list(dest.glob("model*.safetensors"))
    if not shards:
        print(f"[verify] MISSING: no *.safetensors under {dest}", file=sys.stderr)
        return False
    # processor / tokenizer are loaded on init — require the common few
    for req in ("tokenizer_config.json", "preprocessor_config.json"):
        if not (dest / req).is_file():
            print(f"[verify] MISSING: {dest / req}", file=sys.stderr)
            return False
    total_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 ** 3)
    print(f"[verify] OK — {len(shards)} safetensors shard(s), total {total_gb:.2f} GB")
    return True


def download(repo_id: str, dest: Path, hf_home: str | None, approx_gb: int) -> None:
    # Enable hf-transfer if the package is present (gives 2-5x speedup on fast links).
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[download] hf-transfer detected — enabled for faster download")
    except ImportError:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        print("[download] hf-transfer NOT installed (optional) — using stock downloader. "
              "`pip install hf_transfer` for 2-5x speedup.")

    if hf_home:
        os.environ["HF_HOME"] = hf_home
        print(f"[download] HF_HOME override: {hf_home}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub not installed. Run:\n"
            "    pip install huggingface_hub\n"
            "(or install requirements-gpu.txt on this host).",
            file=sys.stderr,
        )
        sys.exit(2)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo : {repo_id}")
    print(f"[download] dest : {dest.resolve()}")
    print(f"[download] size : ~{approx_gb} GB (BF16 estimate)")

    _check_space(dest, approx_gb)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            local_dir_use_symlinks=False,
            # Keep to weights + configs; skip onnx duplicates and old GGUF if ever present.
            ignore_patterns=["*.onnx", "*.gguf", "*.msgpack", "*.h5"],
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "name or service not known" in msg or "connection" in msg or "timeout" in msg:
            print(
                f"ERROR: network failure while downloading.\n"
                f"  reason : {e}\n"
                f"  hints  : check proxy, or set HF_ENDPOINT (e.g. a mirror).",
                file=sys.stderr,
            )
            sys.exit(3)
        if "no space" in msg or "disk full" in msg:
            print(f"ERROR: disk full while writing {dest}. See `df -h`.", file=sys.stderr)
            sys.exit(4)
        if "gated" in msg or "authorized" in msg or "401" in msg:
            print(
                f"ERROR: model access may be gated — run `huggingface-cli login` "
                f"or set HF_TOKEN. Underlying: {e}",
                file=sys.stderr,
            )
            sys.exit(5)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF repo id (default: {DEFAULT_REPO})")
    ap.add_argument("--dest", default=DEFAULT_DEST, help=f"Local directory (default: {DEFAULT_DEST})")
    ap.add_argument("--hf-home", default=None, help="Override HF_HOME (cache root)")
    ap.add_argument("--force", action="store_true", help="Redownload even if verify passes")
    ap.add_argument(
        "--approx-gb",
        type=int,
        default=None,
        help="Override disk-space pre-check (GB). Default: auto from repo id "
             "(3B→7, 7B→16, 72B→150, else 10).",
    )
    args = ap.parse_args()

    # Resolve dest relative to project root (script lives in scripts/).
    dest = Path(args.dest).expanduser()
    if not dest.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        dest = (project_root / dest).resolve()

    if dest.exists() and not args.force and _verify(dest):
        print(f"[skip] {dest} already looks complete — pass --force to redownload.")
        return 0

    approx_gb = args.approx_gb if args.approx_gb else _approx_size_gb(args.repo)
    download(args.repo, dest, args.hf_home, approx_gb)

    if not _verify(dest):
        print(
            "ERROR: verify step failed after download — directory exists but looks "
            "incomplete. Re-run with --force or inspect the path.",
            file=sys.stderr,
        )
        return 1
    print(f"[done] {args.repo} ready at {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
