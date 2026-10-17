#!/usr/bin/env python3
"""Download xlangai/OpenCUA-7B to ./models/OpenCUA-7B/.

Same shape as the other download_*.py scripts: verify + space check +
hf-transfer opt-in.

Usage:
    python3 scripts/download_opencua.py                          # default 7B
    python3 scripts/download_opencua.py --repo xlangai/OpenCUA-32B
    python3 scripts/download_opencua.py --dest /some/path
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_REPO = "xlangai/OpenCUA-7B"
DEFAULT_DEST = "models/OpenCUA-7B"

_SIZE_GB_BY_MARKER = {
    "72b": 150,
    "32b": 70,
    "7b":  18,
}
_FALLBACK_SIZE_GB = 18


def _approx_size_gb(repo_id: str) -> int:
    r = repo_id.lower()
    for marker, gb in _SIZE_GB_BY_MARKER.items():
        if marker in r:
            return gb
    return _FALLBACK_SIZE_GB


def _check_space(dest: Path, need_gb: int) -> None:
    parent = dest.parent if not dest.exists() else dest
    free_gb = shutil.disk_usage(parent).free / (1024 ** 3)
    if free_gb < need_gb + 1:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free at {parent} — need ~{need_gb + 1} GB."
        )


def _verify(dest: Path) -> bool:
    config = dest / "config.json"
    if not config.is_file():
        print(f"[verify] MISSING: {config}", file=sys.stderr)
        return False
    shards = list(dest.glob("model*.safetensors"))
    if not shards:
        print(f"[verify] MISSING: no *.safetensors under {dest}", file=sys.stderr)
        return False
    # OpenCUA ships with a custom modeling_*.py + tokenization files —
    # require the tokenizer & a chat-template-bearing config.
    for req in ("tokenizer_config.json",):
        if not (dest / req).is_file():
            print(f"[verify] MISSING: {dest / req}", file=sys.stderr)
            return False
    total_gb = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / (1024 ** 3)
    print(f"[verify] OK — {len(shards)} safetensors shard(s), total {total_gb:.2f} GB")
    return True


def download(repo_id: str, dest: Path, hf_home: str | None, approx_gb: int) -> None:
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("[download] hf-transfer detected — enabled for faster download")
    except ImportError:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        print("[download] hf-transfer NOT installed — `pip install hf_transfer` "
              "for 2-5x speedup.")

    if hf_home:
        os.environ["HF_HOME"] = hf_home
        print(f"[download] HF_HOME override: {hf_home}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
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
            ignore_patterns=["*.onnx", "*.gguf", "*.msgpack", "*.h5"],
        )
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "name or service not known" in msg or "connection" in msg or "timeout" in msg:
            print(f"ERROR: network failure: {e}", file=sys.stderr); sys.exit(3)
        if "no space" in msg or "disk full" in msg:
            print(f"ERROR: disk full at {dest}.", file=sys.stderr); sys.exit(4)
        if "gated" in msg or "401" in msg:
            print(f"ERROR: gated/auth — run huggingface-cli login. {e}",
                  file=sys.stderr); sys.exit(5)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--hf-home", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--approx-gb", type=int, default=None)
    args = ap.parse_args()

    dest = Path(args.dest).expanduser()
    if not dest.is_absolute():
        dest = (Path(__file__).resolve().parents[1] / dest).resolve()

    if dest.exists() and not args.force and _verify(dest):
        print(f"[skip] {dest} already complete — pass --force to redownload.")
        return 0

    approx_gb = args.approx_gb if args.approx_gb else _approx_size_gb(args.repo)
    download(args.repo, dest, args.hf_home, approx_gb)
    if not _verify(dest):
        print("ERROR: verify failed after download.", file=sys.stderr)
        return 1
    print(f"[done] {args.repo} ready at {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
