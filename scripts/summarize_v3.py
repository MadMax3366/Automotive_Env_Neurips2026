#!/usr/bin/env python3
"""Aggregate run_task.py results across all models in new_test_01/.

Reports
-------
1. Per-model x per-category table:  SR%, avg validator_score, error count.
2. Per-task CSV with the underlying numbers, one row per (model, task_id).
3. Aggregate JSON for downstream tooling.

Why fine-grained
----------------
run_task.py records `validator_score` (0..100) for every task. For
`equals_multi` and `graded` validators the score is the (weighted)
percentage of sub-checks that passed, so partial completion is captured
natively. SR (binary success) is still emitted alongside.

Usage
-----
    python3 scripts/summarize_v3.py                          # default root
    python3 scripts/summarize_v3.py --root new_test_01
    python3 scripts/summarize_v3.py --models m3a_gemini-2_5-flash
    python3 scripts/summarize_v3.py --csv my_summary.csv
    python3 scripts/summarize_v3.py --mode mean              # avg over reruns
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

# Category basename -> matches the four task files shipped under ./task/.
# run_task.py writes outputs as new_test_01/<agent>_<model>/<basename>_NN.json.
CATEGORIES = OrderedDict([
    ("explicit",         "task_explicit"),
    ("implicit",         "task_implicit"),
    ("safety_explicit",  "task_safety_explicit"),
    ("safety_implicit",  "task_safety_implicit"),
])

_NN_RE = re.compile(r"_(\d{2})\.json$")

# Error rows that reduce the SR/avgS denominator (i.e. "task did not have
# a fair chance to be evaluated"). Anything else with an `error` key is
# treated as a *completed-but-failed* run: counted in the denominator
# with score=0 and success=0.
_EXCLUDABLE_ERR_MSGS = {
    # m3a.py raises this exactly when raw_response is falsy AFTER the
    # wrapper has exhausted its own retries (real LLM-API failure, not a
    # task-side fault).
    "Error calling LLM in action selection phase.",
}


def _is_excludable_error(r: dict) -> bool:
    return "error" in r and r["error"] in _EXCLUDABLE_ERR_MSGS


def latest_file_for(model_dir: Path, prefix: str) -> Optional[Path]:
    """Return the highest-numbered _NN.json for prefix under model_dir, or None."""
    candidates = files_for(model_dir, prefix)
    return candidates[-1] if candidates else None


def files_for(model_dir: Path, prefix: str) -> list:
    """Return all matching _NN.json files for prefix, sorted by NN ascending."""
    out = []
    for p in glob.glob(str(model_dir / f"{prefix}_*.json")):
        m = _NN_RE.search(os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return [Path(p) for _, p in sorted(out, key=lambda x: x[0])]


def merge_results_across_files(files: list) -> tuple:
    """Mean-aggregate results by task_id across multiple _NN.json snapshots.

    For each task_id, average across all rerun versions:
      success_rate         = mean(1.0 if success else 0.0)
      avg validator_score  = mean(score; missing or error counts as 0)
      sub_pass / sub_fail  = sums across runs (true ratio over all attempts)

    Returns (merged_results_list, files_used_count, total_results_seen).
    """
    if not files:
        return [], 0, 0
    runs_by_id: dict = {}
    first_order: list = []
    seen_ids = set()
    total_seen = 0
    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception as e:
            print(f"[warn] {fp}: read failed ({e}); skipping", file=sys.stderr)
            continue
        for r in (d.get("results") or []):
            tid = r.get("task_id")
            if not tid:
                continue
            total_seen += 1
            if tid not in seen_ids:
                first_order.append(tid)
                seen_ids.add(tid)
            runs_by_id.setdefault(tid, []).append(r)

    merged = []
    for tid in first_order:
        runs = runs_by_id[tid]
        n_runs = len(runs)
        non_err = [r for r in runs if "error" not in r]
        if not non_err:
            merged.append({
                "task_id": tid,
                "goal_text": runs[0].get("goal_text", ""),
                "error": f"all {n_runs} runs errored",
                "n_runs": n_runs,
                "n_runs_success": 0,
                "sub_pass_sum": 0,
                "sub_fail_sum": 0,
            })
            continue
        success_per_run = [
            1.0 if r.get("success") is True else 0.0 for r in runs
        ]
        score_per_run = [
            float(r.get("validator_score") or 0) for r in runs
        ]
        sub_pass_total = sum(len(r.get("validator_passed") or []) for r in runs)
        sub_fail_total = sum(len(r.get("validator_failed") or []) for r in runs)
        avg_success = sum(success_per_run) / n_runs
        avg_score = sum(score_per_run) / n_runs
        rep = non_err[0]
        merged.append({
            "task_id": tid,
            "goal_text": rep.get("goal_text", ""),
            "success": avg_success,
            "success_rate": avg_success,
            "validator_type": rep.get("validator_type", ""),
            "validator_score": avg_score,
            "validator_passed": rep.get("validator_passed") or [],
            "validator_failed": rep.get("validator_failed") or [],
            "n_runs": n_runs,
            "n_runs_success": int(sum(success_per_run)),
            "sub_pass_sum": sub_pass_total,
            "sub_fail_sum": sub_fail_total,
            "timing": rep.get("timing"),
        })
    return merged, len(files), total_seen


def per_category_metrics(results: list) -> dict:
    """Compute aggregate metrics for one (model, category)'s result list.

    Handles two schemas:
      - native run output: success is bool, no aggregate fields.
      - mean-merged: success is float in [0,1] (per-task SR across runs),
        and sub_pass_sum / sub_fail_sum carry totals.
    """
    n = len(results)
    n_excl_err = sum(1 for r in results if _is_excludable_error(r))
    n_err = n_excl_err
    n_done = n - n_excl_err
    success_total = 0.0
    for r in results:
        if _is_excludable_error(r):
            continue
        if "error" in r:
            continue
        s = r.get("success")
        if s is True:
            success_total += 1.0
        elif isinstance(s, (int, float)) and not isinstance(s, bool):
            success_total += float(s)
    scores = []
    for r in results:
        if _is_excludable_error(r):
            continue
        if "error" in r:
            scores.append(0.0)
        elif r.get("validator_score") is not None:
            scores.append(r["validator_score"])
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    return {
        "n": n,
        "n_done": n_done,
        "n_err": n_err,
        "n_success": success_total,
        "sr_pct": (100.0 * success_total / n_done) if n_done else 0.0,
        "avg_score": avg_score,
    }


def collect_per_task_rows(model_name: str, category: str,
                          results: list, source_path: Path) -> list:
    rows = []
    for r in results:
        timing = r.get("timing") or {}
        rows.append({
            "model": model_name,
            "category": category,
            "task_id": r.get("task_id", ""),
            "goal_text": r.get("goal_text", ""),
            "success": r.get("success"),
            "validator_type": r.get("validator_type", ""),
            "validator_score": r.get("validator_score"),
            "subchecks_passed": len(r.get("validator_passed") or []),
            "subchecks_failed": len(r.get("validator_failed") or []),
            "error": r.get("error", ""),
            "steps": timing.get("total_steps"),
            "llm_calls": timing.get("llm_calls"),
            "duration_s": timing.get("duration_s"),
            "failed_checks": " | ".join(r.get("validator_failed") or []),
            "source_file": str(source_path.relative_to(source_path.parents[1]))
                           if len(source_path.parents) >= 2 else str(source_path),
        })
    return rows


def discover_model_dirs(root: Path, only: Optional[list]) -> list:
    """Find m3a_* / t3a_* subdirs under root (one per agent_model combo).
    If `only` is given, restrict to those names."""
    dirs = sorted([
        p for p in root.iterdir()
        if p.is_dir() and (p.name.startswith("m3a_") or p.name.startswith("t3a_"))
    ])
    if only:
        wanted = set(only)
        dirs = [d for d in dirs if d.name in wanted]
    return dirs


def print_main_table(rows_by_model: "OrderedDict[str, dict]"):
    """rows_by_model[model] = {cat: metrics, ..., '_total': metrics}."""
    cat_keys = list(CATEGORIES.keys())
    name_w = max((len(m) for m in rows_by_model), default=10)
    name_w = max(name_w, len("Model"))
    print(f"\n=== Success rate + average validator_score per category ===\n")
    header = f"{'Model':<{name_w}}  "
    for c in cat_keys + ["TOTAL"]:
        header += f" {c:>22}"
    print(header)
    sub = " " * (name_w + 2)
    for c in cat_keys + ["TOTAL"]:
        sub += f" {'SR%':>7} {'avgS':>7} {'err':>5}"
    print(sub)
    print("-" * len(sub))
    for model, by_cat in rows_by_model.items():
        line = f"{model:<{name_w}}  "
        for c in cat_keys + ["_total"]:
            m = by_cat.get(c)
            if not m:
                line += f" {'-':>7} {'-':>7} {'-':>5}"
                continue
            line += (f" {m['sr_pct']:>6.1f}%"
                     f" {m['avg_score']:>7.1f}"
                     f" {m['n_err']:>5d}")
        print(line)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default="new_test_01",
                    help="Root directory containing <agent>_<model>/ subdirs.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Optional explicit list of model dir names to include.")
    ap.add_argument("--csv", default=None,
                    help="Output per-task CSV path. "
                         "Default: <root>/_aggregate/per_task.csv")
    ap.add_argument("--json", default=None,
                    help="Output aggregate JSON path. "
                         "Default: <root>/_aggregate/summary.json")
    ap.add_argument("--mode", choices=("latest", "mean"), default="latest",
                    help="How to combine multiple _NN.json snapshots per "
                         "(model, category):\n"
                         "  latest (default): use only the highest-numbered "
                         "_NN.json (one fresh run).\n"
                         "  mean: average per-task SR / validator_score across "
                         "every _NN.json (useful when you ran the same suite "
                         "multiple times and want an honest expected-run value).")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"[error] root dir not found: {root}", file=sys.stderr)
        return 2
    model_dirs = discover_model_dirs(root, args.models)
    if not model_dirs:
        print(f"[error] no <agent>_<model>/ subdirs under {root}", file=sys.stderr)
        return 2

    rows_by_model: "OrderedDict[str, dict]" = OrderedDict()
    per_task_rows = []
    files_used: dict = {}

    for mdir in model_dirs:
        model = mdir.name
        rows_by_model[model] = OrderedDict()
        files_used[model] = {}
        total_results = []
        for cat, prefix in CATEGORIES.items():
            if args.mode == "latest":
                f = latest_file_for(mdir, prefix)
                file_list = [f] if f else []
            else:
                file_list = files_for(mdir, prefix)
            if not file_list:
                rows_by_model[model][cat] = None
                files_used[model][cat] = []
                continue
            if args.mode == "latest":
                try:
                    d = json.loads(file_list[0].read_text())
                except Exception as e:
                    print(f"[warn] {file_list[0]}: read failed ({e}); skipping",
                          file=sys.stderr)
                    rows_by_model[model][cat] = None
                    files_used[model][cat] = []
                    continue
                results = d.get("results") or []
                total_seen = len(results)
            else:
                results, _, total_seen = merge_results_across_files(file_list)
            if not results:
                rows_by_model[model][cat] = None
                files_used[model][cat] = [p.name for p in file_list]
                continue
            rows_by_model[model][cat] = per_category_metrics(results)
            files_used[model][cat] = {
                "files": [p.name for p in file_list],
                "raw_seen": total_seen,
                "merged_n": len(results),
            }
            label_file = file_list[-1]
            per_task_rows.extend(collect_per_task_rows(model, cat, results, label_file))
            total_results.extend(results)
        if total_results:
            rows_by_model[model]["_total"] = per_category_metrics(total_results)

    nonempty = OrderedDict(
        (m, by_cat) for m, by_cat in rows_by_model.items()
        if by_cat.get("_total")
    )

    print_main_table(nonempty)
    skipped = [m for m in rows_by_model if m not in nonempty]
    if skipped:
        print(f"\n(skipped — no result files: {', '.join(skipped)})")

    mode_label = {
        "mean": "merged across all _NN.json (mean of runs)",
        "latest": "latest _NN.json only",
    }[args.mode]
    print(f"\n=== Source files used ({mode_label}) ===\n")
    name_w = max((len(m) for m in rows_by_model), default=10)
    for model, by_cat in files_used.items():
        for cat in CATEGORIES:
            entry = by_cat.get(cat)
            if not entry:
                print(f"  {model:<{name_w}}  {cat:<16}  -- missing --")
                continue
            if isinstance(entry, dict):
                files = entry["files"]
                raw = entry["raw_seen"]
                merged = entry["merged_n"]
                if len(files) == 1:
                    print(f"  {model:<{name_w}}  {cat:<16}  {files[0]} "
                          f"({merged} tasks)")
                else:
                    print(f"  {model:<{name_w}}  {cat:<16}  "
                          f"merged {len(files)} files -> {merged} tasks "
                          f"({raw} raw rows): {', '.join(files)}")

    out_dir = root / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else (out_dir / "per_task.csv")
    json_path = Path(args.json) if args.json else (out_dir / "summary.json")

    if per_task_rows:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_task_rows[0].keys()))
            w.writeheader()
            w.writerows(per_task_rows)
        print(f"\nper-task CSV ({len(per_task_rows)} rows): {csv_path.resolve()}")

    summary_blob = {
        "root": str(root.resolve()),
        "categories": list(CATEGORIES.keys()),
        "models": [
            {
                "model": model,
                "by_category": {
                    cat: rows_by_model[model].get(cat) for cat in CATEGORIES
                },
                "total": rows_by_model[model].get("_total"),
                "files_used": files_used.get(model, {}),
            }
            for model in rows_by_model
        ],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary_blob, indent=2, ensure_ascii=False))
    print(f"summary JSON: {json_path.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
