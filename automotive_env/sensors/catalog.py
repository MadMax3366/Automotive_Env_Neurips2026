"""Parse vhal_capability.txt -> data/vhal_catalog.json.

Run: python -m automotive_env.sensors.catalog
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "vhal_capability.txt"
OUTPUT = ROOT / "data" / "vhal_catalog.json"

_PROP_RE = re.compile(
    r"^Property:(0x[0-9a-fA-F]+),\s*Property name:(\S+?),\s*"
    r"access:(0x[0-9a-fA-F]+),\s*changeMode:(0x[0-9a-fA-F]+),"
)
_AREA_RE = re.compile(
    r"areaId:(0x[0-9a-fA-F]+),\s*f min:([-\d.]+),\s*f max:([-\d.]+),\s*"
    r"i min:(-?\d+),\s*i max:(-?\d+),\s*i64 min:(-?\d+),\s*i64 max:(-?\d+)"
)

_ACCESS = {0x1: "R", 0x2: "W", 0x3: "RW"}
_CHANGE_MODE = {0x0: "STATIC", 0x1: "ON_CHANGE", 0x2: "CONTINUOUS"}


def parse(text: str) -> dict:
    catalog: dict = {}
    current = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        m = _PROP_RE.match(line)
        if m:
            pid_hex, name, access_hex, change_hex = m.groups()
            pid = int(pid_hex, 16)
            current = {
                "id": pid,
                "id_hex": pid_hex,
                "access": _ACCESS.get(int(access_hex, 16), access_hex),
                "change_mode": _CHANGE_MODE.get(int(change_hex, 16), change_hex),
                "value_type": "int32",  # filled below
                "areas": {},
            }
            catalog[name] = current
            continue
        if current is None:
            continue
        ma = _AREA_RE.search(line)
        if ma:
            area_hex, fmin, fmax, imin, imax, i64min, i64max = ma.groups()
            current["areas"][area_hex] = {
                "f_min": float(fmin),
                "f_max": float(fmax),
                "i_min": int(imin),
                "i_max": int(imax),
                "i64_min": int(i64min),
                "i64_max": int(i64max),
            }

    for meta in catalog.values():
        if not meta["areas"]:
            meta["areas"] = {"0x0": {"f_min": 0.0, "f_max": 0.0,
                                     "i_min": 0, "i_max": 0,
                                     "i64_min": 0, "i64_max": 0}}
        is_float = any(a["f_min"] != 0.0 or a["f_max"] != 0.0
                       for a in meta["areas"].values())
        meta["value_type"] = "float" if is_float else "int32"
    return catalog


def main():
    if not INPUT.exists():
        raise SystemExit(f"input not found: {INPUT}")
    catalog = parse(INPUT.read_text())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(catalog, indent=2))
    total = len(catalog)
    rw = sum(1 for m in catalog.values() if m["access"] == "RW")
    r_only = sum(1 for m in catalog.values() if m["access"] == "R")
    w_only = sum(1 for m in catalog.values() if m["access"] == "W")
    floats = sum(1 for m in catalog.values() if m["value_type"] == "float")
    print(f"[catalog] wrote {OUTPUT} ({total} properties)")
    print(f"[catalog] access: RW={rw}, R={r_only}, W={w_only}")
    print(f"[catalog] value_type: float={floats}, int32={total - floats}")


if __name__ == "__main__":
    main()
