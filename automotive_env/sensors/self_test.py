"""Self-test: probe every RW VHAL (property, area) with multiple write
strategies and emit data/property_write_methods.json — a lookup table
used by run_task.apply_setup to pick the correct write path per property.

Run: python -m automotive_env.sensors.self_test
"""
import json
import time
from collections import Counter
from pathlib import Path

from automotive_env.sensors.vhal import (
    set_vhal, get_vhal, inject_vhal,
    _ensure_adb_root, _try_dumpsys_ivehicle, _try_cmd_car_service,
    _load_hvac_power_dependents,
)
from automotive_env.sensors.sensor import set_sensor, get_sensor, set_gps
from automotive_env.sensors.driving import emulate_driving_state, set_day_night

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_CATALOG_PATH = _DATA_DIR / "vhal_catalog.json"
_METHODS_PATH = _DATA_DIR / "property_write_methods.json"

_HVAC_POWER_NAME = "HVAC_POWER_ON"
_HVAC_POWER_AREA = 0x75  # ROW_1 (driver+passenger), per vhal_capability.txt


def pick_test_value(meta: dict, area_meta: dict, original):
    vtype = meta["value_type"]
    if vtype == "float":
        lo, hi = area_meta.get("f_min", 0.0), area_meta.get("f_max", 0.0)
        if hi <= lo:
            return None
        mid = (lo + hi) / 2
        if original is not None and abs(float(original) - mid) < 0.01:
            mid = (mid + hi) / 2
        return mid
    lo, hi = area_meta.get("i_min", 0), area_meta.get("i_max", 0)
    if hi == 0 and lo == 0:
        if original in (0, 1):
            return 1 - int(original)
        return 1 if original != 1 else 0
    mid = (lo + hi) // 2
    if original is not None and mid == original:
        mid = mid + 1 if mid < hi else mid - 1
    return mid


def values_equal(a, b, vtype: str) -> bool:
    if a is None or b is None:
        return False
    if vtype == "float":
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return a == b


def _try_write(writer, name: str, area: int, test_val, vtype: str):
    """Run `writer(name, area, test_val)`, sleep, read-back, return (ok, read_back, reason)."""
    ok_call = writer(name, area, test_val)
    if not ok_call:
        return False, None, "call rejected"
    time.sleep(0.1)
    back = get_vhal(name, area)
    if values_equal(back, test_val, vtype):
        return True, back, "ok"
    return False, back, f"wrote {test_val} but read {back}"


def test_vhal_property_smart(name: str, area: int,
                             area_meta: dict, meta: dict) -> dict:
    """Try multiple write strategies; return the one that round-trips."""
    vtype = meta["value_type"]
    area_hex = hex(area)
    result = {
        "name": name, "area": area_hex,
        "original_value": None, "tested_value": None,
        "method": "unwritable", "attempts": [],
    }
    # If this property is gated on HVAC_POWER_ON (per HAL config array),
    # prime POWER=1 on the area before the read. Without this, an HVAC-off
    # emulator makes every dependent property look "unreadable" on the
    # very first get_vhal() and we skip all write-strategy testing below.
    if meta["id"] in _load_hvac_power_dependents():
        set_vhal(_HVAC_POWER_NAME, area, 1)
        time.sleep(0.2)

    original = get_vhal(name, area)
    if original is None:
        result["method"] = "unreadable"
        return result
    result["original_value"] = original

    test_val = pick_test_value(meta, area_meta, original)
    if test_val is None:
        result["method"] = "no_test_value"
        return result
    result["tested_value"] = test_val

    # 1. direct_set
    ok, back, reason = _try_write(set_vhal, name, area, test_val, vtype)
    result["attempts"].append({"method": "direct_set", "ok": ok, "reason": reason})
    if ok:
        set_vhal(name, area, original)
        result["method"] = "direct_set"
        return result

    # 2. inject (for physical/sensor-driven props)
    ok, back, reason = _try_write(inject_vhal, name, area, test_val, vtype)
    result["attempts"].append({"method": "inject", "ok": ok, "reason": reason})
    if ok:
        inject_vhal(name, area, original)
        result["method"] = "inject"
        return result

    # 3. set_after_hvac_power (for HVAC_* subordinate switches)
    if name.startswith("HVAC_") and name != _HVAC_POWER_NAME:
        hvac_orig = get_vhal(_HVAC_POWER_NAME, _HVAC_POWER_AREA)
        powered = set_vhal(_HVAC_POWER_NAME, _HVAC_POWER_AREA, 1)
        time.sleep(0.2)
        ok, back, reason = _try_write(set_vhal, name, area, test_val, vtype)
        result["attempts"].append(
            {"method": "set_after_hvac_power", "ok": ok,
             "reason": reason, "powered_ok": powered})
        if ok:
            set_vhal(name, area, original)
            if hvac_orig is not None:
                set_vhal(_HVAC_POWER_NAME, _HVAC_POWER_AREA, hvac_orig)
            result["method"] = "set_after_hvac_power"
            return result
        if hvac_orig is not None:
            set_vhal(_HVAC_POWER_NAME, _HVAC_POWER_AREA, hvac_orig)

    # 4. dumpsys_ivehicle (requires adb root; bypasses car_service layer)
    if _ensure_adb_root():
        set_ok = _try_dumpsys_ivehicle(meta, area, test_val)
        if set_ok:
            time.sleep(0.1)
            back = get_vhal(name, area)
            ok = values_equal(back, test_val, vtype)
            reason = "ok" if ok else f"wrote {test_val} but read {back}"
            result["attempts"].append(
                {"method": "dumpsys_ivehicle", "ok": ok, "reason": reason})
            if ok:
                _try_dumpsys_ivehicle(meta, area, original)
                result["method"] = "dumpsys_ivehicle"
                return result
        else:
            result["attempts"].append(
                {"method": "dumpsys_ivehicle", "ok": False, "reason": "SET rejected"})
    else:
        result["attempts"].append(
            {"method": "dumpsys_ivehicle", "ok": False, "reason": "adb root unavailable"})

    return result


def test_vhal_rw() -> dict:
    catalog = json.loads(_CATALOG_PATH.read_text())
    results = {"tested": 0, "by_method": Counter(), "per_property": []}
    for name, meta in catalog.items():
        if meta["access"] != "RW":
            continue
        for area_hex, area_meta in meta["areas"].items():
            area = int(area_hex, 16)
            r = test_vhal_property_smart(name, area, area_meta, meta)
            results["by_method"][r["method"]] += 1
            results["per_property"].append(r)
            results["tested"] += 1
    return results


def write_methods_json(vhal_results: dict) -> dict:
    by_prop: dict = {}
    for r in vhal_results["per_property"]:
        entry = {"method": r["method"]}
        if r["method"] == "set_after_hvac_power":
            entry["precondition"] = {
                "name": _HVAC_POWER_NAME,
                "area": _HVAC_POWER_AREA,
                "value": 1,
            }
        if r["method"] == "unwritable":
            entry["notes"] = "all write strategies failed in self_test"
        by_prop.setdefault(r["name"], {"areas": {}})["areas"][r["area"]] = entry
    summary = dict(vhal_results["by_method"])
    summary["total"] = vhal_results["tested"]
    output = {"summary": summary, "by_property": by_prop}
    _METHODS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _METHODS_PATH.write_text(json.dumps(output, indent=2))
    return output


def test_sensors() -> dict:
    results = {"passed": 0, "failed": 0, "details": []}
    cases = [
        ("temperature", (25.0,)),
        ("humidity", (70.0,)),
        ("acceleration", (0.0, 0.0, 9.81)),
    ]
    for name, values in cases:
        ok_set = set_sensor(name, *values)
        time.sleep(0.05)
        back = get_sensor(name)
        ok_final = bool(ok_set and back is not None)
        results["details"].append(
            f"{name}: set={ok_set}, get={back}, ok={ok_final}")
        results["passed" if ok_final else "failed"] += 1
    gps_ok = set_gps(-122.084, 37.422)
    results["details"].append(f"gps: set_gps={gps_ok}")
    results["passed" if gps_ok else "failed"] += 1
    return results


def test_driving() -> dict:
    results = {"passed": 0, "failed": 0, "details": []}
    for state in ["park", "drive", "park"]:
        ok = emulate_driving_state(state)
        results["details"].append(f"driving_state({state}): {ok}")
        results["passed" if ok else "failed"] += 1
    for mode in ["day", "night", "day"]:
        ok = set_day_night(mode)
        results["details"].append(f"day_night({mode}): {ok}")
        results["passed" if ok else "failed"] += 1
    return results


def _bar(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    if not _CATALOG_PATH.exists():
        raise SystemExit(f"catalog missing: {_CATALOG_PATH}; "
                         f"run: python -m automotive_env.sensors.catalog")

    _bar("[1/3] VHAL RW Property Self-Test (smart probe)")
    vhal_r = test_vhal_rw()
    output = write_methods_json(vhal_r)
    print(f"\n[self_test] wrote {_METHODS_PATH}")

    _bar("VHAL Write Methods Breakdown")
    for method, count in sorted(vhal_r["by_method"].items(), key=lambda x: -x[1]):
        print(f"  {method:24s} {count:4d}")
    print(f"  {'TOTAL':24s} {vhal_r['tested']:4d}")
    writable = (vhal_r["by_method"].get("direct_set", 0)
                + vhal_r["by_method"].get("inject", 0)
                + vhal_r["by_method"].get("set_after_hvac_power", 0)
                + vhal_r["by_method"].get("dumpsys_ivehicle", 0))
    print(f"\n  writable (round-trip verified): {writable}/{vhal_r['tested']}")

    unwritable = [r for r in vhal_r["per_property"] if r["method"] == "unwritable"]
    if unwritable:
        print(f"\nUnwritable properties ({len(unwritable)}):")
        for r in unwritable:
            last = r["attempts"][-1]["reason"] if r["attempts"] else "-"
            print(f"  - {r['name']}[{r['area']}]  last={last}")

    _bar("[2/3] Sensor Self-Test")
    sensor_r = test_sensors()
    for d in sensor_r["details"]:
        print(f"  {d}")
    print(f"\npassed={sensor_r['passed']}, failed={sensor_r['failed']}")

    _bar("[3/3] Driving State Self-Test")
    drive_r = test_driving()
    for d in drive_r["details"]:
        print(f"  {d}")
    print(f"\npassed={drive_r['passed']}, failed={drive_r['failed']}")

    _bar("SUMMARY")
    print(f"  VHAL writable: {writable}/{vhal_r['tested']} "
          f"(direct_set={vhal_r['by_method'].get('direct_set', 0)}, "
          f"inject={vhal_r['by_method'].get('inject', 0)}, "
          f"set_after_hvac_power={vhal_r['by_method'].get('set_after_hvac_power', 0)}, "
          f"dumpsys_ivehicle={vhal_r['by_method'].get('dumpsys_ivehicle', 0)})")
    print(f"  Sensor: {sensor_r['passed']}/{sensor_r['passed']+sensor_r['failed']} passed")
    print(f"  Driving: {drive_r['passed']}/{drive_r['passed']+drive_r['failed']} passed")


if __name__ == "__main__":
    main()
