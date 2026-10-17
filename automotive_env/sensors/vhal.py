"""Low-level VHAL wrapper via `adb shell cmd car_service`.

Function style, no classes. Every call spawns a fresh adb process.
Helpers return Python values / bool; never raise. On any failure they
print a warning and return None/False.
"""
import json
import os
import re
import subprocess
import threading
from pathlib import Path

_CATALOG = None
_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "vhal_catalog.json"
# Per-serial root-check cache (thread-safe for parallel workers).
_ROOT_STATE: dict = {}
_ROOT_LOCK = threading.Lock()
_IVEHICLE_SVC = "android.hardware.automotive.vehicle.IVehicle/default"

# Ground-truth dump of the running HAL's property table; used to discover
# which properties are gated by HVAC_POWER_ON.
_CAPABILITY_PATH = Path(__file__).resolve().parents[2] / "vhal_capability.txt"
_HVAC_POWER_NAME = "HVAC_POWER_ON"
_HVAC_POWER_DEPENDENTS: set | None = None


def _load_hvac_power_dependents() -> set:
    """Return property IDs (ints) that require HVAC_POWER_ON=1 before they
    will accept writes or reads.

    Per AAOS VHAL spec the HVAC_POWER_ON config[] array lists every
    property that is gated on it. We parse that directly from
    vhal_capability.txt (the capability dump collected from the running
    HAL). Empty set if the file is missing — behaviour is unchanged for
    anyone who never probed their HAL.
    """
    global _HVAC_POWER_DEPENDENTS
    if _HVAC_POWER_DEPENDENTS is not None:
        return _HVAC_POWER_DEPENDENTS
    deps: set = set()
    try:
        for line in _CAPABILITY_PATH.read_text().splitlines():
            if "HVAC_POWER_ON" in line and "config:[" in line:
                m = re.search(r"config:\[([^\]]*)\]", line)
                if m and m.group(1).strip():
                    deps = {int(x.strip()) for x in m.group(1).split(",")
                            if x.strip()}
                break
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001 - diagnostic only
        print(f"[vhal] warning: failed to parse hvac_power dependents: {e}")
    _HVAC_POWER_DEPENDENTS = deps
    return deps


def _adb_base() -> list:
    """Return base adb command, prefixed with -s <serial> if ANDROID_SERIAL is set."""
    serial = os.environ.get("ANDROID_SERIAL", "")
    if serial:
        return ["adb", "-s", serial]
    return ["adb"]


def _load_catalog() -> dict:
    global _CATALOG
    if _CATALOG is None:
        if not _CATALOG_PATH.exists():
            print(f"[vhal] catalog missing: {_CATALOG_PATH}. "
                  f"Run: python -m automotive_env.sensors.catalog")
            _CATALOG = {}
        else:
            _CATALOG = json.loads(_CATALOG_PATH.read_text())
    return _CATALOG


def _adb_cmd(*args, timeout: int = 10):
    cmd = _adb_base() + ["shell", "cmd", "car_service"] + list(args)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except FileNotFoundError:
        return "", "adb not found", -1


def _format_value(value, vtype: str) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if vtype == "float":
        return f"{float(value):.6f}"
    return str(int(value))


def _ensure_adb_root() -> bool:
    """Check `adb root` once per serial; cache result. Needed for Tier 2 writes."""
    serial = os.environ.get("ANDROID_SERIAL", "_default")
    with _ROOT_LOCK:
        if serial in _ROOT_STATE:
            return _ROOT_STATE[serial]
        try:
            r = subprocess.run(_adb_base() + ["root"],
                               capture_output=True, text=True, timeout=10)
        except Exception as e:
            print(f"[vhal] adb root exception: {e}")
            _ROOT_STATE[serial] = False
            return False
        output = (r.stdout + r.stderr).lower()
        ok = ("root" in output
              and "cannot" not in output
              and "not allowed" not in output)
        _ROOT_STATE[serial] = ok
        if not ok:
            print(f"[vhal] warning: adb root failed "
                  f"({(r.stderr or r.stdout).strip()[:120]}). "
                  f"Tier 2 write (dumpsys) will be unavailable.")
        return ok


def _try_cmd_car_service(meta: dict, area: int, value) -> bool:
    """Tier 1 write via `cmd car_service set-property-value`."""
    val_s = _format_value(value, meta["value_type"])
    out, _err, _ = _adb_cmd("set-property-value",
                            str(meta["id"]), str(area), val_s)
    return "successfully" in out.lower()


def _try_dumpsys_ivehicle(meta: dict, area: int, value) -> bool:
    """Tier 2 write via `dumpsys IVehicle/default --set` (requires adb root).

    Uses hex property ID; name is not accepted by dumpsys --set.
    """
    hex_id = meta["id_hex"]
    vtype = meta["value_type"]
    cmd = _adb_base() + ["shell", "dumpsys", _IVEHICLE_SVC,
                         "--set", hex_id, "-a", str(area)]
    if vtype == "float":
        cmd += ["-f", str(float(value))]
    else:
        cmd += ["-i", str(int(value))]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception as e:
        print(f"[vhal] dumpsys exception: {e}")
        return False
    out = r.stdout + r.stderr
    low = out.lower()
    if "not a valid" in low or "non-integer" in low:
        print(f"[vhal] dumpsys parse error for {hex_id}: {out.strip()[:200]}")
        return False
    if "set property:" in low and "error" not in low:
        return True
    return False


def _value_equals(got, expected, vtype: str) -> bool:
    if got is None:
        return False
    try:
        if vtype == "float":
            return abs(float(got) - float(expected)) < 0.01
        return int(got) == int(expected)
    except (TypeError, ValueError):
        return got == expected


def _readback_matches(name: str, area: int, expected, meta: dict) -> bool:
    """Post-write read-back check.

    Needed because `cmd car_service set-property-value` sometimes reports
    "... set to X successfully" even when the underlying HAL silently
    rejects or overrides the value (observed on HVAC_POWER_ON on the
    GPU-server AAOS image). Without this verify step, set_vhal would
    return True on those writes and never fall back to tier-2 dumpsys.
    """
    got = get_vhal(name, area)
    return _value_equals(got, expected, meta.get("value_type", "int32"))


def _ensure_hvac_power_on(area: int) -> bool:
    """Best-effort enable HVAC_POWER_ON at `area`. Called automatically
    before touching any property listed in HVAC_POWER_ON.config[]."""
    catalog = _load_catalog()
    meta = catalog.get(_HVAC_POWER_NAME)
    if not meta:
        return False
    if (_try_cmd_car_service(meta, area, 1)
            and _readback_matches(_HVAC_POWER_NAME, area, 1, meta)):
        return True
    if (_ensure_adb_root()
            and _try_dumpsys_ivehicle(meta, area, 1)
            and _readback_matches(_HVAC_POWER_NAME, area, 1, meta)):
        return True
    return False


def set_vhal(name: str, area: int, value) -> bool:
    """Write VHAL property with two-tier fallback + HVAC_POWER_ON precondition.

    Tier 1: `cmd car_service set-property-value` (covers RW properties).
    Tier 2: `dumpsys IVehicle/default --set` (requires adb root, covers
            R-only / physical-state properties unreachable via car_service).

    Precondition: if the property is declared as a dependent in the HAL's
    HVAC_POWER_ON config array (see vhal_capability.txt), HVAC_POWER_ON is
    auto-set to 1 on the same area first. Without this, gate-dependent
    props like HVAC_FAN_SPEED / HVAC_FAN_DIRECTION return NOT_AVAILABLE
    for both set and get while HVAC power is off.
    """
    catalog = _load_catalog()
    if name not in catalog:
        print(f"[vhal] warning: {name} not in catalog")
        return False
    meta = catalog[name]

    if name != _HVAC_POWER_NAME and meta["id"] in _load_hvac_power_dependents():
        _ensure_hvac_power_on(area)  # best-effort; downstream will report if it failed

    # Tier 1: verify with read-back to weed out CarService false positives.
    if (_try_cmd_car_service(meta, area, value)
            and _readback_matches(name, area, value, meta)):
        return True

    # Tier 2: dumpsys IVehicle --set (requires adb root).
    if (_ensure_adb_root()
            and _try_dumpsys_ivehicle(meta, area, value)
            and _readback_matches(name, area, value, meta)):
        return True

    print(f"[vhal] set {name}[{hex(area)}]={value} failed (both tiers)")
    return False


def get_vhal(name: str, area: int = 0):
    """Run `car_service get-property-value <id> <area>`; return Python value or None.

    HVAC_POWER_ON gating: if the property is a declared dependent in
    HVAC_POWER_ON.config[], auto-enable HVAC_POWER_ON first so the get
    doesn't silently return None. Without this, validators reading
    HVAC_FAN_SPEED / HVAC_FAN_DIRECTION after an agent interaction get
    None and score 0, even if the agent set the right value.
    """
    catalog = _load_catalog()
    if name not in catalog:
        print(f"[vhal] warning: {name} not in catalog")
        return None
    meta = catalog[name]

    if name != _HVAC_POWER_NAME and meta["id"] in _load_hvac_power_dependents():
        _ensure_hvac_power_on(area)

    out, err, _ = _adb_cmd("get-property-value", str(meta["id"]), str(area))

    low = out.lower()
    if ("can not get property value" in low
            or "not supported" in low
            or "not available" in low
            or "no car property" in low
            or "exception" in (err or "").lower()):
        return None

    vtype = meta["value_type"]
    if vtype == "float":
        m = re.search(r"floatValues:\s*\[([^\]]*)\]", out)
        if m and m.group(1).strip():
            first = m.group(1).split(",")[0].strip()
            try:
                return float(first)
            except ValueError:
                pass
    m = re.search(r"int32Values:\s*\[([^\]]*)\]", out)
    if m and m.group(1).strip():
        first = m.group(1).split(",")[0].strip()
        try:
            return int(first)
        except ValueError:
            pass
    # fallback: int64 or string
    m = re.search(r"int64Values:\s*\[([^\]]*)\]", out)
    if m and m.group(1).strip():
        first = m.group(1).split(",")[0].strip()
        try:
            return int(first)
        except ValueError:
            pass
    m = re.search(r"stringValue(?:s)?:\s*(\S+)", out)
    if m:
        s = m.group(1).rstrip(",")
        return None if s in ("[]", "") else s
    return None


def inject_vhal(name: str, area: int, value) -> bool:
    """Run `car_service inject-vhal-event <id> <area> <value>` (for R-only props)."""
    catalog = _load_catalog()
    if name not in catalog:
        print(f"[vhal] warning: {name} not in catalog")
        return False
    meta = catalog[name]
    val_s = _format_value(value, meta["value_type"])
    out, err, _ = _adb_cmd("inject-vhal-event",
                           str(meta["id"]), str(area), val_s)
    low_err = (err or "").lower()
    if "exception" in low_err or "error" in low_err:
        print(f"[vhal] inject {name}[{hex(area)}]={val_s} failed: "
              f"{err.strip()[:200]}")
        return False
    # inject-vhal-event is silent on success
    return True
