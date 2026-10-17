"""Driving state / day-night wrappers via `adb shell cmd car_service`."""
import os
import subprocess


def _adb_base() -> list:
    serial = os.environ.get("ANDROID_SERIAL", "")
    return ["adb", "-s", serial] if serial else ["adb"]


def _car_cmd(*args, timeout: int = 10):
    try:
        r = subprocess.run(
            _adb_base() + ["shell", "cmd", "car_service", *args],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "timeout", -1
    except FileNotFoundError:
        return "", "adb not found", -1


def emulate_driving_state(mode: str) -> bool:
    """mode: drive / park / reverse / neutral"""
    valid = {"drive", "park", "reverse", "neutral"}
    if mode not in valid:
        print(f"[driving] invalid mode: {mode} (expected {sorted(valid)})")
        return False
    out, err, rc = _car_cmd("emulate-driving-state", mode)
    if rc == 0 and "exception" not in (err or "").lower():
        return True
    print(f"[driving] emulate-driving-state {mode} failed: {(err or out).strip()[:200]}")
    return False


def set_day_night(mode: str) -> bool:
    """mode: day / night / sensor"""
    valid = {"day", "night", "sensor"}
    if mode not in valid:
        print(f"[driving] invalid mode: {mode} (expected {sorted(valid)})")
        return False
    out, err, rc = _car_cmd("day-night-mode", mode)
    if rc == 0 and "exception" not in (err or "").lower():
        return True
    print(f"[driving] day-night-mode {mode} failed: {(err or out).strip()[:200]}")
    return False
