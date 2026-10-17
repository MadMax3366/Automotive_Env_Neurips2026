"""Android emulator console (telnet) wrapper for sensors / GPS.

Each call opens a fresh connection, authenticates if required,
sends the commands, drains the response, closes.
"""
import os
import socket
from pathlib import Path


def _auth_token():
    path = Path.home() / ".emulator_console_auth_token"
    try:
        return path.read_text().strip() if path.exists() else None
    except Exception:
        return None


def _drain(sock: socket.socket, total_timeout: float = 1.0) -> str:
    sock.settimeout(total_timeout)
    data = bytearray()
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            break
        except OSError:
            break
        if not chunk:
            break
        data.extend(chunk)
    return data.decode("utf-8", errors="ignore")


def _telnet_cmd(commands, host: str = "localhost", port: int = None,
                timeout: float = 3.0) -> str:
    if port is None:
        port = int(os.environ.get("EMULATOR_CONSOLE_PORT", 5554))
    try:
        s = socket.create_connection((host, port), timeout=timeout)
    except OSError as e:
        print(f"[sensor] telnet connect failed: {e}")
        return ""
    try:
        banner = _drain(s, total_timeout=0.6)
        if "Authentication required" in banner:
            token = _auth_token()
            if not token:
                print("[sensor] telnet auth required but no token file")
                return banner
            s.sendall(f"auth {token}\n".encode())
            banner += _drain(s, total_timeout=0.6)
            if "KO" in banner.split("auth ")[-1]:
                print(f"[sensor] telnet auth failed")
                return banner
        out = banner
        for c in commands:
            s.sendall(f"{c}\n".encode())
            out += _drain(s, total_timeout=0.4)
        try:
            s.sendall(b"quit\n")
        except OSError:
            pass
        out += _drain(s, total_timeout=0.2)
        return out
    finally:
        try:
            s.close()
        except OSError:
            pass


def set_sensor(name: str, *values) -> bool:
    """name: temperature / humidity / acceleration / gyroscope / ..."""
    vs = ":".join(str(v) for v in values)
    out = _telnet_cmd([f"sensor set {name} {vs}"])
    # Emulator replies OK on success, "KO: ..." on failure.
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    for line in reversed(lines):
        if line.startswith("OK"):
            return True
        if line.startswith("KO"):
            print(f"[sensor] set {name}={vs}: {line}")
            return False
    print(f"[sensor] set {name}={vs}: no OK/KO in response")
    return False


def get_sensor(name: str):
    """Return float for 1-value sensors, tuple for multi-value, or None."""
    out = _telnet_cmd([f"sensor get {name}"])
    # Expected line: "<name> = v" or "<name> = v1:v2:v3"
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith(name):
            continue
        if "=" not in line:
            continue
        val = line.split("=", 1)[1].strip()
        parts = val.split(":")
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            return None
        return nums[0] if len(nums) == 1 else tuple(nums)
    return None


def set_gps(lng: float, lat: float, alt: float = 0.0) -> bool:
    out = _telnet_cmd([f"geo fix {lng} {lat} {alt}"])
    for line in reversed([l.strip() for l in out.splitlines() if l.strip()]):
        if line.startswith("OK"):
            return True
        if line.startswith("KO"):
            print(f"[sensor] geo fix: {line}")
            return False
    return False
