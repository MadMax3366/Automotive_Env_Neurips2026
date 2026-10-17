#!/usr/bin/env bash
# Drop the wlan0 route (10.0.2.0/24) on every running emulator so that
# 10.0.2.x traffic (host gateway, VHAL gRPC, ADB reverse, etc.) goes
# through eth0 — the AAOS image registers both routes at boot and the
# more-specific wlan0 entry hijacks the channel.
#
# Usage:
#   scripts/fix_emulator_route.sh                 # all running emulators
#   scripts/fix_emulator_route.sh 5554 5556       # explicit console ports
#   scripts/fix_emulator_route.sh emulator-5554   # explicit device names

set -euo pipefail

if ! command -v adb >/dev/null 2>&1; then
    echo "ERROR: adb not on PATH" >&2
    exit 2
fi

# ---- pick targets ----------------------------------------------------------
declare -a targets=()
if [[ $# -gt 0 ]]; then
    for arg in "$@"; do
        if [[ "$arg" =~ ^[0-9]+$ ]]; then
            targets+=("emulator-${arg}")
        else
            targets+=("$arg")
        fi
    done
else
    while IFS= read -r line; do
        targets+=("$line")
    done < <(adb devices | awk '$1 ~ /^emulator-[0-9]+$/ && $2 == "device" {print $1}')
fi

if [[ ${#targets[@]} -eq 0 ]]; then
    echo "No running emulators found (adb devices listed none)." >&2
    exit 1
fi

# ---- per-device fix --------------------------------------------------------
fix_one() {
    local dev="$1"
    echo "=== ${dev} ==="

    if ! adb -s "$dev" get-state >/dev/null 2>&1; then
        echo "  [skip] not online"
        return
    fi

    # adb root is a no-op on already-rooted adbd; on userdebug it restarts
    # adbd and the connection drops briefly — wait-for-device handles that.
    adb -s "$dev" root >/dev/null 2>&1 || true
    adb -s "$dev" wait-for-device

    echo "  [before]"
    adb -s "$dev" shell ip route | sed 's/^/    /'

    # Idempotent: ignore "No such process" if the rule was already removed.
    adb -s "$dev" shell 'ip route del 10.0.2.0/24 dev wlan0 2>/dev/null; true'

    echo "  [after]"
    adb -s "$dev" shell ip route | sed 's/^/    /'
}

for dev in "${targets[@]}"; do
    fix_one "$dev"
done

echo "done."
