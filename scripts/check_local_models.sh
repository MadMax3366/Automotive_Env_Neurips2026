#!/usr/bin/env bash
# Audit local model storage:
#   - list every dir under ./models/ with size + safetensors shard count
#   - flag any HF cache hits (~/.cache/huggingface/hub/models--*) so you
#     know if a wrapper accidentally re-downloaded weights to cache
#     instead of local_dir
#   - report HF transformers_modules cache size (this is custom .py code
#     for trust_remote_code models — KB scale, not weights, expected)
#
# Usage:
#   scripts/check_local_models.sh
set -euo pipefail

echo "=========================================="
echo "Local model dirs (./models/)"
echo "=========================================="
if [[ -d models ]]; then
    for d in models/*/; do
        [[ -d "$d" ]] || continue
        name=$(basename "$d")
        sz=$(du -sh "$d" 2>/dev/null | cut -f1)
        n_shards=$(find "$d" -maxdepth 2 -name 'model-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
        n_single=$(find "$d" -maxdepth 2 -name 'model.safetensors' 2>/dev/null | wc -l | tr -d ' ')
        config="MISSING"
        [[ -f "$d/config.json" ]] && config="ok"
        printf "  %-30s  %6s  shards=%s  single=%s  config=%s\n" \
            "$name" "$sz" "$n_shards" "$n_single" "$config"
    done
else
    echo "  (no ./models/ dir)"
fi

echo
echo "=========================================="
echo "HF cache hits — weights that snuck into cache"
echo "=========================================="
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HUB_CACHE="$HF_HOME/hub"
if [[ -d "$HUB_CACHE" ]]; then
    found_any=0
    for d in "$HUB_CACHE"/models--*; do
        [[ -d "$d" ]] || continue
        sz=$(du -sh "$d" 2>/dev/null | cut -f1)
        sz_bytes=$(du -sb "$d" 2>/dev/null | cut -f1)
        # Anything > 100 MB in hub cache is likely a real weight leak
        if [[ "${sz_bytes:-0}" -gt $((100 * 1024 * 1024)) ]]; then
            echo "  ⚠️  $sz   $(basename "$d")  -- weights in cache, not in ./models/"
            found_any=1
        fi
    done
    [[ $found_any -eq 0 ]] && echo "  ok — no large model snapshots in HF hub cache"
else
    echo "  no HF hub cache at $HUB_CACHE (clean state)"
fi

echo
echo "=========================================="
echo "transformers_modules cache (custom .py for trust_remote_code)"
echo "=========================================="
TRM="$HF_HOME/modules/transformers_modules"
if [[ -d "$TRM" ]]; then
    sz=$(du -sh "$TRM" 2>/dev/null | cut -f1)
    n=$(find "$TRM" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ')
    echo "  $sz across $n module dir(s)  (KB-scale .py code; expected for OpenCUA / MiMo-VL)"
    find "$TRM" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sed 's/^/    - /'
else
    echo "  no transformers_modules cache (no trust_remote_code model loaded yet)"
fi

echo
echo "=========================================="
echo "Disk free"
echo "=========================================="
df -h . | tail -1
