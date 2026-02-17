#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/haojc/.openclaw/workspace/quant-system"
CFG="$ROOT/config/runtime.yaml"

if [[ ! -f "$CFG" ]]; then
  echo "[stop] runtime config not found: $CFG"
  exit 1
fi

sed -i 's/^enabled:.*/enabled: false/' "$CFG"
echo "[stop] quant-system disabled"
grep '^enabled:' "$CFG"
