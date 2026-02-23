#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEMPLATE_FILE="${REPO_ROOT}/deploy/raspi/cron/quant-system.cron.tpl"

TARGET_ROOT=""
PYTHON_BIN=""
OUTPUT_FILE="/tmp/quant-system.cron"
APPLY="false"

usage() {
  cat <<'EOF'
Usage:
  scripts/install_raspi_cron.sh [--target-root PATH] [--python PATH] [--output FILE] [--apply]

Options:
  --target-root PATH   Project absolute path on Raspberry Pi.
                       Default: current repo root.
  --python PATH        Python path used in cron.
                       Default: <target-root>/.venv/bin/python
  --output FILE        Rendered cron file path.
                       Default: /tmp/quant-system.cron
  --apply              Install rendered cron using crontab.
  --help               Show this help.

Examples:
  # Render only
  scripts/install_raspi_cron.sh --target-root /home/pi/quant-system

  # Render and install
  scripts/install_raspi_cron.sh \
    --target-root /home/pi/quant-system \
    --python /home/pi/quant-system/.venv/bin/python \
    --apply
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-root)
      TARGET_ROOT="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="${2:-}"
      shift 2
      ;;
    --apply)
      APPLY="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[cron-install] unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${TARGET_ROOT}" ]]; then
  TARGET_ROOT="${REPO_ROOT}"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${TARGET_ROOT}/.venv/bin/python"
fi

if [[ ! -f "${TEMPLATE_FILE}" ]]; then
  echo "[cron-install] template not found: ${TEMPLATE_FILE}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

esc_root="$(printf '%s' "${TARGET_ROOT}" | sed -e 's/[\\/&]/\\&/g')"
esc_python="$(printf '%s' "${PYTHON_BIN}" | sed -e 's/[\\/&]/\\&/g')"

sed \
  -e "s#__ROOT__#${esc_root}#g" \
  -e "s#__PYTHON__#${esc_python}#g" \
  "${TEMPLATE_FILE}" > "${OUTPUT_FILE}"

if [[ -d "${TARGET_ROOT}" || "${APPLY}" == "true" ]]; then
  mkdir -p \
    "${TARGET_ROOT}/logs" \
    "${TARGET_ROOT}/outputs/orders" \
    "${TARGET_ROOT}/outputs/state" \
    "${TARGET_ROOT}/outputs/reports"
else
  echo "[cron-install] target-root not found, skip runtime directory bootstrap: ${TARGET_ROOT}"
fi

echo "[cron-install] rendered: ${OUTPUT_FILE}"
echo "[cron-install] target-root: ${TARGET_ROOT}"
echo "[cron-install] python: ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[cron-install] warning: python not executable at ${PYTHON_BIN}" >&2
fi

if [[ "${APPLY}" == "true" ]]; then
  crontab "${OUTPUT_FILE}"
  echo "[cron-install] installed via crontab"
  crontab -l
else
  echo "[cron-install] not installed (render only). Use --apply to install."
fi
