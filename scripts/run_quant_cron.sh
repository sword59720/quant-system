#!/usr/bin/env bash
set -u

PROJECT_ROOT="/home/node/projects/quant-system"
LOG_FILE="${PROJECT_ROOT}/logs/cron.log"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
CRON_TIMEOUT_SEC="${CRON_TIMEOUT_SEC:-10800}"   # 默认 3 小时

mkdir -p "${PROJECT_ROOT}/logs"

notify() {
  local title="$1"
  local content="$2"
  "${PYTHON_BIN}" - <<'PY' "$title" "$content"
import sys
from core.notify_wecom import send_wecom_message

title = sys.argv[1]
content = sys.argv[2]
ok, detail = send_wecom_message(content, title=title)
print(f"[cron-notify] {'ok' if ok else 'fail'}: {detail}")
PY
}

cd "${PROJECT_ROOT}" || exit 1
start_epoch=$(date +%s)
start_human=$(date '+%F %T %Z')

echo "[$(date '+%F %T')] [cron] run_quant start" >> "${LOG_FILE}"
notify "量化定时任务开始" "[提醒] 定时任务已启动\n时间: ${start_human}\n任务: scripts/run_quant.py"

"${PYTHON_BIN}" -m py_compile scripts/stock_single/fetch_stock_single_data.py >/dev/null 2>&1
if [ $? -ne 0 ]; then
  err_msg="[提醒] 任务启动失败：语法检查未通过（fetch_stock_single_data.py）"
  echo "[$(date '+%F %T')] [cron] precheck failed" >> "${LOG_FILE}"
  notify "量化定时任务失败" "${err_msg}"
  exit 2
fi

set +e
timeout --foreground "${CRON_TIMEOUT_SEC}"s "${PYTHON_BIN}" scripts/run_quant.py --stock-etf true --stock-single false --crypto false >> "${LOG_FILE}" 2>&1
rc=$?
set -e

end_epoch=$(date +%s)
elapsed=$((end_epoch - start_epoch))
end_human=$(date '+%F %T %Z')

if [ ${rc} -eq 0 ]; then
  echo "[$(date '+%F %T')] [cron] run_quant success elapsed=${elapsed}s" >> "${LOG_FILE}"
  notify "量化定时任务完成" "[提醒] 定时任务执行完成\n开始: ${start_human}\n结束: ${end_human}\n耗时: ${elapsed}s"
  exit 0
elif [ ${rc} -eq 124 ]; then
  echo "[$(date '+%F %T')] [cron] run_quant timeout elapsed=${elapsed}s" >> "${LOG_FILE}"
  notify "量化定时任务超时" "[提醒] 定时任务执行超时并被中止\n开始: ${start_human}\n结束: ${end_human}\n超时阈值: ${CRON_TIMEOUT_SEC}s\n已运行: ${elapsed}s"
  exit 124
else
  echo "[$(date '+%F %T')] [cron] run_quant failed rc=${rc} elapsed=${elapsed}s" >> "${LOG_FILE}"
  notify "量化定时任务失败" "[提醒] 定时任务执行失败\n开始: ${start_human}\n结束: ${end_human}\n退出码: ${rc}\n耗时: ${elapsed}s"
  exit ${rc}
fi
