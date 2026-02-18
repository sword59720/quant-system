#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.notify_wecom import send_wecom_message


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    runtime = load_yaml("config/runtime.yaml")
    if not runtime.get("enabled", True):
        print("[health-notify] system disabled")
        return 0

    log_file = os.path.join(runtime["paths"]["output_dir"], "reports", "healthcheck_latest.log")
    if not os.path.exists(log_file):
        print("[health-notify] no healthcheck file")
        return 0

    text = open(log_file, "r", encoding="utf-8", errors="ignore").read().strip()
    if not text:
        print("[health-notify] empty")
        return 0

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    key = [ln for ln in lines if re.search(r"Traceback|ERROR|failed", ln, re.I)]
    if not key:
        print("[health-notify] no alert keywords")
        return 0

    summary = "\n".join(key[:3])
    ok, msg = send_wecom_message(
        f"检测到异常日志（前3条）:\n{summary}",
        title="异常告警",
        dedup_key="healthcheck_alert",
        dedup_hours=24,
    )
    print(f"[health-notify] {'ok' if ok else 'fail'}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
