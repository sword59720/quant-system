#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""发送 Crypto 每日评估（收益率/Sharpe/最大回撤）到企业微信。"""

import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_OK, EXIT_OUTPUT_ERROR


def _pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "N/A"


def main() -> int:
    report = "./outputs/reports/backtest_crypto_report.json"
    if not os.path.exists(report):
        print(f"[crypto-backtest-notify] report not found: {report}")
        return EXIT_OUTPUT_ERROR

    try:
        with open(report, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"[crypto-backtest-notify] report load failed: {e}")
        return EXIT_OUTPUT_ERROR

    c = (obj or {}).get("crypto", {}) or {}
    if "error" in c:
        content = (
            f"时间: {datetime.now().isoformat()}\n"
            f"Crypto每日评估失败: {c.get('error')}"
        )
    else:
        content = "\n".join([
            f"时间: {datetime.now().isoformat()}",
            "Crypto每日纸面评估:",
            f"- 年化收益率: {_pct(c.get('annual_return'))}",
            f"- Sharpe: {c.get('sharpe', 'N/A')}",
            f"- 最大回撤: {_pct(c.get('max_drawdown'))}",
            f"- 最终净值: {c.get('final_nav', 'N/A')}",
            f"- 周期数: {c.get('periods', 'N/A')}",
        ])

    try:
        from core.notify_wecom import send_wecom_message
        dedup_key = f"crypto_daily_eval:{datetime.now().date().isoformat()}"
        ok, detail = send_wecom_message(
            content=content,
            title="Crypto每日评估",
            dedup_key=dedup_key,
            dedup_hours=12,
        )
        print(f"[crypto-backtest-notify] wecom {'ok' if ok else 'fail'}: {detail}")
        return EXIT_OK if ok else EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[crypto-backtest-notify] send failed: {e}")
        return EXIT_OUTPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
