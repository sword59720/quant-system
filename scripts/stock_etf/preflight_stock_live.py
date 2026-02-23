#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实盘交易前置检查（面向树莓派运行环境）。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.stock_broker import SUPPORTED_STOCK_BROKERS, resolve_runtime_stock_broker
from core.stock_broker import resolve_strategy_account_config

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        out = yaml.safe_load(f)
    return out if isinstance(out, dict) else {}


def _has_text(x) -> bool:
    return bool(str(x or "").strip())


def _now_in_timezone(tz_name: str) -> datetime:
    tz = str(tz_name or "Asia/Shanghai").strip() or "Asia/Shanghai"
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz))
        except Exception:
            pass
    if tz == "Asia/Shanghai":
        return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    return datetime.now()


def _ensure_dir(path: str) -> tuple[bool, str]:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return True, f"可写: {path}"
    except Exception as e:
        return False, f"不可写: {path} ({e})"


def _check_file_exists(path: str) -> tuple[bool, str]:
    if os.path.exists(path):
        return True, f"存在: {path}"
    return False, f"缺失: {path}"


def _add(checks: list, item: str, ok: bool, message: str, level: str = "error"):
    checks.append(
        {
            "item": item,
            "ok": bool(ok),
            "level": level,
            "message": message,
        }
    )


def run_preflight(runtime_file: str, broker_file: str) -> dict:
    checks: list[dict] = []

    runtime = _load_yaml(runtime_file)
    brokers = _load_yaml(broker_file)

    enabled = bool(runtime.get("enabled", True))
    env = str(runtime.get("env", "paper")).strip().lower()
    broker, broker_source = resolve_runtime_stock_broker(runtime, strategy="stock_etf")
    tz_name = str(runtime.get("timezone", "Asia/Shanghai")).strip() or "Asia/Shanghai"
    total_capital = float(runtime.get("total_capital", 0) or 0)

    _add(checks, "runtime.enabled", enabled, f"enabled={enabled}", level="warning")
    _add(checks, "runtime.env", env in {"paper", "live"}, f"env={env}")

    now_tz = _now_in_timezone(tz_name)
    _add(
        checks,
        "runtime.timezone",
        True,
        f"timezone={tz_name}, now={now_tz.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        level="warning",
    )
    if tz_name != "Asia/Shanghai":
        _add(
            checks,
            "runtime.timezone.recommend",
            False,
            "A股实盘建议使用 timezone=Asia/Shanghai",
            level="warning",
        )

    if env == "live":
        _add(
            checks,
            "runtime.stock_broker.stock_etf.required",
            _has_text(broker),
            "live 模式必须在 config/runtime.yaml 设置 stock_brokers.stock_etf（或兼容字段 broker）",
        )
        _add(
            checks,
            "runtime.stock_broker.stock_etf.value",
            broker in SUPPORTED_STOCK_BROKERS,
            f"broker={broker or '<empty>'}",
        )
    else:
        if not _has_text(broker):
            _add(
                checks,
                "runtime.stock_broker.stock_etf.default",
                True,
                "paper 模式未显式 broker，将使用默认 guotou 模拟路径",
                level="warning",
            )
    _add(
        checks,
        "runtime.stock_broker.stock_etf.source",
        True,
        f"source={broker_source or 'default(guotou)'}",
        level="warning",
    )

    _add(
        checks,
        "runtime.total_capital",
        total_capital > 0,
        f"total_capital={total_capital}",
    )
    if total_capital < 1000:
        _add(
            checks,
            "runtime.total_capital.min",
            False,
            "资金规模偏小，实盘可能产生不足最小交易单位的订单",
            level="warning",
        )

    paths = runtime.get("paths", {})
    data_dir = str(paths.get("data_dir", "./data"))
    out_dir = str(paths.get("output_dir", "./outputs"))
    log_dir = str(paths.get("log_dir", "./logs"))

    for p in [data_dir, out_dir, log_dir, os.path.join(out_dir, "orders"), os.path.join(out_dir, "state"), os.path.join(out_dir, "reports")]:
        ok, msg = _ensure_dir(p)
        _add(checks, f"path:{p}", ok, msg)

    for p in [
        "scripts/stock_etf/fetch_stock_etf_data.py",
        "scripts/stock_etf/run_stock_etf.py",
        "scripts/stock_etf/generate_trades_stock_etf.py",
        "scripts/stock_etf/execute_trades_stock_etf.py",
        "scripts/stock_etf/trade_stock_etf.py",
    ]:
        ok, msg = _check_file_exists(p)
        _add(checks, f"script:{p}", ok, msg)

    if env == "live":
        _add(
            checks,
            "broker.yaml",
            os.path.exists(broker_file),
            f"需要券商配置文件: {broker_file}",
        )

        broker_cfg, account_source = resolve_strategy_account_config(
            brokers,
            broker=broker,
            strategy="stock_etf",
        )
        _add(
            checks,
            "runtime.stock_account.stock_etf.source",
            True,
            f"source={account_source or '<not_found>'}",
            level="warning",
        )

        if broker == "myquant":
            _add(checks, "myquant.token", _has_text(broker_cfg.get("token")), "myquant.token 不能为空")
            _add(checks, "myquant.account_id", _has_text(broker_cfg.get("account_id")), "myquant.account_id 不能为空")
            has_gm = importlib.util.find_spec("gm") is not None
            _add(checks, "myquant.sdk", has_gm, "需要安装掘金 SDK: pip install gm")

        if broker == "guotou":
            platform = str(broker_cfg.get("platform", "emp")).strip().lower()
            hosting_mode = str(broker_cfg.get("emp", {}).get("hosting_mode", "signal")).strip().lower()
            _add(checks, "guotou.account_id", _has_text(broker_cfg.get("account_id")), "guotou.account_id 不能为空")
            if platform == "emp" and hosting_mode == "signal":
                _add(
                    checks,
                    "guotou.emp.signal",
                    False,
                    "当前代码未实现 guotou EMP signal 实盘连接，请切到 stock_brokers.stock_etf=myquant 或提供 guotou live API 适配",
                )
            else:
                _add(checks, "guotou.platform", True, f"platform={platform}, hosting_mode={hosting_mode}", level="warning")

    errors = [x for x in checks if (not x["ok"]) and x.get("level", "error") == "error"]
    warnings = [x for x in checks if (not x["ok"]) and x.get("level", "error") == "warning"]
    passed = len(errors) == 0

    return {
        "ts": datetime.now().isoformat(),
        "runtime_file": runtime_file,
        "broker_file": broker_file,
        "passed": passed,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="stock_etf 实盘前置检查")
    parser.add_argument("--runtime", default="./config/runtime.yaml", help="runtime 配置路径")
    parser.add_argument("--broker", default="./config/broker.yaml", help="broker 配置路径")
    parser.add_argument("--json-out", default="", help="输出 JSON 报告路径")
    parser.add_argument("--strict", action="store_true", help="warning 也视为失败")
    args = parser.parse_args()

    report = run_preflight(args.runtime, args.broker)
    checks = report.get("checks", [])
    errors = [x for x in checks if (not x["ok"]) and x.get("level", "error") == "error"]
    warnings = [x for x in checks if (not x["ok"]) and x.get("level", "error") == "warning"]

    print("=" * 60)
    print("stock_etf 实盘前置检查")
    print("=" * 60)
    for c in checks:
        mark = "PASS" if c["ok"] else ("WARN" if c.get("level") == "warning" else "FAIL")
        print(f"[{mark}] {c['item']}: {c['message']}")
    print("-" * 60)
    print(f"passed={report['passed']} errors={len(errors)} warnings={len(warnings)}")

    if args.json_out:
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"report: {args.json_out}")

    if errors:
        return 2
    if args.strict and warnings:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
