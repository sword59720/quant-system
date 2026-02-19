#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Optional

import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import EXIT_OK


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _fmt_metric_value(value: Optional[float], unit: str) -> str:
    if value is None:
        return "N/A"
    if unit == "pct":
        return f"{value * 100.0:.2f}%"
    if unit == "ms":
        return f"{value:.1f}ms"
    if unit == "bps":
        return f"{value:.2f}bps"
    return f"{value:.4f}"


def _merge_alert_cfg(user_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = {
        "enabled": False,
        "report_file": "./outputs/reports/execution_quality_daily.json",
        "min_orders": 1,
        "dedup_hours": 12,
        "thresholds": {
            "success_rate_min": 0.95,
            "fill_rate_min": 0.90,
            "reject_rate_max": 0.05,
            "p95_execution_latency_ms_max": 2000.0,
            "p95_abs_slippage_bps_max": 20.0,
        },
    }
    out = dict(base)
    u = user_cfg if isinstance(user_cfg, dict) else {}
    out["enabled"] = bool(u.get("enabled", base["enabled"]))
    out["report_file"] = str(u.get("report_file", base["report_file"]))
    out["min_orders"] = _safe_int(u.get("min_orders"), base["min_orders"])
    out["dedup_hours"] = _safe_int(u.get("dedup_hours"), base["dedup_hours"])
    out["thresholds"] = dict(base["thresholds"])
    if isinstance(u.get("thresholds"), dict):
        out["thresholds"].update(u["thresholds"])
    return out


def evaluate_execution_quality_alert(report: Dict[str, Any], alert_cfg: Dict[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary", {}) or {}
    thresholds = alert_cfg.get("thresholds", {}) or {}
    orders_total = _safe_int(summary.get("orders_total"), 0)
    min_orders = max(0, _safe_int(alert_cfg.get("min_orders"), 1))
    min_orders_reached = orders_total >= min_orders

    checks = [
        ("success_rate", "success_rate_min", "min", "成功率", "pct"),
        ("fill_rate", "fill_rate_min", "min", "成交率", "pct"),
        ("reject_rate", "reject_rate_max", "max", "拒单率", "pct"),
        ("p95_execution_latency_ms", "p95_execution_latency_ms_max", "max", "P95 延迟", "ms"),
        ("p95_abs_slippage_bps", "p95_abs_slippage_bps_max", "max", "P95 绝对滑点", "bps"),
    ]

    breaches = []
    for metric, thr_key, rule, label, unit in checks:
        val = _safe_float(summary.get(metric))
        thr = _safe_float(thresholds.get(thr_key))
        if val is None or thr is None:
            continue
        hit = (val < thr) if rule == "min" else (val > thr)
        if hit:
            breaches.append(
                {
                    "metric": metric,
                    "label": label,
                    "rule": rule,
                    "op": "<" if rule == "min" else ">",
                    "value": float(val),
                    "threshold": float(thr),
                    "unit": unit,
                    "value_text": _fmt_metric_value(float(val), unit),
                    "threshold_text": _fmt_metric_value(float(thr), unit),
                }
            )

    return {
        "date": str(report.get("date") or ""),
        "orders_total": int(orders_total),
        "min_orders": int(min_orders),
        "min_orders_reached": bool(min_orders_reached),
        "breaches": breaches,
        "should_alert": bool(min_orders_reached and len(breaches) > 0),
        "summary": summary,
    }


def build_alert_message(result: Dict[str, Any], report_file: str) -> str:
    summary = result.get("summary", {}) or {}
    lines = [
        f"日期: {result.get('date') or 'unknown'}",
        f"订单数: {result.get('orders_total', 0)}",
        f"成功率: {_fmt_metric_value(_safe_float(summary.get('success_rate')), 'pct')}",
        f"成交率: {_fmt_metric_value(_safe_float(summary.get('fill_rate')), 'pct')}",
        f"拒单率: {_fmt_metric_value(_safe_float(summary.get('reject_rate')), 'pct')}",
    ]
    if _safe_int(summary.get("latency_samples"), 0) > 0:
        lines.append(
            f"P95 延迟: {_fmt_metric_value(_safe_float(summary.get('p95_execution_latency_ms')), 'ms')}"
        )
    if _safe_int(summary.get("slippage_samples"), 0) > 0:
        lines.append(
            f"P95 绝对滑点: {_fmt_metric_value(_safe_float(summary.get('p95_abs_slippage_bps')), 'bps')}"
        )

    lines.append("触发阈值:")
    for b in result.get("breaches", []):
        lines.append(f"- {b['label']}: {b['value_text']} {b['op']} {b['threshold_text']}")
    lines.append(f"报告: {report_file}")
    return "\n".join(lines)


def _build_dedup_key(result: Dict[str, Any]) -> str:
    names = sorted(x.get("metric", "") for x in result.get("breaches", []))
    payload = f"{result.get('date', '')}|{','.join(names)}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"execution_quality_alert:{digest}"


def safe_send_wecom_message(content: str, title: str, dedup_key: str, dedup_hours: int):
    try:
        from core.notify_wecom import send_wecom_message
    except Exception as e:
        return False, f"notify module unavailable: {e}"
    return send_wecom_message(
        content=content,
        title=title,
        dedup_key=dedup_key,
        dedup_hours=dedup_hours,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Send WeCom alert for execution quality threshold breaches.")
    parser.add_argument("--runtime", default="./config/runtime.yaml", help="runtime config path")
    parser.add_argument("--report", default=None, help="execution quality report file")
    parser.add_argument("--dry-run", action="store_true", help="print alert message only")
    args = parser.parse_args()

    try:
        runtime = load_yaml(args.runtime)
    except Exception as e:
        print(f"[exec-quality-alert] runtime load failed: {e}")
        return EXIT_OK

    if not runtime.get("enabled", True):
        print("[exec-quality-alert] system disabled")
        return EXIT_OK

    alert_cfg = _merge_alert_cfg(runtime.get("execution_quality_alert", {}) or {})
    if not alert_cfg.get("enabled", False):
        print("[exec-quality-alert] disabled by runtime config")
        return EXIT_OK

    report_file = str(args.report or alert_cfg.get("report_file") or "").strip()
    if not report_file:
        print("[exec-quality-alert] report file not configured")
        return EXIT_OK

    if not os.path.exists(report_file):
        print(f"[exec-quality-alert] report not found: {report_file}")
        return EXIT_OK

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as e:
        print(f"[exec-quality-alert] report load failed: {e}")
        return EXIT_OK

    try:
        result = evaluate_execution_quality_alert(report, alert_cfg)
        if not result.get("min_orders_reached", False):
            print(
                f"[exec-quality-alert] skipped: orders {result.get('orders_total', 0)}"
                f" < min_orders {result.get('min_orders', 1)}"
            )
            return EXIT_OK

        if not result.get("should_alert", False):
            print("[exec-quality-alert] no threshold breach")
            return EXIT_OK

        content = build_alert_message(result, report_file)
        dedup_hours = max(0, _safe_int(alert_cfg.get("dedup_hours"), 12))
        dedup_key = _build_dedup_key(result)

        if args.dry_run:
            print("[exec-quality-alert] dry-run message:")
            print(content)
            return EXIT_OK

        ok, msg = safe_send_wecom_message(
            content=content,
            title="成交质量告警",
            dedup_key=dedup_key,
            dedup_hours=dedup_hours,
        )
        print(f"[exec-quality-alert] wecom {'ok' if ok else 'fail'}: {msg}")
    except Exception as e:
        print(f"[exec-quality-alert] run failed: {e}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
