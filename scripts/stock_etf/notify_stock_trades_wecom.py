#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推送股票交易指令到企业微信。"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_DISABLED, EXIT_OK, EXIT_OUTPUT_ERROR

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def now_in_timezone(tz_name: str) -> datetime:
    tz = str(tz_name or "Asia/Shanghai").strip() or "Asia/Shanghai"
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(tz))
        except Exception:
            pass
    if tz == "Asia/Shanghai":
        return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8)))
    return datetime.now()


def _fmt_weight_delta(delta_weight: Optional[float]) -> str:
    if delta_weight is None:
        return "N/A"
    return f"{delta_weight * 100.0:+.2f}%"


def _safe_iso_date(iso_ts: str, fallback: str) -> str:
    s = str(iso_ts or "").strip()
    if not s:
        return fallback
    if "T" in s:
        return s.split("T", 1)[0]
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return fallback


def build_dedup_key(trades: Dict[str, Any], tz_name: str) -> str:
    now_date = now_in_timezone(tz_name).strftime("%Y-%m-%d")
    ts = str(trades.get("ts", ""))
    date_tag = _safe_iso_date(ts, now_date)
    orders_payload = [
        {
            "symbol": str(o.get("symbol", "")).strip(),
            "action": str(o.get("action", "")).strip().upper(),
            "amount_quote": round(safe_float(o.get("amount_quote")), 2),
            "delta_weight": round(safe_float(o.get("delta_weight")), 6),
        }
        for o in (trades.get("orders", []) or [])
    ]
    payload = json.dumps(orders_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"stock_trade_orders:{date_tag}:{digest}"


def _position_hint(position_file: str, stale_hours: int) -> Tuple[List[str], bool]:
    lines: List[str] = []
    stale = False
    if not os.path.exists(position_file):
        return [f"持仓文件缺失: {position_file}"], True

    try:
        pos = load_json(position_file)
        items = pos.get("positions", []) if isinstance(pos, dict) else []
        lines.append(f"本地持仓记录数: {len(items)}")
    except Exception as e:
        return [f"持仓文件读取失败: {position_file} ({e})"], True

    try:
        mtime = datetime.fromtimestamp(os.path.getmtime(position_file))
        age_hours = (datetime.now() - mtime).total_seconds() / 3600.0
        lines.append(f"持仓文件更新时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        if age_hours > float(stale_hours):
            lines.append(f"⚠️ 持仓文件已超过 {stale_hours} 小时未更新，请先人工核对券商持仓")
            stale = True
    except Exception:
        pass

    return lines, stale


def _fmt_positions(title: str, positions: List[Dict[str, Any]]) -> List[str]:
    lines = [title]
    if not positions:
        lines.append("  (空仓)")
        return lines
    for p in positions:
        sym = str(p.get("symbol", "")).strip()
        w = p.get("weight")
        if w is None:
            lines.append(f"  - {sym}")
        else:
            lines.append(f"  - {sym} weight={safe_float(w):.4f}")
    return lines


def _load_latest_execution_record() -> Dict[str, Any]:
    files = glob.glob("./outputs/orders/execution_record_*.json")
    if not files:
        return {}
    latest = max(files, key=os.path.getmtime)
    try:
        return load_json(latest)
    except Exception:
        return {}


def build_trade_message(trades: Dict[str, Any], stale_position_hours: int = 36) -> str:
    ts = str(trades.get("ts", "")).strip() or datetime.now().isoformat()
    capital_total = safe_float(trades.get("capital_total"))
    capital_market = safe_float(trades.get("capital_market"))
    orders = trades.get("orders", []) or []
    position_file = str(trades.get("position_file", "./outputs/state/stock_positions.json"))

    lines = [
        f"时间: {ts}",
        f"总资金: ¥{capital_total:,.2f}",
        f"股票资金: ¥{capital_market:,.2f}",
        f"指令数: {len(orders)}",
    ]

    # 优先使用最新 execution_record（含交易前后仓位、数量、价格）
    rec = _load_latest_execution_record()
    if rec:
        lines.append("")
        lines.extend(_fmt_positions("交易前仓位:", rec.get("positions_before", []) or []))
        lines.append("交易指令:")
        rows = rec.get("order_results", []) or []
        if not rows:
            lines.append("  (无)")
        else:
            for i, r in enumerate(rows, 1):
                lines.append(
                    f"{i}. {str(r.get('action','')).upper():<4} {str(r.get('symbol','')).strip()} "
                    f"数量={r.get('quantity','N/A')} 价格={r.get('order_price','N/A')} 金额¥{safe_float(r.get('amount_quote')):,.2f}"
                )
        lines.extend(_fmt_positions("交易后仓位:", rec.get("positions_after", []) or []))
    else:
        if orders:
            lines.append("交易指令:")
            for idx, order in enumerate(orders, 1):
                symbol = str(order.get("symbol", "")).strip()
                action = str(order.get("action", "BUY")).strip().upper()
                amount = safe_float(order.get("amount_quote"))
                delta_weight_val = order.get("delta_weight")
                delta_weight = None if delta_weight_val is None else safe_float(delta_weight_val, 0.0)
                lines.append(
                    f"{idx}. {action:<4} {symbol} 金额¥{amount:,.2f}  Δ仓位{_fmt_weight_delta(delta_weight)}"
                )
        else:
            lines.append("今日无调仓指令（维持当前仓位）。")

    pos_lines, _ = _position_hint(position_file, stale_hours=stale_position_hours)
    lines.append("")
    lines.append("持仓校验:")
    lines.extend(pos_lines)
    lines.append("说明: 指令基于本地持仓差分生成，请以券商端最终可成交结果为准。")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description="发送股票交易指令到企业微信")
    parser.add_argument("--runtime", default="./config/runtime.yaml", help="runtime 配置路径")
    parser.add_argument("--file", default="./outputs/orders/stock_trades.json", help="交易指令文件路径")
    parser.add_argument("--title", default="股票交易指令", help="通知标题")
    parser.add_argument("--dedup-hours", type=int, default=None, help="去重小时（默认取 runtime.stock_trade_notify.dedup_hours）")
    parser.add_argument("--send-empty", action="store_true", help="无指令时也发送通知（默认取 runtime.stock_trade_notify.send_empty）")
    parser.add_argument("--dry-run", action="store_true", help="仅打印消息，不发送")
    args = parser.parse_args()

    try:
        runtime = load_yaml(args.runtime)
    except Exception as e:
        print(f"[stock-trade-notify] runtime load failed: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[stock-trade-notify] system disabled")
        return EXIT_DISABLED

    notify_cfg = runtime.get("stock_trade_notify", {}) if isinstance(runtime.get("stock_trade_notify", {}), dict) else {}
    enabled = bool(notify_cfg.get("enabled", True))
    if not enabled:
        print("[stock-trade-notify] disabled by runtime config")
        return EXIT_OK

    trade_file = str(notify_cfg.get("trades_file", args.file)).strip() or args.file
    if not os.path.exists(trade_file):
        print(f"[stock-trade-notify] trade file not found: {trade_file}")
        return EXIT_OUTPUT_ERROR

    try:
        trades = load_json(trade_file)
    except Exception as e:
        print(f"[stock-trade-notify] trade file load failed: {e}")
        return EXIT_OUTPUT_ERROR

    send_empty = bool(notify_cfg.get("send_empty", False))
    if args.send_empty:
        send_empty = True

    orders = trades.get("orders", []) or []
    if not orders and not send_empty:
        print("[stock-trade-notify] no orders, skipped (send_empty=false)")
        return EXIT_OK

    title = str(notify_cfg.get("title", args.title)).strip() or "股票交易指令"
    dedup_hours_default = int(notify_cfg.get("dedup_hours", 12))
    dedup_hours = dedup_hours_default if args.dedup_hours is None else int(args.dedup_hours)
    stale_position_hours = int(notify_cfg.get("stale_position_hours", 36))
    tz_name = str(runtime.get("timezone", "Asia/Shanghai")).strip() or "Asia/Shanghai"

    dedup_key = build_dedup_key(trades, tz_name=tz_name)
    msg = build_trade_message(trades, stale_position_hours=stale_position_hours)
    print(msg)
    if args.dry_run:
        print("[stock-trade-notify] dry-run only")
        return EXIT_OK

    ok, detail = safe_send_wecom_message(
        content=msg,
        title=title,
        dedup_key=dedup_key,
        dedup_hours=dedup_hours,
    )
    print(f"[stock-trade-notify] wecom {'ok' if ok else 'fail'}: {detail}")
    return EXIT_OK if ok else EXIT_OUTPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
