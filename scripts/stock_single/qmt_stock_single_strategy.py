#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMT execution bridge for stock_single strategy.

Workflow:
1) (optional) refresh stock_single signals by running run_stock_single.py
2) read outputs/orders/stock_single_signals.json
3) convert signal target weights to order list against current positions
4) plan mode: write order plan only
5) live mode: send orders to QMT (xtquant)
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yaml


DEFAULT_RUNTIME_FILE = "./config/runtime.yaml"
DEFAULT_STOCK_SINGLE_FILE = "./config/stock_single.yaml"
DEFAULT_SIGNAL_FILE = "./outputs/orders/stock_single_signals.json"
DEFAULT_POSITION_SNAPSHOT = "./outputs/state/stock_positions.json"
DEFAULT_OUTPUT_DIR = "./outputs/orders"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="QMT bridge for stock_single strategy signals.")
    p.add_argument("--runtime", default=DEFAULT_RUNTIME_FILE, help="runtime yaml path")
    p.add_argument("--stock-single", default=DEFAULT_STOCK_SINGLE_FILE, help="stock_single yaml path")
    p.add_argument("--signals-file", default=None, help="signal file path, default from stock_single.paths.signal_file")
    p.add_argument("--position-snapshot", default=DEFAULT_POSITION_SNAPSHOT, help="local positions snapshot for plan mode")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="output dir for order plan/execution records")

    p.add_argument("--mode", choices=["plan", "live"], default="plan", help="plan=generate orders only, live=send to QMT")
    p.add_argument("--refresh-signals", action="store_true", help="run stock_single pipeline before reading signals")
    p.add_argument(
        "--signal-task",
        choices=["full", "pool", "risk", "hourly"],
        default="full",
        help="task for run_stock_single.py when --refresh-signals is enabled",
    )

    p.add_argument("--qmt-path", default="", help="QMT userdata path for XtQuantTrader (required in live mode)")
    p.add_argument("--account-id", default="", help="QMT stock account id (required in live mode)")
    p.add_argument("--account-type", default="STOCK", help="QMT account type, usually STOCK")
    p.add_argument("--strategy-name", default="stock_single_qmt", help="strategy name for QMT order remarks")

    p.add_argument("--order-price-mode", choices=["signal", "last"], default="signal", help="limit price source")
    p.add_argument("--min-order-value", type=float, default=1000.0, help="drop orders with value below this amount")
    p.add_argument("--min-delta-weight", type=float, default=0.0025, help="ignore tiny target deltas")
    p.add_argument("--max-orders", type=int, default=40, help="max number of orders per run")
    p.add_argument("--dry-run-live", action="store_true", help="live mode but do not send orders")
    return p.parse_args()


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def _clip_lot_shares(shares: int, lot_size: int = 100) -> int:
    if shares <= 0:
        return 0
    return (int(shares) // lot_size) * lot_size


def _call_first(obj: Any, names: List[str], *args, **kwargs):
    last_error = None
    for name in names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            last_error = e
            continue
    if last_error is not None:
        raise last_error
    raise AttributeError(f"none of methods found: {names}")


def _attr_or_key(x: Any, name: str, default=None):
    if isinstance(x, dict):
        return x.get(name, default)
    return getattr(x, name, default)


def _normalize_symbol(sym: str) -> str:
    s = str(sym or "").strip().upper()
    if not s:
        return s
    if "." in s:
        return s
    if len(s) == 6 and s[0] in {"6", "5", "9"}:
        return f"{s}.SH"
    if len(s) == 6:
        return f"{s}.SZ"
    return s


def _refresh_signals(task: str) -> None:
    cmd = [sys.executable, "./scripts/stock_single/run_stock_single.py", "--task", task]
    subprocess.run(cmd, check=True)


def _load_signal_payload(signal_file: str) -> dict:
    if not os.path.exists(signal_file):
        raise FileNotFoundError(f"signal file not found: {signal_file}")
    with open(signal_file, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid signal payload format: {signal_file}")
    payload.setdefault("signals", [])
    return payload


def _signal_map(payload: dict) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for x in payload.get("signals", []) or []:
        sym = _normalize_symbol(str(x.get("symbol", "")))
        if not sym:
            continue
        out[sym] = x
    return out


def _build_targets_from_signals(
    payload: dict,
    *,
    max_positions: int,
    capital_alloc_pct: float,
) -> Tuple[Dict[str, float], set]:
    target_weights: Dict[str, float] = {}
    explicit_sell = set()
    buy_candidates = []
    for x in payload.get("signals", []) or []:
        sym = _normalize_symbol(str(x.get("symbol", "")))
        if not sym:
            continue
        action = str(x.get("action", "HOLD")).strip().upper()
        if action == "BUY":
            buy_candidates.append(
                {
                    "symbol": sym,
                    "score": _to_float(x.get("score"), -1e12),
                    "target_weight": max(0.0, _to_float(x.get("target_weight"), 0.0)),
                }
            )
        elif action == "SELL":
            explicit_sell.add(sym)
            target_weights[sym] = 0.0
    buy_candidates = sorted(buy_candidates, key=lambda z: z["score"], reverse=True)
    if max_positions > 0:
        buy_candidates = buy_candidates[: int(max_positions)]
    for item in buy_candidates:
        target_weights[item["symbol"]] = float(item["target_weight"])
    total_buy_w = sum(target_weights.values())
    cap = max(0.0, float(capital_alloc_pct))
    if total_buy_w > cap and total_buy_w > 1e-9:
        scale = cap / total_buy_w
        for sym in list(target_weights.keys()):
            target_weights[sym] = float(target_weights[sym]) * scale
    return target_weights, explicit_sell


def _load_snapshot_positions(path: str) -> Dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    rows = payload.get("positions", []) if isinstance(payload, dict) else []
    out: Dict[str, dict] = {}
    for r in rows or []:
        sym = _normalize_symbol(str(_attr_or_key(r, "symbol", "")))
        if not sym:
            continue
        out[sym] = {
            "symbol": sym,
            "weight": _to_float(_attr_or_key(r, "weight", 0.0), 0.0),
            "market_value": _to_float(_attr_or_key(r, "market_value", 0.0), 0.0),
            "volume": _safe_int(_attr_or_key(r, "volume", 0), 0),
            "can_use_volume": _safe_int(_attr_or_key(r, "can_use_volume", _attr_or_key(r, "volume", 0)), 0),
        }
    return out


@dataclass
class QmtSession:
    trader: Any
    account: Any
    xtconstant: Any


def _open_qmt_session(qmt_path: str, account_id: str, account_type: str) -> QmtSession:
    try:
        from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
        from xtquant.xttype import StockAccount
        from xtquant import xtconstant
    except Exception as e:
        raise RuntimeError(f"xtquant import failed: {e}")

    class _Callback(XtQuantTraderCallback):
        pass

    session_id = int(time.time()) % 1000000000
    trader = XtQuantTrader(qmt_path, session_id)
    trader.register_callback(_Callback())
    trader.start()
    connect_res = trader.connect()
    if connect_res != 0:
        raise RuntimeError(f"QMT connect failed, code={connect_res}")

    account = StockAccount(account_id, account_type)
    sub_res = trader.subscribe(account)
    if sub_res != 0:
        raise RuntimeError(f"QMT subscribe failed, code={sub_res}")

    return QmtSession(trader=trader, account=account, xtconstant=xtconstant)


def _query_qmt_positions_and_asset(session: QmtSession) -> Tuple[Dict[str, dict], float]:
    trader = session.trader
    account = session.account

    asset = _call_first(
        trader,
        ["query_stock_asset", "query_asset", "query_account_asset"],
        account,
    )
    total_asset = _to_float(_attr_or_key(asset, "total_asset", 0.0), 0.0)
    if total_asset <= 0:
        cash = _to_float(_attr_or_key(asset, "cash", 0.0), 0.0)
        market_value = _to_float(_attr_or_key(asset, "market_value", 0.0), 0.0)
        total_asset = cash + market_value
    if total_asset <= 0:
        raise RuntimeError("QMT returned invalid total_asset <= 0")

    rows = _call_first(
        trader,
        ["query_stock_positions", "query_positions", "query_position"],
        account,
    )
    out: Dict[str, dict] = {}
    for r in rows or []:
        sym = _normalize_symbol(str(_attr_or_key(r, "stock_code", _attr_or_key(r, "symbol", ""))))
        if not sym:
            continue
        market_value = _to_float(_attr_or_key(r, "market_value", 0.0), 0.0)
        vol = _safe_int(_attr_or_key(r, "volume", _attr_or_key(r, "total_volume", 0)), 0)
        can_use = _safe_int(_attr_or_key(r, "can_use_volume", _attr_or_key(r, "enable_volume", vol)), vol)
        weight = market_value / total_asset if total_asset > 0 else 0.0
        out[sym] = {
            "symbol": sym,
            "weight": weight,
            "market_value": market_value,
            "volume": vol,
            "can_use_volume": can_use,
        }
    return out, total_asset


def _pick_price(signal_item: dict, side: str, mode: str) -> float:
    last_price = _to_float(signal_item.get("last_price"), 0.0)
    if mode == "last":
        return last_price
    if side == "BUY":
        px = _to_float(signal_item.get("entry_price"), 0.0)
        return px if px > 0 else last_price
    px = _to_float(signal_item.get("exit_price"), 0.0)
    return px if px > 0 else last_price


def _build_order_plan(
    *,
    signal_payload: dict,
    signal_map: Dict[str, dict],
    target_weights: Dict[str, float],
    explicit_sell: set,
    positions: Dict[str, dict],
    total_asset: float,
    min_delta_weight: float,
    min_order_value: float,
    max_orders: int,
    order_price_mode: str,
) -> dict:
    current_weights = {sym: _to_float(pos.get("weight"), 0.0) for sym, pos in positions.items()}
    symbols = sorted(set(current_weights.keys()) | set(target_weights.keys()) | set(explicit_sell))
    orders: List[dict] = []
    dropped: List[dict] = []

    for sym in symbols:
        cur_w = _to_float(current_weights.get(sym), 0.0)
        tgt_w = _to_float(target_weights.get(sym), 0.0)
        if sym in explicit_sell:
            tgt_w = 0.0
        delta_w = tgt_w - cur_w
        if abs(delta_w) < float(min_delta_weight):
            continue

        side = "BUY" if delta_w > 0 else "SELL"
        sig = signal_map.get(sym, {})
        price = _pick_price(sig, side, order_price_mode)
        if price <= 0:
            dropped.append({"symbol": sym, "reason": "invalid_price", "delta_weight": delta_w})
            continue

        est_value = abs(delta_w) * total_asset
        if est_value < float(min_order_value):
            dropped.append({"symbol": sym, "reason": "below_min_order_value", "delta_weight": delta_w, "value": est_value})
            continue

        raw_shares = int(est_value / price)
        shares = _clip_lot_shares(raw_shares, lot_size=100)
        if shares < 100:
            dropped.append({"symbol": sym, "reason": "shares_below_1lot", "delta_weight": delta_w, "shares": shares})
            continue

        if side == "SELL":
            can_use = _safe_int(positions.get(sym, {}).get("can_use_volume", 0), 0)
            shares = min(shares, _clip_lot_shares(can_use, lot_size=100))
            if shares < 100:
                dropped.append({"symbol": sym, "reason": "no_sellable_shares", "delta_weight": delta_w, "shares": shares})
                continue

        order_value = shares * price
        orders.append(
            {
                "symbol": sym,
                "side": side,
                "delta_weight": round(delta_w, 6),
                "target_weight": round(tgt_w, 6),
                "current_weight": round(cur_w, 6),
                "price": round(price, 3),
                "shares": int(shares),
                "amount_quote": round(order_value, 2),
                "signal_reason": str(sig.get("reason", "")),
            }
        )

    orders = sorted(orders, key=lambda x: (0 if x["side"] == "SELL" else 1, -abs(x["delta_weight"])))
    if len(orders) > int(max_orders):
        dropped.extend([{**x, "reason": "cut_by_max_orders"} for x in orders[int(max_orders) :]])
        orders = orders[: int(max_orders)]

    return {
        "generated_at": datetime.now().isoformat(),
        "market": "CN_STOCK_SINGLE",
        "mode": "qmt_bridge",
        "total_asset": float(total_asset),
        "signal_ts": signal_payload.get("ts"),
        "signal_policy": signal_payload.get("signal_policy", {}),
        "risk_overlay": signal_payload.get("risk_overlay", {}),
        "orders": orders,
        "dropped": dropped,
    }


def _place_orders_qmt(plan: dict, session: QmtSession, strategy_name: str, dry_run_live: bool) -> dict:
    trader = session.trader
    account = session.account
    xtconstant = session.xtconstant

    buy_const = getattr(xtconstant, "STOCK_BUY", 23)
    sell_const = getattr(xtconstant, "STOCK_SELL", 24)
    fix_price_const = getattr(xtconstant, "FIX_PRICE", 11)

    results = []
    for row in plan.get("orders", []):
        side_const = buy_const if row.get("side") == "BUY" else sell_const
        stock_code = _normalize_symbol(row.get("symbol", ""))
        volume = int(row.get("shares", 0))
        price = float(row.get("price", 0.0))
        status = "dry_run"
        order_id = None
        err = None
        t0 = time.time()
        try:
            if not dry_run_live:
                # Common xtquant signature:
                # order_stock(account, stock_code, order_type, order_volume, price_type, price, strategy_name, order_remark)
                order_id = _call_first(
                    trader,
                    ["order_stock"],
                    account,
                    stock_code,
                    side_const,
                    volume,
                    fix_price_const,
                    price,
                    strategy_name,
                    f"stock_single:{row.get('signal_reason', '')[:40]}",
                )
                status = "submitted"
        except Exception as e:
            status = "error"
            err = str(e)
        latency_ms = round((time.time() - t0) * 1000.0, 2)
        results.append(
            {
                **row,
                "stock_code": stock_code,
                "status": status,
                "order_id": order_id,
                "error": err,
                "latency_ms": latency_ms,
            }
        )

    return {
        "ts": datetime.now().isoformat(),
        "strategy_name": strategy_name,
        "dry_run_live": bool(dry_run_live),
        "summary": {
            "orders_total": len(results),
            "submitted": sum(1 for x in results if x["status"] == "submitted"),
            "errors": sum(1 for x in results if x["status"] == "error"),
            "dry_run": sum(1 for x in results if x["status"] == "dry_run"),
        },
        "results": results,
    }


def _resolve_signal_file(args, stock_single_cfg: dict) -> str:
    if args.signals_file:
        return str(args.signals_file)
    return str(stock_single_cfg.get("paths", {}).get("signal_file", DEFAULT_SIGNAL_FILE))


def _write_json(path: str, payload: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    runtime = load_yaml(args.runtime)
    stock_single_cfg = load_yaml(args.stock_single)

    if not runtime.get("enabled", True):
        print("[qmt-stock-single] runtime disabled")
        return 2
    if not stock_single_cfg.get("enabled", False):
        print("[qmt-stock-single] warning: stock_single.enabled=false, continue by request")

    if args.refresh_signals:
        _refresh_signals(args.signal_task)

    signal_file = _resolve_signal_file(args, stock_single_cfg)
    payload = _load_signal_payload(signal_file)
    signal_map = _signal_map(payload)
    target_weights, explicit_sell = _build_targets_from_signals(
        payload,
        max_positions=int(stock_single_cfg.get("max_positions", 10)),
        capital_alloc_pct=float(stock_single_cfg.get("capital_alloc_pct", 1.0)),
    )

    if args.mode == "plan":
        total_capital = _to_float(runtime.get("total_capital"), 0.0)
        positions = _load_snapshot_positions(args.position_snapshot)
        if total_capital <= 0:
            total_capital = _to_float(sum(_to_float(x.get("market_value"), 0.0) for x in positions.values()), 0.0)
        if total_capital <= 0:
            total_capital = 100000.0
        plan = _build_order_plan(
            signal_payload=payload,
            signal_map=signal_map,
            target_weights=target_weights,
            explicit_sell=explicit_sell,
            positions=positions,
            total_asset=total_capital,
            min_delta_weight=args.min_delta_weight,
            min_order_value=args.min_order_value,
            max_orders=args.max_orders,
            order_price_mode=args.order_price_mode,
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(args.output_dir, f"stock_single_qmt_order_plan_{ts}.json")
        _write_json(out_file, plan)
        print(f"[qmt-stock-single] mode=plan orders={len(plan.get('orders', []))} -> {out_file}")
        return 0

    if not args.qmt_path or not args.account_id:
        raise ValueError("live mode requires --qmt-path and --account-id")

    session = _open_qmt_session(args.qmt_path, args.account_id, args.account_type)
    positions, total_asset = _query_qmt_positions_and_asset(session)
    plan = _build_order_plan(
        signal_payload=payload,
        signal_map=signal_map,
        target_weights=target_weights,
        explicit_sell=explicit_sell,
        positions=positions,
        total_asset=total_asset,
        min_delta_weight=args.min_delta_weight,
        min_order_value=args.min_order_value,
        max_orders=args.max_orders,
        order_price_mode=args.order_price_mode,
    )
    exec_res = _place_orders_qmt(plan, session, args.strategy_name, args.dry_run_live)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = os.path.join(args.output_dir, f"stock_single_qmt_order_plan_{ts}.json")
    exec_file = os.path.join(args.output_dir, f"stock_single_qmt_execution_{ts}.json")
    _write_json(plan_file, plan)
    _write_json(exec_file, exec_res)

    summary = exec_res.get("summary", {})
    print(
        "[qmt-stock-single] mode=live "
        f"orders={summary.get('orders_total', 0)} "
        f"submitted={summary.get('submitted', 0)} "
        f"errors={summary.get('errors', 0)} "
        f"dry_run={summary.get('dry_run', 0)}"
    )
    print(f"[qmt-stock-single] plan -> {plan_file}")
    print(f"[qmt-stock-single] exec -> {exec_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
