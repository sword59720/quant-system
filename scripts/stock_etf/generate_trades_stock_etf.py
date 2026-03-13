#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import os
from datetime import datetime

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _normalize_positions(items):
    out = []
    for x in items or []:
        if not isinstance(x, dict):
            continue
        symbol = str(x.get("symbol", "")).strip()
        if not symbol:
            continue
        weight = float(x.get("weight", 0.0) or 0.0)
        quantity = x.get("quantity")
        row = {"symbol": symbol, "weight": round(weight, 6)}
        if quantity is not None:
            row["quantity"] = quantity
        out.append(row)
    return sorted(out, key=lambda x: x["symbol"])


def _targets_to_positions(items):
    out = []
    for x in items or []:
        if not isinstance(x, dict):
            continue
        symbol = str(x.get("symbol", "")).strip()
        if not symbol:
            continue
        out.append({"symbol": symbol, "weight": round(float(x.get("target_weight", 0.0) or 0.0), 6)})
    return sorted(out, key=lambda x: x["symbol"])


def _positions_to_map(items):
    out = {}
    for x in items or []:
        symbol = str(x.get("symbol", "")).strip()
        if not symbol:
            continue
        out[symbol] = float(x.get("weight", 0.0) or 0.0)
    return out


def _latest_close(data_dir: str, symbol: str):
    path = os.path.join(data_dir, "stock", f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    last = None
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    if not last:
        return None
    try:
        return float(last.get("close") or 0.0)
    except Exception:
        return None


def _append_order(orders, symbol: str, action: str, current_weight: float, target_weight: float, capital: float, data_dir: str):
    delta_weight = target_weight - current_weight
    if abs(delta_weight) < 1e-6:
        return

    amount_quote = round(abs(float(delta_weight)) * capital, 2)
    last_price = _latest_close(data_dir, symbol)
    est_shares = None
    est_lots = None
    est_amount_rounded = None
    if last_price and last_price > 0:
        raw_shares = int(amount_quote / last_price)
        est_lots = raw_shares // 100
        est_shares = est_lots * 100
        est_amount_rounded = round(est_shares * last_price, 2)

    order = {
        "symbol": symbol,
        "action": action,
        "current_weight": round(float(current_weight), 6),
        "target_weight": round(float(target_weight), 6),
        "delta_weight": round(float(delta_weight), 6),
        "amount_quote": amount_quote,
    }
    if last_price and last_price > 0:
        order["ref_price"] = round(last_price, 6)
        order["estimated_shares"] = est_shares
        order["estimated_lots"] = est_lots
        order["estimated_amount_quote_rounded"] = est_amount_rounded
    orders.append(order)


def build_stock_orders(target_file: str, pos_file: str, out_file: str, total_capital: float, data_dir: str):
    target = load_json(target_file, default={})
    positions = load_json(pos_file, default={"positions": []})

    market_capital = float(target.get("capital", 0))
    capital = float(total_capital)
    tgt_items = target.get("targets", [])
    cur_items = positions.get("positions", [])

    current_positions = _normalize_positions(cur_items)
    target_positions = _targets_to_positions(tgt_items)
    cur = _positions_to_map(current_positions)
    tgt = _positions_to_map(target_positions)

    symbols = sorted(set(tgt.keys()) | set(cur.keys()))
    orders = []
    for symbol in symbols:
        tw = tgt.get(symbol, 0.0)
        cw = cur.get(symbol, 0.0)
        diff = tw - cw
        if abs(diff) < 1e-6:
            continue
        action = "BUY" if diff > 0 else "SELL"
        _append_order(orders, symbol, action, cw, tw, capital, data_dir)

    estimated_after_positions = []
    for symbol in symbols:
        w = tgt.get(symbol, 0.0)
        if abs(w) < 1e-6:
            continue
        estimated_after_positions.append({"symbol": symbol, "weight": round(float(w), 6)})

    out = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "capital_total": capital,
        "capital_market": market_capital,
        "target_file": target_file,
        "position_file": pos_file,
        "current_positions": current_positions,
        "target_positions": target_positions,
        "estimated_after_positions": estimated_after_positions,
        "orders": orders,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main():
    runtime = load_yaml("config/runtime.yaml")
    data_dir = runtime["paths"]["data_dir"]
    out_dir = os.path.join(runtime["paths"]["output_dir"], "orders")
    state_dir = os.path.join(runtime["paths"]["output_dir"], "state")
    ensure_dir(out_dir)
    ensure_dir(state_dir)

    total_capital = float(runtime.get("total_capital", 0))
    stock_pos = os.path.join(state_dir, "stock_positions.json")
    if not os.path.exists(stock_pos):
        with open(stock_pos, "w", encoding="utf-8") as f:
            json.dump({"positions": []}, f, ensure_ascii=False, indent=2)

    stock_out = build_stock_orders(
        target_file=os.path.join(out_dir, "stock_targets.json"),
        pos_file=stock_pos,
        out_file=os.path.join(out_dir, "stock_trades.json"),
        total_capital=total_capital,
        data_dir=data_dir,
    )

    print("[stock-trades] generated:")
    print(json.dumps(stock_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
