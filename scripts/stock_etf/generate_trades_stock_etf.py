#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def _append_order(orders, symbol: str, action: str, delta_weight: float, capital: float):
    if abs(delta_weight) < 1e-6:
        return
    orders.append(
        {
            "symbol": symbol,
            "action": action,
            "delta_weight": round(float(delta_weight), 6),
            "amount_quote": round(abs(float(delta_weight)) * capital, 2),
        }
    )


def build_stock_orders(target_file: str, pos_file: str, out_file: str, total_capital: float):
    target = load_json(target_file, default={})
    positions = load_json(pos_file, default={"positions": []})

    market_capital = float(target.get("capital", 0))
    capital = float(total_capital)
    tgt_items = target.get("targets", [])
    cur_items = positions.get("positions", [])

    tgt = {x["symbol"]: float(x.get("target_weight", 0)) for x in tgt_items}
    cur = {x["symbol"]: float(x.get("weight", 0)) for x in cur_items}

    symbols = sorted(set(tgt.keys()) | set(cur.keys()))
    orders = []
    for symbol in symbols:
        tw = tgt.get(symbol, 0.0)
        cw = cur.get(symbol, 0.0)
        diff = tw - cw
        if abs(diff) < 1e-6:
            continue
        action = "BUY" if diff > 0 else "SELL"
        _append_order(orders, symbol, action, diff, capital)

    out = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "capital_total": capital,
        "capital_market": market_capital,
        "target_file": target_file,
        "position_file": pos_file,
        "orders": orders,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def main():
    runtime = load_yaml("config/runtime.yaml")
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
    )

    print("[stock-trades] generated:")
    print(json.dumps(stock_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
