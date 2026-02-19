#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import yaml
from datetime import datetime


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


def _build_crypto_contract_orders(symbol: str, cur_weight: float, tgt_weight: float, capital: float):
    orders = []
    cw = float(cur_weight)
    tw = float(tgt_weight)

    # 同号净仓：只需要增减同方向仓位
    if cw >= 0 and tw >= 0:
        diff = tw - cw
        if diff > 0:
            _append_order(orders, symbol, "OPEN_LONG", diff, capital)
        else:
            _append_order(orders, symbol, "CLOSE_LONG", diff, capital)
        return orders
    if cw <= 0 and tw <= 0:
        curr_abs = abs(cw)
        tgt_abs = abs(tw)
        if tgt_abs > curr_abs:
            _append_order(orders, symbol, "OPEN_SHORT", -(tgt_abs - curr_abs), capital)
        else:
            _append_order(orders, symbol, "CLOSE_SHORT", curr_abs - tgt_abs, capital)
        return orders

    # 跨零换向：先平后开
    if cw > 0 and tw < 0:
        _append_order(orders, symbol, "CLOSE_LONG", -cw, capital)
        _append_order(orders, symbol, "OPEN_SHORT", tw, capital)
        return orders
    if cw < 0 and tw > 0:
        _append_order(orders, symbol, "CLOSE_SHORT", abs(cw), capital)
        _append_order(orders, symbol, "OPEN_LONG", tw, capital)
        return orders
    return orders


def build_market_orders(market: str, target_file: str, pos_file: str, out_file: str, total_capital: float):
    target = load_json(target_file, default={})
    positions = load_json(pos_file, default={"positions": []})

    market_capital = float(target.get("capital", 0))
    capital = float(total_capital)
    contract_mode = bool(target.get("contract_mode", False))
    tgt_items = target.get("targets", [])
    cur_items = positions.get("positions", [])

    tgt = {x["symbol"]: float(x.get("target_weight", 0)) for x in tgt_items}
    cur = {x["symbol"]: float(x.get("weight", 0)) for x in cur_items}

    symbols = sorted(set(tgt.keys()) | set(cur.keys()))
    orders = []
    for s in symbols:
        tw = tgt.get(s, 0.0)
        cw = cur.get(s, 0.0)
        if market == "crypto" and contract_mode:
            orders.extend(_build_crypto_contract_orders(s, cw, tw, capital))
            continue

        diff = tw - cw
        if abs(diff) < 1e-6:
            continue
        action = "BUY" if diff > 0 else "SELL"
        _append_order(orders, s, action, diff, capital)

    out = {
        "ts": datetime.now().isoformat(),
        "market": market,
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

    # 建立默认持仓文件（若不存在）
    stock_pos = os.path.join(state_dir, "stock_positions.json")
    crypto_pos = os.path.join(state_dir, "crypto_positions.json")
    if not os.path.exists(stock_pos):
        with open(stock_pos, "w", encoding="utf-8") as f:
            json.dump({"positions": []}, f, ensure_ascii=False, indent=2)
    if not os.path.exists(crypto_pos):
        with open(crypto_pos, "w", encoding="utf-8") as f:
            json.dump({"positions": []}, f, ensure_ascii=False, indent=2)

    stock_out = build_market_orders(
        market="stock",
        target_file=os.path.join(out_dir, "stock_targets.json"),
        pos_file=stock_pos,
        out_file=os.path.join(out_dir, "stock_trades.json"),
        total_capital=total_capital,
    )
    crypto_out = build_market_orders(
        market="crypto",
        target_file=os.path.join(out_dir, "crypto_targets.json"),
        pos_file=crypto_pos,
        out_file=os.path.join(out_dir, "crypto_trades.json"),
        total_capital=total_capital,
    )

    print("[trades] generated:")
    print(json.dumps(stock_out, ensure_ascii=False, indent=2))
    print(json.dumps(crypto_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
