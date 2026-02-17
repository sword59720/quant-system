#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.signal import (
    momentum_score,
    volatility_score,
    max_drawdown_score,
    liquidity_score,
    normalize_rank,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    runtime = load_yaml("config/runtime.yaml")
    stock = load_yaml("config/stock.yaml")
    risk = load_yaml("config/risk.yaml")

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return

    if not stock.get("enabled", False):
        print("[stock] disabled")
        return

    env = runtime.get("env", "paper")
    total_capital = runtime.get("total_capital", 20000)
    alloc_pct = stock.get("capital_alloc_pct", 0.7)

    lbs = stock.get("signal", {}).get("momentum_lookback_days", [20, 60])
    ws = stock.get("signal", {}).get("momentum_weights", [0.6, 0.4])
    top_n = stock.get("signal", {}).get("top_n", 2)

    factor_w = stock.get("signal", {}).get(
        "factor_weights",
        {"momentum": 0.45, "low_vol": 0.25, "drawdown": 0.20, "liquidity": 0.10},
    )

    universe = stock.get("universe", [])
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")

    raw = {}
    benchmark_close = None

    for s in universe:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp)
            if "close" not in df.columns:
                continue
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(close) < 130:
                continue

            m = momentum_score(close, lbs[0], lbs[1], ws[0], ws[1])
            v = volatility_score(close, lb=20)
            d = max_drawdown_score(close, lb=60)
            amt = liquidity_score(df.get("amount", pd.Series(dtype=float)), lb=20)
            if None in [m, v, d] or amt is None:
                continue

            raw[s] = {"momentum": m, "vol": v, "drawdown": d, "liquidity": amt}

            if s == "510300":
                benchmark_close = close
        except Exception:
            continue

    # rank normalize factors
    mom_rank = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
    vol_rank = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)  # lower vol better
    dd_rank = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)  # less negative better
    liq_rank = normalize_rank({k: v["liquidity"] for k, v in raw.items()}, ascending=False)

    scored = []
    for s in raw.keys():
        score = (
            factor_w.get("momentum", 0.45) * mom_rank.get(s, 0)
            + factor_w.get("low_vol", 0.25) * vol_rank.get(s, 0)
            + factor_w.get("drawdown", 0.20) * dd_rank.get(s, 0)
            + factor_w.get("liquidity", 0.10) * liq_rank.get(s, 0)
        )
        scored.append((s, float(score)))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    picks = [x[0] for x in scored[:top_n]]

    # risk switch: benchmark below MA120 => half risk
    risk_switch = {"enabled": False, "reason": "normal", "alloc_multiplier": 1.0}
    if benchmark_close is not None and len(benchmark_close) >= 120:
        ma120 = benchmark_close.tail(120).mean()
        if benchmark_close.iloc[-1] < ma120:
            risk_switch = {"enabled": True, "reason": "benchmark_below_ma120", "alloc_multiplier": 0.5}

    effective_alloc = alloc_pct * risk_switch["alloc_multiplier"]
    stock_capital = total_capital * effective_alloc

    target_weight_each = min(
        risk["position_limits"]["stock_single_max_pct"],
        effective_alloc / max(1, len(picks) if picks else top_n),
    )

    targets = [{"symbol": s, "target_weight": round(target_weight_each, 4)} for s in picks]

    output_dir = runtime["paths"]["output_dir"]
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "orders"))

    out = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "env": env,
        "capital": stock_capital,
        "alloc_pct_effective": round(effective_alloc, 4),
        "risk_switch": risk_switch,
        "factor_weights": factor_w,
        "scores": [{"symbol": s, "score": sc} for s, sc in scored],
        "targets": targets,
        "note": "v3 multifactor targets",
    }

    out_file = os.path.join(output_dir, "orders", "stock_targets.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[stock] done -> {out_file}")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
