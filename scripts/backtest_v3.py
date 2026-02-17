#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import math
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.signal import momentum_score, volatility_score, max_drawdown_score, liquidity_score, normalize_rank


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def annualized_return(nav: pd.Series, periods_per_year: int):
    if len(nav) < 2:
        return 0.0
    total = nav.iloc[-1] / nav.iloc[0]
    years = max((len(nav) - 1) / periods_per_year, 1e-9)
    return float(total ** (1 / years) - 1)


def max_drawdown(nav: pd.Series):
    peak = nav.cummax()
    dd = nav / peak - 1
    return float(dd.min())


def sharpe_ratio(ret: pd.Series, periods_per_year: int):
    if len(ret) < 2 or ret.std() == 0:
        return 0.0
    return float((ret.mean() / ret.std()) * math.sqrt(periods_per_year))


def summarize(nav, ret, ppy):
    nav_s = pd.Series(nav)
    ret_s = pd.Series(ret)
    return {
        "periods": int(len(ret_s)),
        "annual_return": annualized_return(nav_s, ppy),
        "max_drawdown": max_drawdown(nav_s),
        "sharpe": sharpe_ratio(ret_s, ppy),
        "final_nav": float(nav_s.iloc[-1]),
    }


def backtest_stock(runtime, stock_cfg):
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    universe = stock_cfg.get("universe", [])
    alloc_pct = float(stock_cfg.get("capital_alloc_pct", 0.7))
    top_n = stock_cfg.get("signal", {}).get("top_n", 2)
    lbs = stock_cfg.get("signal", {}).get("momentum_lookback_days", [20, 60])
    ws = stock_cfg.get("signal", {}).get("momentum_weights", [0.6, 0.4])
    fw = stock_cfg.get("signal", {}).get("factor_weights", {"momentum": 0.45, "low_vol": 0.25, "drawdown": 0.2, "liquidity": 0.1})

    frames = {}
    for s in universe:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        frames[s] = df.sort_values("date").set_index("date")

    if len(frames) < top_n:
        return {"error": "stock data not enough"}

    dates = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    if len(dates) < 350:
        return {"error": "stock history too short (<350 days common)"}

    fee = 0.0008
    rebalance_threshold = 0.10
    nav, rets = [1.0], []
    weights = {s: 0.0 for s in frames.keys()}

    for i in range(121, len(dates) - 1):
        dt, nxt = dates[i], dates[i + 1]

        # weekly rebalance only Monday
        do_rebalance = pd.Timestamp(dt).weekday() == 0
        target = weights.copy()

        if do_rebalance:
            raw = {}
            for s, df in frames.items():
                hist = df.loc[:dt]
                close = pd.to_numeric(hist["close"], errors="coerce").dropna()
                amount = pd.to_numeric(hist.get("amount", pd.Series(dtype=float)), errors="coerce")
                m = momentum_score(close, lbs[0], lbs[1], ws[0], ws[1])
                v = volatility_score(close, 20)
                d = max_drawdown_score(close, 60)
                l = liquidity_score(amount, 20)
                if None in [m, v, d] or l is None:
                    continue
                raw[s] = {"momentum": m, "vol": v, "drawdown": d, "liquidity": l}

            risk_mult = 1.0
            if "510300" in frames:
                close300 = pd.to_numeric(frames["510300"].loc[:dt, "close"], errors="coerce").dropna()
                if len(close300) >= 120 and close300.iloc[-1] < close300.tail(120).mean():
                    risk_mult = 0.5

            if len(raw) >= top_n:
                mr = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
                vr = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)
                dr = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)
                lr = normalize_rank({k: v["liquidity"] for k, v in raw.items()}, ascending=False)
                score = {
                    s: fw.get("momentum", 0.45) * mr.get(s, 0)
                    + fw.get("low_vol", 0.25) * vr.get(s, 0)
                    + fw.get("drawdown", 0.2) * dr.get(s, 0)
                    + fw.get("liquidity", 0.1) * lr.get(s, 0)
                    for s in raw.keys()
                }
                picks = [x[0] for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]]
                tw = alloc_pct * risk_mult / top_n
                target = {s: (tw if s in picks else 0.0) for s in frames.keys()}
            else:
                target = {s: 0.0 for s in frames.keys()}

            # trade threshold
            turnover = sum(abs(target[s] - weights.get(s, 0.0)) for s in target.keys())
            if turnover < rebalance_threshold:
                target = weights.copy()

        # pnl on next day
        day_ret = 0.0
        for s, w in target.items():
            if w == 0:
                continue
            c0 = frames[s].loc[dt, "close"]
            c1 = frames[s].loc[nxt, "close"]
            day_ret += w * (c1 / c0 - 1)

        turnover = sum(abs(target[s] - weights.get(s, 0.0)) for s in target.keys()) if do_rebalance else 0.0
        day_ret -= turnover * fee

        weights = target
        nav.append(nav[-1] * (1 + day_ret))
        rets.append(day_ret)

    return summarize(nav, rets, 252)


def backtest_crypto(runtime, crypto_cfg):
    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    symbols = crypto_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])
    alloc_pct = float(crypto_cfg.get("capital_alloc_pct", 0.3))
    top_n = crypto_cfg.get("signal", {}).get("top_n", 1)
    lbs = crypto_cfg.get("signal", {}).get("momentum_lookback_bars", [30, 90])
    ws = crypto_cfg.get("signal", {}).get("momentum_weights", [0.6, 0.4])
    fw = crypto_cfg.get("signal", {}).get("factor_weights", {"momentum": 0.60, "low_vol": 0.25, "drawdown": 0.15})
    threshold = float(crypto_cfg.get("defense", {}).get("risk_off_threshold_pct", -3.0)) / 100.0

    frames = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        frames[s] = df.sort_values("date").set_index("date")

    if len(frames) < top_n:
        return {"error": "crypto data not enough"}

    dates = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    if len(dates) < max(lbs) + 120:
        return {"error": "crypto history too short"}

    fee = 0.001
    rebalance_threshold = 0.08
    nav, rets = [1.0], []
    weights = {s: 0.0 for s in frames.keys()}

    for i in range(max(lbs) + 1, len(dates) - 1):
        dt, nxt = dates[i], dates[i + 1]

        raw = {}
        for s, df in frames.items():
            hist = df.loc[:dt]
            close = pd.to_numeric(hist["close"], errors="coerce").dropna()
            m = momentum_score(close, lbs[0], lbs[1], ws[0], ws[1])
            v = volatility_score(close, 20)
            d = max_drawdown_score(close, 60)
            sret = close.iloc[-1] / close.iloc[-lbs[0]] - 1 if len(close) > lbs[0] else -1
            if None in [m, v, d]:
                continue
            raw[s] = {"momentum": m, "vol": v, "drawdown": d, "short_ret": float(sret)}

        target = {s: 0.0 for s in frames.keys()}
        if len(raw) >= top_n:
            # risk off: all short_ret below threshold => all cash
            if not all(v["short_ret"] < threshold for v in raw.values()):
                mr = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
                vr = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)
                dr = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)
                score = {
                    s: fw.get("momentum", 0.60) * mr.get(s, 0)
                    + fw.get("low_vol", 0.25) * vr.get(s, 0)
                    + fw.get("drawdown", 0.15) * dr.get(s, 0)
                    for s in raw.keys()
                }
                picks = [x[0] for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]]
                tw = alloc_pct / top_n
                target = {s: (tw if s in picks else 0.0) for s in frames.keys()}

        turnover = sum(abs(target[s] - weights.get(s, 0.0)) for s in target.keys())
        if turnover < rebalance_threshold:
            target = weights.copy()
            turnover = 0.0

        bar_ret = 0.0
        for s, w in target.items():
            if w == 0:
                continue
            c0 = frames[s].loc[dt, "close"]
            c1 = frames[s].loc[nxt, "close"]
            bar_ret += w * (c1 / c0 - 1)
        bar_ret -= turnover * fee

        weights = target
        nav.append(nav[-1] * (1 + bar_ret))
        rets.append(bar_ret)

    return summarize(nav, rets, 6 * 365)


def main():
    runtime = load_yaml("config/runtime.yaml")
    stock_cfg = load_yaml("config/stock.yaml")
    crypto_cfg = load_yaml("config/crypto.yaml")

    stock = backtest_stock(runtime, stock_cfg)
    crypto = backtest_crypto(runtime, crypto_cfg)

    report = {
        "ts": datetime.now().isoformat(),
        "note": "v3.2 lightweight backtest (alloc-aware, threshold rebalance, risk-off)",
        "stock": stock,
        "crypto": crypto,
    }

    out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(out_dir)
    out = os.path.join(out_dir, "backtest_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[backtest] report -> {out}")


if __name__ == "__main__":
    main()
