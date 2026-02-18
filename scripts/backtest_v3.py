#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import yaml
import math
import tempfile
import argparse
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_OUTPUT_ERROR, EXIT_SIGNAL_ERROR
from core.signal import momentum_score, volatility_score, max_drawdown_score, liquidity_score, normalize_rank
from scripts.paper_forward_stock import run_paper_forward


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


def build_stock_compare(stock_result):
    if not isinstance(stock_result, dict):
        return {"available": False, "reason": "stock result is not a dict"}
    if stock_result.get("mode") != "global_momentum_both":
        return {"available": False, "reason": "stock mode is not global_momentum_both"}

    prod = stock_result.get("production", {})
    res = stock_result.get("research", {})
    if (not isinstance(prod, dict)) or (not isinstance(res, dict)):
        return {"available": False, "reason": "missing production/research result"}

    higher_better = [
        "annual_return",
        "excess_annual_return_vs_alloc",
        "sharpe",
        "final_nav",
    ]
    deltas = {}
    prod_better = []
    res_better = []

    for k in higher_better:
        if k in prod and k in res:
            deltas[k] = float(prod[k] - res[k])
            if prod[k] > res[k]:
                prod_better.append(k)
            elif prod[k] < res[k]:
                res_better.append(k)

    if "max_drawdown" in prod and "max_drawdown" in res:
        deltas["max_drawdown"] = float(prod["max_drawdown"] - res["max_drawdown"])
        if prod["max_drawdown"] > res["max_drawdown"]:
            prod_better.append("max_drawdown")
        elif prod["max_drawdown"] < res["max_drawdown"]:
            res_better.append("max_drawdown")

    periods_equal = None
    if "periods" in prod and "periods" in res:
        periods_equal = bool(int(prod["periods"]) == int(res["periods"]))
        deltas["periods"] = int(prod["periods"]) - int(res["periods"])

    preferred_model = "tie"
    if len(prod_better) > len(res_better):
        preferred_model = "production"
    elif len(prod_better) < len(res_better):
        preferred_model = "research"

    return {
        "available": True,
        "periods_equal": periods_equal,
        "production_better_metrics": prod_better,
        "research_better_metrics": res_better,
        "preferred_model": preferred_model,
        "deltas_production_minus_research": deltas,
    }


def backtest_stock_global_momentum_research(runtime, stock_cfg):
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    benchmark_symbol = stock_cfg.get("benchmark_symbol", "510300")
    defensive_symbol = stock_cfg.get("defensive_symbol", "511010")
    alloc_pct = float(stock_cfg.get("capital_alloc_pct", 0.7))

    model_cfg = stock_cfg.get("global_model", {})
    rebalance_days = int(model_cfg.get("rebalance_days", 20))
    momentum_lb = int(model_cfg.get("momentum_lb", 252))
    ma_window = int(model_cfg.get("ma_window", 200))
    vol_window = int(model_cfg.get("vol_window", 20))
    warmup_min_days = int(model_cfg.get("warmup_min_days", 126))
    top_n = int(model_cfg.get("top_n", 1))
    min_score = float(model_cfg.get("min_score", 0.0))
    fee = float(model_cfg.get("fee", 0.0008))

    universe = sorted(set(stock_cfg.get("universe", []) + [benchmark_symbol, defensive_symbol]))
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

    if benchmark_symbol not in frames or defensive_symbol not in frames:
        return {"error": "benchmark or defensive data missing"}

    dates = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    if len(dates) < 400:
        return {"error": "stock history too short (<400 days common)"}

    prices = pd.DataFrame({s: frames[s].loc[dates, "close"] for s in frames.keys()}, index=dates).sort_index()
    ret = prices.pct_change().fillna(0.0)

    # Keep warmup aligned with paper-forward to make production/research comparable.
    start_idx = max(momentum_lb, ma_window, vol_window, warmup_min_days) + 1
    if start_idx >= len(prices) - 1:
        return {"error": "not enough history for research backtest"}

    symbols = list(prices.columns)
    risk_symbols = [s for s in symbols if s != defensive_symbol]
    weights = {s: 0.0 for s in symbols}
    weights[defensive_symbol] = alloc_pct
    nav = [1.0]
    rets = []
    bench_nav_alloc = [1.0]

    for i in range(start_idx, len(prices) - 1):
        target = weights.copy()
        if i % rebalance_days == 0:
            scores = {}
            vols = {}
            for s in risk_symbols:
                m = prices[s].iloc[i] / prices[s].iloc[i - momentum_lb] - 1.0
                v = float(ret[s].iloc[i - vol_window + 1 : i + 1].std())
                if v <= 0:
                    continue
                vols[s] = v
                scores[s] = float(m / max(v, 1e-6))
            regime_ok = prices[benchmark_symbol].iloc[i] >= prices[benchmark_symbol].iloc[i - ma_window + 1 : i + 1].mean()
            eligible = [s for s in risk_symbols if s in scores and scores[s] > min_score]

            sleeve = {s: 0.0 for s in symbols}
            if (not regime_ok) or (not eligible):
                sleeve[defensive_symbol] = 1.0
            else:
                picks = sorted(eligible, key=lambda s: scores[s], reverse=True)[:top_n]
                inv_vol = {s: 1.0 / max(float(vols[s]), 1e-6) for s in picks}
                total_inv = sum(inv_vol.values())
                for s, v in inv_vol.items():
                    sleeve[s] = v / total_inv

            target = {s: sleeve.get(s, 0.0) * alloc_pct for s in symbols}

        turnover = sum(abs(target[s] - weights.get(s, 0.0)) for s in symbols)
        day_ret = 0.0
        for s, w in target.items():
            if w == 0.0:
                continue
            day_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        day_ret -= turnover * fee

        bench_ret_alloc = alloc_pct * (prices[benchmark_symbol].iloc[i + 1] / prices[benchmark_symbol].iloc[i] - 1.0)

        weights = target
        nav.append(nav[-1] * (1.0 + day_ret))
        rets.append(day_ret)
        bench_nav_alloc.append(bench_nav_alloc[-1] * (1.0 + bench_ret_alloc))

    summary = summarize(nav, rets, 252)
    bench_nav_alloc = pd.Series(bench_nav_alloc)
    benchmark_ann_alloc = annualized_return(bench_nav_alloc, 252)

    summary.update(
        {
            "benchmark_annual_return_alloc": float(benchmark_ann_alloc),
            "excess_annual_return_vs_alloc": float(summary["annual_return"] - benchmark_ann_alloc),
            "mode": "global_momentum_research",
        }
    )
    return summary


def backtest_stock_production(runtime, stock_cfg, risk_cfg):
    with tempfile.TemporaryDirectory() as td:
        rt = json.loads(json.dumps(runtime))
        rt["paths"]["output_dir"] = td
        summary, _history_file, _latest_file = run_paper_forward(rt, stock_cfg, risk_cfg)
    agg = summary["aggregate"]
    latest = summary["latest"]
    return {
        "periods": int(agg["periods"]),
        "annual_return": float(agg["strategy_annual_return"]),
        "max_drawdown": float(agg["strategy_max_drawdown"]),
        "sharpe": float(agg["strategy_sharpe"]),
        "final_nav": float(latest["strategy_nav"]),
        "benchmark_annual_return_alloc": float(agg["benchmark_annual_return_alloc"]),
        "excess_annual_return_vs_alloc": float(agg["excess_annual_return_vs_alloc"]),
        "mode": "global_momentum_production_aligned",
    }


def backtest_stock(runtime, stock_cfg, risk_cfg, backtest_mode_override=None):
    if stock_cfg.get("mode") == "global_momentum":
        backtest_mode = str(backtest_mode_override or stock_cfg.get("backtest_mode", "production")).lower()
        if backtest_mode == "research":
            result = backtest_stock_global_momentum_research(runtime, stock_cfg)
            if "error" in result:
                return {"error": result["error"]}
            return result

        if backtest_mode == "both":
            try:
                prod = backtest_stock_production(runtime, stock_cfg, risk_cfg)
            except Exception as e:
                return {"error": f"global_momentum production backtest failed: {e}"}
            res = backtest_stock_global_momentum_research(runtime, stock_cfg)
            if "error" in res:
                return {"error": res["error"]}
            return {
                "mode": "global_momentum_both",
                "production": prod,
                "research": res,
            }

        try:
            return backtest_stock_production(runtime, stock_cfg, risk_cfg)
        except Exception as e:
            return {"error": f"global_momentum paper-forward backtest failed: {e}"}

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stock-mode",
        choices=["production", "research", "both"],
        default=None,
        help="override stock backtest mode (default: config/stock.yaml backtest_mode)",
    )
    args = parser.parse_args()

    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
        crypto_cfg = load_yaml("config/crypto.yaml")
    except Exception as e:
        print(f"[backtest] config error: {e}")
        return EXIT_CONFIG_ERROR

    try:
        risk_cfg = load_yaml("config/risk.yaml")
        stock = backtest_stock(runtime, stock_cfg, risk_cfg, backtest_mode_override=args.stock_mode)
        crypto = backtest_crypto(runtime, crypto_cfg)
    except Exception as e:
        print(f"[backtest] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    report = {
        "ts": datetime.now().isoformat(),
        "note": "v3.6 lightweight backtest (stock: production/research/both via backtest_mode or --stock-mode; crypto: multifactor)",
        "stock": stock,
        "crypto": crypto,
    }
    compare_report = {
        "ts": report["ts"],
        "note": "stock production vs research comparison summary",
        "stock_compare": build_stock_compare(stock),
    }

    try:
        out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(out_dir)
        out_report = os.path.join(out_dir, "backtest_report.json")
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        out_compare = os.path.join(out_dir, "backtest_compare.json")
        with open(out_compare, "w", encoding="utf-8") as f:
            json.dump(compare_report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[backtest] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[backtest] report -> {out_report}")
    print(f"[backtest] compare -> {out_compare}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
