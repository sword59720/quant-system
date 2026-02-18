#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_OUTPUT_ERROR, EXIT_SIGNAL_ERROR


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def annualized_return(nav: pd.Series, periods_per_year: int = 252) -> float:
    if len(nav) < 2:
        return 0.0
    total = float(nav.iloc[-1] / nav.iloc[0])
    years = max((len(nav) - 1) / periods_per_year, 1e-9)
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def sharpe_ratio(nav: pd.Series, periods_per_year: int = 252) -> float:
    ret = nav.pct_change().dropna()
    if len(ret) < 2:
        return 0.0
    sd = ret.std()
    if sd == 0:
        return 0.0
    return float((ret.mean() / sd) * math.sqrt(periods_per_year))


def summarize(nav: pd.Series) -> dict:
    return {
        "periods": int(max(len(nav) - 1, 0)),
        "annual_return": annualized_return(nav, 252),
        "max_drawdown": max_drawdown(nav),
        "sharpe": sharpe_ratio(nav, 252),
        "final_nav": float(nav.iloc[-1]) if len(nav) else 1.0,
    }


def load_price_frame(data_dir: str, symbols: list) -> pd.DataFrame:
    frames = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        frames[s] = df.sort_values("date").set_index("date")[["close"]]

    if len(frames) < len(symbols):
        missing = [s for s in symbols if s not in frames]
        raise RuntimeError(f"missing price files: {missing}")

    dates = sorted(set.intersection(*[set(v.index) for v in frames.values()]))
    if len(dates) < 400:
        raise RuntimeError("not enough common stock history (<400 rows)")

    px = pd.DataFrame({s: frames[s].loc[dates, "close"] for s in symbols}, index=dates).sort_index()
    return px


def run_strategy(
    prices: pd.DataFrame,
    benchmark_symbol: str,
    defensive_symbol: str,
    params: dict,
    start_idx: int,
    end_idx: int,
) -> pd.Series:
    symbols = list(prices.columns)
    risk_symbols = [s for s in symbols if s != defensive_symbol]
    ret = prices.pct_change().fillna(0.0)

    nav = [1.0]
    weights = {s: 0.0 for s in symbols}
    weights[defensive_symbol] = 1.0

    rebalance_days = int(params["rebalance_days"])
    momentum_lb = int(params["momentum_lb"])
    ma_window = int(params["ma_window"])
    vol_window = int(params["vol_window"])
    top_n = int(params["top_n"])
    fee = float(params["fee"])
    min_score = float(params.get("min_score", 0.0))

    for i in range(start_idx, end_idx - 1):
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

            if (not regime_ok) or (not eligible):
                target = {s: 0.0 for s in symbols}
                target[defensive_symbol] = 1.0
            else:
                picks = sorted(eligible, key=lambda s: scores[s], reverse=True)[:top_n]
                inv_vol = {
                    s: 1.0 / max(float(vols[s]), 1e-6)
                    for s in picks
                }
                total_inv = sum(inv_vol.values())
                target = {s: 0.0 for s in symbols}
                for s, v in inv_vol.items():
                    target[s] = v / total_inv

        turnover = sum(abs(target[s] - weights.get(s, 0.0)) for s in symbols)
        bar_ret = 0.0
        for s, w in target.items():
            if w == 0.0:
                continue
            bar_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        bar_ret -= turnover * fee

        nav.append(nav[-1] * (1.0 + bar_ret))
        weights = target

    return pd.Series(nav)


def run_benchmark(prices: pd.DataFrame, symbol: str, start_idx: int, end_idx: int) -> pd.Series:
    px = prices[symbol].iloc[start_idx:end_idx]
    return (px / px.iloc[0]).reset_index(drop=True)


def compare_metrics(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> dict:
    s = summarize(strategy_nav.reset_index(drop=True))
    b = summarize(benchmark_nav.reset_index(drop=True))
    return {
        "strategy": s,
        "benchmark": b,
        "excess_annual_return": float(s["annual_return"] - b["annual_return"]),
        "excess_final_nav": float(s["final_nav"] - b["final_nav"]),
    }


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
    except Exception as e:
        print(f"[backtest] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_OK
    if not stock_cfg.get("enabled", False):
        print("[stock] disabled")
        return EXIT_OK

    try:
        data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
        output_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(output_dir)

        universe = stock_cfg.get("universe", [])
        benchmark_symbol = stock_cfg.get("benchmark_symbol", "510300")
        defensive_symbol = stock_cfg.get("defensive_symbol", "511010")

        if benchmark_symbol not in universe:
            universe = sorted(set(universe + [benchmark_symbol]))
        if defensive_symbol not in universe:
            universe = sorted(set(universe + [defensive_symbol]))

        prices = load_price_frame(data_dir, universe)

        params = {
            "rebalance_days": int(stock_cfg.get("global_model", {}).get("rebalance_days", 20)),
            "momentum_lb": int(stock_cfg.get("global_model", {}).get("momentum_lb", 252)),
            "top_n": int(stock_cfg.get("global_model", {}).get("top_n", 1)),
            "ma_window": int(stock_cfg.get("global_model", {}).get("ma_window", 200)),
            "vol_window": int(stock_cfg.get("global_model", {}).get("vol_window", 20)),
            "fee": float(stock_cfg.get("global_model", {}).get("fee", 0.0008)),
            "min_score": float(stock_cfg.get("global_model", {}).get("min_score", 0.0)),
        }

        warmup = max(params["momentum_lb"], params["ma_window"], params["vol_window"], 126) + 1
        full_start = warmup
        full_end = len(prices) - 1

        split_date = pd.Timestamp(stock_cfg.get("model_split_date", "2023-01-03"))
        split_idx = int(prices.index.searchsorted(split_date))
        oos_start = max(split_idx, warmup)
        oos_end = len(prices) - 1

        full_s = run_strategy(prices, benchmark_symbol, defensive_symbol, params, full_start, full_end)
        full_b = run_benchmark(prices, benchmark_symbol, full_start, full_end)
        oos_s = run_strategy(prices, benchmark_symbol, defensive_symbol, params, oos_start, oos_end)
        oos_b = run_benchmark(prices, benchmark_symbol, oos_start, oos_end)

        out = {
            "ts": datetime.now().isoformat(),
            "model": "dual_momentum_trend_ivol_defense",
            "note": "12m momentum + MA200 regime filter + inverse-vol weight + defensive bond ETF",
            "assumptions": {
                "benchmark_basis": "raw_100pct_benchmark",
                "execution_guard": "not_applied",
                "risk_overlay": "not_applied",
                "sleeve_exposure": "full_100pct",
            },
            "benchmark_symbol": benchmark_symbol,
            "defensive_symbol": defensive_symbol,
            "params": params,
            "out_of_sample": compare_metrics(oos_s, oos_b),
            "full_period": compare_metrics(full_s, full_b),
            "references": [
                "Time Series Momentum (Moskowitz, Ooi, Pedersen, 2012)",
                "A Century of Evidence on Trend-Following Investing (Hurst, Ooi, Pedersen, 2017)",
                "Volatility-Managed Portfolios (Moreira, Muir, 2017)",
            ],
        }
    except Exception as e:
        print(f"[backtest] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    try:
        out_file = os.path.join(output_dir, "stock_global_model_report.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[backtest] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[backtest] report -> {out_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
