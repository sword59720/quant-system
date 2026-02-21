#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_OUTPUT_ERROR, EXIT_SIGNAL_ERROR
from core.signal import liquidity_score, max_drawdown_score, momentum_score, normalize_rank, volatility_score


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
    return float((nav / nav.cummax() - 1.0).min())


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


def compare_vs_benchmark(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> dict:
    s = summarize(strategy_nav.reset_index(drop=True))
    b = summarize(benchmark_nav.reset_index(drop=True))
    return {
        "strategy": s,
        "benchmark": b,
        "excess_annual_return": float(s["annual_return"] - b["annual_return"]),
        "excess_final_nav": float(s["final_nav"] - b["final_nav"]),
    }


def objective(strategy_nav: pd.Series, benchmark_nav: pd.Series) -> float:
    s = summarize(strategy_nav.reset_index(drop=True))
    b = summarize(benchmark_nav.reset_index(drop=True))
    return (
        2.0 * (s["annual_return"] - b["annual_return"])
        + 0.4 * (s["sharpe"] - b["sharpe"])
        + 0.1 * (s["max_drawdown"] - b["max_drawdown"])
    )


def load_stock_data(data_dir: str, symbols: list) -> tuple[pd.DataFrame, dict]:
    frames = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        frames[s] = df.sort_values("date").set_index("date")

    if len(frames) < len(symbols):
        missing = [s for s in symbols if s not in frames]
        raise RuntimeError(f"missing price files: {missing}")

    dates = sorted(set.intersection(*[set(v.index) for v in frames.values()]))
    if len(dates) < 400:
        raise RuntimeError("not enough common history (<400 rows)")

    prices = pd.DataFrame({s: frames[s].loc[dates, "close"] for s in symbols}, index=dates).sort_index()
    return prices, frames


def run_benchmark(prices: pd.DataFrame, benchmark_symbol: str, start_idx: int, end_idx: int) -> pd.Series:
    px = prices[benchmark_symbol].iloc[start_idx:end_idx]
    return (px / px.iloc[0]).reset_index(drop=True)


def run_global_momentum(
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

    rebalance_days = int(params["rebalance_days"])
    momentum_lb = int(params["momentum_lb"])
    ma_window = int(params["ma_window"])
    vol_window = int(params["vol_window"])
    top_n = int(params["top_n"])
    fee = float(params["fee"])
    min_score = float(params.get("min_score", 0.0))

    weights = {s: 0.0 for s in symbols}
    weights[defensive_symbol] = 1.0
    nav = [1.0]

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

        turnover = sum(abs(target[s] - weights[s]) for s in symbols)
        bar_ret = 0.0
        for s, w in target.items():
            if w != 0.0:
                bar_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        bar_ret -= turnover * fee

        nav.append(nav[-1] * (1.0 + bar_ret))
        weights = target

    return pd.Series(nav)


def run_dual_momentum_top2(
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

    rebalance_days = int(params["rebalance_days"])
    lb_short = int(params["lb_short"])
    lb_long = int(params["lb_long"])
    ma_window = int(params["ma_window"])
    vol_window = int(params["vol_window"])
    top_n = int(params["top_n"])
    fee = float(params["fee"])

    weights = {s: 0.0 for s in symbols}
    weights[defensive_symbol] = 1.0
    nav = [1.0]

    for i in range(start_idx, end_idx - 1):
        target = weights.copy()
        if i % rebalance_days == 0:
            scores = {}
            for s in risk_symbols:
                m1 = prices[s].iloc[i] / prices[s].iloc[i - lb_short] - 1.0
                m2 = prices[s].iloc[i] / prices[s].iloc[i - lb_long] - 1.0
                scores[s] = 0.5 * m1 + 0.5 * m2
            regime_ok = prices[benchmark_symbol].iloc[i] >= prices[benchmark_symbol].iloc[i - ma_window + 1 : i + 1].mean()
            eligible = [s for s in risk_symbols if scores[s] > 0.0]

            if (not regime_ok) or (not eligible):
                target = {s: 0.0 for s in symbols}
                target[defensive_symbol] = 1.0
            else:
                picks = sorted(eligible, key=lambda s: scores[s], reverse=True)[:top_n]
                inv_vol = {
                    s: 1.0 / max(float(ret[s].iloc[i - vol_window + 1 : i + 1].std()), 1e-6)
                    for s in picks
                }
                total_inv = sum(inv_vol.values())
                target = {s: 0.0 for s in symbols}
                for s, v in inv_vol.items():
                    target[s] = v / total_inv

        turnover = sum(abs(target[s] - weights[s]) for s in symbols)
        bar_ret = 0.0
        for s, w in target.items():
            if w != 0.0:
                bar_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        bar_ret -= turnover * fee

        nav.append(nav[-1] * (1.0 + bar_ret))
        weights = target

    return pd.Series(nav)


def run_legacy_multifactor(
    prices: pd.DataFrame,
    raw_frames: dict,
    params: dict,
    start_idx: int,
    end_idx: int,
) -> pd.Series:
    symbols = list(prices.columns)
    top_n = int(params["top_n"])
    fee = float(params["fee"])
    lbs = params["momentum_lookback_days"]
    ws = params["momentum_weights"]
    fw = params["factor_weights"]

    dates = list(prices.index)
    weights = {s: 0.0 for s in symbols}
    nav = [1.0]

    for i in range(start_idx, end_idx - 1):
        dt = dates[i]
        target = weights.copy()
        # Legacy model rebalances on Monday.
        if pd.Timestamp(dt).weekday() == 0:
            raw = {}
            for s in symbols:
                hist = raw_frames[s].loc[:dt]
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
            if "510300" in raw_frames:
                close300 = pd.to_numeric(raw_frames["510300"].loc[:dt, "close"], errors="coerce").dropna()
                if len(close300) >= 120 and close300.iloc[-1] < close300.tail(120).mean():
                    risk_mult = 0.5

            if len(raw) >= top_n:
                mr = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
                vr = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)
                dr = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)
                lr = normalize_rank({k: v["liquidity"] for k, v in raw.items()}, ascending=False)
                score = {}
                for s in raw.keys():
                    score[s] = (
                        fw.get("momentum", 0.45) * mr.get(s, 0.0)
                        + fw.get("low_vol", 0.25) * vr.get(s, 0.0)
                        + fw.get("drawdown", 0.20) * dr.get(s, 0.0)
                        + fw.get("liquidity", 0.10) * lr.get(s, 0.0)
                    )
                picks = [x[0] for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]]
                # Keep full investment but apply legacy risk multiplier.
                tw = risk_mult / top_n
                target = {s: (tw if s in picks else 0.0) for s in symbols}
            else:
                target = {s: 0.0 for s in symbols}

        turnover = sum(abs(target[s] - weights[s]) for s in symbols)
        bar_ret = 0.0
        for s, w in target.items():
            if w != 0.0:
                bar_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        bar_ret -= turnover * fee

        nav.append(nav[-1] * (1.0 + bar_ret))
        weights = target

    return pd.Series(nav)


def run_walk_forward_global(
    prices: pd.DataFrame,
    benchmark_symbol: str,
    defensive_symbol: str,
    base_params: dict,
) -> dict:
    param_grid = list(
        itertools.product(
            [10, 20],
            [180, 252],
            [120, 200],
            [20, 40],
            [1, 2],
        )
    )

    train_window = 756
    test_window = 126
    max_warmup = max(252, 200, 40, 126) + 1
    anchor = max_warmup + train_window

    if anchor >= len(prices) - 2:
        return {"error": "history too short for walk-forward"}

    chain_nav = [1.0]
    segments = []
    param_counter = Counter()
    i = anchor

    while i < len(prices) - 1:
        tr0 = i - train_window
        tr1 = i
        te1 = min(i + test_window, len(prices) - 1)

        best_score = None
        best_params = None
        for rebalance_days, momentum_lb, ma_window, vol_window, top_n in param_grid:
            p = {
                "rebalance_days": rebalance_days,
                "momentum_lb": momentum_lb,
                "ma_window": ma_window,
                "vol_window": vol_window,
                "top_n": top_n,
                "fee": base_params["fee"],
                "min_score": base_params.get("min_score", 0.0),
            }
            warmup = max(momentum_lb, ma_window, vol_window, 126) + 1
            s0 = max(tr0, warmup)
            if tr1 - s0 < 250:
                continue
            s_nav = run_global_momentum(prices, benchmark_symbol, defensive_symbol, p, s0, tr1)
            b_nav = run_benchmark(prices, benchmark_symbol, s0, tr1)
            score = objective(s_nav, b_nav)
            if (best_score is None) or (score > best_score):
                best_score = score
                best_params = p

        if best_params is None:
            break

        test_nav = run_global_momentum(prices, benchmark_symbol, defensive_symbol, best_params, i, te1)
        test_bench = run_benchmark(prices, benchmark_symbol, i, te1)
        seg_result = compare_vs_benchmark(test_nav, test_bench)
        segments.append(
            {
                "train_start": str(prices.index[tr0].date()),
                "train_end": str(prices.index[tr1 - 1].date()),
                "test_start": str(prices.index[i].date()),
                "test_end": str(prices.index[te1 - 1].date()),
                "params": best_params,
                "result": seg_result,
            }
        )

        param_counter.update([tuple(best_params.items())])
        scaled = test_nav / test_nav.iloc[0] * chain_nav[-1]
        chain_nav.extend(scaled.iloc[1:].tolist())
        i = te1

    if len(chain_nav) < 2:
        return {"error": "walk-forward produced empty nav"}

    chain_nav_s = pd.Series(chain_nav)
    start_idx = len(prices) - len(chain_nav_s)
    chain_bench = run_benchmark(prices, benchmark_symbol, start_idx, len(prices))
    top_params = []
    for params_tuple, cnt in param_counter.most_common(3):
        top_params.append({"count": int(cnt), "params": dict(params_tuple)})

    return {
        "train_window_days": train_window,
        "test_window_days": test_window,
        "segments": segments,
        "top_selected_params": top_params,
        "aggregate": compare_vs_benchmark(chain_nav_s, chain_bench),
    }


def evaluate_model_windows(
    strategy_nav_full: pd.Series,
    benchmark_nav_full: pd.Series,
    strategy_nav_oos_2023: pd.Series,
    benchmark_nav_oos_2023: pd.Series,
    strategy_nav_oos_2024: pd.Series,
    benchmark_nav_oos_2024: pd.Series,
) -> dict:
    return {
        "full_period": compare_vs_benchmark(strategy_nav_full, benchmark_nav_full),
        "oos_2023": compare_vs_benchmark(strategy_nav_oos_2023, benchmark_nav_oos_2023),
        "oos_2024": compare_vs_benchmark(strategy_nav_oos_2024, benchmark_nav_oos_2024),
    }


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
    except Exception as e:
        print(f"[backtest] config error: {e}")
        return EXIT_CONFIG_ERROR

    try:
        data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
        out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(out_dir)

        benchmark_symbol = stock_cfg.get("benchmark_symbol", "510300")
        defensive_symbol = stock_cfg.get("defensive_symbol", "511010")
        universe = sorted(set(stock_cfg.get("universe", []) + [benchmark_symbol, defensive_symbol]))

        prices, raw_frames = load_stock_data(data_dir, universe)

        global_params = {
            "rebalance_days": int(stock_cfg.get("global_model", {}).get("rebalance_days", 20)),
            "momentum_lb": int(stock_cfg.get("global_model", {}).get("momentum_lb", 252)),
            "ma_window": int(stock_cfg.get("global_model", {}).get("ma_window", 200)),
            "vol_window": int(stock_cfg.get("global_model", {}).get("vol_window", 20)),
            "top_n": int(stock_cfg.get("global_model", {}).get("top_n", 1)),
            "fee": float(stock_cfg.get("global_model", {}).get("fee", 0.0008)),
            "min_score": float(stock_cfg.get("global_model", {}).get("min_score", 0.0)),
        }

        dual_params = {
            "rebalance_days": 10,
            "lb_short": 126,
            "lb_long": 252,
            "ma_window": 200,
            "vol_window": 20,
            "top_n": 2,
            "fee": global_params["fee"],
        }

        legacy_params = {
            "top_n": int(stock_cfg.get("signal", {}).get("top_n", 2)),
            "momentum_lookback_days": stock_cfg.get("signal", {}).get("momentum_lookback_days", [20, 60]),
            "momentum_weights": stock_cfg.get("signal", {}).get("momentum_weights", [0.6, 0.4]),
            "factor_weights": stock_cfg.get("signal", {}).get(
                "factor_weights",
                {"momentum": 0.45, "low_vol": 0.25, "drawdown": 0.20, "liquidity": 0.10},
            ),
            "fee": global_params["fee"],
        }

        warmup_global = max(global_params["momentum_lb"], global_params["ma_window"], global_params["vol_window"], 126) + 1
        warmup_dual = max(dual_params["lb_long"], dual_params["ma_window"], dual_params["vol_window"], 126) + 1
        warmup_legacy = max(legacy_params["momentum_lookback_days"][1], 120, 60, 20, 126) + 1
        start_idx = max(warmup_global, warmup_dual, warmup_legacy)
        end_idx = len(prices) - 1

        oos_2023_idx = max(start_idx, int(prices.index.searchsorted(pd.Timestamp("2023-01-03"))))
        oos_2024_idx = max(start_idx, int(prices.index.searchsorted(pd.Timestamp("2024-01-02"))))

        benchmark_full = run_benchmark(prices, benchmark_symbol, start_idx, end_idx)
        benchmark_oos_2023 = run_benchmark(prices, benchmark_symbol, oos_2023_idx, end_idx)
        benchmark_oos_2024 = run_benchmark(prices, benchmark_symbol, oos_2024_idx, end_idx)

        model_results = []

        global_full = run_global_momentum(prices, benchmark_symbol, defensive_symbol, global_params, start_idx, end_idx)
        global_oos_2023 = run_global_momentum(prices, benchmark_symbol, defensive_symbol, global_params, oos_2023_idx, end_idx)
        global_oos_2024 = run_global_momentum(prices, benchmark_symbol, defensive_symbol, global_params, oos_2024_idx, end_idx)
        model_results.append(
            {
                "model": "global_momentum_current",
                "params": global_params,
                "window_results": evaluate_model_windows(
                    global_full,
                    benchmark_full,
                    global_oos_2023,
                    benchmark_oos_2023,
                    global_oos_2024,
                    benchmark_oos_2024,
                ),
            }
        )

        dual_full = run_dual_momentum_top2(prices, benchmark_symbol, defensive_symbol, dual_params, start_idx, end_idx)
        dual_oos_2023 = run_dual_momentum_top2(prices, benchmark_symbol, defensive_symbol, dual_params, oos_2023_idx, end_idx)
        dual_oos_2024 = run_dual_momentum_top2(prices, benchmark_symbol, defensive_symbol, dual_params, oos_2024_idx, end_idx)
        model_results.append(
            {
                "model": "dual_momentum_top2",
                "params": dual_params,
                "window_results": evaluate_model_windows(
                    dual_full,
                    benchmark_full,
                    dual_oos_2023,
                    benchmark_oos_2023,
                    dual_oos_2024,
                    benchmark_oos_2024,
                ),
            }
        )

        legacy_full = run_legacy_multifactor(prices, raw_frames, legacy_params, start_idx, end_idx)
        legacy_oos_2023 = run_legacy_multifactor(prices, raw_frames, legacy_params, oos_2023_idx, end_idx)
        legacy_oos_2024 = run_legacy_multifactor(prices, raw_frames, legacy_params, oos_2024_idx, end_idx)
        model_results.append(
            {
                "model": "legacy_multifactor_weekly",
                "params": legacy_params,
                "window_results": evaluate_model_windows(
                    legacy_full,
                    benchmark_full,
                    legacy_oos_2023,
                    benchmark_oos_2023,
                    legacy_oos_2024,
                    benchmark_oos_2024,
                ),
            }
        )

        fee_scenarios = []
        for fee in [0.0008, 0.0012, 0.0015]:
            p = dict(global_params)
            p["fee"] = fee
            nav_full = run_global_momentum(prices, benchmark_symbol, defensive_symbol, p, start_idx, end_idx)
            nav_oos = run_global_momentum(prices, benchmark_symbol, defensive_symbol, p, oos_2023_idx, end_idx)
            fee_scenarios.append(
                {
                    "fee": fee,
                    "full_period": compare_vs_benchmark(nav_full, benchmark_full),
                    "oos_2023": compare_vs_benchmark(nav_oos, benchmark_oos_2023),
                }
            )

        perturbations = []
        for rebalance_days, momentum_lb, ma_window, vol_window, top_n in [
            (10, 252, 200, 20, 1),
            (20, 180, 200, 20, 1),
            (20, 252, 120, 20, 1),
            (20, 252, 240, 20, 1),
            (20, 252, 200, 40, 1),
            (20, 252, 200, 20, 2),
        ]:
            p = dict(global_params)
            p.update(
                {
                    "rebalance_days": rebalance_days,
                    "momentum_lb": momentum_lb,
                    "ma_window": ma_window,
                    "vol_window": vol_window,
                    "top_n": top_n,
                }
            )
            nav_oos = run_global_momentum(prices, benchmark_symbol, defensive_symbol, p, oos_2023_idx, end_idx)
            perturbations.append({"params": p, "oos_2023": compare_vs_benchmark(nav_oos, benchmark_oos_2023)})

        walk_forward = run_walk_forward_global(prices, benchmark_symbol, defensive_symbol, global_params)

        model_results = sorted(
            model_results,
            key=lambda x: x["window_results"]["oos_2023"]["excess_annual_return"],
            reverse=True,
        )

        best_model = model_results[0]["model"]
        current_oos = model_results[0]["window_results"]["oos_2023"]["excess_annual_return"]
        current_oos_2024 = model_results[0]["window_results"]["oos_2024"]["excess_annual_return"]
        recommendation = {
            "selected_model": best_model,
            "reason": "best out-of-sample excess annual return on current dataset",
            "gate_passed": bool(current_oos > 0 and current_oos_2024 > 0),
        }

        out = {
            "ts": datetime.now().isoformat(),
            "note": "stock model lab: comparison + robustness + walk-forward",
            "assumptions": {
                "benchmark_basis": "raw_100pct_benchmark",
                "execution_guard": "not_applied",
                "risk_overlay": "not_applied",
                "sleeve_exposure": "full_100pct",
            },
            "date_ranges": {
                "full_start": str(prices.index[start_idx].date()),
                "full_end": str(prices.index[end_idx - 1].date()),
                "oos_2023_start": str(prices.index[oos_2023_idx].date()),
                "oos_2024_start": str(prices.index[oos_2024_idx].date()),
            },
            "benchmark_symbol": benchmark_symbol,
            "defensive_symbol": defensive_symbol,
            "model_comparison": model_results,
            "robustness": {
                "fee_sensitivity": fee_scenarios,
                "parameter_perturbation": perturbations,
                "walk_forward": walk_forward,
            },
            "recommendation": recommendation,
        }
    except Exception as e:
        print(f"[backtest] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    try:
        out_file = os.path.join(out_dir, "stock_model_lab_report.json")
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
