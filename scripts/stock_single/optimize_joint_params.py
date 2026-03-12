#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_DISABLED, EXIT_OK, EXIT_SIGNAL_ERROR
from scripts.stock_single import scoring_core
from scripts.stock_single.backtest_stock_single import (
    annualized_return,
    build_panel,
    build_target_weights,
    load_benchmark_close,
    load_daily_frames,
    max_drawdown,
    resolve_signal_thresholds,
    sharpe_ratio,
)
from scripts.stock_single.fetch_stock_single_data import load_symbols


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Joint parameter optimization for stock_single with MDD constraint.")
    parser.add_argument("--start-date", default=None, help="Backtest start date, format YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Backtest end date, format YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=20260226, help="Random seed for sampling candidates")
    parser.add_argument("--coarse-evals", type=int, default=180, help="Max coarse-stage evaluations")
    parser.add_argument("--fine-evals", type=int, default=120, help="Max fine-stage evaluations")
    parser.add_argument("--anchors", type=int, default=10, help="Top anchors from coarse stage for fine search")
    parser.add_argument(
        "--force-capital-alloc",
        type=float,
        default=None,
        help="Force capital_alloc_pct to a fixed value (e.g. 1.0)",
    )
    return parser.parse_args()


def candidate_key(candidate: dict) -> Tuple[float, int, float, float, int, int]:
    return (
        float(candidate["capital_alloc_pct"]),
        int(candidate["max_positions"]),
        float(candidate["single_max"]),
        float(candidate["target_weight"]),
        int(candidate["buy_top_k"]),
        int(candidate["hold_top_k"]),
    )


@dataclass
class BacktestContext:
    runtime: dict
    stock_single: dict
    symbols: List[str]
    dates: pd.Index
    close_df: pd.DataFrame
    ret_df: pd.DataFrame
    vol20: pd.DataFrame
    score_df: pd.DataFrame
    benchmark_close: pd.Series
    signal_ret_df: pd.DataFrame
    vol_z_df: pd.DataFrame
    tradable_by_i: Dict[int, set]
    score_row_by_i: Dict[int, pd.Series]
    trade_ret_by_i: Dict[int, pd.Series]
    benchmark_base_ret_by_i: Dict[int, float]
    start_idx: int
    loop_end_i: int
    signal_cfg: dict
    risk_cfg: dict
    fast_cfg: dict
    industry_diversification: dict
    fee_buy_bps: float
    fee_sell_bps: float
    stamp_tax_sell_bps: float
    slippage_bps: float
    buy_cost_rate: float
    sell_cost_rate: float
    buy_threshold: float
    sell_threshold: float
    mdd_limit: float
    trigger_portfolio_1d: float
    trigger_single_1d: float
    trigger_vol_zscore: float
    block_new_buys_on_trigger: bool
    force_sell_on_single_crash: bool
    feature_coverage: dict
    base_score_weights: dict


def build_context(runtime: dict, stock_single: dict, args) -> BacktestContext:
    symbols_meta = load_symbols(stock_single)
    if not symbols_meta:
        raise RuntimeError("no A-share stock symbols found")

    data_cfg = stock_single.get("data", {})
    daily_dir = data_cfg.get("daily_output_dir", "./data/stock_single/daily")
    frames = load_daily_frames(daily_dir, symbols_meta)
    if not frames:
        raise RuntimeError(f"no valid daily files found in {daily_dir}")

    symbols = sorted(frames.keys())
    close_df = build_panel(frames, "close")
    vol_df = build_panel(frames, "volume")
    amt_df = build_panel(frames, "amount")
    if close_df.empty:
        raise RuntimeError("daily close panel is empty")

    backtest_cfg = stock_single.get("backtest", {})
    signal_cfg = dict(stock_single.get("signal", {}))
    risk_cfg = dict(stock_single.get("risk", {}))
    risk_cfg_opt = copy.deepcopy(risk_cfg)
    # Keep dynamic risk budget for adaptive exposure; disable stop-loss in optimization
    # loop to avoid extra noise from simplified cost-basis assumptions.
    risk_cfg_opt.setdefault("dynamic_risk_budget", {})["enabled"] = bool(
        risk_cfg_opt.get("dynamic_risk_budget", {}).get("enabled", True)
    )
    risk_cfg_opt.setdefault("stop_loss", {})["enabled"] = False
    fast_cfg = dict(risk_cfg.get("fast_check", {}))
    industry_diversification = dict(stock_single.get("industry_diversification", {}))

    fee_buy_bps = float(backtest_cfg.get("fee_buy_bps", 1.5))
    fee_sell_bps = float(backtest_cfg.get("fee_sell_bps", 1.5))
    stamp_tax_sell_bps = float(backtest_cfg.get("stamp_tax_sell_bps", 5.0))
    slippage_bps = float(backtest_cfg.get("slippage_bps", 2.5))
    buy_cost_rate = (fee_buy_bps + slippage_bps) * 1e-4
    sell_cost_rate = (fee_sell_bps + stamp_tax_sell_bps + slippage_bps) * 1e-4

    score_bundle = scoring_core.compute_score_bundle(
        close_df=close_df,
        vol_df=vol_df,
        amt_df=amt_df,
        data_cfg=data_cfg,
        backtest_cfg=backtest_cfg,
    )
    ret_df = score_bundle["ret_df"]
    vol20 = score_bundle["vol20"]
    score_df = score_bundle["score_df"]
    feature_coverage = score_bundle["coverage"]
    base_score_weights = score_bundle["score_weights"]

    dates = close_df.index
    warmup = int(max(backtest_cfg.get("warmup_days", 80), 61))
    if len(dates) < warmup + 3:
        raise RuntimeError("not enough history for backtest")

    start_idx = warmup + 1
    if args.start_date:
        start_idx = max(start_idx, int(dates.searchsorted(pd.Timestamp(args.start_date), side="left")))
    end_price_idx = len(dates) - 1
    if args.end_date:
        end_price_idx = min(end_price_idx, int(dates.searchsorted(pd.Timestamp(args.end_date), side="right") - 1))
    loop_end_i = min(len(dates) - 2, end_price_idx - 1)
    if loop_end_i < start_idx:
        raise RuntimeError("invalid date range after warmup")

    benchmark_symbol = str(backtest_cfg.get("benchmark_symbol", "510300"))
    benchmark_close = load_benchmark_close(runtime, benchmark_symbol).reindex(dates)

    buy_threshold = float(signal_cfg.get("buy_threshold", 1.0))
    sell_threshold = float(signal_cfg.get("sell_threshold", -0.5))
    mdd_limit = float(backtest_cfg.get("optimize_max_drawdown_limit", -0.15))
    trigger_portfolio_1d = float(
        fast_cfg.get(
            "trigger_portfolio_ret_1d",
            float(fast_cfg.get("trigger_portfolio_ret_5m", -0.008)) * (48.0 ** 0.5),
        )
    )
    trigger_single_1d = float(
        fast_cfg.get(
            "trigger_single_ret_1d",
            float(fast_cfg.get("trigger_single_ret_5m", -0.020)) * (48.0 ** 0.5),
        )
    )
    trigger_vol_zscore = float(fast_cfg.get("trigger_vol_zscore", 3.0))
    block_new_buys_on_trigger = bool(fast_cfg.get("block_new_buys_on_trigger", True))
    force_sell_on_single_crash = bool(fast_cfg.get("force_sell_on_single_crash", True))

    signal_ret_df = close_df.pct_change(fill_method=None).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
    ret_abs = signal_ret_df.abs()
    vol_z_df = ((ret_abs - ret_abs.rolling(20, min_periods=10).mean()) / ret_abs.rolling(20, min_periods=10).std())
    vol_z_df = vol_z_df.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)

    tradable_by_i: Dict[int, set] = {}
    score_row_by_i: Dict[int, pd.Series] = {}
    trade_ret_by_i: Dict[int, pd.Series] = {}
    benchmark_base_ret_by_i: Dict[int, float] = {}
    for i in range(start_idx, loop_end_i + 1):
        idx_signal = i - 1
        trade_date = dates[i]
        next_date = dates[i + 1]

        px0 = close_df.loc[trade_date]
        px1 = close_df.loc[next_date]
        tradable_mask = px0.notna() & px1.notna() & (px0 > 0)
        tradable_symbols = set(px0.index[tradable_mask.values].tolist())
        tradable_by_i[i] = tradable_symbols

        score_row_by_i[i] = scoring_core.apply_realtime_adjustments(
            score_row=score_df.iloc[idx_signal],
            tradable_symbols=tradable_symbols,
            close_df=close_df,
            vol20=vol20,
            idx_signal=idx_signal,
            signal_cfg=signal_cfg,
        )

        row_ret = (px1 / px0 - 1.0).replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)
        trade_ret_by_i[i] = row_ret

        if (
            (not benchmark_close.empty)
            and pd.notna(benchmark_close.get(trade_date))
            and pd.notna(benchmark_close.get(next_date))
            and float(benchmark_close.loc[trade_date]) > 0
        ):
            benchmark_base_ret_by_i[i] = float(benchmark_close.loc[next_date] / benchmark_close.loc[trade_date] - 1.0)
        else:
            benchmark_base_ret_by_i[i] = float(row_ret[tradable_mask].mean()) if tradable_mask.any() else 0.0

    return BacktestContext(
        runtime=runtime,
        stock_single=stock_single,
        symbols=symbols,
        dates=dates,
        close_df=close_df,
        ret_df=ret_df,
        vol20=vol20,
        score_df=score_df,
        benchmark_close=benchmark_close,
        signal_ret_df=signal_ret_df,
        vol_z_df=vol_z_df,
        tradable_by_i=tradable_by_i,
        score_row_by_i=score_row_by_i,
        trade_ret_by_i=trade_ret_by_i,
        benchmark_base_ret_by_i=benchmark_base_ret_by_i,
        start_idx=start_idx,
        loop_end_i=loop_end_i,
        signal_cfg=signal_cfg,
        risk_cfg=risk_cfg_opt,
        fast_cfg=fast_cfg,
        industry_diversification=industry_diversification,
        fee_buy_bps=fee_buy_bps,
        fee_sell_bps=fee_sell_bps,
        stamp_tax_sell_bps=stamp_tax_sell_bps,
        slippage_bps=slippage_bps,
        buy_cost_rate=buy_cost_rate,
        sell_cost_rate=sell_cost_rate,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        mdd_limit=mdd_limit,
        trigger_portfolio_1d=trigger_portfolio_1d,
        trigger_single_1d=trigger_single_1d,
        trigger_vol_zscore=trigger_vol_zscore,
        block_new_buys_on_trigger=block_new_buys_on_trigger,
        force_sell_on_single_crash=force_sell_on_single_crash,
        feature_coverage=feature_coverage,
        base_score_weights=base_score_weights,
    )


def compute_risk_overlay_fast(
    ctx: BacktestContext,
    idx_signal: int,
    weights: Dict[str, float],
    symbols: List[str],
) -> dict:
    if idx_signal <= 0:
        return {
            "stage": "normal",
            "triggered": False,
            "block_new_buys": False,
            "force_sell_symbols": [],
            "portfolio_ret_1d": 0.0,
            "single_crash_count": 0,
            "vol_spike_count": 0,
        }

    ret_row = ctx.signal_ret_df.iloc[idx_signal]
    z_row = ctx.vol_z_df.iloc[idx_signal]

    portfolio_ret_1d = 0.0
    for s, w in weights.items():
        portfolio_ret_1d += float(w) * float(ret_row.get(s, 0.0))

    single_crash_symbols = sorted([s for s in symbols if float(ret_row.get(s, 0.0)) <= ctx.trigger_single_1d])
    vol_spike_symbols = sorted(
        [
            s
            for s in symbols
            if float(z_row.get(s, 0.0)) >= ctx.trigger_vol_zscore and float(ret_row.get(s, 0.0)) < 0.0
        ]
    )

    portfolio_trigger = bool(portfolio_ret_1d <= ctx.trigger_portfolio_1d)
    single_trigger = bool(len(single_crash_symbols) > 0)
    vol_trigger = bool(len(vol_spike_symbols) > 0)
    triggered = bool(portfolio_trigger or single_trigger or vol_trigger)

    force_sell = sorted(set(single_crash_symbols + vol_spike_symbols))
    if not ctx.force_sell_on_single_crash:
        force_sell = []
    block_new_buys = bool(triggered and ctx.block_new_buys_on_trigger)

    return {
        "stage": "risk" if triggered else "normal",
        "triggered": triggered,
        "block_new_buys": block_new_buys,
        "force_sell_symbols": force_sell,
        "portfolio_ret_1d": float(portfolio_ret_1d),
        "single_crash_count": int(len(single_crash_symbols)),
        "vol_spike_count": int(len(vol_spike_symbols)),
    }


def evaluate_candidate(ctx: BacktestContext, candidate: dict) -> dict:
    signal_cfg = dict(ctx.signal_cfg)
    signal_cfg["trigger_mode"] = "topk"
    signal_cfg["buy_top_k"] = int(candidate["buy_top_k"])
    signal_cfg["hold_top_k"] = int(max(candidate["hold_top_k"], candidate["buy_top_k"]))

    capital_alloc_pct = float(candidate["capital_alloc_pct"])
    max_positions = int(candidate["max_positions"])
    target_weight = float(candidate["target_weight"])
    single_max = float(candidate["single_max"])

    weights = {s: 0.0 for s in ctx.symbols}
    strategy_rets: List[float] = []
    benchmark_rets: List[float] = []

    for i in range(ctx.start_idx, ctx.loop_end_i + 1):
        idx_signal = i - 1
        tradable_symbols = ctx.tradable_by_i[i]

        risk_overlay = compute_risk_overlay_fast(
            ctx=ctx,
            idx_signal=idx_signal,
            weights=weights,
            symbols=ctx.symbols,
        )
        score_row = ctx.score_row_by_i[i]
        eff_buy_threshold, eff_sell_threshold, _ = resolve_signal_thresholds(
            score_row=score_row,
            tradable_symbols=tradable_symbols,
            signal_cfg=signal_cfg,
            max_positions=max_positions,
            buy_threshold_static=ctx.buy_threshold,
            sell_threshold_static=ctx.sell_threshold,
        )

        target, _ = build_target_weights(
            symbols=ctx.symbols,
            current_weights=weights,
            score_row=score_row,
            tradable_symbols=tradable_symbols,
            risk_overlay=risk_overlay,
            capital_alloc_pct=capital_alloc_pct,
            max_positions=max_positions,
            target_weight=target_weight,
            single_max=single_max,
            buy_threshold=eff_buy_threshold,
            sell_threshold=eff_sell_threshold,
            close_df=ctx.close_df,
            idx_signal=idx_signal,
            risk_cfg=ctx.risk_cfg,
            industry_diversification=ctx.industry_diversification,
        )

        buy_turnover = 0.0
        sell_turnover = 0.0
        for s in ctx.symbols:
            delta = float(target.get(s, 0.0) - weights.get(s, 0.0))
            if delta > 0:
                buy_turnover += delta
            elif delta < 0:
                sell_turnover += -delta
        trade_cost = buy_turnover * ctx.buy_cost_rate + sell_turnover * ctx.sell_cost_rate

        weights = target

        port_ret = 0.0
        row_ret = ctx.trade_ret_by_i[i]
        for s, w in weights.items():
            if w <= 0.0:
                continue
            port_ret += float(w) * float(row_ret.get(s, 0.0))
        port_ret -= trade_cost
        strategy_rets.append(float(port_ret))
        benchmark_rets.append(float(capital_alloc_pct * ctx.benchmark_base_ret_by_i[i]))

    if len(strategy_rets) < 2:
        raise RuntimeError("backtest periods too short after filtering")

    ret_s = pd.Series(strategy_rets, dtype=float)
    ret_b = pd.Series(benchmark_rets, dtype=float)
    nav_s = pd.concat([pd.Series([1.0], dtype=float), (1.0 + ret_s).cumprod()], ignore_index=True)
    nav_b = pd.concat([pd.Series([1.0], dtype=float), (1.0 + ret_b).cumprod()], ignore_index=True)

    annual = annualized_return(nav_s, 252)
    bench_annual = annualized_return(nav_b, 252)
    mdd = max_drawdown(nav_s)
    sharpe = sharpe_ratio(ret_s, 252)
    feasible = bool(mdd >= ctx.mdd_limit)

    out = {
        "params": {
            "capital_alloc_pct": capital_alloc_pct,
            "max_positions": max_positions,
            "single_max": single_max,
            "target_weight": target_weight,
            "buy_top_k": int(candidate["buy_top_k"]),
            "hold_top_k": int(candidate["hold_top_k"]),
        },
        "annual_return": float(annual),
        "benchmark_annual_return": float(bench_annual),
        "excess_annual_return": float(annual - bench_annual),
        "max_drawdown": float(mdd),
        "sharpe": float(sharpe),
        "feasible": feasible,
    }
    return out


def better_feasible(candidate: dict, incumbent: dict) -> bool:
    if incumbent is None:
        return True
    if candidate["annual_return"] > incumbent["annual_return"]:
        return True
    if candidate["annual_return"] < incumbent["annual_return"]:
        return False
    if candidate["max_drawdown"] > incumbent["max_drawdown"]:
        return True
    if candidate["max_drawdown"] < incumbent["max_drawdown"]:
        return False
    return candidate["sharpe"] > incumbent["sharpe"]


def better_infeasible(candidate: dict, incumbent: dict) -> bool:
    if incumbent is None:
        return True
    if candidate["max_drawdown"] > incumbent["max_drawdown"]:
        return True
    if candidate["max_drawdown"] < incumbent["max_drawdown"]:
        return False
    if candidate["annual_return"] > incumbent["annual_return"]:
        return True
    if candidate["annual_return"] < incumbent["annual_return"]:
        return False
    return candidate["sharpe"] > incumbent["sharpe"]


def generate_coarse_candidates(rng: random.Random) -> List[dict]:
    capital_alloc_values = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00]
    max_positions_values = [4, 6, 8, 10, 12]
    single_max_values = [0.04, 0.06, 0.08, 0.10, 0.12, 0.15]
    target_weight_values = [0.04, 0.06, 0.08, 0.10]
    buy_top_k_values = [4, 6, 8, 10, 12, 15]
    hold_gap_values = [2, 4, 6, 8, 10]

    out = []
    for cap, mp, sm, tw, bt, hg in product(
        capital_alloc_values,
        max_positions_values,
        single_max_values,
        target_weight_values,
        buy_top_k_values,
        hold_gap_values,
    ):
        hold_top_k = bt + hg
        out.append(
            {
                "capital_alloc_pct": cap,
                "max_positions": mp,
                "single_max": sm,
                "target_weight": tw,
                "buy_top_k": bt,
                "hold_top_k": hold_top_k,
            }
        )
    rng.shuffle(out)
    return out


def generate_fine_candidates(anchors: List[dict], coarse_candidates: List[dict]) -> List[dict]:
    cap_vals = sorted({float(x["capital_alloc_pct"]) for x in coarse_candidates})
    mp_vals = sorted({int(x["max_positions"]) for x in coarse_candidates})
    sm_vals = sorted({float(x["single_max"]) for x in coarse_candidates})
    tw_vals = sorted({float(x["target_weight"]) for x in coarse_candidates})
    bt_vals = sorted({int(x["buy_top_k"]) for x in coarse_candidates})
    hk_vals = sorted({int(x["hold_top_k"]) for x in coarse_candidates})

    idx_map = {
        "capital_alloc_pct": {v: i for i, v in enumerate(cap_vals)},
        "max_positions": {v: i for i, v in enumerate(mp_vals)},
        "single_max": {v: i for i, v in enumerate(sm_vals)},
        "target_weight": {v: i for i, v in enumerate(tw_vals)},
        "buy_top_k": {v: i for i, v in enumerate(bt_vals)},
        "hold_top_k": {v: i for i, v in enumerate(hk_vals)},
    }

    fine = {}
    for anchor in anchors:
        p = anchor["params"]
        cap_i = idx_map["capital_alloc_pct"][float(p["capital_alloc_pct"])]
        mp_i = idx_map["max_positions"][int(p["max_positions"])]
        sm_i = idx_map["single_max"][float(p["single_max"])]
        tw_i = idx_map["target_weight"][float(p["target_weight"])]
        bt_i = idx_map["buy_top_k"][int(p["buy_top_k"])]
        hk_i = idx_map["hold_top_k"][int(p["hold_top_k"])]

        cap_nei = [cap_vals[i] for i in range(max(0, cap_i - 1), min(len(cap_vals), cap_i + 2))]
        mp_nei = [mp_vals[i] for i in range(max(0, mp_i - 1), min(len(mp_vals), mp_i + 2))]
        sm_nei = [sm_vals[i] for i in range(max(0, sm_i - 1), min(len(sm_vals), sm_i + 2))]
        tw_nei = [tw_vals[i] for i in range(max(0, tw_i - 1), min(len(tw_vals), tw_i + 2))]
        bt_nei = [bt_vals[i] for i in range(max(0, bt_i - 1), min(len(bt_vals), bt_i + 2))]
        hk_nei = [hk_vals[i] for i in range(max(0, hk_i - 1), min(len(hk_vals), hk_i + 2))]

        for cap, mp, sm, tw, bt, hk in product(cap_nei, mp_nei, sm_nei, tw_nei, bt_nei, hk_nei):
            if hk < bt:
                continue
            c = {
                "capital_alloc_pct": cap,
                "max_positions": mp,
                "single_max": sm,
                "target_weight": tw,
                "buy_top_k": bt,
                "hold_top_k": hk,
            }
            fine[candidate_key(c)] = c
    return list(fine.values())


def rank_results(results: List[dict]) -> Tuple[dict, dict, int]:
    best_feasible = None
    best_infeasible = None
    feasible_count = 0
    for r in results:
        if bool(r.get("feasible", False)):
            feasible_count += 1
            if better_feasible(r, best_feasible):
                best_feasible = r
        else:
            if better_infeasible(r, best_infeasible):
                best_infeasible = r
    best_overall = best_feasible if best_feasible is not None else best_infeasible
    return best_overall, best_feasible, feasible_count


def top_results(results: List[dict], n: int = 10) -> List[dict]:
    feasible = [r for r in results if bool(r.get("feasible", False))]
    infeasible = [r for r in results if not bool(r.get("feasible", False))]
    feasible_sorted = sorted(
        feasible,
        key=lambda x: (x["annual_return"], x["max_drawdown"], x["sharpe"]),
        reverse=True,
    )
    infeasible_sorted = sorted(
        infeasible,
        key=lambda x: (x["max_drawdown"], x["annual_return"], x["sharpe"]),
        reverse=True,
    )
    mixed = feasible_sorted + infeasible_sorted
    return mixed[:n]


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()

    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[optimize-joint] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    try:
        ctx = build_context(runtime, stock_single, args)
    except Exception as e:
        print(f"[optimize-joint] init error: {e}")
        return EXIT_SIGNAL_ERROR

    print(
        f"[optimize-joint] periods={ctx.loop_end_i - ctx.start_idx + 1}, symbols={len(ctx.symbols)}, "
        f"mdd_limit={ctx.mdd_limit:.2%}"
    )

    rng = random.Random(args.seed)
    coarse_all = generate_coarse_candidates(rng)
    if args.force_capital_alloc is not None:
        force_cap = float(args.force_capital_alloc)
        coarse_all = [c for c in coarse_all if abs(float(c["capital_alloc_pct"]) - force_cap) < 1e-12]
        if not coarse_all:
            print(f"[optimize-joint] no candidates left after force_capital_alloc={force_cap}")
            return EXIT_SIGNAL_ERROR
    coarse_candidates = coarse_all[: max(1, int(args.coarse_evals))]

    evaluated = set()
    results_coarse = []
    for idx, cand in enumerate(coarse_candidates, start=1):
        key = candidate_key(cand)
        if key in evaluated:
            continue
        evaluated.add(key)
        res = evaluate_candidate(ctx, cand)
        results_coarse.append(res)
        if idx % 10 == 0 or idx == len(coarse_candidates):
            print(
                f"[coarse] {idx}/{len(coarse_candidates)} "
                f"annual={res['annual_return']:.4f} mdd={res['max_drawdown']:.4f} "
                f"sharpe={res['sharpe']:.4f} feasible={res['feasible']}"
            )

    best_coarse, best_feasible_coarse, feasible_count_coarse = rank_results(results_coarse)
    anchors = top_results(results_coarse, max(1, int(args.anchors)))
    fine_pool = generate_fine_candidates(anchors, coarse_all)
    if args.force_capital_alloc is not None:
        force_cap = float(args.force_capital_alloc)
        fine_pool = [c for c in fine_pool if abs(float(c["capital_alloc_pct"]) - force_cap) < 1e-12]
    rng.shuffle(fine_pool)

    results_fine = []
    fine_budget = max(0, int(args.fine_evals))
    used_fine = 0
    for cand in fine_pool:
        if used_fine >= fine_budget:
            break
        key = candidate_key(cand)
        if key in evaluated:
            continue
        evaluated.add(key)
        res = evaluate_candidate(ctx, cand)
        results_fine.append(res)
        used_fine += 1
        if used_fine % 10 == 0 or used_fine == fine_budget:
            print(
                f"[fine] {used_fine}/{fine_budget} "
                f"annual={res['annual_return']:.4f} mdd={res['max_drawdown']:.4f} "
                f"sharpe={res['sharpe']:.4f} feasible={res['feasible']}"
            )

    all_results = results_coarse + results_fine
    best_overall, best_feasible, feasible_count = rank_results(all_results)
    if best_overall is None:
        print("[optimize-joint] no valid results")
        return EXIT_SIGNAL_ERROR

    out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, "stock_single_joint_optimization.json")

    report = {
        "ts": datetime.now().isoformat(),
        "objective": {
            "primary": "annual_return_max",
            "constraint": {"max_drawdown_gte": ctx.mdd_limit},
            "monitor": ["sharpe"],
        },
        "data": {
            "symbols": len(ctx.symbols),
            "periods": int(ctx.loop_end_i - ctx.start_idx + 1),
            "start_signal_date": str(ctx.dates[ctx.start_idx - 1].date().isoformat()),
            "end_signal_date": str(ctx.dates[ctx.loop_end_i - 1].date().isoformat()),
            "feature_coverage": ctx.feature_coverage,
        },
        "search": {
            "seed": int(args.seed),
            "coarse_evals": int(len(results_coarse)),
            "fine_evals": int(len(results_fine)),
            "total_evals": int(len(all_results)),
            "coarse_feasible": int(feasible_count_coarse),
            "total_feasible": int(feasible_count),
        },
        "best_overall": best_overall,
        "best_feasible": best_feasible,
        "best_coarse": best_coarse,
        "best_feasible_coarse": best_feasible_coarse,
        "top10": top_results(all_results, 10),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[optimize-joint] done")
    if best_feasible is None:
        print(
            f"[optimize-joint] warning: no feasible solution found under MDD>={ctx.mdd_limit:.2%}; "
            "best_overall is least-infeasible"
        )
    print(f"[optimize-joint] total_evals={len(all_results)}, feasible={feasible_count}")
    print(
        "[optimize-joint] best_overall: "
        f"annual={best_overall['annual_return']:.4f}, "
        f"mdd={best_overall['max_drawdown']:.4f}, "
        f"sharpe={best_overall['sharpe']:.4f}, "
        f"params={best_overall['params']}"
    )
    if best_feasible is not None:
        print(
            "[optimize-joint] best_feasible: "
            f"annual={best_feasible['annual_return']:.4f}, "
            f"mdd={best_feasible['max_drawdown']:.4f}, "
            f"sharpe={best_feasible['sharpe']:.4f}, "
            f"params={best_feasible['params']}"
        )
    print(f"[optimize-joint] report -> {out_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
