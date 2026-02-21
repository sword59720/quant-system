#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import sys
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
    EXIT_SIGNAL_ERROR,
)
from scripts.stock_single.fetch_stock_single_data import is_a_share_stock_code, load_symbols, normalize_symbol


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def annualized_return(nav: pd.Series, periods_per_year: int = 252) -> float:
    if len(nav) < 2:
        return 0.0
    years = max((len(nav) - 1) / periods_per_year, 1e-9)
    total = float(nav.iloc[-1] / nav.iloc[0])
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    return float((nav / nav.cummax() - 1.0).min())


def sharpe_ratio(ret: pd.Series, periods_per_year: int = 252) -> float:
    if len(ret) < 2:
        return 0.0
    sd = float(ret.std())
    if sd == 0.0:
        return 0.0
    return float((ret.mean() / sd) * math.sqrt(periods_per_year))


def cross_section_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1, skipna=True)
    sd = df.std(axis=1, skipna=True, ddof=0).replace(0.0, float("nan"))
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z


def load_daily_frames(daily_dir: str, symbols: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    frames = {}
    for code, canonical in symbols:
        fp = os.path.join(daily_dir, f"{canonical}.csv")
        if not os.path.exists(fp):
            fp = os.path.join(daily_dir, f"{code}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x = x.dropna(subset=["date"]).sort_values("date").set_index("date")
        frames[canonical] = x
    return frames


def build_panel(frames: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    series_map = {}
    for sym, df in frames.items():
        if column not in df.columns:
            continue
        series_map[sym] = pd.to_numeric(df[column], errors="coerce")
    if not series_map:
        return pd.DataFrame()
    panel = pd.DataFrame(series_map).sort_index()
    return panel


def empty_aligned_panel(index: pd.Index, columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=columns, dtype=float)


def load_factor_panels(
    index: pd.Index,
    symbols: List[str],
    data_cfg: dict,
    value_cols: Dict[str, List[str]],
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    valuation_dir = data_cfg.get("valuation_output_dir", "./data/stock_single/valuation")
    flow_dir = data_cfg.get("fund_flow_output_dir", "./data/stock_single/fund_flow")
    dir_map = {
        "pe_ttm": valuation_dir,
        "pb": valuation_dir,
        "main_net_inflow": flow_dir,
        "main_net_inflow_ratio": flow_dir,
    }
    for factor_name in value_cols.keys():
        out[factor_name] = empty_aligned_panel(index=index, columns=symbols)

    for symbol in symbols:
        for factor_name, candidates in value_cols.items():
            base_dir = dir_map.get(factor_name, "")
            if not base_dir:
                continue
            fp = os.path.join(base_dir, f"{symbol}.csv")
            if not os.path.exists(fp):
                continue
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            if "date" not in df.columns:
                continue
            x = df.copy()
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["date"]).sort_values("date")
            src_col = None
            for c in candidates:
                if c in x.columns:
                    src_col = c
                    break
            if src_col is None:
                continue
            s = pd.to_numeric(x[src_col], errors="coerce").set_axis(x["date"])
            out[factor_name].loc[:, symbol] = s.reindex(index).values
    return out


def compute_risk_overlay(
    idx_signal: int,
    symbols: List[str],
    weights: Dict[str, float],
    close_df: pd.DataFrame,
    ret_df: pd.DataFrame,
    fast_cfg: dict,
) -> dict:
    if idx_signal <= 0:
        return {
            "stage": "normal",
            "triggered": False,
            "block_new_buys": False,
            "force_sell_symbols": [],
            "portfolio_ret_1d": 0.0,
            "single_crash_count": 0,
        }

    trigger_portfolio_1d = float(
        fast_cfg.get(
            "trigger_portfolio_ret_1d",
            float(fast_cfg.get("trigger_portfolio_ret_5m", -0.008)) * math.sqrt(48.0),
        )
    )
    trigger_single_1d = float(
        fast_cfg.get(
            "trigger_single_ret_1d",
            float(fast_cfg.get("trigger_single_ret_5m", -0.020)) * math.sqrt(48.0),
        )
    )
    trigger_vol_z = float(fast_cfg.get("trigger_vol_zscore", 3.0))

    dt_signal = close_df.index[idx_signal]
    dt_prev = close_df.index[idx_signal - 1]
    ret_row = (close_df.loc[dt_signal] / close_df.loc[dt_prev] - 1.0).fillna(0.0)

    portfolio_ret_1d = 0.0
    for s, w in weights.items():
        portfolio_ret_1d += float(w) * float(ret_row.get(s, 0.0))

    single_crash_symbols = sorted([s for s in symbols if float(ret_row.get(s, 0.0)) <= trigger_single_1d])

    ret_abs = ret_df.abs()
    roll_mu = ret_abs.rolling(20, min_periods=10).mean()
    roll_sd = ret_abs.rolling(20, min_periods=10).std()
    z_row = ((ret_abs - roll_mu) / roll_sd).iloc[idx_signal].fillna(0.0)
    vol_spike_symbols = sorted(
        [
            s
            for s in symbols
            if float(z_row.get(s, 0.0)) >= trigger_vol_z and float(ret_row.get(s, 0.0)) < 0.0
        ]
    )

    portfolio_trigger = bool(portfolio_ret_1d <= trigger_portfolio_1d)
    single_trigger = bool(len(single_crash_symbols) > 0)
    vol_trigger = bool(len(vol_spike_symbols) > 0)
    triggered = bool(portfolio_trigger or single_trigger or vol_trigger)

    force_sell = sorted(set(single_crash_symbols + vol_spike_symbols))
    block_new_buys = bool(triggered and fast_cfg.get("block_new_buys_on_trigger", True))
    if not bool(fast_cfg.get("force_sell_on_single_crash", True)):
        force_sell = []

    return {
        "stage": "risk" if triggered else "normal",
        "triggered": triggered,
        "block_new_buys": block_new_buys,
        "force_sell_symbols": force_sell,
        "portfolio_ret_1d": float(portfolio_ret_1d),
        "single_crash_count": int(len(single_crash_symbols)),
        "vol_spike_count": int(len(vol_spike_symbols)),
        "trigger_flags": {
            "portfolio_drop": portfolio_trigger,
            "single_crash": single_trigger,
            "vol_spike": vol_trigger,
        },
        "thresholds_1d": {
            "trigger_portfolio_ret_1d": trigger_portfolio_1d,
            "trigger_single_ret_1d": trigger_single_1d,
            "trigger_vol_zscore": trigger_vol_z,
        },
    }


def build_target_weights(
    symbols: List[str],
    current_weights: Dict[str, float],
    score_row: pd.Series,
    tradable_symbols: set,
    risk_overlay: dict,
    capital_alloc_pct: float,
    max_positions: int,
    target_weight: float,
    single_max: float,
    buy_threshold: float,
    sell_threshold: float,
) -> Tuple[Dict[str, float], dict]:
    weights = dict(current_weights)
    target = {s: 0.0 for s in symbols}

    locked_symbols = sorted([s for s in symbols if weights.get(s, 0.0) > 1e-8 and s not in tradable_symbols])
    for s in locked_symbols:
        target[s] = float(weights.get(s, 0.0))
    locked_weight = float(sum(target[s] for s in locked_symbols))

    force_sell = set([str(s) for s in risk_overlay.get("force_sell_symbols", [])])
    block_new_buys = bool(risk_overlay.get("block_new_buys", False))

    held_tradable = [s for s in symbols if weights.get(s, 0.0) > 1e-8 and s in tradable_symbols]
    keep_symbols = []
    sell_symbols = []
    for s in held_tradable:
        score = float(score_row.get(s, float("nan")))
        if s in force_sell:
            sell_symbols.append(s)
            continue
        if pd.notna(score) and score <= sell_threshold:
            sell_symbols.append(s)
            continue
        keep_symbols.append(s)

    slot_limit = max(0, int(max_positions) - len(locked_symbols))
    if slot_limit <= 0:
        keep_symbols = []
    elif len(keep_symbols) > slot_limit:
        keep_symbols = sorted(
            keep_symbols,
            key=lambda s: float(score_row.get(s, -999.0)),
            reverse=True,
        )[:slot_limit]

    buy_candidates = []
    if not block_new_buys and slot_limit > len(keep_symbols):
        for s in sorted(tradable_symbols, key=lambda x: float(score_row.get(x, -999.0)), reverse=True):
            if s in keep_symbols:
                continue
            score = float(score_row.get(s, float("nan")))
            if pd.isna(score) or score < buy_threshold:
                continue
            buy_candidates.append(s)
        buy_candidates = buy_candidates[: max(0, slot_limit - len(keep_symbols))]

    final_symbols = keep_symbols + buy_candidates

    available_alloc = max(0.0, float(capital_alloc_pct) - locked_weight)
    if final_symbols and available_alloc > 1e-9:
        per_symbol = min(float(target_weight), float(single_max))
        if per_symbol > 1e-9:
            base_total = per_symbol * len(final_symbols)
            if base_total > available_alloc:
                per_symbol = available_alloc / len(final_symbols)

            for s in final_symbols:
                target[s] = per_symbol

    meta = {
        "locked_symbols": locked_symbols,
        "keep_symbols": keep_symbols,
        "buy_candidates": buy_candidates,
        "sell_symbols": sorted(sell_symbols),
        "block_new_buys": block_new_buys,
    }
    return target, meta

def resolve_signal_thresholds(
    score_row: pd.Series,
    tradable_symbols: set,
    signal_cfg: dict,
    max_positions: int,
    buy_threshold_static: float,
    sell_threshold_static: float,
) -> Tuple[float, float, dict]:
    mode = str(signal_cfg.get("trigger_mode", "threshold")).strip().lower()
    if mode not in {"threshold", "quantile", "topk"}:
        mode = "threshold"

    if tradable_symbols:
        valid = score_row.reindex(sorted(tradable_symbols)).dropna().astype(float)
    else:
        valid = score_row.dropna().astype(float)
    if valid.empty:
        return buy_threshold_static, sell_threshold_static, {
            "mode": "threshold",
            "fallback": "no_valid_scores",
            "buy_threshold": float(buy_threshold_static),
            "sell_threshold": float(sell_threshold_static),
        }

    if mode == "quantile":
        q_buy = min(float(signal_cfg.get("buy_score_quantile", 0.75)), 1.0)
        q_sell = min(float(signal_cfg.get("sell_score_quantile", 0.35)), 1.0)
        if q_sell >= q_buy:
            q_sell = max(0.0, q_buy - 0.10)
        buy_threshold = float(valid.quantile(q_buy))
        sell_threshold = float(valid.quantile(q_sell))
        if sell_threshold >= buy_threshold:
            sell_threshold = buy_threshold - 1e-6
        return buy_threshold, sell_threshold, {
            "mode": "quantile",
            "buy_score_quantile": q_buy,
            "sell_score_quantile": q_sell,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }

    if mode == "topk":
        ranked = valid.sort_values(ascending=False).reset_index(drop=True)
        buy_top_k = int(signal_cfg.get("buy_top_k", max_positions))
        buy_top_k = max(1, min(buy_top_k, len(ranked)))
        hold_top_k = int(signal_cfg.get("hold_top_k", max(buy_top_k + 1, buy_top_k * 2)))
        hold_top_k = max(buy_top_k, min(hold_top_k, len(ranked)))

        buy_threshold = float(ranked.iloc[buy_top_k - 1])
        sell_threshold = float(ranked.iloc[hold_top_k - 1])
        if sell_threshold >= buy_threshold:
            sell_threshold = buy_threshold - 1e-6
        return buy_threshold, sell_threshold, {
            "mode": "topk",
            "buy_top_k": buy_top_k,
            "hold_top_k": hold_top_k,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }

    return buy_threshold_static, sell_threshold_static, {
        "mode": "threshold",
        "buy_threshold": float(buy_threshold_static),
        "sell_threshold": float(sell_threshold_static),
    }

def backtest_with_weights(runtime, stock_single, score_weights):
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

    max_positions = int(stock_single.get("max_positions", 10))
    capital_alloc_pct = float(stock_single.get("capital_alloc_pct", 0.30))
    signal_cfg = stock_single.get("signal", {})
    risk_cfg = stock_single.get("risk", {})
    fast_cfg = risk_cfg.get("fast_check", {})
    backtest_cfg = stock_single.get("backtest", {})

    buy_threshold = float(signal_cfg.get("buy_threshold", 1.0))
    sell_threshold = float(signal_cfg.get("sell_threshold", -0.5))
    target_weight = float(signal_cfg.get("per_signal_target_weight", 0.08))
    single_max = float(risk_cfg.get("single_max_pct", 0.12))

    fee_buy_bps = float(backtest_cfg.get("fee_buy_bps", 1.5))
    fee_sell_bps = float(backtest_cfg.get("fee_sell_bps", 1.5))
    stamp_tax_sell_bps = float(backtest_cfg.get("stamp_tax_sell_bps", 5.0))
    slippage_bps = float(backtest_cfg.get("slippage_bps", 2.5))

    buy_cost_rate = (fee_buy_bps + slippage_bps) * 1e-4
    sell_cost_rate = (fee_sell_bps + stamp_tax_sell_bps + slippage_bps) * 1e-4

    ret_df = close_df.pct_change(fill_method=None)
    mom20 = close_df / close_df.shift(20) - 1.0
    mom60 = close_df / close_df.shift(60) - 1.0
    rev5 = close_df / close_df.shift(5) - 1.0
    vol20 = ret_df.rolling(20, min_periods=10).std()
    vol_base = vol_df if not vol_df.empty else amt_df
    if vol_base.empty:
        vol_base = close_df.copy()
    vol_spike = vol_base / vol_base.rolling(20, min_periods=10).mean() - 1.0

    vol_change = vol20 / vol20.shift(10) - 1.0
    volume_change = vol_base / vol_base.shift(10) - 1.0

    sentiment = {}
    for s in close_df.columns:
        if s in close_df.columns and s in vol_base.columns:
            price_change = close_df[s].pct_change(5)
            volume_change_s = vol_base[s].pct_change(5)
            sentiment[s] = price_change * volume_change_s
    sentiment = pd.DataFrame(sentiment, index=close_df.index)

    factor_panels = load_factor_panels(
        index=close_df.index,
        symbols=symbols,
        data_cfg=data_cfg,
        value_cols={
            "pe_ttm": ["pe_ttm"],
            "pb": ["pb"],
            "main_net_inflow": ["main_net_inflow"],
            "main_net_inflow_ratio": ["main_net_inflow_ratio"],
        },
    )
    pe_ttm = factor_panels["pe_ttm"]
    pb = factor_panels["pb"]
    main_flow = factor_panels["main_net_inflow"]
    main_flow_ratio = factor_panels["main_net_inflow_ratio"]

    ep = 1.0 / pe_ttm.where(pe_ttm > 0)
    bp = 1.0 / pb.where(pb > 0)
    flow_ratio = main_flow_ratio.copy()
    if flow_ratio.empty or flow_ratio.notna().sum().sum() == 0:
        amt_safe = amt_df.where(amt_df > 0)
        flow_ratio = main_flow / amt_safe
    flow_5d = flow_ratio.rolling(5, min_periods=3).sum()

    z_mom20 = cross_section_zscore(mom20)
    z_mom60 = cross_section_zscore(mom60)
    z_rev5 = cross_section_zscore(rev5)
    z_vol20 = cross_section_zscore(vol20)
    z_vol_spike = cross_section_zscore(vol_spike)
    z_ep = cross_section_zscore(ep)
    z_bp = cross_section_zscore(bp)
    z_flow = cross_section_zscore(flow_5d)
    z_vol_change = cross_section_zscore(vol_change)
    z_volume_change = cross_section_zscore(volume_change)
    z_sentiment = cross_section_zscore(sentiment)

    score_df = (
        float(score_weights.get("mom20", 0.35)) * z_mom20.fillna(0.0)
        + float(score_weights.get("mom60", 0.30)) * z_mom60.fillna(0.0)
        + float(score_weights.get("rev5", -0.12)) * z_rev5.fillna(0.0)
        + float(score_weights.get("vol20", -0.10)) * z_vol20.fillna(0.0)
        + float(score_weights.get("vol_spike", 0.08)) * z_vol_spike.fillna(0.0)
        + float(score_weights.get("ep", 0.12)) * z_ep.fillna(0.0)
        + float(score_weights.get("bp", 0.08)) * z_bp.fillna(0.0)
        + float(score_weights.get("main_flow", 0.25)) * z_flow.fillna(0.0)
        + float(score_weights.get("vol_change", 0.18)) * z_vol_change.fillna(0.0)
        + float(score_weights.get("volume_change", 0.18)) * z_volume_change.fillna(0.0)
        + float(score_weights.get("sentiment", 0.15)) * z_sentiment.fillna(0.0)
    )

    warmup = int(max(backtest_cfg.get("warmup_days", 80), 61))
    dates = close_df.index
    if len(dates) < warmup + 3:
        raise RuntimeError("not enough history for backtest")

    start_idx = warmup + 1
    end_price_idx = len(dates) - 1
    loop_end_i = min(len(dates) - 2, end_price_idx - 1)
    if loop_end_i < start_idx:
        raise RuntimeError("invalid date range after warmup")

    weights = {s: 0.0 for s in symbols}
    strategy_nav = 1.0
    strategy_rets = []

    for i in range(start_idx, loop_end_i + 1):
        idx_signal = i - 1
        signal_date = dates[idx_signal]
        trade_date = dates[i]
        next_date = dates[i + 1]

        tradable_symbols = set(
            [
                s
                for s in symbols
                if pd.notna(close_df.at[trade_date, s]) and pd.notna(close_df.at[next_date, s])
            ]
        )

        risk_overlay = compute_risk_overlay(
            idx_signal=idx_signal,
            symbols=symbols,
            weights=weights,
            close_df=close_df,
            ret_df=ret_df,
            fast_cfg=fast_cfg,
        )
        score_row = score_df.iloc[idx_signal]

        time_series_momentum_window = int(signal_cfg.get("time_series_momentum_window", 20))
        volatility_adjusted_signal = bool(signal_cfg.get("volatility_adjusted_signal", True))

        ts_momentum = {}
        for s in tradable_symbols:
            if s in close_df.columns:
                if idx_signal >= time_series_momentum_window:
                    ts_momentum[s] = float(close_df.iloc[idx_signal][s] / close_df.iloc[idx_signal - time_series_momentum_window][s] - 1.0)
                else:
                    ts_momentum[s] = 0.0
            else:
                ts_momentum[s] = 0.0

        if volatility_adjusted_signal:
            for s in tradable_symbols:
                if s in vol20.columns and pd.notna(vol20.iloc[idx_signal][s]) and vol20.iloc[idx_signal][s] > 0:
                    if pd.notna(score_row[s]):
                        score_row[s] = score_row[s] / vol20.iloc[idx_signal][s]

        for s in tradable_symbols:
            if ts_momentum.get(s, 0.0) < 0:
                if pd.notna(score_row[s]):
                    score_row[s] *= 0.5

        eff_buy_threshold, eff_sell_threshold, threshold_meta = resolve_signal_thresholds(
            score_row=score_row,
            tradable_symbols=tradable_symbols,
            signal_cfg=signal_cfg,
            max_positions=max_positions,
            buy_threshold_static=buy_threshold,
            sell_threshold_static=sell_threshold,
        )

        target, target_meta = build_target_weights(
            symbols=symbols,
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
        )

        buy_turnover = 0.0
        sell_turnover = 0.0
        for s in symbols:
            delta = float(target.get(s, 0.0) - weights.get(s, 0.0))
            if delta > 0:
                buy_turnover += delta
            elif delta < 0:
                sell_turnover += -delta

        trade_cost = buy_turnover * buy_cost_rate + sell_turnover * sell_cost_rate

        weights = target

        port_ret = 0.0
        for s, w in weights.items():
            if w <= 0.0:
                continue
            px0 = close_df.at[trade_date, s] if s in close_df.columns else pd.NA
            px1 = close_df.at[next_date, s] if s in close_df.columns else pd.NA
            if pd.isna(px0) or pd.isna(px1) or float(px0) <= 0:
                continue
            port_ret += float(w) * (float(px1) / float(px0) - 1.0)
        port_ret -= trade_cost

        strategy_nav *= 1.0 + port_ret
        strategy_rets.append(float(port_ret))

    if len(strategy_rets) < 2:
        raise RuntimeError("backtest periods too short after filtering")

    nav_s = pd.Series([1.0] + [strategy_nav] * len(strategy_rets))
    ret_s = pd.Series(strategy_rets)

    annual_return = annualized_return(nav_s, 252)
    sharpe = sharpe_ratio(ret_s, 252)
    max_dd = max_drawdown(nav_s)

    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "score_weights": score_weights
    }

def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[optimize-factor-weights] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    # 定义因子权重搜索空间
    mom20_range = [0.3, 0.35, 0.4, 0.45]
    mom60_range = [0.25, 0.3, 0.35, 0.4]
    rev5_range = [-0.15, -0.12, -0.1, -0.08]
    vol20_range = [-0.12, -0.1, -0.08, -0.06]
    vol_spike_range = [0.05, 0.08, 0.1, 0.12]
    ep_range = [0.1, 0.12, 0.15, 0.18]
    bp_range = [0.06, 0.08, 0.1, 0.12]
    main_flow_range = [0.2, 0.25, 0.3, 0.35]
    vol_change_range = [0.15, 0.18, 0.2, 0.22]
    volume_change_range = [0.15, 0.18, 0.2, 0.22]
    sentiment_range = [0.1, 0.15, 0.2, 0.25]

    best_result = {
        "annual_return": -float('inf'),
        "sharpe": -float('inf'),
        "max_drawdown": float('inf'),
        "score_weights": {}
    }

    results = []

    # 生成所有可能的权重组合
    # 为了减少计算量，我们只选择部分组合进行测试
    for mom20, mom60, main_flow, sentiment in product(mom20_range, mom60_range, main_flow_range, sentiment_range):
        score_weights = {
            "mom20": mom20,
            "mom60": mom60,
            "rev5": -0.12,
            "vol20": -0.1,
            "vol_spike": 0.08,
            "ep": 0.12,
            "bp": 0.08,
            "main_flow": main_flow,
            "vol_change": 0.18,
            "volume_change": 0.18,
            "sentiment": sentiment
        }

        try:
            result = backtest_with_weights(runtime, stock_single, score_weights)
            results.append(result)
            print(f"Testing weights: {score_weights}")
            print(f"Result: Annual Return={result['annual_return']:.4f}, Sharpe={result['sharpe']:.4f}, Max DD={result['max_drawdown']:.4f}")

            # 更新最佳结果
            if result['annual_return'] > best_result['annual_return'] and result['sharpe'] > best_result['sharpe']:
                best_result = result
        except Exception as e:
            print(f"Error testing weights {score_weights}: {e}")
            continue

    # 输出结果
    out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(out_dir)
    result_file = os.path.join(out_dir, "factor_weights_optimization.json")

    output = {
        "ts": datetime.now().isoformat(),
        "best_result": best_result,
        "all_results": results
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nBest result:")
    print(f"Annual Return: {best_result['annual_return']:.4f}")
    print(f"Sharpe Ratio: {best_result['sharpe']:.4f}")
    print(f"Max Drawdown: {best_result['max_drawdown']:.4f}")
    print(f"Optimal Weights: {best_result['score_weights']}")
    print(f"Results saved to: {result_file}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
