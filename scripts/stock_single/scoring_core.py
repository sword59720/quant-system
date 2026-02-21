#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Tuple

import pandas as pd


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
    return pd.DataFrame(series_map).sort_index()


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
    flow_dir = data_cfg.get(
        "fund_flow_factor_output_dir",
        data_cfg.get("fund_flow_output_dir", "./data/stock_single/fund_flow"),
    )
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


def factor_coverage(panel: pd.DataFrame) -> float:
    if panel.empty:
        return 0.0
    return float(panel.notna().sum().sum() / float(panel.shape[0] * panel.shape[1]))


def _safe_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([float("inf"), float("-inf")], float("nan"))


def compute_score_bundle(
    close_df: pd.DataFrame,
    vol_df: pd.DataFrame,
    amt_df: pd.DataFrame,
    data_cfg: dict,
    backtest_cfg: dict,
) -> dict:
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

    def calculate_boll(price: pd.Series, volume: pd.Series, window: int = 20, num_std: int = 2):
        rolling_mean = price.rolling(window=window).mean()
        rolling_std = price.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        boll_percent = (price - lower_band) / (upper_band - lower_band)
        boll_width = (upper_band - lower_band) / rolling_mean
        boll_width_change = boll_width.diff()
        boll_open_up = (boll_width_change > 0).astype(int)
        boll_open_down = (boll_width_change < 0).astype(int)
        upper_break = (price > upper_band).astype(int)
        lower_break = (price < lower_band).astype(int)

        volume_mean = volume.rolling(window=window, min_periods=10).mean()
        volume_change_s = volume / volume_mean - 1.0
        volume_shrink = (volume_change_s < -0.2).astype(int)
        buy_signal = (lower_break & volume_shrink & boll_open_up).astype(int)
        sell_signal = (upper_break & volume_shrink & boll_open_down).astype(int)
        return boll_percent, boll_width, buy_signal, sell_signal

    boll_percent_map = {}
    boll_width_map = {}
    buy_signal_map = {}
    sell_signal_map = {}
    for s in close_df.columns:
        if s not in vol_base.columns:
            continue
        bp, bw, bs, ss = calculate_boll(close_df[s], vol_base[s])
        boll_percent_map[s] = bp
        boll_width_map[s] = bw
        buy_signal_map[s] = bs
        sell_signal_map[s] = ss
    boll_percent = _safe_df(pd.DataFrame(boll_percent_map, index=close_df.index))
    boll_width = _safe_df(pd.DataFrame(boll_width_map, index=close_df.index))
    buy_signal = _safe_df(pd.DataFrame(buy_signal_map, index=close_df.index))
    sell_signal = _safe_df(pd.DataFrame(sell_signal_map, index=close_df.index))

    sentiment_map = {}
    for s in close_df.columns:
        if s not in vol_base.columns:
            continue
        price_change = close_df[s].pct_change(fill_method=None, periods=5)
        volume_change_s = vol_base[s].pct_change(fill_method=None, periods=5)
        sentiment_map[s] = price_change * volume_change_s
    sentiment = _safe_df(pd.DataFrame(sentiment_map, index=close_df.index))

    factor_panels = load_factor_panels(
        index=close_df.index,
        symbols=sorted(close_df.columns),
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

    z_mom20 = cross_section_zscore(_safe_df(mom20))
    z_mom60 = cross_section_zscore(_safe_df(mom60))
    z_rev5 = cross_section_zscore(_safe_df(rev5))
    z_vol20 = cross_section_zscore(_safe_df(vol20))
    z_vol_spike = cross_section_zscore(_safe_df(vol_spike))
    z_ep = cross_section_zscore(_safe_df(ep))
    z_bp = cross_section_zscore(_safe_df(bp))
    z_flow = cross_section_zscore(_safe_df(flow_5d))
    z_vol_change = cross_section_zscore(_safe_df(vol_change))
    z_volume_change = cross_section_zscore(_safe_df(volume_change))
    z_sentiment = cross_section_zscore(_safe_df(sentiment))
    z_boll_percent = cross_section_zscore(_safe_df(boll_percent))
    z_boll_width = cross_section_zscore(_safe_df(boll_width))
    z_buy_signal = cross_section_zscore(_safe_df(buy_signal))
    z_sell_signal = cross_section_zscore(_safe_df(sell_signal))
    improved_boll_signal = z_boll_percent - z_buy_signal * 3.0 + z_sell_signal * 3.0

    score_weights = backtest_cfg.get(
        "score_weights",
        {
            "mom20": 2.00,
            "mom60": 1.50,
            "rev5": -0.30,
            "vol20": -0.25,
            "vol_spike": 0.40,
            "ep": 0.05,
            "bp": 0.05,
            "main_flow": 2.00,
            "vol_change": 0.60,
            "volume_change": 0.60,
            "sentiment": 0.80,
            "boll_percent": -0.60,
            "boll_width": 0.30,
            "boll_break": -1.00,
        },
    )
    score_df = (
        float(score_weights.get("mom20", 2.00)) * z_mom20.fillna(0.0)
        + float(score_weights.get("mom60", 1.50)) * z_mom60.fillna(0.0)
        + float(score_weights.get("rev5", -0.30)) * z_rev5.fillna(0.0)
        + float(score_weights.get("vol20", -0.25)) * z_vol20.fillna(0.0)
        + float(score_weights.get("vol_spike", 0.40)) * z_vol_spike.fillna(0.0)
        + float(score_weights.get("ep", 0.05)) * z_ep.fillna(0.0)
        + float(score_weights.get("bp", 0.05)) * z_bp.fillna(0.0)
        + float(score_weights.get("main_flow", 2.00)) * z_flow.fillna(0.0)
        + float(score_weights.get("vol_change", 0.60)) * z_vol_change.fillna(0.0)
        + float(score_weights.get("volume_change", 0.60)) * z_volume_change.fillna(0.0)
        + float(score_weights.get("sentiment", 0.80)) * z_sentiment.fillna(0.0)
        + float(score_weights.get("boll_percent", -0.60)) * z_boll_percent.fillna(0.0)
        + float(score_weights.get("boll_width", 0.30)) * z_boll_width.fillna(0.0)
        + float(score_weights.get("boll_break", -1.00)) * improved_boll_signal.fillna(0.0)
    )

    coverage_start = pd.Timestamp(str(backtest_cfg.get("factor_coverage_start_date", "2025-08-20")))
    relevant_index = close_df.index[close_df.index >= coverage_start]
    if len(relevant_index) == 0:
        relevant_index = close_df.index

    coverage = {
        "pe_ttm": factor_coverage(pe_ttm.loc[relevant_index]),
        "pb": factor_coverage(pb.loc[relevant_index]),
        "ep": factor_coverage(ep.loc[relevant_index]),
        "bp": factor_coverage(bp.loc[relevant_index]),
        "main_net_inflow": factor_coverage(main_flow.loc[relevant_index]),
        "main_net_inflow_ratio": factor_coverage(main_flow_ratio.loc[relevant_index]),
        "flow_5d": factor_coverage(flow_5d.loc[relevant_index]),
    }
    return {
        "ret_df": ret_df,
        "vol20": vol20,
        "score_df": score_df,
        "coverage": coverage,
        "score_weights": score_weights,
    }


def apply_realtime_adjustments(
    score_row: pd.Series,
    tradable_symbols: set,
    close_df: pd.DataFrame,
    vol20: pd.DataFrame,
    idx_signal: int,
    signal_cfg: dict,
) -> pd.Series:
    out = score_row.copy()
    time_series_momentum_window = int(signal_cfg.get("time_series_momentum_window", 20))
    volatility_adjusted_signal = bool(signal_cfg.get("volatility_adjusted_signal", True))

    if volatility_adjusted_signal:
        for s in tradable_symbols:
            if s in vol20.columns and pd.notna(vol20.iloc[idx_signal].get(s)) and float(vol20.iloc[idx_signal][s]) > 0:
                if pd.notna(out.get(s)):
                    out[s] = float(out[s]) / float(vol20.iloc[idx_signal][s])

    for s in tradable_symbols:
        if s not in close_df.columns or idx_signal < time_series_momentum_window:
            continue
        px_now = close_df.iloc[idx_signal].get(s)
        px_ref = close_df.iloc[idx_signal - time_series_momentum_window].get(s)
        if pd.isna(px_now) or pd.isna(px_ref) or float(px_ref) <= 0:
            continue
        ts_momentum = float(px_now / px_ref - 1.0)
        if ts_momentum < 0 and pd.notna(out.get(s)):
            out[s] = float(out[s]) * 0.5
    return out


def compute_latest_atr14(frames: Dict[str, pd.DataFrame], close_df: pd.DataFrame, idx_signal: int) -> pd.Series:
    out = {}
    for s in close_df.columns:
        df = frames.get(s, pd.DataFrame())
        close_s = pd.to_numeric(close_df[s], errors="coerce")
        if not df.empty and "high" in df.columns and "low" in df.columns:
            high_s = pd.to_numeric(df["high"], errors="coerce").reindex(close_df.index)
            low_s = pd.to_numeric(df["low"], errors="coerce").reindex(close_df.index)
        else:
            high_s = close_s
            low_s = close_s
        prev_close = close_s.shift(1)
        tr = pd.concat(
            [
                (high_s - low_s).abs(),
                (high_s - prev_close).abs(),
                (low_s - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(14, min_periods=5).mean()
        atr_v = atr.iloc[idx_signal] if idx_signal < len(atr) else float("nan")
        if pd.isna(atr_v) or float(atr_v) <= 0:
            rv = close_s.pct_change(fill_method=None).rolling(14, min_periods=5).std()
            rv_v = rv.iloc[idx_signal] if idx_signal < len(rv) else float("nan")
            last = close_s.iloc[idx_signal] if idx_signal < len(close_s) else float("nan")
            if pd.notna(rv_v) and pd.notna(last):
                atr_v = abs(float(rv_v) * float(last))
        if pd.isna(atr_v) or float(atr_v) <= 0:
            last = close_s.iloc[idx_signal] if idx_signal < len(close_s) else float("nan")
            if pd.notna(last) and float(last) > 0:
                atr_v = max(float(last) * 0.02, 0.01)
            else:
                atr_v = 0.01
        out[s] = float(atr_v)
    return pd.Series(out)
