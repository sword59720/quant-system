#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Optional

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
from core.execution_guard import resolve_min_turnover
from core.exposure_gate import evaluate_exposure_gate


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
        raise RuntimeError(f"missing stock data files: {missing}")

    dates = sorted(set.intersection(*[set(v.index) for v in frames.values()]))
    if len(dates) < 400:
        raise RuntimeError("not enough common stock history (<400 rows)")

    return pd.DataFrame({s: frames[s].loc[dates, "close"] for s in symbols}, index=dates).sort_index()


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


def sharpe_ratio(ret: pd.Series, periods_per_year: int = 252) -> float:
    if len(ret) < 2:
        return 0.0
    sd = ret.std()
    if sd == 0:
        return 0.0
    return float((ret.mean() / sd) * math.sqrt(periods_per_year))


def weights_to_json(weights: dict) -> str:
    clean = {k: round(float(v), 6) for k, v in weights.items() if float(v) > 1e-8}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def calc_turnover(old_w: dict, new_w: dict, symbols: list) -> float:
    return float(sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in symbols))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_risk_alloc_multiplier(
    prices: pd.DataFrame,
    i: int,
    benchmark_symbol: str,
    params: dict,
) -> tuple[float, dict]:
    cfg = params.get("risk_governor", {}) if isinstance(params, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    meta = {
        "enabled": enabled,
        "alloc_mult": 1.0,
        "vol_mult": 1.0,
        "dd_cap_applied": False,
        "momentum_cap_applied": False,
        "benchmark_vol": None,
        "benchmark_dd": None,
        "benchmark_momentum": None,
    }
    if not enabled:
        return 1.0, meta

    close = prices[benchmark_symbol]
    vol_window = int(cfg.get("vol_window", 20))
    target_daily_vol = float(cfg.get("target_daily_vol", 0.012))
    min_alloc_mult = float(cfg.get("min_alloc_mult", 0.30))
    max_alloc_mult = float(cfg.get("max_alloc_mult", 1.00))
    dd_window = int(cfg.get("dd_window", 120))
    dd_trigger = float(cfg.get("dd_trigger", -0.08))
    dd_alloc_mult = float(cfg.get("dd_alloc_mult", 0.45))
    momentum_window = int(cfg.get("momentum_window", 20))
    momentum_trigger = float(cfg.get("momentum_trigger", -0.03))
    momentum_alloc_mult = float(cfg.get("momentum_alloc_mult", 0.60))

    ret = close.pct_change()
    r0 = max(1, i - vol_window + 1)
    vol_slice = ret.iloc[r0 : i + 1].dropna()
    vol_mult = 1.0
    if len(vol_slice) >= 2:
        bench_vol = float(vol_slice.std())
        meta["benchmark_vol"] = bench_vol
        if bench_vol > 1e-8:
            vol_mult = target_daily_vol / bench_vol
    vol_mult = _clamp(float(vol_mult), min_alloc_mult, max_alloc_mult)

    dd_start = max(0, i - dd_window + 1)
    dd_window_close = close.iloc[dd_start : i + 1]
    if len(dd_window_close) > 0:
        peak = float(dd_window_close.max())
        dd_now = float(close.iloc[i] / max(peak, 1e-8) - 1.0)
        meta["benchmark_dd"] = dd_now
    else:
        dd_now = 0.0

    mom_now = None
    if i >= momentum_window:
        mom_now = float(close.iloc[i] / close.iloc[i - momentum_window] - 1.0)
        meta["benchmark_momentum"] = mom_now

    alloc_mult = vol_mult
    if dd_now <= dd_trigger:
        alloc_mult = min(alloc_mult, dd_alloc_mult)
        meta["dd_cap_applied"] = True
    if (mom_now is not None) and (mom_now <= momentum_trigger):
        alloc_mult = min(alloc_mult, momentum_alloc_mult)
        meta["momentum_cap_applied"] = True

    alloc_mult = _clamp(float(alloc_mult), min_alloc_mult, max_alloc_mult)
    meta["alloc_mult"] = alloc_mult
    meta["vol_mult"] = vol_mult
    return alloc_mult, meta


def build_risk_on_weights(
    scores: dict,
    picks: list,
    score_mix: float,
    score_floor: float,
    score_power: float,
) -> dict:
    if not picks:
        return {}
    mix = max(0.0, min(1.0, float(score_mix)))
    power = max(0.1, float(score_power))

    inv_vol = {s: 1.0 / max(float(scores[s]["vol"]), 1e-6) for s in picks}
    inv_total = sum(inv_vol.values())
    ivol_weights = {s: (inv_vol[s] / inv_total if inv_total > 0 else 1.0 / len(picks)) for s in picks}
    if mix <= 1e-12:
        return ivol_weights

    score_raw = {}
    for s in picks:
        val = max(float(scores[s]["score"]) - float(score_floor), 0.0)
        score_raw[s] = val**power
    score_total = sum(score_raw.values())
    if score_total <= 1e-12:
        score_weights = {s: 1.0 / len(picks) for s in picks}
    else:
        score_weights = {s: score_raw[s] / score_total for s in picks}

    mixed = {s: (1.0 - mix) * ivol_weights[s] + mix * score_weights[s] for s in picks}
    mixed_total = sum(mixed.values())
    if mixed_total <= 1e-12:
        return ivol_weights
    return {s: mixed[s] / mixed_total for s in picks}


def _normalize_float_weights(weights: list, n: int) -> list[float]:
    if n <= 0:
        return []
    out = []
    for w in (weights or []):
        try:
            out.append(float(w))
        except (TypeError, ValueError):
            continue
    if len(out) != n:
        return [1.0 / n] * n
    total = sum(max(x, 0.0) for x in out)
    if total <= 1e-12:
        return [1.0 / n] * n
    return [max(x, 0.0) / total for x in out]


def _resolve_structural_cfg(params: dict) -> dict:
    cfg = params.get("structural_upgrade", {}) if isinstance(params, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    windows = []
    for x in cfg.get("momentum_windows", [20, 60, 180]):
        try:
            w = int(x)
        except (TypeError, ValueError):
            continue
        if w > 1:
            windows.append(w)
    if not windows:
        windows = [max(2, int(params.get("momentum_lb", 252)))]
    weights = _normalize_float_weights(cfg.get("momentum_weights", [0.5, 0.3, 0.2]), len(windows))

    breadth_on_threshold = float(cfg.get("breadth_on_threshold", 0.60))
    breadth_off_threshold = float(cfg.get("breadth_off_threshold", 0.35))
    if breadth_off_threshold > breadth_on_threshold:
        breadth_off_threshold = breadth_on_threshold

    trend_mult_min = float(cfg.get("trend_alloc_mult_min", 0.75))
    trend_mult_max = float(cfg.get("trend_alloc_mult_max", 1.35))
    if trend_mult_max < trend_mult_min:
        trend_mult_max = trend_mult_min

    breadth_mult_min = float(cfg.get("breadth_alloc_mult_min", 0.70))
    breadth_mult_max = float(cfg.get("breadth_alloc_mult_max", 1.25))
    if breadth_mult_max < breadth_mult_min:
        breadth_mult_max = breadth_mult_min

    total_mult_min = float(cfg.get("total_alloc_mult_min", 0.70))
    total_mult_max = float(cfg.get("total_alloc_mult_max", 1.60))
    if total_mult_max < total_mult_min:
        total_mult_max = total_mult_min

    phase2_total_mult_min = float(cfg.get("phase2_total_mult_min", 0.80))
    phase2_total_mult_max = float(cfg.get("phase2_total_mult_max", 1.20))
    if phase2_total_mult_max < phase2_total_mult_min:
        phase2_total_mult_max = phase2_total_mult_min

    signal_strength_min_alloc_mult = float(cfg.get("signal_strength_min_alloc_mult", 0.70))
    signal_strength_max_alloc_mult = float(cfg.get("signal_strength_max_alloc_mult", 1.10))
    if signal_strength_max_alloc_mult < signal_strength_min_alloc_mult:
        signal_strength_max_alloc_mult = signal_strength_min_alloc_mult

    adaptive_min_top_n = max(1, int(cfg.get("phase2_adaptive_min_top_n", 1)))
    adaptive_max_top_n = max(adaptive_min_top_n, int(cfg.get("phase2_adaptive_max_top_n", int(params.get("top_n", 1)))))

    phase2_stress_regime_stepdown = int(cfg.get("phase2_stress_regime_stepdown", 0))
    phase2_stress_regime_stepdown = max(0, min(2, phase2_stress_regime_stepdown))

    return {
        "enabled": enabled,
        "momentum_windows": windows,
        "momentum_weights": weights,
        "eligibility_require_trend": bool(cfg.get("eligibility_require_trend", False)),
        "score_blend": _clamp(float(cfg.get("score_blend", 0.35)), 0.0, 1.0),
        "relative_momentum_window": max(2, int(cfg.get("relative_momentum_window", 60))),
        "relative_momentum_weight": float(cfg.get("relative_momentum_weight", 0.20)),
        "asset_trend_ma_window": max(2, int(cfg.get("asset_trend_ma_window", 80))),
        "breadth_on_threshold": breadth_on_threshold,
        "breadth_off_threshold": breadth_off_threshold,
        "neutral_alloc_mult": max(0.0, float(cfg.get("neutral_alloc_mult", 0.65))),
        "neutral_top_n": max(1, int(cfg.get("neutral_top_n", max(1, int(params.get("top_n", 1)) - 1)))),
        "trend_strength_k": float(cfg.get("trend_strength_k", 8.0)),
        "trend_alloc_mult_min": trend_mult_min,
        "trend_alloc_mult_max": trend_mult_max,
        "breadth_alloc_mult_min": breadth_mult_min,
        "breadth_alloc_mult_max": breadth_mult_max,
        "total_alloc_mult_min": total_mult_min,
        "total_alloc_mult_max": total_mult_max,
        "phase2_enabled": bool(cfg.get("phase2_enabled", False)),
        "phase2_vol_short_window": max(2, int(cfg.get("phase2_vol_short_window", 15))),
        "phase2_vol_long_window": max(5, int(cfg.get("phase2_vol_long_window", 80))),
        "phase2_vol_calm_ratio": float(cfg.get("phase2_vol_calm_ratio", 0.85)),
        "phase2_vol_stress_ratio": float(cfg.get("phase2_vol_stress_ratio", 1.20)),
        "phase2_vol_mult_calm": float(cfg.get("phase2_vol_mult_calm", 1.08)),
        "phase2_vol_mult_normal": float(cfg.get("phase2_vol_mult_normal", 1.00)),
        "phase2_vol_mult_stress": float(cfg.get("phase2_vol_mult_stress", 0.78)),
        "phase2_stress_regime_stepdown": phase2_stress_regime_stepdown,
        "phase2_recovery_enabled": bool(cfg.get("phase2_recovery_enabled", True)),
        "phase2_recovery_window": max(20, int(cfg.get("phase2_recovery_window", 180))),
        "phase2_recovery_trigger_drawdown": float(cfg.get("phase2_recovery_trigger_drawdown", -0.08)),
        "phase2_recovery_momentum_window": max(2, int(cfg.get("phase2_recovery_momentum_window", 20))),
        "phase2_recovery_max_mult": float(cfg.get("phase2_recovery_max_mult", 1.20)),
        "phase2_total_mult_min": phase2_total_mult_min,
        "phase2_total_mult_max": phase2_total_mult_max,
        "phase2_adaptive_enabled": bool(cfg.get("phase2_adaptive_enabled", True)),
        "phase2_adaptive_min_top_n": adaptive_min_top_n,
        "phase2_adaptive_max_top_n": adaptive_max_top_n,
        "phase2_adaptive_calm_top_n_delta": int(cfg.get("phase2_adaptive_calm_top_n_delta", -1)),
        "phase2_adaptive_stress_top_n_delta": int(cfg.get("phase2_adaptive_stress_top_n_delta", 1)),
        "phase2_adaptive_high_dispersion_threshold": float(
            cfg.get("phase2_adaptive_high_dispersion_threshold", 0.22)
        ),
        "phase2_adaptive_high_dispersion_top_n_delta": int(cfg.get("phase2_adaptive_high_dispersion_top_n_delta", -1)),
        "phase2_adaptive_score_power_calm_mult": float(cfg.get("phase2_adaptive_score_power_calm_mult", 1.20)),
        "phase2_adaptive_score_power_stress_mult": float(cfg.get("phase2_adaptive_score_power_stress_mult", 0.85)),
        "signal_strength_enabled": bool(cfg.get("signal_strength_enabled", False)),
        "signal_strength_floor": float(cfg.get("signal_strength_floor", 0.0)),
        "signal_strength_ref": max(1e-6, float(cfg.get("signal_strength_ref", 0.06))),
        "signal_strength_curve": max(0.25, float(cfg.get("signal_strength_curve", 1.0))),
        "signal_strength_min_alloc_mult": signal_strength_min_alloc_mult,
        "signal_strength_max_alloc_mult": signal_strength_max_alloc_mult,
    }


def _weighted_momentum(series: pd.Series, i: int, windows: list[int], weights: list[float]) -> Optional[float]:
    if len(windows) != len(weights) or not windows:
        return None
    vals = []
    for w in windows:
        if i < w:
            return None
        prev = float(series.iloc[i - w])
        if abs(prev) <= 1e-12:
            return None
        vals.append(float(series.iloc[i] / prev - 1.0))
    return float(sum(weights[k] * vals[k] for k in range(len(vals))))


def _compute_vol_state_multiplier(
    benchmark_ret: pd.Series,
    i: int,
    cfg: dict,
) -> tuple[str, float, Optional[float]]:
    sw = int(cfg.get("phase2_vol_short_window", 15))
    lw = max(sw + 1, int(cfg.get("phase2_vol_long_window", 80)))
    if i < lw:
        return "normal", float(cfg.get("phase2_vol_mult_normal", 1.0)), None

    short_vol = float(benchmark_ret.iloc[i - sw + 1 : i + 1].std())
    long_vol = float(benchmark_ret.iloc[i - lw + 1 : i + 1].std())
    if long_vol <= 1e-8:
        return "normal", float(cfg.get("phase2_vol_mult_normal", 1.0)), None

    ratio = float(short_vol / long_vol)
    calm_r = float(cfg.get("phase2_vol_calm_ratio", 0.85))
    stress_r = float(cfg.get("phase2_vol_stress_ratio", 1.20))
    if ratio <= calm_r:
        return "calm", float(cfg.get("phase2_vol_mult_calm", 1.08)), ratio
    if ratio >= stress_r:
        return "stress", float(cfg.get("phase2_vol_mult_stress", 0.78)), ratio
    return "normal", float(cfg.get("phase2_vol_mult_normal", 1.0)), ratio


def _compute_drawdown_recovery_multiplier(
    benchmark_close: pd.Series,
    i: int,
    cfg: dict,
) -> tuple[float, dict]:
    meta = {
        "enabled": bool(cfg.get("phase2_recovery_enabled", True)),
        "drawdown": None,
        "drawdown_depth": None,
        "recovery_progress": None,
        "momentum": None,
    }
    if not meta["enabled"]:
        return 1.0, meta

    window = int(cfg.get("phase2_recovery_window", 180))
    trigger_dd = float(cfg.get("phase2_recovery_trigger_drawdown", -0.08))
    mom_w = int(cfg.get("phase2_recovery_momentum_window", 20))
    max_mult = max(1.0, float(cfg.get("phase2_recovery_max_mult", 1.20)))
    if i < max(window, mom_w):
        return 1.0, meta

    sl = benchmark_close.iloc[i - window + 1 : i + 1]
    peak = float(sl.max())
    trough = float(sl.min())
    px = float(benchmark_close.iloc[i])
    if peak <= 1e-12:
        return 1.0, meta

    dd_now = float(px / peak - 1.0)
    dd_depth = float(trough / peak - 1.0)
    meta["drawdown"] = dd_now
    meta["drawdown_depth"] = dd_depth
    if dd_depth > trigger_dd:
        return 1.0, meta

    if (peak - trough) <= 1e-12:
        return 1.0, meta
    recovery_progress = float((px - trough) / (peak - trough))
    recovery_progress = _clamp(recovery_progress, 0.0, 1.0)
    meta["recovery_progress"] = recovery_progress

    prev = float(benchmark_close.iloc[i - mom_w])
    if abs(prev) <= 1e-12:
        return 1.0, meta
    mom = float(px / prev - 1.0)
    meta["momentum"] = mom
    if mom <= 0.0:
        return 1.0, meta

    mult = 1.0 + (max_mult - 1.0) * recovery_progress
    return float(_clamp(mult, 1.0, max_mult)), meta


def _score_dispersion(scores: dict, eligible: list[str]) -> float:
    vals = sorted([float(scores[s]["score"]) for s in eligible if s in scores], reverse=True)
    if len(vals) < 2:
        return 0.0
    ref = max(abs(vals[0]), 1e-6)
    if len(vals) >= 3:
        return float(max(0.0, (vals[0] - vals[2]) / ref))
    return float(max(0.0, (vals[0] - vals[-1]) / ref))


def build_signal(
    prices: pd.DataFrame,
    i: int,
    symbols: list,
    benchmark_symbol: str,
    defensive_symbol: str,
    alloc_pct: float,
    single_max: float,
    params: dict,
) -> tuple[dict, bool, str, list]:
    risk_symbols = [s for s in symbols if s != defensive_symbol]
    momentum_lb = int(params["momentum_lb"])
    ma_window = int(params["ma_window"])
    vol_window = int(params["vol_window"])
    top_n = int(params["top_n"])
    min_score = float(params.get("min_score", 0.0))
    risk_on_score_mix = float(params.get("risk_on_score_mix", 0.0))
    risk_on_score_floor = float(params.get("risk_on_score_floor", min_score))
    risk_on_score_power = float(params.get("risk_on_score_power", 1.0))
    defensive_bypass_single_max = bool(params.get("defensive_bypass_single_max", True))
    structural_cfg = _resolve_structural_cfg(params)
    structural_enabled = bool(structural_cfg["enabled"])

    ret = prices.pct_change().fillna(0.0)
    scores = {}
    benchmark_close = prices[benchmark_symbol]
    benchmark_ma = None
    benchmark_trend_on = False
    if i + 1 >= ma_window:
        benchmark_ma = float(benchmark_close.iloc[i - ma_window + 1 : i + 1].mean())
        benchmark_trend_on = bool(prices[benchmark_symbol].iloc[i] >= benchmark_ma)

    benchmark_rel_momentum = None
    rel_w = int(structural_cfg["relative_momentum_window"])
    if structural_enabled and i >= rel_w:
        bench_prev = float(benchmark_close.iloc[i - rel_w])
        if abs(bench_prev) > 1e-12:
            benchmark_rel_momentum = float(benchmark_close.iloc[i] / bench_prev - 1.0)

    for s in risk_symbols:
        close = prices[s]
        need = max(momentum_lb, vol_window)
        if structural_enabled:
            need = max(
                need,
                max(structural_cfg["momentum_windows"]),
                rel_w,
                int(structural_cfg["asset_trend_ma_window"]),
            )
        if i < need:
            continue

        if structural_enabled:
            long_prev = float(close.iloc[i - momentum_lb])
            if abs(long_prev) <= 1e-12:
                continue
            long_m = float(close.iloc[i] / long_prev - 1.0)
            comp_m = _weighted_momentum(
                series=close,
                i=i,
                windows=structural_cfg["momentum_windows"],
                weights=structural_cfg["momentum_weights"],
            )
            if comp_m is None:
                continue
            rel_m = 0.0
            if benchmark_rel_momentum is not None and i >= rel_w:
                sym_prev = float(close.iloc[i - rel_w])
                if abs(sym_prev) > 1e-12:
                    sym_rel = float(close.iloc[i] / sym_prev - 1.0)
                    rel_m = float(sym_rel - benchmark_rel_momentum)
            struct_signal = float(comp_m + structural_cfg["relative_momentum_weight"] * rel_m)
            blend = float(structural_cfg["score_blend"])
            m = float((1.0 - blend) * long_m + blend * struct_signal)
            asset_ma_window = int(structural_cfg["asset_trend_ma_window"])
            asset_ma = float(close.iloc[i - asset_ma_window + 1 : i + 1].mean())
            trend_ok = bool(close.iloc[i] >= asset_ma)
        else:
            m = float(prices[s].iloc[i] / prices[s].iloc[i - momentum_lb] - 1.0)
            comp_m = m
            rel_m = 0.0
            trend_ok = True

        vol = float(ret[s].iloc[i - vol_window + 1 : i + 1].std())
        if vol <= 0:
            continue
        scores[s] = {
            "momentum": float(m),
            "composite_momentum": float(comp_m),
            "relative_momentum": float(rel_m),
            "trend_ok": bool(trend_ok),
            "vol": vol,
            "score": float(m / max(vol, 1e-6)),
        }

    breadth_universe = [s for s in scores.keys() if s != benchmark_symbol] or list(scores.keys())
    breadth = 0.0
    if breadth_universe:
        breadth_count = sum(
            1
            for s in breadth_universe
            if bool(scores[s]["trend_ok"]) and float(scores[s]["composite_momentum"]) > 0.0
        )
        breadth = float(breadth_count / len(breadth_universe))

    if structural_enabled:
        if benchmark_trend_on and breadth >= float(structural_cfg["breadth_on_threshold"]):
            regime_state = "risk_on"
        elif benchmark_trend_on and breadth >= float(structural_cfg["breadth_off_threshold"]):
            regime_state = "neutral"
        else:
            regime_state = "risk_off"
    else:
        regime_state = "risk_on" if benchmark_trend_on else "risk_off"
    regime_state_initial = str(regime_state)
    phase2_pre_state = "normal"
    phase2_pre_ratio = None
    phase2_pre_mult = 1.0
    phase2_stepdown_applied = False
    if structural_enabled and bool(structural_cfg.get("phase2_enabled", False)):
        phase2_pre_state, phase2_pre_mult, phase2_pre_ratio = _compute_vol_state_multiplier(
            benchmark_ret=ret[benchmark_symbol],
            i=i,
            cfg=structural_cfg,
        )
        stepdown = int(structural_cfg.get("phase2_stress_regime_stepdown", 0))
        if phase2_pre_state == "stress" and stepdown > 0:
            if stepdown >= 2 and regime_state in {"risk_on", "neutral"}:
                regime_state = "risk_off"
                phase2_stepdown_applied = True
            elif stepdown >= 1 and regime_state == "risk_on":
                regime_state = "neutral"
                phase2_stepdown_applied = True

    regime_on = bool(regime_state != "risk_off")
    if structural_enabled:
        eligible = [
            s
            for s in scores
            if scores[s]["score"] > min_score
            and ((not structural_cfg["eligibility_require_trend"]) or scores[s]["trend_ok"])
            and scores[s]["composite_momentum"] > 0.0
        ]
    else:
        eligible = [s for s in scores if scores[s]["score"] > min_score]

    target = {s: 0.0 for s in symbols}
    reason = "risk_off"
    risk_weights = {}
    risk_governor_meta = {
        "enabled": bool(params.get("risk_governor", {}).get("enabled", False)) if isinstance(params, dict) else False,
        "alloc_mult": 1.0,
        "structural_enabled": bool(structural_enabled),
        "regime_state": str(regime_state),
        "regime_state_initial": str(regime_state_initial),
        "phase2_stress_stepdown": int(structural_cfg.get("phase2_stress_regime_stepdown", 0)),
        "phase2_regime_stepdown_applied": bool(phase2_stepdown_applied),
        "benchmark_trend_on": bool(benchmark_trend_on),
        "benchmark_ma": benchmark_ma,
        "breadth": float(breadth),
        "neutral_alloc_mult": float(structural_cfg["neutral_alloc_mult"]),
        "trend_alloc_mult": 1.0,
        "breadth_alloc_mult": 1.0,
        "structural_alloc_mult": 1.0,
        "risk_alloc_base": float(alloc_pct),
        "risk_alloc_before_rg": float(alloc_pct),
        "phase2_enabled": bool(structural_enabled and structural_cfg.get("phase2_enabled", False)),
        "phase2_vol_state": str(phase2_pre_state),
        "phase2_vol_ratio": phase2_pre_ratio,
        "phase2_vol_mult": float(phase2_pre_mult),
        "phase2_recovery_mult": 1.0,
        "phase2_recovery_drawdown": None,
        "phase2_recovery_progress": None,
        "phase2_alloc_mult": 1.0,
        "signal_strength_enabled": bool(structural_cfg.get("signal_strength_enabled", False)),
        "signal_strength_value": None,
        "signal_strength_alloc_mult": 1.0,
        "score_dispersion": 0.0,
        "top_n_effective": int(top_n),
        "score_power_effective": float(risk_on_score_power),
    }
    if regime_state in {"risk_on", "neutral"} and eligible:
        risk_alloc_base = float(alloc_pct)
        if regime_state == "neutral":
            risk_alloc_base *= float(structural_cfg["neutral_alloc_mult"])

        trend_alloc_mult = 1.0
        breadth_alloc_mult = 1.0
        structural_alloc_mult = 1.0
        if structural_enabled and benchmark_ma is not None and abs(float(benchmark_ma)) > 1e-12:
            trend_strength = float(prices[benchmark_symbol].iloc[i] / benchmark_ma - 1.0)
            trend_alloc_mult = _clamp(
                1.0 + trend_strength * float(structural_cfg["trend_strength_k"]),
                float(structural_cfg["trend_alloc_mult_min"]),
                float(structural_cfg["trend_alloc_mult_max"]),
            )
            breadth_target = (
                float(structural_cfg["breadth_on_threshold"])
                if regime_state == "risk_on"
                else float(structural_cfg["breadth_off_threshold"])
            )
            breadth_alloc_mult = _clamp(
                float(breadth / max(breadth_target, 1e-6)),
                float(structural_cfg["breadth_alloc_mult_min"]),
                float(structural_cfg["breadth_alloc_mult_max"]),
            )
            structural_alloc_mult = _clamp(
                float(trend_alloc_mult * breadth_alloc_mult),
                float(structural_cfg["total_alloc_mult_min"]),
                float(structural_cfg["total_alloc_mult_max"]),
            )

        top_n_base = int(top_n if regime_state == "risk_on" else min(top_n, structural_cfg["neutral_top_n"]))
        top_n_eff = max(1, top_n_base)
        score_power_eff = float(risk_on_score_power)
        phase2_alloc_mult = 1.0
        phase2_vol_state = "normal"
        phase2_vol_ratio = None
        phase2_vol_mult = 1.0
        phase2_recovery_mult = 1.0
        phase2_recovery_meta = {}
        if structural_enabled and bool(structural_cfg.get("phase2_enabled", False)):
            phase2_vol_state = str(phase2_pre_state)
            phase2_vol_mult = float(phase2_pre_mult)
            phase2_vol_ratio = phase2_pre_ratio
            phase2_recovery_mult, phase2_recovery_meta = _compute_drawdown_recovery_multiplier(
                benchmark_close=benchmark_close,
                i=i,
                cfg=structural_cfg,
            )
            phase2_alloc_mult = _clamp(
                float(phase2_vol_mult * phase2_recovery_mult),
                float(structural_cfg["phase2_total_mult_min"]),
                float(structural_cfg["phase2_total_mult_max"]),
            )
            if bool(structural_cfg.get("phase2_adaptive_enabled", True)):
                score_disp = _score_dispersion(scores=scores, eligible=eligible)
                if phase2_vol_state == "calm":
                    top_n_eff += int(structural_cfg["phase2_adaptive_calm_top_n_delta"])
                    score_power_eff *= float(structural_cfg["phase2_adaptive_score_power_calm_mult"])
                elif phase2_vol_state == "stress":
                    top_n_eff += int(structural_cfg["phase2_adaptive_stress_top_n_delta"])
                    score_power_eff *= float(structural_cfg["phase2_adaptive_score_power_stress_mult"])
                if score_disp >= float(structural_cfg["phase2_adaptive_high_dispersion_threshold"]):
                    top_n_eff += int(structural_cfg["phase2_adaptive_high_dispersion_top_n_delta"])
                top_n_eff = int(
                    _clamp(
                        float(top_n_eff),
                        float(structural_cfg["phase2_adaptive_min_top_n"]),
                        float(structural_cfg["phase2_adaptive_max_top_n"]),
                    )
                )
                top_n_eff = min(top_n_eff, max(1, len(eligible)))
                risk_governor_meta["score_dispersion"] = float(score_disp)

        picks = sorted(eligible, key=lambda s: scores[s]["score"], reverse=True)[: max(1, top_n_eff)]
        signal_strength_mult = 1.0
        signal_strength_value = None
        if bool(structural_cfg.get("signal_strength_enabled", False)) and picks:
            strength_floor = float(structural_cfg.get("signal_strength_floor", 0.0))
            strength_ref = max(1e-6, float(structural_cfg.get("signal_strength_ref", 0.06)))
            strength_curve = max(0.25, float(structural_cfg.get("signal_strength_curve", 1.0)))
            strength_vals = [
                max(float(scores[s]["composite_momentum"]) - strength_floor, 0.0) for s in picks if s in scores
            ]
            if strength_vals:
                signal_strength_value = float(sum(strength_vals) / len(strength_vals))
                signal_strength_mult = _clamp(
                    float((signal_strength_value / strength_ref) ** strength_curve),
                    float(structural_cfg.get("signal_strength_min_alloc_mult", 0.70)),
                    float(structural_cfg.get("signal_strength_max_alloc_mult", 1.10)),
                )

        risk_alloc_pre_rg = float(risk_alloc_base * structural_alloc_mult * phase2_alloc_mult * signal_strength_mult)
        alloc_mult, rg_meta = compute_risk_alloc_multiplier(
            prices=prices,
            i=i,
            benchmark_symbol=benchmark_symbol,
            params=params,
        )
        risk_governor_meta.update(rg_meta)
        risk_governor_meta["trend_alloc_mult"] = float(trend_alloc_mult)
        risk_governor_meta["breadth_alloc_mult"] = float(breadth_alloc_mult)
        risk_governor_meta["structural_alloc_mult"] = float(structural_alloc_mult)
        risk_governor_meta["risk_alloc_base"] = float(risk_alloc_base)
        risk_governor_meta["risk_alloc_before_rg"] = float(risk_alloc_pre_rg)
        risk_governor_meta["phase2_vol_state"] = str(phase2_vol_state)
        risk_governor_meta["phase2_vol_ratio"] = phase2_vol_ratio
        risk_governor_meta["phase2_vol_mult"] = float(phase2_vol_mult)
        risk_governor_meta["phase2_recovery_mult"] = float(phase2_recovery_mult)
        risk_governor_meta["phase2_recovery_drawdown"] = phase2_recovery_meta.get("drawdown")
        risk_governor_meta["phase2_recovery_progress"] = phase2_recovery_meta.get("recovery_progress")
        risk_governor_meta["phase2_alloc_mult"] = float(phase2_alloc_mult)
        risk_governor_meta["signal_strength_value"] = signal_strength_value
        risk_governor_meta["signal_strength_alloc_mult"] = float(signal_strength_mult)
        risk_governor_meta["top_n_effective"] = int(top_n_eff)
        risk_governor_meta["score_power_effective"] = float(score_power_eff)

        risk_alloc = float(risk_alloc_pre_rg * alloc_mult)
        risk_weights = build_risk_on_weights(
            scores=scores,
            picks=picks,
            score_mix=risk_on_score_mix,
            score_floor=risk_on_score_floor,
            score_power=score_power_eff,
        )
        for s, w in risk_weights.items():
            target[s] = risk_alloc * w
        for s in picks:
            if (not defensive_bypass_single_max) or s != defensive_symbol:
                target[s] = min(target[s], single_max)
        used = sum(target.values())
        if used < alloc_pct:
            if (not defensive_bypass_single_max):
                room = max(0.0, single_max - target[defensive_symbol])
                target[defensive_symbol] += min(room, alloc_pct - used)
            else:
                target[defensive_symbol] += alloc_pct - used
        reason = "risk_on" if regime_state == "risk_on" else "risk_on_neutral"
    else:
        target[defensive_symbol] = alloc_pct
        if not benchmark_trend_on:
            reason = "benchmark_below_ma"
        elif phase2_stepdown_applied and regime_state == "risk_off":
            reason = "phase2_stress_risk_off"
        elif regime_state == "risk_off":
            reason = "breadth_risk_off"
        elif not eligible:
            reason = "no_positive_momentum"

    score_rows = []
    for s in sorted(scores.keys(), key=lambda x: scores[x]["score"], reverse=True):
        score_rows.append(
            {
                "symbol": s,
                "momentum": round(scores[s]["momentum"], 6),
                "composite_momentum": round(scores[s]["composite_momentum"], 6),
                "relative_momentum": round(scores[s]["relative_momentum"], 6),
                "trend_ok": bool(scores[s]["trend_ok"]),
                "vol": round(scores[s]["vol"], 6),
                "score": round(scores[s]["score"], 6),
                "risk_weight": round(float(risk_weights.get(s, 0.0)), 6),
            }
        )
    return target, regime_on, reason, score_rows, risk_governor_meta


def run_paper_forward(runtime, stock, risk):
    model_cfg = stock.get("global_model", {})
    guard_cfg = model_cfg.get("execution_guard", {})
    overlay_cfg = model_cfg.get("risk_overlay", {})
    exposure_cfg = model_cfg.get("exposure_gate", {})

    benchmark_symbol = stock.get("benchmark_symbol", "510300")
    defensive_symbol = stock.get("defensive_symbol", "511010")
    symbols = sorted(set(stock.get("universe", []) + [benchmark_symbol, defensive_symbol]))
    alloc_pct = float(stock.get("capital_alloc_pct", 0.7))
    single_max = float(risk["position_limits"]["stock_single_max_pct"])

    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    report_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(report_dir)

    prices = load_price_frame(data_dir, symbols)
    warmup_min_days = int(model_cfg.get("warmup_min_days", 126))
    struct_cfg = model_cfg.get("structural_upgrade", {}) or {}
    struct_windows = []
    for x in struct_cfg.get("momentum_windows", []):
        try:
            w = int(x)
        except (TypeError, ValueError):
            continue
        if w > 1:
            struct_windows.append(w)
    struct_max_window = max(struct_windows) if (bool(struct_cfg.get("enabled", False)) and struct_windows) else 0
    struct_rel_window = (
        int(struct_cfg.get("relative_momentum_window", 60)) if bool(struct_cfg.get("enabled", False)) else 0
    )
    struct_asset_ma = int(struct_cfg.get("asset_trend_ma_window", 80)) if bool(struct_cfg.get("enabled", False)) else 0
    struct_phase2_enabled = bool(struct_cfg.get("enabled", False)) and bool(struct_cfg.get("phase2_enabled", False))
    struct_phase2_vol_long = int(struct_cfg.get("phase2_vol_long_window", 80)) if struct_phase2_enabled else 0
    struct_phase2_recovery_window = int(struct_cfg.get("phase2_recovery_window", 180)) if struct_phase2_enabled else 0
    struct_phase2_recovery_mom = int(struct_cfg.get("phase2_recovery_momentum_window", 20)) if struct_phase2_enabled else 0
    warmup = max(
        int(model_cfg.get("momentum_lb", 252)),
        int(model_cfg.get("ma_window", 200)),
        int(model_cfg.get("vol_window", 20)),
        struct_max_window,
        struct_rel_window,
        struct_asset_ma,
        struct_phase2_vol_long,
        struct_phase2_recovery_window,
        struct_phase2_recovery_mom,
        warmup_min_days,
    ) + 1
    if warmup >= len(prices) - 1:
        raise RuntimeError("not enough history for paper-forward simulation")

    fee = float(model_cfg.get("fee", 0.0008))
    min_rebalance_days = int(guard_cfg.get("min_rebalance_days", model_cfg.get("rebalance_days", 20)))
    min_turnover_base = float(guard_cfg.get("min_turnover", 0.05))
    force_regime = bool(guard_cfg.get("force_rebalance_on_regime_change", True))
    guard_enabled = bool(guard_cfg.get("enabled", True))
    defensive_bypass_single_max = bool(model_cfg.get("defensive_bypass_single_max", True))
    overlay_enabled = bool(overlay_cfg.get("enabled", False))
    overlay_trigger_excess_20d = float(overlay_cfg.get("trigger_excess_20d_vs_alloc", -0.02))
    overlay_trigger_dd = float(overlay_cfg.get("trigger_strategy_drawdown", -0.12))
    overlay_release_excess_20d = float(overlay_cfg.get("release_excess_20d_vs_alloc", 0.01))
    overlay_release_dd = float(overlay_cfg.get("release_strategy_drawdown", -0.06))
    overlay_min_defense_days = int(overlay_cfg.get("min_defense_days", 10))
    overlay_sticky_mode = bool(overlay_cfg.get("sticky_mode", True))

    weights = {s: 0.0 for s in symbols}
    weights[defensive_symbol] = alloc_pct
    last_rebalance_date = None
    last_regime_on = None
    overlay_state_active = False
    overlay_activated_date = None

    strategy_nav = 1.0
    benchmark_nav_raw = 1.0
    benchmark_nav_alloc = 1.0
    strategy_peak = 1.0
    benchmark_peak_raw = 1.0
    benchmark_peak_alloc = 1.0

    records = []
    hist_strategy_ret = []
    hist_benchmark_ret_alloc = []
    exposure_stage_count = Counter()
    min_turnover_meta = {"enabled": False, "base_min_turnover": min_turnover_base, "effective_min_turnover": min_turnover_base}
    last_exposure_meta = {
        "enabled": bool(exposure_cfg.get("enabled", False)),
        "base_alloc_pct": alloc_pct,
        "effective_alloc_pct": alloc_pct,
        "stage": "disabled",
        "reason": "init",
    }
    for i in range(warmup, len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        alloc_effective, exposure_meta = evaluate_exposure_gate(
            hist_strategy_ret,
            hist_benchmark_ret_alloc,
            alloc_pct,
            exposure_cfg,
        )
        min_turnover, min_turnover_meta = resolve_min_turnover(min_turnover_base, alloc_effective, guard_cfg)
        proposed, regime_on, signal_reason, score_rows, risk_governor_meta = build_signal(
            prices=prices,
            i=i,
            symbols=symbols,
            benchmark_symbol=benchmark_symbol,
            defensive_symbol=defensive_symbol,
            alloc_pct=alloc_effective,
            single_max=single_max,
            params=model_cfg,
        )
        proposed_before_overlay = proposed.copy()
        overlay_meta = {
            "enabled": bool(overlay_enabled),
            "trigger_excess_20d_vs_alloc": float(overlay_trigger_excess_20d),
            "trigger_strategy_drawdown": float(overlay_trigger_dd),
            "release_excess_20d_vs_alloc": float(overlay_release_excess_20d),
            "release_strategy_drawdown": float(overlay_release_dd),
            "min_defense_days": int(overlay_min_defense_days),
            "sticky_mode": bool(overlay_sticky_mode),
            "triggered": False,
            "trigger_reasons": [],
            "released": False,
            "release_reasons": [],
            "state_active_before": bool(overlay_state_active),
            "state_active_after": bool(overlay_state_active),
            "activated_date": overlay_activated_date.isoformat() if overlay_activated_date else None,
            "days_in_defense": None,
            "metrics": {
                "excess_return_20d_vs_alloc": None,
                "strategy_dd": float(strategy_nav / strategy_peak - 1.0),
            },
        }
        overlay_force_rebalance = False
        overlay_force_reason = None
        if overlay_enabled:
            excess_20d = None
            if hist_strategy_ret:
                strat_s = pd.Series(hist_strategy_ret, dtype=float)
                bench_s = pd.Series(hist_benchmark_ret_alloc, dtype=float)
                ex_s = strat_s - bench_s
                excess_20d = float(ex_s.tail(20).sum())
            strategy_dd_now = float(strategy_nav / strategy_peak - 1.0)
            overlay_meta["metrics"]["excess_return_20d_vs_alloc"] = excess_20d
            overlay_meta["metrics"]["strategy_dd"] = strategy_dd_now

            days_in_defense = None
            if overlay_activated_date is not None:
                days_in_defense = int((date.date() - overlay_activated_date).days)
            overlay_meta["days_in_defense"] = days_in_defense

            trigger_reasons = []
            if (excess_20d is not None) and (float(excess_20d) <= overlay_trigger_excess_20d):
                trigger_reasons.append("excess_20d_below_threshold")
            if strategy_dd_now <= overlay_trigger_dd:
                trigger_reasons.append("drawdown_below_threshold")

            released = False
            release_reasons = []
            if overlay_sticky_mode and overlay_state_active:
                release_ok = True
                if (excess_20d is None) or (float(excess_20d) < overlay_release_excess_20d):
                    release_ok = False
                if strategy_dd_now < overlay_release_dd:
                    release_ok = False
                if (days_in_defense is None) or (days_in_defense < overlay_min_defense_days):
                    release_ok = False
                if release_ok:
                    overlay_state_active = False
                    released = True
                    release_reasons.append("release_threshold_reached")
                    overlay_activated_date = None
            elif (not overlay_sticky_mode) and overlay_state_active and (not trigger_reasons):
                overlay_state_active = False
                released = True
                release_reasons.append("trigger_cleared_non_sticky")
                overlay_activated_date = None

            if trigger_reasons:
                overlay_state_active = True
                if overlay_activated_date is None:
                    overlay_activated_date = date.date()

            if overlay_state_active:
                tw = float(alloc_effective)
                if not defensive_bypass_single_max:
                    tw = min(tw, float(single_max))
                proposed = {s: 0.0 for s in symbols}
                proposed[defensive_symbol] = tw
                if calc_turnover(proposed_before_overlay, proposed, symbols) > 1e-8:
                    overlay_force_rebalance = True
                    overlay_force_reason = "risk_overlay_active"
            elif released:
                overlay_force_rebalance = True
                overlay_force_reason = "risk_overlay_released"

            overlay_meta["triggered"] = bool(trigger_reasons)
            overlay_meta["trigger_reasons"] = trigger_reasons
            overlay_meta["released"] = bool(released)
            overlay_meta["release_reasons"] = release_reasons
            overlay_meta["state_active_after"] = bool(overlay_state_active)
            overlay_meta["activated_date"] = overlay_activated_date.isoformat() if overlay_activated_date else None

        action = "rebalance"
        action_reason = "normal"
        executed = proposed
        proposed_turnover = calc_turnover(weights, proposed, symbols)
        days_since = None

        if overlay_force_rebalance:
            action = "rebalance"
            action_reason = str(overlay_force_reason or "risk_overlay")
            executed = proposed
        elif guard_enabled and last_rebalance_date is not None:
            days_since = (date.date() - last_rebalance_date).days
            if force_regime and last_regime_on is not None and regime_on != last_regime_on:
                action = "rebalance"
                action_reason = "regime_changed"
                executed = proposed
            elif days_since < min_rebalance_days:
                action = "hold"
                action_reason = "min_rebalance_days"
                executed = weights.copy()
            elif proposed_turnover < min_turnover:
                action = "hold"
                action_reason = "turnover_below_threshold"
                executed = weights.copy()

        turnover = calc_turnover(weights, executed, symbols)
        if action == "rebalance":
            last_rebalance_date = date.date()

        strategy_ret = 0.0
        for s, w in executed.items():
            if w == 0.0:
                continue
            strategy_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        strategy_ret -= turnover * fee

        benchmark_ret_raw = prices[benchmark_symbol].iloc[i + 1] / prices[benchmark_symbol].iloc[i] - 1.0
        benchmark_ret_alloc = alloc_pct * benchmark_ret_raw
        excess_ret_vs_raw = strategy_ret - benchmark_ret_raw
        excess_ret_vs_alloc = strategy_ret - benchmark_ret_alloc

        strategy_nav *= 1.0 + strategy_ret
        benchmark_nav_raw *= 1.0 + benchmark_ret_raw
        benchmark_nav_alloc *= 1.0 + benchmark_ret_alloc
        strategy_peak = max(strategy_peak, strategy_nav)
        benchmark_peak_raw = max(benchmark_peak_raw, benchmark_nav_raw)
        benchmark_peak_alloc = max(benchmark_peak_alloc, benchmark_nav_alloc)

        records.append(
            {
                "date": date.date().isoformat(),
                "next_date": next_date.date().isoformat(),
                "action": action,
                "action_reason": action_reason,
                "alloc_pct_base": alloc_pct,
                "alloc_pct_effective": float(alloc_effective),
                "exposure_gate_stage": str(exposure_meta.get("stage", "unknown")),
                "exposure_gate_reason": str(exposure_meta.get("reason", "")),
                "exposure_gate_agg_excess_ann_worst": exposure_meta.get("agg_excess_ann_worst"),
                "min_turnover_effective": float(min_turnover),
                "regime_on": bool(regime_on),
                "signal_reason": signal_reason,
                "risk_alloc_mult": float(risk_governor_meta.get("alloc_mult", 1.0)),
                "risk_governor": json.dumps(risk_governor_meta, ensure_ascii=False),
                "risk_overlay": json.dumps(overlay_meta, ensure_ascii=False),
                "days_since_last_rebalance": days_since,
                "proposed_turnover": round(proposed_turnover, 6),
                "executed_turnover": round(turnover, 6),
                "strategy_ret": strategy_ret,
                "benchmark_ret_raw": benchmark_ret_raw,
                "benchmark_ret_alloc": benchmark_ret_alloc,
                "excess_ret_vs_raw": excess_ret_vs_raw,
                "excess_ret_vs_alloc": excess_ret_vs_alloc,
                "strategy_nav": strategy_nav,
                "benchmark_nav_raw": benchmark_nav_raw,
                "benchmark_nav_alloc": benchmark_nav_alloc,
                "excess_nav_vs_raw": strategy_nav - benchmark_nav_raw,
                "excess_nav_vs_alloc": strategy_nav - benchmark_nav_alloc,
                "strategy_dd": strategy_nav / strategy_peak - 1.0,
                "benchmark_dd_raw": benchmark_nav_raw / benchmark_peak_raw - 1.0,
                "benchmark_dd_alloc": benchmark_nav_alloc / benchmark_peak_alloc - 1.0,
                "executed_weights": weights_to_json(executed),
                "proposed_weights": weights_to_json(proposed),
                "scores": json.dumps(score_rows, ensure_ascii=False),
            }
        )
        hist_strategy_ret.append(float(strategy_ret))
        hist_benchmark_ret_alloc.append(float(benchmark_ret_alloc))
        exposure_stage_count[str(exposure_meta.get("stage", "unknown"))] += 1
        last_exposure_meta = exposure_meta
        weights = executed
        last_regime_on = regime_on

    df = pd.DataFrame(records)
    history_file = os.path.join(report_dir, "stock_paper_forward_history.csv")
    df.to_csv(history_file, index=False, encoding="utf-8")

    strategy_nav_series = pd.Series([1.0] + df["strategy_nav"].tolist())
    benchmark_nav_raw_series = pd.Series([1.0] + df["benchmark_nav_raw"].tolist())
    benchmark_nav_alloc_series = pd.Series([1.0] + df["benchmark_nav_alloc"].tolist())
    strategy_ret_series = df["strategy_ret"]
    benchmark_ret_raw_series = df["benchmark_ret_raw"]
    benchmark_ret_alloc_series = df["benchmark_ret_alloc"]
    excess_ret_vs_raw_series = df["excess_ret_vs_raw"]
    excess_ret_vs_alloc_series = df["excess_ret_vs_alloc"]

    last = df.iloc[-1]
    rolling_20_raw = excess_ret_vs_raw_series.tail(20).sum() if len(df) >= 20 else excess_ret_vs_raw_series.sum()
    rolling_60_raw = excess_ret_vs_raw_series.tail(60).sum() if len(df) >= 60 else excess_ret_vs_raw_series.sum()
    rolling_20_alloc = excess_ret_vs_alloc_series.tail(20).sum() if len(df) >= 20 else excess_ret_vs_alloc_series.sum()
    rolling_60_alloc = excess_ret_vs_alloc_series.tail(60).sum() if len(df) >= 60 else excess_ret_vs_alloc_series.sum()
    action_cnt = Counter(df["action"].tolist())
    reason_cnt = Counter(df["action_reason"].tolist())

    strategy_ann = annualized_return(strategy_nav_series, 252)
    benchmark_ann_raw = annualized_return(benchmark_nav_raw_series, 252)
    benchmark_ann_alloc = annualized_return(benchmark_nav_alloc_series, 252)

    summary = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "mode": "global_momentum",
        "history_file": history_file,
        "date_range": {"from": str(df["date"].iloc[0]), "to": str(df["date"].iloc[-1])},
        "latest": {
            "date": str(last["date"]),
            "next_date": str(last["next_date"]),
            "action": str(last["action"]),
            "action_reason": str(last["action_reason"]),
            "regime_on": bool(last["regime_on"]),
            "signal_reason": str(last["signal_reason"]),
            "executed_turnover": float(last["executed_turnover"]),
            "strategy_nav": float(last["strategy_nav"]),
            "benchmark_nav_raw": float(last["benchmark_nav_raw"]),
            "benchmark_nav_alloc": float(last["benchmark_nav_alloc"]),
            "excess_nav_vs_raw": float(last["excess_nav_vs_raw"]),
            "excess_nav_vs_alloc": float(last["excess_nav_vs_alloc"]),
            "strategy_dd": float(last["strategy_dd"]),
            "benchmark_dd_raw": float(last["benchmark_dd_raw"]),
            "benchmark_dd_alloc": float(last["benchmark_dd_alloc"]),
        },
        "aggregate": {
            "periods": int(len(df)),
            "strategy_annual_return": strategy_ann,
            "benchmark_annual_return_raw": benchmark_ann_raw,
            "benchmark_annual_return_alloc": benchmark_ann_alloc,
            "excess_annual_return_vs_raw": strategy_ann - benchmark_ann_raw,
            "excess_annual_return_vs_alloc": strategy_ann - benchmark_ann_alloc,
            "strategy_max_drawdown": max_drawdown(strategy_nav_series),
            "benchmark_max_drawdown_raw": max_drawdown(benchmark_nav_raw_series),
            "benchmark_max_drawdown_alloc": max_drawdown(benchmark_nav_alloc_series),
            "strategy_sharpe": sharpe_ratio(strategy_ret_series, 252),
            "benchmark_sharpe_raw": sharpe_ratio(benchmark_ret_raw_series, 252),
            "benchmark_sharpe_alloc": sharpe_ratio(benchmark_ret_alloc_series, 252),
            "excess_sharpe_vs_raw": sharpe_ratio(excess_ret_vs_raw_series, 252),
            "excess_sharpe_vs_alloc": sharpe_ratio(excess_ret_vs_alloc_series, 252),
            "total_turnover": float(df["executed_turnover"].sum()),
            "avg_turnover": float(df["executed_turnover"].mean()),
            "trades": int((df["executed_turnover"] > 1e-9).sum()),
            "action_count": dict(action_cnt),
            "reason_count": dict(reason_cnt),
        },
        "rolling": {
            "excess_return_20d_vs_raw": float(rolling_20_raw),
            "excess_return_60d_vs_raw": float(rolling_60_raw),
            "excess_return_20d_vs_alloc": float(rolling_20_alloc),
            "excess_return_60d_vs_alloc": float(rolling_60_alloc),
        },
        "guard_params": {
            "enabled": guard_enabled,
            "min_rebalance_days": min_rebalance_days,
            "min_turnover_base": min_turnover_base,
            "min_turnover": float(last["min_turnover_effective"]),
            "dynamic_min_turnover": min_turnover_meta,
            "force_rebalance_on_regime_change": force_regime,
        },
        "risk_governor": {
            "enabled": bool(model_cfg.get("risk_governor", {}).get("enabled", False)),
            "params": model_cfg.get("risk_governor", {}),
            "latest_alloc_mult": float(last.get("risk_alloc_mult", 1.0)),
        },
        "exposure_gate": {
            "enabled": bool(exposure_cfg.get("enabled", False)),
            "stage_count": dict(exposure_stage_count),
            "latest": last_exposure_meta,
            "params": exposure_cfg,
        },
    }

    latest_file = os.path.join(report_dir, "stock_paper_forward_latest.json")
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, history_file, latest_file


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock = load_yaml("config/stock.yaml")
        risk = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[paper-forward] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock.get("enabled", False):
        print("[stock] disabled")
        return EXIT_DISABLED
    if stock.get("mode") != "global_momentum":
        print("[stock] mode is not global_momentum, skip paper forward report")
        return EXIT_OK

    try:
        summary, history_file, latest_file = run_paper_forward(runtime, stock, risk)
    except OSError as e:
        print(f"[paper-forward] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[paper-forward] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[paper-forward] history -> {history_file}")
    print(f"[paper-forward] latest  -> {latest_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
