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


def load_price_frame(data_dir: str, symbols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        x = (
            df.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
        )
        if x.empty:
            continue
        frames[s] = x["close"]

    if len(frames) < len(symbols):
        missing = [s for s in symbols if s not in frames]
        raise RuntimeError(f"missing stock data files: {missing}")

    all_dates = sorted(set.union(*[set(v.index) for v in frames.values()]))
    if len(all_dates) < 400:
        raise RuntimeError("not enough stock history (<400 rows union)")

    union_index = pd.DatetimeIndex(all_dates)
    prices_raw = pd.DataFrame({s: frames[s].reindex(union_index) for s in symbols}, index=union_index).sort_index()
    availability = prices_raw.notna()

    # Use forward-fill on/after listing dates so non-trading gaps map to 0 return.
    # Before listing, values remain NaN and are filtered by per-symbol history guards.
    prices = prices_raw.ffill()
    return prices, availability


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


def _parse_symbol_weight_caps(raw_caps, default_cap: float) -> dict:
    if not isinstance(raw_caps, dict):
        return {}
    out = {}
    for k, v in raw_caps.items():
        sym = str(k or "").strip()
        if not sym:
            continue
        try:
            cap = float(v)
        except (TypeError, ValueError):
            continue
        if cap <= 0:
            continue
        out[sym] = float(_clamp(cap, 0.0, float(default_cap)))
    return out


def _cap_for_symbol(symbol: str, default_cap: float, caps: dict) -> float:
    if isinstance(caps, dict) and symbol in caps:
        return float(caps[symbol])
    return float(default_cap)


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


def _resolve_regime_profile(
    params: dict,
    regime_state: str,
    base_top_n: int,
    base_min_score: float,
    base_score_mix: float,
    base_score_floor: float,
    base_score_power: float,
) -> dict:
    cfg = params.get("regime_profiles", {}) if isinstance(params, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    raw = {}
    if enabled:
        x = cfg.get(regime_state, {})
        if isinstance(x, dict):
            raw = x

    def _pick_int(name: str, default: int, minimum: int = 1) -> int:
        try:
            return max(minimum, int(raw.get(name, default)))
        except (TypeError, ValueError):
            return max(minimum, int(default))

    def _pick_float(name: str, default: float) -> float:
        try:
            return float(raw.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    return {
        "enabled": bool(enabled),
        "regime": str(regime_state),
        "top_n": _pick_int("top_n", base_top_n, 1),
        "min_score": _pick_float("min_score", base_min_score),
        "score_mix": _clamp(_pick_float("score_mix", base_score_mix), 0.0, 1.0),
        "score_floor": _pick_float("score_floor", base_score_floor),
        "score_power": max(0.1, _pick_float("score_power", base_score_power)),
        "alloc_mult": max(0.0, _pick_float("alloc_mult", 1.0)),
        "turbo_enabled": bool(raw.get("turbo_enabled", False)),
        "turbo_trend_strength_min": _pick_float("turbo_trend_strength_min", 0.03),
        "turbo_breadth_min": _pick_float("turbo_breadth_min", 0.80),
        "turbo_alloc_mult_add": _pick_float("turbo_alloc_mult_add", 0.05),
        "turbo_top_n_add": _pick_int("turbo_top_n_add", 1, 0),
    }


def _resolve_defensive_rotation_cfg(params: dict, fallback_symbol: str) -> dict:
    cfg = params.get("defensive_rotation", {}) if isinstance(params, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    raw_symbols = cfg.get("symbols", [fallback_symbol])
    symbols = sorted(set([str(x) for x in raw_symbols] + [str(fallback_symbol)]))
    windows = []
    for x in cfg.get("momentum_windows", [20, 60, 120]):
        try:
            w = int(x)
        except (TypeError, ValueError):
            continue
        if w > 1:
            windows.append(w)
    if not windows:
        windows = [20, 60, 120]
    weights = _normalize_float_weights(cfg.get("momentum_weights", [0.5, 0.3, 0.2]), len(windows))
    adaptive_cfg = cfg.get("adaptive_allocation", {}) if isinstance(cfg.get("adaptive_allocation", {}), dict) else {}
    adaptive_enabled = bool(adaptive_cfg.get("enabled", False))
    adaptive_min_ratio = float(adaptive_cfg.get("min_ratio", 0.0))
    adaptive_max_ratio = float(adaptive_cfg.get("max_ratio", 0.60))
    if adaptive_max_ratio < adaptive_min_ratio:
        adaptive_max_ratio = adaptive_min_ratio
    return {
        "enabled": enabled,
        "symbols": symbols,
        "top_n": max(1, int(cfg.get("top_n", 1))),
        "momentum_windows": windows,
        "momentum_weights": weights,
        "vol_window": max(5, int(cfg.get("vol_window", 20))),
        "trend_ma_window": max(5, int(cfg.get("trend_ma_window", 60))),
        "min_momentum": float(cfg.get("min_momentum", -0.02)),
        "trend_filter": bool(cfg.get("trend_filter", True)),
        "score_power": max(0.1, float(cfg.get("score_power", 1.0))),
        "adaptive_enabled": adaptive_enabled,
        "adaptive_base_ratio": float(adaptive_cfg.get("base_ratio", 0.0)),
        "adaptive_min_ratio": adaptive_min_ratio,
        "adaptive_max_ratio": adaptive_max_ratio,
        "adaptive_neutral_add": float(adaptive_cfg.get("neutral_add", 0.0)),
        "adaptive_trend_ref": max(1e-6, float(adaptive_cfg.get("trend_ref", 0.08))),
        "adaptive_trend_weight": max(0.0, float(adaptive_cfg.get("trend_weight", 0.0))),
        "adaptive_breadth_ref": _clamp(float(adaptive_cfg.get("breadth_ref", 0.60)), 1e-6, 1.0 - 1e-6),
        "adaptive_breadth_weight": max(0.0, float(adaptive_cfg.get("breadth_weight", 0.0))),
        "adaptive_drawdown_window": max(20, int(adaptive_cfg.get("drawdown_window", 60))),
        "adaptive_drawdown_ref": float(adaptive_cfg.get("drawdown_ref", -0.08)),
        "adaptive_drawdown_weight": max(0.0, float(adaptive_cfg.get("drawdown_weight", 0.0))),
        "adaptive_recovery_window": max(5, int(adaptive_cfg.get("recovery_window", 20))),
        "adaptive_recovery_ref": max(1e-6, float(adaptive_cfg.get("recovery_ref", 0.05))),
        "adaptive_recovery_weight": max(0.0, float(adaptive_cfg.get("recovery_weight", 0.0))),
        "adaptive_calm_add": float(adaptive_cfg.get("calm_add", 0.0)),
        "adaptive_stress_add": float(adaptive_cfg.get("stress_add", 0.0)),
        "fallback_symbol": str(fallback_symbol),
    }


def _build_defensive_weights(
    prices: pd.DataFrame,
    obs_count: pd.DataFrame,
    i: int,
    defensive_cfg: dict,
) -> tuple[dict, dict]:
    fallback_symbol = str(defensive_cfg.get("fallback_symbol", ""))
    meta = {
        "enabled": bool(defensive_cfg.get("enabled", False)),
        "selected_symbols": [fallback_symbol] if fallback_symbol else [],
        "candidates": [],
        "reason": "fallback_default",
    }
    if (not meta["enabled"]) or (not fallback_symbol):
        return ({fallback_symbol: 1.0} if fallback_symbol else {}), meta

    candidates = []
    for symbol in defensive_cfg.get("symbols", []):
        if symbol not in prices.columns:
            continue
        need = max(
            max(defensive_cfg["momentum_windows"]),
            int(defensive_cfg["vol_window"]),
            int(defensive_cfg["trend_ma_window"]),
        )
        obs = int(obs_count[symbol].iloc[i]) if symbol in obs_count.columns else 0
        if obs <= need:
            continue
        series = prices[symbol].iloc[: i + 1]
        momentum = _weighted_momentum(
            series=series,
            i=len(series) - 1,
            windows=defensive_cfg["momentum_windows"],
            weights=defensive_cfg["momentum_weights"],
        )
        if momentum is None:
            continue
        ret = series.pct_change().dropna().tail(int(defensive_cfg["vol_window"]))
        vol = float(ret.std()) if len(ret) >= 2 else None
        if vol is None or vol <= 0:
            continue
        ma = float(series.tail(int(defensive_cfg["trend_ma_window"])).mean())
        trend_ok = bool(series.iloc[-1] >= ma) if ma > 0 else True
        score = max(float(momentum), float(defensive_cfg["min_momentum"]))
        if defensive_cfg.get("trend_filter", True) and (not trend_ok):
            score *= 0.5
        score = max(score, 0.0) ** float(defensive_cfg["score_power"])
        candidates.append(
            {
                "symbol": symbol,
                "momentum": float(momentum),
                "vol": float(vol),
                "trend_ok": bool(trend_ok),
                "score": float(score),
            }
        )

    meta["candidates"] = candidates
    if not candidates:
        return {fallback_symbol: 1.0}, meta

    ranked = sorted(candidates, key=lambda x: (x["score"], x["momentum"]), reverse=True)
    positive = [x for x in ranked if x["momentum"] >= float(defensive_cfg["min_momentum"])]
    chosen = positive[: int(defensive_cfg["top_n"])] if positive else []
    if not chosen:
        chosen = [min(ranked, key=lambda x: x["vol"])]
        meta["reason"] = "fallback_low_vol"
    else:
        meta["reason"] = "momentum_rotation"

    inv_vol = {x["symbol"]: 1.0 / max(float(x["vol"]), 1e-6) for x in chosen}
    total_inv = sum(inv_vol.values())
    if total_inv <= 1e-12:
        weights = {x["symbol"]: 1.0 / len(chosen) for x in chosen}
    else:
        weights = {sym: inv_vol[sym] / total_inv for sym in inv_vol}
    meta["selected_symbols"] = list(weights.keys())
    return weights, meta


def _signed_unit_scale(value: float, negative_ref: float, positive_ref: float) -> float:
    if value >= 0.0:
        return _clamp(float(value) / max(float(positive_ref), 1e-6), -1.0, 1.0)
    return _clamp(float(value) / max(float(abs(negative_ref)), 1e-6), -1.0, 1.0)


def _compute_defensive_allocation_ratio(
    benchmark_close: pd.Series,
    i: int,
    regime_state: str,
    trend_strength: Optional[float],
    breadth: float,
    phase2_state: str,
    defensive_cfg: dict,
) -> tuple[float, dict]:
    meta = {
        "enabled": bool(defensive_cfg.get("adaptive_enabled", False)),
        "regime_state": str(regime_state),
        "base_ratio": 0.0,
        "trend_term": 0.0,
        "breadth_term": 0.0,
        "drawdown_term": 0.0,
        "recovery_term": 0.0,
        "phase_term": 0.0,
        "drawdown_now": None,
        "recent_return": None,
        "target_ratio": 0.0,
    }
    if regime_state == "risk_off":
        meta["target_ratio"] = 1.0
        return 1.0, meta
    if (not meta["enabled"]) or benchmark_close is None or i < 4:
        return 0.0, meta

    ratio = float(defensive_cfg.get("adaptive_base_ratio", 0.0))
    if regime_state == "neutral":
        ratio += float(defensive_cfg.get("adaptive_neutral_add", 0.0))

    trend_term = 0.0
    if trend_strength is not None:
        trend_term = float(defensive_cfg.get("adaptive_trend_weight", 0.0)) * _signed_unit_scale(
            float(trend_strength),
            negative_ref=float(defensive_cfg.get("adaptive_trend_ref", 0.08)),
            positive_ref=float(defensive_cfg.get("adaptive_trend_ref", 0.08)),
        )
        ratio -= trend_term

    breadth_ref = float(defensive_cfg.get("adaptive_breadth_ref", 0.60))
    breadth_norm = _signed_unit_scale(
        float(breadth) - breadth_ref,
        negative_ref=breadth_ref,
        positive_ref=1.0 - breadth_ref,
    )
    breadth_term = float(defensive_cfg.get("adaptive_breadth_weight", 0.0)) * breadth_norm
    ratio -= breadth_term

    dd_window = int(defensive_cfg.get("adaptive_drawdown_window", 60))
    start = max(0, i - dd_window + 1)
    dd_slice = benchmark_close.iloc[start : i + 1]
    peak = float(dd_slice.max()) if len(dd_slice) > 0 else float(benchmark_close.iloc[i])
    drawdown_now = float(benchmark_close.iloc[i] / max(peak, 1e-8) - 1.0)
    drawdown_term = float(defensive_cfg.get("adaptive_drawdown_weight", 0.0)) * _clamp(
        abs(min(drawdown_now, 0.0)) / max(abs(float(defensive_cfg.get("adaptive_drawdown_ref", -0.08))), 1e-6),
        0.0,
        1.0,
    )
    ratio += drawdown_term

    recovery_term = 0.0
    rec_window = int(defensive_cfg.get("adaptive_recovery_window", 20))
    if i >= rec_window:
        recent_return = float(benchmark_close.iloc[i] / benchmark_close.iloc[i - rec_window] - 1.0)
        meta["recent_return"] = recent_return
        recovery_term = float(defensive_cfg.get("adaptive_recovery_weight", 0.0)) * _clamp(
            max(recent_return, 0.0) / max(float(defensive_cfg.get("adaptive_recovery_ref", 0.05)), 1e-6),
            0.0,
            1.0,
        )
        ratio -= recovery_term

    phase_term = 0.0
    if str(phase2_state) == "stress":
        phase_term = float(defensive_cfg.get("adaptive_stress_add", 0.0))
        ratio += phase_term
    elif str(phase2_state) == "calm":
        phase_term = float(defensive_cfg.get("adaptive_calm_add", 0.0))
        ratio += phase_term

    ratio = _clamp(
        float(ratio),
        float(defensive_cfg.get("adaptive_min_ratio", 0.0)),
        float(defensive_cfg.get("adaptive_max_ratio", 0.60)),
    )
    meta.update(
        {
            "base_ratio": float(defensive_cfg.get("adaptive_base_ratio", 0.0)),
            "trend_term": float(trend_term),
            "breadth_term": float(breadth_term),
            "drawdown_term": float(drawdown_term),
            "recovery_term": float(recovery_term),
            "phase_term": float(phase_term),
            "drawdown_now": float(drawdown_now),
            "target_ratio": float(ratio),
        }
    )
    return float(ratio), meta


def _resolve_dual_budget_cfg(params: dict) -> dict:
    cfg = params.get("dual_layer_budget", {}) if isinstance(params, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    total_mult_min = float(cfg.get("total_mult_min", 0.30))
    total_mult_max = float(cfg.get("total_mult_max", 1.35))
    if total_mult_max < total_mult_min:
        total_mult_max = total_mult_min
    return {
        "enabled": enabled,
        "risk_on_base_mult": float(cfg.get("risk_on_base_mult", 1.00)),
        "neutral_base_mult": float(cfg.get("neutral_base_mult", 0.80)),
        "risk_off_base_mult": float(cfg.get("risk_off_base_mult", 0.35)),
        "strong_trend_threshold": float(cfg.get("strong_trend_threshold", 0.08)),
        "weak_trend_threshold": float(cfg.get("weak_trend_threshold", 0.0)),
        "strong_trend_mult": float(cfg.get("strong_trend_mult", 1.10)),
        "weak_trend_mult": float(cfg.get("weak_trend_mult", 0.92)),
        "strong_breadth_threshold": float(cfg.get("strong_breadth_threshold", 0.72)),
        "weak_breadth_threshold": float(cfg.get("weak_breadth_threshold", 0.45)),
        "strong_breadth_mult": float(cfg.get("strong_breadth_mult", 1.08)),
        "weak_breadth_mult": float(cfg.get("weak_breadth_mult", 0.90)),
        "drawdown_window": max(20, int(cfg.get("drawdown_window", 60))),
        "mild_drawdown_trigger": float(cfg.get("mild_drawdown_trigger", -0.06)),
        "hard_drawdown_trigger": float(cfg.get("hard_drawdown_trigger", -0.10)),
        "mild_drawdown_mult": float(cfg.get("mild_drawdown_mult", 0.92)),
        "hard_drawdown_mult": float(cfg.get("hard_drawdown_mult", 0.72)),
        "recovery_window": max(5, int(cfg.get("recovery_window", 20))),
        "recovery_threshold": float(cfg.get("recovery_threshold", 0.05)),
        "recovery_mult": float(cfg.get("recovery_mult", 1.05)),
        "total_mult_min": total_mult_min,
        "total_mult_max": total_mult_max,
    }


def _compute_dual_budget_multiplier(
    benchmark_close: pd.Series,
    i: int,
    regime_state: str,
    trend_strength: Optional[float],
    breadth: float,
    cfg: dict,
) -> tuple[float, dict]:
    meta = {
        "enabled": bool(cfg.get("enabled", False)),
        "regime_state": str(regime_state),
        "base_mult": 1.0,
        "trend_mult": 1.0,
        "breadth_mult": 1.0,
        "drawdown_mult": 1.0,
        "recovery_mult": 1.0,
        "drawdown_now": None,
        "recent_return": None,
        "alloc_mult": 1.0,
    }
    if (not meta["enabled"]) or benchmark_close is None or i < 5:
        return 1.0, meta

    base_mult = float(cfg.get(f"{regime_state}_base_mult", 1.0))
    trend_mult = 1.0
    if trend_strength is not None:
        if trend_strength >= float(cfg["strong_trend_threshold"]):
            trend_mult = float(cfg["strong_trend_mult"])
        elif trend_strength <= float(cfg["weak_trend_threshold"]):
            trend_mult = float(cfg["weak_trend_mult"])

    breadth_mult = 1.0
    if breadth >= float(cfg["strong_breadth_threshold"]):
        breadth_mult = float(cfg["strong_breadth_mult"])
    elif breadth <= float(cfg["weak_breadth_threshold"]):
        breadth_mult = float(cfg["weak_breadth_mult"])

    dd_window = int(cfg["drawdown_window"])
    start = max(0, i - dd_window + 1)
    dd_slice = benchmark_close.iloc[start : i + 1]
    peak = float(dd_slice.max()) if len(dd_slice) > 0 else float(benchmark_close.iloc[i])
    drawdown_now = float(benchmark_close.iloc[i] / max(peak, 1e-8) - 1.0)
    drawdown_mult = 1.0
    if drawdown_now <= float(cfg["hard_drawdown_trigger"]):
        drawdown_mult = float(cfg["hard_drawdown_mult"])
    elif drawdown_now <= float(cfg["mild_drawdown_trigger"]):
        drawdown_mult = float(cfg["mild_drawdown_mult"])

    recovery_mult = 1.0
    rec_window = int(cfg["recovery_window"])
    if i >= rec_window:
        recent_return = float(benchmark_close.iloc[i] / benchmark_close.iloc[i - rec_window] - 1.0)
        meta["recent_return"] = recent_return
        if (recent_return >= float(cfg["recovery_threshold"])) and (regime_state == "risk_on"):
            recovery_mult = float(cfg["recovery_mult"])

    alloc_mult = _clamp(
        float(base_mult * trend_mult * breadth_mult * drawdown_mult * recovery_mult),
        float(cfg["total_mult_min"]),
        float(cfg["total_mult_max"]),
    )
    meta.update(
        {
            "base_mult": float(base_mult),
            "trend_mult": float(trend_mult),
            "breadth_mult": float(breadth_mult),
            "drawdown_mult": float(drawdown_mult),
            "recovery_mult": float(recovery_mult),
            "drawdown_now": float(drawdown_now),
            "alloc_mult": float(alloc_mult),
        }
    )
    return alloc_mult, meta


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
    obs_count: pd.DataFrame,
    i: int,
    symbols: list,
    benchmark_symbol: str,
    defensive_symbol: str,
    alloc_pct: float,
    single_max: float,
    params: dict,
) -> tuple[dict, bool, str, list]:
    benchmark_tradeable = bool(params.get("benchmark_tradeable", True))
    defensive_rotation_cfg = _resolve_defensive_rotation_cfg(params, defensive_symbol)
    dual_budget_cfg = _resolve_dual_budget_cfg(params)
    defensive_symbols = list(defensive_rotation_cfg.get("symbols", [defensive_symbol]))
    defensive_symbol_set = set(defensive_symbols)
    risk_symbols = [s for s in symbols if s not in defensive_symbol_set and (benchmark_tradeable or s != benchmark_symbol)]
    momentum_lb = int(params["momentum_lb"])
    ma_window = int(params["ma_window"])
    vol_window = int(params["vol_window"])
    top_n_base_cfg = int(params["top_n"])
    min_score_base = float(params.get("min_score", 0.0))
    score_mix_base = float(params.get("risk_on_score_mix", 0.0))
    score_floor_base = float(params.get("risk_on_score_floor", min_score_base))
    score_power_base = float(params.get("risk_on_score_power", 1.0))
    defensive_bypass_single_max = bool(params.get("defensive_bypass_single_max", True))
    symbol_weight_caps = _parse_symbol_weight_caps(params.get("symbol_weight_caps", {}), float(single_max))
    structural_cfg = _resolve_structural_cfg(params)
    structural_enabled = bool(structural_cfg["enabled"])
    defensive_weights, defensive_meta = _build_defensive_weights(prices, obs_count, i, defensive_rotation_cfg)

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
        obs = int(obs_count[s].iloc[i]) if s in obs_count.columns else 0
        if obs <= need:
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

    regime_profile = _resolve_regime_profile(
        params=params,
        regime_state=str(regime_state),
        base_top_n=top_n_base_cfg,
        base_min_score=min_score_base,
        base_score_mix=score_mix_base,
        base_score_floor=score_floor_base,
        base_score_power=score_power_base,
    )
    min_score_eff = float(regime_profile["min_score"])
    score_mix_eff = float(regime_profile["score_mix"])
    score_floor_eff = float(regime_profile["score_floor"])
    score_power_eff_base = float(regime_profile["score_power"])
    top_n_profile = int(regime_profile["top_n"])
    regime_alloc_mult = float(regime_profile["alloc_mult"])
    trend_strength_now = None
    if benchmark_ma is not None and abs(float(benchmark_ma)) > 1e-12:
        trend_strength_now = float(prices[benchmark_symbol].iloc[i] / benchmark_ma - 1.0)
    turbo_triggered = False
    if (
        regime_state == "risk_on"
        and bool(regime_profile.get("turbo_enabled", False))
        and (trend_strength_now is not None)
        and (trend_strength_now >= float(regime_profile.get("turbo_trend_strength_min", 0.03)))
        and (breadth >= float(regime_profile.get("turbo_breadth_min", 0.80)))
    ):
        turbo_triggered = True
        regime_alloc_mult += float(regime_profile.get("turbo_alloc_mult_add", 0.0))
        top_n_profile += int(regime_profile.get("turbo_top_n_add", 0))
    top_n_profile = max(1, int(top_n_profile))
    regime_alloc_mult = max(0.0, float(regime_alloc_mult))

    regime_on = bool(regime_state != "risk_off")
    if structural_enabled:
        eligible = [
            s
            for s in scores
            if scores[s]["score"] > min_score_eff
            and ((not structural_cfg["eligibility_require_trend"]) or scores[s]["trend_ok"])
            and scores[s]["composite_momentum"] > 0.0
        ]
    else:
        eligible = [s for s in scores if scores[s]["score"] > min_score_eff]

    target = {s: 0.0 for s in symbols}
    reason = "risk_off"
    risk_weights = {}
    risk_governor_meta = {
        "enabled": bool(params.get("risk_governor", {}).get("enabled", False)) if isinstance(params, dict) else False,
        "alloc_mult": 1.0,
        "dual_budget_enabled": bool(dual_budget_cfg.get("enabled", False)),
        "dual_budget_alloc_mult": 1.0,
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
        "symbol_weight_caps": symbol_weight_caps,
        "regime_profile_enabled": bool(regime_profile["enabled"]),
        "regime_profile_name": str(regime_profile["regime"]),
        "regime_profile_top_n": int(top_n_profile),
        "regime_profile_min_score": float(min_score_eff),
        "regime_profile_score_mix": float(score_mix_eff),
        "regime_profile_score_floor": float(score_floor_eff),
        "regime_profile_score_power": float(score_power_eff_base),
        "regime_profile_alloc_mult": float(regime_alloc_mult),
        "regime_profile_turbo_enabled": bool(regime_profile.get("turbo_enabled", False)),
        "regime_profile_turbo_triggered": bool(turbo_triggered),
        "regime_profile_turbo_trend_strength_min": float(regime_profile.get("turbo_trend_strength_min", 0.03)),
        "regime_profile_turbo_breadth_min": float(regime_profile.get("turbo_breadth_min", 0.80)),
        "regime_profile_turbo_alloc_mult_add": float(regime_profile.get("turbo_alloc_mult_add", 0.05)),
        "regime_profile_turbo_top_n_add": int(regime_profile.get("turbo_top_n_add", 1)),
        "score_dispersion": 0.0,
        "top_n_effective": int(top_n_profile),
        "score_power_effective": float(score_power_eff_base),
        "defensive_rotation": defensive_meta,
    }
    target_total_alloc = float(alloc_pct)
    if regime_state in {"risk_on", "neutral"} and eligible:
        risk_alloc_base = float(alloc_pct)
        if regime_state == "neutral":
            risk_alloc_base *= float(structural_cfg["neutral_alloc_mult"])
        risk_alloc_base *= float(regime_alloc_mult)

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

        top_n_base = int(top_n_profile if regime_state == "risk_on" else min(top_n_profile, structural_cfg["neutral_top_n"]))
        top_n_eff = max(1, top_n_base)
        score_power_eff = float(score_power_eff_base)
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

        dual_alloc_mult, dual_meta = _compute_dual_budget_multiplier(
            benchmark_close=benchmark_close,
            i=i,
            regime_state=regime_state,
            trend_strength=trend_strength_now,
            breadth=breadth,
            cfg=dual_budget_cfg,
        )
        risk_governor_meta["dual_budget"] = dual_meta
        risk_governor_meta["dual_budget_alloc_mult"] = float(dual_alloc_mult)
        target_total_alloc = float(risk_alloc_pre_rg * alloc_mult * dual_alloc_mult)
        defensive_ratio, defensive_alloc_meta = _compute_defensive_allocation_ratio(
            benchmark_close=benchmark_close,
            i=i,
            regime_state=regime_state,
            trend_strength=trend_strength_now,
            breadth=breadth,
            phase2_state=phase2_vol_state,
            defensive_cfg=defensive_rotation_cfg,
        )
        risk_governor_meta["defensive_rotation"]["adaptive_allocation"] = defensive_alloc_meta
        risk_governor_meta["defensive_rotation"]["target_ratio"] = float(defensive_ratio)
        risk_weights = build_risk_on_weights(
            scores=scores,
            picks=picks,
            score_mix=score_mix_eff,
            score_floor=score_floor_eff,
            score_power=score_power_eff,
        )
        reason = "risk_on" if regime_state == "risk_on" else "risk_on_neutral"
    else:
        dual_alloc_mult, dual_meta = _compute_dual_budget_multiplier(
            benchmark_close=benchmark_close,
            i=i,
            regime_state=regime_state,
            trend_strength=trend_strength_now,
            breadth=breadth,
            cfg=dual_budget_cfg,
        )
        risk_governor_meta["dual_budget"] = dual_meta
        risk_governor_meta["dual_budget_alloc_mult"] = float(dual_alloc_mult)
        target_total_alloc = float(min(alloc_pct, max(0.0, alloc_pct * dual_alloc_mult)))
        risk_governor_meta["defensive_rotation"]["adaptive_allocation"] = {
            "enabled": bool(defensive_rotation_cfg.get("adaptive_enabled", False)),
            "regime_state": str(regime_state),
            "target_ratio": 1.0,
        }
        risk_governor_meta["defensive_rotation"]["target_ratio"] = 1.0
        if not benchmark_trend_on:
            reason = "benchmark_below_ma"
        elif phase2_stepdown_applied and regime_state == "risk_off":
            reason = "phase2_stress_risk_off"
        elif regime_state == "risk_off":
            reason = "breadth_risk_off"
        elif not eligible:
            reason = "no_positive_momentum"

    target_total_alloc = float(min(alloc_pct, max(0.0, target_total_alloc)))
    if regime_state in {"risk_on", "neutral"} and eligible:
        defensive_target_alloc = float(
            max(0.0, target_total_alloc * float(risk_governor_meta["defensive_rotation"].get("target_ratio", 0.0)))
        )
        risk_target_alloc = float(max(0.0, target_total_alloc - defensive_target_alloc))
        raw_targets = {s: risk_target_alloc * weight for s, weight in risk_weights.items()}
        if defensive_target_alloc > 1e-8:
            defensive_fill = defensive_weights if defensive_weights else {defensive_symbol: 1.0}
            for sym, weight in defensive_fill.items():
                raw_targets[sym] = raw_targets.get(sym, 0.0) + defensive_target_alloc * weight
    else:
        defensive_fill = defensive_weights if defensive_weights else {defensive_symbol: 1.0}
        raw_targets = {sym: float(target_total_alloc * weight) for sym, weight in defensive_fill.items()}

    capped_targets = {}
    for sym, weight in raw_targets.items():
        if (not defensive_bypass_single_max) or sym not in defensive_symbol_set:
            capped_targets[sym] = min(float(weight), _cap_for_symbol(sym, float(single_max), symbol_weight_caps))
        else:
            capped_targets[sym] = float(weight)

    used = sum(capped_targets.values())
    remain = max(0.0, target_total_alloc - used)
    if remain > 1e-8:
        defensive_fill = defensive_weights if defensive_weights else {defensive_symbol: 1.0}
        for sym, weight in defensive_fill.items():
            add_w = float(remain * weight)
            if sym not in capped_targets:
                capped_targets[sym] = 0.0
            if (not defensive_bypass_single_max) or sym not in defensive_symbol_set:
                defensive_cap = _cap_for_symbol(sym, float(single_max), symbol_weight_caps)
                room = max(0.0, defensive_cap - capped_targets[sym])
                capped_targets[sym] += min(room, add_w)
            else:
                capped_targets[sym] += add_w
    target.update(capped_targets)

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
    risk_governor_meta["alloc_target"] = float(target_total_alloc)
    return target, regime_on, reason, score_rows, risk_governor_meta


def run_paper_forward(runtime, stock, risk):
    model_cfg = stock.get("global_model", {})
    guard_cfg = model_cfg.get("execution_guard", {})
    overlay_cfg = model_cfg.get("risk_overlay", {})
    exposure_cfg = model_cfg.get("exposure_gate", {})

    benchmark_symbol = stock.get("benchmark_symbol", "510300")
    defensive_symbol = stock.get("defensive_symbol", "511010")
    defensive_rotation_cfg = _resolve_defensive_rotation_cfg(model_cfg, defensive_symbol)
    symbols = sorted(
        set(stock.get("universe", []) + [benchmark_symbol, defensive_symbol] + list(defensive_rotation_cfg.get("symbols", [])))
    )
    alloc_pct = float(stock.get("capital_alloc_pct", 0.7))
    single_max = float(risk["position_limits"]["stock_single_max_pct"])
    symbol_weight_caps = _parse_symbol_weight_caps(model_cfg.get("symbol_weight_caps", {}), float(single_max))

    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    report_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(report_dir)

    prices, availability = load_price_frame(data_dir, symbols)
    obs_count = availability.cumsum()
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
    required_hist_days = max(
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
    )
    start_scan = required_hist_days + 1
    if start_scan >= len(prices) - 1:
        raise RuntimeError("not enough history for paper-forward simulation")

    start_idx = None
    for i in range(start_scan, len(prices) - 1):
        bench_obs = int(obs_count[benchmark_symbol].iloc[i])
        def_obs = int(obs_count[defensive_symbol].iloc[i])
        if bench_obs <= required_hist_days or def_obs <= required_hist_days:
            continue
        if not bool(availability[benchmark_symbol].iloc[i]) or not bool(availability[benchmark_symbol].iloc[i + 1]):
            continue
        if not bool(availability[defensive_symbol].iloc[i]) or not bool(availability[defensive_symbol].iloc[i + 1]):
            continue
        start_idx = i
        break
    if start_idx is None:
        raise RuntimeError("not enough benchmark/defensive history for paper-forward simulation")

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
    for i in range(start_idx, len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        if not bool(availability[benchmark_symbol].iloc[i]) or not bool(availability[benchmark_symbol].iloc[i + 1]):
            continue
        alloc_effective, exposure_meta = evaluate_exposure_gate(
            hist_strategy_ret,
            hist_benchmark_ret_alloc,
            alloc_pct,
            exposure_cfg,
        )
        proposed, regime_on, signal_reason, score_rows, risk_governor_meta = build_signal(
            prices=prices,
            obs_count=obs_count,
            i=i,
            symbols=symbols,
            benchmark_symbol=benchmark_symbol,
            defensive_symbol=defensive_symbol,
            alloc_pct=alloc_effective,
            single_max=single_max,
            params=model_cfg,
        )
        alloc_effective_final = float(risk_governor_meta.get("alloc_target", alloc_effective))
        min_turnover, min_turnover_meta = resolve_min_turnover(min_turnover_base, alloc_effective_final, guard_cfg)
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
                    tw = min(tw, _cap_for_symbol(defensive_symbol, float(single_max), symbol_weight_caps))
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
                "alloc_pct_effective": float(alloc_effective_final),
                "alloc_pct_after_exposure_gate": float(alloc_effective),
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

    if not records:
        raise RuntimeError("no tradable periods after history/availability filters")

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
