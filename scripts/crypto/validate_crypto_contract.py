#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.crypto_model import resolve_crypto_model_cfg


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def summarize_returns(rets, periods_per_year):
    if len(rets) < 2:
        return None
    arr = np.array(rets, dtype=float)
    nav = np.cumprod(np.concatenate([[1.0], 1.0 + arr]))
    ann = float(nav[-1] ** (periods_per_year / len(arr)) - 1.0)
    mdd = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
    sharpe = float((arr.mean() / arr.std()) * math.sqrt(periods_per_year)) if arr.std() > 1e-12 else 0.0
    return {
        "periods": int(len(arr)),
        "annual_return": ann,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "final_nav": float(nav[-1]),
    }


def load_frames(runtime, symbols):
    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    out = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if ("date" not in df.columns) or ("close" not in df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"])
        out[s] = df.sort_values("date").set_index("date")
    return out


def run_contract_backtest(runtime, crypto_cfg, date_from=None, slip_bps=0.0, funding_bps_per_bar=0.0, fee=None):
    model_meta = resolve_crypto_model_cfg(crypto_cfg)
    signal_cfg = model_meta["signal"]
    defense_cfg = model_meta["defense"]

    symbols = crypto_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])
    frames = load_frames(runtime, symbols)
    if len(frames) < max(1, int(signal_cfg.get("top_n", 1))):
        return {"error": "crypto data not enough"}

    dates = sorted(set.intersection(*[set(f.index) for f in frames.values()]))
    if date_from is not None:
        cutoff = pd.Timestamp(date_from)
        dates = [d for d in dates if d >= cutoff]

    lbs = signal_cfg.get("momentum_lookback_bars", [90, 180])
    ma_window = int(signal_cfg.get("ma_window_bars", 180))
    vol_window = int(signal_cfg.get("vol_window_bars", 20))
    warmup = max(max(lbs), ma_window, vol_window) + 2
    if len(dates) <= warmup + 20:
        return {"error": f"history too short: {len(dates)}"}

    ws = signal_cfg.get("momentum_weights", [1.0, 0.0])
    top_n = int(signal_cfg.get("top_n", 1))
    short_top_n = int(signal_cfg.get("short_top_n", top_n))
    engine = str(signal_cfg.get("engine", "advanced_rmm")).strip().lower()
    is_advanced_engine = engine in {"advanced_rmm", "advanced_ls_rmm", "advanced_ls_cs"}
    momentum_threshold = float(signal_cfg.get("momentum_threshold_pct", 0.0)) / 100.0
    threshold = float(defense_cfg.get("risk_off_threshold_pct", -2.0)) / 100.0
    alloc_pct = float(crypto_cfg.get("capital_alloc_pct", 0.3))
    rebalance_threshold = float(signal_cfg.get("rebalance_threshold", 0.02))
    periods_per_year = int(signal_cfg.get("periods_per_year", 6 * 365))
    trading_fee = float(signal_cfg.get("fee", 0.001) if fee is None else fee)

    trade_cfg = crypto_cfg.get("trade", {}) or {}
    allow_short = bool(trade_cfg.get("allow_short", False))
    short_on_risk_off = bool(signal_cfg.get("short_on_risk_off", False))
    ls_long_alloc_pct = float(signal_cfg.get("ls_long_alloc_pct", alloc_pct))
    ls_short_alloc_pct = float(signal_cfg.get("ls_short_alloc_pct", alloc_pct))
    ls_short_requires_risk_off = bool(signal_cfg.get("ls_short_requires_risk_off", False))
    ls_dynamic_budget_enabled = bool(signal_cfg.get("ls_dynamic_budget_enabled", False))
    ls_regime_momentum_threshold = float(
        signal_cfg.get(
            "ls_regime_momentum_threshold_pct",
            signal_cfg.get("momentum_threshold_pct", 0.0),
        )
    ) / 100.0
    ls_regime_trend_breadth = float(signal_cfg.get("ls_regime_trend_breadth", 0.6))
    ls_regime_up_long_alloc_pct = float(signal_cfg.get("ls_regime_up_long_alloc_pct", ls_long_alloc_pct))
    ls_regime_up_short_alloc_pct = float(signal_cfg.get("ls_regime_up_short_alloc_pct", ls_short_alloc_pct))
    ls_regime_neutral_long_alloc_pct = float(
        signal_cfg.get("ls_regime_neutral_long_alloc_pct", ls_long_alloc_pct)
    )
    ls_regime_neutral_short_alloc_pct = float(
        signal_cfg.get("ls_regime_neutral_short_alloc_pct", ls_short_alloc_pct)
    )
    ls_neutral_exposure_multiplier = float(signal_cfg.get("ls_neutral_exposure_multiplier", 1.0))
    ls_regime_down_long_alloc_pct = float(signal_cfg.get("ls_regime_down_long_alloc_pct", ls_long_alloc_pct))
    ls_regime_down_short_alloc_pct = float(signal_cfg.get("ls_regime_down_short_alloc_pct", ls_short_alloc_pct))
    ls_confidence_scaling_enabled = bool(signal_cfg.get("ls_confidence_scaling_enabled", False))
    ls_confidence_min_spread = float(signal_cfg.get("ls_confidence_min_spread", 0.12))
    ls_confidence_full_spread = float(
        signal_cfg.get(
            "ls_confidence_full_spread",
            max(ls_confidence_min_spread + 1e-6, 0.32),
        )
    )
    max_exposure_pct = float(
        signal_cfg.get(
            "max_exposure_pct",
            alloc_pct * max(1.0, float(trade_cfg.get("leverage", 1.0))),
        )
    )
    single_cap = float(signal_cfg.get("single_max_pct", max_exposure_pct))

    rm_cfg = signal_cfg.get("risk_managed", {}) or {}
    rm_enabled = bool(rm_cfg.get("enabled", False))
    rm_target_vol = float(rm_cfg.get("target_vol_annual", 0.0))
    rm_regime_target_vol_enabled = bool(rm_cfg.get("regime_target_vol_enabled", False))
    rm_target_vol_up = float(rm_cfg.get("target_vol_up_annual", rm_target_vol))
    rm_target_vol_neutral = float(rm_cfg.get("target_vol_neutral_annual", rm_target_vol))
    rm_target_vol_down = float(rm_cfg.get("target_vol_down_annual", rm_target_vol))
    rm_vol_lb = int(rm_cfg.get("vol_lookback_bars", vol_window))
    rm_lev_min = float(rm_cfg.get("min_leverage", 0.2))
    rm_lev_max = float(rm_cfg.get("max_leverage", 3.0))

    dd_cfg = signal_cfg.get("drawdown_throttle", {}) or {}
    dd_enabled = bool(dd_cfg.get("enabled", False))
    dd_trigger = float(dd_cfg.get("trigger_dd", -0.08))
    dd_reduced_mult = float(dd_cfg.get("reduced_alloc_multiplier", 0.0))

    weights = {s: 0.0 for s in frames.keys()}
    nav = [1.0]
    rets = []
    lev_hist = []

    for i in range(warmup, len(dates) - 1):
        dt, nxt = dates[i], dates[i + 1]

        raw = {}
        for s, df in frames.items():
            close = pd.to_numeric(df.loc[:dt, "close"], errors="coerce").dropna()
            if len(close) <= warmup:
                continue
            m = float(close.iloc[-1] / close.iloc[-int(lbs[0])] - 1.0) * float(ws[0])
            m += float(close.iloc[-1] / close.iloc[-int(lbs[1])] - 1.0) * float(ws[1])

            lr = np.log(close.astype(float)).diff().dropna().tail(vol_window)
            if lr.empty:
                continue
            vol_ann = float(lr.std() * np.sqrt(periods_per_year))
            if vol_ann <= 0:
                continue

            trend_on = bool(close.iloc[-1] >= close.tail(ma_window).mean()) if len(close) >= ma_window else False
            sret = float(close.iloc[-1] / close.iloc[-int(lbs[0])] - 1.0)
            raw[s] = {
                "momentum": m,
                "vol_ann": vol_ann,
                "trend_on": trend_on,
                "short_ret": sret,
            }

        target = {s: 0.0 for s in frames.keys()}
        lev_now = 1.0
        if len(raw) >= top_n:
            score = {s: raw[s]["momentum"] / max(raw[s]["vol_ann"], 1e-8) for s in raw.keys()}
            ranked_desc = [x[0] for x in sorted(score.items(), key=lambda x: x[1], reverse=True)]
            ranked_asc = [x[0] for x in sorted(score.items(), key=lambda x: x[1])]
            if engine == "advanced_ls_cs":
                long_candidates = [
                    s for s in ranked_desc if raw[s]["momentum"] > momentum_threshold
                ][: max(1, top_n)]
                short_candidates = [
                    s for s in ranked_asc if raw[s]["momentum"] < -momentum_threshold
                ][: max(1, short_top_n)]
                if not long_candidates:
                    long_candidates = ranked_desc[: max(1, top_n)]
                if not short_candidates:
                    short_candidates = ranked_asc[: max(1, short_top_n)]
            else:
                long_candidates = [
                    s for s in ranked_desc if raw[s]["trend_on"] and raw[s]["momentum"] > momentum_threshold
                ][: max(1, top_n)]
                short_candidates = [
                    s for s in ranked_asc if (not raw[s]["trend_on"]) and raw[s]["momentum"] < -momentum_threshold
                ][: max(1, short_top_n)]

            target_map = {}
            risk_off = all(v["short_ret"] < threshold for v in raw.values())
            avg_momentum = float(np.mean([v["momentum"] for v in raw.values()])) if raw else 0.0
            trend_breadth = float(np.mean([1.0 if v["trend_on"] else 0.0 for v in raw.values()])) if raw else 0.0
            down_breadth = 1.0 - trend_breadth
            regime_state = "neutral"
            if (avg_momentum <= -ls_regime_momentum_threshold) and (down_breadth >= ls_regime_trend_breadth):
                regime_state = "risk_off"
            elif (avg_momentum >= ls_regime_momentum_threshold) and (trend_breadth >= ls_regime_trend_breadth):
                regime_state = "risk_on"
            if is_advanced_engine and engine in {"advanced_ls_rmm", "advanced_ls_cs"}:
                long_budget = max(0.0, ls_long_alloc_pct)
                short_budget = max(0.0, ls_short_alloc_pct)
                if ls_dynamic_budget_enabled:
                    if regime_state == "risk_on":
                        long_budget = max(0.0, ls_regime_up_long_alloc_pct)
                        short_budget = max(0.0, ls_regime_up_short_alloc_pct)
                    elif regime_state == "risk_off":
                        long_budget = max(0.0, ls_regime_down_long_alloc_pct)
                        short_budget = max(0.0, ls_regime_down_short_alloc_pct)
                    else:
                        long_budget = max(0.0, ls_regime_neutral_long_alloc_pct)
                        short_budget = max(0.0, ls_regime_neutral_short_alloc_pct)
                        neutral_mult = max(0.0, min(1.5, ls_neutral_exposure_multiplier))
                        long_budget *= neutral_mult
                        short_budget *= neutral_mult
                if long_candidates and long_budget > 0:
                    inv = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in long_candidates}
                    inv_sum = sum(inv.values())
                    if inv_sum > 0:
                        for s in long_candidates:
                            target_map[s] = target_map.get(s, 0.0) + min(long_budget * (inv[s] / inv_sum), single_cap)

                short_gate = (regime_state == "risk_off") if ls_dynamic_budget_enabled else risk_off
                short_enabled = (
                    allow_short
                    and short_candidates
                    and ((not ls_short_requires_risk_off) or short_gate)
                    and short_budget > 0
                )
                if short_enabled:
                    inv = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in short_candidates}
                    inv_sum = sum(inv.values())
                    if inv_sum > 0:
                        for s in short_candidates:
                            target_map[s] = target_map.get(s, 0.0) - min(short_budget * (inv[s] / inv_sum), single_cap)
                if ls_confidence_scaling_enabled and target_map:
                    spread = float(max(score.values()) - min(score.values())) if score else 0.0
                    hi = max(ls_confidence_full_spread, ls_confidence_min_spread + 1e-8)
                    if spread <= ls_confidence_min_spread:
                        confidence = 0.0
                    elif spread >= hi:
                        confidence = 1.0
                    else:
                        confidence = (spread - ls_confidence_min_spread) / (hi - ls_confidence_min_spread)
                    target_map = {s: w * confidence for s, w in target_map.items()}
            else:
                if (not risk_off) and long_candidates:
                    inv = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in long_candidates}
                    inv_sum = sum(inv.values())
                    if inv_sum > 0:
                        for s in long_candidates:
                            target_map[s] = min(alloc_pct * (inv[s] / inv_sum), single_cap)
                elif allow_short and short_on_risk_off and short_candidates:
                    inv = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in short_candidates}
                    inv_sum = sum(inv.values())
                    if inv_sum > 0:
                        for s in short_candidates:
                            target_map[s] = -min(alloc_pct * (inv[s] / inv_sum), single_cap)

            if dd_enabled and target_map:
                peak = max(nav)
                curr_dd = nav[-1] / peak - 1.0 if peak > 0 else 0.0
                if curr_dd <= dd_trigger:
                    target_map = {s: w * dd_reduced_mult for s, w in target_map.items()}

            if rm_enabled and target_map and len(rets) >= rm_vol_lb:
                rv = float(np.std(rets[-rm_vol_lb:]) * np.sqrt(periods_per_year))
                rm_target_vol_eff = rm_target_vol
                if rm_regime_target_vol_enabled:
                    if regime_state == "risk_on":
                        rm_target_vol_eff = rm_target_vol_up
                    elif regime_state == "risk_off":
                        rm_target_vol_eff = rm_target_vol_down
                    else:
                        rm_target_vol_eff = rm_target_vol_neutral
                if rv > 1e-8 and rm_target_vol_eff > 0:
                    lev_now = rm_target_vol_eff / rv
                    lev_now = max(rm_lev_min, min(rm_lev_max, lev_now))
                    target_map = {s: w * lev_now for s, w in target_map.items()}

            gross = float(sum(abs(w) for w in target_map.values()))
            if gross > max_exposure_pct and gross > 1e-8:
                scale = max_exposure_pct / gross
                target_map = {s: w * scale for s, w in target_map.items()}

            for s in target.keys():
                target[s] = float(target_map.get(s, 0.0))

        turnover = float(sum(abs(target[s] - weights.get(s, 0.0)) for s in target.keys()))
        if turnover < rebalance_threshold:
            target = weights.copy()
            turnover = 0.0

        gross_exposure = float(sum(abs(v) for v in target.values()))
        bar_ret = 0.0
        for s, w in target.items():
            if abs(w) < 1e-12:
                continue
            c0 = float(frames[s].loc[dt, "close"])
            c1 = float(frames[s].loc[nxt, "close"])
            bar_ret += w * (c1 / c0 - 1.0)

        trade_cost = turnover * (trading_fee + float(slip_bps) / 10000.0)
        funding_cost = gross_exposure * (float(funding_bps_per_bar) / 10000.0)
        bar_ret -= trade_cost + funding_cost

        weights = target
        nav.append(nav[-1] * (1.0 + bar_ret))
        rets.append(bar_ret)
        lev_hist.append(lev_now)

    out = summarize_returns(rets, periods_per_year)
    if out is None:
        return {"error": "not enough returns"}
    out["avg_leverage_multiplier"] = float(np.mean(lev_hist)) if lev_hist else 1.0
    out["max_leverage_multiplier"] = float(np.max(lev_hist)) if lev_hist else 1.0
    out["ls_dynamic_budget_enabled"] = bool(ls_dynamic_budget_enabled)
    out["date_from"] = None if date_from is None else str(pd.Timestamp(date_from).date())
    out["slippage_bps"] = float(slip_bps)
    out["funding_bps_per_bar"] = float(funding_bps_per_bar)
    return out


def main():
    runtime = load_yaml("config/runtime.yaml")
    crypto = load_yaml("config/crypto.yaml")

    result = {
        "ts": datetime.now().isoformat(),
        "market": "crypto",
        "mode": "contract_validation",
        "model_meta": resolve_crypto_model_cfg(crypto),
    }

    baseline = run_contract_backtest(runtime, crypto, date_from=None, slip_bps=0.0, funding_bps_per_bar=0.0)
    result["full_sample"] = baseline

    frames = load_frames(runtime, crypto.get("symbols", ["BTC/USDT", "ETH/USDT"]))
    dates = sorted(set.intersection(*[set(f.index) for f in frames.values()])) if frames else []
    if dates:
        cut_70 = dates[int(len(dates) * 0.7)]
        cut_80 = dates[int(len(dates) * 0.8)]
        result["oos"] = {
            "oos_30pct": run_contract_backtest(runtime, crypto, date_from=cut_70),
            "oos_20pct": run_contract_backtest(runtime, crypto, date_from=cut_80),
        }
        rolling = []
        for ratio in [0.55, 0.65, 0.75, 0.85]:
            idx = int(len(dates) * ratio)
            if idx >= len(dates) - 100:
                continue
            st = dates[idx]
            m = run_contract_backtest(runtime, crypto, date_from=st)
            if isinstance(m, dict):
                m["window_ratio"] = float(ratio)
                m["window_from"] = str(pd.Timestamp(st).date())
                rolling.append(m)
        result["rolling_oos"] = rolling
    else:
        result["oos"] = {"error": "no dates"}
        result["rolling_oos"] = []

    stress_cases = [
        {"name": "base", "slippage_bps": 0.0, "funding_bps_per_bar": 0.0},
        {"name": "stress_mild", "slippage_bps": 3.0, "funding_bps_per_bar": 1.0},
        {"name": "stress_medium", "slippage_bps": 5.0, "funding_bps_per_bar": 2.0},
        {"name": "stress_harsh", "slippage_bps": 10.0, "funding_bps_per_bar": 3.0},
    ]
    stress = []
    for sc in stress_cases:
        out = run_contract_backtest(
            runtime,
            crypto,
            date_from=None,
            slip_bps=sc["slippage_bps"],
            funding_bps_per_bar=sc["funding_bps_per_bar"],
        )
        out["name"] = sc["name"]
        stress.append(out)
    result["stress"] = stress

    def pass_gate(m, ann_min, mdd_min):
        return (
            isinstance(m, dict)
            and ("annual_return" in m)
            and ("max_drawdown" in m)
            and (float(m["annual_return"]) >= ann_min)
            and (float(m["max_drawdown"]) >= mdd_min)
        )

    o70 = ((result.get("oos", {}) or {}).get("oos_30pct")) or {}
    o80 = ((result.get("oos", {}) or {}).get("oos_20pct")) or {}
    stress_mild = next((x for x in stress if x.get("name") == "stress_mild"), {})
    stress_metrics = [x for x in stress if isinstance(x, dict)]
    rolling_metrics = [x for x in (result.get("rolling_oos") or []) if isinstance(x, dict)]

    legacy_target = {"annual_return_min": 0.20, "max_drawdown_min": -0.10}
    oos_focus_target = {
        "full_annual_return_min": 0.08,
        "oos_annual_return_min": 0.00,
        "full_max_drawdown_min": -0.10,
        "oos_max_drawdown_min": -0.08,
        "stress_mild_annual_return_min": 0.00,
        "stress_mild_max_drawdown_min": -0.12,
    }

    result["gates"] = {
        "legacy_target": legacy_target,
        "legacy_full_pass": pass_gate(
            result["full_sample"], legacy_target["annual_return_min"], legacy_target["max_drawdown_min"]
        ),
        "legacy_oos_all_pass": pass_gate(o70, legacy_target["annual_return_min"], legacy_target["max_drawdown_min"])
        and pass_gate(o80, legacy_target["annual_return_min"], legacy_target["max_drawdown_min"]),
        "legacy_stress_all_pass": bool(stress_metrics)
        and all(pass_gate(x, legacy_target["annual_return_min"], legacy_target["max_drawdown_min"]) for x in stress_metrics),
        "oos_focus_target": oos_focus_target,
        "oos_focus_pass": (
            pass_gate(
                result["full_sample"],
                oos_focus_target["full_annual_return_min"],
                oos_focus_target["full_max_drawdown_min"],
            )
            and pass_gate(
                o70,
                oos_focus_target["oos_annual_return_min"],
                oos_focus_target["oos_max_drawdown_min"],
            )
            and pass_gate(
                o80,
                oos_focus_target["oos_annual_return_min"],
                oos_focus_target["oos_max_drawdown_min"],
            )
            and pass_gate(
                stress_mild,
                oos_focus_target["stress_mild_annual_return_min"],
                oos_focus_target["stress_mild_max_drawdown_min"],
            )
        ),
        "rolling_oos_all_pass": bool(rolling_metrics)
        and all(
            pass_gate(
                x,
                oos_focus_target["oos_annual_return_min"],
                oos_focus_target["oos_max_drawdown_min"],
            )
            for x in rolling_metrics
        ),
    }

    out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, "crypto_contract_validation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[crypto-validate] report -> {out_file}")


if __name__ == "__main__":
    main()
