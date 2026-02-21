#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_OUTPUT_ERROR, EXIT_SIGNAL_ERROR
from core.crypto_model import resolve_crypto_model_cfg
from core.signal import momentum_score, volatility_score, max_drawdown_score, liquidity_score, normalize_rank
from scripts.crypto.validate_crypto_contract import run_contract_backtest


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

def backtest_crypto(runtime, crypto_cfg):
    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    symbols = crypto_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])
    alloc_pct = float(crypto_cfg.get("capital_alloc_pct", 0.3))
    model_meta = resolve_crypto_model_cfg(crypto_cfg)
    signal_cfg = model_meta["signal"]
    defense_cfg = model_meta["defense"]
    market_type = model_meta["market_type"]
    contract_mode = model_meta["contract_mode"]
    engine = str(signal_cfg.get("engine", "multifactor_v1")).strip().lower()
    if contract_mode and engine in {"advanced_rmm", "advanced_ls_rmm", "advanced_ls_cs"}:
        res = run_contract_backtest(runtime, crypto_cfg)
        if "error" in res:
            return res
        res.update(
            {
                "model_profile": model_meta["profile_key"],
                "model_name": model_meta["profile_name"],
                "market_type": market_type,
                "contract_mode": contract_mode,
                "engine": engine,
                "allow_short": bool((crypto_cfg.get("trade", {}) or {}).get("allow_short", False)),
                "short_on_risk_off": bool(signal_cfg.get("short_on_risk_off", False)),
                "ls_dynamic_budget_enabled": bool(signal_cfg.get("ls_dynamic_budget_enabled", False)),
                "ls_long_alloc_pct": float(signal_cfg.get("ls_long_alloc_pct", crypto_cfg.get("capital_alloc_pct", 0.3))),
                "ls_short_alloc_pct": float(signal_cfg.get("ls_short_alloc_pct", crypto_cfg.get("capital_alloc_pct", 0.3))),
                "ls_short_requires_risk_off": bool(signal_cfg.get("ls_short_requires_risk_off", False)),
            }
        )
        return res

    top_n = signal_cfg.get("top_n", 1)
    short_top_n = int(signal_cfg.get("short_top_n", top_n))
    short_on_risk_off = bool(signal_cfg.get("short_on_risk_off", False))
    is_advanced_engine = engine in {"advanced_rmm", "advanced_ls_rmm", "advanced_ls_cs"}
    lbs = signal_cfg.get("momentum_lookback_bars", [30, 90])
    ws = signal_cfg.get("momentum_weights", [0.6, 0.4])
    fw = signal_cfg.get("factor_weights", {"momentum": 0.60, "low_vol": 0.25, "drawdown": 0.15})
    threshold = float(defense_cfg.get("risk_off_threshold_pct", -3.0)) / 100.0
    trade_cfg = crypto_cfg.get("trade", {}) or {}
    allow_short = bool(trade_cfg.get("allow_short", False))
    ma_window = int(signal_cfg.get("ma_window_bars", 180))
    vol_window = int(signal_cfg.get("vol_window_bars", 20))
    momentum_threshold = float(signal_cfg.get("momentum_threshold_pct", 0.0)) / 100.0
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
    ppy = int(signal_cfg.get("periods_per_year", 6 * 365))

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
    if len(dates) < max(max(lbs), ma_window, vol_window) + 120:
        return {"error": "crypto history too short"}

    fee = float(signal_cfg.get("fee", 0.001))
    rebalance_threshold = float(signal_cfg.get("rebalance_threshold", 0.08))
    max_exposure_pct = float(
        signal_cfg.get(
            "max_exposure_pct",
            alloc_pct * max(1.0, float(trade_cfg.get("leverage", 1.0))) if contract_mode else alloc_pct,
        )
    )
    single_cap = float(signal_cfg.get("single_max_pct", max_exposure_pct))

    # 风险管理：目标波动 + 动态杠杆
    rm_cfg = signal_cfg.get("risk_managed", {}) or {}
    rm_enabled = bool(rm_cfg.get("enabled", False))
    rm_target_vol = float(rm_cfg.get("target_vol_annual", 0.0))
    rm_regime_target_vol_enabled = bool(rm_cfg.get("regime_target_vol_enabled", False))
    rm_target_vol_up = float(rm_cfg.get("target_vol_up_annual", rm_target_vol))
    rm_target_vol_neutral = float(rm_cfg.get("target_vol_neutral_annual", rm_target_vol))
    rm_target_vol_down = float(rm_cfg.get("target_vol_down_annual", rm_target_vol))
    rm_vol_lb = int(rm_cfg.get("vol_lookback_bars", vol_window))
    rm_lev_min = float(rm_cfg.get("min_leverage", 0.2))
    rm_lev_max = float(rm_cfg.get("max_leverage", max(1.0, float(trade_cfg.get("leverage", 1.0)))))

    # 回撤节流
    dd_cfg = signal_cfg.get("drawdown_throttle", {}) or {}
    dd_enabled = bool(dd_cfg.get("enabled", False))
    dd_trigger = float(dd_cfg.get("trigger_dd", -0.08))
    dd_reduced_mult = float(dd_cfg.get("reduced_alloc_multiplier", 0.0))

    nav, rets = [1.0], []
    weights = {s: 0.0 for s in frames.keys()}
    lev_hist = []

    for i in range(max(max(lbs), ma_window, vol_window) + 1, len(dates) - 1):
        dt, nxt = dates[i], dates[i + 1]

        raw = {}
        for s, df in frames.items():
            hist = df.loc[:dt]
            close = pd.to_numeric(hist["close"], errors="coerce").dropna()
            if len(close) <= max(max(lbs), ma_window, vol_window) + 2:
                continue
            m = momentum_score(close, lbs[0], lbs[1], ws[0], ws[1])
            v = volatility_score(close, 20)
            d = max_drawdown_score(close, 60)
            sret = close.iloc[-1] / close.iloc[-lbs[0]] - 1 if len(close) > lbs[0] else -1
            lr = np.log(close.astype(float)).diff().dropna().tail(vol_window)
            if lr.empty:
                continue
            vol_ann = float(lr.std() * np.sqrt(ppy))
            trend_on = bool(close.iloc[-1] >= close.tail(ma_window).mean()) if len(close) >= ma_window else False
            if None in [m, v, d] or vol_ann <= 0:
                continue
            raw[s] = {
                "momentum": float(m),
                "vol": float(v),
                "vol_ann": float(vol_ann),
                "drawdown": float(d),
                "short_ret": float(sret),
                "trend_on": trend_on,
            }

        target = {s: 0.0 for s in frames.keys()}
        lev_now = 1.0
        if len(raw) >= top_n:
            if is_advanced_engine:
                score = {s: raw[s]["momentum"] / max(raw[s]["vol_ann"], 1e-8) for s in raw.keys()}
                ranked_desc = [x[0] for x in sorted(score.items(), key=lambda x: x[1], reverse=True)]
                ranked_asc = [x[0] for x in sorted(score.items(), key=lambda x: x[1])]
                if engine == "advanced_ls_cs":
                    long_candidates = [
                        s for s in ranked_desc if raw[s]["momentum"] > momentum_threshold
                    ][: max(1, int(top_n))]
                    short_candidates = [
                        s for s in ranked_asc if raw[s]["momentum"] < -momentum_threshold
                    ][: max(1, int(short_top_n))]
                    if not long_candidates:
                        long_candidates = ranked_desc[: max(1, int(top_n))]
                    if not short_candidates:
                        short_candidates = ranked_asc[: max(1, int(short_top_n))]
                else:
                    long_candidates = [
                        s for s in ranked_desc if raw[s]["trend_on"] and raw[s]["momentum"] > momentum_threshold
                    ][: max(1, int(top_n))]
                    short_candidates = [
                        s for s in ranked_asc if (not raw[s]["trend_on"]) and raw[s]["momentum"] < -momentum_threshold
                    ][: max(1, int(short_top_n))]

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
                if engine in {"advanced_ls_rmm", "advanced_ls_cs"}:
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
                        inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in long_candidates}
                        inv_sum = sum(inv_vol.values())
                        if inv_sum > 0:
                            for s in long_candidates:
                                target_map[s] = target_map.get(s, 0.0) + min(long_budget * (inv_vol[s] / inv_sum), single_cap)

                    short_gate = (regime_state == "risk_off") if ls_dynamic_budget_enabled else risk_off
                    short_enabled = (
                        contract_mode
                        and allow_short
                        and short_candidates
                        and ((not ls_short_requires_risk_off) or short_gate)
                        and short_budget > 0
                    )
                    if short_enabled:
                        inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in short_candidates}
                        inv_sum = sum(inv_vol.values())
                        if inv_sum > 0:
                            for s in short_candidates:
                                target_map[s] = target_map.get(s, 0.0) - min(short_budget * (inv_vol[s] / inv_sum), single_cap)
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
                    if not risk_off and long_candidates:
                        inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in long_candidates}
                        inv_sum = sum(inv_vol.values())
                        if inv_sum > 0:
                            for s in long_candidates:
                                target_map[s] = min(alloc_pct * (inv_vol[s] / inv_sum), single_cap)
                    elif contract_mode and allow_short and short_on_risk_off and short_candidates:
                        inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in short_candidates}
                        inv_sum = sum(inv_vol.values())
                        if inv_sum > 0:
                            for s in short_candidates:
                                target_map[s] = -min(alloc_pct * (inv_vol[s] / inv_sum), single_cap)

                if dd_enabled and target_map:
                    peak = max(nav)
                    curr_dd = nav[-1] / peak - 1 if peak > 0 else 0.0
                    if curr_dd <= dd_trigger:
                        target_map = {s: w * dd_reduced_mult for s, w in target_map.items()}

                if rm_enabled and target_map:
                    if len(rets) >= rm_vol_lb:
                        rv = float(np.std(rets[-rm_vol_lb:]) * np.sqrt(ppy))
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

                for s in frames.keys():
                    target[s] = float(target_map.get(s, 0.0))
            else:
                mr = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
                vr = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)
                dr = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)
                score = {
                    s: fw.get("momentum", 0.60) * mr.get(s, 0)
                    + fw.get("low_vol", 0.25) * vr.get(s, 0)
                    + fw.get("drawdown", 0.15) * dr.get(s, 0)
                    for s in raw.keys()
                }
                risk_off = all(v["short_ret"] < threshold for v in raw.values())
                if not risk_off:
                    picks = [x[0] for x in sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_n]]
                    tw = alloc_pct / top_n
                    target = {s: (tw if s in picks else 0.0) for s in frames.keys()}
                elif contract_mode and allow_short and short_on_risk_off:
                    short_picks = [x[0] for x in sorted(score.items(), key=lambda x: x[1])[: max(1, short_top_n)]]
                    tw = alloc_pct / max(1, len(short_picks))
                    target = {s: (-tw if s in short_picks else 0.0) for s in frames.keys()}
                else:
                    target = {s: 0.0 for s in frames.keys()}

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
        lev_hist.append(float(lev_now))

    out = summarize(nav, rets, ppy)
    out.update(
        {
            "model_profile": model_meta["profile_key"],
            "model_name": model_meta["profile_name"],
            "market_type": market_type,
            "contract_mode": contract_mode,
            "engine": engine,
            "allow_short": allow_short,
            "short_on_risk_off": short_on_risk_off,
            "ls_dynamic_budget_enabled": bool(ls_dynamic_budget_enabled),
            "ls_long_alloc_pct": float(ls_long_alloc_pct),
            "ls_short_alloc_pct": float(ls_short_alloc_pct),
            "ls_short_requires_risk_off": bool(ls_short_requires_risk_off),
            "avg_leverage_multiplier": float(np.mean(lev_hist)) if lev_hist else 1.0,
            "max_leverage_multiplier": float(np.max(lev_hist)) if lev_hist else 1.0,
        }
    )
    return out


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        crypto_cfg = load_yaml("config/crypto.yaml")
    except Exception as e:
        print(f"[backtest-crypto] config error: {e}")
        return EXIT_CONFIG_ERROR

    try:
        crypto = backtest_crypto(runtime, crypto_cfg)
    except Exception as e:
        print(f"[backtest-crypto] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    report = {
        "ts": datetime.now().isoformat(),
        "note": "crypto lightweight backtest",
        "crypto": crypto,
    }

    try:
        out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(out_dir)
        out_report = os.path.join(out_dir, "backtest_crypto_report.json")
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[backtest-crypto] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[backtest-crypto] report -> {out_report}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
