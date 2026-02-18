#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
    EXIT_SIGNAL_ERROR,
)
from core.signal import (
    momentum_score,
    volatility_score,
    max_drawdown_score,
    liquidity_score,
    normalize_rank,
)
from core.execution_guard import resolve_min_turnover
from core.exposure_gate import evaluate_exposure_gate


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_close_series(path):
    if not os.path.exists(path):
        return None, None
    df = pd.read_csv(path)
    if "close" not in df.columns:
        return None, None
    last_date = None
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            last_date = dates.iloc[-1].date().isoformat()
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        return None, last_date
    return close, last_date


def load_stock_return_history(path):
    if (not path) or (not os.path.exists(path)):
        return [], []
    try:
        df = pd.read_csv(path)
    except Exception:
        return [], []
    if ("strategy_ret" not in df.columns) or ("benchmark_ret_alloc" not in df.columns):
        return [], []
    s = pd.to_numeric(df["strategy_ret"], errors="coerce").dropna().tolist()
    b = pd.to_numeric(df["benchmark_ret_alloc"], errors="coerce").dropna().tolist()
    n = min(len(s), len(b))
    if n <= 0:
        return [], []
    return s[-n:], b[-n:]


def targets_to_map(targets):
    out = {}
    for row in targets:
        s = str(row.get("symbol", "")).strip()
        if not s:
            continue
        out[s] = float(row.get("target_weight", 0.0))
    return out


def calc_turnover(a_targets, b_targets):
    a = targets_to_map(a_targets)
    b = targets_to_map(b_targets)
    symbols = set(a.keys()) | set(b.keys())
    return float(sum(abs(a.get(s, 0.0) - b.get(s, 0.0)) for s in symbols))


def apply_risk_overlay(runtime, stock, out):
    model_cfg = stock.get("global_model", {})
    overlay_cfg = model_cfg.get("risk_overlay", {})
    enabled = bool(overlay_cfg.get("enabled", False))
    monitor_file = overlay_cfg.get(
        "monitor_file",
        os.path.join(runtime["paths"]["output_dir"], "reports", "stock_paper_forward_latest.json"),
    )
    threshold_excess_20d = float(overlay_cfg.get("trigger_excess_20d_vs_alloc", -0.02))
    threshold_dd = float(overlay_cfg.get("trigger_strategy_drawdown", -0.12))
    release_excess_20d = float(overlay_cfg.get("release_excess_20d_vs_alloc", 0.01))
    release_dd = float(overlay_cfg.get("release_strategy_drawdown", -0.06))
    min_defense_days = int(overlay_cfg.get("min_defense_days", 10))
    sticky_mode = bool(overlay_cfg.get("sticky_mode", True))
    defensive_bypass_single_max = bool(model_cfg.get("defensive_bypass_single_max", True))
    single_max = float(overlay_cfg.get("single_max_override", -1.0))
    if single_max <= 0:
        single_max = 1.0

    output_dir = runtime["paths"]["output_dir"]
    state_dir = os.path.join(output_dir, "state")
    ensure_dir(state_dir)
    state_file = os.path.join(state_dir, "stock_risk_overlay_state.json")
    state = load_json(state_file, default={})

    meta = {
        "enabled": enabled,
        "monitor_file": monitor_file,
        "trigger_excess_20d_vs_alloc": threshold_excess_20d,
        "trigger_strategy_drawdown": threshold_dd,
        "release_excess_20d_vs_alloc": release_excess_20d,
        "release_strategy_drawdown": release_dd,
        "min_defense_days": min_defense_days,
        "sticky_mode": sticky_mode,
        "triggered": False,
        "trigger_reasons": [],
        "released": False,
        "release_reasons": [],
        "metrics": {},
        "state_active": bool(state.get("active", False)),
        "activated_date": state.get("activated_date"),
        "days_in_defense": None,
    }

    if not enabled:
        state.update(
            {
                "active": False,
                "activated_date": None,
                "updated_at": datetime.now().isoformat(),
            }
        )
        save_json(state_file, state)
        out["risk_overlay"] = meta
        return out

    latest = load_json(monitor_file, default={})
    rolling = latest.get("rolling", {})
    latest_block = latest.get("latest", {})
    excess_20d = rolling.get("excess_return_20d_vs_alloc")
    strategy_dd = latest_block.get("strategy_dd")
    meta["metrics"] = {
        "excess_return_20d_vs_alloc": excess_20d,
        "strategy_dd": strategy_dd,
    }

    reasons = []
    if (excess_20d is not None) and (float(excess_20d) <= threshold_excess_20d):
        reasons.append("excess_20d_below_threshold")
    if (strategy_dd is not None) and (float(strategy_dd) <= threshold_dd):
        reasons.append("drawdown_below_threshold")

    market_date = out.get("market_date")
    asof_date = datetime.now().date()
    if market_date:
        try:
            asof_date = datetime.strptime(market_date, "%Y-%m-%d").date()
        except Exception:
            pass

    active = bool(state.get("active", False))
    activated_date = state.get("activated_date")
    days_in_defense = None
    if activated_date:
        try:
            act_date = datetime.strptime(activated_date, "%Y-%m-%d").date()
            days_in_defense = (asof_date - act_date).days
        except Exception:
            days_in_defense = None
    meta["days_in_defense"] = days_in_defense

    released = False
    release_reasons = []
    if sticky_mode and active:
        release_ok = True
        if excess_20d is None or float(excess_20d) < release_excess_20d:
            release_ok = False
        if strategy_dd is None or float(strategy_dd) < release_dd:
            release_ok = False
        if days_in_defense is None or days_in_defense < min_defense_days:
            release_ok = False
        if release_ok:
            active = False
            released = True
            release_reasons.append("release_threshold_reached")
            state["activated_date"] = None
    elif (not sticky_mode) and active and (not reasons):
        # Non-sticky mode exits defense as soon as trigger conditions clear.
        active = False
        released = True
        release_reasons.append("trigger_cleared_non_sticky")
        state["activated_date"] = None

    if reasons:
        active = True
        if not state.get("activated_date"):
            state["activated_date"] = asof_date.isoformat()

    if active:
        alloc = float(out.get("alloc_pct_effective", stock.get("capital_alloc_pct", 0.7)))
        defensive_symbol = out.get("defensive_symbol", stock.get("defensive_symbol", "511010"))
        tw = alloc if defensive_bypass_single_max else min(alloc, single_max)
        out["proposed_targets_before_overlay"] = out.get("targets", [])
        defensive_targets = [{"symbol": defensive_symbol, "target_weight": round(float(tw), 4)}]
        out["targets"] = defensive_targets
        if calc_turnover(out["proposed_targets_before_overlay"], defensive_targets) > 1e-8:
            out["force_rebalance"] = True
            out["force_rebalance_reason"] = "risk_overlay_active"
    elif released:
        out["force_rebalance"] = True
        out["force_rebalance_reason"] = "risk_overlay_released"

    meta["triggered"] = bool(reasons)
    meta["trigger_reasons"] = reasons
    meta["released"] = released
    meta["release_reasons"] = release_reasons
    meta["state_active"] = active
    meta["activated_date"] = state.get("activated_date")

    state.update(
        {
            "active": active,
            "last_triggered": bool(reasons),
            "last_trigger_reasons": reasons,
            "last_released": released,
            "last_release_reasons": release_reasons,
            "updated_at": datetime.now().isoformat(),
        }
    )
    if (not active) and (not reasons) and (not released):
        # Keep activated_date only when still active.
        state["activated_date"] = None
    save_json(state_file, state)

    out["risk_overlay"] = meta
    return out


def apply_execution_guard(runtime, stock, out):
    model_cfg = stock.get("global_model", {})
    guard_cfg = model_cfg.get("execution_guard", {})
    enabled = bool(guard_cfg.get("enabled", True))
    min_rebalance_days = int(guard_cfg.get("min_rebalance_days", model_cfg.get("rebalance_days", 20)))
    min_turnover_base = float(guard_cfg.get("min_turnover", 0.05))
    alloc_effective = float(out.get("alloc_pct_effective", stock.get("capital_alloc_pct", 0.7)))
    min_turnover, min_turnover_meta = resolve_min_turnover(min_turnover_base, alloc_effective, guard_cfg)
    force_on_regime_change = bool(guard_cfg.get("force_rebalance_on_regime_change", True))

    output_dir = runtime["paths"]["output_dir"]
    state_dir = os.path.join(output_dir, "state")
    ensure_dir(state_dir)
    state_file = os.path.join(state_dir, "stock_signal_state.json")

    state = load_json(state_file, default={})
    prev_targets = state.get("last_targets", [])
    prev_regime_on = bool(state.get("last_regime_on", False))
    prev_rebalance_date = state.get("last_rebalance_date")

    market_date = out.get("market_date")
    asof_date = datetime.now().date()
    if market_date:
        try:
            asof_date = datetime.strptime(market_date, "%Y-%m-%d").date()
        except Exception:
            pass

    turnover = calc_turnover(prev_targets, out["targets"])
    regime_changed = bool(out.get("regime_on", False)) != prev_regime_on
    days_since = None
    if prev_rebalance_date:
        try:
            prev_date = datetime.strptime(prev_rebalance_date, "%Y-%m-%d").date()
            days_since = (asof_date - prev_date).days
        except Exception:
            days_since = None

    force_rebalance = bool(out.get("force_rebalance", False))
    action = "rebalance"
    reason = "force_rebalance" if force_rebalance else "normal"
    effective_targets = out["targets"]

    if force_rebalance:
        action = "rebalance"
        reason = str(out.get("force_rebalance_reason", "force_rebalance"))
    elif enabled and prev_targets:
        if force_on_regime_change and regime_changed:
            action = "rebalance"
            reason = "regime_changed"
        elif (days_since is not None) and (days_since < min_rebalance_days):
            action = "hold"
            reason = "min_rebalance_days"
            effective_targets = prev_targets
        elif turnover < min_turnover:
            action = "hold"
            reason = "turnover_below_threshold"
            effective_targets = prev_targets

    out["execution_guard"] = {
        "enabled": enabled,
        "action": action,
        "reason": reason,
        "min_rebalance_days": min_rebalance_days,
        "min_turnover_base": round(min_turnover_base, 6),
        "min_turnover": round(min_turnover, 6),
        "dynamic_min_turnover": {
            "enabled": bool(min_turnover_meta.get("enabled", False)),
            "alloc_pct": None
            if min_turnover_meta.get("alloc_pct") is None
            else round(float(min_turnover_meta["alloc_pct"]), 6),
            "alloc_ref": None
            if min_turnover_meta.get("alloc_ref") is None
            else round(float(min_turnover_meta["alloc_ref"]), 6),
            "ratio": None
            if min_turnover_meta.get("ratio") is None
            else round(float(min_turnover_meta["ratio"]), 6),
            "multiplier": None
            if min_turnover_meta.get("multiplier") is None
            else round(float(min_turnover_meta["multiplier"]), 6),
        },
        "force_rebalance_on_regime_change": force_on_regime_change,
        "days_since_last_rebalance": days_since,
        "proposed_turnover": round(turnover, 6),
    }
    out["proposed_targets"] = out["targets"]
    out["targets"] = effective_targets

    new_state = {
        "ts": datetime.now().isoformat(),
        "last_market_date": asof_date.isoformat(),
        "last_regime_on": bool(out.get("regime_on", False)),
        "last_signal_reason": out.get("signal_reason", ""),
        "last_targets": out["targets"],
        "last_proposed_targets": out.get("proposed_targets", []),
        "last_execution_action": action,
        "last_execution_reason": reason,
        "last_rebalance_date": prev_rebalance_date,
    }
    if action == "rebalance":
        new_state["last_rebalance_date"] = asof_date.isoformat()
    save_json(state_file, new_state)
    return out


def run_global_momentum(runtime, stock, risk):
    env = runtime.get("env", "paper")
    total_capital = float(runtime.get("total_capital", 20000))
    alloc_pct = float(stock.get("capital_alloc_pct", 0.7))

    model_cfg = stock.get("global_model", {})
    benchmark_symbol = stock.get("benchmark_symbol", "510300")
    defensive_symbol = stock.get("defensive_symbol", "511010")
    top_n = int(model_cfg.get("top_n", 1))
    momentum_lb = int(model_cfg.get("momentum_lb", 252))
    ma_window = int(model_cfg.get("ma_window", 200))
    vol_window = int(model_cfg.get("vol_window", 20))
    min_score = float(model_cfg.get("min_score", 0.0))
    defensive_bypass_single_max = bool(model_cfg.get("defensive_bypass_single_max", True))

    universe = stock.get("universe", [])
    symbols = sorted(set(universe + [benchmark_symbol, defensive_symbol]))
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")

    closes = {}
    market_dates = {}
    vol = {}
    momentum = {}
    score = {}

    for s in symbols:
        fp = os.path.join(data_dir, f"{s}.csv")
        close, last_date = load_close_series(fp)
        if close is None:
            continue
        closes[s] = close
        market_dates[s] = last_date

    if benchmark_symbol not in closes:
        raise RuntimeError(f"benchmark data missing: {benchmark_symbol}")
    if defensive_symbol not in closes:
        raise RuntimeError(f"defensive data missing: {defensive_symbol}")

    risk_symbols = [s for s in closes.keys() if s != defensive_symbol]
    for s in risk_symbols:
        close = closes[s]
        if len(close) < max(momentum_lb, vol_window) + 2:
            continue
        m = float(close.iloc[-1] / close.iloc[-momentum_lb] - 1.0)
        ret = close.pct_change().dropna().tail(vol_window)
        if ret.empty:
            continue
        v = float(ret.std())
        if v <= 0:
            continue
        momentum[s] = m
        vol[s] = v
        score[s] = m / max(v, 1e-6)

    benchmark_close = closes[benchmark_symbol]
    regime_on = False
    if len(benchmark_close) >= ma_window:
        ma = float(benchmark_close.tail(ma_window).mean())
        regime_on = bool(benchmark_close.iloc[-1] >= ma)

    eligible = [s for s in score.keys() if score[s] > min_score]
    sleeve_weights = {}
    signal_reason = "risk_off"

    if regime_on and eligible:
        picks = sorted(eligible, key=lambda x: score[x], reverse=True)[:top_n]
        inv_vol = {s: 1.0 / max(vol[s], 1e-6) for s in picks}
        inv_total = sum(inv_vol.values())
        sleeve_weights = {s: inv_vol[s] / inv_total for s in picks}
        signal_reason = "risk_on"
    else:
        sleeve_weights = {defensive_symbol: 1.0}
        if not regime_on:
            signal_reason = "benchmark_below_ma"
        elif not eligible:
            signal_reason = "no_positive_momentum"

    gate_cfg = model_cfg.get("exposure_gate", {})
    history_file = gate_cfg.get(
        "history_file",
        os.path.join(runtime["paths"]["output_dir"], "reports", "stock_paper_forward_history.csv"),
    )
    hist_strategy_ret, hist_bench_alloc = load_stock_return_history(history_file)
    effective_alloc, exposure_meta = evaluate_exposure_gate(hist_strategy_ret, hist_bench_alloc, alloc_pct, gate_cfg)
    stock_capital = total_capital * effective_alloc
    single_max = float(risk["position_limits"]["stock_single_max_pct"])

    raw_targets = {s: effective_alloc * sleeve_w for s, sleeve_w in sleeve_weights.items()}
    capped_targets = {}
    for s, tw in raw_targets.items():
        if (not defensive_bypass_single_max) or s != defensive_symbol:
            capped_targets[s] = min(float(tw), single_max)
        else:
            capped_targets[s] = float(tw)

    used_alloc = sum(capped_targets.values())
    remain_alloc = max(0.0, effective_alloc - used_alloc)
    if remain_alloc > 1e-8:
        if defensive_symbol not in capped_targets:
            capped_targets[defensive_symbol] = 0.0
        if (not defensive_bypass_single_max):
            defensive_room = max(0.0, single_max - capped_targets[defensive_symbol])
            capped_targets[defensive_symbol] += min(remain_alloc, defensive_room)
        else:
            capped_targets[defensive_symbol] += remain_alloc

    targets = [
        {"symbol": s, "target_weight": round(float(w), 4)}
        for s, w in sorted(capped_targets.items(), key=lambda x: x[1], reverse=True)
        if w > 1e-8
    ]

    scores = []
    for s in sorted(score.keys(), key=lambda x: score[x], reverse=True):
        scores.append(
            {
                "symbol": s,
                "momentum": round(float(momentum[s]), 6),
                "vol": round(float(vol[s]), 6),
                "score": round(float(score[s]), 6),
            }
        )

    out = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "market_date": market_dates.get(benchmark_symbol),
        "env": env,
        "mode": "global_momentum",
        "model": "dual_momentum_trend_ivol_defense",
        "capital": stock_capital,
        "alloc_pct_effective": round(effective_alloc, 4),
        "alloc_pct_base": round(alloc_pct, 4),
        "benchmark_symbol": benchmark_symbol,
        "defensive_symbol": defensive_symbol,
        "regime_on": regime_on,
        "signal_reason": signal_reason,
        "params": {
            "top_n": top_n,
            "momentum_lb": momentum_lb,
            "ma_window": ma_window,
            "vol_window": vol_window,
            "min_score": min_score,
            "defensive_bypass_single_max": defensive_bypass_single_max,
        },
        "scores": scores,
        "targets": targets,
        "exposure_gate": exposure_meta,
        "note": "global momentum production targets",
    }
    out = apply_risk_overlay(runtime, stock, out)
    return apply_execution_guard(runtime, stock, out)


def run_legacy_multifactor(runtime, stock, risk):
    env = runtime.get("env", "paper")
    total_capital = runtime.get("total_capital", 20000)
    alloc_pct = stock.get("capital_alloc_pct", 0.7)

    lbs = stock.get("signal", {}).get("momentum_lookback_days", [20, 60])
    ws = stock.get("signal", {}).get("momentum_weights", [0.6, 0.4])
    top_n = stock.get("signal", {}).get("top_n", 2)

    factor_w = stock.get("signal", {}).get(
        "factor_weights",
        {"momentum": 0.45, "low_vol": 0.25, "drawdown": 0.20, "liquidity": 0.10},
    )

    universe = stock.get("universe", [])
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")

    raw = {}
    benchmark_close = None

    for s in universe:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp)
            if "close" not in df.columns:
                continue
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(close) < 130:
                continue

            m = momentum_score(close, lbs[0], lbs[1], ws[0], ws[1])
            v = volatility_score(close, lb=20)
            d = max_drawdown_score(close, lb=60)
            amt = liquidity_score(df.get("amount", pd.Series(dtype=float)), lb=20)
            if None in [m, v, d] or amt is None:
                continue

            raw[s] = {"momentum": m, "vol": v, "drawdown": d, "liquidity": amt}

            if s == "510300":
                benchmark_close = close
        except Exception:
            continue

    # rank normalize factors
    mom_rank = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
    vol_rank = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)  # lower vol better
    dd_rank = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)  # less negative better
    liq_rank = normalize_rank({k: v["liquidity"] for k, v in raw.items()}, ascending=False)

    scored = []
    for s in raw.keys():
        score = (
            factor_w.get("momentum", 0.45) * mom_rank.get(s, 0)
            + factor_w.get("low_vol", 0.25) * vol_rank.get(s, 0)
            + factor_w.get("drawdown", 0.20) * dd_rank.get(s, 0)
            + factor_w.get("liquidity", 0.10) * liq_rank.get(s, 0)
        )
        scored.append((s, float(score)))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    picks = [x[0] for x in scored[:top_n]]

    # risk switch: benchmark below MA120 => half risk
    risk_switch = {"enabled": False, "reason": "normal", "alloc_multiplier": 1.0}
    if benchmark_close is not None and len(benchmark_close) >= 120:
        ma120 = benchmark_close.tail(120).mean()
        if benchmark_close.iloc[-1] < ma120:
            risk_switch = {"enabled": True, "reason": "benchmark_below_ma120", "alloc_multiplier": 0.5}

    effective_alloc = alloc_pct * risk_switch["alloc_multiplier"]
    stock_capital = total_capital * effective_alloc

    target_weight_each = min(
        risk["position_limits"]["stock_single_max_pct"],
        effective_alloc / max(1, len(picks) if picks else top_n),
    )

    targets = [{"symbol": s, "target_weight": round(target_weight_each, 4)} for s in picks]

    out = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "env": env,
        "mode": "legacy_multifactor",
        "capital": stock_capital,
        "alloc_pct_effective": round(effective_alloc, 4),
        "risk_switch": risk_switch,
        "factor_weights": factor_w,
        "scores": [{"symbol": s, "score": sc} for s, sc in scored],
        "targets": targets,
        "note": "legacy multifactor targets",
    }
    return out


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock = load_yaml("config/stock.yaml")
        risk = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[stock] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    if not stock.get("enabled", False):
        print("[stock] disabled")
        return EXIT_DISABLED

    try:
        mode = stock.get("mode", "global_momentum")
        if mode == "global_momentum":
            out = run_global_momentum(runtime, stock, risk)
        else:
            out = run_legacy_multifactor(runtime, stock, risk)
    except Exception as e:
        print(f"[stock] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    try:
        output_dir = runtime["paths"]["output_dir"]
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "orders"))

        out_file = os.path.join(output_dir, "orders", "stock_targets.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[stock] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(f"[stock] done -> {out_file}")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
