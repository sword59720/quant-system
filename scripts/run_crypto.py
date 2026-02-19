#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_FORMAT_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
    EXIT_SIGNAL_ERROR,
)
from core.crypto_model import resolve_crypto_model_cfg
from core.signal import momentum_score, volatility_score, max_drawdown_score, normalize_rank


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_send_wecom_message(content, title, dedup_key=None, dedup_hours=0):
    try:
        from core.notify_wecom import send_wecom_message
    except Exception as e:
        return False, f"notify module unavailable: {e}"
    return send_wecom_message(content, title=title, dedup_key=dedup_key, dedup_hours=dedup_hours)


def _weighted_momentum(close: pd.Series, lbs, ws):
    if len(lbs) < 2:
        lbs = [30, 90]
    if len(ws) < 2:
        ws = [0.6, 0.4]
    lb1, lb2 = int(lbs[0]), int(lbs[1])
    w1, w2 = float(ws[0]), float(ws[1])
    if len(close) <= max(lb1, lb2):
        return None
    m1 = float(close.iloc[-1] / close.iloc[-lb1] - 1.0)
    m2 = float(close.iloc[-1] / close.iloc[-lb2] - 1.0)
    return m1 * w1 + m2 * w2


def _annualized_vol(close: pd.Series, lb: int, periods_per_year: int):
    if len(close) <= lb + 2:
        return None
    lr = np.log(close.astype(float)).diff().dropna().tail(lb)
    if lr.empty:
        return None
    return float(lr.std() * np.sqrt(periods_per_year))


def _estimate_portfolio_vol_annual(series_map, target_map, lookback: int, periods_per_year: int):
    symbols = [s for s, w in target_map.items() if abs(w) > 1e-12 and s in series_map]
    if not symbols:
        return None
    returns = []
    for s in symbols:
        r = series_map[s].pct_change().dropna().tail(lookback)
        returns.append(r.rename(s))
    if not returns:
        return None
    ret_df = pd.concat(returns, axis=1).dropna()
    if len(ret_df) < max(5, lookback // 3):
        return None
    cov = ret_df.cov().values * periods_per_year
    w = np.array([target_map[s] for s in symbols], dtype=float)
    var = float(w.T @ cov @ w)
    if var <= 0:
        return None
    return float(np.sqrt(var))


def build_crypto_targets(runtime, crypto, risk):
    if not runtime.get("enabled", True):
        raise RuntimeError("runtime disabled")
    if not crypto.get("enabled", False):
        raise RuntimeError("crypto disabled")

    env = runtime.get("env", "paper")
    total_capital = runtime.get("total_capital", 20000)
    alloc_pct = crypto.get("capital_alloc_pct", 0.3)
    crypto_capital = total_capital * alloc_pct

    model_meta = resolve_crypto_model_cfg(crypto)
    signal_cfg = model_meta["signal"]
    defense_cfg = model_meta["defense"]
    trade_market_type = model_meta["market_type"]
    contract_mode = model_meta["contract_mode"]
    engine = str(signal_cfg.get("engine", "multifactor_v1")).strip().lower()

    lbs = signal_cfg.get("momentum_lookback_bars", [30, 90])
    ws = signal_cfg.get("momentum_weights", [0.6, 0.4])
    top_n = signal_cfg.get("top_n", 1)
    symbols = crypto.get("symbols", [])
    trade_cfg = crypto.get("trade", {}) or {}
    allow_short = bool(trade_cfg.get("allow_short", False))
    short_on_risk_off = bool(signal_cfg.get("short_on_risk_off", False))
    short_top_n = int(signal_cfg.get("short_top_n", top_n))
    ppy = int(signal_cfg.get("periods_per_year", 6 * 365))
    ma_window = int(signal_cfg.get("ma_window_bars", 180))
    vol_window = int(signal_cfg.get("vol_window_bars", 20))
    momentum_threshold = float(signal_cfg.get("momentum_threshold_pct", 0.0)) / 100.0
    is_advanced_engine = engine in {"advanced_rmm", "advanced_ls_rmm"}
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
    ls_regime_down_long_alloc_pct = float(signal_cfg.get("ls_regime_down_long_alloc_pct", ls_long_alloc_pct))
    ls_regime_down_short_alloc_pct = float(signal_cfg.get("ls_regime_down_short_alloc_pct", ls_short_alloc_pct))

    factor_w = signal_cfg.get(
        "factor_weights", {"momentum": 0.60, "low_vol": 0.25, "drawdown": 0.15}
    )

    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    raw = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp)
            if "close" not in df.columns:
                continue
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(close) < max(lbs) + 5:
                continue

            m = _weighted_momentum(close, lbs, ws)
            v = volatility_score(close, lb=20)
            d = max_drawdown_score(close, lb=60)
            sret = float(close.iloc[-1] / close.iloc[-lbs[0]] - 1)
            if m is None or None in [v, d]:
                continue
            trend_on = False
            if len(close) >= ma_window:
                trend_on = bool(close.iloc[-1] >= close.tail(ma_window).mean())

            vol_ann = _annualized_vol(close, vol_window, ppy)
            if vol_ann is None:
                continue

            raw[s] = {
                "momentum": m,
                "vol": v,
                "vol_ann": vol_ann,
                "drawdown": d,
                "short_ret": sret,
                "trend_on": trend_on,
                "close": close,
                "score": float(m / max(vol_ann, 1e-8)),
            }
        except Exception:
            continue

    if not raw:
        raise RuntimeError("no valid crypto symbol data")

    scored = []
    if is_advanced_engine:
        for s, rv in raw.items():
            scored.append((s, float(rv["score"])))
    else:
        mom_rank = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
        vol_rank = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)
        dd_rank = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)
        for s in raw.keys():
            score = (
                factor_w.get("momentum", 0.60) * mom_rank.get(s, 0)
                + factor_w.get("low_vol", 0.25) * vol_rank.get(s, 0)
                + factor_w.get("drawdown", 0.15) * dd_rank.get(s, 0)
            )
            scored.append((s, float(score)))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    picks = [x[0] for x in scored[: max(1, int(top_n))]]
    short_picks = [x[0] for x in sorted(scored, key=lambda x: x[1])[: max(1, int(short_top_n))]]

    # risk off rule: all short-term returns below threshold => USDT defense
    threshold = float(defense_cfg.get("risk_off_threshold_pct", -3.0)) / 100.0
    risk_off = False
    if raw and all(v["short_ret"] < threshold for v in raw.values()):
        risk_off = True
    regime_state = "neutral"
    avg_momentum = None
    trend_breadth = None
    if raw:
        avg_momentum = float(np.mean([v["momentum"] for v in raw.values()]))
        trend_breadth = float(np.mean([1.0 if v["trend_on"] else 0.0 for v in raw.values()]))
        down_breadth = 1.0 - trend_breadth
        if (avg_momentum <= -ls_regime_momentum_threshold) and (down_breadth >= ls_regime_trend_breadth):
            regime_state = "risk_off"
        elif (avg_momentum >= ls_regime_momentum_threshold) and (trend_breadth >= ls_regime_trend_breadth):
            regime_state = "risk_on"

    targets = []
    leverage_multiplier = 1.0
    est_portfolio_vol_ann = None
    long_budget_used = 0.0
    short_budget_used = 0.0
    if is_advanced_engine:
        single_cap = float(
            signal_cfg.get(
                "single_max_pct",
                risk["position_limits"].get("crypto_single_max_pct", alloc_pct),
            )
        )
        long_candidates = [s for s in picks if raw[s]["trend_on"] and raw[s]["momentum"] > momentum_threshold]
        short_candidates = [
            s for s in short_picks if (not raw[s]["trend_on"]) and raw[s]["momentum"] < -momentum_threshold
        ]
        target_map = {}
        if engine == "advanced_ls_rmm":
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
            long_budget_used = long_budget
            short_budget_used = short_budget
            if long_candidates and long_budget > 0:
                inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in long_candidates}
                inv_sum = sum(inv_vol.values())
                if inv_sum > 0:
                    for s in long_candidates:
                        w = long_budget * (inv_vol[s] / inv_sum)
                        target_map[s] = target_map.get(s, 0.0) + min(w, single_cap)

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
                        w = short_budget * (inv_vol[s] / inv_sum)
                        target_map[s] = target_map.get(s, 0.0) - min(w, single_cap)
        else:
            if not risk_off and long_candidates:
                inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in long_candidates}
                inv_sum = sum(inv_vol.values())
                if inv_sum > 0:
                    for s in long_candidates:
                        w = alloc_pct * (inv_vol[s] / inv_sum)
                        target_map[s] = min(w, single_cap)
            elif contract_mode and allow_short and short_on_risk_off and short_candidates:
                inv_vol = {s: 1.0 / max(raw[s]["vol_ann"], 1e-8) for s in short_candidates}
                inv_sum = sum(inv_vol.values())
                if inv_sum > 0:
                    for s in short_candidates:
                        w = alloc_pct * (inv_vol[s] / inv_sum)
                        target_map[s] = -min(w, single_cap)

        rm_cfg = signal_cfg.get("risk_managed", {}) or {}
        rm_enabled = bool(rm_cfg.get("enabled", False))
        if rm_enabled and target_map:
            target_vol_ann = float(rm_cfg.get("target_vol_annual", 0.0))
            vol_lb = int(rm_cfg.get("vol_lookback_bars", vol_window))
            lev_max = float(rm_cfg.get("max_leverage", max(1, int(trade_cfg.get("leverage", 1)))))
            lev_min = float(rm_cfg.get("min_leverage", 0.2))
            series_map = {s: raw[s]["close"] for s in target_map.keys()}
            est_portfolio_vol_ann = _estimate_portfolio_vol_annual(series_map, target_map, vol_lb, ppy)
            if (target_vol_ann > 0) and est_portfolio_vol_ann and est_portfolio_vol_ann > 1e-8:
                leverage_multiplier = target_vol_ann / est_portfolio_vol_ann
                leverage_multiplier = max(lev_min, min(lev_max, leverage_multiplier))
                target_map = {s: w * leverage_multiplier for s, w in target_map.items()}

        gross = float(sum(abs(w) for w in target_map.values()))
        max_exposure_pct = float(
            signal_cfg.get(
                "max_exposure_pct",
                alloc_pct * max(1.0, float(trade_cfg.get("leverage", 1.0))) if contract_mode else alloc_pct,
            )
        )
        if gross > max_exposure_pct and gross > 1e-8:
            scale = max_exposure_pct / gross
            target_map = {s: w * scale for s, w in target_map.items()}

        targets = [{"symbol": s, "target_weight": round(float(w), 4)} for s, w in target_map.items() if abs(w) > 1e-8]
        targets = sorted(targets, key=lambda x: abs(x["target_weight"]), reverse=True)
    elif not risk_off and picks:
        target_weight_each = min(
            risk["position_limits"]["crypto_single_max_pct"],
            crypto.get("capital_alloc_pct", 0.3) / max(1, len(picks)),
        )
        targets = [{"symbol": s, "target_weight": round(target_weight_each, 4)} for s in picks]
    elif contract_mode and allow_short and short_on_risk_off and short_picks:
        short_weight_each = min(
            risk["position_limits"]["crypto_single_max_pct"],
            crypto.get("capital_alloc_pct", 0.3) / max(1, len(short_picks)),
        )
        targets = [{"symbol": s, "target_weight": round(-short_weight_each, 4)} for s in short_picks]
    elif (not contract_mode) and defense_cfg.get("use_usdt_defense", True):
        targets = [{"symbol": "USDT", "target_weight": round(alloc_pct, 4)}]
    else:
        # 合约模式 risk_off 默认平仓到 0 仓（不产生目标仓位）
        targets = []

    note = "v3 dual-side long/short targets" if engine == "advanced_ls_rmm" else "v3 multifactor targets"
    return {
        "ts": datetime.now().isoformat(),
        "market": "crypto",
        "env": env,
        "capital": crypto_capital,
        "model_profile": model_meta["profile_key"],
        "model_name": model_meta["profile_name"],
        "market_type": trade_market_type,
        "contract_mode": contract_mode,
        "engine": engine,
        "allow_short": allow_short,
        "short_on_risk_off": short_on_risk_off,
        "regime_state": regime_state,
        "avg_momentum": None if avg_momentum is None else round(float(avg_momentum), 6),
        "trend_breadth": None if trend_breadth is None else round(float(trend_breadth), 6),
        "ls_dynamic_budget_enabled": ls_dynamic_budget_enabled,
        "long_budget_used": round(float(long_budget_used), 4),
        "short_budget_used": round(float(short_budget_used), 4),
        "ls_long_alloc_pct": round(float(ls_long_alloc_pct), 4),
        "ls_short_alloc_pct": round(float(ls_short_alloc_pct), 4),
        "ls_short_requires_risk_off": ls_short_requires_risk_off,
        "risk_off": risk_off,
        "leverage_multiplier": round(float(leverage_multiplier), 4),
        "est_portfolio_vol_annual": None if est_portfolio_vol_ann is None else round(float(est_portfolio_vol_ann), 6),
        "factor_weights": factor_w,
        "scores": [{"symbol": s, "score": sc} for s, sc in scored],
        "targets": targets,
        "note": note,
    }


def maybe_auto_execute_live(runtime, crypto, target_file):
    exec_cfg = crypto.get("execution", {}) or {}
    enabled = bool(exec_cfg.get("auto_place_order", False))
    env = str(runtime.get("env", "paper")).strip().lower()
    min_notional = float(exec_cfg.get("min_order_notional", 0.0) or 0.0)

    result = {
        "enabled": enabled,
        "env": env,
        "reason": "disabled",
        "trades_file": "",
        "orders_total": 0,
        "orders_after_filter": 0,
        "dropped_small_orders": 0,
        "executed": False,
    }
    if not enabled:
        return result
    if env != "live":
        result["reason"] = "env_not_live"
        return result

    from scripts.generate_trades import build_market_orders
    from scripts.execute_trades import execute_trades

    output_dir = runtime["paths"]["output_dir"]
    orders_dir = os.path.join(output_dir, "orders")
    state_dir = os.path.join(output_dir, "state")
    ensure_dir(orders_dir)
    ensure_dir(state_dir)

    pos_file = os.path.join(state_dir, "crypto_positions.json")
    if not os.path.exists(pos_file):
        with open(pos_file, "w", encoding="utf-8") as f:
            json.dump({"positions": []}, f, ensure_ascii=False, indent=2)

    trades_file = os.path.join(orders_dir, "crypto_trades.json")
    plan = build_market_orders(
        market="crypto",
        target_file=target_file,
        pos_file=pos_file,
        out_file=trades_file,
        total_capital=float(runtime.get("total_capital", 0)),
    )

    orders = plan.get("orders", [])
    result["trades_file"] = trades_file
    result["orders_total"] = len(orders)

    if min_notional > 0:
        kept = []
        dropped = 0
        for x in orders:
            amt = float(x.get("amount_quote", 0) or 0)
            if amt >= min_notional:
                kept.append(x)
            else:
                dropped += 1
        plan["orders"] = kept
        with open(trades_file, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
        orders = kept
        result["dropped_small_orders"] = dropped

    result["orders_after_filter"] = len(orders)
    if not orders:
        result["reason"] = "no_orders"
        result["executed"] = True
        return result

    ok = execute_trades(trades_file=trades_file, dry_run=False)
    result["executed"] = bool(ok)
    result["reason"] = "ok" if ok else "execution_failed"
    return result


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        crypto = load_yaml("config/crypto.yaml")
        risk = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[crypto] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    if not crypto.get("enabled", False):
        print("[crypto] disabled")
        return EXIT_DISABLED

    try:
        out = build_crypto_targets(runtime, crypto, risk)
    except Exception as e:
        print(f"[crypto] data/signal error: {e}")
        return EXIT_DATA_FORMAT_ERROR

    try:
        output_dir = runtime["paths"]["output_dir"]
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "orders"))
        out_file = os.path.join(output_dir, "orders", "crypto_targets.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[crypto] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(f"[crypto] done -> {out_file}")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    auto_meta = {}
    try:
        auto_meta = maybe_auto_execute_live(runtime, crypto, out_file)
        out["auto_execution"] = auto_meta
        if auto_meta.get("enabled", False):
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"[crypto-auto] {json.dumps(auto_meta, ensure_ascii=False)}")
    except Exception as e:
        print(f"[crypto-auto] failed: {e}")
        return EXIT_SIGNAL_ERROR

    if auto_meta.get("enabled", False) and auto_meta.get("reason") == "execution_failed":
        print("[crypto-auto] execution failed")
        return EXIT_SIGNAL_ERROR

    # 企业微信通知：目标仓位 + 风控状态
    try:
        tgt = ", ".join([f"{x['symbol']}:{x['target_weight']:.2f}" for x in out.get("targets", [])]) or "无"
        lines = [
            f"时间: {out.get('ts','')}",
            f"市场: crypto",
            f"risk_off: {out.get('risk_off', False)}",
            f"目标仓位: {tgt}",
        ]
        ok, msg = safe_send_wecom_message("\n".join(lines), title="目标仓位更新")
        print(f"[notify] wecom {'ok' if ok else 'fail'}: {msg}")
        if out.get("risk_off", False):
            safe_send_wecom_message(
                "币圈风控触发 risk_off，已切换/保持防守仓位。",
                title="风控状态异常触发",
                dedup_key="risk_crypto_trigger",
                dedup_hours=24,
            )
    except Exception as e:
        print(f"[notify] wecom error: {e}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
