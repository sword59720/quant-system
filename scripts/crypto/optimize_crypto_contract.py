#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import os
import random
import sys
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.crypto.validate_crypto_contract import run_contract_backtest


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pick_dates(crypto_cfg):
    symbols = crypto_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])
    sets = []
    for s in symbols:
        fp = os.path.join("data", "crypto", f"{s.replace('/', '_')}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns:
            continue
        sets.append(set(pd.to_datetime(df["date"], errors="coerce").dropna()))
    if not sets:
        return None, None
    dates = sorted(set.intersection(*sets))
    if len(dates) < 20:
        return None, None
    cut70 = dates[int(len(dates) * 0.7)]
    cut80 = dates[int(len(dates) * 0.8)]
    return cut70, cut80


def apply_contract_params(base_cfg, params):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("trade", {})
    cfg["trade"]["allow_short"] = bool(params["allow_short"])

    cm = cfg.setdefault("contract_model", {})
    signal = cm.setdefault("signal", {})
    defense = cm.setdefault("defense", {})

    signal["engine"] = str(params["engine"])
    signal["top_n"] = int(params["top_n"])
    signal["short_top_n"] = int(params["short_top_n"])
    signal["short_on_risk_off"] = bool(params["short_on_risk_off"])
    signal["momentum_lookback_bars"] = list(params["momentum_lookback_bars"])
    signal["momentum_weights"] = list(params["momentum_weights"])
    signal["ma_window_bars"] = int(params["ma_window_bars"])
    signal["vol_window_bars"] = int(params["vol_window_bars"])
    signal["momentum_threshold_pct"] = float(params["momentum_threshold_pct"])
    signal["rebalance_threshold"] = float(params["rebalance_threshold"])
    signal["max_exposure_pct"] = float(params["max_exposure_pct"])
    signal["ls_long_alloc_pct"] = float(params["ls_long_alloc_pct"])
    signal["ls_short_alloc_pct"] = float(params["ls_short_alloc_pct"])
    signal["ls_short_requires_risk_off"] = bool(params["ls_short_requires_risk_off"])
    signal["ls_dynamic_budget_enabled"] = bool(params["ls_dynamic_budget_enabled"])
    signal["ls_regime_momentum_threshold_pct"] = float(params["ls_regime_momentum_threshold_pct"])
    signal["ls_regime_trend_breadth"] = float(params["ls_regime_trend_breadth"])
    signal["ls_regime_up_long_alloc_pct"] = float(params["ls_regime_up_long_alloc_pct"])
    signal["ls_regime_up_short_alloc_pct"] = float(params["ls_regime_up_short_alloc_pct"])
    signal["ls_regime_neutral_long_alloc_pct"] = float(params["ls_regime_neutral_long_alloc_pct"])
    signal["ls_regime_neutral_short_alloc_pct"] = float(params["ls_regime_neutral_short_alloc_pct"])
    signal["ls_neutral_exposure_multiplier"] = float(params["ls_neutral_exposure_multiplier"])
    signal["ls_regime_down_long_alloc_pct"] = float(params["ls_regime_down_long_alloc_pct"])
    signal["ls_regime_down_short_alloc_pct"] = float(params["ls_regime_down_short_alloc_pct"])
    signal["ls_confidence_scaling_enabled"] = bool(params["ls_confidence_scaling_enabled"])
    signal["ls_confidence_min_spread"] = float(params["ls_confidence_min_spread"])
    signal["ls_confidence_full_spread"] = float(params["ls_confidence_full_spread"])

    rm = signal.setdefault("risk_managed", {})
    rm["enabled"] = True
    rm["target_vol_annual"] = float(params["target_vol_annual"])
    rm["vol_lookback_bars"] = int(params["vol_lookback_bars"])
    rm["min_leverage"] = float(params["min_leverage"])
    rm["max_leverage"] = float(params["max_leverage"])

    dd = signal.setdefault("drawdown_throttle", {})
    dd["enabled"] = True
    dd["trigger_dd"] = float(params["trigger_dd"])
    dd["reduced_alloc_multiplier"] = float(params["reduced_alloc_multiplier"])

    defense["risk_off_threshold_pct"] = float(params["risk_off_threshold_pct"])
    return cfg


def metric_ok(m):
    return isinstance(m, dict) and ("annual_return" in m) and ("max_drawdown" in m)


def score_candidate(full, o70, o80, stress):
    o70_ann = float(o70["annual_return"])
    o80_ann = float(o80["annual_return"])
    stress_ann = float(stress["annual_return"])
    robust_oos_ann = min(o70_ann, o80_ann)
    avg_oos_ann = 0.5 * (o70_ann + o80_ann)
    full_ann = float(full["annual_return"])
    full_sharpe = float(full.get("sharpe", 0.0))

    worst_dd = min(
        float(full["max_drawdown"]),
        float(o70["max_drawdown"]),
        float(o80["max_drawdown"]),
        float(stress["max_drawdown"]),
    )
    dd_penalty = max(0.0, (-0.10 - worst_dd)) * 3.0
    downside_penalty = max(0.0, -robust_oos_ann) * 1.8 + max(0.0, -stress_ann) * 0.8
    return (
        (0.50 * robust_oos_ann)
        + (0.20 * avg_oos_ann)
        + (0.15 * full_ann)
        + (0.10 * stress_ann)
        + (0.05 * full_sharpe)
        - dd_penalty
        - downside_penalty
    )


def strict_target_pass(full, o70, o80, stress):
    return (
        (float(full["annual_return"]) >= 0.08)
        and (float(full["max_drawdown"]) >= -0.10)
        and (float(o70["annual_return"]) >= 0.05)
        and (float(o70["max_drawdown"]) >= -0.08)
        and (float(o80["annual_return"]) >= 0.05)
        and (float(o80["max_drawdown"]) >= -0.08)
        and (float(stress["annual_return"]) >= 0.00)
        and (float(stress["max_drawdown"]) >= -0.12)
    )


def ultimate_target_pass(full, o70, o80, stress):
    return (
        (float(full["annual_return"]) >= 0.50)
        and (float(full["max_drawdown"]) >= -0.10)
        and (float(o70["annual_return"]) >= 0.10)
        and (float(o70["max_drawdown"]) >= -0.10)
        and (float(o80["annual_return"]) >= 0.10)
        and (float(o80["max_drawdown"]) >= -0.10)
        and (float(stress["annual_return"]) >= 0.00)
        and (float(stress["max_drawdown"]) >= -0.12)
    )


def main():
    parser = argparse.ArgumentParser(description="Optimize crypto contract model with robust OOS/stress objective")
    parser.add_argument("--trials", type=int, default=40, help="random trials")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--engine",
        type=str,
        default="all",
        choices=["all", "advanced_rmm", "advanced_ls_rmm", "advanced_ls_cs"],
        help="engine filter",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    runtime = load_yaml("config/runtime.yaml")
    crypto = load_yaml("config/crypto.yaml")

    cut70, cut80 = pick_dates(crypto)
    if cut70 is None:
        raise SystemExit("[crypto-opt] no sufficient data for optimization")

    engine_space = (
        ["advanced_rmm", "advanced_ls_rmm", "advanced_ls_cs"]
        if args.engine == "all"
        else [args.engine]
    )
    space = {
        "engine": engine_space,
        "allow_short": [False, True],
        "short_on_risk_off": [False, True],
        "top_n": [1, 2],
        "short_top_n": [1, 2],
        "momentum_lookback_bars": [[30, 90], [60, 120], [90, 180], [120, 240]],
        "momentum_weights": [[1.0, 0.0], [0.8, 0.2]],
        "ma_window_bars": [90, 120, 180, 240],
        "vol_window_bars": [20, 30, 40],
        "momentum_threshold_pct": [0.0, 0.5, 1.0, 2.0],
        "rebalance_threshold": [0.01, 0.02, 0.04],
        "max_exposure_pct": [0.5, 0.8, 1.0, 1.2],
        "risk_off_threshold_pct": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
        "ls_long_alloc_pct": [0.20, 0.30, 0.40],
        "ls_short_alloc_pct": [0.20, 0.30, 0.40],
        "ls_short_requires_risk_off": [False, True],
        "ls_dynamic_budget_enabled": [False, True],
        "ls_regime_momentum_threshold_pct": [0.2, 0.5, 1.0, 1.5],
        "ls_regime_trend_breadth": [0.5, 0.6, 0.7],
        "ls_regime_up_long_alloc_pct": [0.4, 0.6, 0.8, 1.0],
        "ls_regime_up_short_alloc_pct": [0.0, 0.1, 0.2, 0.3],
        "ls_regime_neutral_long_alloc_pct": [0.2, 0.3, 0.4, 0.5],
        "ls_regime_neutral_short_alloc_pct": [0.2, 0.3, 0.4, 0.5],
        "ls_neutral_exposure_multiplier": [0.4, 0.6, 0.8, 1.0, 1.2],
        "ls_regime_down_long_alloc_pct": [0.0, 0.1, 0.2, 0.3],
        "ls_regime_down_short_alloc_pct": [0.4, 0.6, 0.8, 1.0],
        "ls_confidence_scaling_enabled": [False, True],
        "ls_confidence_min_spread": [0.08, 0.10, 0.12, 0.16, 0.20],
        "ls_confidence_full_spread": [0.20, 0.24, 0.28, 0.32, 0.40],
        "target_vol_annual": [0.12, 0.16, 0.20, 0.25, 0.30],
        "vol_lookback_bars": [20, 30, 40],
        "min_leverage": [0.2],
        "max_leverage": [1.0, 1.2, 1.5, 2.0],
        "trigger_dd": [-0.05, -0.06, -0.08, -0.10],
        "reduced_alloc_multiplier": [0.0, 0.1, 0.2, 0.3, 0.5],
    }

    records = []
    for i in range(args.trials):
        params = {k: random.choice(v) for k, v in space.items()}
        if params["ma_window_bars"] < params["momentum_lookback_bars"][1]:
            params["ma_window_bars"] = params["momentum_lookback_bars"][1]
        if params["ls_confidence_full_spread"] <= params["ls_confidence_min_spread"]:
            params["ls_confidence_full_spread"] = round(params["ls_confidence_min_spread"] + 0.08, 4)
        if not params["allow_short"]:
            params["short_on_risk_off"] = False
            params["ls_short_alloc_pct"] = 0.0
            params["ls_regime_up_short_alloc_pct"] = 0.0
            params["ls_regime_neutral_short_alloc_pct"] = 0.0
            params["ls_regime_down_short_alloc_pct"] = 0.0
        if params["engine"] not in {"advanced_ls_rmm", "advanced_ls_cs"}:
            params["ls_short_requires_risk_off"] = bool(params["short_on_risk_off"])
            params["ls_long_alloc_pct"] = float(crypto.get("capital_alloc_pct", 0.3))
            params["ls_short_alloc_pct"] = float(crypto.get("capital_alloc_pct", 0.3))
            params["ls_dynamic_budget_enabled"] = False
            params["ls_confidence_scaling_enabled"] = False
            params["ls_regime_momentum_threshold_pct"] = float(params["momentum_threshold_pct"])
            params["ls_regime_trend_breadth"] = 0.6
            params["ls_regime_up_long_alloc_pct"] = float(params["ls_long_alloc_pct"])
            params["ls_regime_up_short_alloc_pct"] = float(params["ls_short_alloc_pct"])
            params["ls_regime_neutral_long_alloc_pct"] = float(params["ls_long_alloc_pct"])
            params["ls_regime_neutral_short_alloc_pct"] = float(params["ls_short_alloc_pct"])
            params["ls_regime_down_long_alloc_pct"] = float(params["ls_long_alloc_pct"])
            params["ls_regime_down_short_alloc_pct"] = float(params["ls_short_alloc_pct"])

        cfg = apply_contract_params(crypto, params)
        full = run_contract_backtest(runtime, cfg)
        o70 = run_contract_backtest(runtime, cfg, date_from=cut70)
        o80 = run_contract_backtest(runtime, cfg, date_from=cut80)
        stress = run_contract_backtest(runtime, cfg, slip_bps=3.0, funding_bps_per_bar=1.0)
        if not all(metric_ok(x) for x in [full, o70, o80, stress]):
            continue

        score = score_candidate(full, o70, o80, stress)
        records.append(
            {
                "trial": i + 1,
                "params": params,
                "score": score,
                "full": full,
                "oos_30pct": o70,
                "oos_20pct": o80,
                "stress_mild": stress,
                "strict_target_pass": strict_target_pass(full, o70, o80, stress),
                "ultimate_target_pass": ultimate_target_pass(full, o70, o80, stress),
            }
        )

    records = sorted(records, key=lambda x: x["score"], reverse=True)
    strict = [x for x in records if x["strict_target_pass"]]
    ultimate = [x for x in records if x.get("ultimate_target_pass")]
    out = {
        "ts": datetime.now().isoformat(),
        "engine_filter": args.engine,
        "trials_requested": args.trials,
        "trials_completed": len(records),
        "cutoff_30pct": str(pd.Timestamp(cut70)),
        "cutoff_20pct": str(pd.Timestamp(cut80)),
        "best_by_score": records[0] if records else None,
        "best_strict_target": strict[0] if strict else None,
        "best_ultimate_target": ultimate[0] if ultimate else None,
        "strict_target_count": len(strict),
        "ultimate_target_count": len(ultimate),
        "top10": records[:10],
    }

    out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, "crypto_contract_search.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[crypto-opt] report -> {out_file}")


if __name__ == "__main__":
    main()
