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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.validate_crypto_contract import run_contract_backtest


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def pick_dates(crypto_cfg, date_from=None):
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
    if date_from:
        cutoff = pd.Timestamp(date_from)
        dates = [d for d in dates if d >= cutoff]
    if len(dates) < 200:
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
    rm["vol_lookback_bars"] = int(params["rm_vol_lookback_bars"])
    rm["min_leverage"] = float(params["min_leverage"])
    rm["max_leverage"] = float(params["max_leverage"])

    dd = signal.setdefault("drawdown_throttle", {})
    dd["enabled"] = True
    dd["trigger_dd"] = float(params["trigger_dd"])
    dd["reduced_alloc_multiplier"] = float(params["reduced_alloc_multiplier"])

    defense["risk_off_threshold_pct"] = float(params["risk_off_threshold_pct"])
    return cfg


def mget(m, key, default=0.0):
    return float(m.get(key, default)) if isinstance(m, dict) else float(default)


def objective_score(full, o70, o80, stress):
    full_ann = mget(full, "annual_return")
    full_sh = mget(full, "sharpe")
    full_dd = mget(full, "max_drawdown")
    o70_ann = mget(o70, "annual_return")
    o80_ann = mget(o80, "annual_return")
    o70_sh = mget(o70, "sharpe")
    o80_sh = mget(o80, "sharpe")
    stress_ann = mget(stress, "annual_return")
    stress_dd = mget(stress, "max_drawdown")

    robust_oos_ann = min(o70_ann, o80_ann)
    robust_oos_sh = min(o70_sh, o80_sh)
    dd_worst = min(full_dd, mget(o70, "max_drawdown"), mget(o80, "max_drawdown"), stress_dd)

    reward = (
        0.55 * full_ann
        + 0.20 * robust_oos_ann
        + 0.12 * stress_ann
        + 0.08 * full_sh
        + 0.05 * robust_oos_sh
    )

    dd_penalty = max(0.0, -0.10 - dd_worst) * 6.0
    sharpe_penalty = max(0.0, 1.0 - full_sh) * 0.25
    return reward - dd_penalty - sharpe_penalty


def hard_target_pass(full, min_ann, min_sharpe, max_dd):
    return (
        mget(full, "annual_return") >= float(min_ann)
        and mget(full, "sharpe") >= float(min_sharpe)
        and mget(full, "max_drawdown") >= float(max_dd)
    )


def robust_guard_pass(o70, o80, stress):
    return (
        mget(o70, "annual_return") >= 0.0
        and mget(o80, "annual_return") >= 0.0
        and mget(o70, "max_drawdown") >= -0.12
        and mget(o80, "max_drawdown") >= -0.12
        and mget(stress, "annual_return") >= 0.0
        and mget(stress, "max_drawdown") >= -0.15
    )


def main():
    parser = argparse.ArgumentParser(description="Target-driven crypto contract optimizer")
    parser.add_argument("--trials", type=int, default=80, help="random trials")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--min-ann", type=float, default=0.50, help="target min annual return")
    parser.add_argument("--max-dd", type=float, default=-0.10, help="target max drawdown floor")
    parser.add_argument("--min-sharpe", type=float, default=1.0, help="target min sharpe")
    parser.add_argument("--allow-short-only", action="store_true", help="force allow_short=true")
    parser.add_argument(
        "--engine",
        type=str,
        default="advanced_ls_rmm",
        choices=["advanced_rmm", "advanced_ls_rmm", "advanced_ls_cs"],
    )
    parser.add_argument("--stress-slip-bps", type=float, default=3.0)
    parser.add_argument("--stress-funding-bps", type=float, default=1.0)
    parser.add_argument("--date-from", type=str, default=None, help="backtest start date, e.g. 2024-10-08")
    parser.add_argument(
        "--report-file",
        type=str,
        default="crypto_target_search.json",
        help="report file name under outputs/reports",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    runtime = load_yaml("config/runtime.yaml")
    crypto = load_yaml("config/crypto.yaml")
    cut70, cut80 = pick_dates(crypto, date_from=args.date_from)
    if cut70 is None:
        raise SystemExit("[crypto-opt-target] no sufficient common date history")

    space = {
        "engine": [args.engine],
        "allow_short": [True] if args.allow_short_only else [False, True],
        "short_on_risk_off": [False, True],
        "top_n": [1, 2, 3],
        "short_top_n": [1, 2, 3],
        "momentum_lookback_bars": [[30, 90], [60, 120], [90, 180], [120, 240]],
        "momentum_weights": [[1.0, 0.0], [0.8, 0.2]],
        "ma_window_bars": [120, 180, 240, 360],
        "vol_window_bars": [20, 30, 40, 60],
        "momentum_threshold_pct": [0.0, 0.5, 1.0, 1.5, 2.0],
        "rebalance_threshold": [0.005, 0.01, 0.02, 0.03, 0.05],
        "max_exposure_pct": [0.6, 0.8, 1.0, 1.2, 1.5],
        "risk_off_threshold_pct": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0],
        "ls_long_alloc_pct": [0.2, 0.3, 0.4, 0.5, 0.6],
        "ls_short_alloc_pct": [0.0, 0.1, 0.2, 0.3, 0.4],
        "ls_short_requires_risk_off": [False, True],
        "ls_dynamic_budget_enabled": [False, True],
        "ls_regime_momentum_threshold_pct": [0.2, 0.5, 1.0, 1.5],
        "ls_regime_trend_breadth": [0.5, 0.6, 0.7],
        "ls_regime_up_long_alloc_pct": [0.4, 0.6, 0.8, 1.0],
        "ls_regime_up_short_alloc_pct": [0.0, 0.1, 0.2, 0.3],
        "ls_regime_neutral_long_alloc_pct": [0.2, 0.3, 0.4, 0.5],
        "ls_regime_neutral_short_alloc_pct": [0.0, 0.1, 0.2, 0.3, 0.4],
        "ls_neutral_exposure_multiplier": [0.4, 0.6, 0.8, 1.0, 1.2],
        "ls_regime_down_long_alloc_pct": [0.0, 0.1, 0.2, 0.3],
        "ls_regime_down_short_alloc_pct": [0.2, 0.4, 0.6, 0.8, 1.0],
        "ls_confidence_scaling_enabled": [False, True],
        "ls_confidence_min_spread": [0.08, 0.10, 0.12, 0.16, 0.20],
        "ls_confidence_full_spread": [0.20, 0.24, 0.28, 0.32, 0.40],
        "target_vol_annual": [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30],
        "rm_vol_lookback_bars": [20, 30, 40, 60, 90],
        "min_leverage": [0.2],
        "max_leverage": [1.0, 1.2, 1.5, 2.0, 2.5],
        "trigger_dd": [-0.03, -0.04, -0.05, -0.06, -0.08, -0.10],
        "reduced_alloc_multiplier": [0.0, 0.1, 0.2, 0.3, 0.5],
    }

    records = []
    for i in range(args.trials):
        params = {k: random.choice(v) for k, v in space.items()}
        if params["ma_window_bars"] < params["momentum_lookback_bars"][1]:
            params["ma_window_bars"] = params["momentum_lookback_bars"][1]
        if params["ls_confidence_full_spread"] <= params["ls_confidence_min_spread"]:
            params["ls_confidence_full_spread"] = round(params["ls_confidence_min_spread"] + 0.08, 4)

        # keep long/short budgets reasonable
        if params["ls_short_alloc_pct"] > params["ls_long_alloc_pct"] + 0.2:
            params["ls_short_alloc_pct"] = max(0.0, params["ls_long_alloc_pct"] - 0.1)

        # For non-LS engine, convert to directional mode.
        if params["engine"] not in {"advanced_ls_rmm", "advanced_ls_cs"}:
            params["ls_dynamic_budget_enabled"] = False
            params["ls_confidence_scaling_enabled"] = False
            params["ls_regime_up_long_alloc_pct"] = params["ls_long_alloc_pct"]
            params["ls_regime_up_short_alloc_pct"] = params["ls_short_alloc_pct"]
            params["ls_regime_neutral_long_alloc_pct"] = params["ls_long_alloc_pct"]
            params["ls_regime_neutral_short_alloc_pct"] = params["ls_short_alloc_pct"]
            params["ls_regime_down_long_alloc_pct"] = params["ls_long_alloc_pct"]
            params["ls_regime_down_short_alloc_pct"] = params["ls_short_alloc_pct"]

        if not params["allow_short"]:
            params["short_on_risk_off"] = False
            params["ls_short_alloc_pct"] = 0.0
            params["ls_regime_up_short_alloc_pct"] = 0.0
            params["ls_regime_neutral_short_alloc_pct"] = 0.0
            params["ls_regime_down_short_alloc_pct"] = 0.0

        cfg = apply_contract_params(crypto, params)
        full = run_contract_backtest(runtime, cfg, date_from=args.date_from)
        if "error" in full:
            continue

        # quick prune to save time
        if (mget(full, "annual_return") < -0.05) or (mget(full, "max_drawdown") < -0.35):
            continue

        o70 = run_contract_backtest(runtime, cfg, date_from=cut70)
        o80 = run_contract_backtest(runtime, cfg, date_from=cut80)
        stress = run_contract_backtest(
            runtime,
            cfg,
            slip_bps=float(args.stress_slip_bps),
            funding_bps_per_bar=float(args.stress_funding_bps),
        )
        if any("error" in x for x in [o70, o80, stress]):
            continue

        score = objective_score(full, o70, o80, stress)
        hit_hard_target = hard_target_pass(full, args.min_ann, args.min_sharpe, args.max_dd)
        hit_robust_guard = robust_guard_pass(o70, o80, stress)
        records.append(
            {
                "trial": i + 1,
                "score": float(score),
                "hard_target_pass": bool(hit_hard_target),
                "robust_guard_pass": bool(hit_robust_guard),
                "params": params,
                "full": full,
                "oos_30pct": o70,
                "oos_20pct": o80,
                "stress_mild": stress,
            }
        )
        print(
            f"[trial {i+1}/{args.trials}] score={score:.4f} "
            f"ann={mget(full,'annual_return'):.3f} dd={mget(full,'max_drawdown'):.3f} "
            f"sh={mget(full,'sharpe'):.2f} o70={mget(o70,'annual_return'):.3f} "
            f"o80={mget(o80,'annual_return'):.3f}",
            flush=True,
        )

    records = sorted(records, key=lambda x: x["score"], reverse=True)
    hard_pass = [x for x in records if x["hard_target_pass"]]
    robust_pass = [x for x in records if x["robust_guard_pass"]]
    robust_hard_pass = [x for x in records if x["hard_target_pass"] and x["robust_guard_pass"]]

    out = {
        "ts": datetime.now().isoformat(),
        "trials_requested": int(args.trials),
        "trials_kept": int(len(records)),
        "seed": int(args.seed),
        "cutoff_30pct": str(pd.Timestamp(cut70)),
        "cutoff_20pct": str(pd.Timestamp(cut80)),
        "target": {"min_ann": args.min_ann, "min_sharpe": args.min_sharpe, "max_dd": args.max_dd},
        "date_from": args.date_from,
        "engine": args.engine,
        "allow_short_only": bool(args.allow_short_only),
        "best_by_score": records[0] if records else None,
        "best_hard_target": hard_pass[0] if hard_pass else None,
        "best_robust_guard": robust_pass[0] if robust_pass else None,
        "best_hard_and_robust": robust_hard_pass[0] if robust_hard_pass else None,
        "hard_target_count": len(hard_pass),
        "robust_guard_count": len(robust_pass),
        "hard_and_robust_count": len(robust_hard_pass),
        "top10": records[:10],
    }

    out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, args.report_file)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[crypto-opt-target] report -> {out_file}")


if __name__ == "__main__":
    main()
