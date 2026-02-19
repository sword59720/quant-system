#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
    EXIT_SIGNAL_ERROR,
)
from scripts.paper_forward_stock import run_paper_forward


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


def compute_window_metrics(df: pd.DataFrame, start_date: Optional[str]) -> dict:
    x = df if start_date is None else df[df["date"] >= pd.Timestamp(start_date)]
    if len(x) < 2:
        return {
            "rows": int(len(x)),
            "strategy_annual_return": 0.0,
            "strategy_max_drawdown": 0.0,
            "strategy_sharpe": 0.0,
        }

    strategy_nav = pd.Series([1.0] + x["strategy_nav"].tolist())
    return {
        "rows": int(len(x)),
        "strategy_annual_return": float(annualized_return(strategy_nav, 252)),
        "strategy_max_drawdown": float(max_drawdown(strategy_nav)),
        "strategy_sharpe": float(sharpe_ratio(x["strategy_ret"], 252)),
    }


def base_candidate(stock_cfg: dict) -> dict:
    gm = stock_cfg.get("global_model", {})
    guard = gm.get("execution_guard", {})
    dyn = guard.get("dynamic_min_turnover", {})
    gate = gm.get("exposure_gate", {})
    thr = gate.get("thresholds", {})
    caps = gate.get("alloc_caps", {})
    return {
        "capital_alloc_pct": 1.0,
        "momentum_lb": int(gm.get("momentum_lb", 252)),
        "ma_window": int(gm.get("ma_window", 200)),
        "vol_window": int(gm.get("vol_window", 20)),
        "top_n": int(gm.get("top_n", 1)),
        "min_score": float(gm.get("min_score", 0.0)),
        "rebalance_days": int(gm.get("rebalance_days", 20)),
        "guard_enabled": bool(guard.get("enabled", True)),
        "min_rebalance_days": int(guard.get("min_rebalance_days", gm.get("rebalance_days", 20))),
        "min_turnover": float(guard.get("min_turnover", 0.05)),
        "force_rebalance_on_regime_change": bool(guard.get("force_rebalance_on_regime_change", True)),
        "dynamic_min_turnover_enabled": bool(dyn.get("enabled", False)),
        "exposure_gate_enabled": bool(gate.get("enabled", False)),
        "exposure_warn_worst_min": float(thr.get("warn_worst_min", -0.10)),
        "exposure_risk_worst_min": float(thr.get("risk_worst_min", -0.14)),
        "exposure_cap_pass": float(caps.get("pass", 1.0)),
        "exposure_cap_warn": float(caps.get("warn", 0.9)),
        "exposure_cap_risk": float(caps.get("risk", 0.8)),
    }


def apply_candidate(stock_cfg: dict, candidate: dict) -> dict:
    s = copy.deepcopy(stock_cfg)
    s["capital_alloc_pct"] = 1.0
    gm = s.setdefault("global_model", {})
    gm["momentum_lb"] = int(candidate["momentum_lb"])
    gm["ma_window"] = int(candidate["ma_window"])
    gm["vol_window"] = int(candidate["vol_window"])
    gm["top_n"] = int(candidate["top_n"])
    gm["min_score"] = float(candidate.get("min_score", 0.0))
    gm["rebalance_days"] = int(candidate["rebalance_days"])

    guard = gm.setdefault("execution_guard", {})
    guard["enabled"] = bool(candidate["guard_enabled"])
    guard["min_rebalance_days"] = int(candidate["min_rebalance_days"])
    guard["min_turnover"] = float(candidate["min_turnover"])
    guard["force_rebalance_on_regime_change"] = bool(candidate["force_rebalance_on_regime_change"])
    dyn = guard.setdefault("dynamic_min_turnover", {})
    dyn["enabled"] = bool(candidate["dynamic_min_turnover_enabled"])

    gate = gm.setdefault("exposure_gate", {})
    gate["enabled"] = bool(candidate["exposure_gate_enabled"])
    if gate["enabled"]:
        thr = gate.setdefault("thresholds", {})
        thr["warn_worst_min"] = float(candidate["exposure_warn_worst_min"])
        thr["risk_worst_min"] = float(candidate["exposure_risk_worst_min"])
        caps = gate.setdefault("alloc_caps", {})
        caps["pass"] = min(1.0, float(candidate["exposure_cap_pass"]))
        caps["warn"] = min(caps["pass"], float(candidate["exposure_cap_warn"]))
        caps["risk"] = min(caps["warn"], float(candidate["exposure_cap_risk"]))
    return s


def evaluate_candidate(runtime: dict, stock_cfg: dict, risk_cfg: dict, candidate: dict) -> dict:
    eval_stock = apply_candidate(stock_cfg, candidate)
    with tempfile.TemporaryDirectory() as td:
        rt = copy.deepcopy(runtime)
        rt["paths"]["output_dir"] = td
        summary, history_file, _latest = run_paper_forward(rt, eval_stock, risk_cfg)
        df = pd.read_csv(history_file)

    df["date"] = pd.to_datetime(df["date"])
    agg = summary["aggregate"]
    metrics = {
        "full": {
            "strategy_annual_return": float(agg["strategy_annual_return"]),
            "strategy_max_drawdown": float(agg["strategy_max_drawdown"]),
            "strategy_sharpe": float(agg["strategy_sharpe"]),
            "rows": int(agg["periods"]),
        },
        "oos_2023": compute_window_metrics(df, "2023-01-03"),
        "oos_2024": compute_window_metrics(df, "2024-01-02"),
        "oos_2025": compute_window_metrics(df, "2025-01-02"),
    }
    return {
        "params": {
            "capital_alloc_pct": 1.0,
            "momentum_lb": int(candidate["momentum_lb"]),
            "ma_window": int(candidate["ma_window"]),
            "vol_window": int(candidate["vol_window"]),
            "top_n": int(candidate["top_n"]),
            "min_score": float(candidate["min_score"]),
            "rebalance_days": int(candidate["rebalance_days"]),
            "guard_enabled": bool(candidate["guard_enabled"]),
            "min_rebalance_days": int(candidate["min_rebalance_days"]),
            "min_turnover": float(candidate["min_turnover"]),
            "force_rebalance_on_regime_change": bool(candidate["force_rebalance_on_regime_change"]),
            "dynamic_min_turnover_enabled": bool(candidate["dynamic_min_turnover_enabled"]),
            "exposure_gate_enabled": bool(candidate["exposure_gate_enabled"]),
            "exposure_warn_worst_min": float(candidate["exposure_warn_worst_min"]),
            "exposure_risk_worst_min": float(candidate["exposure_risk_worst_min"]),
            "exposure_cap_pass": float(candidate["exposure_cap_pass"]),
            "exposure_cap_warn": float(candidate["exposure_cap_warn"]),
            "exposure_cap_risk": float(candidate["exposure_cap_risk"]),
        },
        "metrics": metrics,
        "aggregate": {
            "trades": int(agg["trades"]),
            "total_turnover": float(agg["total_turnover"]),
            "rolling_excess_20d_vs_alloc": float(summary["rolling"]["excess_return_20d_vs_alloc"]),
            "rolling_excess_60d_vs_alloc": float(summary["rolling"]["excess_return_60d_vs_alloc"]),
        },
    }


def stage1_candidates(stock_cfg: dict) -> list[dict]:
    base = base_candidate(stock_cfg)
    out = []
    for momentum_lb in [126, 252]:
        for ma_window in [160, 180, 200]:
            for vol_window in [20, 50]:
                for top_n in [1, 2, 4]:
                    for rebalance_days in [10, 20]:
                        for min_rebalance_days in [20, 30]:
                            for min_turnover in [0.01, 0.03]:
                                for force_regime in [False, True]:
                                    c = dict(base)
                                    c.update(
                                        {
                                            "momentum_lb": int(momentum_lb),
                                            "ma_window": int(ma_window),
                                            "vol_window": int(vol_window),
                                            "top_n": int(top_n),
                                            "rebalance_days": int(rebalance_days),
                                            "min_rebalance_days": int(min_rebalance_days),
                                            "min_turnover": float(min_turnover),
                                            "force_rebalance_on_regime_change": bool(force_regime),
                                            "exposure_gate_enabled": False,
                                        }
                                    )
                                    out.append(c)
    return out


def stage2_exposure_candidates(best_core: list[dict]) -> list[dict]:
    out = []
    for row in best_core:
        p = dict(row["params"])
        # keep one baseline without exposure gate
        p0 = dict(p)
        p0["exposure_gate_enabled"] = False
        out.append(p0)

        for warn_thr in [-0.06, -0.08, -0.10]:
            for risk_thr in [-0.10, -0.12, -0.14]:
                if risk_thr > warn_thr:
                    continue
                for warn_factor in [0.90, 0.95]:
                    for risk_factor in [0.75, 0.85]:
                        c = dict(p)
                        c.update(
                            {
                                "exposure_gate_enabled": True,
                                "exposure_warn_worst_min": float(warn_thr),
                                "exposure_risk_worst_min": float(risk_thr),
                                "exposure_cap_pass": 1.0,
                                "exposure_cap_warn": float(min(1.0, warn_factor)),
                                "exposure_cap_risk": float(min(warn_factor, risk_factor)),
                            }
                        )
                        out.append(c)
    return out


def run_stage(name: str, runtime: dict, stock_cfg: dict, risk_cfg: dict, candidates: list[dict]) -> list[dict]:
    out = []
    total = len(candidates)
    for i, c in enumerate(candidates, 1):
        r = evaluate_candidate(runtime, stock_cfg, risk_cfg, c)
        r["stage"] = name
        out.append(r)
        if i % 20 == 0 or i == total:
            print(f"[dd-opt:{name}] {i}/{total}", flush=True)
    return out


def pick_best(results: list[dict], ann_floor: float) -> tuple[dict, bool]:
    feasible = [r for r in results if float(r["metrics"]["full"]["strategy_annual_return"]) >= ann_floor]
    if feasible:
        best = sorted(
            feasible,
            key=lambda r: (
                float(r["metrics"]["full"]["strategy_max_drawdown"]),
                float(r["metrics"]["full"]["strategy_annual_return"]),
                float(r["metrics"]["full"]["strategy_sharpe"]),
                -int(r["aggregate"]["trades"]),
            ),
            reverse=True,
        )[0]
        return best, True

    best = sorted(
        results,
        key=lambda r: (
            float(r["metrics"]["full"]["strategy_annual_return"]),
            float(r["metrics"]["full"]["strategy_max_drawdown"]),
            float(r["metrics"]["full"]["strategy_sharpe"]),
        ),
        reverse=True,
    )[0]
    return best, False


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
        risk_cfg = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[dd-opt] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_cfg.get("enabled", False):
        print("[stock] disabled")
        return EXIT_DISABLED
    if stock_cfg.get("mode") != "global_momentum":
        print("[stock] mode is not global_momentum, skip optimization")
        return EXIT_DISABLED

    try:
        out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(out_dir)

        baseline_candidate = base_candidate(stock_cfg)
        baseline = evaluate_candidate(runtime, stock_cfg, risk_cfg, baseline_candidate)
        baseline["stage"] = "baseline"
        ann_floor = float(baseline["metrics"]["full"]["strategy_annual_return"])

        s1 = run_stage("stage1_core", runtime, stock_cfg, risk_cfg, stage1_candidates(stock_cfg))
        # keep top 4 by annual return for exposure refinement
        top_core = sorted(
            s1,
            key=lambda r: (
                float(r["metrics"]["full"]["strategy_annual_return"]),
                float(r["metrics"]["full"]["strategy_sharpe"]),
            ),
            reverse=True,
        )[:4]
        s2 = run_stage("stage2_exposure", runtime, stock_cfg, risk_cfg, stage2_exposure_candidates(top_core))

        all_results = [baseline] + s1 + s2
        recommended, feasible_found = pick_best(all_results, ann_floor)

        report = {
            "ts": datetime.now().isoformat(),
            "note": "alloc=1.0 drawdown optimization with annual-return floor",
            "constraint": {
                "alloc_fixed": 1.0,
                "annual_return_floor": ann_floor,
            },
            "baseline": baseline,
            "stage1_count": len(s1),
            "stage2_count": len(s2),
            "stage1_top10_by_annual": sorted(
                s1,
                key=lambda r: float(r["metrics"]["full"]["strategy_annual_return"]),
                reverse=True,
            )[:10],
            "stage2_top10_by_annual": sorted(
                s2,
                key=lambda r: float(r["metrics"]["full"]["strategy_annual_return"]),
                reverse=True,
            )[:10],
            "recommended": recommended,
            "feasible_found": bool(feasible_found),
            "improvement_vs_baseline_full": {
                "annual_return_delta": float(
                    recommended["metrics"]["full"]["strategy_annual_return"] - baseline["metrics"]["full"]["strategy_annual_return"]
                ),
                "max_drawdown_delta": float(
                    recommended["metrics"]["full"]["strategy_max_drawdown"] - baseline["metrics"]["full"]["strategy_max_drawdown"]
                ),
                "sharpe_delta": float(
                    recommended["metrics"]["full"]["strategy_sharpe"] - baseline["metrics"]["full"]["strategy_sharpe"]
                ),
            },
        }

        out_file = os.path.join(out_dir, "stock_alloc1_drawdown_optimization.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[dd-opt] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[dd-opt] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    print(f"[dd-opt] report -> {out_file}")
    print("[dd-opt] feasible_found:", report["feasible_found"])
    print("[dd-opt] baseline_full:", json.dumps(report["baseline"]["metrics"]["full"], ensure_ascii=False))
    print("[dd-opt] recommended_full:", json.dumps(report["recommended"]["metrics"]["full"], ensure_ascii=False))
    print("[dd-opt] recommended_params:", json.dumps(report["recommended"]["params"], ensure_ascii=False))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
