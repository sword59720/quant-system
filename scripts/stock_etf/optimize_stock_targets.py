#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
    EXIT_SIGNAL_ERROR,
)
from scripts.stock_etf.paper_forward_stock import run_paper_forward

TARGET_ANNUAL_RETURN = 0.20
TARGET_MAX_DRAWDOWN = -0.10
TARGET_SHARPE = 1.00


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
            "benchmark_annual_return_alloc": 0.0,
            "excess_annual_return_vs_alloc": 0.0,
            "strategy_max_drawdown": 0.0,
            "benchmark_max_drawdown_alloc": 0.0,
            "strategy_sharpe": 0.0,
            "benchmark_sharpe_alloc": 0.0,
        }

    strategy_nav = pd.Series([1.0] + x["strategy_nav"].tolist())
    benchmark_nav = pd.Series([1.0] + x["benchmark_nav_alloc"].tolist())
    strategy_ann = annualized_return(strategy_nav, 252)
    benchmark_ann = annualized_return(benchmark_nav, 252)
    return {
        "rows": int(len(x)),
        "strategy_annual_return": float(strategy_ann),
        "benchmark_annual_return_alloc": float(benchmark_ann),
        "excess_annual_return_vs_alloc": float(strategy_ann - benchmark_ann),
        "strategy_max_drawdown": max_drawdown(strategy_nav),
        "benchmark_max_drawdown_alloc": max_drawdown(benchmark_nav),
        "strategy_sharpe": sharpe_ratio(x["strategy_ret"], 252),
        "benchmark_sharpe_alloc": sharpe_ratio(x["benchmark_ret_alloc"], 252),
    }


def _uniq_int(values: list[int], lo: int, hi: int) -> list[int]:
    return sorted(set(int(v) for v in values if lo <= int(v) <= hi))


def _uniq_float(values: list[float], lo: float, hi: float, ndigits: int = 4) -> list[float]:
    out = []
    for v in values:
        x = round(float(v), ndigits)
        if lo <= x <= hi:
            out.append(float(x))
    return sorted(set(out))


def extract_base_candidate(stock_cfg: dict) -> dict:
    gm = stock_cfg.get("global_model", {})
    guard = gm.get("execution_guard", {})
    dyn = guard.get("dynamic_min_turnover", {})
    gate = gm.get("exposure_gate", {})
    thr = gate.get("thresholds", {})
    caps = gate.get("alloc_caps", {})
    alloc = float(stock_cfg.get("capital_alloc_pct", 0.7))
    pass_cap = float(caps.get("pass", alloc))
    warn_cap = float(caps.get("warn", min(pass_cap, alloc)))
    risk_cap = float(caps.get("risk", min(warn_cap, alloc)))
    return {
        "capital_alloc_pct": alloc,
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
        "exposure_gate_lookback_days": [int(x) for x in gate.get("lookback_days", [63, 126])],
        "exposure_warn_worst_min": float(thr.get("warn_worst_min", -0.10)),
        "exposure_risk_worst_min": float(thr.get("risk_worst_min", -0.14)),
        "exposure_cap_pass": float(pass_cap),
        "exposure_cap_warn": float(warn_cap),
        "exposure_cap_risk": float(risk_cap),
    }


def apply_candidate(stock_cfg: dict, candidate: dict) -> dict:
    s = copy.deepcopy(stock_cfg)
    s["capital_alloc_pct"] = float(candidate["capital_alloc_pct"])

    gm = s.setdefault("global_model", {})
    gm["momentum_lb"] = int(candidate["momentum_lb"])
    gm["ma_window"] = int(candidate["ma_window"])
    gm["vol_window"] = int(candidate["vol_window"])
    gm["top_n"] = int(candidate["top_n"])
    gm["min_score"] = float(candidate.get("min_score", 0.0))
    gm["rebalance_days"] = int(candidate.get("rebalance_days", gm.get("rebalance_days", 20)))

    guard = gm.setdefault("execution_guard", {})
    guard["enabled"] = bool(candidate.get("guard_enabled", True))
    guard["min_rebalance_days"] = int(candidate["min_rebalance_days"])
    guard["min_turnover"] = float(candidate["min_turnover"])
    guard["force_rebalance_on_regime_change"] = bool(candidate["force_rebalance_on_regime_change"])

    dyn = guard.setdefault("dynamic_min_turnover", {})
    dyn["enabled"] = bool(candidate.get("dynamic_min_turnover_enabled", dyn.get("enabled", False)))

    gate = gm.setdefault("exposure_gate", {})
    gate_enabled = bool(candidate.get("exposure_gate_enabled", gate.get("enabled", False)))
    gate["enabled"] = gate_enabled
    if gate_enabled:
        base_alloc = float(s["capital_alloc_pct"])
        gate["lookback_days"] = [int(x) for x in candidate.get("exposure_gate_lookback_days", [63, 126])]
        thr = gate.setdefault("thresholds", {})
        thr["warn_worst_min"] = float(candidate.get("exposure_warn_worst_min", -0.10))
        thr["risk_worst_min"] = float(candidate.get("exposure_risk_worst_min", -0.14))

        cap_pass = float(candidate.get("exposure_cap_pass", base_alloc))
        cap_warn = float(candidate.get("exposure_cap_warn", min(base_alloc, cap_pass)))
        cap_risk = float(candidate.get("exposure_cap_risk", min(cap_warn, base_alloc)))
        cap_pass = min(cap_pass, base_alloc)
        cap_warn = min(cap_warn, cap_pass)
        cap_risk = min(cap_risk, cap_warn)

        caps = gate.setdefault("alloc_caps", {})
        caps["pass"] = float(cap_pass)
        caps["warn"] = float(cap_warn)
        caps["risk"] = float(cap_risk)

    return s


def evaluate_target(full_metrics: dict) -> dict:
    ann = float(full_metrics["strategy_annual_return"])
    mdd = float(full_metrics["strategy_max_drawdown"])
    shp = float(full_metrics["strategy_sharpe"])

    ann_gap = max(0.0, TARGET_ANNUAL_RETURN - ann)
    mdd_gap = max(0.0, TARGET_MAX_DRAWDOWN - mdd)
    shp_gap = max(0.0, TARGET_SHARPE - shp)

    met_ann = ann >= TARGET_ANNUAL_RETURN
    met_mdd = mdd >= TARGET_MAX_DRAWDOWN
    met_shp = shp >= TARGET_SHARPE
    met_count = int(met_ann) + int(met_mdd) + int(met_shp)

    # Gap weighted by practical importance for this target profile.
    gap_score = float((8.0 * ann_gap) + (14.0 * mdd_gap) + (6.0 * shp_gap))
    return {
        "targets_met_count": met_count,
        "all_met": bool(met_count == 3),
        "met": {
            "annual_return": bool(met_ann),
            "max_drawdown": bool(met_mdd),
            "sharpe": bool(met_shp),
        },
        "gaps": {
            "annual_return": float(ann_gap),
            "max_drawdown": float(mdd_gap),
            "sharpe": float(shp_gap),
        },
        "gap_score": gap_score,
        "targets": {
            "annual_return": TARGET_ANNUAL_RETURN,
            "max_drawdown": TARGET_MAX_DRAWDOWN,
            "sharpe": TARGET_SHARPE,
        },
    }


def score_candidate(metrics: dict, trades: int) -> tuple[float, dict]:
    full_m = metrics["full"]
    o23 = metrics["oos_2023"]
    o24 = metrics["oos_2024"]
    o25 = metrics["oos_2025"]
    target = evaluate_target(full_m)

    robust_ann = (
        0.4 * float(o23["strategy_annual_return"])
        + 0.8 * float(o24["strategy_annual_return"])
        + 1.0 * float(o25["strategy_annual_return"])
    )
    score = (
        200.0 * target["targets_met_count"]
        - 300.0 * target["gap_score"]
        + 25.0 * float(full_m["strategy_annual_return"])
        + 8.0 * float(full_m["strategy_sharpe"])
        + 15.0 * float(full_m["strategy_max_drawdown"])
        + 4.0 * robust_ann
        - 0.03 * float(trades)
    )
    return float(score), target


def is_stable(metrics: dict) -> bool:
    full_m = metrics["full"]
    o23 = metrics["oos_2023"]
    o24 = metrics["oos_2024"]
    o25 = metrics["oos_2025"]
    return bool(
        full_m["strategy_annual_return"] > 0
        and o23["strategy_annual_return"] > 0
        and o24["strategy_annual_return"] > 0
        and o25["strategy_annual_return"] > 0
        and full_m["strategy_max_drawdown"] > -0.18
    )


def evaluate_candidate(runtime: dict, stock_cfg: dict, risk_cfg: dict, candidate: dict) -> dict:
    eval_stock = apply_candidate(stock_cfg, candidate)
    with tempfile.TemporaryDirectory() as td:
        rt = copy.deepcopy(runtime)
        rt["paths"]["output_dir"] = td
        summary, history_file, _latest_file = run_paper_forward(rt, eval_stock, risk_cfg)
        df = pd.read_csv(history_file)

    df["date"] = pd.to_datetime(df["date"])
    metrics = {
        "full": compute_window_metrics(df, None),
        "oos_2023": compute_window_metrics(df, "2023-01-03"),
        "oos_2024": compute_window_metrics(df, "2024-01-02"),
        "oos_2025": compute_window_metrics(df, "2025-01-02"),
    }
    score, target = score_candidate(metrics, int(summary["aggregate"]["trades"]))
    return {
        "params": {
            "capital_alloc_pct": float(candidate["capital_alloc_pct"]),
            "momentum_lb": int(candidate["momentum_lb"]),
            "ma_window": int(candidate["ma_window"]),
            "vol_window": int(candidate["vol_window"]),
            "top_n": int(candidate["top_n"]),
            "min_score": float(candidate.get("min_score", 0.0)),
            "rebalance_days": int(candidate.get("rebalance_days", 20)),
            "guard_enabled": bool(candidate.get("guard_enabled", True)),
            "min_rebalance_days": int(candidate["min_rebalance_days"]),
            "min_turnover": float(candidate["min_turnover"]),
            "force_rebalance_on_regime_change": bool(candidate["force_rebalance_on_regime_change"]),
            "dynamic_min_turnover_enabled": bool(candidate.get("dynamic_min_turnover_enabled", True)),
            "exposure_gate_enabled": bool(candidate.get("exposure_gate_enabled", False)),
            "exposure_warn_worst_min": float(candidate.get("exposure_warn_worst_min", -0.10)),
            "exposure_risk_worst_min": float(candidate.get("exposure_risk_worst_min", -0.14)),
            "exposure_cap_pass": float(candidate.get("exposure_cap_pass", candidate["capital_alloc_pct"])),
            "exposure_cap_warn": float(
                candidate.get("exposure_cap_warn", min(candidate["capital_alloc_pct"], candidate["capital_alloc_pct"]))
            ),
            "exposure_cap_risk": float(
                candidate.get("exposure_cap_risk", min(candidate["capital_alloc_pct"], candidate["capital_alloc_pct"]))
            ),
        },
        "metrics": metrics,
        "aggregate": {
            "trades": int(summary["aggregate"]["trades"]),
            "total_turnover": float(summary["aggregate"]["total_turnover"]),
            "rolling_excess_20d_vs_alloc": float(summary["rolling"]["excess_return_20d_vs_alloc"]),
            "rolling_excess_60d_vs_alloc": float(summary["rolling"]["excess_return_60d_vs_alloc"]),
        },
        "target": target,
        "stable_pass": is_stable(metrics),
        "score": score,
    }


def stage1_candidates(stock_cfg: dict) -> list[dict]:
    base = extract_base_candidate(stock_cfg)
    out = []
    for alloc in [0.70, 0.85, 1.00]:
        for momentum_lb in [126, 252]:
            for ma_window in [160, 180, 200]:
                for vol_window in [20, 50]:
                    for top_n in [1, 2, 4]:
                        for rebalance_days in [10, 20]:
                            c = dict(base)
                            c.update(
                                {
                                    "capital_alloc_pct": float(alloc),
                                    "momentum_lb": int(momentum_lb),
                                    "ma_window": int(ma_window),
                                    "vol_window": int(vol_window),
                                    "top_n": int(top_n),
                                    "rebalance_days": int(rebalance_days),
                                    "exposure_gate_enabled": False,
                                }
                            )
                            out.append(c)
    return out


def stage2_candidates(top_core: list[dict]) -> list[dict]:
    out = []
    for core in top_core:
        p = dict(core["params"])
        for min_days in [10, 20, 30]:
            for min_turn in [0.01, 0.03, 0.05]:
                for force_rebalance in [False, True]:
                    for guard_enabled in [True, False]:
                        for dyn_enabled in [True, False]:
                            c = dict(p)
                            c.update(
                                {
                                    "min_rebalance_days": int(min_days),
                                    "min_turnover": float(min_turn),
                                    "force_rebalance_on_regime_change": bool(force_rebalance),
                                    "guard_enabled": bool(guard_enabled),
                                    "dynamic_min_turnover_enabled": bool(dyn_enabled),
                                    "exposure_gate_enabled": False,
                                }
                            )
                            out.append(c)
    return out


def stage3_candidates(best_stage2: dict) -> list[dict]:
    p = best_stage2["params"]
    momentum_grid = _uniq_int([p["momentum_lb"] - 32, p["momentum_lb"], p["momentum_lb"] + 32], 60, 320)
    ma_grid = _uniq_int([p["ma_window"] - 20, p["ma_window"], p["ma_window"] + 20], 80, 260)
    vol_grid = _uniq_int([p["vol_window"] - 10, p["vol_window"], p["vol_window"] + 10], 5, 80)
    alloc_grid = _uniq_float(
        [p["capital_alloc_pct"] - 0.10, p["capital_alloc_pct"], p["capital_alloc_pct"] + 0.10], 0.50, 1.20, 3
    )
    min_turn_grid = _uniq_float([p["min_turnover"] - 0.01, p["min_turnover"], p["min_turnover"] + 0.01], 0.005, 0.12, 4)

    out = []
    for momentum_lb in momentum_grid:
        for ma_window in ma_grid:
            for vol_window in vol_grid:
                for alloc in alloc_grid:
                    for min_turn in min_turn_grid:
                        c = dict(p)
                        c.update(
                            {
                                "capital_alloc_pct": float(alloc),
                                "momentum_lb": int(momentum_lb),
                                "ma_window": int(ma_window),
                                "vol_window": int(vol_window),
                                "min_turnover": float(min_turn),
                                "exposure_gate_enabled": False,
                            }
                        )
                        out.append(c)
    return out


def stage4_exposure_candidates(best_stage3: dict) -> list[dict]:
    p = best_stage3["params"]
    alloc = float(p["capital_alloc_pct"])
    out = []

    base_disabled = dict(p)
    base_disabled["exposure_gate_enabled"] = False
    out.append(base_disabled)

    for warn_thr in [-0.06, -0.08, -0.10]:
        for risk_thr in [-0.10, -0.12, -0.14]:
            if risk_thr > warn_thr:
                continue
            for warn_factor in [0.85, 0.90]:
                for risk_factor in [0.65, 0.75]:
                    cap_pass = alloc
                    cap_warn = min(alloc, alloc * warn_factor)
                    cap_risk = min(cap_warn, alloc * risk_factor)
                    c = dict(p)
                    c.update(
                        {
                            "exposure_gate_enabled": True,
                            "exposure_gate_lookback_days": [63, 126],
                            "exposure_warn_worst_min": float(warn_thr),
                            "exposure_risk_worst_min": float(risk_thr),
                            "exposure_cap_pass": float(cap_pass),
                            "exposure_cap_warn": float(cap_warn),
                            "exposure_cap_risk": float(cap_risk),
                        }
                    )
                    out.append(c)
    return out


def sort_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda x: (
            int(x["target"]["targets_met_count"]),
            bool(x["target"]["all_met"]),
            -float(x["target"]["gap_score"]),
            bool(x["stable_pass"]),
            float(x["metrics"]["full"]["strategy_annual_return"]),
            float(x["metrics"]["full"]["strategy_sharpe"]),
            float(x["metrics"]["full"]["strategy_max_drawdown"]),
            float(x["metrics"]["oos_2025"]["strategy_annual_return"]),
            -int(x["aggregate"]["trades"]),
            float(x["score"]),
        ),
        reverse=True,
    )


def pick_recommended(results: list[dict]) -> dict:
    strict = [r for r in results if r["target"]["all_met"] and r["stable_pass"]]
    if strict:
        return strict[0]
    all_met = [r for r in results if r["target"]["all_met"]]
    if all_met:
        return all_met[0]
    stable = [r for r in results if r["stable_pass"]]
    if stable:
        return stable[0]
    return results[0]


def run_stage(stage_name: str, runtime: dict, stock_cfg: dict, risk_cfg: dict, candidates: list[dict]) -> list[dict]:
    total = len(candidates)
    results = []
    for i, c in enumerate(candidates, 1):
        result = evaluate_candidate(runtime, stock_cfg, risk_cfg, c)
        result["stage"] = stage_name
        results.append(result)
        if i % 20 == 0 or i == total:
            print(f"[target-opt:{stage_name}] {i}/{total}", flush=True)
    return sort_results(results)


def main():
    parser = argparse.ArgumentParser(description="Optimize stock production model toward return/DD/Sharpe targets.")
    parser.add_argument(
        "--output",
        default="outputs/reports/stock_target_optimization_report.json",
        help="output json report path",
    )
    args = parser.parse_args()

    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
        risk_cfg = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[target-opt] config error: {e}")
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
        out_dir = os.path.dirname(args.output) or "."
        ensure_dir(out_dir)

        baseline_params = extract_base_candidate(stock_cfg)
        baseline = evaluate_candidate(runtime, stock_cfg, risk_cfg, baseline_params)
        baseline["stage"] = "baseline"

        s1_results = run_stage("stage1_core_alloc", runtime, stock_cfg, risk_cfg, stage1_candidates(stock_cfg))
        top_core = s1_results[:4]

        s2_results = run_stage("stage2_guard", runtime, stock_cfg, risk_cfg, stage2_candidates(top_core))
        best_stage2 = pick_recommended(s2_results)

        s3_results = run_stage("stage3_local", runtime, stock_cfg, risk_cfg, stage3_candidates(best_stage2))
        best_stage3 = pick_recommended(s3_results)

        s4_results = run_stage("stage4_exposure", runtime, stock_cfg, risk_cfg, stage4_exposure_candidates(best_stage3))

        all_results = sort_results([baseline] + s1_results + s2_results + s3_results + s4_results)
        recommended = pick_recommended(all_results)

        report = {
            "ts": datetime.now().isoformat(),
            "note": "stock production target optimization (annual>=20%, mdd>=-10%, sharpe>=1.0)",
            "targets": {
                "annual_return_min": TARGET_ANNUAL_RETURN,
                "max_drawdown_min": TARGET_MAX_DRAWDOWN,
                "sharpe_min": TARGET_SHARPE,
            },
            "baseline": baseline,
            "stages": {
                "stage1_count": len(s1_results),
                "stage2_count": len(s2_results),
                "stage3_count": len(s3_results),
                "stage4_count": len(s4_results),
                "stage1_top5": s1_results[:5],
                "stage2_top5": s2_results[:5],
                "stage3_top8": s3_results[:8],
                "stage4_top8": s4_results[:8],
            },
            "recommended": recommended,
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
                "targets_met_count_delta": int(
                    recommended["target"]["targets_met_count"] - baseline["target"]["targets_met_count"]
                ),
            },
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[target-opt] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[target-opt] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    print(f"[target-opt] report -> {args.output}")
    print(
        "[target-opt] recommended params:",
        json.dumps(report["recommended"]["params"], ensure_ascii=False),
    )
    print(
        "[target-opt] recommended full metrics:",
        json.dumps(report["recommended"]["metrics"]["full"], ensure_ascii=False),
    )
    print(
        "[target-opt] targets met:",
        json.dumps(report["recommended"]["target"], ensure_ascii=False),
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
