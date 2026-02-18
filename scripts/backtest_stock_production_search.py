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


def score_candidate(metrics: dict, trades: int) -> tuple[float, float]:
    full_m = metrics["full"]
    y23_m = metrics["oos_2023"]
    y24_m = metrics["oos_2024"]
    y25_m = metrics["oos_2025"]
    sharpe_delta = full_m["strategy_sharpe"] - full_m["benchmark_sharpe_alloc"]
    dd_improve = full_m["benchmark_max_drawdown_alloc"] - full_m["strategy_max_drawdown"]

    train_score = (
        1.2 * full_m["excess_annual_return_vs_alloc"]
        + 1.8 * y23_m["excess_annual_return_vs_alloc"]
        + 2.2 * y24_m["excess_annual_return_vs_alloc"]
        + 0.12 * sharpe_delta
        + 0.10 * dd_improve
        - 0.0004 * trades
    )
    overall_score = train_score + 2.2 * y25_m["excess_annual_return_vs_alloc"]
    return float(train_score), float(overall_score)


def is_stable(metrics: dict) -> bool:
    full_m = metrics["full"]
    y23_m = metrics["oos_2023"]
    y24_m = metrics["oos_2024"]
    y25_m = metrics["oos_2025"]
    return bool(
        full_m["excess_annual_return_vs_alloc"] > 0
        and y23_m["excess_annual_return_vs_alloc"] > 0
        and y24_m["excess_annual_return_vs_alloc"] > 0
        and y25_m["excess_annual_return_vs_alloc"] > 0
        and full_m["strategy_max_drawdown"] > -0.12
    )


def apply_candidate(stock_cfg: dict, candidate: dict) -> dict:
    s = copy.deepcopy(stock_cfg)
    gm = s["global_model"]

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
    return s


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
    train_score, overall_score = score_candidate(metrics, int(summary["aggregate"]["trades"]))
    out = {
        "params": {
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
        },
        "metrics": metrics,
        "aggregate": {
            "trades": int(summary["aggregate"]["trades"]),
            "total_turnover": float(summary["aggregate"]["total_turnover"]),
            "rolling_excess_20d_vs_alloc": float(summary["rolling"]["excess_return_20d_vs_alloc"]),
            "rolling_excess_60d_vs_alloc": float(summary["rolling"]["excess_return_60d_vs_alloc"]),
        },
        "stable_pass": is_stable(metrics),
        "train_score": train_score,
        "overall_score": overall_score,
    }
    return out


def stage1_candidates(stock_cfg: dict) -> list[dict]:
    guard = stock_cfg.get("global_model", {}).get("execution_guard", {})
    base_min_days = int(guard.get("min_rebalance_days", 20))
    base_min_turn = float(guard.get("min_turnover", 0.05))
    base_force = bool(guard.get("force_rebalance_on_regime_change", True))

    out = []
    for momentum_lb in [126, 252]:
        for ma_window in [120, 200]:
            for vol_window in [20, 40]:
                for top_n in [1, 2, 3, 4]:
                    out.append(
                        {
                            "momentum_lb": momentum_lb,
                            "ma_window": ma_window,
                            "vol_window": vol_window,
                            "top_n": top_n,
                            "min_score": 0.0,
                            "rebalance_days": 20,
                            "guard_enabled": True,
                            "min_rebalance_days": base_min_days,
                            "min_turnover": base_min_turn,
                            "force_rebalance_on_regime_change": base_force,
                        }
                    )
    return out


def stage2_candidates(top_core: list[dict]) -> list[dict]:
    out = []
    for core in top_core:
        for min_days in [10, 15, 20, 25, 30]:
            for min_turn in [0.01, 0.02, 0.03, 0.05, 0.08]:
                for force_rebalance in [True, False]:
                    c = dict(core["params"])
                    c.update(
                        {
                            "min_rebalance_days": min_days,
                            "min_turnover": min_turn,
                            "force_rebalance_on_regime_change": force_rebalance,
                        }
                    )
                    out.append(c)
    return out


def stage3_candidates(best_stage2: dict) -> list[dict]:
    p = best_stage2["params"]

    def uniq_int(vals: list[int], lower: int) -> list[int]:
        return sorted(set([v for v in vals if v >= lower]))

    momentum_grid = uniq_int([p["momentum_lb"] - 32, p["momentum_lb"], p["momentum_lb"] + 28], 60)
    ma_grid = uniq_int([p["ma_window"] - 20, p["ma_window"], p["ma_window"] + 20], 20)
    vol_grid = uniq_int([p["vol_window"] - 10, p["vol_window"], p["vol_window"] + 10], 5)
    top_n_grid = [int(p["top_n"])]
    min_days_grid = uniq_int([p["min_rebalance_days"] - 5, p["min_rebalance_days"], p["min_rebalance_days"] + 5], 1)
    min_turn_grid = sorted(
        set(
            round(v, 4)
            for v in [
                p["min_turnover"] - 0.01,
                p["min_turnover"],
                p["min_turnover"] + 0.01,
            ]
            if v > 0
        )
    )

    out = []
    for momentum_lb in momentum_grid:
        for ma_window in ma_grid:
            for vol_window in vol_grid:
                for top_n in top_n_grid:
                    for min_days in min_days_grid:
                        for min_turn in min_turn_grid:
                            out.append(
                                {
                                    "momentum_lb": momentum_lb,
                                    "ma_window": ma_window,
                                    "vol_window": vol_window,
                                    "top_n": top_n,
                                    "min_score": 0.0,
                                    "rebalance_days": int(p["rebalance_days"]),
                                    "guard_enabled": bool(p["guard_enabled"]),
                                    "min_rebalance_days": min_days,
                                    "min_turnover": float(min_turn),
                                    "force_rebalance_on_regime_change": bool(p["force_rebalance_on_regime_change"]),
                                }
                            )
    return out


def sort_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda x: (
            bool(x["stable_pass"]),
            x["train_score"],
            x["metrics"]["oos_2025"]["excess_annual_return_vs_alloc"],
            x["metrics"]["full"]["excess_annual_return_vs_alloc"],
        ),
        reverse=True,
    )


def run_stage(stage_name: str, runtime: dict, stock_cfg: dict, risk_cfg: dict, candidates: list[dict]) -> list[dict]:
    total = len(candidates)
    results = []
    for i, c in enumerate(candidates, 1):
        result = evaluate_candidate(runtime, stock_cfg, risk_cfg, c)
        result["stage"] = stage_name
        results.append(result)
        if i % 20 == 0 or i == total:
            print(f"[search:{stage_name}] {i}/{total}", flush=True)
    return sort_results(results)


def pick_recommended(results: list[dict]) -> dict:
    stable = [r for r in results if r["stable_pass"]]
    if stable:
        return stable[0]
    return results[0]


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
        risk_cfg = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[search] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_cfg.get("enabled", False):
        print("[stock] disabled")
        return EXIT_DISABLED
    if stock_cfg.get("mode") != "global_momentum":
        print("[stock] mode is not global_momentum, skip search")
        return EXIT_DISABLED

    try:
        out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(out_dir)

        baseline_params = {
            "momentum_lb": int(stock_cfg.get("global_model", {}).get("momentum_lb", 252)),
            "ma_window": int(stock_cfg.get("global_model", {}).get("ma_window", 200)),
            "vol_window": int(stock_cfg.get("global_model", {}).get("vol_window", 20)),
            "top_n": int(stock_cfg.get("global_model", {}).get("top_n", 1)),
            "min_score": float(stock_cfg.get("global_model", {}).get("min_score", 0.0)),
            "rebalance_days": int(stock_cfg.get("global_model", {}).get("rebalance_days", 20)),
            "guard_enabled": bool(stock_cfg.get("global_model", {}).get("execution_guard", {}).get("enabled", True)),
            "min_rebalance_days": int(
                stock_cfg.get("global_model", {}).get("execution_guard", {}).get("min_rebalance_days", 20)
            ),
            "min_turnover": float(stock_cfg.get("global_model", {}).get("execution_guard", {}).get("min_turnover", 0.05)),
            "force_rebalance_on_regime_change": bool(
                stock_cfg.get("global_model", {})
                .get("execution_guard", {})
                .get("force_rebalance_on_regime_change", True)
            ),
        }
        baseline = evaluate_candidate(runtime, stock_cfg, risk_cfg, baseline_params)
        baseline["stage"] = "baseline"

        # Stage 1: coarse core params (fixed guard as current config).
        s1_results = run_stage("stage1_core", runtime, stock_cfg, risk_cfg, stage1_candidates(stock_cfg))
        top_core = s1_results[:4]

        # Stage 2: guard refinement around the best stage-1 cores.
        s2_results = run_stage("stage2_guard", runtime, stock_cfg, risk_cfg, stage2_candidates(top_core))
        best_stage2 = pick_recommended(s2_results)

        # Stage 3: local fine search around stage-2 best.
        s3_results = run_stage("stage3_fine", runtime, stock_cfg, risk_cfg, stage3_candidates(best_stage2))
        best_stage3 = pick_recommended(s3_results)

        all_results = sort_results([baseline] + s1_results + s2_results + s3_results)
        recommended = pick_recommended(all_results)

        report = {
            "ts": datetime.now().isoformat(),
            "note": "production-aligned stock model parameter search (paper-forward engine)",
            "selection_policy": {
                "objective": "stable profitability + safety",
                "train_windows": ["full", "oos_2023", "oos_2024"],
                "holdout_window": "oos_2025",
                "stable_gate": {
                    "full_excess_gt_0": True,
                    "oos_2023_excess_gt_0": True,
                    "oos_2024_excess_gt_0": True,
                    "oos_2025_excess_gt_0": True,
                    "full_max_drawdown_gt_-0.12": True,
                },
            },
            "baseline": baseline,
            "stages": {
                "stage1_count": len(s1_results),
                "stage2_count": len(s2_results),
                "stage3_count": len(s3_results),
                "stage1_top5": s1_results[:5],
                "stage2_top5": s2_results[:5],
                "stage3_top10": s3_results[:10],
            },
            "recommended": recommended,
            "improvement_vs_baseline": {
                "full_excess_ann_delta": float(
                    recommended["metrics"]["full"]["excess_annual_return_vs_alloc"]
                    - baseline["metrics"]["full"]["excess_annual_return_vs_alloc"]
                ),
                "oos_2024_excess_ann_delta": float(
                    recommended["metrics"]["oos_2024"]["excess_annual_return_vs_alloc"]
                    - baseline["metrics"]["oos_2024"]["excess_annual_return_vs_alloc"]
                ),
                "oos_2025_excess_ann_delta": float(
                    recommended["metrics"]["oos_2025"]["excess_annual_return_vs_alloc"]
                    - baseline["metrics"]["oos_2025"]["excess_annual_return_vs_alloc"]
                ),
                "full_drawdown_delta": float(
                    recommended["metrics"]["full"]["strategy_max_drawdown"]
                    - baseline["metrics"]["full"]["strategy_max_drawdown"]
                ),
                "trade_count_delta": int(recommended["aggregate"]["trades"] - baseline["aggregate"]["trades"]),
            },
        }

        out_file = os.path.join(out_dir, "stock_production_model_search.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[search] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[search] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    print(f"[search] report -> {out_file}")
    print(
        "[search] recommended params:",
        json.dumps(report["recommended"]["params"], ensure_ascii=False),
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
