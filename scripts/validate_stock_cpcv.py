#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
import tempfile
from datetime import datetime
from itertools import combinations

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
    if sd == 0:
        return 0.0
    return float((ret.mean() / sd) * math.sqrt(periods_per_year))


def annualize_mean(ret: pd.Series, periods_per_year: int = 252):
    if len(ret) == 0:
        return None
    return float(ret.mean() * periods_per_year)


def build_groups(n_rows: int, n_groups: int) -> list[tuple[int, int]]:
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_rows < n_groups:
        raise ValueError("n_rows must be >= n_groups")
    groups = []
    for i in range(n_groups):
        start = int(i * n_rows / n_groups)
        end = int((i + 1) * n_rows / n_groups)
        groups.append((start, end))
    return groups


def build_fold_indices(
    n_rows: int,
    groups: list[tuple[int, int]],
    test_group_ids: tuple[int, ...],
    embargo_days: int,
) -> list[int]:
    idx = []
    for gid in test_group_ids:
        start, end = groups[gid]
        s = start + embargo_days
        e = end - embargo_days
        if e <= s:
            continue
        idx.extend(range(s, e))
    idx = sorted(set(i for i in idx if 0 <= i < n_rows))
    return idx


def fold_metrics(df: pd.DataFrame) -> dict:
    if len(df) < 2:
        return {
            "rows": int(len(df)),
            "strategy_ann": 0.0,
            "benchmark_ann": 0.0,
            "excess_ann": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
        }
    s_nav = pd.Series([1.0] + (1.0 + df["strategy_ret"]).cumprod().tolist())
    b_nav = pd.Series([1.0] + (1.0 + df["benchmark_ret_alloc"]).cumprod().tolist())
    s_ann = annualized_return(s_nav, 252)
    b_ann = annualized_return(b_nav, 252)
    excess_daily = df["strategy_ret"] - df["benchmark_ret_alloc"]
    up_mask = df["benchmark_ret_alloc"] >= 0.0
    down_mask = df["benchmark_ret_alloc"] < 0.0
    up_excess_ann = annualize_mean(excess_daily[up_mask], 252)
    down_excess_ann = annualize_mean(excess_daily[down_mask], 252)
    return {
        "rows": int(len(df)),
        "strategy_ann": float(s_ann),
        "benchmark_ann": float(b_ann),
        "excess_ann": float(s_ann - b_ann),
        "max_drawdown": max_drawdown(s_nav),
        "sharpe": sharpe_ratio(df["strategy_ret"], 252),
        "up_days": int(up_mask.sum()),
        "down_days": int(down_mask.sum()),
        "up_excess_ann": up_excess_ann,
        "down_excess_ann": down_excess_ann,
    }


def summarize_folds(folds: list[dict], thresholds: dict) -> dict:
    if not folds:
        return {"error": "no valid folds"}

    excess = pd.Series([f["metrics"]["excess_ann"] for f in folds], dtype=float)
    sharpe = pd.Series([f["metrics"]["sharpe"] for f in folds], dtype=float)
    max_dd = pd.Series([f["metrics"]["max_drawdown"] for f in folds], dtype=float)
    up_excess = pd.Series(
        [
            f["metrics"]["up_excess_ann"]
            for f in folds
            if f["metrics"].get("up_excess_ann") is not None
        ],
        dtype=float,
    )
    down_excess = pd.Series(
        [
            f["metrics"]["down_excess_ann"]
            for f in folds
            if f["metrics"].get("down_excess_ann") is not None
        ],
        dtype=float,
    )

    summary = {
        "folds_total": int(len(folds)),
        "excess_ann_mean": float(excess.mean()),
        "excess_ann_median": float(excess.median()),
        "excess_ann_std": float(excess.std(ddof=0)),
        "excess_ann_worst": float(excess.min()),
        "excess_ann_win_rate": float((excess > 0.0).mean()),
        "sharpe_median": float(sharpe.median()),
        "max_drawdown_worst": float(max_dd.min()),
        "up_excess_ann_mean": float(up_excess.mean()) if len(up_excess) > 0 else None,
        "down_excess_ann_mean": float(down_excess.mean()) if len(down_excess) > 0 else None,
        "down_excess_ann_worst": float(down_excess.min()) if len(down_excess) > 0 else None,
    }

    checks = {
        "excess_ann_mean_min": summary["excess_ann_mean"] >= float(thresholds["excess_ann_mean_min"]),
        "excess_ann_median_min": summary["excess_ann_median"] >= float(thresholds["excess_ann_median_min"]),
        "excess_ann_worst_min": summary["excess_ann_worst"] >= float(thresholds["excess_ann_worst_min"]),
        "excess_ann_win_rate_min": summary["excess_ann_win_rate"] >= float(thresholds["excess_ann_win_rate_min"]),
        "sharpe_median_min": summary["sharpe_median"] >= float(thresholds["sharpe_median_min"]),
        "max_drawdown_worst_min": summary["max_drawdown_worst"] >= float(thresholds["max_drawdown_worst_min"]),
    }

    if "up_excess_ann_mean_min" in thresholds:
        if summary["up_excess_ann_mean"] is None:
            checks["up_excess_ann_mean_min"] = False
        else:
            checks["up_excess_ann_mean_min"] = summary["up_excess_ann_mean"] >= float(thresholds["up_excess_ann_mean_min"])

    if "down_excess_ann_mean_min" in thresholds:
        if summary["down_excess_ann_mean"] is None:
            checks["down_excess_ann_mean_min"] = False
        else:
            checks["down_excess_ann_mean_min"] = summary["down_excess_ann_mean"] >= float(thresholds["down_excess_ann_mean_min"])

    if "down_excess_ann_worst_min" in thresholds:
        if summary["down_excess_ann_worst"] is None:
            checks["down_excess_ann_worst_min"] = False
        else:
            checks["down_excess_ann_worst_min"] = summary["down_excess_ann_worst"] >= float(thresholds["down_excess_ann_worst_min"])

    summary["threshold_checks"] = checks
    summary["gate_passed"] = bool(all(checks.values()))
    return summary


def evaluate_window(df: pd.DataFrame, cfg: dict, thresholds: dict) -> dict:
    rows = len(df)
    n_groups = int(cfg["n_groups"])
    k = int(cfg["test_groups"])
    embargo = int(cfg["embargo_days"])
    min_fold_days = int(cfg.get("min_fold_days", 60))

    groups = build_groups(rows, n_groups)
    fold_defs = list(combinations(range(n_groups), k))
    fold_results = []
    skipped = 0

    for test_ids in fold_defs:
        idx = build_fold_indices(rows, groups, test_ids, embargo)
        if len(idx) < min_fold_days:
            skipped += 1
            continue
        m = fold_metrics(df.iloc[idx].reset_index(drop=True))
        fold_results.append(
            {
                "test_groups": list(test_ids),
                "rows": int(len(idx)),
                "metrics": m,
            }
        )

    summary = summarize_folds(fold_results, thresholds)
    return {
        "rows": int(rows),
        "n_groups": n_groups,
        "test_groups": k,
        "embargo_days": embargo,
        "min_fold_days": min_fold_days,
        "fold_candidates": int(len(fold_defs)),
        "fold_used": int(len(fold_results)),
        "fold_skipped": int(skipped),
        "summary": summary,
        "folds": fold_results,
    }


def ranking_score(summary: dict, fold_used: int, fold_candidates: int) -> float:
    if "error" in summary:
        return float("-inf")
    gate_bonus = 1000.0 if summary.get("gate_passed", False) else 0.0
    coverage = 0.0 if fold_candidates == 0 else float(fold_used) / float(fold_candidates)
    score = (
        gate_bonus
        + 120.0 * float(summary.get("excess_ann_median", 0.0))
        + 90.0 * float(summary.get("excess_ann_mean", 0.0))
        + 80.0 * float(summary.get("excess_ann_worst", -1.0))
        + 40.0 * float(summary.get("up_excess_ann_mean") or 0.0)
        + 60.0 * float(summary.get("down_excess_ann_mean") or 0.0)
        + 50.0 * float(summary.get("down_excess_ann_worst") or -1.0)
        + 20.0 * float(summary.get("excess_ann_win_rate", 0.0))
        + 10.0 * float(summary.get("sharpe_median", 0.0))
        + 20.0 * float(summary.get("max_drawdown_worst", -1.0))
        + 10.0 * coverage
        - 20.0 * float(summary.get("excess_ann_std", 0.0))
    )
    return float(score)


def search_window_configs(df: pd.DataFrame, win_cfg: dict, search_cfg: dict, thresholds: dict) -> dict:
    n_candidates = [int(x) for x in search_cfg.get("n_groups_candidates", [8, 10, 12])]
    k_candidates = [int(x) for x in search_cfg.get("test_groups_candidates", [2, 3])]
    embargo_candidates = [int(x) for x in search_cfg.get("embargo_days_candidates", [10, 15, 20, 30])]
    min_fold_days = int(win_cfg.get("min_fold_days", 60))
    max_results = int(search_cfg.get("max_results", 20))

    results = []
    total_eval = 0
    for n in sorted(set(n_candidates)):
        for k in sorted(set(k_candidates)):
            if k >= n:
                continue
            for embargo in sorted(set(embargo_candidates)):
                trial_cfg = {
                    "n_groups": n,
                    "test_groups": k,
                    "embargo_days": embargo,
                    "min_fold_days": min_fold_days,
                }
                try:
                    out = evaluate_window(df, trial_cfg, thresholds)
                except Exception as e:
                    results.append({"params": trial_cfg, "error": str(e)})
                    continue
                total_eval += 1
                summary = out["summary"]
                score = ranking_score(summary, out["fold_used"], out["fold_candidates"])
                results.append(
                    {
                        "params": trial_cfg,
                        "score": score,
                        "fold_used": out["fold_used"],
                        "fold_candidates": out["fold_candidates"],
                        "summary": summary,
                    }
                )

    valid = [r for r in results if "summary" in r]
    valid = sorted(valid, key=lambda x: x["score"], reverse=True)
    passed = [r for r in valid if bool(r["summary"].get("gate_passed", False))]
    best = passed[0] if passed else (valid[0] if valid else None)
    return {
        "evaluated": int(total_eval),
        "valid_results": int(len(valid)),
        "gate_passed_results": int(len(passed)),
        "selection_policy": "best_gate_passed_first",
        "best": best,
        "top_results": valid[:max_results],
    }


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock = load_yaml("config/stock.yaml")
        risk = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[cpcv] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock.get("enabled", False):
        print("[stock] disabled")
        return EXIT_DISABLED
    if stock.get("mode") != "global_momentum":
        print("[stock] mode is not global_momentum, skip cpcv validation")
        return EXIT_DISABLED

    cpcv_cfg = stock.get("validation", {}).get("cpcv", {})
    if not bool(cpcv_cfg.get("enabled", False)):
        print("[cpcv] disabled by config/stock.yaml")
        return EXIT_DISABLED

    try:
        with tempfile.TemporaryDirectory() as td:
            rt = json.loads(json.dumps(runtime))
            rt["paths"]["output_dir"] = td
            _summary, history_file, _latest_file = run_paper_forward(rt, stock, risk)
            history = pd.read_csv(history_file)

        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date").reset_index(drop=True)
        thresholds = cpcv_cfg["thresholds"]

        windows = cpcv_cfg["windows"]
        search_cfg = cpcv_cfg.get("search", {})
        do_search = bool(search_cfg.get("enabled", False))
        out_windows = {}
        out_search = {}
        for win_name, win_cfg in windows.items():
            start_date = win_cfg.get("start_date")
            win_df = history if not start_date else history[history["date"] >= pd.Timestamp(start_date)]
            win_df = win_df.reset_index(drop=True)
            if len(win_df) < max(int(win_cfg["n_groups"]) * 10, 120):
                out_windows[win_name] = {"error": f"window too short: {len(win_df)} rows"}
                if do_search:
                    out_search[win_name] = {"error": f"window too short: {len(win_df)} rows"}
                continue
            out_windows[win_name] = evaluate_window(win_df, win_cfg, thresholds)
            if do_search:
                out_search[win_name] = search_window_configs(win_df, win_cfg, search_cfg, thresholds)
    except OSError as e:
        print(f"[cpcv] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[cpcv] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    report = {
        "ts": datetime.now().isoformat(),
        "note": "stock CPCV validation report (production-aligned daily return path)",
        "thresholds": thresholds,
        "windows": out_windows,
        "search": out_search if do_search else {},
    }

    try:
        out_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
        ensure_dir(out_dir)
        out_file = os.path.join(out_dir, "stock_cpcv_report.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[cpcv] output error: {e}")
        return EXIT_OUTPUT_ERROR

    compact = {
        "ts": report["ts"],
        "thresholds": report["thresholds"],
        "windows": {},
        "search": {},
    }
    for win_name, win_data in report["windows"].items():
        if "error" in win_data:
            compact["windows"][win_name] = {"error": win_data["error"]}
        else:
            compact["windows"][win_name] = {
                "rows": win_data["rows"],
                "fold_used": win_data["fold_used"],
                "summary": win_data["summary"],
            }

    for win_name, search_data in report["search"].items():
        if "error" in search_data:
            compact["search"][win_name] = {"error": search_data["error"]}
            continue
        best = search_data.get("best")
        compact["search"][win_name] = {
            "evaluated": search_data["evaluated"],
            "valid_results": search_data["valid_results"],
            "gate_passed_results": search_data["gate_passed_results"],
            "selection_policy": search_data["selection_policy"],
            "best_params": best["params"] if best else None,
            "best_gate_passed": bool(best["summary"]["gate_passed"]) if best else None,
            "best_score": float(best["score"]) if best else None,
        }

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"[cpcv] report -> {out_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
