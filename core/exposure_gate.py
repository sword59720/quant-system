#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from itertools import combinations


def _annualized_return_from_ret(ret, periods_per_year=252):
    n = len(ret)
    if n < 2:
        return 0.0
    nav = 1.0
    for x in ret:
        nav *= 1.0 + float(x)
    years = max(float(n) / float(periods_per_year), 1e-9)
    return float(nav ** (1.0 / years) - 1.0)


def _build_groups(n_rows, n_groups):
    if n_groups < 2 or n_rows < n_groups:
        return []
    out = []
    for i in range(n_groups):
        s = int(i * n_rows / n_groups)
        e = int((i + 1) * n_rows / n_groups)
        out.append((s, e))
    return out


def _build_fold_indices(n_rows, groups, test_group_ids, embargo_days):
    idx = []
    for gid in test_group_ids:
        s, e = groups[gid]
        s2 = s + embargo_days
        e2 = e - embargo_days
        if e2 <= s2:
            continue
        idx.extend(range(s2, e2))
    idx = sorted(set(i for i in idx if 0 <= i < n_rows))
    return idx


def compute_cpcv_excess_worst(
    strategy_ret,
    benchmark_ret_alloc,
    n_groups=6,
    test_groups=2,
    embargo_days=5,
    min_fold_days=20,
):
    n_rows = min(len(strategy_ret), len(benchmark_ret_alloc))
    groups = _build_groups(n_rows, int(n_groups))
    if not groups:
        return None, {"error": "window too short for groups"}

    fold_defs = list(combinations(range(int(n_groups)), int(test_groups)))
    excess_list = []
    for test_ids in fold_defs:
        idx = _build_fold_indices(n_rows, groups, test_ids, int(embargo_days))
        if len(idx) < int(min_fold_days):
            continue
        s = [strategy_ret[i] for i in idx]
        b = [benchmark_ret_alloc[i] for i in idx]
        ex_ann = _annualized_return_from_ret(s, 252) - _annualized_return_from_ret(b, 252)
        if math.isnan(ex_ann):
            continue
        excess_list.append(float(ex_ann))

    if not excess_list:
        return None, {"error": "no valid folds"}

    return float(min(excess_list)), {
        "fold_candidates": int(len(fold_defs)),
        "fold_used": int(len(excess_list)),
        "excess_ann_mean": float(sum(excess_list) / len(excess_list)),
    }


def evaluate_exposure_gate(strategy_ret_hist, benchmark_ret_alloc_hist, base_alloc_pct, gate_cfg):
    enabled = bool(gate_cfg.get("enabled", False))
    meta = {
        "enabled": enabled,
        "base_alloc_pct": float(base_alloc_pct),
        "effective_alloc_pct": float(base_alloc_pct),
        "stage": "disabled",
        "reason": "gate_disabled",
        "lookback_eval": [],
        "agg_excess_ann_worst": None,
    }
    if not enabled:
        return float(base_alloc_pct), meta

    n = min(len(strategy_ret_hist), len(benchmark_ret_alloc_hist))
    if n < 20:
        meta.update(
            {
                "stage": "insufficient_data",
                "reason": f"history_too_short:{n}",
            }
        )
        return float(base_alloc_pct), meta

    lookbacks = [int(x) for x in gate_cfg.get("lookback_days", [63, 126]) if int(x) > 0]
    cpcv_cfg = gate_cfg.get("cpcv", {})
    n_groups = int(cpcv_cfg.get("n_groups", 6))
    test_groups = int(cpcv_cfg.get("test_groups", 2))
    embargo_days = int(cpcv_cfg.get("embargo_days", 5))
    min_fold_days = int(cpcv_cfg.get("min_fold_days", 20))

    per_win = []
    worsts = []
    for win in sorted(set(lookbacks)):
        if n < win:
            per_win.append({"window_days": win, "error": f"history_too_short:{n}"})
            continue
        s = strategy_ret_hist[-win:]
        b = benchmark_ret_alloc_hist[-win:]
        worst, detail = compute_cpcv_excess_worst(
            s,
            b,
            n_groups=n_groups,
            test_groups=test_groups,
            embargo_days=embargo_days,
            min_fold_days=min_fold_days,
        )
        row = {"window_days": int(win)}
        row.update(detail)
        if worst is None:
            row["error"] = row.get("error", "no_valid_folds")
        else:
            row["excess_ann_worst"] = float(worst)
            worsts.append(float(worst))
        per_win.append(row)

    if not worsts:
        meta.update(
            {
                "stage": "insufficient_data",
                "reason": "no_valid_window",
                "lookback_eval": per_win,
            }
        )
        return float(base_alloc_pct), meta

    agg_worst = float(min(worsts))
    thr = gate_cfg.get("thresholds", {})
    warn_worst_min = float(thr.get("warn_worst_min", -0.10))
    risk_worst_min = float(thr.get("risk_worst_min", -0.14))
    if risk_worst_min > warn_worst_min:
        risk_worst_min, warn_worst_min = warn_worst_min, risk_worst_min

    caps = gate_cfg.get("alloc_caps", {})
    cap_pass = float(caps.get("pass", base_alloc_pct))
    cap_warn = float(caps.get("warn", min(base_alloc_pct, cap_pass)))
    cap_risk = float(caps.get("risk", min(cap_warn, base_alloc_pct)))
    cap_pass = min(cap_pass, float(base_alloc_pct))
    cap_warn = min(cap_warn, cap_pass)
    cap_risk = min(cap_risk, cap_warn)

    stage = "pass"
    cap = cap_pass
    reason = "worst_pass"
    if agg_worst < risk_worst_min:
        stage = "risk"
        cap = cap_risk
        reason = "worst_below_risk_threshold"
    elif agg_worst < warn_worst_min:
        stage = "warn"
        cap = cap_warn
        reason = "worst_below_warn_threshold"

    eff = float(min(float(base_alloc_pct), float(cap)))
    meta.update(
        {
            "stage": stage,
            "reason": reason,
            "effective_alloc_pct": eff,
            "agg_excess_ann_worst": agg_worst,
            "lookback_eval": per_win,
            "thresholds": {
                "warn_worst_min": warn_worst_min,
                "risk_worst_min": risk_worst_min,
            },
            "alloc_caps": {
                "pass": cap_pass,
                "warn": cap_warn,
                "risk": cap_risk,
            },
        }
    )
    return eff, meta
