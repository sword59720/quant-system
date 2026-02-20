#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import hashlib
import subprocess
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
from core.notify_wecom import send_wecom_message


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


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize_symbols(symbols):
    out = []
    seen = set()
    for s in symbols or []:
        x = str(s or "").strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return sorted(out)


def _build_universe_symbols(stock):
    universe = _normalize_symbols(stock.get("universe", []))
    benchmark = str(stock.get("benchmark_symbol", "510300")).strip()
    defensive = str(stock.get("defensive_symbol", "511010")).strip()
    return _normalize_symbols(universe + [benchmark, defensive])


def _hash_symbols(symbols):
    payload = ",".join(_normalize_symbols(symbols))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def classify_universe_change_level(has_previous, change_count, change_ratio, small_thr, medium_thr):
    if not has_previous:
        return "init"
    if int(change_count) <= 0:
        return "none"
    if float(change_ratio) <= float(small_thr):
        return "small"
    if float(change_ratio) <= float(medium_thr):
        return "medium"
    return "large"


def run_universe_quick_check(runtime, symbols, min_history_rows):
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    checks = []
    missing_symbols = []
    invalid_symbols = []
    insufficient_symbols = []
    ready_symbols = []
    last_dates = []

    for symbol in _normalize_symbols(symbols):
        fp = os.path.join(data_dir, f"{symbol}.csv")
        item = {
            "symbol": symbol,
            "file": fp,
            "exists": os.path.exists(fp),
            "has_columns": False,
            "rows": 0,
            "last_date": None,
            "ready": False,
        }
        if not item["exists"]:
            missing_symbols.append(symbol)
            checks.append(item)
            continue

        try:
            df = pd.read_csv(fp)
        except Exception:
            invalid_symbols.append(symbol)
            checks.append(item)
            continue

        if ("date" not in df.columns) or ("close" not in df.columns):
            invalid_symbols.append(symbol)
            checks.append(item)
            continue

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        item["has_columns"] = True
        item["rows"] = int(len(close))
        if not dates.empty:
            item["last_date"] = dates.iloc[-1].date().isoformat()
            last_dates.append(item["last_date"])
        item["ready"] = bool(len(close) >= int(min_history_rows))

        if item["ready"]:
            ready_symbols.append(symbol)
        else:
            insufficient_symbols.append(symbol)
        checks.append(item)

    total = int(len(checks))
    ready_count = int(len(ready_symbols))
    ready_ratio = float(ready_count / total) if total else 0.0

    common_last_date = None
    if checks:
        valid_dates = [x["last_date"] for x in checks if x.get("last_date")]
        if valid_dates:
            common_last_date = min(valid_dates)

    return {
        "total_symbols": total,
        "ready_symbols": ready_count,
        "ready_ratio": ready_ratio,
        "min_history_rows": int(min_history_rows),
        "missing_symbols": sorted(missing_symbols),
        "invalid_symbols": sorted(invalid_symbols),
        "insufficient_symbols": sorted(insufficient_symbols),
        "common_last_date": common_last_date,
        "checks": checks,
    }


def _load_backtest_baseline(path):
    data = load_json(path, default={})
    if not isinstance(data, dict):
        return None
    stock_block = data.get("stock", {})
    if not isinstance(stock_block, dict):
        return None
    keys = ["annual_return", "max_drawdown", "sharpe", "final_nav", "periods"]
    if any(k not in stock_block for k in keys):
        return None
    return {k: stock_block.get(k) for k in keys}


def evaluate_backtest_gate(current, baseline, gate_cfg):
    out = {
        "required": True,
        "available": False,
        "passed": True,
        "reason": "ok",
        "baseline": baseline,
        "current": current,
        "thresholds": {
            "annual_return_drop_max": float(gate_cfg.get("annual_return_drop_max", 0.02)),
            "sharpe_drop_max": float(gate_cfg.get("sharpe_drop_max", 0.20)),
            "max_drawdown_widen_max": float(gate_cfg.get("max_drawdown_widen_max", 0.03)),
        },
        "checks": {},
        "deltas": {},
    }

    if not isinstance(current, dict) or ("error" in current):
        out["passed"] = False
        out["reason"] = f"backtest_unavailable: {current.get('error') if isinstance(current, dict) else 'invalid'}"
        return out

    if baseline is None:
        out["available"] = True
        out["passed"] = True
        out["reason"] = "baseline_missing"
        return out

    cur_ann = _safe_float(current.get("annual_return"))
    cur_sharpe = _safe_float(current.get("sharpe"))
    cur_mdd = _safe_float(current.get("max_drawdown"))
    base_ann = _safe_float(baseline.get("annual_return"))
    base_sharpe = _safe_float(baseline.get("sharpe"))
    base_mdd = _safe_float(baseline.get("max_drawdown"))
    if None in [cur_ann, cur_sharpe, cur_mdd, base_ann, base_sharpe, base_mdd]:
        out["passed"] = False
        out["reason"] = "metric_missing"
        return out

    ann_drop = float(base_ann - cur_ann)
    sharpe_drop = float(base_sharpe - cur_sharpe)
    mdd_widen = float(abs(cur_mdd) - abs(base_mdd))
    thr = out["thresholds"]
    checks = {
        "annual_return_drop_ok": ann_drop <= float(thr["annual_return_drop_max"]),
        "sharpe_drop_ok": sharpe_drop <= float(thr["sharpe_drop_max"]),
        "max_drawdown_widen_ok": mdd_widen <= float(thr["max_drawdown_widen_max"]),
    }
    out["checks"] = checks
    out["deltas"] = {
        "annual_return": float(cur_ann - base_ann),
        "sharpe": float(cur_sharpe - base_sharpe),
        "max_drawdown": float(cur_mdd - base_mdd),
        "max_drawdown_widen": mdd_widen,
    }
    out["available"] = True
    out["passed"] = bool(all(checks.values()))
    if not out["passed"]:
        out["reason"] = "gate_failed"
    return out


def run_cpcv_gate(runtime):
    report_file = os.path.join(runtime["paths"]["output_dir"], "reports", "stock_cpcv_report.json")
    cmd = [sys.executable, "scripts/validate_stock_cpcv.py"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return {
            "required": True,
            "available": False,
            "passed": False,
            "reason": f"cpcv_run_failed: {e}",
            "returncode": None,
            "windows": {},
        }

    if proc.returncode == EXIT_DISABLED:
        return {
            "required": True,
            "available": False,
            "passed": False,
            "reason": "cpcv_disabled",
            "returncode": int(proc.returncode),
            "windows": {},
        }

    report = load_json(report_file, default={})
    windows = {}
    all_pass = True
    if isinstance(report, dict):
        for name, data in (report.get("windows", {}) or {}).items():
            if (not isinstance(data, dict)) or ("summary" not in data):
                windows[name] = False
                all_pass = False
                continue
            passed = bool((data.get("summary") or {}).get("gate_passed", False))
            windows[name] = passed
            all_pass = all_pass and passed
    else:
        all_pass = False

    if not windows:
        all_pass = False

    reason = "ok" if all_pass else "gate_failed"
    if proc.returncode != EXIT_OK:
        reason = f"cpcv_script_exit_{proc.returncode}"
        all_pass = False

    return {
        "required": True,
        "available": bool(windows),
        "passed": bool(all_pass),
        "reason": reason,
        "returncode": int(proc.returncode),
        "windows": windows,
    }


def apply_universe_change_guard(runtime, stock, risk):
    val_cfg = stock.get("validation", {}) or {}
    guard_cfg = val_cfg.get("universe_change_guard", {}) or {}
    enabled = bool(guard_cfg.get("enabled", False))
    output_dir = runtime["paths"]["output_dir"]

    state_file = guard_cfg.get(
        "state_file",
        os.path.join(output_dir, "state", "stock_universe_state.json"),
    )
    report_file = guard_cfg.get(
        "report_file",
        os.path.join(output_dir, "reports", "stock_universe_change_latest.json"),
    )
    ensure_dir(os.path.dirname(state_file))
    ensure_dir(os.path.dirname(report_file))

    symbols = _build_universe_symbols(stock)
    current_hash = _hash_symbols(symbols)
    now_ts = datetime.now().isoformat()

    if not enabled:
        out = {
            "enabled": False,
            "level": "disabled",
            "alert_required": False,
            "report_file": report_file,
            "state_file": state_file,
        }
        save_json(report_file, {"ts": now_ts, **out})
        save_json(
            state_file,
            {
                "ts": now_ts,
                "enabled": False,
                "symbols": symbols,
                "symbols_hash": current_hash,
                "last_level": "disabled",
            },
        )
        return out

    state = load_json(state_file, default={})
    prev_symbols = state.get("symbols")
    if not isinstance(prev_symbols, list):
        prev_symbols = None
    else:
        prev_symbols = _normalize_symbols(prev_symbols)

    prev_set = set(prev_symbols or [])
    cur_set = set(symbols)
    added = sorted(cur_set - prev_set)
    removed = sorted(prev_set - cur_set)
    change_count = int(len(added) + len(removed))
    base_count = int(len(prev_symbols) if prev_symbols is not None else len(symbols))
    change_ratio = float(change_count / max(1, base_count))

    small_thr = float(guard_cfg.get("small_change_ratio", 0.10))
    medium_thr = float(guard_cfg.get("medium_change_ratio", 0.30))
    if medium_thr < small_thr:
        medium_thr = small_thr
    level = classify_universe_change_level(
        has_previous=(prev_symbols is not None),
        change_count=change_count,
        change_ratio=change_ratio,
        small_thr=small_thr,
        medium_thr=medium_thr,
    )

    model_cfg = stock.get("global_model", {}) or {}
    min_rows_default = max(
        int(model_cfg.get("momentum_lb", 252)),
        int(model_cfg.get("ma_window", 200)),
        int(model_cfg.get("vol_window", 30)),
        int(model_cfg.get("warmup_min_days", 126)),
    ) + 2
    min_rows = int(guard_cfg.get("min_history_rows", min_rows_default))
    min_ready_ratio = float(guard_cfg.get("min_ready_ratio", 0.90))

    quick = run_universe_quick_check(runtime, symbols, min_rows)
    checks = {"quick_check": quick}
    issues = []
    if quick.get("missing_symbols"):
        issues.append("missing_data_files")
    if quick.get("invalid_symbols"):
        issues.append("invalid_data_format")
    if float(quick.get("ready_ratio", 0.0)) < min_ready_ratio:
        issues.append("insufficient_history_coverage")

    backtest_gate_cfg = guard_cfg.get("backtest_gate", {}) or {}
    run_backtest = bool(guard_cfg.get("run_backtest_on_medium_large", True))
    if level in {"medium", "large"} and run_backtest:
        baseline_file = backtest_gate_cfg.get(
            "baseline_file",
            os.path.join(output_dir, "reports", "backtest_report.json"),
        )
        baseline = _load_backtest_baseline(baseline_file)
        bt_current = {"error": "not_run"}
        try:
            from scripts.backtest_v3 import backtest_stock

            bt_current = backtest_stock(runtime, stock, risk, backtest_mode_override="production")
        except Exception as e:
            bt_current = {"error": str(e)}

        bt_gate = evaluate_backtest_gate(bt_current, baseline, backtest_gate_cfg)
        bt_gate["baseline_file"] = baseline_file
        checks["backtest_gate"] = bt_gate
        if not bool(bt_gate.get("passed", False)):
            issues.append("backtest_gate_failed")

    run_cpcv = bool(guard_cfg.get("run_cpcv_on_large", True))
    if level == "large" and run_cpcv:
        cpcv = run_cpcv_gate(runtime)
        checks["cpcv_gate"] = cpcv
        if not bool(cpcv.get("passed", False)):
            issues.append("cpcv_gate_failed")

    status = "pass"
    if issues:
        status = "warn"
    if ("backtest_gate_failed" in issues) or ("cpcv_gate_failed" in issues):
        status = "risk"

    report = {
        "ts": now_ts,
        "enabled": True,
        "level": level,
        "status": status,
        "universe_size": int(len(symbols)),
        "change_count": change_count,
        "change_ratio": round(change_ratio, 6),
        "thresholds": {
            "small_change_ratio": small_thr,
            "medium_change_ratio": medium_thr,
            "min_history_rows": min_rows,
            "min_ready_ratio": min_ready_ratio,
        },
        "changes": {
            "added": added,
            "removed": removed,
        },
        "issues": sorted(set(issues)),
        "checks": checks,
        "alert_required": bool(issues and level not in {"none", "init"}),
        "symbols_hash": current_hash,
        "state_file": state_file,
        "report_file": report_file,
    }
    save_json(report_file, report)
    save_json(
        state_file,
        {
            "ts": now_ts,
            "enabled": True,
            "symbols": symbols,
            "symbols_hash": current_hash,
            "last_level": level,
            "last_status": status,
            "last_change_count": change_count,
            "last_change_ratio": round(change_ratio, 6),
            "last_issues": sorted(set(issues)),
            "last_report_file": report_file,
        },
    )
    return report


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


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def compute_risk_alloc_multiplier(benchmark_close, model_cfg):
    cfg = model_cfg.get("risk_governor", {}) if isinstance(model_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    meta = {
        "enabled": enabled,
        "alloc_mult": 1.0,
        "vol_mult": 1.0,
        "dd_cap_applied": False,
        "momentum_cap_applied": False,
        "benchmark_vol": None,
        "benchmark_dd": None,
        "benchmark_momentum": None,
    }
    if (not enabled) or (benchmark_close is None) or len(benchmark_close) < 3:
        return 1.0, meta

    vol_window = int(cfg.get("vol_window", 20))
    target_daily_vol = float(cfg.get("target_daily_vol", 0.012))
    min_alloc_mult = float(cfg.get("min_alloc_mult", 0.30))
    max_alloc_mult = float(cfg.get("max_alloc_mult", 1.00))
    dd_window = int(cfg.get("dd_window", 120))
    dd_trigger = float(cfg.get("dd_trigger", -0.08))
    dd_alloc_mult = float(cfg.get("dd_alloc_mult", 0.45))
    momentum_window = int(cfg.get("momentum_window", 20))
    momentum_trigger = float(cfg.get("momentum_trigger", -0.03))
    momentum_alloc_mult = float(cfg.get("momentum_alloc_mult", 0.60))

    ret = benchmark_close.pct_change().dropna()
    vol_mult = 1.0
    if len(ret) >= 2:
        bench_vol = float(ret.tail(vol_window).std())
        meta["benchmark_vol"] = bench_vol
        if bench_vol > 1e-8:
            vol_mult = target_daily_vol / bench_vol
    vol_mult = _clamp(float(vol_mult), min_alloc_mult, max_alloc_mult)

    dd_slice = benchmark_close.tail(dd_window)
    peak = float(dd_slice.max()) if len(dd_slice) > 0 else float(benchmark_close.iloc[-1])
    dd_now = float(benchmark_close.iloc[-1] / max(peak, 1e-8) - 1.0)
    meta["benchmark_dd"] = dd_now

    mom_now = None
    if len(benchmark_close) > momentum_window:
        mom_now = float(benchmark_close.iloc[-1] / benchmark_close.iloc[-1 - momentum_window] - 1.0)
        meta["benchmark_momentum"] = mom_now

    alloc_mult = vol_mult
    if dd_now <= dd_trigger:
        alloc_mult = min(alloc_mult, dd_alloc_mult)
        meta["dd_cap_applied"] = True
    if (mom_now is not None) and (mom_now <= momentum_trigger):
        alloc_mult = min(alloc_mult, momentum_alloc_mult)
        meta["momentum_cap_applied"] = True

    alloc_mult = _clamp(float(alloc_mult), min_alloc_mult, max_alloc_mult)
    meta["alloc_mult"] = alloc_mult
    meta["vol_mult"] = vol_mult
    return alloc_mult, meta


def build_risk_on_weights(score_map, vol_map, picks, score_mix, score_floor, score_power):
    if not picks:
        return {}
    mix = max(0.0, min(1.0, float(score_mix)))
    power = max(0.1, float(score_power))

    inv_vol = {s: 1.0 / max(float(vol_map[s]), 1e-6) for s in picks}
    inv_total = sum(inv_vol.values())
    ivol_weights = {s: (inv_vol[s] / inv_total if inv_total > 0 else 1.0 / len(picks)) for s in picks}
    if mix <= 1e-12:
        return ivol_weights

    score_raw = {}
    for s in picks:
        val = max(float(score_map[s]) - float(score_floor), 0.0)
        score_raw[s] = val**power
    score_total = sum(score_raw.values())
    if score_total <= 1e-12:
        score_weights = {s: 1.0 / len(picks) for s in picks}
    else:
        score_weights = {s: score_raw[s] / score_total for s in picks}

    mixed = {s: (1.0 - mix) * ivol_weights[s] + mix * score_weights[s] for s in picks}
    mixed_total = sum(mixed.values())
    if mixed_total <= 1e-12:
        return ivol_weights
    return {s: mixed[s] / mixed_total for s in picks}


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
    risk_on_score_mix = float(model_cfg.get("risk_on_score_mix", 0.0))
    risk_on_score_floor = float(model_cfg.get("risk_on_score_floor", min_score))
    risk_on_score_power = float(model_cfg.get("risk_on_score_power", 1.0))
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

    gate_cfg = model_cfg.get("exposure_gate", {})
    history_file = gate_cfg.get(
        "history_file",
        os.path.join(runtime["paths"]["output_dir"], "reports", "stock_paper_forward_history.csv"),
    )
    hist_strategy_ret, hist_bench_alloc = load_stock_return_history(history_file)
    effective_alloc, exposure_meta = evaluate_exposure_gate(hist_strategy_ret, hist_bench_alloc, alloc_pct, gate_cfg)
    stock_capital = total_capital * effective_alloc
    single_max = float(risk["position_limits"]["stock_single_max_pct"])

    eligible = [s for s in score.keys() if score[s] > min_score]
    sleeve_weights = {}
    signal_reason = "risk_off"
    risk_weights = {}
    risk_governor_meta = {"enabled": bool(model_cfg.get("risk_governor", {}).get("enabled", False)), "alloc_mult": 1.0}
    risk_alloc = effective_alloc

    if regime_on and eligible:
        alloc_mult, risk_governor_meta = compute_risk_alloc_multiplier(benchmark_close, model_cfg)
        risk_alloc = effective_alloc * alloc_mult
        picks = sorted(eligible, key=lambda x: score[x], reverse=True)[:top_n]
        risk_weights = build_risk_on_weights(
            score_map=score,
            vol_map=vol,
            picks=picks,
            score_mix=risk_on_score_mix,
            score_floor=risk_on_score_floor,
            score_power=risk_on_score_power,
        )
        sleeve_weights = risk_weights
        signal_reason = "risk_on"
    else:
        sleeve_weights = {defensive_symbol: 1.0}
        if not regime_on:
            signal_reason = "benchmark_below_ma"
        elif not eligible:
            signal_reason = "no_positive_momentum"

    raw_targets = {s: risk_alloc * sleeve_w for s, sleeve_w in sleeve_weights.items()}
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
                "risk_weight": round(float(risk_weights.get(s, 0.0)), 6),
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
            "risk_on_score_mix": risk_on_score_mix,
            "risk_on_score_floor": risk_on_score_floor,
            "risk_on_score_power": risk_on_score_power,
            "defensive_bypass_single_max": defensive_bypass_single_max,
        },
        "scores": scores,
        "targets": targets,
        "risk_governor": risk_governor_meta,
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
        universe_guard = apply_universe_change_guard(runtime, stock, risk)
        out["universe_change_guard"] = universe_guard
    except Exception as e:
        out["universe_change_guard"] = {
            "enabled": False,
            "level": "error",
            "status": "warn",
            "issues": [f"guard_failed:{e}"],
            "alert_required": False,
        }
        print(f"[stock] universe guard warning: {e}")

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

    # 企业微信通知：目标仓位 + 风控处罚/执行拦截
    try:
        tgt = ", ".join([f"{x['symbol']}:{x['target_weight']:.2f}" for x in out.get("targets", [])]) or "无"
        lines = [
            f"时间: {out.get('ts','')}",
            f"市场: stock",
            f"模式: {out.get('mode','')}",
            f"目标仓位: {tgt}",
        ]
        ok, msg = send_wecom_message("\n".join(lines), title="目标仓位更新")
        print(f"[notify] wecom {'ok' if ok else 'fail'}: {msg}")

        # 风控异常处罚通知
        overlay = out.get("risk_overlay", {})
        guard = out.get("execution_guard", {})
        punish = False
        reason = []
        if overlay.get("state_active", False) or overlay.get("triggered", False):
            punish = True
            reason.append(f"risk_overlay={overlay.get('reason', 'active')}")
        if guard.get("action") == "hold":
            punish = True
            reason.append(f"execution_guard={guard.get('reason','hold')}")
        if punish:
            send_wecom_message(
                "风控处罚触发：" + ", ".join(reason) + f"\n当前目标仓位: {tgt}",
                title="风控状态异常触发",
                dedup_key="risk_stock_trigger",
                dedup_hours=24,
            )

        # 股票池变化分级告警
        u_guard = out.get("universe_change_guard", {}) or {}
        if bool(u_guard.get("alert_required", False)):
            level = str(u_guard.get("level", "unknown"))
            status = str(u_guard.get("status", "warn"))
            issues = ", ".join(u_guard.get("issues", [])) or "none"
            changes = u_guard.get("changes", {}) or {}
            added = ",".join(changes.get("added", [])) or "无"
            removed = ",".join(changes.get("removed", [])) or "无"
            lines = [
                f"时间: {out.get('ts','')}",
                f"等级: {level}",
                f"状态: {status}",
                f"变更: +[{added}] -[{removed}]",
                f"问题: {issues}",
            ]
            symbols_hash = str(u_guard.get("symbols_hash", ""))[:16]
            dedup_key = f"stock_universe_guard_{symbols_hash}" if symbols_hash else "stock_universe_guard"
            send_wecom_message(
                "\n".join(lines),
                title="股票池变更告警",
                dedup_key=dedup_key,
                dedup_hours=24,
            )
    except Exception as e:
        print(f"[notify] wecom error: {e}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
