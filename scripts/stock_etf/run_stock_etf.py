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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
    cmd = [sys.executable, "scripts/stock_etf/backtest_stock_etf_cpcv.py"]
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
            from scripts.stock_etf.backtest_stock_etf import backtest_stock

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


def _normalize_float_weights(weights, n):
    if n <= 0:
        return []
    out = []
    for w in (weights or []):
        try:
            out.append(float(w))
        except (TypeError, ValueError):
            continue
    if len(out) != n:
        return [1.0 / n] * n
    total = sum(max(x, 0.0) for x in out)
    if total <= 1e-12:
        return [1.0 / n] * n
    return [max(x, 0.0) / total for x in out]


def _resolve_structural_cfg(model_cfg):
    cfg = model_cfg.get("structural_upgrade", {}) if isinstance(model_cfg, dict) else {}
    enabled = bool(cfg.get("enabled", False))
    windows = []
    for x in cfg.get("momentum_windows", [20, 60, 180]):
        try:
            w = int(x)
        except (TypeError, ValueError):
            continue
        if w > 1:
            windows.append(w)
    if not windows:
        windows = [max(2, int(model_cfg.get("momentum_lb", 252)))]
    weights = _normalize_float_weights(cfg.get("momentum_weights", [0.5, 0.3, 0.2]), len(windows))

    breadth_on_threshold = float(cfg.get("breadth_on_threshold", 0.60))
    breadth_off_threshold = float(cfg.get("breadth_off_threshold", 0.35))
    if breadth_off_threshold > breadth_on_threshold:
        breadth_off_threshold = breadth_on_threshold

    trend_mult_min = float(cfg.get("trend_alloc_mult_min", 0.75))
    trend_mult_max = float(cfg.get("trend_alloc_mult_max", 1.35))
    if trend_mult_max < trend_mult_min:
        trend_mult_max = trend_mult_min

    breadth_mult_min = float(cfg.get("breadth_alloc_mult_min", 0.70))
    breadth_mult_max = float(cfg.get("breadth_alloc_mult_max", 1.25))
    if breadth_mult_max < breadth_mult_min:
        breadth_mult_max = breadth_mult_min

    total_mult_min = float(cfg.get("total_alloc_mult_min", 0.70))
    total_mult_max = float(cfg.get("total_alloc_mult_max", 1.60))
    if total_mult_max < total_mult_min:
        total_mult_max = total_mult_min

    phase2_total_mult_min = float(cfg.get("phase2_total_mult_min", 0.80))
    phase2_total_mult_max = float(cfg.get("phase2_total_mult_max", 1.20))
    if phase2_total_mult_max < phase2_total_mult_min:
        phase2_total_mult_max = phase2_total_mult_min

    signal_strength_min_alloc_mult = float(cfg.get("signal_strength_min_alloc_mult", 0.70))
    signal_strength_max_alloc_mult = float(cfg.get("signal_strength_max_alloc_mult", 1.10))
    if signal_strength_max_alloc_mult < signal_strength_min_alloc_mult:
        signal_strength_max_alloc_mult = signal_strength_min_alloc_mult

    top_n = max(1, int(model_cfg.get("top_n", 1)))
    adaptive_min_top_n = max(1, int(cfg.get("phase2_adaptive_min_top_n", 1)))
    adaptive_max_top_n = max(adaptive_min_top_n, int(cfg.get("phase2_adaptive_max_top_n", top_n)))
    phase2_stress_regime_stepdown = int(cfg.get("phase2_stress_regime_stepdown", 0))
    phase2_stress_regime_stepdown = max(0, min(2, phase2_stress_regime_stepdown))

    return {
        "enabled": enabled,
        "momentum_windows": windows,
        "momentum_weights": weights,
        "eligibility_require_trend": bool(cfg.get("eligibility_require_trend", False)),
        "score_blend": _clamp(float(cfg.get("score_blend", 0.35)), 0.0, 1.0),
        "relative_momentum_window": max(2, int(cfg.get("relative_momentum_window", 60))),
        "relative_momentum_weight": float(cfg.get("relative_momentum_weight", 0.20)),
        "asset_trend_ma_window": max(2, int(cfg.get("asset_trend_ma_window", 80))),
        "breadth_on_threshold": breadth_on_threshold,
        "breadth_off_threshold": breadth_off_threshold,
        "neutral_alloc_mult": max(0.0, float(cfg.get("neutral_alloc_mult", 0.65))),
        "neutral_top_n": max(1, int(cfg.get("neutral_top_n", max(1, top_n - 1)))),
        "trend_strength_k": float(cfg.get("trend_strength_k", 8.0)),
        "trend_alloc_mult_min": trend_mult_min,
        "trend_alloc_mult_max": trend_mult_max,
        "breadth_alloc_mult_min": breadth_mult_min,
        "breadth_alloc_mult_max": breadth_mult_max,
        "total_alloc_mult_min": total_mult_min,
        "total_alloc_mult_max": total_mult_max,
        "phase2_enabled": bool(cfg.get("phase2_enabled", False)),
        "phase2_vol_short_window": max(2, int(cfg.get("phase2_vol_short_window", 15))),
        "phase2_vol_long_window": max(5, int(cfg.get("phase2_vol_long_window", 80))),
        "phase2_vol_calm_ratio": float(cfg.get("phase2_vol_calm_ratio", 0.85)),
        "phase2_vol_stress_ratio": float(cfg.get("phase2_vol_stress_ratio", 1.20)),
        "phase2_vol_mult_calm": float(cfg.get("phase2_vol_mult_calm", 1.08)),
        "phase2_vol_mult_normal": float(cfg.get("phase2_vol_mult_normal", 1.00)),
        "phase2_vol_mult_stress": float(cfg.get("phase2_vol_mult_stress", 0.78)),
        "phase2_stress_regime_stepdown": phase2_stress_regime_stepdown,
        "phase2_recovery_enabled": bool(cfg.get("phase2_recovery_enabled", True)),
        "phase2_recovery_window": max(20, int(cfg.get("phase2_recovery_window", 180))),
        "phase2_recovery_trigger_drawdown": float(cfg.get("phase2_recovery_trigger_drawdown", -0.08)),
        "phase2_recovery_momentum_window": max(2, int(cfg.get("phase2_recovery_momentum_window", 20))),
        "phase2_recovery_max_mult": float(cfg.get("phase2_recovery_max_mult", 1.20)),
        "phase2_total_mult_min": phase2_total_mult_min,
        "phase2_total_mult_max": phase2_total_mult_max,
        "phase2_adaptive_enabled": bool(cfg.get("phase2_adaptive_enabled", True)),
        "phase2_adaptive_min_top_n": adaptive_min_top_n,
        "phase2_adaptive_max_top_n": adaptive_max_top_n,
        "phase2_adaptive_calm_top_n_delta": int(cfg.get("phase2_adaptive_calm_top_n_delta", -1)),
        "phase2_adaptive_stress_top_n_delta": int(cfg.get("phase2_adaptive_stress_top_n_delta", 1)),
        "phase2_adaptive_high_dispersion_threshold": float(
            cfg.get("phase2_adaptive_high_dispersion_threshold", 0.22)
        ),
        "phase2_adaptive_high_dispersion_top_n_delta": int(cfg.get("phase2_adaptive_high_dispersion_top_n_delta", -1)),
        "phase2_adaptive_score_power_calm_mult": float(cfg.get("phase2_adaptive_score_power_calm_mult", 1.20)),
        "phase2_adaptive_score_power_stress_mult": float(cfg.get("phase2_adaptive_score_power_stress_mult", 0.85)),
        "signal_strength_enabled": bool(cfg.get("signal_strength_enabled", False)),
        "signal_strength_floor": float(cfg.get("signal_strength_floor", 0.0)),
        "signal_strength_ref": max(1e-6, float(cfg.get("signal_strength_ref", 0.06))),
        "signal_strength_curve": max(0.25, float(cfg.get("signal_strength_curve", 1.0))),
        "signal_strength_min_alloc_mult": signal_strength_min_alloc_mult,
        "signal_strength_max_alloc_mult": signal_strength_max_alloc_mult,
    }


def _weighted_momentum(close, windows, weights):
    if len(windows) != len(weights) or not windows:
        return None
    vals = []
    for w in windows:
        if len(close) <= w:
            return None
        prev = float(close.iloc[-1 - w])
        if abs(prev) <= 1e-12:
            return None
        vals.append(float(close.iloc[-1] / prev - 1.0))
    return float(sum(weights[k] * vals[k] for k in range(len(vals))))


def _compute_vol_state_multiplier(benchmark_ret, cfg):
    sw = int(cfg.get("phase2_vol_short_window", 15))
    lw = max(sw + 1, int(cfg.get("phase2_vol_long_window", 80)))
    if len(benchmark_ret) < lw:
        return "normal", float(cfg.get("phase2_vol_mult_normal", 1.0)), None
    short_vol = float(benchmark_ret.tail(sw).std())
    long_vol = float(benchmark_ret.tail(lw).std())
    if long_vol <= 1e-8:
        return "normal", float(cfg.get("phase2_vol_mult_normal", 1.0)), None

    ratio = float(short_vol / long_vol)
    calm_r = float(cfg.get("phase2_vol_calm_ratio", 0.85))
    stress_r = float(cfg.get("phase2_vol_stress_ratio", 1.20))
    if ratio <= calm_r:
        return "calm", float(cfg.get("phase2_vol_mult_calm", 1.08)), ratio
    if ratio >= stress_r:
        return "stress", float(cfg.get("phase2_vol_mult_stress", 0.78)), ratio
    return "normal", float(cfg.get("phase2_vol_mult_normal", 1.0)), ratio


def _compute_drawdown_recovery_multiplier(benchmark_close, cfg):
    meta = {
        "enabled": bool(cfg.get("phase2_recovery_enabled", True)),
        "drawdown": None,
        "drawdown_depth": None,
        "recovery_progress": None,
        "momentum": None,
    }
    if not meta["enabled"]:
        return 1.0, meta

    window = int(cfg.get("phase2_recovery_window", 180))
    trigger_dd = float(cfg.get("phase2_recovery_trigger_drawdown", -0.08))
    mom_w = int(cfg.get("phase2_recovery_momentum_window", 20))
    max_mult = max(1.0, float(cfg.get("phase2_recovery_max_mult", 1.20)))
    if len(benchmark_close) <= max(window, mom_w):
        return 1.0, meta

    sl = benchmark_close.tail(window)
    peak = float(sl.max())
    trough = float(sl.min())
    px = float(benchmark_close.iloc[-1])
    if peak <= 1e-12:
        return 1.0, meta

    dd_now = float(px / peak - 1.0)
    dd_depth = float(trough / peak - 1.0)
    meta["drawdown"] = dd_now
    meta["drawdown_depth"] = dd_depth
    if dd_depth > trigger_dd:
        return 1.0, meta

    if (peak - trough) <= 1e-12:
        return 1.0, meta
    recovery_progress = float((px - trough) / (peak - trough))
    recovery_progress = _clamp(recovery_progress, 0.0, 1.0)
    meta["recovery_progress"] = recovery_progress

    prev = float(benchmark_close.iloc[-1 - mom_w])
    if abs(prev) <= 1e-12:
        return 1.0, meta
    mom = float(px / prev - 1.0)
    meta["momentum"] = mom
    if mom <= 0.0:
        return 1.0, meta

    mult = 1.0 + (max_mult - 1.0) * recovery_progress
    return float(_clamp(mult, 1.0, max_mult)), meta


def _score_dispersion(score_map, eligible):
    vals = sorted([float(score_map[s]) for s in eligible if s in score_map], reverse=True)
    if len(vals) < 2:
        return 0.0
    ref = max(abs(vals[0]), 1e-6)
    if len(vals) >= 3:
        return float(max(0.0, (vals[0] - vals[2]) / ref))
    return float(max(0.0, (vals[0] - vals[-1]) / ref))


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
    prev_targets_raw = state.get("last_targets", [])
    allowed_symbols = set(stock.get("universe", []))
    allowed_symbols.add(str(stock.get("benchmark_symbol", "510300")))
    allowed_symbols.add(str(stock.get("defensive_symbol", "511010")))
    prev_targets = []
    dropped_targets = 0
    for t in prev_targets_raw:
        if not isinstance(t, dict):
            continue
        sym = str(t.get("symbol", ""))
        if sym in allowed_symbols:
            prev_targets.append({"symbol": sym, "target_weight": float(t.get("target_weight", 0.0))})
        else:
            dropped_targets += 1
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
        "dropped_stale_targets": int(dropped_targets),
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
    structural_cfg = _resolve_structural_cfg(model_cfg)
    structural_enabled = bool(structural_cfg["enabled"])

    universe = stock.get("universe", [])
    symbols = sorted(set(universe + [benchmark_symbol, defensive_symbol]))
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")

    closes = {}
    market_dates = {}
    vol = {}
    momentum = {}
    composite_momentum = {}
    relative_momentum = {}
    trend_ok_map = {}
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
    benchmark_close = closes[benchmark_symbol]
    benchmark_ma = None
    benchmark_trend_on = False
    if len(benchmark_close) >= ma_window:
        benchmark_ma = float(benchmark_close.tail(ma_window).mean())
        benchmark_trend_on = bool(benchmark_close.iloc[-1] >= benchmark_ma)

    bench_rel_momentum = None
    rel_window = int(structural_cfg["relative_momentum_window"])
    if structural_enabled and len(benchmark_close) > rel_window:
        bench_prev = float(benchmark_close.iloc[-1 - rel_window])
        if abs(bench_prev) > 1e-12:
            bench_rel_momentum = float(benchmark_close.iloc[-1] / bench_prev - 1.0)

    for s in risk_symbols:
        close = closes[s]
        needed = max(momentum_lb, vol_window) + 2
        if structural_enabled:
            needed = max(
                needed,
                max(structural_cfg["momentum_windows"]) + 2,
                int(structural_cfg["relative_momentum_window"]) + 2,
                int(structural_cfg["asset_trend_ma_window"]) + 2,
            )
        if len(close) < needed:
            continue

        if structural_enabled:
            long_prev = float(close.iloc[-1 - momentum_lb])
            if abs(long_prev) <= 1e-12:
                continue
            long_m = float(close.iloc[-1] / long_prev - 1.0)
            comp_m = _weighted_momentum(
                close=close,
                windows=structural_cfg["momentum_windows"],
                weights=structural_cfg["momentum_weights"],
            )
            if comp_m is None:
                continue
            rel_m = 0.0
            if bench_rel_momentum is not None and len(close) > rel_window:
                sym_prev = float(close.iloc[-1 - rel_window])
                if abs(sym_prev) > 1e-12:
                    sym_rel = float(close.iloc[-1] / sym_prev - 1.0)
                    rel_m = float(sym_rel - bench_rel_momentum)
            struct_signal = float(comp_m + structural_cfg["relative_momentum_weight"] * rel_m)
            blend = float(structural_cfg["score_blend"])
            m = float((1.0 - blend) * long_m + blend * struct_signal)
            asset_ma_window = int(structural_cfg["asset_trend_ma_window"])
            asset_ma = float(close.tail(asset_ma_window).mean())
            trend_ok = bool(close.iloc[-1] >= asset_ma)
        else:
            m = float(close.iloc[-1] / close.iloc[-momentum_lb] - 1.0)
            comp_m = m
            rel_m = 0.0
            trend_ok = True

        ret = close.pct_change().dropna().tail(vol_window)
        if ret.empty:
            continue
        v = float(ret.std())
        if v <= 0:
            continue
        momentum[s] = m
        composite_momentum[s] = float(comp_m)
        relative_momentum[s] = float(rel_m)
        trend_ok_map[s] = bool(trend_ok)
        vol[s] = v
        score[s] = m / max(v, 1e-6)

    breadth_universe = [s for s in score.keys() if s != benchmark_symbol] or list(score.keys())
    breadth = 0.0
    if breadth_universe:
        breadth_count = sum(
            1
            for s in breadth_universe
            if bool(trend_ok_map.get(s, False)) and float(composite_momentum.get(s, 0.0)) > 0.0
        )
        breadth = float(breadth_count / len(breadth_universe))

    if structural_enabled:
        if benchmark_trend_on and breadth >= float(structural_cfg["breadth_on_threshold"]):
            regime_state = "risk_on"
        elif benchmark_trend_on and breadth >= float(structural_cfg["breadth_off_threshold"]):
            regime_state = "neutral"
        else:
            regime_state = "risk_off"
    else:
        regime_state = "risk_on" if benchmark_trend_on else "risk_off"
    regime_state_initial = str(regime_state)
    bench_ret = benchmark_close.pct_change().dropna()
    phase2_pre_state = "normal"
    phase2_pre_ratio = None
    phase2_pre_mult = 1.0
    phase2_stepdown_applied = False
    if structural_enabled and bool(structural_cfg.get("phase2_enabled", False)):
        phase2_pre_state, phase2_pre_mult, phase2_pre_ratio = _compute_vol_state_multiplier(
            benchmark_ret=bench_ret,
            cfg=structural_cfg,
        )
        stepdown = int(structural_cfg.get("phase2_stress_regime_stepdown", 0))
        if phase2_pre_state == "stress" and stepdown > 0:
            if stepdown >= 2 and regime_state in {"risk_on", "neutral"}:
                regime_state = "risk_off"
                phase2_stepdown_applied = True
            elif stepdown >= 1 and regime_state == "risk_on":
                regime_state = "neutral"
                phase2_stepdown_applied = True
    regime_on = bool(regime_state != "risk_off")

    gate_cfg = model_cfg.get("exposure_gate", {})
    history_file = gate_cfg.get(
        "history_file",
        os.path.join(runtime["paths"]["output_dir"], "reports", "stock_paper_forward_history.csv"),
    )
    hist_strategy_ret, hist_bench_alloc = load_stock_return_history(history_file)
    effective_alloc, exposure_meta = evaluate_exposure_gate(hist_strategy_ret, hist_bench_alloc, alloc_pct, gate_cfg)
    stock_capital = total_capital * effective_alloc
    single_max = float(risk["position_limits"]["stock_single_max_pct"])

    if structural_enabled:
        eligible = [
            s
            for s in score.keys()
            if score[s] > min_score
            and ((not structural_cfg["eligibility_require_trend"]) or bool(trend_ok_map.get(s, False)))
            and float(composite_momentum.get(s, 0.0)) > 0.0
        ]
    else:
        eligible = [s for s in score.keys() if score[s] > min_score]
    sleeve_weights = {}
    signal_reason = "risk_off"
    risk_weights = {}
    risk_governor_meta = {
        "enabled": bool(model_cfg.get("risk_governor", {}).get("enabled", False)),
        "alloc_mult": 1.0,
        "structural_enabled": bool(structural_enabled),
        "regime_state": str(regime_state),
        "regime_state_initial": str(regime_state_initial),
        "phase2_stress_stepdown": int(structural_cfg.get("phase2_stress_regime_stepdown", 0)),
        "phase2_regime_stepdown_applied": bool(phase2_stepdown_applied),
        "benchmark_trend_on": bool(benchmark_trend_on),
        "benchmark_ma": benchmark_ma,
        "breadth": float(breadth),
        "neutral_alloc_mult": float(structural_cfg["neutral_alloc_mult"]),
        "trend_alloc_mult": 1.0,
        "breadth_alloc_mult": 1.0,
        "structural_alloc_mult": 1.0,
        "risk_alloc_base": float(effective_alloc),
        "risk_alloc_before_rg": float(effective_alloc),
        "phase2_enabled": bool(structural_enabled and structural_cfg.get("phase2_enabled", False)),
        "phase2_vol_state": str(phase2_pre_state),
        "phase2_vol_ratio": phase2_pre_ratio,
        "phase2_vol_mult": float(phase2_pre_mult),
        "phase2_recovery_mult": 1.0,
        "phase2_recovery_drawdown": None,
        "phase2_recovery_progress": None,
        "phase2_alloc_mult": 1.0,
        "signal_strength_enabled": bool(structural_cfg.get("signal_strength_enabled", False)),
        "signal_strength_value": None,
        "signal_strength_alloc_mult": 1.0,
        "score_dispersion": 0.0,
        "top_n_effective": int(top_n),
        "score_power_effective": float(risk_on_score_power),
    }
    risk_alloc = effective_alloc

    if regime_state in {"risk_on", "neutral"} and eligible:
        risk_alloc_base = float(effective_alloc)
        if regime_state == "neutral":
            risk_alloc_base *= float(structural_cfg["neutral_alloc_mult"])

        trend_alloc_mult = 1.0
        breadth_alloc_mult = 1.0
        structural_alloc_mult = 1.0
        if structural_enabled and benchmark_ma is not None and abs(float(benchmark_ma)) > 1e-12:
            trend_strength = float(benchmark_close.iloc[-1] / benchmark_ma - 1.0)
            trend_alloc_mult = _clamp(
                1.0 + trend_strength * float(structural_cfg["trend_strength_k"]),
                float(structural_cfg["trend_alloc_mult_min"]),
                float(structural_cfg["trend_alloc_mult_max"]),
            )
            breadth_target = (
                float(structural_cfg["breadth_on_threshold"])
                if regime_state == "risk_on"
                else float(structural_cfg["breadth_off_threshold"])
            )
            breadth_alloc_mult = _clamp(
                float(breadth / max(breadth_target, 1e-6)),
                float(structural_cfg["breadth_alloc_mult_min"]),
                float(structural_cfg["breadth_alloc_mult_max"]),
            )
            structural_alloc_mult = _clamp(
                float(trend_alloc_mult * breadth_alloc_mult),
                float(structural_cfg["total_alloc_mult_min"]),
                float(structural_cfg["total_alloc_mult_max"]),
            )

        top_n_base = int(top_n if regime_state == "risk_on" else min(top_n, structural_cfg["neutral_top_n"]))
        top_n_eff = max(1, top_n_base)
        score_power_eff = float(risk_on_score_power)
        phase2_alloc_mult = 1.0
        phase2_vol_state = "normal"
        phase2_vol_ratio = None
        phase2_vol_mult = 1.0
        phase2_recovery_mult = 1.0
        phase2_recovery_meta = {}
        if structural_enabled and bool(structural_cfg.get("phase2_enabled", False)):
            phase2_vol_state = str(phase2_pre_state)
            phase2_vol_mult = float(phase2_pre_mult)
            phase2_vol_ratio = phase2_pre_ratio
            phase2_recovery_mult, phase2_recovery_meta = _compute_drawdown_recovery_multiplier(
                benchmark_close=benchmark_close,
                cfg=structural_cfg,
            )
            phase2_alloc_mult = _clamp(
                float(phase2_vol_mult * phase2_recovery_mult),
                float(structural_cfg["phase2_total_mult_min"]),
                float(structural_cfg["phase2_total_mult_max"]),
            )
            if bool(structural_cfg.get("phase2_adaptive_enabled", True)):
                score_disp = _score_dispersion(score_map=score, eligible=eligible)
                if phase2_vol_state == "calm":
                    top_n_eff += int(structural_cfg["phase2_adaptive_calm_top_n_delta"])
                    score_power_eff *= float(structural_cfg["phase2_adaptive_score_power_calm_mult"])
                elif phase2_vol_state == "stress":
                    top_n_eff += int(structural_cfg["phase2_adaptive_stress_top_n_delta"])
                    score_power_eff *= float(structural_cfg["phase2_adaptive_score_power_stress_mult"])
                if score_disp >= float(structural_cfg["phase2_adaptive_high_dispersion_threshold"]):
                    top_n_eff += int(structural_cfg["phase2_adaptive_high_dispersion_top_n_delta"])
                top_n_eff = int(
                    _clamp(
                        float(top_n_eff),
                        float(structural_cfg["phase2_adaptive_min_top_n"]),
                        float(structural_cfg["phase2_adaptive_max_top_n"]),
                    )
                )
                top_n_eff = min(top_n_eff, max(1, len(eligible)))
                risk_governor_meta["score_dispersion"] = float(score_disp)

        picks = sorted(eligible, key=lambda x: score[x], reverse=True)[: max(1, top_n_eff)]
        signal_strength_mult = 1.0
        signal_strength_value = None
        if bool(structural_cfg.get("signal_strength_enabled", False)) and picks:
            strength_floor = float(structural_cfg.get("signal_strength_floor", 0.0))
            strength_ref = max(1e-6, float(structural_cfg.get("signal_strength_ref", 0.06)))
            strength_curve = max(0.25, float(structural_cfg.get("signal_strength_curve", 1.0)))
            strength_vals = [
                max(float(composite_momentum.get(s, momentum.get(s, 0.0))) - strength_floor, 0.0) for s in picks
            ]
            if strength_vals:
                signal_strength_value = float(sum(strength_vals) / len(strength_vals))
                signal_strength_mult = _clamp(
                    float((signal_strength_value / strength_ref) ** strength_curve),
                    float(structural_cfg.get("signal_strength_min_alloc_mult", 0.70)),
                    float(structural_cfg.get("signal_strength_max_alloc_mult", 1.10)),
                )

        risk_alloc_pre_rg = float(risk_alloc_base * structural_alloc_mult * phase2_alloc_mult * signal_strength_mult)
        alloc_mult, rg_meta = compute_risk_alloc_multiplier(benchmark_close, model_cfg)
        risk_governor_meta.update(rg_meta)
        risk_governor_meta["trend_alloc_mult"] = float(trend_alloc_mult)
        risk_governor_meta["breadth_alloc_mult"] = float(breadth_alloc_mult)
        risk_governor_meta["structural_alloc_mult"] = float(structural_alloc_mult)
        risk_governor_meta["risk_alloc_base"] = float(risk_alloc_base)
        risk_governor_meta["risk_alloc_before_rg"] = float(risk_alloc_pre_rg)
        risk_governor_meta["phase2_vol_state"] = str(phase2_vol_state)
        risk_governor_meta["phase2_vol_ratio"] = phase2_vol_ratio
        risk_governor_meta["phase2_vol_mult"] = float(phase2_vol_mult)
        risk_governor_meta["phase2_recovery_mult"] = float(phase2_recovery_mult)
        risk_governor_meta["phase2_recovery_drawdown"] = phase2_recovery_meta.get("drawdown")
        risk_governor_meta["phase2_recovery_progress"] = phase2_recovery_meta.get("recovery_progress")
        risk_governor_meta["phase2_alloc_mult"] = float(phase2_alloc_mult)
        risk_governor_meta["signal_strength_value"] = signal_strength_value
        risk_governor_meta["signal_strength_alloc_mult"] = float(signal_strength_mult)
        risk_governor_meta["top_n_effective"] = int(top_n_eff)
        risk_governor_meta["score_power_effective"] = float(score_power_eff)

        risk_alloc = float(risk_alloc_pre_rg * alloc_mult)
        risk_weights = build_risk_on_weights(
            score_map=score,
            vol_map=vol,
            picks=picks,
            score_mix=risk_on_score_mix,
            score_floor=risk_on_score_floor,
            score_power=score_power_eff,
        )
        sleeve_weights = risk_weights
        signal_reason = "risk_on" if regime_state == "risk_on" else "risk_on_neutral"
    else:
        sleeve_weights = {defensive_symbol: 1.0}
        if not benchmark_trend_on:
            signal_reason = "benchmark_below_ma"
        elif phase2_stepdown_applied and regime_state == "risk_off":
            signal_reason = "phase2_stress_risk_off"
        elif regime_state == "risk_off":
            signal_reason = "breadth_risk_off"
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
                "composite_momentum": round(float(composite_momentum.get(s, momentum[s])), 6),
                "relative_momentum": round(float(relative_momentum.get(s, 0.0)), 6),
                "trend_ok": bool(trend_ok_map.get(s, True)),
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
            "structural_upgrade": structural_cfg,
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
