#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import sys
from collections import Counter
from datetime import datetime

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
from core.execution_guard import resolve_min_turnover
from core.exposure_gate import evaluate_exposure_gate


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_price_frame(data_dir: str, symbols: list) -> pd.DataFrame:
    frames = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"])
        frames[s] = df.sort_values("date").set_index("date")[["close"]]

    if len(frames) < len(symbols):
        missing = [s for s in symbols if s not in frames]
        raise RuntimeError(f"missing stock data files: {missing}")

    dates = sorted(set.intersection(*[set(v.index) for v in frames.values()]))
    if len(dates) < 400:
        raise RuntimeError("not enough common stock history (<400 rows)")

    return pd.DataFrame({s: frames[s].loc[dates, "close"] for s in symbols}, index=dates).sort_index()


def annualized_return(nav: pd.Series, periods_per_year: int = 252) -> float:
    if len(nav) < 2:
        return 0.0
    total = float(nav.iloc[-1] / nav.iloc[0])
    years = max((len(nav) - 1) / periods_per_year, 1e-9)
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    return float((nav / nav.cummax() - 1.0).min())


def sharpe_ratio(ret: pd.Series, periods_per_year: int = 252) -> float:
    if len(ret) < 2:
        return 0.0
    sd = ret.std()
    if sd == 0:
        return 0.0
    return float((ret.mean() / sd) * math.sqrt(periods_per_year))


def weights_to_json(weights: dict) -> str:
    clean = {k: round(float(v), 6) for k, v in weights.items() if float(v) > 1e-8}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True)


def calc_turnover(old_w: dict, new_w: dict, symbols: list) -> float:
    return float(sum(abs(new_w.get(s, 0.0) - old_w.get(s, 0.0)) for s in symbols))


def build_signal(
    prices: pd.DataFrame,
    i: int,
    symbols: list,
    benchmark_symbol: str,
    defensive_symbol: str,
    alloc_pct: float,
    single_max: float,
    params: dict,
) -> tuple[dict, bool, str, list]:
    risk_symbols = [s for s in symbols if s != defensive_symbol]
    momentum_lb = int(params["momentum_lb"])
    ma_window = int(params["ma_window"])
    vol_window = int(params["vol_window"])
    top_n = int(params["top_n"])
    min_score = float(params.get("min_score", 0.0))
    defensive_bypass_single_max = bool(params.get("defensive_bypass_single_max", True))

    ret = prices.pct_change().fillna(0.0)
    scores = {}
    for s in risk_symbols:
        m = prices[s].iloc[i] / prices[s].iloc[i - momentum_lb] - 1.0
        vol = float(ret[s].iloc[i - vol_window + 1 : i + 1].std())
        if vol <= 0:
            continue
        scores[s] = {"momentum": float(m), "vol": vol, "score": float(m / max(vol, 1e-6))}

    regime_on = bool(prices[benchmark_symbol].iloc[i] >= prices[benchmark_symbol].iloc[i - ma_window + 1 : i + 1].mean())
    eligible = [s for s in scores if scores[s]["score"] > min_score]

    target = {s: 0.0 for s in symbols}
    reason = "risk_off"
    if regime_on and eligible:
        picks = sorted(eligible, key=lambda s: scores[s]["score"], reverse=True)[:top_n]
        inv_vol = {s: 1.0 / max(scores[s]["vol"], 1e-6) for s in picks}
        total_inv = sum(inv_vol.values())
        for s, v in inv_vol.items():
            target[s] = alloc_pct * (v / total_inv)
        for s in picks:
            if (not defensive_bypass_single_max) or s != defensive_symbol:
                target[s] = min(target[s], single_max)
        used = sum(target.values())
        if used < alloc_pct:
            if (not defensive_bypass_single_max):
                room = max(0.0, single_max - target[defensive_symbol])
                target[defensive_symbol] += min(room, alloc_pct - used)
            else:
                target[defensive_symbol] += alloc_pct - used
        reason = "risk_on"
    else:
        target[defensive_symbol] = alloc_pct
        if not regime_on:
            reason = "benchmark_below_ma"
        elif not eligible:
            reason = "no_positive_momentum"

    score_rows = []
    for s in sorted(scores.keys(), key=lambda x: scores[x]["score"], reverse=True):
        score_rows.append(
            {
                "symbol": s,
                "momentum": round(scores[s]["momentum"], 6),
                "vol": round(scores[s]["vol"], 6),
                "score": round(scores[s]["score"], 6),
            }
        )
    return target, regime_on, reason, score_rows


def run_paper_forward(runtime, stock, risk):
    model_cfg = stock.get("global_model", {})
    guard_cfg = model_cfg.get("execution_guard", {})
    exposure_cfg = model_cfg.get("exposure_gate", {})

    benchmark_symbol = stock.get("benchmark_symbol", "510300")
    defensive_symbol = stock.get("defensive_symbol", "511010")
    symbols = sorted(set(stock.get("universe", []) + [benchmark_symbol, defensive_symbol]))
    alloc_pct = float(stock.get("capital_alloc_pct", 0.7))
    single_max = float(risk["position_limits"]["stock_single_max_pct"])

    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    report_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    ensure_dir(report_dir)

    prices = load_price_frame(data_dir, symbols)
    warmup_min_days = int(model_cfg.get("warmup_min_days", 126))
    warmup = max(
        int(model_cfg.get("momentum_lb", 252)),
        int(model_cfg.get("ma_window", 200)),
        int(model_cfg.get("vol_window", 20)),
        warmup_min_days,
    ) + 1
    if warmup >= len(prices) - 1:
        raise RuntimeError("not enough history for paper-forward simulation")

    fee = float(model_cfg.get("fee", 0.0008))
    min_rebalance_days = int(guard_cfg.get("min_rebalance_days", model_cfg.get("rebalance_days", 20)))
    min_turnover_base = float(guard_cfg.get("min_turnover", 0.05))
    force_regime = bool(guard_cfg.get("force_rebalance_on_regime_change", True))
    guard_enabled = bool(guard_cfg.get("enabled", True))

    weights = {s: 0.0 for s in symbols}
    weights[defensive_symbol] = alloc_pct
    last_rebalance_date = None
    last_regime_on = None

    strategy_nav = 1.0
    benchmark_nav_raw = 1.0
    benchmark_nav_alloc = 1.0
    strategy_peak = 1.0
    benchmark_peak_raw = 1.0
    benchmark_peak_alloc = 1.0

    records = []
    hist_strategy_ret = []
    hist_benchmark_ret_alloc = []
    exposure_stage_count = Counter()
    min_turnover_meta = {"enabled": False, "base_min_turnover": min_turnover_base, "effective_min_turnover": min_turnover_base}
    last_exposure_meta = {
        "enabled": bool(exposure_cfg.get("enabled", False)),
        "base_alloc_pct": alloc_pct,
        "effective_alloc_pct": alloc_pct,
        "stage": "disabled",
        "reason": "init",
    }
    for i in range(warmup, len(prices) - 1):
        date = prices.index[i]
        next_date = prices.index[i + 1]
        alloc_effective, exposure_meta = evaluate_exposure_gate(
            hist_strategy_ret,
            hist_benchmark_ret_alloc,
            alloc_pct,
            exposure_cfg,
        )
        min_turnover, min_turnover_meta = resolve_min_turnover(min_turnover_base, alloc_effective, guard_cfg)
        proposed, regime_on, signal_reason, score_rows = build_signal(
            prices=prices,
            i=i,
            symbols=symbols,
            benchmark_symbol=benchmark_symbol,
            defensive_symbol=defensive_symbol,
            alloc_pct=alloc_effective,
            single_max=single_max,
            params=model_cfg,
        )

        action = "rebalance"
        action_reason = "normal"
        executed = proposed
        proposed_turnover = calc_turnover(weights, proposed, symbols)
        days_since = None

        if guard_enabled and last_rebalance_date is not None:
            days_since = (date.date() - last_rebalance_date).days
            if force_regime and last_regime_on is not None and regime_on != last_regime_on:
                action = "rebalance"
                action_reason = "regime_changed"
                executed = proposed
            elif days_since < min_rebalance_days:
                action = "hold"
                action_reason = "min_rebalance_days"
                executed = weights.copy()
            elif proposed_turnover < min_turnover:
                action = "hold"
                action_reason = "turnover_below_threshold"
                executed = weights.copy()

        turnover = calc_turnover(weights, executed, symbols)
        if action == "rebalance":
            last_rebalance_date = date.date()

        strategy_ret = 0.0
        for s, w in executed.items():
            if w == 0.0:
                continue
            strategy_ret += w * (prices[s].iloc[i + 1] / prices[s].iloc[i] - 1.0)
        strategy_ret -= turnover * fee

        benchmark_ret_raw = prices[benchmark_symbol].iloc[i + 1] / prices[benchmark_symbol].iloc[i] - 1.0
        benchmark_ret_alloc = alloc_pct * benchmark_ret_raw
        excess_ret_vs_raw = strategy_ret - benchmark_ret_raw
        excess_ret_vs_alloc = strategy_ret - benchmark_ret_alloc

        strategy_nav *= 1.0 + strategy_ret
        benchmark_nav_raw *= 1.0 + benchmark_ret_raw
        benchmark_nav_alloc *= 1.0 + benchmark_ret_alloc
        strategy_peak = max(strategy_peak, strategy_nav)
        benchmark_peak_raw = max(benchmark_peak_raw, benchmark_nav_raw)
        benchmark_peak_alloc = max(benchmark_peak_alloc, benchmark_nav_alloc)

        records.append(
            {
                "date": date.date().isoformat(),
                "next_date": next_date.date().isoformat(),
                "action": action,
                "action_reason": action_reason,
                "alloc_pct_base": alloc_pct,
                "alloc_pct_effective": float(alloc_effective),
                "exposure_gate_stage": str(exposure_meta.get("stage", "unknown")),
                "exposure_gate_reason": str(exposure_meta.get("reason", "")),
                "exposure_gate_agg_excess_ann_worst": exposure_meta.get("agg_excess_ann_worst"),
                "min_turnover_effective": float(min_turnover),
                "regime_on": bool(regime_on),
                "signal_reason": signal_reason,
                "days_since_last_rebalance": days_since,
                "proposed_turnover": round(proposed_turnover, 6),
                "executed_turnover": round(turnover, 6),
                "strategy_ret": strategy_ret,
                "benchmark_ret_raw": benchmark_ret_raw,
                "benchmark_ret_alloc": benchmark_ret_alloc,
                "excess_ret_vs_raw": excess_ret_vs_raw,
                "excess_ret_vs_alloc": excess_ret_vs_alloc,
                "strategy_nav": strategy_nav,
                "benchmark_nav_raw": benchmark_nav_raw,
                "benchmark_nav_alloc": benchmark_nav_alloc,
                "excess_nav_vs_raw": strategy_nav - benchmark_nav_raw,
                "excess_nav_vs_alloc": strategy_nav - benchmark_nav_alloc,
                "strategy_dd": strategy_nav / strategy_peak - 1.0,
                "benchmark_dd_raw": benchmark_nav_raw / benchmark_peak_raw - 1.0,
                "benchmark_dd_alloc": benchmark_nav_alloc / benchmark_peak_alloc - 1.0,
                "executed_weights": weights_to_json(executed),
                "proposed_weights": weights_to_json(proposed),
                "scores": json.dumps(score_rows, ensure_ascii=False),
            }
        )
        hist_strategy_ret.append(float(strategy_ret))
        hist_benchmark_ret_alloc.append(float(benchmark_ret_alloc))
        exposure_stage_count[str(exposure_meta.get("stage", "unknown"))] += 1
        last_exposure_meta = exposure_meta
        weights = executed
        last_regime_on = regime_on

    df = pd.DataFrame(records)
    history_file = os.path.join(report_dir, "stock_paper_forward_history.csv")
    df.to_csv(history_file, index=False, encoding="utf-8")

    strategy_nav_series = pd.Series([1.0] + df["strategy_nav"].tolist())
    benchmark_nav_raw_series = pd.Series([1.0] + df["benchmark_nav_raw"].tolist())
    benchmark_nav_alloc_series = pd.Series([1.0] + df["benchmark_nav_alloc"].tolist())
    strategy_ret_series = df["strategy_ret"]
    benchmark_ret_raw_series = df["benchmark_ret_raw"]
    benchmark_ret_alloc_series = df["benchmark_ret_alloc"]
    excess_ret_vs_raw_series = df["excess_ret_vs_raw"]
    excess_ret_vs_alloc_series = df["excess_ret_vs_alloc"]

    last = df.iloc[-1]
    rolling_20_raw = excess_ret_vs_raw_series.tail(20).sum() if len(df) >= 20 else excess_ret_vs_raw_series.sum()
    rolling_60_raw = excess_ret_vs_raw_series.tail(60).sum() if len(df) >= 60 else excess_ret_vs_raw_series.sum()
    rolling_20_alloc = excess_ret_vs_alloc_series.tail(20).sum() if len(df) >= 20 else excess_ret_vs_alloc_series.sum()
    rolling_60_alloc = excess_ret_vs_alloc_series.tail(60).sum() if len(df) >= 60 else excess_ret_vs_alloc_series.sum()
    action_cnt = Counter(df["action"].tolist())
    reason_cnt = Counter(df["action_reason"].tolist())

    strategy_ann = annualized_return(strategy_nav_series, 252)
    benchmark_ann_raw = annualized_return(benchmark_nav_raw_series, 252)
    benchmark_ann_alloc = annualized_return(benchmark_nav_alloc_series, 252)

    summary = {
        "ts": datetime.now().isoformat(),
        "market": "stock",
        "mode": "global_momentum",
        "history_file": history_file,
        "date_range": {"from": str(df["date"].iloc[0]), "to": str(df["date"].iloc[-1])},
        "latest": {
            "date": str(last["date"]),
            "next_date": str(last["next_date"]),
            "action": str(last["action"]),
            "action_reason": str(last["action_reason"]),
            "regime_on": bool(last["regime_on"]),
            "signal_reason": str(last["signal_reason"]),
            "executed_turnover": float(last["executed_turnover"]),
            "strategy_nav": float(last["strategy_nav"]),
            "benchmark_nav_raw": float(last["benchmark_nav_raw"]),
            "benchmark_nav_alloc": float(last["benchmark_nav_alloc"]),
            "excess_nav_vs_raw": float(last["excess_nav_vs_raw"]),
            "excess_nav_vs_alloc": float(last["excess_nav_vs_alloc"]),
            "strategy_dd": float(last["strategy_dd"]),
            "benchmark_dd_raw": float(last["benchmark_dd_raw"]),
            "benchmark_dd_alloc": float(last["benchmark_dd_alloc"]),
        },
        "aggregate": {
            "periods": int(len(df)),
            "strategy_annual_return": strategy_ann,
            "benchmark_annual_return_raw": benchmark_ann_raw,
            "benchmark_annual_return_alloc": benchmark_ann_alloc,
            "excess_annual_return_vs_raw": strategy_ann - benchmark_ann_raw,
            "excess_annual_return_vs_alloc": strategy_ann - benchmark_ann_alloc,
            "strategy_max_drawdown": max_drawdown(strategy_nav_series),
            "benchmark_max_drawdown_raw": max_drawdown(benchmark_nav_raw_series),
            "benchmark_max_drawdown_alloc": max_drawdown(benchmark_nav_alloc_series),
            "strategy_sharpe": sharpe_ratio(strategy_ret_series, 252),
            "benchmark_sharpe_raw": sharpe_ratio(benchmark_ret_raw_series, 252),
            "benchmark_sharpe_alloc": sharpe_ratio(benchmark_ret_alloc_series, 252),
            "excess_sharpe_vs_raw": sharpe_ratio(excess_ret_vs_raw_series, 252),
            "excess_sharpe_vs_alloc": sharpe_ratio(excess_ret_vs_alloc_series, 252),
            "total_turnover": float(df["executed_turnover"].sum()),
            "avg_turnover": float(df["executed_turnover"].mean()),
            "trades": int((df["executed_turnover"] > 1e-9).sum()),
            "action_count": dict(action_cnt),
            "reason_count": dict(reason_cnt),
        },
        "rolling": {
            "excess_return_20d_vs_raw": float(rolling_20_raw),
            "excess_return_60d_vs_raw": float(rolling_60_raw),
            "excess_return_20d_vs_alloc": float(rolling_20_alloc),
            "excess_return_60d_vs_alloc": float(rolling_60_alloc),
        },
        "guard_params": {
            "enabled": guard_enabled,
            "min_rebalance_days": min_rebalance_days,
            "min_turnover_base": min_turnover_base,
            "min_turnover": float(last["min_turnover_effective"]),
            "dynamic_min_turnover": min_turnover_meta,
            "force_rebalance_on_regime_change": force_regime,
        },
        "exposure_gate": {
            "enabled": bool(exposure_cfg.get("enabled", False)),
            "stage_count": dict(exposure_stage_count),
            "latest": last_exposure_meta,
            "params": exposure_cfg,
        },
    }

    latest_file = os.path.join(report_dir, "stock_paper_forward_latest.json")
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary, history_file, latest_file


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock = load_yaml("config/stock.yaml")
        risk = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[paper-forward] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock.get("enabled", False):
        print("[stock] disabled")
        return EXIT_DISABLED
    if stock.get("mode") != "global_momentum":
        print("[stock] mode is not global_momentum, skip paper forward report")
        return EXIT_OK

    try:
        summary, history_file, latest_file = run_paper_forward(runtime, stock, risk)
    except OSError as e:
        print(f"[paper-forward] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[paper-forward] signal error: {e}")
        return EXIT_SIGNAL_ERROR

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[paper-forward] history -> {history_file}")
    print(f"[paper-forward] latest  -> {latest_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
