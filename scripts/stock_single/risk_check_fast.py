#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_DISABLED, EXIT_OK, EXIT_OUTPUT_ERROR


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_pool_symbols(stock_single: dict) -> list[str]:
    pool_file = stock_single.get("paths", {}).get("pool_file", "./outputs/orders/stock_single_pool.json")
    if not os.path.exists(pool_file):
        return []
    with open(pool_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [str(x).strip() for x in payload.get("symbols", []) if str(x).strip()]


def load_snapshot(snapshot_file: str) -> pd.DataFrame:
    if not os.path.exists(snapshot_file):
        return pd.DataFrame(columns=["symbol", "ret_5m"])
    df = pd.read_csv(snapshot_file)
    if "symbol" not in df.columns or "ret_5m" not in df.columns:
        raise RuntimeError("risk snapshot missing required columns: symbol, ret_5m")
    x = df.copy()
    x["symbol"] = x["symbol"].astype(str).str.strip()
    x["ret_5m"] = pd.to_numeric(x["ret_5m"], errors="coerce")
    x = x.dropna(subset=["symbol", "ret_5m"])
    return x


def calc_portfolio_ret_5m(snapshot_df: pd.DataFrame) -> float:
    if snapshot_df.empty:
        return 0.0
    if "weight" not in snapshot_df.columns:
        return float(snapshot_df["ret_5m"].mean())

    w = pd.to_numeric(snapshot_df["weight"], errors="coerce").fillna(0.0)
    w = w.clip(lower=0.0)
    w_sum = float(w.sum())
    if w_sum <= 1e-12:
        return float(snapshot_df["ret_5m"].mean())
    return float((snapshot_df["ret_5m"] * w / w_sum).sum())


def evaluate_fast_risk(stock_single: dict) -> tuple[dict, dict]:
    risk_cfg = stock_single.get("risk", {}).get("fast_check", {})
    snapshot_file = risk_cfg.get("snapshot_file", "./data/stock_single/intraday_risk_latest.csv")
    trigger_portfolio = float(risk_cfg.get("trigger_portfolio_ret_5m", -0.008))
    trigger_single = float(risk_cfg.get("trigger_single_ret_5m", -0.020))
    trigger_vol = float(risk_cfg.get("trigger_vol_zscore", 3.0))

    pool_symbols = set(load_pool_symbols(stock_single))
    snapshot_df = load_snapshot(snapshot_file)
    snapshot_status = "ok" if os.path.exists(snapshot_file) else "missing_snapshot"

    if pool_symbols:
        snapshot_df = snapshot_df[snapshot_df["symbol"].isin(pool_symbols)].copy()

    portfolio_ret_5m = calc_portfolio_ret_5m(snapshot_df)
    single_crash_symbols = (
        sorted(snapshot_df.loc[snapshot_df["ret_5m"] <= trigger_single, "symbol"].astype(str).tolist())
        if not snapshot_df.empty
        else []
    )
    if "vol_zscore" in snapshot_df.columns:
        z = pd.to_numeric(snapshot_df["vol_zscore"], errors="coerce")
        vol_spike_symbols = sorted(snapshot_df.loc[z >= trigger_vol, "symbol"].astype(str).tolist())
    else:
        vol_spike_symbols = []

    portfolio_trigger = bool(portfolio_ret_5m <= trigger_portfolio)
    single_trigger = bool(len(single_crash_symbols) > 0)
    vol_trigger = bool(len(vol_spike_symbols) > 0)
    triggered = bool(portfolio_trigger or single_trigger or vol_trigger)

    block_new_buys = bool(triggered and risk_cfg.get("block_new_buys_on_trigger", True))
    force_sell_symbols = (
        single_crash_symbols if bool(risk_cfg.get("force_sell_on_single_crash", True)) else []
    )
    stage = "risk" if triggered else "normal"

    state = {
        "ts": datetime.now().isoformat(),
        "market": "CN_STOCK_SINGLE",
        "mode": "fast_risk_check_5m",
        "snapshot_file": snapshot_file,
        "snapshot_status": snapshot_status,
        "universe_count": int(len(snapshot_df)),
        "stage": stage,
        "triggered": triggered,
        "metrics": {
            "portfolio_ret_5m": float(round(portfolio_ret_5m, 8)),
            "single_crash_count": int(len(single_crash_symbols)),
            "vol_spike_count": int(len(vol_spike_symbols)),
        },
        "trigger_flags": {
            "portfolio_drop": portfolio_trigger,
            "single_crash": single_trigger,
            "vol_spike": vol_trigger,
        },
        "thresholds": {
            "trigger_portfolio_ret_5m": trigger_portfolio,
            "trigger_single_ret_5m": trigger_single,
            "trigger_vol_zscore": trigger_vol,
        },
        "actions": {
            "block_new_buys": block_new_buys,
            "force_sell_symbols": force_sell_symbols,
        },
    }

    alerts = {
        "ts": state["ts"],
        "triggered": triggered,
        "alerts": [
            {
                "level": "risk",
                "type": "portfolio_drop_5m",
                "detail": f"portfolio_ret_5m={portfolio_ret_5m:.4f} <= {trigger_portfolio:.4f}",
            }
        ]
        if portfolio_trigger
        else [],
    }
    if single_trigger:
        alerts["alerts"].append(
            {
                "level": "risk",
                "type": "single_crash_5m",
                "detail": ",".join(single_crash_symbols),
            }
        )
    if vol_trigger:
        alerts["alerts"].append(
            {
                "level": "warn",
                "type": "vol_spike_zscore",
                "detail": ",".join(vol_spike_symbols),
            }
        )
    return state, alerts


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[stock-single-risk] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_single.get("enabled", False):
        print("[stock-single] disabled")
        return EXIT_DISABLED

    fast_cfg = stock_single.get("risk", {}).get("fast_check", {})
    if not bool(fast_cfg.get("enabled", True)):
        print("[stock-single-risk] fast_check disabled")
        return EXIT_DISABLED

    try:
        state, alerts = evaluate_fast_risk(stock_single)
        paths = stock_single.get("paths", {})
        state_file = paths.get("risk_state_file", "./outputs/orders/stock_single_risk_state.json")
        alert_file = paths.get("risk_alert_file", "./outputs/orders/stock_single_risk_alerts.json")
        ensure_dir(os.path.dirname(state_file) or ".")
        ensure_dir(os.path.dirname(alert_file) or ".")
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        with open(alert_file, "w", encoding="utf-8") as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"[stock-single-risk] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[stock-single-risk] failed: {e}")
        return EXIT_CONFIG_ERROR

    print(
        "[stock-single-risk] "
        f"stage={state['stage']} triggered={state['triggered']} "
        f"universe={state['universe_count']} "
        f"portfolio_ret_5m={state['metrics']['portfolio_ret_5m']:.4f}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
