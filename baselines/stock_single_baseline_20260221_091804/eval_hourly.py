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


def load_pool(stock_single: dict) -> list[str]:
    pool_file = stock_single.get("paths", {}).get("pool_file", "./outputs/orders/stock_single_pool.json")
    if not os.path.exists(pool_file):
        return []
    with open(pool_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [str(x) for x in payload.get("symbols", [])]


def load_risk_state(stock_single: dict) -> dict:
    state_file = stock_single.get("paths", {}).get("risk_state_file", "./outputs/orders/stock_single_risk_state.json")
    if not os.path.exists(state_file):
        return {
            "source": "default",
            "stage": "unknown",
            "triggered": False,
            "actions": {"block_new_buys": False, "force_sell_symbols": []},
        }
    with open(state_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {
        "source": state_file,
        "stage": str(payload.get("stage", "unknown")),
        "triggered": bool(payload.get("triggered", False)),
        "actions": payload.get("actions", {}) or {},
    }
    out.setdefault("actions", {})
    out["actions"].setdefault("block_new_buys", False)
    out["actions"].setdefault("force_sell_symbols", [])
    return out


def load_score_snapshot() -> pd.DataFrame:
    fp = "./data/stock_single/hourly_scores_latest.csv"
    if not os.path.exists(fp):
        return pd.DataFrame(columns=["symbol", "score", "last_price", "atr14"])
    df = pd.read_csv(fp)
    for col in ["symbol", "score", "last_price", "atr14"]:
        if col not in df.columns:
            raise RuntimeError(f"hourly score file missing column: {col}")
    return df


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def resolve_signal_thresholds(
    score_df: pd.DataFrame,
    symbols: list[str],
    signal_cfg: dict,
    max_positions: int,
) -> tuple[float, float, dict]:
    mode = str(signal_cfg.get("trigger_mode", "threshold")).strip().lower()
    if mode not in {"threshold", "quantile", "topk"}:
        mode = "threshold"

    buy_threshold_static = float(signal_cfg.get("buy_threshold", 1.0))
    sell_threshold_static = float(signal_cfg.get("sell_threshold", -0.5))

    x = score_df[score_df["symbol"].astype(str).isin(set(symbols))].copy()
    x["score"] = pd.to_numeric(x["score"], errors="coerce")
    valid = x["score"].dropna().astype(float)
    if valid.empty:
        return buy_threshold_static, sell_threshold_static, {
            "mode": "threshold",
            "fallback": "no_valid_scores",
            "buy_threshold": buy_threshold_static,
            "sell_threshold": sell_threshold_static,
        }

    if mode == "quantile":
        q_buy = clamp(float(signal_cfg.get("buy_score_quantile", 0.75)), 0.0, 1.0)
        q_sell = clamp(float(signal_cfg.get("sell_score_quantile", 0.35)), 0.0, 1.0)
        if q_sell >= q_buy:
            q_sell = max(0.0, q_buy - 0.10)
        buy_threshold = float(valid.quantile(q_buy))
        sell_threshold = float(valid.quantile(q_sell))
        if sell_threshold >= buy_threshold:
            sell_threshold = buy_threshold - 1e-6
        return buy_threshold, sell_threshold, {
            "mode": "quantile",
            "buy_score_quantile": q_buy,
            "sell_score_quantile": q_sell,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }

    if mode == "topk":
        ranked = valid.sort_values(ascending=False).reset_index(drop=True)
        buy_top_k = int(signal_cfg.get("buy_top_k", max_positions))
        buy_top_k = max(1, min(buy_top_k, len(ranked)))
        hold_top_k = int(signal_cfg.get("hold_top_k", max(buy_top_k + 1, buy_top_k * 2)))
        hold_top_k = max(buy_top_k, min(hold_top_k, len(ranked)))

        buy_threshold = float(ranked.iloc[buy_top_k - 1])
        sell_threshold = float(ranked.iloc[hold_top_k - 1])
        if sell_threshold >= buy_threshold:
            sell_threshold = buy_threshold - 1e-6
        return buy_threshold, sell_threshold, {
            "mode": "topk",
            "buy_top_k": buy_top_k,
            "hold_top_k": hold_top_k,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }

    return buy_threshold_static, sell_threshold_static, {
        "mode": "threshold",
        "buy_threshold": buy_threshold_static,
        "sell_threshold": sell_threshold_static,
    }


def build_signals(runtime: dict, stock_single: dict) -> tuple[list[dict], str]:
    symbols = load_pool(stock_single)
    score_df = load_score_snapshot()
    score_map = {str(r["symbol"]): r for _, r in score_df.iterrows()}
    risk_state = load_risk_state(stock_single)
    force_sell_symbols = set([str(x) for x in risk_state["actions"].get("force_sell_symbols", [])])
    block_new_buys = bool(risk_state["actions"].get("block_new_buys", False))

    s_cfg = stock_single.get("signal", {})
    r_cfg = stock_single.get("risk", {})
    max_positions = int(stock_single.get("max_positions", 10))
    buy_threshold, sell_threshold, threshold_meta = resolve_signal_thresholds(
        score_df=score_df,
        symbols=symbols,
        signal_cfg=s_cfg,
        max_positions=max_positions,
    )
    buy_buffer = float(s_cfg.get("buy_buffer_bps", 15.0)) * 1e-4
    sell_buffer = float(s_cfg.get("sell_buffer_bps", 15.0)) * 1e-4
    target_w = float(s_cfg.get("per_signal_target_weight", 0.08))
    single_max = float(r_cfg.get("single_max_pct", 0.12))

    signals = []
    for symbol in symbols:
        row = score_map.get(symbol)
        if row is None:
            signals.append(
                {
                    "symbol": symbol,
                    "action": "HOLD",
                    "reason": "missing_score_snapshot",
                }
            )
            continue

        score = float(row["score"])
        last_price = float(row["last_price"])
        atr = max(float(row["atr14"]), 0.01)

        action = "HOLD"
        reason = "score_in_neutral_range"
        entry_price = None
        exit_price = None
        stop_price = None
        target_weight = 0.0

        if symbol in force_sell_symbols:
            action = "SELL"
            reason = "fast_risk_force_sell"
            exit_price = round(last_price * (1.0 - sell_buffer), 3)
        elif score >= buy_threshold:
            action = "BUY"
            reason = f"score={score:.3f}>=buy_threshold"
            entry_price = round(last_price * (1.0 + buy_buffer), 3)
            stop_price = round(entry_price - 1.2 * atr, 3)
            exit_price = round(entry_price + 2.2 * atr, 3)
            target_weight = min(target_w, single_max)
        elif score <= sell_threshold:
            action = "SELL"
            reason = f"score={score:.3f}<=sell_threshold"
            exit_price = round(last_price * (1.0 - sell_buffer), 3)

        if action == "BUY" and block_new_buys:
            action = "HOLD"
            entry_price = None
            stop_price = None
            target_weight = 0.0
            reason = f"{reason};blocked_by_fast_risk"

        signals.append(
            {
                "symbol": symbol,
                "action": action,
                "score": round(score, 6),
                "last_price": round(last_price, 3),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_weight": round(float(target_weight), 4),
                "reason": reason,
                "risk_stage": risk_state["stage"],
            }
        )

    out_file = stock_single.get("paths", {}).get("signal_file", "./outputs/orders/stock_single_signals.json")
    ensure_dir(os.path.dirname(out_file) or ".")
    payload = {
        "ts": datetime.now().isoformat(),
        "market": "CN_STOCK_SINGLE",
        "mode": "single_stock_hourly",
        "pool_size": len(symbols),
        "risk_overlay": {
            "source": risk_state["source"],
            "stage": risk_state["stage"],
            "triggered": risk_state["triggered"],
            "block_new_buys": block_new_buys,
            "force_sell_count": len(force_sell_symbols),
        },
        "signal_policy": {
            "mode": str(threshold_meta.get("mode", "threshold")),
            "buy_threshold_effective": float(buy_threshold),
            "sell_threshold_effective": float(sell_threshold),
        },
        "signals": signals,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return signals, out_file


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[stock-single-hourly] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_single.get("enabled", False):
        print("[stock-single] disabled")
        return EXIT_DISABLED

    try:
        signals, out_file = build_signals(runtime, stock_single)
    except OSError as e:
        print(f"[stock-single-hourly] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[stock-single-hourly] failed: {e}")
        return EXIT_CONFIG_ERROR

    print(f"[stock-single-hourly] signals={len(signals)} -> {out_file}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
