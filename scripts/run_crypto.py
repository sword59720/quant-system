#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import json
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_FORMAT_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
    EXIT_OUTPUT_ERROR,
)
from core.signal import momentum_score, volatility_score, max_drawdown_score, normalize_rank
from core.notify_wecom import send_wecom_message


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_crypto_targets(runtime, crypto, risk):
    if not runtime.get("enabled", True):
        raise RuntimeError("runtime disabled")
    if not crypto.get("enabled", False):
        raise RuntimeError("crypto disabled")

    env = runtime.get("env", "paper")
    total_capital = runtime.get("total_capital", 20000)
    alloc_pct = crypto.get("capital_alloc_pct", 0.3)
    crypto_capital = total_capital * alloc_pct

    lbs = crypto.get("signal", {}).get("momentum_lookback_bars", [30, 90])
    ws = crypto.get("signal", {}).get("momentum_weights", [0.6, 0.4])
    top_n = crypto.get("signal", {}).get("top_n", 1)
    symbols = crypto.get("symbols", [])

    factor_w = crypto.get("signal", {}).get(
        "factor_weights", {"momentum": 0.60, "low_vol": 0.25, "drawdown": 0.15}
    )

    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    raw = {}
    for s in symbols:
        fp = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp)
            if "close" not in df.columns:
                continue
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(close) < max(lbs) + 5:
                continue

            m = momentum_score(close, lbs[0], lbs[1], ws[0], ws[1])
            v = volatility_score(close, lb=20)
            d = max_drawdown_score(close, lb=60)
            sret = float(close.iloc[-1] / close.iloc[-lbs[0]] - 1)
            if None in [m, v, d]:
                continue
            raw[s] = {"momentum": m, "vol": v, "drawdown": d, "short_ret": sret}
        except Exception:
            continue

    if not raw:
        raise RuntimeError("no valid crypto symbol data")

    mom_rank = normalize_rank({k: v["momentum"] for k, v in raw.items()}, ascending=False)
    vol_rank = normalize_rank({k: v["vol"] for k, v in raw.items()}, ascending=True)
    dd_rank = normalize_rank({k: v["drawdown"] for k, v in raw.items()}, ascending=False)

    scored = []
    for s in raw.keys():
        score = (
            factor_w.get("momentum", 0.60) * mom_rank.get(s, 0)
            + factor_w.get("low_vol", 0.25) * vol_rank.get(s, 0)
            + factor_w.get("drawdown", 0.15) * dd_rank.get(s, 0)
        )
        scored.append((s, float(score)))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    picks = [x[0] for x in scored[:top_n]]

    # risk off rule: all short-term returns below threshold => USDT defense
    threshold = float(crypto.get("defense", {}).get("risk_off_threshold_pct", -3.0)) / 100.0
    risk_off = False
    if raw and all(v["short_ret"] < threshold for v in raw.values()):
        risk_off = True

    targets = []
    if not risk_off and picks:
        target_weight_each = min(
            risk["position_limits"]["crypto_single_max_pct"],
            crypto.get("capital_alloc_pct", 0.3) / max(1, len(picks)),
        )
        targets = [{"symbol": s, "target_weight": round(target_weight_each, 4)} for s in picks]
    elif crypto.get("defense", {}).get("use_usdt_defense", True):
        targets = [{"symbol": "USDT", "target_weight": round(alloc_pct, 4)}]

    return {
        "ts": datetime.now().isoformat(),
        "market": "crypto",
        "env": env,
        "capital": crypto_capital,
        "risk_off": risk_off,
        "factor_weights": factor_w,
        "scores": [{"symbol": s, "score": sc} for s, sc in scored],
        "targets": targets,
        "note": "v3 multifactor targets",
    }

def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        crypto = load_yaml("config/crypto.yaml")
        risk = load_yaml("config/risk.yaml")
    except Exception as e:
        print(f"[crypto] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    if not crypto.get("enabled", False):
        print("[crypto] disabled")
        return EXIT_DISABLED

    try:
        out = build_crypto_targets(runtime, crypto, risk)
    except Exception as e:
        print(f"[crypto] data/signal error: {e}")
        return EXIT_DATA_FORMAT_ERROR

    try:
        output_dir = runtime["paths"]["output_dir"]
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "orders"))
        out_file = os.path.join(output_dir, "orders", "crypto_targets.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[crypto] output error: {e}")
        return EXIT_OUTPUT_ERROR

    print(f"[crypto] done -> {out_file}")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 企业微信通知：目标仓位 + 风控状态
    try:
        tgt = ", ".join([f"{x['symbol']}:{x['target_weight']:.2f}" for x in out.get("targets", [])]) or "无"
        lines = [
            f"时间: {out.get('ts','')}",
            f"市场: crypto",
            f"risk_off: {out.get('risk_off', False)}",
            f"目标仓位: {tgt}",
        ]
        ok, msg = send_wecom_message("\n".join(lines), title="目标仓位更新")
        print(f"[notify] wecom {'ok' if ok else 'fail'}: {msg}")
        if out.get("risk_off", False):
            send_wecom_message(
                "币圈风控触发 risk_off，已切换/保持防守仓位。",
                title="风控状态异常触发",
                dedup_key="risk_crypto_trigger",
                dedup_hours=24,
            )
    except Exception as e:
        print(f"[notify] wecom error: {e}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
