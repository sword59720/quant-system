#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import yaml
import requests
import pandas as pd
from datetime import datetime


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_htx_symbol(symbol: str) -> str:
    # BTC/USDT -> btcusdt
    return symbol.replace("/", "").lower()


def fetch_kline(symbol: str, period: str = "4hour", size: int = 500, timeout: int = 10):
    url = "https://api.huobi.pro/market/history/kline"
    params = {"symbol": to_htx_symbol(symbol), "period": period, "size": size}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"htx error: {data}")
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    # huobi returns latest first
    df = pd.DataFrame(rows)
    # expected cols: id, open, close, low, high, amount, vol, count
    df["date"] = pd.to_datetime(df["id"], unit="s")
    out = df[["date", "open", "high", "low", "close", "amount"]].copy()
    out = out.rename(columns={"amount": "volume"})
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main():
    runtime = load_yaml("config/runtime.yaml")
    crypto = load_yaml("config/crypto.yaml")

    symbols = crypto.get("symbols", ["BTC/USDT", "ETH/USDT"])
    period = "4hour"
    if crypto.get("signal", {}).get("timeframe") == "1h":
        period = "60min"

    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    ensure_dir(data_dir)

    ok = 0
    fail = 0
    for s in symbols:
        try:
            df = fetch_kline(s, period=period, size=500, timeout=10)
            if df.empty:
                raise RuntimeError("empty data")
            out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[htx-data] {s} -> {out} rows={len(df)}")
            ok += 1
        except Exception as e:
            print(f"[htx-data] {s} failed: {e}")
            fail += 1
        time.sleep(0.2)

    print(f"[htx-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
