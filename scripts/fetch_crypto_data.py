#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import yaml
import requests
import pandas as pd
import ccxt
from datetime import datetime


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_htx_symbol(symbol: str) -> str:
    return symbol.replace("/", "").lower()


def fetch_htx_klines(symbol: str, timeframe: str = "4h", size: int = 2000, timeout: int = 10) -> pd.DataFrame:
    period = "4hour" if timeframe == "4h" else "60min"
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

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["id"], unit="s")
    out = df[["date", "open", "high", "low", "close", "amount"]].copy()
    out = out.rename(columns={"amount": "volume"})
    out = out.sort_values("date").reset_index(drop=True)
    return out


def fetch_ccxt_klines(ex, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df


def main():
    runtime = load_yaml("config/runtime.yaml")
    crypto = load_yaml("config/crypto.yaml")

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return

    exchange_name = crypto.get("exchange", "okx")
    timeframe = crypto.get("signal", {}).get("timeframe", "4h")
    symbols = crypto.get("symbols", [])

    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    ensure_dir(data_dir)

    ok = 0
    fail = 0

    # HTX direct mode (bypass ccxt load_markets issue)
    if exchange_name in ["htx", "huobi", "htx_direct"]:
        for s in symbols:
            try:
                df = fetch_htx_klines(s, timeframe=timeframe, size=2000, timeout=10)
                if df.empty:
                    raise RuntimeError("empty data")
                out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
                df.to_csv(out, index=False, encoding="utf-8")
                print(f"[crypto-data:htx] {s} -> {out} rows={len(df)}")
                ok += 1
            except Exception as e:
                print(f"[crypto-data:htx] {s} failed: {e}")
                fail += 1
            time.sleep(0.2)
        print(f"[crypto-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")
        return

    # default ccxt mode
    candidates = [exchange_name, "okx", "bybit"]
    ex = None
    for name in candidates:
        try:
            ex_class = getattr(ccxt, name)
            tmp = ex_class({"enableRateLimit": True, "timeout": 10000})
            tmp.load_markets()
            ex = tmp
            exchange_name = name
            break
        except Exception:
            continue

    if ex is None:
        print("[crypto-data] no available crypto exchange endpoint, skip fetch and keep cached data")
        return

    for s in symbols:
        try:
            df = fetch_ccxt_klines(ex, s, timeframe=timeframe, limit=500)
            if df.empty:
                raise RuntimeError("empty data")
            out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[crypto-data:{exchange_name}] {s} -> {out} rows={len(df)}")
            ok += 1
        except Exception as e:
            print(f"[crypto-data:{exchange_name}] {s} failed: {e}")
            fail += 1

    print(f"[crypto-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
