#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import requests
import pandas as pd
import ccxt
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import (
    EXIT_CONFIG_ERROR,
    EXIT_DATA_FETCH_ERROR,
    EXIT_DISABLED,
    EXIT_OK,
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_htx_symbol(symbol: str) -> str:
    return symbol.replace("/", "").lower()


def fetch_htx_klines(
    symbol: str,
    timeframe: str = "4h",
    size: int = 2000,
    timeout: int = 10,
    base_url: str = "https://api.huobi.pro/market/history/kline",
) -> pd.DataFrame:
    period = "4hour" if timeframe == "4h" else "60min"
    params = {"symbol": to_htx_symbol(symbol), "period": period, "size": size}
    r = requests.get(base_url, params=params, timeout=timeout)
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


def has_valid_cache(path: str, min_rows: int = 200, max_age_days: int = 5) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if ("close" not in df.columns) or (len(df) < min_rows):
        return False
    if "date" not in df.columns:
        return False
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    age_days = (datetime.now().date() - dates.iloc[-1].date()).days
    return age_days <= max_age_days


def load_ccxt_exchange(candidates: list):
    for name in candidates:
        try:
            ex_class = getattr(ccxt, name)
            tmp = ex_class({"enableRateLimit": True, "timeout": 10000})
            tmp.load_markets()
            return name, tmp
        except Exception:
            continue
    return None, None


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        crypto = load_yaml("config/crypto.yaml")
    except Exception as e:
        print(f"[crypto-data] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    exchange_name = crypto.get("exchange", "okx")
    timeframe = crypto.get("signal", {}).get("timeframe", "4h")
    symbols = crypto.get("symbols", [])

    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    ensure_dir(data_dir)

    ok = 0
    fail = 0
    cache_guard = crypto.get("data_guard", {})
    min_rows = int(cache_guard.get("min_cache_rows", 200))
    max_age_days = int(cache_guard.get("max_cache_age_days", 5))

    # HTX direct mode (bypass ccxt load_markets issue)
    if exchange_name in ["htx", "huobi", "htx_direct"]:
        htx_urls = [
            "https://api.huobi.pro/market/history/kline",
            "https://api-aws.huobi.pro/market/history/kline",
        ]
        _, fallback_ex = load_ccxt_exchange(["okx", "bybit"])
        for s in symbols:
            out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
            last_err = None
            try:
                df = pd.DataFrame()
                for base_url in htx_urls:
                    for timeout in [8, 12, 20]:
                        try:
                            df = fetch_htx_klines(
                                s,
                                timeframe=timeframe,
                                size=2000,
                                timeout=timeout,
                                base_url=base_url,
                            )
                            if not df.empty:
                                break
                        except Exception as e:
                            last_err = e
                            continue
                    if not df.empty:
                        break

                if df.empty and fallback_ex is not None:
                    try:
                        df = fetch_ccxt_klines(fallback_ex, s, timeframe=timeframe, limit=1000)
                    except Exception as e:
                        last_err = e

                if df.empty:
                    raise RuntimeError(last_err or "empty data")

                df.to_csv(out, index=False, encoding="utf-8")
                print(f"[crypto-data:htx] {s} -> {out} rows={len(df)}")
                ok += 1
            except Exception as e:
                if has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days):
                    print(f"[crypto-data:htx] {s} fetch failed, keep cache -> {out} ({e})")
                    ok += 1
                else:
                    print(f"[crypto-data:htx] {s} failed: {e}")
                    fail += 1
            time.sleep(0.2)
        print(f"[crypto-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")
        return EXIT_OK if fail == 0 else EXIT_DATA_FETCH_ERROR

    # default ccxt mode
    candidates = [exchange_name, "okx", "bybit"]
    exchange_name, ex = load_ccxt_exchange(candidates)

    if ex is None:
        missing = []
        for s in symbols:
            out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
            if not has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days):
                missing.append(s)
        if missing:
            print(f"[crypto-data] no available exchange endpoint and missing cache: {missing}")
            return EXIT_DATA_FETCH_ERROR
        print("[crypto-data] no available exchange endpoint, keep cached data")
        return EXIT_OK

    for s in symbols:
        out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        try:
            df = fetch_ccxt_klines(ex, s, timeframe=timeframe, limit=500)
            if df.empty:
                raise RuntimeError("empty data")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[crypto-data:{exchange_name}] {s} -> {out} rows={len(df)}")
            ok += 1
        except Exception as e:
            if has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days):
                print(f"[crypto-data:{exchange_name}] {s} fetch failed, keep cache -> {out} ({e})")
                ok += 1
            else:
                print(f"[crypto-data:{exchange_name}] {s} failed: {e}")
                fail += 1

    print(f"[crypto-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")
    return EXIT_OK if fail == 0 else EXIT_DATA_FETCH_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
