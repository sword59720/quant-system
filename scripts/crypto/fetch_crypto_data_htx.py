#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import yaml
import requests
import pandas as pd
from datetime import datetime


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_timezone_name(x) -> str:
    tz = str(x or "Asia/Shanghai").strip()
    if not tz:
        tz = "Asia/Shanghai"
    try:
        _ = pd.Timestamp.now(tz=tz)
        return tz
    except Exception:
        return "Asia/Shanghai"


def _epoch_to_local_naive(values, unit: str, timezone_name: str):
    dt = pd.to_datetime(values, unit=unit, utc=True, errors="coerce")
    try:
        return dt.dt.tz_convert(timezone_name).dt.tz_localize(None)
    except Exception:
        if str(timezone_name).strip() == "Asia/Shanghai":
            return (dt + pd.Timedelta(hours=8)).dt.tz_localize(None)
        return dt.dt.tz_localize(None)


def to_htx_symbol(symbol: str) -> str:
    # BTC/USDT -> btcusdt
    return symbol.replace("/", "").lower()


def _error_chain_message(exc: Exception) -> str:
    if exc is None:
        return ""
    chain = []
    cur = exc
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        chain.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return " | ".join(chain).lower()


def is_proxy_connection_error(exc: Exception) -> bool:
    msg = _error_chain_message(exc)
    keywords = [
        "proxyerror",
        "unable to connect to proxy",
        "cannot connect to proxy",
        "failed to establish a new connection",
        "connection refused",
        "tunnel connection failed",
        "proxy",
    ]
    return any(k in msg for k in keywords)


def http_get_with_proxy_policy(
    url: str,
    params: dict,
    timeout: int,
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    trust_env_enabled: bool = True,
):
    mode = str(proxy_mode or "auto").strip().lower()
    if mode not in {"auto", "env", "direct"}:
        mode = "auto"

    if mode == "direct":
        sess = requests.Session()
        sess.trust_env = bool(trust_env_enabled)
        try:
            return sess.get(url, params=params, timeout=timeout)
        finally:
            sess.close()

    try:
        return requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        if not (mode == "auto" and proxy_auto_bypass_on_error and is_proxy_connection_error(e)):
            raise
        sess = requests.Session()
        sess.trust_env = bool(trust_env_enabled)
        try:
            return sess.get(url, params=params, timeout=timeout)
        finally:
            sess.close()


def fetch_kline(
    symbol: str,
    period: str = "4hour",
    size: int = 500,
    timeout: int = 10,
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    trust_env_enabled: bool = True,
    timezone_name: str = "Asia/Shanghai",
):
    url = "https://api.huobi.pro/market/history/kline"
    params = {"symbol": to_htx_symbol(symbol), "period": period, "size": size}
    r = http_get_with_proxy_policy(
        url,
        params=params,
        timeout=timeout,
        proxy_mode=proxy_mode,
        proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
        trust_env_enabled=trust_env_enabled,
    )
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
    df["date"] = _epoch_to_local_naive(df["id"], unit="s", timezone_name=timezone_name)
    out = df[["date", "open", "high", "low", "close", "amount"]].copy()
    out = out.rename(columns={"amount": "volume"})
    out = out.sort_values("date").reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Fetch HTX crypto data")
    parser.add_argument(
        "--trust-env",
        choices=["true", "false"],
        default=None,
        help="override network.trust_env_enabled (default from config, usually true)",
    )
    args = parser.parse_args()

    runtime = load_yaml("config/runtime.yaml")
    crypto = load_yaml("config/crypto.yaml")
    timezone_name = resolve_timezone_name(runtime.get("timezone", "Asia/Shanghai"))
    print(f"[htx-data] timezone={timezone_name} (csv date stored as local clock)")
    network_cfg = crypto.get("network", {}) or {}
    proxy_mode = str(network_cfg.get("http_proxy_mode", "auto")).strip().lower()
    if proxy_mode not in {"auto", "env", "direct"}:
        proxy_mode = "auto"
    proxy_auto_bypass_on_error = bool(network_cfg.get("proxy_auto_bypass_on_error", True))
    trust_env_enabled = bool(network_cfg.get("trust_env_enabled", True))
    if args.trust_env is not None:
        trust_env_enabled = (str(args.trust_env).lower() == "true")

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
            df = fetch_kline(
                s,
                period=period,
                size=500,
                timeout=10,
                proxy_mode=proxy_mode,
                proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
                trust_env_enabled=trust_env_enabled,
                timezone_name=timezone_name,
            )
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
