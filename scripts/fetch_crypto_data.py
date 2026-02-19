#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import yaml
import requests
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone
from typing import Dict

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
            # Fallback for systems without timezone database (e.g. minimal Raspberry Pi image).
            return (dt + pd.Timedelta(hours=8)).dt.tz_localize(None)
        return dt.dt.tz_localize(None)


def _now_date_in_timezone(timezone_name: str):
    try:
        return pd.Timestamp.now(tz=timezone_name).date()
    except Exception:
        if str(timezone_name).strip() == "Asia/Shanghai":
            return (datetime.now(timezone.utc) + timedelta(hours=8)).date()
        return datetime.now(timezone.utc).date()


def to_htx_symbol(symbol: str) -> str:
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
    params: Dict,
    timeout: int,
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
):
    mode = str(proxy_mode or "auto").strip().lower()
    if mode not in {"auto", "env", "direct"}:
        mode = "auto"

    if mode == "direct":
        sess = requests.Session()
        sess.trust_env = False
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
        sess.trust_env = False
        try:
            return sess.get(url, params=params, timeout=timeout)
        finally:
            sess.close()


def fetch_htx_klines(
    symbol: str,
    timeframe: str = "4h",
    size: int = 2000,
    timeout: int = 10,
    base_url: str = "https://api.huobi.pro/market/history/kline",
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    timezone_name: str = "Asia/Shanghai",
) -> pd.DataFrame:
    period = "4hour" if timeframe == "4h" else "60min"
    params = {"symbol": to_htx_symbol(symbol), "period": period, "size": size}
    r = http_get_with_proxy_policy(
        base_url,
        params=params,
        timeout=timeout,
        proxy_mode=proxy_mode,
        proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "ok":
        raise RuntimeError(f"htx error: {data}")
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = _epoch_to_local_naive(df["id"], unit="s", timezone_name=timezone_name)
    out = df[["date", "open", "high", "low", "close", "amount"]].copy()
    out = out.rename(columns={"amount": "volume"})
    out = out.sort_values("date").reset_index(drop=True)
    return out


def fetch_ccxt_klines(
    ex,
    symbol: str,
    timeframe: str,
    limit: int = 500,
    timezone_name: str = "Asia/Shanghai",
) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["date"] = _epoch_to_local_naive(df["ts"], unit="ms", timezone_name=timezone_name)
    df = df[["date", "open", "high", "low", "close", "volume"]]
    return df


def has_valid_cache(path: str, min_rows: int = 200, max_age_days: int = 5, now_date=None) -> bool:
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
    current_date = now_date if now_date is not None else datetime.now().date()
    age_days = (current_date - dates.iloc[-1].date()).days
    return age_days <= max_age_days


def is_dns_resolution_error(exc: Exception) -> bool:
    if exc is None:
        return False

    msg = _error_chain_message(exc)
    keywords = [
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "failed to resolve",
        "name resolution",
        "getaddrinfo failed",
        "gaierror",
    ]
    return any(k in msg for k in keywords)


def list_ready_symbols(data_dir: str, symbols: list, min_rows: int, max_age_days: int, now_date=None) -> list:
    ready = []
    for s in symbols:
        out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        if has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days, now_date=now_date):
            ready.append(s)
    return ready


def should_soft_fail_on_dns(
    total_fail: int,
    dns_fail: int,
    ready_count: int,
    enabled: bool,
    min_ready_symbols: int,
) -> bool:
    if (not enabled) or total_fail <= 0:
        return False
    if dns_fail != total_fail:
        return False
    return ready_count >= max(1, int(min_ready_symbols))


def load_ccxt_exchange(candidates: list):
    errors = {}
    for name in candidates:
        try:
            ex_class = getattr(ccxt, name)
            tmp = ex_class({"enableRateLimit": True, "timeout": 10000})
            tmp.load_markets()
            return name, tmp, errors
        except Exception as e:
            errors[name] = e
            continue
    return None, None, errors


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

    timezone_name = resolve_timezone_name(runtime.get("timezone", "Asia/Shanghai"))
    now_date = _now_date_in_timezone(timezone_name)
    print(f"[crypto-data] timezone={timezone_name} (csv date stored as local clock)")

    exchange_name = crypto.get("exchange", "okx")
    timeframe = crypto.get("signal", {}).get("timeframe", "4h")
    symbols = crypto.get("symbols", [])

    data_dir = os.path.join(runtime["paths"]["data_dir"], "crypto")
    ensure_dir(data_dir)

    ok = 0
    fail = 0
    dns_fail = 0
    cache_guard = crypto.get("data_guard", {})
    min_rows = int(cache_guard.get("min_cache_rows", 200))
    max_age_days = int(cache_guard.get("max_cache_age_days", 5))
    dns_soft_enabled = bool(cache_guard.get("dns_block_soft_fail_enabled", True))
    default_min_ready = 1 if len(symbols) <= 1 else 2
    dns_min_ready_symbols = int(cache_guard.get("dns_block_min_ready_symbols", default_min_ready))
    network_cfg = crypto.get("network", {}) or {}
    proxy_mode = str(network_cfg.get("http_proxy_mode", "auto")).strip().lower()
    if proxy_mode not in {"auto", "env", "direct"}:
        proxy_mode = "auto"
    proxy_auto_bypass_on_error = bool(network_cfg.get("proxy_auto_bypass_on_error", True))

    # HTX direct mode (bypass ccxt load_markets issue)
    if exchange_name in ["htx", "huobi", "htx_direct"]:
        htx_urls = [
            "https://api.huobi.pro/market/history/kline",
            "https://api-aws.huobi.pro/market/history/kline",
        ]
        _, fallback_ex, _ = load_ccxt_exchange(["okx", "bybit"])
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
                                proxy_mode=proxy_mode,
                                proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
                                timezone_name=timezone_name,
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
                        df = fetch_ccxt_klines(
                            fallback_ex,
                            s,
                            timeframe=timeframe,
                            limit=1000,
                            timezone_name=timezone_name,
                        )
                    except Exception as e:
                        last_err = e

                if df.empty:
                    raise RuntimeError(last_err or "empty data")

                df.to_csv(out, index=False, encoding="utf-8")
                print(f"[crypto-data:htx] {s} -> {out} rows={len(df)}")
                ok += 1
            except Exception as e:
                if has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days, now_date=now_date):
                    print(f"[crypto-data:htx] {s} fetch failed, keep cache -> {out} ({e})")
                    ok += 1
                else:
                    print(f"[crypto-data:htx] {s} failed: {e}")
                    if is_dns_resolution_error(e):
                        dns_fail += 1
                    fail += 1
            time.sleep(0.2)
        print(f"[crypto-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")
        if fail == 0:
            return EXIT_OK
        ready_symbols = list_ready_symbols(
            data_dir,
            symbols,
            min_rows=min_rows,
            max_age_days=max_age_days,
            now_date=now_date,
        )
        if should_soft_fail_on_dns(
            total_fail=fail,
            dns_fail=dns_fail,
            ready_count=len(ready_symbols),
            enabled=dns_soft_enabled,
            min_ready_symbols=dns_min_ready_symbols,
        ):
            print(
                "[crypto-data] DNS blocked some symbols, continue with ready cache/data: "
                f"ready={len(ready_symbols)} required>={dns_min_ready_symbols}"
            )
            return EXIT_OK
        return EXIT_DATA_FETCH_ERROR

    # default ccxt mode
    candidates = [exchange_name, "okx", "bybit"]
    exchange_name, ex, exchange_errors = load_ccxt_exchange(candidates)

    if ex is None:
        missing = []
        for s in symbols:
            out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
            if not has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days, now_date=now_date):
                missing.append(s)
        if missing:
            ready_symbols = list_ready_symbols(
                data_dir,
                symbols,
                min_rows=min_rows,
                max_age_days=max_age_days,
                now_date=now_date,
            )
            all_dns = bool(exchange_errors) and all(
                is_dns_resolution_error(err) for err in exchange_errors.values()
            )
            dns_fail_missing = len(missing) if all_dns else 0
            if should_soft_fail_on_dns(
                total_fail=len(missing),
                dns_fail=dns_fail_missing,
                ready_count=len(ready_symbols),
                enabled=dns_soft_enabled,
                min_ready_symbols=dns_min_ready_symbols,
            ):
                print(
                    "[crypto-data] no exchange endpoint (DNS blocked), continue with ready cache/data: "
                    f"ready={len(ready_symbols)} required>={dns_min_ready_symbols}, missing={missing}"
                )
                return EXIT_OK
            print(f"[crypto-data] no available exchange endpoint and missing cache: {missing}")
            return EXIT_DATA_FETCH_ERROR
        print("[crypto-data] no available exchange endpoint, keep cached data")
        return EXIT_OK

    for s in symbols:
        out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
        try:
            df = fetch_ccxt_klines(ex, s, timeframe=timeframe, limit=500, timezone_name=timezone_name)
            if df.empty:
                raise RuntimeError("empty data")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[crypto-data:{exchange_name}] {s} -> {out} rows={len(df)}")
            ok += 1
        except Exception as e:
            if has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days, now_date=now_date):
                print(f"[crypto-data:{exchange_name}] {s} fetch failed, keep cache -> {out} ({e})")
                ok += 1
            else:
                print(f"[crypto-data:{exchange_name}] {s} failed: {e}")
                if is_dns_resolution_error(e):
                    dns_fail += 1
                fail += 1

    print(f"[crypto-data] done ok={ok} fail={fail} @ {datetime.now().isoformat()}")
    if fail == 0:
        return EXIT_OK
    ready_symbols = list_ready_symbols(
        data_dir,
        symbols,
        min_rows=min_rows,
        max_age_days=max_age_days,
        now_date=now_date,
    )
    if should_soft_fail_on_dns(
        total_fail=fail,
        dns_fail=dns_fail,
        ready_count=len(ready_symbols),
        enabled=dns_soft_enabled,
        min_ready_symbols=dns_min_ready_symbols,
    ):
        print(
            "[crypto-data] DNS blocked some symbols, continue with ready cache/data: "
            f"ready={len(ready_symbols)} required>={dns_min_ready_symbols}"
        )
        return EXIT_OK
    return EXIT_DATA_FETCH_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
