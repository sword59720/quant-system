#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import yaml
import requests
import pandas as pd
import ccxt
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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


def to_okx_inst_id(symbol: str) -> str:
    # BTC/USDT -> BTC-USDT
    return symbol.replace("/", "-").upper()


def to_okx_bar(timeframe: str) -> str:
    tf = str(timeframe or "").strip().lower()
    mapping = {"1h": "1H", "4h": "4H", "1d": "1D"}
    return mapping.get(tf, "4H")


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


def fetch_htx_klines(
    symbol: str,
    timeframe: str = "4h",
    size: int = 2000,
    timeout: int = 10,
    base_url: str = "https://api.huobi.pro/market/history/kline",
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    trust_env_enabled: bool = True,
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
        trust_env_enabled=trust_env_enabled,
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


def parse_start_ts_seconds(start_str: str):
    if not start_str:
        return None
    ts = pd.Timestamp(start_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp())


def fetch_htx_klines_page_with_id(
    symbol: str,
    timeframe: str = "4h",
    size: int = 2000,
    timeout: int = 10,
    base_url: str = "https://api.huobi.pro/market/history/kline",
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    trust_env_enabled: bool = True,
    timezone_name: str = "Asia/Shanghai",
    to_ts: int = None,
) -> pd.DataFrame:
    period = "4hour" if timeframe == "4h" else "60min"
    params = {"symbol": to_htx_symbol(symbol), "period": period, "size": int(size)}
    if to_ts is not None:
        params["to"] = int(to_ts)

    r = http_get_with_proxy_policy(
        base_url,
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

    df = pd.DataFrame(rows)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["id"] = df["id"].astype("int64")
    df["date"] = _epoch_to_local_naive(df["id"], unit="s", timezone_name=timezone_name)
    out = df[["id", "date", "open", "high", "low", "close", "amount"]].copy()
    out = out.rename(columns={"amount": "volume"})
    out = out.sort_values("id").drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    return out


def fetch_htx_klines_history(
    symbol: str,
    timeframe: str = "4h",
    start_ts_seconds: int = None,
    max_bars: int = 2000,
    page_size: int = 2000,
    timeout: int = 10,
    base_url: str = "https://api.huobi.pro/market/history/kline",
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    trust_env_enabled: bool = True,
    timezone_name: str = "Asia/Shanghai",
) -> pd.DataFrame:
    if max_bars <= 0:
        return pd.DataFrame()

    all_pages = []
    seen_ids = set()
    prev_oldest = None
    cursor_to = None
    page_size = max(50, min(2000, int(page_size)))

    while len(seen_ids) < max_bars:
        page = fetch_htx_klines_page_with_id(
            symbol=symbol,
            timeframe=timeframe,
            size=page_size,
            timeout=timeout,
            base_url=base_url,
            proxy_mode=proxy_mode,
            proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
            trust_env_enabled=trust_env_enabled,
            timezone_name=timezone_name,
            to_ts=cursor_to,
        )
        if page.empty:
            break

        if seen_ids:
            page = page[~page["id"].isin(seen_ids)].copy()
            if page.empty:
                break

        ids = page["id"].astype("int64").tolist()
        seen_ids.update(ids)
        all_pages.append(page)

        oldest = int(min(ids))
        if (start_ts_seconds is not None) and (oldest <= int(start_ts_seconds)):
            break
        if (prev_oldest is not None) and (oldest >= prev_oldest):
            break
        prev_oldest = oldest
        cursor_to = oldest - 1
        time.sleep(0.1)

    if not all_pages:
        return pd.DataFrame()

    out = (
        pd.concat(all_pages, ignore_index=True)
        .sort_values("id")
        .drop_duplicates(subset=["id"], keep="last")
        .reset_index(drop=True)
    )
    if start_ts_seconds is not None:
        out = out[out["id"] >= int(start_ts_seconds)].copy()
    if len(out) > max_bars:
        out = out.tail(max_bars).copy()
    out = out[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    return out


def fetch_okx_klines_history(
    symbol: str,
    timeframe: str = "4h",
    start_ts_seconds: Optional[int] = None,
    max_bars: int = 20000,
    page_size: int = 100,
    timeout: int = 10,
    base_url: str = "https://www.okx.com/api/v5/market/history-candles",
    proxy_mode: str = "auto",
    proxy_auto_bypass_on_error: bool = True,
    trust_env_enabled: bool = True,
    timezone_name: str = "Asia/Shanghai",
    max_retries: int = 3,
) -> pd.DataFrame:
    if max_bars <= 0:
        return pd.DataFrame()

    all_pages = []
    seen_ts = set()
    prev_oldest = None
    cursor_after = None
    page_size = max(10, min(100, int(page_size)))
    start_ts_ms = int(start_ts_seconds) * 1000 if start_ts_seconds is not None else None

    while len(seen_ts) < max_bars:
        req_limit = min(page_size, max_bars - len(seen_ts))
        params = {
            "instId": to_okx_inst_id(symbol),
            "bar": to_okx_bar(timeframe),
            "limit": str(req_limit),
        }
        if cursor_after is not None:
            params["after"] = str(int(cursor_after))

        rows = None
        last_err = None
        for i in range(max(1, int(max_retries))):
            try:
                r = http_get_with_proxy_policy(
                    base_url,
                    params=params,
                    timeout=timeout,
                    proxy_mode=proxy_mode,
                    proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
                    trust_env_enabled=trust_env_enabled,
                )
                r.raise_for_status()
                data = r.json()
                if str(data.get("code")) != "0":
                    raise RuntimeError(f"okx error: code={data.get('code')} msg={data.get('msg')}")
                rows = data.get("data", [])
                break
            except Exception as e:
                last_err = e
                time.sleep(0.3 * (i + 1))
        if rows is None:
            raise RuntimeError(last_err or "okx history request failed")
        if not rows:
            break

        page = pd.DataFrame(rows)
        page = page.rename(columns={0: "ts", 1: "open", 2: "high", 3: "low", 4: "close", 5: "volume"})
        page["ts"] = pd.to_numeric(page["ts"], errors="coerce")
        page = page.dropna(subset=["ts"]).copy()
        if page.empty:
            break
        page["ts"] = page["ts"].astype("int64")
        for c in ["open", "high", "low", "close", "volume"]:
            page[c] = pd.to_numeric(page[c], errors="coerce")
        page = page.dropna(subset=["open", "high", "low", "close", "volume"]).copy()
        if page.empty:
            break

        if seen_ts:
            page = page[~page["ts"].isin(seen_ts)].copy()
            if page.empty:
                break

        bars = page["ts"].astype("int64").tolist()
        seen_ts.update(bars)
        all_pages.append(page[["ts", "open", "high", "low", "close", "volume"]].copy())

        oldest = int(min(bars))
        if (start_ts_ms is not None) and (oldest <= start_ts_ms):
            break
        if (prev_oldest is not None) and (oldest >= prev_oldest):
            break
        prev_oldest = oldest
        cursor_after = oldest
        time.sleep(0.08)

    if not all_pages:
        return pd.DataFrame()

    out = (
        pd.concat(all_pages, ignore_index=True)
        .sort_values("ts")
        .drop_duplicates(subset=["ts"], keep="last")
        .reset_index(drop=True)
    )
    if start_ts_ms is not None:
        out = out[out["ts"] >= int(start_ts_ms)].copy()
    if len(out) > max_bars:
        out = out.tail(max_bars).copy()
    out["date"] = _epoch_to_local_naive(out["ts"], unit="ms", timezone_name=timezone_name)
    out = out[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
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


def fetch_ccxt_klines_history(
    ex,
    symbol: str,
    timeframe: str,
    start_ts_ms: int,
    max_bars: int = 20000,
    limit: int = 1000,
    timezone_name: str = "Asia/Shanghai",
) -> pd.DataFrame:
    all_rows = []
    seen_ts = set()
    since = int(start_ts_ms)
    limit = max(50, min(1000, int(limit)))
    max_bars = max(200, int(max_bars))
    prev_last_ts = None

    while len(seen_ts) < max_bars:
        batch_limit = min(limit, max_bars - len(seen_ts))
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch_limit)
        if not ohlcv:
            break

        appended = 0
        for row in ohlcv:
            ts = int(row[0])
            if ts not in seen_ts:
                seen_ts.add(ts)
                all_rows.append(row)
                appended += 1

        last_ts = int(ohlcv[-1][0])
        if (prev_last_ts is not None) and (last_ts <= prev_last_ts):
            break
        prev_last_ts = last_ts
        since = last_ts + 1

        if len(ohlcv) < batch_limit:
            break
        time.sleep(max(0.05, float(getattr(ex, "rateLimit", 200)) / 1000.0))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    if len(df) > max_bars:
        df = df.tail(max_bars).copy()
    df["date"] = _epoch_to_local_naive(df["ts"], unit="ms", timezone_name=timezone_name)
    return df[["date", "open", "high", "low", "close", "volume"]]


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
    parser = argparse.ArgumentParser(description="Fetch crypto data from HTX/ccxt")
    parser.add_argument(
        "--history-start",
        type=str,
        default=None,
        help="history start date, e.g. 2020-01-01 (UTC)",
    )
    parser.add_argument(
        "--max-bars",
        type=int,
        default=2000,
        help="max bars per symbol to keep (default 2000)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=2000,
        help="HTX page size per request, max 2000",
    )
    parser.add_argument(
        "--trust-env",
        choices=["true", "false"],
        default=None,
        help="override network.trust_env_enabled (default from config, usually true)",
    )
    args = parser.parse_args()

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
    trust_env_enabled = bool(network_cfg.get("trust_env_enabled", True))
    if args.trust_env is not None:
        trust_env_enabled = (str(args.trust_env).lower() == "true")
    max_bars = max(200, int(args.max_bars))
    page_size = max(50, min(2000, int(args.page_size)))
    history_start_ts = parse_start_ts_seconds(args.history_start)
    if args.history_start:
        print(
            f"[crypto-data] history mode enabled: start={args.history_start} "
            f"max_bars={max_bars} page_size={page_size}"
        )

    # HTX direct mode (bypass ccxt load_markets issue)
    if exchange_name in ["htx", "huobi", "htx_direct"]:
        htx_urls = [
            "https://api.huobi.pro/market/history/kline",
            "https://api-aws.huobi.pro/market/history/kline",
        ]
        okx_history_url = "https://www.okx.com/api/v5/market/history-candles"
        fallback_name, fallback_ex = None, None
        fallback_loaded = False

        def ensure_fallback_exchange():
            nonlocal fallback_name, fallback_ex, fallback_loaded
            if fallback_loaded:
                return fallback_name, fallback_ex
            fallback_loaded = True
            fallback_name, fallback_ex, _ = load_ccxt_exchange(["okx", "bybit"])
            return fallback_name, fallback_ex

        for s in symbols:
            out = os.path.join(data_dir, f"{s.replace('/', '_')}.csv")
            last_err = None
            try:
                df = pd.DataFrame()
                for base_url in htx_urls:
                    for timeout in [8, 12, 20]:
                        try:
                            if history_start_ts is not None:
                                df = fetch_htx_klines_history(
                                    s,
                                    timeframe=timeframe,
                                    start_ts_seconds=history_start_ts,
                                    max_bars=max_bars,
                                    page_size=page_size,
                                    timeout=timeout,
                                    base_url=base_url,
                                    proxy_mode=proxy_mode,
                                    proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
                                    trust_env_enabled=trust_env_enabled,
                                    timezone_name=timezone_name,
                                )
                            else:
                                df = fetch_htx_klines(
                                    s,
                                    timeframe=timeframe,
                                    size=max_bars,
                                    timeout=timeout,
                                    base_url=base_url,
                                    proxy_mode=proxy_mode,
                                    proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
                                    trust_env_enabled=trust_env_enabled,
                                    timezone_name=timezone_name,
                                )
                            if not df.empty:
                                break
                        except Exception as e:
                            last_err = e
                            continue
                    if not df.empty:
                        break

                # In history mode, use OKX direct API fallback for multi-year data
                # when HTX endpoint returns only recent capped bars.
                if history_start_ts is not None:
                    try:
                        okx_df = fetch_okx_klines_history(
                            s,
                            timeframe=timeframe,
                            start_ts_seconds=history_start_ts,
                            max_bars=max_bars,
                            page_size=min(100, page_size),
                            timeout=12,
                            base_url=okx_history_url,
                            proxy_mode=proxy_mode,
                            proxy_auto_bypass_on_error=proxy_auto_bypass_on_error,
                            trust_env_enabled=trust_env_enabled,
                            timezone_name=timezone_name,
                            max_retries=3,
                        )
                        if (not okx_df.empty) and (len(okx_df) > len(df)):
                            df = okx_df
                            print(
                                "[crypto-data:history-fallback:okx_direct] "
                                f"{s} rows={len(df)}"
                            )
                    except Exception as e:
                        last_err = e

                # HTX direct endpoint is typically capped around recent 2000 bars.
                # We already have OKX direct history fallback above.
                # Keep ccxt fallback as the last resort to avoid slow load_markets in normal path.
                if (history_start_ts is not None) and df.empty:
                    fallback_name_tmp, fallback_ex_tmp = ensure_fallback_exchange()
                    if fallback_ex_tmp is not None:
                        try:
                            alt_df = fetch_ccxt_klines_history(
                                fallback_ex_tmp,
                                s,
                                timeframe=timeframe,
                                start_ts_ms=int(history_start_ts) * 1000,
                                max_bars=max_bars,
                                limit=1000,
                                timezone_name=timezone_name,
                            )
                            if (not alt_df.empty) and (len(alt_df) > len(df)):
                                df = alt_df
                                print(
                                    f"[crypto-data:history-fallback:{fallback_name_tmp}] "
                                    f"{s} rows={len(df)}"
                                )
                        except Exception as e:
                            last_err = e

                if df.empty:
                    _, fallback_ex_tmp = ensure_fallback_exchange()
                else:
                    fallback_ex_tmp = None
                if df.empty and fallback_ex_tmp is not None:
                    try:
                        df = fetch_ccxt_klines(
                            fallback_ex_tmp,
                            s,
                            timeframe=timeframe,
                            limit=min(1000, max_bars),
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
            if history_start_ts is not None:
                df = fetch_ccxt_klines_history(
                    ex,
                    s,
                    timeframe=timeframe,
                    start_ts_ms=int(history_start_ts) * 1000,
                    max_bars=max_bars,
                    limit=1000,
                    timezone_name=timezone_name,
                )
            else:
                df = fetch_ccxt_klines(
                    ex,
                    s,
                    timeframe=timeframe,
                    limit=min(500, max_bars),
                    timezone_name=timezone_name,
                )
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
