#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import pandas as pd
from datetime import datetime, timedelta

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


def fetch_with_akshare(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.fund_etf_hist_em(
        symbol=symbol,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # 标准化字段
    col_map = {
        "日期": "date",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    df = df.rename(columns=col_map)
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    df = df[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"]) 
    df = df.sort_values("date").reset_index(drop=True)
    return df


def has_valid_cache(path: str, min_rows: int = 200, max_age_days: int = 15) -> bool:
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


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
    except Exception as e:
        print(f"[stock-data] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    symbols = stock_cfg.get("universe", [])
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    ensure_dir(data_dir)

    end = datetime.now().date()
    start = end - timedelta(days=365 * 8)
    start_date = start.strftime("%Y%m%d")
    end_date = end.strftime("%Y%m%d")

    ok = 0
    fail = 0
    cache_guard = stock_cfg.get("data_guard", {})
    min_rows = int(cache_guard.get("min_cache_rows", 200))
    max_age_days = int(cache_guard.get("max_cache_age_days", 15))

    for s in symbols:
        out = os.path.join(data_dir, f"{s}.csv")
        try:
            df = fetch_with_akshare(s, start_date=start_date, end_date=end_date)
            if df.empty:
                raise RuntimeError("empty data")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[stock-data] {s} -> {out} rows={len(df)}")
            ok += 1
        except Exception as e:
            if has_valid_cache(out, min_rows=min_rows, max_age_days=max_age_days):
                print(f"[stock-data] {s} fetch failed, keep cache -> {out} ({e})")
                ok += 1
            else:
                print(f"[stock-data] {s} failed: {e}")
                fail += 1

    print(f"[stock-data] done ok={ok} fail={fail}")
    return EXIT_OK if fail == 0 else EXIT_DATA_FETCH_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
