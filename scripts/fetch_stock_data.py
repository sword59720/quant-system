#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import pandas as pd
from datetime import datetime, timedelta


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


def main():
    runtime = load_yaml("config/runtime.yaml")
    stock_cfg = load_yaml("config/stock.yaml")

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return

    symbols = stock_cfg.get("universe", [])
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    ensure_dir(data_dir)

    end = datetime.now().date()
    start = end - timedelta(days=365 * 8)
    start_date = start.strftime("%Y%m%d")
    end_date = end.strftime("%Y%m%d")

    ok = 0
    fail = 0

    for s in symbols:
        try:
            df = fetch_with_akshare(s, start_date=start_date, end_date=end_date)
            if df.empty:
                raise RuntimeError("empty data")
            out = os.path.join(data_dir, f"{s}.csv")
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"[stock-data] {s} -> {out} rows={len(df)}")
            ok += 1
        except Exception as e:
            print(f"[stock-data] {s} failed: {e}")
            fail += 1

    print(f"[stock-data] done ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
