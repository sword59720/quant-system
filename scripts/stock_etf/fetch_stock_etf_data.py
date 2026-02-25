#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import pandas as pd
from datetime import datetime, timedelta

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


def _normalize_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    keep_cols = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
    if "date" not in keep_cols or "close" not in keep_cols:
        return pd.DataFrame()
    out = df[keep_cols].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["close"]).reset_index(drop=True)
    return out


def fetch_with_akshare_em(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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
    return _normalize_ohlc_df(df)


def fetch_with_akshare_sina(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    # fund_etf_hist_sina 需要带市场前缀的代码
    if symbol.startswith(("0", "1", "2", "3")):
        code = f"sz{symbol}"
    else:
        code = f"sh{symbol}"

    df = ak.fund_etf_hist_sina(symbol=code)
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_ohlc_df(df)
    if df.empty:
        return df

    # 与 EM 路径保持一致，按请求区间裁剪
    s = pd.to_datetime(start_date, format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")
    if pd.notna(s):
        df = df[df["date"] >= s]
    if pd.notna(e):
        df = df[df["date"] <= e]
    return df.reset_index(drop=True)


def fetch_with_baostock(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import baostock as bs

    if symbol.startswith(("0", "1", "2", "3")):
        code = f"sz.{symbol}"
    else:
        code = f"sh.{symbol}"

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"baostock login failed: {lg.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            code,
            "date,open,high,low,close,volume,amount",
            start_date=datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d"),
            end_date=datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag="2",  # 前复权，对齐 akshare qfq
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query failed: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=rs.fields)
        return _normalize_ohlc_df(df)
    finally:
        try:
            bs.logout()
        except Exception:
            pass


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


def disable_proxy_env() -> None:
    # 仅对当前进程生效：临时禁用代理，避免影响 openclaw 主进程
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "UNDICI_PROXY",
    ]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="temporarily disable proxy env vars for this fetch process only",
    )
    args = parser.parse_args()

    if args.no_proxy:
        disable_proxy_env()

    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_cfg = load_yaml("config/stock.yaml")
    except Exception as e:
        print(f"[stock-data] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED

    benchmark_symbol = stock_cfg.get("benchmark_symbol", "510300")
    defensive_symbol = stock_cfg.get("defensive_symbol", "518880")
    symbols = sorted(set((stock_cfg.get("universe", []) or []) + [benchmark_symbol, defensive_symbol]))
    data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
    ensure_dir(data_dir)

    end = datetime.now().date()
    # 保留更长历史，避免覆盖后导致老回测区间（如 2015 起）数据不足
    start = end - timedelta(days=365 * 15)
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
            bs_err = None
            em_err = None
            sina_err = None
            try:
                df = fetch_with_baostock(s, start_date=start_date, end_date=end_date)
                source = "baostock"
            except Exception as e:
                bs_err = e
                df = pd.DataFrame()
                source = "baostock"

            if df.empty:
                try:
                    df = fetch_with_akshare_em(s, start_date=start_date, end_date=end_date)
                    source = "eastmoney"
                except Exception as e:
                    em_err = e
                    df = pd.DataFrame()
                    source = "eastmoney"

            if df.empty:
                try:
                    df = fetch_with_akshare_sina(s, start_date=start_date, end_date=end_date)
                    source = "sina"
                except Exception as e:
                    sina_err = e
                    df = pd.DataFrame()
                    source = "sina"

            if df.empty:
                errs = []
                if bs_err is not None:
                    errs.append(f"baostock={bs_err}")
                if em_err is not None:
                    errs.append(f"eastmoney={em_err}")
                if sina_err is not None:
                    errs.append(f"sina={sina_err}")
                if errs:
                    raise RuntimeError("all sources empty/failed, " + "; ".join(errs))
                raise RuntimeError("all sources empty")

            old_rows = 0
            if os.path.exists(out):
                try:
                    old_df = pd.read_csv(out)
                    old_df = _normalize_ohlc_df(old_df)
                    old_rows = len(old_df)
                    if not old_df.empty:
                        df = pd.concat([old_df, df], ignore_index=True)
                        df = _normalize_ohlc_df(df)
                except Exception:
                    pass

            df.to_csv(out, index=False, encoding="utf-8")
            print(
                f"[stock-data] {s} -> {out} rows={len(df)} source={source}"
                + (f" merged_old={old_rows}" if old_rows else "")
            )
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
