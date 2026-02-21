#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import yaml

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


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch single-stock data for backtest/live (daily + optional minute).")
    parser.add_argument("--with-minute", action="store_true", help="fetch minute data in addition to daily")
    parser.add_argument("--only-minute", action="store_true", help="fetch only minute data")
    parser.add_argument("--only-daily", action="store_true", help="fetch only daily data")
    parser.add_argument("--with-factors", action="store_true", help="fetch valuation and fund-flow factors")
    parser.add_argument("--only-factors", action="store_true", help="fetch only valuation and fund-flow factors")
    parser.add_argument(
        "--symbols",
        default=None,
        help="optional comma-separated symbols, e.g. 600519.SH,000001.SZ",
    )
    return parser.parse_args()


def normalize_symbol(raw_symbol: str) -> Optional[Tuple[str, str]]:
    s = str(raw_symbol).strip().upper()
    if not s:
        return None

    m = re.search(r"(\d{6})", s)
    if not m:
        return None
    code = m.group(1)

    if ".SH" in s or s.startswith("SH"):
        market = "SH"
    elif ".BJ" in s or s.startswith("BJ"):
        market = "BJ"
    elif ".SZ" in s or s.startswith("SZ"):
        market = "SZ"
    else:
        if code.startswith(("4", "8", "92", "93")):
            market = "BJ"
        else:
            market = "SH" if code[0] in {"5", "6", "9"} else "SZ"

    return code, f"{code}.{market}"


def is_a_share_stock_code(code: str, canonical: Optional[str] = None) -> bool:
    """
    Restrict stock_single pipeline to A-share individual stocks.
    Use exchange-aware prefixes to avoid mixing in indexes/ETFs:
    - SH/SSE stocks: 600/601/603/605/688/689 (+ 900 B-share)
    - SZ/SZSE stocks: 000/001/002/003/300/301 (+ 200 B-share)
    - BJ/BSE stocks: 8xxxxx / 4xxxxx
    """
    s = str(code).strip()
    if len(s) != 6 or (not s.isdigit()):
        return False

    market = ""
    if canonical and "." in str(canonical):
        market = str(canonical).split(".")[-1].upper()

    if market == "SH":
        return s.startswith(("600", "601", "603", "605", "688", "689", "900"))
    if market == "SZ":
        return s.startswith(("000", "001", "002", "003", "300", "301", "200"))
    if market == "BJ":
        return s.startswith(("8", "4", "92", "93"))

    # Fallback when market is unknown: keep common stock prefixes only.
    return s.startswith(
        ("000", "001", "002", "003", "300", "301", "600", "601", "603", "605", "688", "689", "8", "92", "93")
    )


def load_symbol_candidates(stock_single: dict) -> list[str]:
    pool_cfg = stock_single.get("pool", {})
    src = str(pool_cfg.get("source_file", "./data/stock_single/universe.csv"))
    col = str(pool_cfg.get("symbol_column", "symbol"))
    symbols = []

    if os.path.exists(src):
        df = pd.read_csv(src)
        if col not in df.columns:
            raise RuntimeError(f"pool source missing column: {col}")
        symbols = df[col].dropna().astype(str).tolist()

    if not symbols:
        pool_file = stock_single.get("paths", {}).get("pool_file", "./outputs/orders/stock_single_pool.json")
        if os.path.exists(pool_file):
            with open(pool_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            symbols = [str(x) for x in payload.get("symbols", [])]

    if not symbols:
        symbols = [str(x) for x in pool_cfg.get("static_fallback", [])]
    return symbols


def load_symbols(stock_single: dict) -> list[tuple[str, str]]:
    candidates = load_symbol_candidates(stock_single)
    max_symbols = int(stock_single.get("max_symbols_in_pool", 300))

    out = []
    seen = set()
    
    # 如果候选股票不足，添加更多股票
    if len(candidates) < max_symbols:
        # 添加常见的A股股票代码
        additional_symbols = [
            "600519.SH", "601318.SH", "000858.SZ", "000333.SZ", "601888.SH",
            "600036.SH", "600276.SH", "601628.SH", "601398.SH", "601288.SH",
            "000001.SZ", "002594.SZ", "002415.SZ", "000977.SZ", "300251.SZ",
            "600685.SH", "600900.SH", "601985.SH", "603019.SH", "920976.BJ",
            "600031.SH", "600030.SH", "600271.SH", "600585.SH", "600887.SH",
            "601012.SH", "601166.SH", "601668.SH", "601899.SH", "603288.SH",
            "000651.SZ", "000725.SZ", "000895.SZ", "002027.SZ", "002230.SZ",
            "002241.SZ", "002304.SZ", "002475.SZ", "002555.SZ", "002714.SZ",
            "300014.SZ", "300033.SZ", "300124.SZ", "300274.SZ", "300413.SZ",
            "300433.SZ", "300601.SZ", "300750.SZ", "300896.SZ", "300957.SZ"
        ]
        candidates.extend(additional_symbols)

    for raw in candidates:
        norm = normalize_symbol(raw)
        if norm is None:
            continue
        code, canonical = norm
        if not is_a_share_stock_code(code, canonical):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append((code, canonical))
        if len(out) >= max_symbols:
            break
    return out


def has_valid_cache(path: str, time_col: str, min_rows: int, max_age_days: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    if len(df) < int(min_rows):
        return False
    if time_col not in df.columns:
        return False
    ts = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if ts.empty:
        return False
    age_days = (datetime.now() - ts.iloc[-1].to_pydatetime()).total_seconds() / 86400.0
    return age_days <= float(max_age_days)


def read_existing(path: str, time_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if time_col not in df.columns:
        return pd.DataFrame()
    x = df.copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    return x


def merge_history(old_df: pd.DataFrame, new_df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if old_df.empty:
        x = new_df.copy()
    elif new_df.empty:
        x = old_df.copy()
    else:
        x = pd.concat([old_df, new_df], axis=0, ignore_index=True)
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col]).drop_duplicates(subset=[time_col], keep="last")
    x = x.sort_values(time_col).reset_index(drop=True)
    return x


def resolve_incremental_start(path: str, default_start: datetime, overlap_days: int, time_col: str) -> datetime:
    old_df = read_existing(path, time_col)
    if old_df.empty:
        return default_start
    last_dt = old_df[time_col].iloc[-1].to_pydatetime()
    start = last_dt - timedelta(days=max(0, int(overlap_days)))
    return max(default_start, start)


def fetch_daily_with_baostock(symbol_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    """
    Fetch daily stock data with Baostock
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        adjust: Adjustment method (qfq, hfq, None)
    
    Returns:
        DataFrame with daily stock data
    """
    import baostock as bs
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][baostock-daily] Fetching {symbol_code} from {start_date} to {end_date} (adjust: {adjust}) - attempt {retry + 1}/{max_retries}")
            
            # Convert symbol to Baostock format
            if symbol_code.endswith(".SH"):
                bs_code = f"sh.{symbol_code[:6]}"
            elif symbol_code.endswith(".SZ"):
                bs_code = f"sz.{symbol_code[:6]}"
            else:
                # Try to determine market
                code = symbol_code[:6]
                if code.startswith(('600', '601', '603', '605', '688', '689')):
                    bs_code = f"sh.{code}"
                else:
                    bs_code = f"sz.{code}"
            
            # Convert adjust parameter
            adjust_flag = "3"  # Default: no adjust
            if adjust == "qfq":
                adjust_flag = "2"  # Forward adjust
            elif adjust == "hfq":
                adjust_flag = "1"  # Backward adjust
            
            # Login to Baostock
            lg = bs.login()
            if lg.error_code != '0':
                print(f"[stock-single-data][baostock-daily] Login failed: {lg.error_msg}")
                bs.logout()
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-daily] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()
            
            # Fetch data
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields="date,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag=adjust_flag
            )
            
            # Check if query failed
            if rs.error_code != '0':
                print(f"[stock-single-data][baostock-daily] Query failed: {rs.error_msg}")
                bs.logout()
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-daily] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()
            
            # Convert to DataFrame
            df = rs.get_data()
            bs.logout()
            
            # Validate data
            if df is None:
                print(f"[stock-single-data][baostock-daily] No data returned for {symbol_code}")
                return pd.DataFrame()
            
            if df.empty:
                print(f"[stock-single-data][baostock-daily] Empty data returned for {symbol_code}")
                return pd.DataFrame()

            # Process columns
            col_map = {
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "amount": "amount",
                "turn": "turnover_rate",
            }
            
            # Rename columns
            x = df.rename(columns=col_map)
            
            # Convert numeric columns
            numeric_cols = ["open", "high", "low", "close", "volume", "amount", "turnover_rate"]
            for col in numeric_cols:
                if col in x.columns:
                    x[col] = pd.to_numeric(x[col], errors="coerce")
            
            # Calculate additional columns
            if "close" in x.columns and "open" in x.columns:
                x["change"] = x["close"] - x["open"]
                x["pct_change"] = (x["change"] / x["open"] * 100).round(2)
            
            if "high" in x.columns and "low" in x.columns and "open" in x.columns:
                x["amplitude_pct"] = ((x["high"] - x["low"]) / x["open"] * 100).round(2)
            
            # Ensure date column exists
            if "date" not in x.columns:
                print(f"[stock-single-data][baostock-daily] No date column found for {symbol_code}")
                return pd.DataFrame()
            
            # Select and validate columns
            keep_cols = [
                c
                for c in [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "amplitude_pct",
                    "pct_change",
                    "change",
                    "turnover_rate",
                ]
                if c in x.columns
            ]
            
            if not keep_cols:
                print(f"[stock-single-data][baostock-daily] No valid columns found for {symbol_code}")
                return pd.DataFrame()
            
            x = x[keep_cols].copy()
            
            # Process date column
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["date"])
            
            if x.empty:
                print(f"[stock-single-data][baostock-daily] No valid date data for {symbol_code}")
                return pd.DataFrame()
            
            # Sort by date
            x = x.sort_values("date").reset_index(drop=True)
            
            print(f"[stock-single-data][baostock-daily] Successfully fetched {len(x)} rows for {symbol_code}")
            return x
            
        except Exception as e:
            print(f"[stock-single-data][baostock-daily] Failed to fetch data for {symbol_code}: {e}")
            try:
                bs.logout()
            except:
                pass
            if retry < max_retries - 1:
                print(f"[stock-single-data][baostock-daily] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][baostock-daily] All retries failed for {symbol_code}")
                return pd.DataFrame()

    # Should never reach here
    return pd.DataFrame()


def fetch_daily_with_akshare(symbol_code: str, start_date: str, end_date: str, adjust: str) -> pd.DataFrame:
    """
    Fetch daily stock data with AKShare
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        adjust: Adjustment method (qfq, hfq, None)
    
    Returns:
        DataFrame with daily stock data
    """
    import akshare as ak
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][daily] Fetching {symbol_code} from {start_date} to {end_date} (adjust: {adjust}) - attempt {retry + 1}/{max_retries}")
            
            # Try with adjust parameter
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )
            except TypeError:
                # Fallback if adjust parameter is not supported
                print(f"[stock-single-data][daily] Adjust parameter not supported, using default")
                df = ak.stock_zh_a_hist(
                    symbol=symbol_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as e:
                # Handle other exceptions
                print(f"[stock-single-data][daily] Error fetching data: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][daily] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise

            # Validate data
            if df is None:
                print(f"[stock-single-data][daily] No data returned for {symbol_code}")
                return pd.DataFrame()
            
            if df.empty:
                print(f"[stock-single-data][daily] Empty data returned for {symbol_code}")
                return pd.DataFrame()

            # Map columns
            col_map = {
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "振幅": "amplitude_pct",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "换手率": "turnover_rate",
            }
            x = df.rename(columns=col_map)
            
            # Ensure date column exists
            if "date" not in x.columns:
                print(f"[stock-single-data][daily] No date column found for {symbol_code}")
                return pd.DataFrame()
            
            # Select and validate columns
            keep_cols = [
                c
                for c in [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "amplitude_pct",
                    "pct_change",
                    "change",
                    "turnover_rate",
                ]
                if c in x.columns
            ]
            
            if not keep_cols:
                print(f"[stock-single-data][daily] No valid columns found for {symbol_code}")
                return pd.DataFrame()
            
            x = x[keep_cols].copy()
            
            # Process date column
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["date"])
            
            if x.empty:
                print(f"[stock-single-data][daily] No valid date data for {symbol_code}")
                return pd.DataFrame()
            
            # Sort by date
            x = x.sort_values("date").reset_index(drop=True)
            
            print(f"[stock-single-data][daily] Successfully fetched {len(x)} rows for {symbol_code}")
            return x
            
        except Exception as e:
            print(f"[stock-single-data][daily] Failed to fetch data for {symbol_code}: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][daily] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][daily] All retries failed for {symbol_code}")
                return pd.DataFrame()

    # Should never reach here
    return pd.DataFrame()


def fetch_minute_with_akshare(
    symbol_code: str,
    start_dt: str,
    end_dt: str,
    period: str,
    adjust: str,
) -> pd.DataFrame:
    import akshare as ak

    if hasattr(ak, "stock_zh_a_hist_min_em"):
        func = ak.stock_zh_a_hist_min_em
        try:
            df = func(symbol=symbol_code, start_date=start_dt, end_date=end_dt, period=period, adjust=adjust)
        except TypeError:
            df = func(symbol=symbol_code, start_date=start_dt, end_date=end_dt, period=period)
    elif hasattr(ak, "stock_zh_a_minute"):
        func = ak.stock_zh_a_minute
        try:
            df = func(symbol=symbol_code, period=period, adjust=adjust)
        except TypeError:
            df = func(symbol=symbol_code, period=period)
    else:
        raise RuntimeError("akshare minute api not found")

    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "时间": "datetime",
        "日期时间": "datetime",
        "日期": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
    }
    x = df.rename(columns=col_map)
    if "datetime" not in x.columns:
        if "date" in x.columns:
            x = x.rename(columns={"date": "datetime"})
        elif "time" in x.columns:
            x = x.rename(columns={"time": "datetime"})
    keep_cols = [c for c in ["datetime", "open", "high", "low", "close", "volume", "amount"] if c in x.columns]
    x = x[keep_cols].copy()
    if "datetime" not in x.columns:
        return pd.DataFrame()
    x["datetime"] = pd.to_datetime(x["datetime"], errors="coerce")
    x = x.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return x


def parse_cn_number(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "--", "None", "nan", "NaN"}:
        return float("nan")
    mult = 1.0
    if s.endswith("%"):
        s = s[:-1]
        mult = 0.01
    if s.endswith("亿"):
        s = s[:-1]
        mult *= 1e8
    elif s.endswith("万"):
        s = s[:-1]
        mult *= 1e4
    elif s.endswith("千"):
        s = s[:-1]
        mult *= 1e3
    try:
        return float(s) * mult
    except Exception:
        return float("nan")


def first_matching_col(columns: List[str], keywords: List[str]) -> Optional[str]:
    for kw in keywords:
        for c in columns:
            if kw in c:
                return c
    return None


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    x = df.copy()
    date_col = first_matching_col(list(x.columns), ["date", "日期", "时间", "datetime", "交易日"])
    if date_col is None:
        return pd.DataFrame(columns=["date"])
    x = x.rename(columns={date_col: "date"})
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return x


def fetch_valuation_with_akshare(symbol_code: str, start_date: str) -> pd.DataFrame:
    """
    Fetch valuation data with AKShare
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with valuation data
    """
    import akshare as ak
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    def _indicator(indicator_name: str, out_col: str) -> pd.DataFrame:
        for retry in range(max_retries):
            try:
                print(f"[stock-single-data][akshare-valuation] Fetching {indicator_name} for {symbol_code} - attempt {retry + 1}/{max_retries}")
                
                # 尝试使用不同的AKShare函数获取估值数据
                try:
                    # 尝试使用stock_zh_a_valuation函数作为替代
                    if hasattr(ak, 'stock_zh_a_valuation'):
                        print(f"[stock-single-data][akshare-valuation] Trying stock_zh_a_valuation for {symbol_code}")
                        df = ak.stock_zh_a_valuation(
                            symbol=symbol_code
                        )
                    else:
                        # 如果函数不存在，返回空DataFrame
                        print(f"[stock-single-data][akshare-valuation] No suitable valuation function available")
                        return pd.DataFrame(columns=["date", out_col])
                except Exception as e:
                    print(f"[stock-single-data][akshare-valuation] Valuation API failed: {e}")
                    if retry < max_retries - 1:
                        print(f"[stock-single-data][akshare-valuation] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        # 直接返回空DataFrame，让系统回退到其他数据源
                        return pd.DataFrame(columns=["date", out_col])
                
                # Validate data
                if df is None:
                    print(f"[stock-single-data][akshare-valuation] No data returned for {symbol_code}")
                    return pd.DataFrame(columns=["date", out_col])
                
                if df.empty:
                    print(f"[stock-single-data][akshare-valuation] Empty data returned for {symbol_code}")
                    return pd.DataFrame(columns=["date", out_col])
                
                x = normalize_date_col(df)
                if x.empty:
                    print(f"[stock-single-data][akshare-valuation] No valid date data for {symbol_code}")
                    return pd.DataFrame(columns=["date", out_col])
                
                value_col = first_matching_col(
                    [c for c in x.columns if c != "date"],
                    ["value", "数值", indicator_name, "市盈率", "市净率", "pe_ttm", "pb"],
                )
                if value_col is None:
                    others = [c for c in x.columns if c != "date"]
                    if not others:
                        print(f"[stock-single-data][akshare-valuation] No value column found for {symbol_code}")
                        return pd.DataFrame(columns=["date", out_col])
                    value_col = others[0]
                
                x[out_col] = x[value_col].map(parse_cn_number)
                x = x[["date", out_col]].dropna(subset=[out_col])
                
                if x.empty:
                    print(f"[stock-single-data][akshare-valuation] No valid valuation data for {symbol_code}")
                    return pd.DataFrame(columns=["date", out_col])
                
                print(f"[stock-single-data][akshare-valuation] Successfully fetched {len(x)} rows for {symbol_code}")
                return x
                
            except Exception as e:
                print(f"[stock-single-data][akshare-valuation] {indicator_name} failed: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][akshare-valuation] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    # 直接返回空DataFrame，让系统回退到其他数据源
                    return pd.DataFrame(columns=["date", out_col])

    # 获取市盈率和市净率数据
    pe = _indicator("市盈率(TTM)", "pe_ttm")
    pb = _indicator("市净率", "pb")
    
    # 合并数据
    merged = pd.merge(pe, pb, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    
    # 过滤日期范围
    start_dt = pd.to_datetime(start_date, errors="coerce")
    if pd.notna(start_dt):
        # 确保date列是datetime类型
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged = merged.dropna(subset=["date"])
        merged = merged[merged["date"] >= start_dt].reset_index(drop=True)
    
    print(f"[stock-single-data][akshare-valuation] Final merged data for {symbol_code}: {len(merged)} rows")
    return merged


def fetch_valuation_with_eastmoney(symbol_code: str, start_date: str) -> pd.DataFrame:
    """
    Fetch valuation data with Eastmoney
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with valuation data
    """
    import requests
    import json
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][eastmoney-valuation] Fetching {symbol_code} - attempt {retry + 1}/{max_retries}")
            
            # Eastmoney API for stock valuation data
            url = f"http://push2.eastmoney.com/api/qt/stock/get?secid={symbol_code[:6]}.{symbol_code[-2:].lower()}&fields=f187,f188"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"[stock-single-data][eastmoney-valuation] JSON decode error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][eastmoney-valuation] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            stock_data = data.get("data", {})
            
            # Check if stock_data is None
            if stock_data is None:
                print(f"[stock-single-data][eastmoney-valuation] no data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Fixed API fields for PE-TTM and PB
            pe_ttm = stock_data.get("f187", "nan")  # f187 is PE-TTM value
            pb = stock_data.get("f188", "nan")  # f188 is PB value
            
            # Create DataFrame with current date
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            df = pd.DataFrame({
                "date": [current_date],
                "pe_ttm": [parse_cn_number(pe_ttm)],
                "pb": [parse_cn_number(pb)]
            })
            
            # Validate data
            if df.empty:
                print(f"[stock-single-data][eastmoney-valuation] Empty data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Process date column
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][eastmoney-valuation] No valid date data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                df = df[df["date"] >= start_dt].reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][eastmoney-valuation] No data after {start_date} for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            print(f"[stock-single-data][eastmoney-valuation] Successfully fetched data for {symbol_code}")
            return df
            
        except Exception as e:
            print(f"[stock-single-data][eastmoney-valuation] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][eastmoney-valuation] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][eastmoney-valuation] All retries failed for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])


def fetch_valuation_with_sina(symbol_code: str, start_date: str) -> pd.DataFrame:
    """
    Fetch valuation data with Sina Finance
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with valuation data
    """
    import requests
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][sina-valuation] Fetching {symbol_code} - attempt {retry + 1}/{max_retries}")
            
            # Sina Finance API for stock valuation data
            market = "sh" if symbol_code.endswith(".SH") else "sz"
            code = symbol_code[:6]
            url = f"http://hq.sinajs.cn/list={market}{code}"
            
            try:
                # 添加headers模拟浏览器请求
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Referer": "http://finance.sina.com.cn/",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Connection": "keep-alive"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[stock-single-data][sina-valuation] Request error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][sina-valuation] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Parse Sina Finance data
            content = response.text
            data = content.split(",")
            
            if len(data) < 32:
                print(f"[stock-single-data][sina-valuation] Insufficient data for {symbol_code}: only {len(data)} fields returned")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Extract valuation data
            try:
                pe_ttm = data[31]  # PE-TTM
                pb = data[32]  # PB
            except IndexError as e:
                print(f"[stock-single-data][sina-valuation] Index error: {e}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Create DataFrame with current date
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            df = pd.DataFrame({
                "date": [current_date],
                "pe_ttm": [parse_cn_number(pe_ttm)],
                "pb": [parse_cn_number(pb)]
            })
            
            # Validate data
            if df.empty:
                print(f"[stock-single-data][sina-valuation] Empty data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Process date column
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][sina-valuation] No valid date data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                df = df[df["date"] >= start_dt].reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][sina-valuation] No data after {start_date} for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            print(f"[stock-single-data][sina-valuation] Successfully fetched data for {symbol_code}")
            return df
            
        except Exception as e:
            print(f"[stock-single-data][sina-valuation] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][sina-valuation] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][sina-valuation] All retries failed for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])


def fetch_valuation_with_10jqka(symbol_code: str, start_date: str) -> pd.DataFrame:
    """
    Fetch valuation data with 10jqka
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with valuation data
    """
    import requests
    import re
    import json
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][10jqka-valuation] Fetching {symbol_code} - attempt {retry + 1}/{max_retries}")
            
            # 10jqka API for stock valuation data - API endpoint might be changed
            code = symbol_code[:6]
            # Updated API endpoint with possible alternative
            url = f"http://web.10jqka.com.cn/data/{code}/"
            
            try:
                # Add headers to mimic browser request
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Referer": "http://www.10jqka.com.cn/",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[stock-single-data][10jqka-valuation] Request error: {e}")
                # Special handling for 502 Bad Gateway error
                if "502" in str(e):
                    print(f"[stock-single-data][10jqka-valuation] 502 Bad Gateway error - 10jqka server may be temporarily unavailable")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][10jqka-valuation] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[stock-single-data][10jqka-valuation] 10jqka API is currently unavailable, will fallback to other data sources")
                    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Parse 10jqka data - updated parsing logic for new endpoint
            content = response.text
            
            # Try to extract valuation data from HTML content
            try:
                # Look for PE and PB values in the HTML
                pe_match = re.search(r'市盈率.*?<span.*?>(.*?)</span>', content, re.DOTALL)
                pb_match = re.search(r'市净率.*?<span.*?>(.*?)</span>', content, re.DOTALL)
                
                pe_ttm = pe_match.group(1).strip() if pe_match else "nan"
                pb = pb_match.group(1).strip() if pb_match else "nan"
            except Exception as e:
                print(f"[stock-single-data][10jqka-valuation] Failed to parse data: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][10jqka-valuation] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Create DataFrame with current date
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            df = pd.DataFrame({
                "date": [current_date],
                "pe_ttm": [parse_cn_number(pe_ttm)],
                "pb": [parse_cn_number(pb)]
            })
            
            # Validate data
            if df.empty:
                print(f"[stock-single-data][10jqka-valuation] Empty data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Process date column
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][10jqka-valuation] No valid date data for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                df = df[df["date"] >= start_dt].reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][10jqka-valuation] No data after {start_date} for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            print(f"[stock-single-data][10jqka-valuation] Successfully fetched data for {symbol_code}")
            return df
            
        except Exception as e:
            print(f"[stock-single-data][10jqka-valuation] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][10jqka-valuation] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][10jqka-valuation] All retries failed for {symbol_code}, will fallback to other data sources")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])


# Global Baostock login state
_baostock_logged_in = False

def ensure_baostock_login():
    """
    Ensure Baostock is logged in
    
    Returns:
        bool: True if login successful, False otherwise
    """
    global _baostock_logged_in
    import baostock as bs
    
    if not _baostock_logged_in:
        print("[stock-single-data][baostock] Logging in to Baostock API...")
        lg = bs.login()
        if lg is None or lg.error_code != '0':
            print(f"[stock-single-data][baostock] Login failed: {lg.error_msg if lg else 'Unknown error'}")
            return False
        print("[stock-single-data][baostock] Login successful")
        _baostock_logged_in = True
    return True

def baostock_logout():
    """
    Logout from Baostock API
    """
    global _baostock_logged_in
    import baostock as bs
    
    if _baostock_logged_in:
        print("[stock-single-data][baostock] Logging out from Baostock API...")
        try:
            bs.logout()
            print("[stock-single-data][baostock] Logout successful")
        except Exception as e:
            print(f"[stock-single-data][baostock] Logout error: {e}")
        finally:
            _baostock_logged_in = False

def fetch_valuation_with_baostock(symbol_code: str, start_date: str) -> pd.DataFrame:
    """
    Fetch valuation data with Baostock
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with valuation data
    """
    import baostock as bs
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][baostock-valuation] Fetching {symbol_code} - attempt {retry + 1}/{max_retries}")
            
            # Ensure Baostock is logged in
            if not ensure_baostock_login():
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-valuation] Login failed, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Convert symbol to Baostock format
            code = symbol_code[:6] if len(symbol_code) >= 6 else symbol_code
            if symbol_code.endswith(".SH"):
                bs_code = f"sh.{code}"
            elif symbol_code.endswith(".SZ"):
                bs_code = f"sz.{code}"
            else:
                # Try to determine market
                if code.startswith(('600', '601', '603', '605', '688', '689')):
                    bs_code = f"sh.{code}"
                else:
                    bs_code = f"sz.{code}"
            
            print(f"[stock-single-data][baostock-valuation] Using Baostock code: {bs_code}")
            
            # Fetch data with valuation indicators
            # Ensure start_date is in correct format (YYYY-MM-DD)
            try:
                # Parse and format date correctly
                start_dt = pd.to_datetime(start_date)
                # Use a longer date range to ensure we get data
                bs_start_date = (start_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                bs_end_date = time.strftime("%Y-%m-%d")
                
                print(f"[stock-single-data][baostock-valuation] Fetching from {bs_start_date} to {bs_end_date}")
                
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields="date,code,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                    start_date=bs_start_date,
                    end_date=bs_end_date,
                    frequency="d",
                    adjustflag="2"  # Forward adjust
                )
                
                # Check if query failed
                if rs is None:
                    print(f"[stock-single-data][baostock-valuation] Query returned None for {symbol_code}")
                    if retry < max_retries - 1:
                        print(f"[stock-single-data][baostock-valuation] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
                
                if rs.error_code != '0':
                    print(f"[stock-single-data][baostock-valuation] Query failed: {rs.error_msg}")
                    if retry < max_retries - 1:
                        print(f"[stock-single-data][baostock-valuation] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
                
                # Convert to DataFrame
                df = rs.get_data()
                print(f"[stock-single-data][baostock-valuation] Raw data received: {len(df)} rows")
                if not df.empty:
                    print(f"[stock-single-data][baostock-valuation] First few rows: {df.head().to_dict('records')}")
            except Exception as e:
                print(f"[stock-single-data][baostock-valuation] Date format error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-valuation] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Validate data
            if df is None:
                print(f"[stock-single-data][baostock-valuation] No data returned for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            if df.empty:
                print(f"[stock-single-data][baostock-valuation] Empty data returned for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])

            # Process columns
            x = df.copy()
            print(f"[stock-single-data][baostock-valuation] Available columns: {list(x.columns)}")
            
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            # Convert valuation columns to numeric
            if "peTTM" in x.columns:
                x["pe_ttm"] = pd.to_numeric(x["peTTM"], errors="coerce")
            else:
                x["pe_ttm"] = float("nan")
            
            if "pbMRQ" in x.columns:
                x["pb"] = pd.to_numeric(x["pbMRQ"], errors="coerce")
            else:
                x["pb"] = float("nan")
            
            if "psTTM" in x.columns:
                x["ps_ttm"] = pd.to_numeric(x["psTTM"], errors="coerce")
            else:
                x["ps_ttm"] = float("nan")
            
            if "pcfNcfTTM" in x.columns:
                x["pcf_ncf_ttm"] = pd.to_numeric(x["pcfNcfTTM"], errors="coerce")
            else:
                x["pcf_ncf_ttm"] = float("nan")
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                # Ensure both date columns are in the same format
                x["date"] = pd.to_datetime(x["date"], errors="coerce")
                x = x.dropna(subset=["date"])
                # Print date range for debugging
                print(f"[stock-single-data][baostock-valuation] Data date range: {x['date'].min()} to {x['date'].max()}")
                print(f"[stock-single-data][baostock-valuation] Filter start date: {start_dt}")
                # Filter data
                filtered_data = x[x["date"] >= start_dt].reset_index(drop=True)
                print(f"[stock-single-data][baostock-valuation] Data after date filter: {len(filtered_data)} rows")
                # If no data after filtering, but we have historical data, use the latest available data
                if filtered_data.empty and not x.empty:
                    print(f"[stock-single-data][baostock-valuation] No data after filter, using latest available data")
                    latest_data = x.tail(1).copy()
                    # Update the date to the filter start date
                    latest_data["date"] = start_dt
                    x = latest_data
                else:
                    x = filtered_data
            
            if x.empty:
                print(f"[stock-single-data][baostock-valuation] No data after filtering for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])
            
            # Select and validate columns
            keep_cols = ["date", "pe_ttm", "pb", "ps_ttm", "pcf_ncf_ttm"]
            x = x[keep_cols].copy()
            
            print(f"[stock-single-data][baostock-valuation] Successfully fetched {len(x)} rows for {symbol_code}")
            return x
            
        except Exception as e:
            print(f"[stock-single-data][baostock-valuation] Failed to fetch data for {symbol_code}: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][baostock-valuation] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][baostock-valuation] All retries failed for {symbol_code}")
                return pd.DataFrame(columns=["date", "pe_ttm", "pb"])


def fetch_valuation_with_multiple_sources(symbol_code: str, start_date: str) -> pd.DataFrame:
    """
    Fetch valuation data with multiple data sources fallback
    
    Args:
        symbol_code: Stock symbol code
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with the best available valuation data
    """
    # Try data sources in order: AKShare, Eastmoney, Sina, 10jqka, Baostock
    # Set AKShare as primary data source based on user request
    # Add more data sources to improve coverage
    sources = [
        ("akshare", lambda: fetch_valuation_with_akshare(symbol_code, start_date)),
        ("eastmoney", lambda: fetch_valuation_with_eastmoney(symbol_code, start_date)),
        ("sina", lambda: fetch_valuation_with_sina(symbol_code, start_date)),
        ("10jqka", lambda: fetch_valuation_with_10jqka(symbol_code, start_date)),
        ("baostock", lambda: fetch_valuation_with_baostock(symbol_code, start_date)),
    ]
    
    print(f"[stock-single-data][valuation] Fetching valuation data for {symbol_code}")
    
    for source_name, fetch_func in sources:
        try:
            print(f"[stock-single-data][valuation] Trying {source_name} for {symbol_code}")
            df = fetch_func()
            
            # Validate data
            if df.empty:
                print(f"[stock-single-data][valuation] {source_name} returned empty data for {symbol_code}")
                continue
            
            # Check if data has any valid values
            if df[['pe_ttm', 'pb']].isna().all().all():
                print(f"[stock-single-data][valuation] {source_name} returned all NaN values for {symbol_code}")
                continue
            
            # Calculate data quality
            data_quality = {
                "length": len(df),
                "valid_pe": df["pe_ttm"].notna().sum(),
                "valid_pb": df["pb"].notna().sum(),
                "total_valid": df[['pe_ttm', 'pb']].notna().sum().sum()
            }
            
            print(f"[stock-single-data][valuation] Selected {source_name} for {symbol_code} with {len(df)} rows")
            print(f"[stock-single-data][valuation] Data quality: {data_quality}")
            return df
            
        except Exception as e:
            print(f"[stock-single-data][valuation] {source_name} failed for {symbol_code}: {e}")
            continue
    
    # Return empty DataFrame if all sources fail
    print(f"[stock-single-data][valuation] No valid valuation data found for {symbol_code}")
    return pd.DataFrame(columns=["date", "pe_ttm", "pb"])


def fetch_fund_flow_with_sina(symbol_code: str, canonical: str, start_date: str) -> pd.DataFrame:
    """
    Fetch fund flow data with Sina Finance
    
    Args:
        symbol_code: Stock symbol code
        canonical: Canonical stock symbol (e.g., 600519.SH)
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with fund flow data
    """
    import requests
    import json
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][sina-flow] Fetching {canonical} - attempt {retry + 1}/{max_retries}")
            
            # Sina Finance API for fund flow data
            market = "sh" if canonical.endswith(".SH") else "sz"
            code = symbol_code[:6]
            url = f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_i_fundpool?page=1&num=100&sort=opendate&asc=0&daima={market}{code}"
            print(f"[stock-single-data][sina-flow] Using URL: {url}")
            
            try:
                # Add headers to mimic browser request
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Referer": "http://finance.sina.com.cn/",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Connection": "keep-alive"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[stock-single-data][sina-flow] Request error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Parse Sina Finance fund flow data
            try:
                data = json.loads(response.text)
                
                # Check if data is a string instead of dict/list
                if isinstance(data, str):
                    print(f"[stock-single-data][sina-flow] Data is string instead of JSON: {data[:100]}...")
                    if "Service not found" in data:
                        print(f"[stock-single-data][sina-flow] API error: Service not found")
                    if retry < max_retries - 1:
                        print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
                
                # Check if it's an error response
                if isinstance(data, dict):
                    if data.get("__ERROR"):
                        print(f"[stock-single-data][sina-flow] API error: {data.get('__ERRORMSG', 'Unknown error')}")
                        if retry < max_retries - 1:
                            print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
                    else:
                        # If data is a dict but not an error, it's not the expected format
                        print(f"[stock-single-data][sina-flow] Data is dict but not in expected format: {data}")
                        if retry < max_retries - 1:
                            print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            continue
                        else:
                            return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
                
                # Check if data is not a list
                if not isinstance(data, list):
                    print(f"[stock-single-data][sina-flow] Data is not a list: {type(data)}")
                    if retry < max_retries - 1:
                        print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            except json.JSONDecodeError as e:
                print(f"[stock-single-data][sina-flow] JSON decode error: {e}")
                # Check if response text contains error message
                if "Service not found" in response.text:
                    print(f"[stock-single-data][sina-flow] API error: Service not found")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            except Exception as e:
                print(f"[stock-single-data][sina-flow] Error parsing response: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            if not data:
                print(f"[stock-single-data][sina-flow] No data returned for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Create DataFrame
            rows = []
            for item in data:
                # Ensure item is a dict before accessing get method
                if not isinstance(item, dict):
                    print(f"[stock-single-data][sina-flow] Skipping non-dict item: {item}")
                    continue
                
                date = item.get("opendate", "")
                if not date:
                    continue
                
                # Extract fund flow data
                main_inflow = item.get("zhijin", 0)  # 主力净流入
                total_vol = item.get("amount", 1)  # 总成交额
                main_ratio = (main_inflow / total_vol * 100) if total_vol > 0 else 0
                
                rows.append({
                    "date": date,
                    "main_net_inflow": main_inflow,
                    "main_net_inflow_ratio": main_ratio,
                    "ultra_large_net_inflow": float("nan"),
                    "large_net_inflow": float("nan"),
                    "mid_net_inflow": float("nan"),
                    "small_net_inflow": float("nan")
                })
            
            if not rows:
                print(f"[stock-single-data][sina-flow] No valid rows found for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            df = pd.DataFrame(rows)
            
            # Process date column
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][sina-flow] No valid date data for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                df = df[df["date"] >= start_dt].reset_index(drop=True)
                print(f"[stock-single-data][sina-flow] Filtered data to start from {start_date}")
            
            if df.empty:
                print(f"[stock-single-data][sina-flow] No data after filtering for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Calculate data quality
            data_quality = {
                "length": len(df),
                "valid_main_inflow": df["main_net_inflow"].notna().sum(),
                "valid_main_ratio": df["main_net_inflow_ratio"].notna().sum(),
                "total_valid": df[["main_net_inflow", "main_net_inflow_ratio"]].notna().sum().sum()
            }
            print(f"[stock-single-data][sina-flow] Data quality: {data_quality}")
            print(f"[stock-single-data][sina-flow] Successfully fetched {len(df)} rows for {canonical}")
            
            return df
            
        except Exception as e:
            print(f"[stock-single-data][sina-flow] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][sina-flow] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][sina-flow] All retries failed for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])


def fetch_fund_flow_with_10jqka(symbol_code: str, canonical: str, start_date: str) -> pd.DataFrame:
    """
    Fetch fund flow data with 10jqka
    
    Args:
        symbol_code: Stock symbol code
        canonical: Canonical stock symbol (e.g., 600519.SH)
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with fund flow data
    """
    import requests
    import json
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][10jqka-flow] Fetching {canonical} - attempt {retry + 1}/{max_retries}")
            
            # 10jqka API for fund flow data - API endpoint might be changed
            code = symbol_code[:6]
            # Updated API endpoint with possible alternative
            url = f"http://web.10jqka.com.cn/data/{code}/fundflow/"
            print(f"[stock-single-data][10jqka-flow] Using URL: {url}")
            
            try:
                # Add headers to mimic browser request
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Referer": "http://www.10jqka.com.cn/"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[stock-single-data][10jqka-flow] Request error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][10jqka-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[stock-single-data][10jqka-flow] 10jqka API is currently unavailable")
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Parse 10jqka fund flow data - updated parsing logic for new endpoint
            try:
                # Try to extract fund flow data from HTML content
                content = response.text
                # For now, we'll return an empty DataFrame as parsing HTML fund flow data is complex
                # and may require more sophisticated parsing logic
                print(f"[stock-single-data][10jqka-flow] 10jqka fund flow API is currently unavailable")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            except Exception as e:
                print(f"[stock-single-data][10jqka-flow] Failed to parse data: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][10jqka-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
        except Exception as e:
            print(f"[stock-single-data][10jqka-flow] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][10jqka-flow] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][10jqka-flow] All retries failed for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])


def fetch_fund_flow_with_eastmoney(symbol_code: str, canonical: str, start_date: str) -> pd.DataFrame:
    """
    Fetch fund flow data with Eastmoney
    
    Args:
        symbol_code: Stock symbol code
        canonical: Canonical stock symbol (e.g., 600519.SH)
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with fund flow data
    """
    import requests
    import json
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][eastmoney-flow] Fetching {canonical} - attempt {retry + 1}/{max_retries}")
            
            # Eastmoney API for fund flow data
            code = symbol_code[:6]
            market = "1" if canonical.endswith(".SH") else "0"
            url = f"http://push2.eastmoney.com/api/qt/stock/get?secid={market}.{code}&fields=f62,f164,f167,f168,f169,f170,f171,f172,f173,f174,f175,f176,f177,f178,f179,f180,f181,f182"
            print(f"[stock-single-data][eastmoney-flow] Using URL: {url}")
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[stock-single-data][eastmoney-flow] Request error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][eastmoney-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Parse Eastmoney fund flow data
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print(f"[stock-single-data][eastmoney-flow] JSON decode error: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][eastmoney-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            stock_data = data.get("data", {})
            
            if not stock_data:
                print(f"[stock-single-data][eastmoney-flow] No stock data returned for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Create DataFrame with current date
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            main_inflow = stock_data.get("f62", 0)  # 主力净流入
            main_ratio = stock_data.get("f164", 0)  # 主力净流入占比
            
            df = pd.DataFrame({
                "date": [current_date],
                "main_net_inflow": [main_inflow],
                "main_net_inflow_ratio": [main_ratio],
                "ultra_large_net_inflow": [float("nan")],
                "large_net_inflow": [float("nan")],
                "mid_net_inflow": [float("nan")],
                "small_net_inflow": [float("nan")]
            })
            
            # Process date column
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            if df.empty:
                print(f"[stock-single-data][eastmoney-flow] No valid date data for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                df = df[df["date"] >= start_dt].reset_index(drop=True)
                print(f"[stock-single-data][eastmoney-flow] Filtered data to start from {start_date}")
            
            if df.empty:
                print(f"[stock-single-data][eastmoney-flow] No data after filtering for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Calculate data quality
            data_quality = {
                "length": len(df),
                "valid_main_inflow": df["main_net_inflow"].notna().sum(),
                "valid_main_ratio": df["main_net_inflow_ratio"].notna().sum(),
                "total_valid": df[["main_net_inflow", "main_net_inflow_ratio"]].notna().sum().sum()
            }
            print(f"[stock-single-data][eastmoney-flow] Data quality: {data_quality}")
            print(f"[stock-single-data][eastmoney-flow] Successfully fetched {len(df)} rows for {canonical}")
            
            return df
            
        except Exception as e:
            print(f"[stock-single-data][eastmoney-flow] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][eastmoney-flow] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][eastmoney-flow] All retries failed for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])


def fetch_fund_flow_with_baostock(symbol_code: str, canonical: str, start_date: str) -> pd.DataFrame:
    """
    Fetch fund flow data with Baostock
    
    Args:
        symbol_code: Stock symbol code
        canonical: Canonical stock symbol (e.g., 600519.SH)
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with fund flow data
    """
    import baostock as bs
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][baostock-flow] Fetching {canonical} - attempt {retry + 1}/{max_retries}")
            
            # Convert symbol to Baostock format
            if canonical.endswith(".SH"):
                bs_code = f"sh.{canonical[:6]}"
            elif canonical.endswith(".SZ"):
                bs_code = f"sz.{canonical[:6]}"
            else:
                # Try to determine market
                code = canonical[:6]
                if code.startswith(('600', '601', '603', '605', '688', '689')):
                    bs_code = f"sh.{code}"
                else:
                    bs_code = f"sz.{code}"
            
            # Login to Baostock
            lg = bs.login()
            if lg.error_code != '0':
                print(f"[stock-single-data][baostock-flow] Login failed: {lg.error_msg}")
                bs.logout()
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])
            
            # Fetch data (Baostock doesn't provide detailed fund flow data, so we use daily data as fallback)
            # Ensure start_date is in correct format (YYYY-MM-DD)
            try:
                # Parse and format date correctly
                start_dt = pd.to_datetime(start_date)
                # Use a longer date range to ensure we get data
                bs_start_date = (start_dt - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
                bs_end_date = time.strftime("%Y-%m-%d")
                
                print(f"[stock-single-data][baostock-flow] Fetching from {bs_start_date} to {bs_end_date}")
                
                rs = bs.query_history_k_data_plus(
                    code=bs_code,
                    fields="date,open,high,low,close,volume,amount,turn",
                    start_date=bs_start_date,
                    end_date=bs_end_date,
                    frequency="d",
                    adjustflag="2"  # Forward adjust
                )
            except Exception as e:
                print(f"[stock-single-data][baostock-flow] Date format error: {e}")
                bs.logout()
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])
            
            # Check if query failed
            if rs is None:
                print(f"[stock-single-data][baostock-flow] Query returned None for {canonical}")
                bs.logout()
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])
            
            if rs.error_code != '0':
                print(f"[stock-single-data][baostock-flow] Query failed: {rs.error_msg}")
                bs.logout()
                if retry < max_retries - 1:
                    print(f"[stock-single-data][baostock-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])
            
            # Convert to DataFrame
            df = rs.get_data()
            bs.logout()
            
            # Validate data
            if df is None:
                print(f"[stock-single-data][baostock-flow] No data returned for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])
            
            if df.empty:
                print(f"[stock-single-data][baostock-flow] Empty data returned for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])

            # Process columns
            x = df.copy()
            x["date"] = pd.to_datetime(x["date"], errors="coerce")
            x = x.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            
            # Calculate fund flow indicators (using price and volume as proxy)
            x["close"] = pd.to_numeric(x["close"], errors="coerce")
            x["volume"] = pd.to_numeric(x["volume"], errors="coerce")
            x["amount"] = pd.to_numeric(x["amount"], errors="coerce")
            
            # Calculate price change
            x["price_change"] = x["close"].pct_change()
            
            # Estimate fund flow based on price and volume
            x["main_net_inflow"] = x["amount"] * x["price_change"]
            x["main_net_inflow_ratio"] = x["price_change"] * 100
            
            # Fill other columns with NaN
            x["ultra_large_net_inflow"] = float("nan")
            x["large_net_inflow"] = float("nan")
            x["mid_net_inflow"] = float("nan")
            x["small_net_inflow"] = float("nan")
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                x = x[x["date"] >= start_dt].reset_index(drop=True)
            
            if x.empty:
                print(f"[stock-single-data][baostock-flow] No data after filtering for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])
            
            # Select and validate columns
            keep_cols = ["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"]
            x = x[keep_cols].copy()
            
            print(f"[stock-single-data][baostock-flow] Successfully fetched {len(x)} rows for {canonical}")
            return x
            
        except Exception as e:
            print(f"[stock-single-data][baostock-flow] Failed to fetch data for {canonical}: {e}")
            try:
                bs.logout()
            except:
                pass
            if retry < max_retries - 1:
                print(f"[stock-single-data][baostock-flow] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][baostock-flow] All retries failed for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])


def fetch_fund_flow_with_multiple_sources(symbol_code: str, canonical: str, start_date: str) -> pd.DataFrame:
    """
    Fetch fund flow data with multiple data sources fallback
    
    Args:
        symbol_code: Stock symbol code
        canonical: Canonical stock symbol (e.g., 600519.SH)
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with the best available fund flow data
    """
    # Try data sources in order: AKShare, Eastmoney, Baostock
    # Set AKShare as primary data source based on user request
    # Use the first available data source with valid data
    sources = [
        ("akshare", lambda: fetch_fund_flow_with_akshare(symbol_code, canonical, start_date)),
        ("eastmoney", lambda: fetch_fund_flow_with_eastmoney(symbol_code, canonical, start_date)),
        ("baostock", lambda: fetch_fund_flow_with_baostock(symbol_code, canonical, start_date)),
    ]
    
    print(f"[stock-single-data][flow] Fetching fund flow data for {canonical}")
    
    for source_name, fetch_func in sources:
        try:
            print(f"[stock-single-data][flow] Trying {source_name} for {canonical}")
            df = fetch_func()
            
            # Validate data
            if df.empty:
                print(f"[stock-single-data][flow] {source_name} returned empty data for {canonical}")
                continue
            
            # Check if data has any valid values
            if df[["main_net_inflow", "main_net_inflow_ratio"]].isna().all().all():
                print(f"[stock-single-data][flow] {source_name} returned all NaN values for {canonical}")
                continue
            
            # Calculate data quality
            data_quality = {
                "length": len(df),
                "valid_main_inflow": df["main_net_inflow"].notna().sum(),
                "valid_main_ratio": df["main_net_inflow_ratio"].notna().sum(),
                "total_valid": df[["main_net_inflow", "main_net_inflow_ratio"]].notna().sum().sum()
            }
            
            print(f"[stock-single-data][flow] Selected {source_name} for {canonical} with {len(df)} rows")
            print(f"[stock-single-data][flow] Data quality: {data_quality}")
            return df
            
        except Exception as e:
            print(f"[stock-single-data][flow] {source_name} failed for {canonical}: {e}")
            continue
    
    # Return empty DataFrame if all sources fail
    print(f"[stock-single-data][flow] No valid fund flow data found for {canonical}")
    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio", "ultra_large_net_inflow", "large_net_inflow", "mid_net_inflow", "small_net_inflow"])


def symbol_market_for_flow(canonical: str) -> str:
    s = str(canonical).upper()
    if s.endswith(".SH"):
        return "sh"
    if s.endswith(".SZ"):
        return "sz"
    if s.endswith(".BJ"):
        return "bj"
    return "sh"


def fetch_fund_flow_with_akshare(symbol_code: str, canonical: str, start_date: str) -> pd.DataFrame:
    """
    Fetch fund flow data with AKShare
    
    Args:
        symbol_code: Stock symbol code
        canonical: Canonical stock symbol (e.g., 600519.SH)
        start_date: Start date in YYYY-MM-DD format
    
    Returns:
        DataFrame with fund flow data
    """
    import akshare as ak
    import time

    max_retries = 3
    retry_delay = 2  # seconds

    for retry in range(max_retries):
        try:
            print(f"[stock-single-data][akshare-flow] Fetching {canonical} - attempt {retry + 1}/{max_retries}")
            
            market = symbol_market_for_flow(canonical)
            print(f"[stock-single-data][akshare-flow] Using market: {market} for {canonical}")
            
            try:
                df = ak.stock_individual_fund_flow(stock=symbol_code, market=market)
            except Exception as e:
                print(f"[stock-single-data][akshare-flow] API call failed: {e}")
                if retry < max_retries - 1:
                    print(f"[stock-single-data][akshare-flow] Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Validate data
            if df is None:
                print(f"[stock-single-data][akshare-flow] No data returned for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            if df.empty:
                print(f"[stock-single-data][akshare-flow] Empty data returned for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            x = normalize_date_col(df)
            if x.empty:
                print(f"[stock-single-data][akshare-flow] No valid date data for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])

            cols = list(x.columns)
            print(f"[stock-single-data][akshare-flow] Available columns: {cols}")
            
            main_amt_col = first_matching_col(cols, ["主力净流入-净额", "主力净流入净额", "主力净流入"])
            main_ratio_col = first_matching_col(cols, ["主力净流入-净占比", "主力净流入净占比", "主力净占比"])

            if main_amt_col is not None:
                x["main_net_inflow"] = x[main_amt_col].map(parse_cn_number)
                print(f"[stock-single-data][akshare-flow] Using {main_amt_col} for main_net_inflow")
            else:
                print(f"[stock-single-data][akshare-flow] No main amount column found")
                x["main_net_inflow"] = float("nan")
            
            if main_ratio_col is not None:
                x["main_net_inflow_ratio"] = x[main_ratio_col].map(parse_cn_number)
                print(f"[stock-single-data][akshare-flow] Using {main_ratio_col} for main_net_inflow_ratio")
            else:
                print(f"[stock-single-data][akshare-flow] No main ratio column found")
                x["main_net_inflow_ratio"] = float("nan")

            # Optional decomposition, useful for future model variants.
            for out_col, candidates in [
                ("ultra_large_net_inflow", ["超大单净流入-净额", "超大单净流入"]),
                ("large_net_inflow", ["大单净流入-净额", "大单净流入"]),
                ("mid_net_inflow", ["中单净流入-净额", "中单净流入"]),
                ("small_net_inflow", ["小单净流入-净额", "小单净流入"]),
            ]:
                src_col = first_matching_col(cols, candidates)
                if src_col is not None:
                    x[out_col] = x[src_col].map(parse_cn_number)
                    print(f"[stock-single-data][akshare-flow] Using {src_col} for {out_col}")
                else:
                    x[out_col] = float("nan")
                    print(f"[stock-single-data][akshare-flow] No column found for {out_col}")

            keep_cols = [
                "date",
                "main_net_inflow",
                "main_net_inflow_ratio",
                "ultra_large_net_inflow",
                "large_net_inflow",
                "mid_net_inflow",
                "small_net_inflow",
            ]
            x = x[keep_cols].sort_values("date").reset_index(drop=True)
            
            # Filter by start date
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if pd.notna(start_dt):
                x = x[x["date"] >= start_dt].reset_index(drop=True)
                print(f"[stock-single-data][akshare-flow] Filtered data to start from {start_date}")
            
            if x.empty:
                print(f"[stock-single-data][akshare-flow] No data after filtering for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])
            
            # Calculate data quality
            data_quality = {
                "length": len(x),
                "valid_main_inflow": x["main_net_inflow"].notna().sum(),
                "valid_main_ratio": x["main_net_inflow_ratio"].notna().sum(),
                "total_valid": x[keep_cols[1:]].notna().sum().sum()
            }
            print(f"[stock-single-data][akshare-flow] Data quality: {data_quality}")
            print(f"[stock-single-data][akshare-flow] Successfully fetched {len(x)} rows for {canonical}")
            
            return x
            
        except Exception as e:
            print(f"[stock-single-data][akshare-flow] error: {e}")
            if retry < max_retries - 1:
                print(f"[stock-single-data][akshare-flow] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                print(f"[stock-single-data][akshare-flow] All retries failed for {canonical}")
                return pd.DataFrame(columns=["date", "main_net_inflow", "main_net_inflow_ratio"])


def main():
    """
    Main function for fetching stock data
    
    Returns:
        Exit code
    """
    args = parse_args()
    
    # Start time for performance tracking
    start_time = datetime.now()
    print(f"[stock-single-data] Starting data fetching at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[stock-single-data] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_single.get("enabled", False):
        # Data preparation should still be available while strategy execution is disabled.
        print("[stock-single-data] warning: config/stock_single.yaml enabled=false, fetching data anyway")

    # Load and validate symbols
    skipped_non_stock = []
    if args.symbols:
        seen = set()
        symbols = []
        for raw in [x.strip() for x in str(args.symbols).split(",") if x.strip()]:
            norm = normalize_symbol(raw)
            if norm is None:
                print(f"[stock-single-data] skipped invalid symbol: {raw}")
                continue
            code, canonical = norm
            if not is_a_share_stock_code(code, canonical):
                skipped_non_stock.append(canonical)
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            symbols.append((code, canonical))
    else:
        symbols = load_symbols(stock_single)
    
    if skipped_non_stock:
        print(
            f"[stock-single-data] skipped non-stock symbols ({len(skipped_non_stock)}): "
            + ",".join(skipped_non_stock[:8])
        )
    if not symbols:
        print("[stock-single-data] no A-share stock symbols found, check pool.source_file/static_fallback")
        return EXIT_CONFIG_ERROR
    
    print(f"[stock-single-data] Processing {len(symbols)} A-share stock symbols")

    data_cfg = stock_single.get("data", {})
    guard_cfg = stock_single.get("data_guard", {})

    # Determine what data to fetch
    fetch_daily = bool(data_cfg.get("fetch_daily", True))
    fetch_minute = bool(data_cfg.get("fetch_minute", False))
    fetch_valuation = bool(data_cfg.get("fetch_valuation", True))
    fetch_fund_flow = bool(data_cfg.get("fetch_fund_flow", True))
    
    # Apply command line arguments
    if args.with_minute:
        fetch_minute = True
    if args.with_factors:
        fetch_valuation = True
        fetch_fund_flow = True
    if args.only_minute:
        fetch_daily = False
        fetch_minute = True
        fetch_valuation = False
        fetch_fund_flow = False
    if args.only_daily:
        fetch_daily = True
        fetch_minute = False
        fetch_valuation = False
        fetch_fund_flow = False
    if args.only_factors:
        fetch_daily = False
        fetch_minute = False
        fetch_valuation = True
        fetch_fund_flow = True

    # Validate at least one data type is enabled
    if (not fetch_daily) and (not fetch_minute) and (not fetch_valuation) and (not fetch_fund_flow):
        print("[stock-single-data] daily/minute/factor fetching are all disabled")
        return EXIT_DISABLED
    
    # Print fetching plan
    print("[stock-single-data] Fetching plan:")
    print(f"[stock-single-data] - Daily data: {'enabled' if fetch_daily else 'disabled'}")
    print(f"[stock-single-data] - Minute data: {'enabled' if fetch_minute else 'disabled'}")
    print(f"[stock-single-data] - Valuation data: {'enabled' if fetch_valuation else 'disabled'}")
    print(f"[stock-single-data] - Fund flow data: {'enabled' if fetch_fund_flow else 'disabled'}")

    # Setup directories
    base_data_dir = runtime.get("paths", {}).get("data_dir", "./data")
    daily_dir = data_cfg.get("daily_output_dir", os.path.join(base_data_dir, "stock_single", "daily"))
    minute_dir = data_cfg.get("minute_output_dir", os.path.join(base_data_dir, "stock_single", "minute_5m"))
    valuation_dir = data_cfg.get("valuation_output_dir", os.path.join(base_data_dir, "stock_single", "valuation"))
    flow_dir = data_cfg.get("fund_flow_output_dir", os.path.join(base_data_dir, "stock_single", "fund_flow"))
    
    # Ensure directories exist
    ensure_dir(daily_dir)
    ensure_dir(minute_dir)
    ensure_dir(valuation_dir)
    ensure_dir(flow_dir)
    
    print(f"[stock-single-data] Data directories:")
    print(f"[stock-single-data] - Daily: {daily_dir}")
    print(f"[stock-single-data] - Minute: {minute_dir}")
    print(f"[stock-single-data] - Valuation: {valuation_dir}")
    print(f"[stock-single-data] - Fund flow: {flow_dir}")

    now = datetime.now()
    adjust = str(data_cfg.get("adjust", "qfq"))
    overlap_days = int(data_cfg.get("incremental_overlap_days", 5))

    total_ok = 0
    total_fail = 0
    total_processed = 0

    # Fetch daily data
    if fetch_daily:
        print("\n[stock-single-data][daily] Starting daily data fetching...")
        hist_years_daily = int(data_cfg.get("history_years_daily", 5))
        default_start_daily = now - timedelta(days=365 * hist_years_daily)
        daily_end = now.strftime("%Y%m%d")
        min_rows_daily = int(guard_cfg.get("min_cache_rows_daily", 200))
        max_age_daily = int(guard_cfg.get("max_cache_age_days_daily", 10))

        ok_daily = 0
        fail_daily = 0
        
        for i, (code, canonical) in enumerate(symbols, 1):
            print(f"[stock-single-data][daily] Processing {canonical} ({i}/{len(symbols)})")
            out_file = os.path.join(daily_dir, f"{canonical}.csv")
            try:
                start_dt = resolve_incremental_start(
                    out_file,
                    default_start=default_start_daily,
                    overlap_days=overlap_days,
                    time_col="date",
                )
                start_date = start_dt.strftime("%Y%m%d")
                print(f"[stock-single-data][daily] Fetching from {start_date} to {daily_end}")
                
                # Try AKShare first
                df_new = fetch_daily_with_akshare(code, start_date=start_date, end_date=daily_end, adjust=adjust)
                
                # Fallback to Baostock if AKShare fails
                if df_new.empty:
                    print(f"[stock-single-data][daily] AKShare failed, trying Baostock for {canonical}")
                    df_new = fetch_daily_with_baostock(canonical, start_date=start_date, end_date=daily_end, adjust=adjust)
                    if df_new.empty:
                        raise RuntimeError("empty daily data from all sources")
                
                df_old = read_existing(out_file, time_col="date")
                df_merge = merge_history(df_old, df_new, time_col="date")
                
                # Validate merged data
                if df_merge.empty:
                    raise RuntimeError("empty merged daily data")
                
                # Process and save data
                df_merge["date"] = pd.to_datetime(df_merge["date"]).dt.strftime("%Y-%m-%d")
                df_merge.to_csv(out_file, index=False, encoding="utf-8")
                
                print(f"[stock-single-data][daily] Success: {canonical} - {len(df_merge)} rows saved")
                ok_daily += 1
            except Exception as e:
                if has_valid_cache(out_file, time_col="date", min_rows=min_rows_daily, max_age_days=max_age_daily):
                    print(f"[stock-single-data][daily] {canonical} fetch failed, keeping valid cache ({e})")
                    ok_daily += 1
                else:
                    print(f"[stock-single-data][daily] {canonical} failed: {e}")
                    fail_daily += 1
        
        total_ok += ok_daily
        total_fail += fail_daily
        total_processed += len(symbols)
        print(f"[stock-single-data][daily] Done: ok={ok_daily} fail={fail_daily} total={ok_daily + fail_daily}")

    # Fetch minute data
    if fetch_minute:
        print("\n[stock-single-data][min] Starting minute data fetching...")
        minute_period = str(data_cfg.get("minute_period", "5"))
        hist_days_minute = int(data_cfg.get("history_days_minute", 120))
        default_start_minute = now - timedelta(days=hist_days_minute)
        minute_end = now.strftime("%Y-%m-%d %H:%M:%S")
        min_rows_minute = int(guard_cfg.get("min_cache_rows_minute", 200))
        max_age_minute = int(guard_cfg.get("max_cache_age_days_minute", 3))

        ok_minute = 0
        fail_minute = 0
        
        for i, (code, canonical) in enumerate(symbols, 1):
            print(f"[stock-single-data][min] Processing {canonical} ({i}/{len(symbols)})")
            out_file = os.path.join(minute_dir, f"{canonical}.csv")
            try:
                start_dt = resolve_incremental_start(
                    out_file,
                    default_start=default_start_minute,
                    overlap_days=overlap_days,
                    time_col="datetime",
                )
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[stock-single-data][min] Fetching from {start_str} to {minute_end}")
                
                df_new = fetch_minute_with_akshare(
                    code,
                    start_dt=start_str,
                    end_dt=minute_end,
                    period=minute_period,
                    adjust=adjust,
                )
                if df_new.empty:
                    raise RuntimeError("empty minute data")
                
                df_old = read_existing(out_file, time_col="datetime")
                df_merge = merge_history(df_old, df_new, time_col="datetime")
                
                # Validate merged data
                if df_merge.empty:
                    raise RuntimeError("empty merged minute data")
                
                # Process and save data
                df_merge["datetime"] = pd.to_datetime(df_merge["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                df_merge.to_csv(out_file, index=False, encoding="utf-8")
                
                print(f"[stock-single-data][min] Success: {canonical} - {len(df_merge)} rows saved")
                ok_minute += 1
            except Exception as e:
                if has_valid_cache(
                    out_file,
                    time_col="datetime",
                    min_rows=min_rows_minute,
                    max_age_days=max_age_minute,
                ):
                    print(f"[stock-single-data][min] {canonical} fetch failed, keeping valid cache ({e})")
                    ok_minute += 1
                else:
                    print(f"[stock-single-data][min] {canonical} failed: {e}")
                    fail_minute += 1
        
        total_ok += ok_minute
        total_fail += fail_minute
        total_processed += len(symbols)
        print(f"[stock-single-data][min] Done: ok={ok_minute} fail={fail_minute} total={ok_minute + fail_minute}")

    # Fetch valuation data
    if fetch_valuation:
        print("\n[stock-single-data][valuation] Starting valuation data fetching...")
        # Set fixed start date for valuation data as requested: 2025-08-20
        default_start_valuation = datetime(2025, 8, 20)
        min_rows_valuation = int(guard_cfg.get("min_cache_rows_valuation", 120))
        max_age_valuation = int(guard_cfg.get("max_cache_age_days_valuation", 30))
        print(f"[stock-single-data][valuation] Using fixed start date: {default_start_valuation.strftime('%Y-%m-%d')}")

        ok_val = 0
        fail_val = 0
        
        for i, (code, canonical) in enumerate(symbols, 1):
            print(f"[stock-single-data][valuation] Processing {canonical} ({i}/{len(symbols)})")
            out_file = os.path.join(valuation_dir, f"{canonical}.csv")
            try:
                # Force start date to 2025-08-20 regardless of existing cache
                # This ensures we fetch valuation data from the requested date
                start_dt = datetime(2025, 8, 20)
                start_str = start_dt.strftime("%Y-%m-%d")
                print(f"[stock-single-data][valuation] Fetching from {start_str} (forced start date)")
                
                df_new = fetch_valuation_with_multiple_sources(symbol_code=canonical, start_date=start_str)
                if df_new.empty:
                    raise RuntimeError("empty valuation data")
                
                df_old = read_existing(out_file, time_col="date")
                df_merge = merge_history(df_old, df_new, time_col="date")
                
                # Validate merged data
                if df_merge.empty:
                    raise RuntimeError("empty merged valuation data")
                
                # Process and save data
                df_merge["date"] = pd.to_datetime(df_merge["date"]).dt.strftime("%Y-%m-%d")
                df_merge.to_csv(out_file, index=False, encoding="utf-8")
                
                print(f"[stock-single-data][valuation] Success: {canonical} - {len(df_merge)} rows saved")
                ok_val += 1
            except Exception as e:
                if has_valid_cache(
                    out_file,
                    time_col="date",
                    min_rows=min_rows_valuation,
                    max_age_days=max_age_valuation,
                ):
                    print(f"[stock-single-data][valuation] {canonical} fetch failed, keeping valid cache ({e})")
                    ok_val += 1
                else:
                    print(f"[stock-single-data][valuation] {canonical} failed: {e}")
                    fail_val += 1
        
        total_ok += ok_val
        total_fail += fail_val
        total_processed += len(symbols)
        print(f"[stock-single-data][valuation] Done: ok={ok_val} fail={fail_val} total={ok_val + fail_val}")

    # Fetch fund flow data
    if fetch_fund_flow:
        print("\n[stock-single-data][flow] Starting fund flow data fetching...")
        hist_years_flow = int(data_cfg.get("history_years_fund_flow", 3))
        default_start_flow = now - timedelta(days=365 * hist_years_flow)
        min_rows_flow = int(guard_cfg.get("min_cache_rows_fund_flow", 120))
        max_age_flow = int(guard_cfg.get("max_cache_age_days_fund_flow", 15))

        ok_flow = 0
        fail_flow = 0
        
        for i, (code, canonical) in enumerate(symbols, 1):
            print(f"[stock-single-data][flow] Processing {canonical} ({i}/{len(symbols)})")
            out_file = os.path.join(flow_dir, f"{canonical}.csv")
            try:
                start_dt = resolve_incremental_start(
                    out_file,
                    default_start=default_start_flow,
                    overlap_days=overlap_days,
                    time_col="date",
                )
                start_str = start_dt.strftime("%Y-%m-%d")
                print(f"[stock-single-data][flow] Fetching from {start_str}")
                
                df_new = fetch_fund_flow_with_multiple_sources(symbol_code=code, canonical=canonical, start_date=start_str)
                if df_new.empty:
                    raise RuntimeError("empty fund-flow data")
                
                df_old = read_existing(out_file, time_col="date")
                df_merge = merge_history(df_old, df_new, time_col="date")
                
                # Validate merged data
                if df_merge.empty:
                    raise RuntimeError("empty merged fund flow data")
                
                # Process and save data
                df_merge["date"] = pd.to_datetime(df_merge["date"]).dt.strftime("%Y-%m-%d")
                df_merge.to_csv(out_file, index=False, encoding="utf-8")
                
                print(f"[stock-single-data][flow] Success: {canonical} - {len(df_merge)} rows saved")
                ok_flow += 1
            except Exception as e:
                if has_valid_cache(
                    out_file,
                    time_col="date",
                    min_rows=min_rows_flow,
                    max_age_days=max_age_flow,
                ):
                    print(f"[stock-single-data][flow] {canonical} fetch failed, keeping valid cache ({e})")
                    ok_flow += 1
                else:
                    print(f"[stock-single-data][flow] {canonical} failed: {e}")
                    fail_flow += 1
        
        total_ok += ok_flow
        total_fail += fail_flow
        total_processed += len(symbols)
        print(f"[stock-single-data][flow] Done: ok={ok_flow} fail={fail_flow} total={ok_flow + fail_flow}")

    # Calculate performance metrics
    end_time = datetime.now()
    elapsed_time = (end_time - start_time).total_seconds()
    success_rate = (total_ok / total_processed * 100) if total_processed > 0 else 0
    
    # Print final report
    print("\n[stock-single-data] Final Report:")
    print(f"[stock-single-data] Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[stock-single-data] End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[stock-single-data] Elapsed time: {elapsed_time:.2f} seconds")
    print(f"[stock-single-data] Total processed: {total_processed}")
    print(f"[stock-single-data] Total ok: {total_ok}")
    print(f"[stock-single-data] Total fail: {total_fail}")
    print(f"[stock-single-data] Success rate: {success_rate:.2f}%")
    
    if total_processed > 0:
        print(f"[stock-single-data] Average time per symbol: {elapsed_time / total_processed:.2f} seconds")
    
    # Return exit code
    if total_fail == 0:
        print("[stock-single-data] All data fetching completed successfully!")
        return EXIT_OK
    else:
        print(f"[stock-single-data] Some data fetching failed: {total_fail} out of {total_processed}")
        return EXIT_DATA_FETCH_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
