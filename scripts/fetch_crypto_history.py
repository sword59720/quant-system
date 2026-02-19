#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto 历史数据获取脚本
拉取主流币种 2020年至今的 4小时 K线数据，用于策略回测
"""

import os
import sys
import time
import pandas as pd
import ccxt
from datetime import datetime, timedelta

# 目标币种 (Top 10 流动性好的)
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "LTC/USDT", "BCH/USDT", "UNI/USDT"
]
TIMEFRAME = "4h"
START_DATE = "2020-01-01 00:00:00"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch_history(exchange, symbol, timeframe, start_ts):
    """
    分批拉取历史数据
    """
    all_ohlcv = []
    since = start_ts
    limit = 1000
    
    print(f"Fetch {symbol} since {datetime.fromtimestamp(since/1000)}")
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            print(f"  Got {len(ohlcv)} rows, last: {datetime.fromtimestamp(ohlcv[-1][0]/1000)}")
            
            # 更新 since
            last_ts = ohlcv[-1][0]
            # 加上一个 timeframe 的毫秒数，避免重复
            # 4h = 4 * 60 * 60 * 1000 = 14400000
            since = last_ts + 1
            
            if len(ohlcv) < limit:
                break
                
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(5)
            continue

    return all_ohlcv

def main():
    data_dir = "./data/crypto_history"
    ensure_dir(data_dir)
    
    # 使用 Binance (数据最全)
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'timeout': 30000
    })
    
    start_ts = int(datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    
    for symbol in SYMBOLS:
        safe_symbol = symbol.replace("/", "_")
        file_path = os.path.join(data_dir, f"{safe_symbol}.csv")
        
        if os.path.exists(file_path):
            print(f"Skip {symbol}, file exists: {file_path}")
            continue
            
        print(f"Fetching {symbol}...")
        try:
            ohlcv = fetch_history(exchange, symbol, TIMEFRAME, start_ts)
            
            if not ohlcv:
                print(f"  No data for {symbol}")
                continue
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 保存
            df.to_csv(file_path, index=False)
            print(f"Saved {symbol} to {file_path}, {len(df)} rows")
            
        except Exception as e:
            print(f"Failed to fetch {symbol}: {e}")

if __name__ == "__main__":
    main()
