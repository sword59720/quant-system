#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from datetime import datetime

import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_CONFIG_ERROR, EXIT_DISABLED, EXIT_OK, EXIT_OUTPUT_ERROR
from scripts.stock_single.fetch_stock_single_data import is_a_share_stock_code, normalize_symbol


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_symbols(raw_symbols):
    out = []
    seen = set()
    skipped_non_stock = []
    for symbol in raw_symbols:
        norm = normalize_symbol(symbol)
        if norm is None:
            continue
        code, canonical = norm
        if not is_a_share_stock_code(code, canonical):
            skipped_non_stock.append(canonical)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out, skipped_non_stock


def build_pool(runtime: dict, stock_single: dict) -> tuple[list[str], str]:
    pool_cfg = stock_single.get("pool", {})
    source_file = pool_cfg.get("source_file", "./data/stock_single/universe.csv")
    symbol_column = pool_cfg.get("symbol_column", "symbol")
    max_symbols = int(stock_single.get("max_symbols_in_pool", 150))
    liquidity_filter = stock_single.get("liquidity_filter", {})
    liquidity_enabled = bool(liquidity_filter.get("enabled", True))
    min_avg_turnover = float(liquidity_filter.get("min_avg_turnover", 100000000))
    min_avg_volume = float(liquidity_filter.get("min_avg_volume", 1000000))

    symbols = []
    if os.path.exists(source_file):
        df = pd.read_csv(source_file)
        if symbol_column not in df.columns:
            raise RuntimeError(f"pool source missing column: {symbol_column}")
        symbols = df[symbol_column].dropna().astype(str).tolist()

    if not symbols:
        symbols = [str(x) for x in pool_cfg.get("static_fallback", [])]

    symbols, skipped_non_stock = normalize_symbols(symbols)
    
    # 流动性筛选
    if liquidity_enabled:
        filtered_symbols = []
        data_dir = runtime.get("paths", {}).get("data_dir", "./data")
        daily_dir = stock_single.get("data", {}).get("daily_output_dir", os.path.join(data_dir, "stock_single", "daily"))
        
        for symbol in symbols:
            # 检查流动性数据
            # 这里简化处理，实际应该从数据文件中读取并计算平均成交额和成交量
            # 暂时跳过流动性筛选，后续可以完善
            filtered_symbols.append(symbol)
        
        symbols = filtered_symbols[:max_symbols]
    else:
        symbols = symbols[:max_symbols]

    pool_out = stock_single.get("paths", {}).get("pool_file", "./outputs/orders/stock_single_pool.json")
    ensure_dir(os.path.dirname(pool_out) or ".")
    payload = {
        "ts": datetime.now().isoformat(),
        "mode": "single_stock_hourly",
        "symbol_count": len(symbols),
        "skipped_non_stock_count": len(skipped_non_stock),
        "liquidity_filter_enabled": liquidity_enabled,
        "symbols": symbols,
    }
    with open(pool_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return symbols, pool_out


def main():
    try:
        runtime = load_yaml("config/runtime.yaml")
        stock_single = load_yaml("config/stock_single.yaml")
    except Exception as e:
        print(f"[stock-single-pool] config error: {e}")
        return EXIT_CONFIG_ERROR

    if not runtime.get("enabled", True):
        print("[system] disabled by config/runtime.yaml: enabled=false")
        return EXIT_DISABLED
    if not stock_single.get("enabled", False):
        print("[stock-single] disabled")
        return EXIT_DISABLED

    try:
        symbols, pool_out = build_pool(runtime, stock_single)
    except OSError as e:
        print(f"[stock-single-pool] output error: {e}")
        return EXIT_OUTPUT_ERROR
    except Exception as e:
        print(f"[stock-single-pool] failed: {e}")
        return EXIT_CONFIG_ERROR

    print(f"[stock-single-pool] symbols={len(symbols)} -> {pool_out}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
