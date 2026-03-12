#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import akshare as ak

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def fetch_universe():
    print("[universe-gen] fetching index components...")
    
    # 1. 沪深300
    try:
        print("[universe-gen] fetching 000300 (CSI300)...")
        df_300 = ak.index_stock_cons_weight_csindex(symbol="000300")
        df_300["source"] = "000300"
    except Exception as e:
        print(f"[universe-gen] CSI300 fetch failed: {e}")
        df_300 = pd.DataFrame()

    # 2. 中证500
    try:
        print("[universe-gen] fetching 000905 (CSI500)...")
        df_500 = ak.index_stock_cons_weight_csindex(symbol="000905")
        df_500["source"] = "000905"
    except Exception as e:
        print(f"[universe-gen] CSI500 fetch failed: {e}")
        df_500 = pd.DataFrame()

    # 3. 创业板指
    try:
        print("[universe-gen] fetching 399006 (ChiNext)...")
        # 尝试备用接口：新浪接口
        try:
            df_cyb = ak.index_stock_cons_sina(symbol="399006")
            df_cyb["source"] = "399006"
        except Exception:
            # 备用：从巨潮获取 (index_stock_cons)
            df_cyb = ak.index_stock_cons(symbol="399006")
            df_cyb["source"] = "399006"
            
    except Exception as e:
        print(f"[universe-gen] ChiNext fetch failed: {e}")
        df_cyb = pd.DataFrame()

    # 合并
    dfs = [d for d in [df_300, df_500, df_cyb] if not d.empty]
    if not dfs:
        print("[universe-gen] no data fetched!")
        return

    full = pd.concat(dfs, ignore_index=True)
    
    # 打印一下所有列名以便调试
    print("[universe-gen] columns:", full.columns.tolist())
    
    # 1. 先把 symbol 数据统一提取出来
    # 优先级：成分券代码 > symbol > code > 证券代码 > 代码 > 品种代码
    symbol_series = None
    
    if "成分券代码" in full.columns:
        symbol_series = full["成分券代码"]
    elif "symbol" in full.columns:
        symbol_series = full["symbol"]
    elif "code" in full.columns:
        symbol_series = full["code"]
    elif "证券代码" in full.columns:
        symbol_series = full["证券代码"]
    
    if symbol_series is None:
        print("[universe-gen] error: cannot find symbol column!")
        return
        
    # 2. 提取 name 数据
    name_series = None
    if "成分券名称" in full.columns:
        name_series = full["成分券名称"]
    elif "name" in full.columns:
        name_series = full["name"]
    elif "证券名称" in full.columns:
        name_series = full["证券名称"]
        
    # 3. 创建新的干净 DataFrame
    clean_df = pd.DataFrame()
    clean_df["raw_symbol"] = symbol_series
    
    if name_series is not None:
        clean_df["name"] = name_series
    else:
        clean_df["name"] = ""
        
    if "source" in full.columns:
        clean_df["source"] = full["source"]
    else:
        clean_df["source"] = "unknown"
        
    if "所属行业" in full.columns:
        clean_df["industry"] = full["所属行业"]
    
    # 4. 格式化 symbol (例如: 600519 -> 600519.SH)
    def fmt_code(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        # 如果是带小数点的浮点数转成字符串，去掉 .0
        if s.endswith(".0"):
            s = s[:-2]
            
        # 补齐6位
        s = s.zfill(6)
        
        # 如果已经是带后缀的格式，跳过
        if "." in s and ("SH" in s or "SZ" in s or "BJ" in s):
            return s
            
        # 如果是 sina 格式 (sh600519)，去掉前缀
        s_lower = s.lower()
        if s_lower.startswith("sh") or s_lower.startswith("sz"):
             s = s[2:]
             
        if s.startswith("6") or s.startswith("5") or s.startswith("9"):
            return f"{s}.SH"
        else:
            return f"{s}.SZ"

    clean_df["symbol"] = clean_df["raw_symbol"].apply(fmt_code)
    
    # 去除无效 symbol
    clean_df = clean_df[clean_df["symbol"] != ""]
    
    # 去重 (保留第一次出现的)
    clean_df = clean_df.drop_duplicates(subset=["symbol"], keep="first")
    
    # 导出
    out_dir = os.path.join(PROJECT_ROOT, "data", "stock_single")
    ensure_dir(out_dir)
    out_file = os.path.join(out_dir, "universe.csv")
    
    out_cols = ["symbol", "name", "source"]
    if "industry" in clean_df.columns:
        out_cols.append("industry")
        
    clean_df[out_cols].to_csv(out_file, index=False, encoding="utf-8")
    print(f"[universe-gen] done! {len(clean_df)} symbols saved to {out_file}")
    
    # 预览
    print(clean_df.head())

if __name__ == "__main__":
    fetch_universe()
