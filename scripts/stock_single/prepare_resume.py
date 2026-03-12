import pandas as pd
import sys
import os

# 读取完整的 universe
universe_path = "/home/haojc/.openclaw/workspace/quant-system/data/stock_single/universe.csv"
df = pd.read_csv(universe_path)

# 找到 601166.SH 的位置
# 上次失败是在 601166 (94/120)，这里的 120 可能是 max_symbols_in_pool 限制后的数量
# 我们直接从 601166.SH 开始截取剩下的所有股票
start_idx = df[df['symbol'] == '601166.SH'].index[0]
remaining_df = df.iloc[start_idx:]

# 保存为临时 universe
temp_universe = "/home/haojc/.openclaw/workspace/quant-system/data/stock_single/universe_resume.csv"
remaining_df.to_csv(temp_universe, index=False)

print(f"Resume universe saved to {temp_universe}, contains {len(remaining_df)} symbols.")
