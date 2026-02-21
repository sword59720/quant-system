#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto 高级回测引擎
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from strategies.crypto.adaptive_momentum import AdaptiveMomentumStrategy

class BacktestEngine:
    def __init__(self, data_dir, start_date, end_date, initial_capital=10000.0):
        self.data_dir = data_dir
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.capital = initial_capital
        self.positions = {'USDT': initial_capital} # 初始全USDT
        self.history = []
        self.data_cache = {}
        self.fee_rate = 0.001 # 0.1%
        
    def load_data(self):
        print("Loading data...")
        for f in os.listdir(self.data_dir):
            if f.endswith('.csv'):
                symbol = f.replace('_', '/')
                path = os.path.join(self.data_dir, f)
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                self.data_cache[symbol.replace('.csv', '')] = df
        print(f"Loaded {len(self.data_cache)} symbols")

    def get_price(self, symbol, date):
        if symbol == 'USDT': return 1.0
        df = self.data_cache.get(symbol)
        if df is None: return None
        # 找到最近的一个收盘价
        row = df[df['date'] <= date].iloc[-1:]
        if row.empty: return None
        return float(row['close'].values[0])

    def run(self, strategy):
        print("Running backtest...")
        
        # 生成时间轴 (4小时)
        current = self.start_date
        while current <= self.end_date:
            # 1. 计算总资产
            total_value = 0
            for sym, amt in self.positions.items():
                price = self.get_price(sym, current)
                if price is None: price = 0
                total_value += amt * price
            
            self.history.append({
                'date': current,
                'equity': total_value
            })
            
            # 2. 获取策略信号
            targets = strategy.get_signal(self.data_cache, current)
            
            # 3. 执行调仓 (简化版：假设按收盘价成交)
            if targets:
                # 卖出不在目标中或权重降低的
                # 买入目标
                # 这里使用简化的全仓再平衡逻辑
                
                # 先全部转为 USDT (虚拟)
                cash = total_value * (1 - self.fee_rate) # 卖出费率
                self.positions = {}
                
                for tgt in targets:
                    sym = tgt['symbol']
                    weight = tgt['weight']
                    alloc_cash = cash * weight
                    
                    if sym == 'USDT':
                        self.positions['USDT'] = self.positions.get('USDT', 0) + alloc_cash
                    else:
                        price = self.get_price(sym, current)
                        if price:
                            amount = (alloc_cash * (1 - self.fee_rate)) / price # 买入费率
                            self.positions[sym] = amount
            
            current += timedelta(hours=4)
            
        print("Backtest finished.")
        return pd.DataFrame(self.history)

    def analyze(self, df):
        df['ret'] = df['equity'].pct_change()
        
        total_ret = (df['equity'].iloc[-1] / df['equity'].iloc[0]) - 1
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        ann_ret = (1 + total_ret) ** (365 / days) - 1
        vol = df['ret'].std() * np.sqrt(365 * 6) # 4h bar
        sharpe = (ann_ret - 0.02) / vol
        
        # Max Drawdown
        df['max_equity'] = df['equity'].cummax()
        df['dd'] = df['equity'] / df['max_equity'] - 1
        max_dd = df['dd'].min()
        
        print("\n" + "="*40)
        print("Performance Report")
        print("="*40)
        print(f"Total Return: {total_ret*100:.2f}%")
        print(f"Ann Return:   {ann_ret*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd*100:.2f}%")
        print("="*40)
        
        # Plot
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['equity'], label='Strategy')
            plt.title('Crypto Adaptive Momentum Strategy')
            plt.legend()
            plt.grid(True)
            plt.savefig('./outputs/reports/crypto_backtest.png')
            print("Chart saved to outputs/reports/crypto_backtest.png")
        except:
            pass

if __name__ == "__main__":
    # 配置
    params = {
        'lookback_periods': [20, 60, 120], # 4h bar
        'top_n': 2,
        'risk_off_threshold': 0.0
    }
    
    strategy = AdaptiveMomentumStrategy(params)
    engine = BacktestEngine(
        data_dir="./data/crypto_history",
        start_date="2021-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d")
    )
    
    engine.load_data()
    res = engine.run(strategy)
    engine.analyze(res)
