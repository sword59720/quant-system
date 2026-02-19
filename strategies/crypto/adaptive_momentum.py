# -*- coding: utf-8 -*-
"""
自适应多因子动量策略 (Adaptive Multi-factor Momentum)
"""
import pandas as pd
import numpy as np

class AdaptiveMomentumStrategy:
    def __init__(self, params):
        self.lookback_periods = params.get('lookback_periods', [20, 60, 120]) # 4h bar
        self.vol_period = params.get('vol_period', 20)
        self.top_n = params.get('top_n', 2)
        self.risk_off_threshold = params.get('risk_off_threshold', -0.05)
        
    def calculate_momentum(self, close_prices):
        """计算综合动量得分"""
        momentum_score = 0
        weights = [0.5, 0.3, 0.2] # 短期权重高
        
        for i, period in enumerate(self.lookback_periods):
            if len(close_prices) > period:
                # 简单收益率
                ret = close_prices.iloc[-1] / close_prices.iloc[-period] - 1
                momentum_score += ret * weights[i]
                
        return momentum_score

    def calculate_volatility(self, close_prices):
        """计算波动率 (标准差)"""
        if len(close_prices) > self.vol_period:
            return close_prices.pct_change().tail(self.vol_period).std()
        return 0.01 # 默认值

    def get_signal(self, data_dict, current_date):
        """
        生成交易信号
        data_dict: {symbol: dataframe}
        """
        scores = []
        
        # 1. 计算每个标的的得分
        for symbol, df in data_dict.items():
            # 截取截止到 current_date 的数据
            hist = df[df['date'] < current_date]
            if hist.empty:
                continue
                
            close = hist['close']
            if len(close) < max(self.lookback_periods):
                continue
                
            mom = self.calculate_momentum(close)
            vol = self.calculate_volatility(close)
            
            # 波动率调整动量 (Risk Parity 思想)
            # 波动率越低，得分越高
            adj_score = mom / (vol + 1e-6)
            
            scores.append({
                'symbol': symbol,
                'score': adj_score,
                'raw_mom': mom,
                'vol': vol
            })
            
        if not scores:
            return []
            
        # 2. 排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 3. 风控检查 (Risk Off)
        # 如果 Top 1 的原始动量都小于阈值，或者大盘(BTC)趋势向下
        top_picks = scores[:self.top_n]
        
        # 简单风控：所有备选标的动量都为负
        if all(x['raw_mom'] < self.risk_off_threshold for x in top_picks):
            return [{'symbol': 'USDT', 'weight': 1.0}]
            
        # 4. 生成目标仓位
        targets = []
        total_inv_vol = sum(1/x['vol'] for x in top_picks)
        
        for item in top_picks:
            # 波动率倒数加权
            weight = (1/item['vol']) / total_inv_vol
            targets.append({
                'symbol': item['symbol'],
                'weight': weight
            })
            
        return targets
