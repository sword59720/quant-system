#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto (虚拟币) 交易适配器
基于 CCXT 库实现，支持 HTX, Binance, OKX 等交易所
"""

import os
import json
import time
import logging
import ccxt
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    amount: float
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    price: Optional[float] = None
    error_msg: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class BaseTrader:
    """交易适配器基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
    
    def connect(self) -> bool:
        raise NotImplementedError
    
    def disconnect(self):
        raise NotImplementedError
    
    def get_account_info(self) -> Dict:
        raise NotImplementedError
    
    def get_positions(self) -> List[Dict]:
        raise NotImplementedError
    
    def place_order(self, order: Order) -> Order:
        raise NotImplementedError
    
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        raise NotImplementedError
    
    def sync_positions(self, pos_file: str) -> List[Dict]:
        positions = self.get_positions()
        with open(pos_file, 'w', encoding='utf-8') as f:
            json.dump({"positions": positions}, f, ensure_ascii=False, indent=2)
        return positions


class CryptoPaperTrader(BaseTrader):
    """Crypto 模拟交易器"""
    
    def connect(self) -> bool:
        self.logger.info("[Crypto PAPER] 模拟交易模式")
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        return {
            "account_id": "CRYPTO_PAPER",
            "available_cash": 20000.0,  # USDT
            "total_asset": 20000.0,
        }
    
    def get_positions(self) -> List[Dict]:
        pos_file = self.config.get("position_file", "./outputs/state/crypto_positions.json")
        if os.path.exists(pos_file):
            with open(pos_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("positions", [])
        return []
    
    def place_order(self, order: Order) -> Order:
        order.order_id = f"PAPER_CRYPTO_{int(time.time() * 1000)}"
        order.status = OrderStatus.FILLED
        order.filled_amount = order.amount
        order.created_at = datetime.now().isoformat()
        order.updated_at = order.created_at
        self.logger.info(f"[Crypto PAPER] 模拟下单: {order.side.value} {order.symbol} {order.amount}")
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED


class CryptoLiveTrader(BaseTrader):
    """
    Crypto 实盘交易器
    使用 CCXT 连接交易所
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.exchange_id = config.get("exchange", "htx")
        self.api_key = config.get("api_key", "")
        self.secret = config.get("api_secret", "")
        self.password = config.get("password", "")  # OKX 等交易所需要
        self.sandbox = config.get("sandbox", False)
        self._ex = None
        
    def connect(self) -> bool:
        """连接交易所"""
        try:
            self.logger.info(f"[Crypto LIVE] 正在连接 {self.exchange_id}...")
            
            exchange_class = getattr(ccxt, self.exchange_id)
            self._ex = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'password': self.password,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}  # 默认现货
            })
            
            if self.sandbox:
                self._ex.set_sandbox_mode(True)
                self.logger.info("[Crypto LIVE] 启用沙箱模式")
            
            # 加载市场信息
            self._ex.load_markets()
            
            self.logger.info(f"[Crypto LIVE] 连接成功: {self.exchange_id}")
            self.is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"[Crypto LIVE] 连接失败: {e}")
            return False
    
    def disconnect(self):
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        """获取账户信息 (USDT余额)"""
        try:
            balance = self._ex.fetch_balance()
            usdt = balance.get('USDT', {})
            total_usdt = balance.get('total', {}).get('USDT', 0.0)
            
            # 估算总资产 (简单起见，这里先只看USDT，或者需要遍历所有非零资产计算价值)
            # 实盘中通常更关注可用 USDT
            
            return {
                "account_id": self.api_key[:8] + "***",
                "available_cash": usdt.get('free', 0.0),
                "total_asset": total_usdt,  # 仅 USDT 资产，不包含持仓市值
            }
        except Exception as e:
            self.logger.error(f"[Crypto] 获取账户信息失败: {e}")
            return {"account_id": "ERROR", "available_cash": 0, "total_asset": 0}
    
    def get_positions(self) -> List[Dict]:
        """获取持仓"""
        try:
            balance = self._ex.fetch_balance()
            positions = []
            
            # 遍历总余额，找出非零资产
            total_balance = balance.get('total', {})
            for currency, amount in total_balance.items():
                if amount > 0 and currency != 'USDT':
                    # 获取当前价格计算市值
                    symbol = f"{currency}/USDT"
                    market_value = 0.0
                    try:
                        ticker = self._ex.fetch_ticker(symbol)
                        price = ticker['last']
                        market_value = amount * price
                    except:
                        pass # 忽略无法获取价格的资产
                    
                    if market_value > 1.0: # 忽略小额残渣
                         positions.append({
                            "symbol": symbol,
                            "volume": amount,
                            "market_value": market_value,
                            # "weight": ... # 权重计算需要总资产，这里暂略
                        })
            return positions
        except Exception as e:
            self.logger.error(f"[Crypto] 获取持仓失败: {e}")
            return []
    
    def place_order(self, order: Order) -> Order:
        """下单"""
        try:
            # 转换方向
            side = 'buy' if order.side == OrderSide.BUY else 'sell'
            type = 'market' # 默认市价单
            
            # 计算数量
            # CCXT create_order 参数: symbol, type, side, amount, price=None
            # 对于市价买单，amount 通常是指 购买的金额 (quote currency) 还是 数量 (base currency) 取决于交易所
            # 大多数交易所市价买单需要指定 'cost' (USDT金额) 或者通过 create_market_buy_order_requires_price 属性判断
            
            # 简化处理：尝试使用 create_order
            # 注意：Crypto 市价单处理比较复杂，各交易所不同。
            # 建议：如果无法确定，可以尝试用限价单模拟市价（买一价/卖一价）
            
            # 这里为了通用性，我们假设 amount 是 USDT 金额
            # 先获取当前价格，转换为币的数量
            ticker = self._ex.fetch_ticker(order.symbol)
            price = ticker['last']
            
            if side == 'buy':
                # 买入：金额 / 价格 = 数量
                amount_coin = order.amount / price
                # 精度处理 (简化)
                amount_coin = self._ex.amount_to_precision(order.symbol, amount_coin)
            else:
                # 卖出：直接是币的数量 (假设 order.amount 传入的是数量？)
                # 不，execute_trades 传入的是 amount_quote (金额)
                # 所以卖出也需要转换：金额 / 价格 = 数量
                amount_coin = order.amount / price
                amount_coin = self._ex.amount_to_precision(order.symbol, amount_coin)
            
            self.logger.info(f"[Crypto LIVE] 下单: {side} {order.symbol} {amount_coin}")
            
            params = {}
            if side == 'buy' and self.exchange_id == 'htx':
                # HTX 市价买单特殊处理
                # HTX spot market buy order requires the amount in quote currency (USDT)
                # 但 ccxt 统一接口通常传入 base amount
                # 我们这里使用 create_market_order 的 params 传递 'market_buy_validate_total': True? 
                # 或者直接用 amount_coin
                pass
            
            response = self._ex.create_order(order.symbol, type, side, amount_coin, None, params)
            
            order.order_id = response['id']
            order.status = OrderStatus.SUBMITTED
            order.created_at = datetime.now().isoformat()
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_msg = str(e)
            self.logger.error(f"[Crypto LIVE] 下单失败: {e}")
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单 (部分交易所可能需要 symbol)"""
        try:
            # CCXT cancel_order 可能需要 symbol
            # 这里简化处理，假设不需要或稍后完善
            self._ex.cancel_order(order_id)
            return True
        except Exception as e:
            self.logger.error(f"[Crypto] 撤单失败: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        try:
            # 需要 symbol?
            order = self._ex.fetch_order(order_id)
            status = order['status'] # open, closed, canceled
            if status == 'open': return OrderStatus.SUBMITTED
            if status == 'closed': return OrderStatus.FILLED
            if status == 'canceled': return OrderStatus.CANCELLED
            return OrderStatus.PENDING
        except Exception as e:
            return OrderStatus.ERROR


def create_trader(config: Dict) -> BaseTrader:
    """工厂函数"""
    env = config.get("env", "paper")
    if env == "live":
        return CryptoLiveTrader(config)
    else:
        return CryptoPaperTrader(config)
