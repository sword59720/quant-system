#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
掘金量化(MyQuant)交易适配器
支持通过掘金量化平台实现实盘交易

掘金量化官网: https://www.myquant.cn/
支持券商: 东方财富、广发证券、招商证券、国盛证券等
"""

import os
import json
import time
import logging
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


class MyQuantPaperTrader(BaseTrader):
    """掘金量化模拟交易器"""
    
    def connect(self) -> bool:
        self.logger.info("[MyQuant PAPER] 掘金量化模拟交易模式")
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        return {
            "account_id": "MYQUANT_PAPER",
            "available_cash": 20000.0,
            "total_asset": 20000.0,
        }
    
    def get_positions(self) -> List[Dict]:
        pos_file = self.config.get("position_file", "./outputs/state/stock_positions.json")
        if os.path.exists(pos_file):
            with open(pos_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("positions", [])
        return []
    
    def place_order(self, order: Order) -> Order:
        order.order_id = f"MYQUANT_PAPER_{int(time.time() * 1000)}"
        order.status = OrderStatus.FILLED
        order.filled_amount = order.amount
        order.created_at = datetime.now().isoformat()
        order.updated_at = order.created_at
        self.logger.info(f"[MyQuant PAPER] 模拟下单: {order.side.value} {order.symbol} {order.amount}")
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED


class MyQuantLiveTrader(BaseTrader):
    """
    掘金量化实盘交易器
    
    使用掘金量化终端或API进行实盘交易
    需要先安装掘金量化SDK: pip install gm
    
    参考文档: https://www.myquant.cn/docs/
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.token = config.get("token", "")
        self.account_id = config.get("account_id", "")
        self._gm = None  # 掘金SDK模块
        
    def connect(self) -> bool:
        """连接掘金量化平台"""
        try:
            # 导入掘金量化SDK
            try:
                import gm
                self._gm = gm
            except ImportError:
                raise ImportError(
                    "请先安装掘金量化SDK: pip install gm\n"
                    "或访问: https://www.myquant.cn/docs/"
                )
            
            self.logger.info("[MyQuant LIVE] 正在连接掘金量化平台...")
            
            # 设置token
            if not self.token:
                raise ValueError("掘金量化token未配置")
            
            # 初始化
            self._gm.set_token(self.token)
            
            self.logger.info("[MyQuant LIVE] 连接成功")
            self.is_connected = True
            return True
            
        except Exception as e:
            self.logger.error(f"[MyQuant LIVE] 连接失败: {e}")
            return False
    
    def disconnect(self):
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        try:
            # 使用掘金API获取账户信息
            # 参考: https://www.myquant.cn/docs/api/account/get_cash.html
            cash = self._gm.get_cash()
            return {
                "account_id": self.account_id,
                "available_cash": cash.get("available", 0),
                "total_asset": cash.get("total_asset", 0),
            }
        except Exception as e:
            self.logger.error(f"[MyQuant] 获取账户信息失败: {e}")
            return {"account_id": self.account_id, "available_cash": 0, "total_asset": 0}
    
    def get_positions(self) -> List[Dict]:
        """获取持仓"""
        try:
            # 使用掘金API获取持仓
            # 参考: https://www.myquant.cn/docs/api/account/get_positions.html
            positions = self._gm.get_positions()
            return [
                {
                    "symbol": p.get("symbol", ""),
                    "weight": p.get("market_value", 0) / self.get_account_info().get("total_asset", 1),
                    "market_value": p.get("market_value", 0),
                    "volume": p.get("volume", 0),
                }
                for p in positions
            ]
        except Exception as e:
            self.logger.error(f"[MyQuant] 获取持仓失败: {e}")
            return []
    
    def place_order(self, order: Order) -> Order:
        """
        下单
        
        掘金量化order_volume单位是"股"，不是"手"
        """
        try:
            # 获取当前价格计算股数
            # TODO: 需要获取实时价格
            current_price = self._get_current_price(order.symbol)
            volume = int(order.amount / current_price) if current_price > 0 else 0
            
            # 调整为一手（100股）的整数倍
            volume = (volume // 100) * 100
            
            if volume < 100:
                order.status = OrderStatus.REJECTED
                order.error_msg = "委托数量不足一手"
                return order
            
            # 掘金下单API
            # 参考: https://www.myquant.cn/docs/api/order/order_volume.html
            side = self._gm.OrderSide_Buy if order.side == OrderSide.BUY else self._gm.OrderSide_Sell
            
            result = self._gm.order_volume(
                symbol=order.symbol,
                volume=volume,
                side=side,
                order_type=self._gm.OrderType_Market,  # 市价单
                position_effect=self._gm.PositionEffect_Open if order.side == OrderSide.BUY else self._gm.PositionEffect_Close
            )
            
            order.order_id = result.get("cl_ord_id", "")
            order.status = OrderStatus.SUBMITTED
            order.created_at = datetime.now().isoformat()
            
            self.logger.info(
                f"[MyQuant LIVE] 下单成功: {order.side.value} {order.symbol} "
                f"数量:{volume}股 订单ID:{order.order_id}"
            )
            
            return order
            
        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_msg = str(e)
            self.logger.error(f"[MyQuant LIVE] 下单失败: {e}")
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        try:
            self._gm.cancel_order(order_id)
            return True
        except Exception as e:
            self.logger.error(f"[MyQuant] 撤单失败: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """查询订单状态"""
        try:
            order_info = self._gm.get_order(order_id)
            status_map = {
                "PendingSubmit": OrderStatus.PENDING,
                "Submitted": OrderStatus.SUBMITTED,
                "PartialFilled": OrderStatus.PARTIAL_FILLED,
                "Filled": OrderStatus.FILLED,
                "Cancelled": OrderStatus.CANCELLED,
                "Rejected": OrderStatus.REJECTED,
            }
            return status_map.get(order_info.get("status"), OrderStatus.ERROR)
        except Exception as e:
            self.logger.error(f"[MyQuant] 查询订单状态失败: {e}")
            return OrderStatus.ERROR
    
    def _get_current_price(self, symbol: str) -> float:
        """获取最新价格"""
        try:
            tick = self._gm.get_ticks(symbol)
            return tick.get("price", 0) if tick else 0
        except:
            return 0


def create_trader(config: Dict) -> BaseTrader:
    """工厂函数：创建掘金量化交易器实例"""
    env = config.get("env", "paper")
    if env == "live":
        return MyQuantLiveTrader(config)
    else:
        return MyQuantPaperTrader(config)
