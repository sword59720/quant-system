#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
国投证券交易适配器
支持策略编程与算法交易API
"""

import os
import json
import time
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
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


class BaseTrader(ABC):
    """交易适配器基类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """连接交易服务器"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """获取当前持仓"""
        pass
    
    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """下单"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """查询订单状态"""
        pass
    
    def sync_positions(self, pos_file: str) -> List[Dict]:
        """同步持仓到本地文件"""
        positions = self.get_positions()
        with open(pos_file, 'w', encoding='utf-8') as f:
            json.dump({"positions": positions}, f, ensure_ascii=False, indent=2)
        return positions


class GuotouPaperTrader(BaseTrader):
    """
    国投证券模拟交易器
    用于测试，不执行真实交易
    """
    
    def connect(self) -> bool:
        self.logger.info("[PAPER] 模拟交易模式 - 不执行真实下单")
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        total_capital = float(self.config.get("total_capital", 20000.0) or 20000.0)
        return {
            "account_id": "PAPER_ACCOUNT",
            "available_cash": total_capital,
            "total_asset": total_capital,
        }
    
    def get_positions(self) -> List[Dict]:
        # 从本地文件读取模拟持仓
        pos_file = self.config.get("position_file", "./outputs/state/stock_positions.json")
        if os.path.exists(pos_file):
            with open(pos_file, 'r', encoding='utf-8') as f:
                return json.load(f).get("positions", [])
        return []
    
    def place_order(self, order: Order) -> Order:
        order.order_id = f"PAPER_{int(time.time() * 1000)}"
        order.status = OrderStatus.FILLED
        order.filled_amount = order.amount
        order.created_at = datetime.now().isoformat()
        order.updated_at = order.created_at
        self.logger.info(f"[PAPER] 模拟下单: {order.side.value} {order.symbol} {order.amount}")
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.FILLED


class GuotouEMPTrader(BaseTrader):
    """
    国投证券 EMP 策略托管平台交易器
    
    EMP (Strategy Hosting Platform) 是国投证券提供的策略托管服务：
    - AlphaT策略: T0算法（日内回转/自动做T）
    - ACT策略: 算法交易（拆单/智能执行）
    
    使用方式：
    1. 信号同步模式: 本地生成信号，推送到EMP执行
    2. 策略托管模式: 将策略部署到EMP服务器运行
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.account_id = config.get("account_id", "")
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.api_endpoint = config.get("api_endpoint", "")
        self.emp_config = config.get("emp", {})
        self.trade_client = None
        
        # EMP 平台特性
        self.use_alphat = self.emp_config.get("use_alphat", False)  # T0算法
        self.use_act = self.emp_config.get("use_act", False)        # 算法交易
        self.hosting_mode = self.emp_config.get("hosting_mode", "signal")  # signal | hosted
        
    def connect(self) -> bool:
        """
        连接国投证券EMP平台
        
        根据 hosting_mode 不同，连接方式不同：
        - signal: 连接信号同步API
        - hosted: 验证托管环境
        """
        try:
            self.logger.info("[EMP] 正在连接国投证券策略托管平台...")
            self.logger.info(f"[EMP] 模式: {self.hosting_mode}")
            self.logger.info(f"[EMP] AlphaT(T0): {self.use_alphat}")
            self.logger.info(f"[EMP] ACT(算法): {self.use_act}")
            
            if self.hosting_mode == "hosted":
                # 策略托管在EMP服务器上运行
                # 只需验证环境
                self.logger.info("[EMP] 托管模式 - 策略在EMP服务器运行")
                self.is_connected = True
                return True
            
            else:
                # 信号同步模式 - 本地策略，远程下单
                # TODO: 替换为EMP实际的API连接代码
                # 可能的接口形式：
                # 1. REST API: 通过HTTPS发送交易信号
                # 2. SDK: 国投提供的Python SDK
                # 3. 消息队列: 通过MQ发送信号
                
                raise NotImplementedError(
                    "请替换为国投证券EMP平台实际的API连接代码。\n"
                    "联系客户经理获取：\n"
                    "- EMP平台API文档\n"
                    "- Python SDK（如有）\n"
                    "- 接入示例代码\n"
                    "\n通常EMP接入方式：\n"
                    "1. 信号同步: 通过REST API推送目标仓位\n"
                    "2. 策略托管: 将quant-system部署到EMP服务器"
                )
            
        except Exception as e:
            self.logger.error(f"[EMP] 连接失败: {e}")
            return False
    
    def disconnect(self):
        if self.trade_client:
            pass
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        """获取账户信息"""
        # TODO: 替换为EMP实际API调用
        return {
            "account_id": self.account_id,
            "available_cash": 0.0,
            "total_asset": 0.0,
        }
    
    def get_positions(self) -> List[Dict]:
        """获取持仓"""
        # TODO: 替换为EMP实际API调用
        return []
    
    def place_order(self, order: Order) -> Order:
        """
        下单到EMP平台
        
        根据配置使用不同算法：
        - AlphaT: T0算法（适合已有底仓做T）
        - ACT: 算法交易（适合大单拆单执行）
        """
        try:
            order.created_at = datetime.now().isoformat()
            
            if self.use_alphat:
                # 使用 AlphaT T0算法
                # T0算法需要已有底仓，做日内回转
                order.order_id = f"ALPHAT_{int(time.time() * 1000)}"
                self.logger.info(
                    f"[EMP/AlphaT] T0算法下单: {order.side.value} {order.symbol} "
                    f"金额:¥{order.amount:.2f}"
                )
                # TODO: 调用AlphaT API
                # alphat_client.submit_order(
                #     symbol=order.symbol,
                #     side=order.side.value,
                #     amount=order.amount,
                #     algorithm="T0"
                # )
                
            elif self.use_act:
                # 使用 ACT 算法交易
                # 智能拆单、TWAP/VWAP执行
                order.order_id = f"ACT_{int(time.time() * 1000)}"
                self.logger.info(
                    f"[EMP/ACT] 算法交易下单: {order.side.value} {order.symbol} "
                    f"金额:¥{order.amount:.2f}"
                )
                # TODO: 调用ACT API
                # act_client.submit_algo_order(
                #     symbol=order.symbol,
                #     side=order.side.value,
                #     amount=order.amount,
                #     algo_type="TWAP"  # 或 VWAP
                # )
                
            else:
                # 普通下单
                order.order_id = f"EMP_{int(time.time() * 1000)}"
                self.logger.info(
                    f"[EMP] 普通下单: {order.side.value} {order.symbol} "
                    f"金额:¥{order.amount:.2f}"
                )
            
            order.status = OrderStatus.SUBMITTED
            return order
            
        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_msg = str(e)
            self.logger.error(f"[EMP] 下单失败: {e}")
            return order
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        # TODO: 替换为EMP实际API调用
        return True
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        """查询订单状态"""
        # TODO: 替换为EMP实际API调用
        return OrderStatus.SUBMITTED


class GuotouLiveTrader(BaseTrader):
    """
    国投证券传统API交易器（备用）
    适用于直接API接入（非EMP模式）
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.account_id = config.get("account_id", "")
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.api_endpoint = config.get("api_endpoint", "")
        self.trade_client = None
        
    def connect(self) -> bool:
        """连接传统交易API"""
        self.logger.info("[LIVE] 传统API模式（备用）")
        self.is_connected = True
        return True
    
    def disconnect(self):
        self.is_connected = False
    
    def get_account_info(self) -> Dict:
        return {"account_id": self.account_id, "available_cash": 0.0, "total_asset": 0.0}
    
    def get_positions(self) -> List[Dict]:
        return []
    
    def place_order(self, order: Order) -> Order:
        order.order_id = f"LIVE_{int(time.time() * 1000)}"
        order.status = OrderStatus.ERROR
        order.error_msg = "传统API模式未实现，请使用EMP模式"
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        return False
    
    def get_order_status(self, order_id: str) -> OrderStatus:
        return OrderStatus.ERROR


def create_trader(config: Dict) -> BaseTrader:
    """工厂函数：创建交易器实例"""
    env = config.get("env", "paper")
    platform = config.get("platform", "emp")  # emp | traditional
    
    if env == "live":
        if platform == "emp":
            return GuotouEMPTrader(config)
        else:
            return GuotouLiveTrader(config)
    else:
        return GuotouPaperTrader(config)
