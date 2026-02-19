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
from typing import Any, Dict, List, Optional, Set
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
    reduce_only: bool = False
    position_side: Optional[str] = None
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
        self.exchange_id_raw = str(config.get("exchange", "htx"))
        self.exchange_id = self._normalize_exchange_id(self.exchange_id_raw)
        self.api_key = config.get("api_key", "")
        self.secret = config.get("api_secret", "")
        self.password = config.get("password", "")  # OKX 等交易所需要
        self.sandbox = config.get("sandbox", False)
        trade_cfg = config.get("trade", {}) or {}
        market_hint = trade_cfg.get("market_type", config.get("market", "CRYPTO_SPOT"))
        self.market_type = self._normalize_market_type(str(market_hint))
        self.margin_mode = str(trade_cfg.get("margin_mode", "cross")).lower()
        self.leverage = int(trade_cfg.get("leverage", 1))
        self.position_mode = str(trade_cfg.get("position_mode", "oneway")).lower()
        self._ex = None
        self._symbol_cache = {}
        self._derivative_ready_symbols: Set[str] = set()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_exchange_id(raw: str) -> str:
        v = str(raw or "htx").strip().lower()
        alias = {
            "htx_direct": "htx",
            "huobi": "htx",
            "huobipro": "htx",
            "huobi_pro": "htx",
            "okex": "okx",
        }
        return alias.get(v, v)

    @staticmethod
    def _normalize_market_type(raw: str) -> str:
        v = raw.strip().lower()
        if v in {"spot", "crypto_spot"}:
            return "spot"
        if v in {"future", "futures", "crypto_futures"}:
            return "future"
        if v in {"swap", "perp", "perpetual", "crypto_swap"}:
            return "swap"
        if "future" in v:
            return "future"
        if "swap" in v or "perp" in v:
            return "swap"
        return "spot"

    @property
    def is_contract_market(self) -> bool:
        return self.market_type in {"swap", "future"}

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = str(symbol or "").strip()
        if ":" in s:
            return s.split(":", 1)[0]
        return s

    def _resolve_symbol_for_trading(self, symbol: str) -> str:
        key = self._normalize_symbol(symbol)
        if key in self._symbol_cache:
            return self._symbol_cache[key]
        if (self._ex is None) or (not getattr(self._ex, "markets", None)):
            self._symbol_cache[key] = key
            return key

        markets = self._ex.markets
        if not self.is_contract_market:
            if key in markets:
                self._symbol_cache[key] = key
                return key
            self._symbol_cache[key] = key
            return key

        # 合约模式优先匹配同 base/quote 的线性合约（如 BTC/USDT:USDT）
        if key in markets and (markets[key].get("swap") or markets[key].get("future") or markets[key].get("contract")):
            self._symbol_cache[key] = key
            return key

        base = ""
        quote = ""
        if "/" in key:
            base, quote = key.split("/", 1)
            quote = quote.split(":", 1)[0]

        candidates = []
        for m in markets.values():
            if not (m.get("swap") or m.get("future") or m.get("contract")):
                continue
            if base and m.get("base") != base:
                continue
            if quote and m.get("quote") != quote:
                continue
            candidates.append(m)

        if candidates:
            # 优先：linear/usdt 结算 -> swap -> active
            def score(market):
                settle = str(market.get("settle", "")).upper()
                return (
                    1 if settle == quote.upper() else 0,
                    1 if market.get("swap") else 0,
                    1 if market.get("active", True) else 0,
                )

            best = sorted(candidates, key=score, reverse=True)[0]
            resolved = best.get("symbol", key)
            self._symbol_cache[key] = resolved
            return resolved

        self._symbol_cache[key] = key
        return key

    def _to_order_amount(self, notional_quote: float, price: float, market: Dict, symbol: str) -> float:
        if price <= 0:
            raise ValueError(f"invalid price for {symbol}: {price}")
        if notional_quote <= 0:
            raise ValueError(f"invalid order notional for {symbol}: {notional_quote}")

        if market.get("spot"):
            base_amount = notional_quote / price
            amount = self._to_float(self._ex.amount_to_precision(symbol, base_amount))
        elif market.get("contract") or market.get("swap") or market.get("future") or self.is_contract_market:
            # 合约数量通常按“张”提交，张数 = 标的数量 / contractSize
            contract_size = self._to_float(market.get("contractSize"), 1.0)
            if contract_size <= 0:
                contract_size = 1.0
            base_amount = notional_quote / price
            contracts = base_amount / contract_size
            amount = self._to_float(self._ex.amount_to_precision(symbol, contracts))
        else:
            base_amount = notional_quote / price
            amount = self._to_float(self._ex.amount_to_precision(symbol, base_amount))

        if amount <= 0:
            raise ValueError(
                f"order amount rounded to 0 for {symbol}; notional={notional_quote}, price={price}"
            )
        return amount

    @staticmethod
    def _is_param_error(exc: Exception) -> bool:
        msg = str(exc or "").lower()
        keys = [
            "invalid parameter",
            "unknown parameter",
            "unexpected parameter",
            "badrequest",
            "illegal parameter",
            "param",
            "parameter",
        ]
        return any(k in msg for k in keys)

    def _build_contract_order_params(self, order: Order) -> Dict[str, Any]:
        if not self.is_contract_market:
            return {}

        exid = self.exchange_id
        params: Dict[str, Any] = {}
        position_side = str(order.position_side or "").upper()

        if exid == "okx":
            if self.margin_mode in {"cross", "isolated"}:
                params["tdMode"] = self.margin_mode
            if position_side in {"LONG", "SHORT"}:
                params["posSide"] = position_side.lower()
            if order.reduce_only:
                params["reduceOnly"] = True
            return params

        if exid in {"binance", "binanceusdm", "binancecoinm"}:
            if position_side in {"LONG", "SHORT"}:
                params["positionSide"] = position_side
            if order.reduce_only:
                params["reduceOnly"] = True
            return params

        if order.reduce_only:
            params["reduceOnly"] = True
        if position_side in {"LONG", "SHORT"} and self.position_mode in {"hedge", "hedged", "two_way"}:
            params["positionSide"] = position_side
        return params

    def _ensure_derivative_symbol_ready(self, symbol: str):
        if not self.is_contract_market or symbol in self._derivative_ready_symbols:
            return
        try:
            if hasattr(self._ex, "set_position_mode"):
                hedged = self.position_mode in {"hedge", "hedged", "two_way"}
                self._ex.set_position_mode(hedged, symbol)
        except Exception as e:
            self.logger.warning(f"[Crypto LIVE] set_position_mode skip ({symbol}): {e}")
        try:
            if hasattr(self._ex, "set_margin_mode"):
                self._ex.set_margin_mode(self.margin_mode, symbol)
        except Exception as e:
            self.logger.warning(f"[Crypto LIVE] set_margin_mode skip ({symbol}): {e}")
        try:
            if hasattr(self._ex, "set_leverage") and self.leverage > 0:
                self._ex.set_leverage(self.leverage, symbol)
        except Exception as e:
            self.logger.warning(f"[Crypto LIVE] set_leverage skip ({symbol}): {e}")

        self._derivative_ready_symbols.add(symbol)
        
    def connect(self) -> bool:
        """连接交易所"""
        try:
            name_extra = ""
            raw = self.exchange_id_raw.strip().lower()
            if raw and raw != self.exchange_id:
                name_extra = f", raw={raw}"
            self.logger.info(
                f"[Crypto LIVE] 正在连接 {self.exchange_id} (market_type={self.market_type}{name_extra})..."
            )

            if not hasattr(ccxt, self.exchange_id):
                raise ValueError(f"unsupported exchange: {self.exchange_id_raw} -> {self.exchange_id}")
            exchange_class = getattr(ccxt, self.exchange_id)
            options = {"defaultType": self.market_type}
            if self.is_contract_market:
                options.setdefault("defaultSubType", "linear")
            self._ex = exchange_class({
                'apiKey': self.api_key,
                'secret': self.secret,
                'password': self.password,
                'enableRateLimit': True,
                'options': options,
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
            usdt = balance.get("USDT", {})
            total_usdt = self._to_float(balance.get("total", {}).get("USDT"))
            if total_usdt <= 0:
                total_usdt = self._to_float(balance.get("USDT", {}).get("total"))
            
            available = self._to_float(usdt.get("free"))
            if available <= 0:
                available = self._to_float(balance.get("free", {}).get("USDT"))
            
            return {
                "account_id": self.api_key[:8] + "***",
                "available_cash": available,
                "total_asset": total_usdt,
                "market_type": self.market_type,
            }
        except Exception as e:
            self.logger.error(f"[Crypto] 获取账户信息失败: {e}")
            return {"account_id": "ERROR", "available_cash": 0, "total_asset": 0}
    
    def get_positions(self) -> List[Dict]:
        """获取持仓"""
        if self.is_contract_market:
            return self._get_derivative_positions()
        return self._get_spot_positions()

    def _get_spot_positions(self) -> List[Dict]:
        try:
            balance = self._ex.fetch_balance()
            positions = []
            
            total_balance = balance.get("total", {})
            usdt_asset = self._to_float(total_balance.get("USDT"))
            total_asset = usdt_asset

            cached_assets = []
            for currency, amount in total_balance.items():
                amount_val = self._to_float(amount)
                if amount_val > 0 and currency != "USDT":
                    symbol = f"{currency}/USDT"
                    market_value = 0.0
                    try:
                        resolved_symbol = self._resolve_symbol_for_trading(symbol)
                        ticker = self._ex.fetch_ticker(resolved_symbol)
                        price = self._to_float(ticker.get("last"))
                        market_value = amount_val * price
                    except Exception:
                        # 忽略无法获取价格的资产
                        continue
                    
                    if market_value > 1.0:
                        cached_assets.append((symbol, amount_val, market_value))
                        total_asset += market_value

            if total_asset <= 0:
                return []

            if usdt_asset > 1.0:
                positions.append(
                    {
                        "symbol": "USDT",
                        "volume": usdt_asset,
                        "market_value": usdt_asset,
                        "weight": usdt_asset / total_asset,
                    }
                )

            for symbol, amount_val, market_value in cached_assets:
                positions.append(
                    {
                        "symbol": symbol,
                        "volume": amount_val,
                        "market_value": market_value,
                        "weight": market_value / total_asset,
                    }
                )
            return positions
        except Exception as e:
            self.logger.error(f"[Crypto] 获取持仓失败: {e}")
            return []

    def _get_derivative_positions(self) -> List[Dict]:
        try:
            if not hasattr(self._ex, "fetch_positions"):
                self.logger.warning("[Crypto] 交易所不支持 fetch_positions，返回空持仓")
                return []

            account = self.get_account_info()
            total_asset = self._to_float(account.get("total_asset"), 0.0)
            if total_asset <= 0:
                total_asset = 1.0

            raw_positions = self._ex.fetch_positions()
            positions = []
            for p in raw_positions:
                symbol_raw = str(p.get("symbol", "")).strip()
                if not symbol_raw:
                    continue
                symbol = self._normalize_symbol(symbol_raw)

                side = str(p.get("side", "")).lower()
                contracts = self._to_float(p.get("contracts"))
                notional = self._to_float(p.get("notional"))

                if contracts <= 0:
                    amt_raw = self._to_float((p.get("info", {}) or {}).get("positionAmt"))
                    if amt_raw != 0:
                        contracts = abs(amt_raw)
                        side = "long" if amt_raw > 0 else "short"

                if abs(notional) < 1e-8:
                    mark_price = self._to_float(p.get("markPrice"), self._to_float(p.get("entryPrice")))
                    contract_size = self._to_float(p.get("contractSize"), 1.0)
                    if contracts > 0 and mark_price > 0:
                        notional = contracts * contract_size * mark_price

                signed_notional = abs(notional)
                if side == "short" or (side == "" and notional < 0):
                    signed_notional = -signed_notional

                if abs(signed_notional) < 1.0:
                    continue

                positions.append(
                    {
                        "symbol": symbol,
                        "contracts": contracts,
                        "market_value": abs(signed_notional),
                        "notional": signed_notional,
                        "side": "SHORT" if signed_notional < 0 else "LONG",
                        "weight": signed_notional / total_asset,
                    }
                )

            return positions
        except Exception as e:
            self.logger.error(f"[Crypto] 获取合约持仓失败: {e}")
            return []
    
    def place_order(self, order: Order) -> Order:
        """下单"""
        try:
            side = "buy" if order.side == OrderSide.BUY else "sell"
            order_type = "market"
            symbol = self._resolve_symbol_for_trading(order.symbol)

            ticker = self._ex.fetch_ticker(symbol)
            price = self._to_float(ticker.get("last"))
            market = self._ex.market(symbol)
            amount = self._to_order_amount(order.amount, price, market, symbol)

            params: Dict[str, Any] = {}
            if self.is_contract_market:
                self._ensure_derivative_symbol_ready(symbol)
                params = self._build_contract_order_params(order)

            self.logger.info(
                f"[Crypto LIVE] 下单: side={side} symbol={symbol} amount={amount} "
                f"(notional={order.amount:.2f}USDT)"
            )
            
            try:
                response = self._ex.create_order(symbol, order_type, side, amount, None, params)
            except Exception as e:
                if self.is_contract_market and params and self._is_param_error(e):
                    self.logger.warning(
                        f"[Crypto LIVE] 参数兼容重试: exchange={self.exchange_id} symbol={symbol} params={params} err={e}"
                    )
                    response = self._ex.create_order(symbol, order_type, side, amount, None, {})
                else:
                    raise
            
            order.order_id = response.get("id")
            order.status = OrderStatus.SUBMITTED
            order.created_at = datetime.now().isoformat()
            order.updated_at = order.created_at
            order.price = price
            order.filled_amount = float(amount)
            
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
