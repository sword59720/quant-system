#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from adapters.crypto_trader import CryptoLiveTrader, Order, OrderSide


class TestCryptoLiveTrader(unittest.TestCase):
    def _build_trader(self, exchange: str, position_mode: str = "oneway") -> CryptoLiveTrader:
        cfg = {
            "env": "live",
            "exchange": exchange,
            "trade": {
                "market_type": "swap",
                "position_mode": position_mode,
                "margin_mode": "cross",
                "leverage": 1,
            },
        }
        return CryptoLiveTrader(cfg)

    def test_exchange_alias_htx_direct(self):
        trader = self._build_trader("htx_direct")
        self.assertEqual(trader.exchange_id, "htx")

    def test_contract_params_htx_oneway(self):
        trader = self._build_trader("htx_direct", position_mode="oneway")
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            amount=100,
            reduce_only=True,
            position_side="SHORT",
        )
        params = trader._build_contract_order_params(order)
        self.assertTrue(params.get("reduceOnly"))
        self.assertNotIn("positionSide", params)

    def test_contract_params_binance_hedge(self):
        trader = self._build_trader("binanceusdm", position_mode="hedge")
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            amount=100,
            reduce_only=True,
            position_side="SHORT",
        )
        params = trader._build_contract_order_params(order)
        self.assertEqual(params.get("positionSide"), "SHORT")
        self.assertTrue(params.get("reduceOnly"))


if __name__ == "__main__":
    unittest.main()
