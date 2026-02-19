#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from scripts.execute_trades import build_order_for_market
from adapters.crypto_trader import OrderSide as CryptoOrderSide
from adapters.guotou_trader import OrderSide as StockOrderSide


class TestExecuteTradesOrderBuild(unittest.TestCase):
    def test_build_stock_order(self):
        order = build_order_for_market(
            "stock",
            {"symbol": "510300", "action": "BUY", "amount_quote": 1000},
        )
        self.assertEqual(order.side, StockOrderSide.BUY)
        self.assertEqual(order.amount, 1000)

    def test_build_crypto_order_basic(self):
        order = build_order_for_market(
            "crypto",
            {"symbol": "BTC/USDT", "action": "SELL", "amount_quote": 500},
        )
        self.assertEqual(order.side, CryptoOrderSide.SELL)
        self.assertFalse(order.reduce_only)
        self.assertIsNone(order.position_side)

    def test_build_crypto_order_close_short(self):
        order = build_order_for_market(
            "crypto",
            {"symbol": "BTC/USDT", "action": "CLOSE_SHORT", "amount_quote": 200},
        )
        self.assertEqual(order.side, CryptoOrderSide.BUY)
        self.assertTrue(order.reduce_only)
        self.assertEqual(order.position_side, "SHORT")

    def test_build_crypto_order_invalid_action(self):
        with self.assertRaises(ValueError):
            build_order_for_market(
                "crypto",
                {"symbol": "BTC/USDT", "action": "UNKNOWN", "amount_quote": 100},
            )


if __name__ == "__main__":
    unittest.main()
