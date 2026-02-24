#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

from scripts.stock_etf.notify_stock_trades_wecom import build_dedup_key, build_trade_message


class TestNotifyStockTradesWecom(unittest.TestCase):
    def test_build_dedup_key_same_orders_same_key(self):
        trades = {
            "ts": "2026-02-24T16:12:00",
            "orders": [
                {"symbol": "159915", "action": "BUY", "amount_quote": 1000, "delta_weight": 0.5},
                {"symbol": "518880", "action": "SELL", "amount_quote": 800, "delta_weight": -0.4},
            ],
        }
        with patch("scripts.stock_etf.notify_stock_trades_wecom.now_in_timezone") as now_mock:
            now_mock.return_value.strftime.return_value = "2026-02-24"
            k1 = build_dedup_key(trades, tz_name="Asia/Shanghai")
            k2 = build_dedup_key(trades, tz_name="Asia/Shanghai")
        self.assertEqual(k1, k2)
        self.assertTrue(k1.startswith("stock_trade_orders:2026-02-24:"))

    def test_build_trade_message_empty_orders(self):
        trades = {
            "ts": "2026-02-24T16:12:00",
            "capital_total": 100000,
            "capital_market": 100000,
            "orders": [],
            "position_file": "/tmp/not_exists.json",
        }
        msg = build_trade_message(trades, stale_position_hours=12)
        self.assertIn("今日无调仓指令", msg)
        self.assertIn("持仓文件缺失", msg)


if __name__ == "__main__":
    unittest.main()
