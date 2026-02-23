#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from core.stock_broker import resolve_runtime_stock_broker, resolve_strategy_account_config


class TestStockBrokerRouting(unittest.TestCase):
    def test_strategy_override_has_highest_priority(self):
        runtime = {
            "broker": "guotou",
            "stock_brokers": {
                "stock_etf": "myquant",
            },
        }
        broker, source = resolve_runtime_stock_broker(runtime, strategy="stock_etf")
        self.assertEqual(broker, "myquant")
        self.assertEqual(source, "stock_brokers.stock_etf")

    def test_broker_dict_supports_default_and_strategy(self):
        runtime = {
            "broker": {
                "default": "guotou",
                "stock_single": "myquant",
            }
        }
        broker, source = resolve_runtime_stock_broker(runtime, strategy="stock_single")
        self.assertEqual(broker, "myquant")
        self.assertEqual(source, "broker.stock_single")

        broker, source = resolve_runtime_stock_broker(runtime, strategy="stock_etf")
        self.assertEqual(broker, "guotou")
        self.assertEqual(source, "broker.default")

    def test_legacy_scalar_broker_is_still_supported(self):
        runtime = {"broker": "guotou"}
        broker, source = resolve_runtime_stock_broker(runtime, strategy="stock_etf")
        self.assertEqual(broker, "guotou")
        self.assertEqual(source, "broker")

    def test_empty_runtime_returns_empty_broker(self):
        broker, source = resolve_runtime_stock_broker({}, strategy="stock_etf")
        self.assertEqual(broker, "")
        self.assertEqual(source, "")

    def test_strategy_can_use_different_brokers(self):
        runtime = {
            "stock_brokers": {
                "stock_etf": "guotou",
                "stock_single": "myquant",
            }
        }
        broker, source = resolve_runtime_stock_broker(runtime, strategy="stock_etf")
        self.assertEqual(broker, "guotou")
        self.assertEqual(source, "stock_brokers.stock_etf")

        broker, source = resolve_runtime_stock_broker(runtime, strategy="stock_single")
        self.assertEqual(broker, "myquant")
        self.assertEqual(source, "stock_brokers.stock_single")

    def test_strategy_account_config_merges_base_and_strategy_override(self):
        broker_full = {
            "guotou": {
                "platform": "emp",
                "trade_mode": "live",
                "order_type": "MARKET",
                "emp": {
                    "hosting_mode": "hosted",
                    "use_alphat": False,
                    "use_act": False,
                },
                "accounts": {
                    "stock_etf": {
                        "account_id": "ETF_ACC",
                        "api_key": "ETF_KEY",
                        "emp": {"use_act": True},
                    }
                },
            }
        }
        cfg, source = resolve_strategy_account_config(broker_full, broker="guotou", strategy="stock_etf")
        self.assertEqual(source, "guotou.accounts.stock_etf")
        self.assertEqual(cfg.get("account_id"), "ETF_ACC")
        self.assertEqual(cfg.get("api_key"), "ETF_KEY")
        self.assertEqual(cfg.get("platform"), "emp")
        self.assertEqual(cfg.get("emp", {}).get("hosting_mode"), "hosted")
        self.assertTrue(cfg.get("emp", {}).get("use_act"))

    def test_strategy_account_config_fallback_to_root(self):
        broker_full = {
            "guotou": {
                "platform": "emp",
                "account_id": "ROOT_ACC",
            }
        }
        cfg, source = resolve_strategy_account_config(broker_full, broker="guotou", strategy="stock_single")
        self.assertEqual(source, "guotou")
        self.assertEqual(cfg.get("account_id"), "ROOT_ACC")


if __name__ == "__main__":
    unittest.main()
