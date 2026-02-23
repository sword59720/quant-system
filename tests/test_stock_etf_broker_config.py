#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

import yaml

from scripts.stock_etf.execute_trades_stock_etf import load_config


class TestStockEtfBrokerConfig(unittest.TestCase):
    def test_prefers_strategy_account_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir, exist_ok=True)

            runtime = {
                "env": "live",
                "stock_brokers": {"stock_etf": "guotou"},
                "timezone": "Asia/Shanghai",
                "paths": {},
            }
            broker = {
                "guotou": {
                    "platform": "emp",
                    "emp": {"hosting_mode": "hosted"},
                    "accounts": {
                        "stock_etf": {
                            "account_id": "ETF_ACC",
                        },
                        "stock_single": {
                            "account_id": "SINGLE_ACC",
                        },
                    },
                }
            }

            with open(os.path.join(config_dir, "runtime.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(runtime, f, allow_unicode=True, sort_keys=False)
            with open(os.path.join(config_dir, "broker.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(broker, f, allow_unicode=True, sort_keys=False)

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                cfg = load_config()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(cfg.get("broker"), "guotou")
            self.assertEqual(cfg.get("account_id"), "ETF_ACC")
            self.assertEqual(cfg.get("_runtime_account_source"), "guotou.accounts.stock_etf")

    def test_fallback_to_legacy_root_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir, exist_ok=True)

            runtime = {
                "env": "live",
                "broker": "guotou",
                "timezone": "Asia/Shanghai",
                "paths": {},
            }
            broker = {
                "guotou": {
                    "platform": "emp",
                    "emp": {"hosting_mode": "hosted"},
                    "account_id": "ROOT_ACC",
                }
            }

            with open(os.path.join(config_dir, "runtime.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(runtime, f, allow_unicode=True, sort_keys=False)
            with open(os.path.join(config_dir, "broker.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(broker, f, allow_unicode=True, sort_keys=False)

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                cfg = load_config()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(cfg.get("account_id"), "ROOT_ACC")
            self.assertEqual(cfg.get("_runtime_account_source"), "guotou")

    def test_cross_broker_strategy_selection_uses_same_account_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "config")
            os.makedirs(config_dir, exist_ok=True)

            runtime = {
                "env": "live",
                "stock_brokers": {"stock_etf": "myquant", "stock_single": "guotou"},
                "timezone": "Asia/Shanghai",
                "paths": {},
            }
            broker = {
                "myquant": {
                    "platform": "myquant",
                    "api": {"endpoint": "https://www.myquant.cn/api", "timeout": 30},
                    "accounts": {
                        "stock_etf": {
                            "token": "ETF_TOKEN",
                            "account_id": "ETF_MQ_ACC",
                        },
                        "stock_single": {
                            "token": "SINGLE_TOKEN",
                            "account_id": "SINGLE_MQ_ACC",
                        },
                    },
                },
                "guotou": {
                    "platform": "emp",
                    "emp": {"hosting_mode": "hosted"},
                    "accounts": {
                        "stock_single": {"account_id": "SINGLE_GT_ACC"},
                    },
                },
            }

            with open(os.path.join(config_dir, "runtime.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(runtime, f, allow_unicode=True, sort_keys=False)
            with open(os.path.join(config_dir, "broker.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(broker, f, allow_unicode=True, sort_keys=False)

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                cfg = load_config()
            finally:
                os.chdir(old_cwd)

            self.assertEqual(cfg.get("broker"), "myquant")
            self.assertEqual(cfg.get("token"), "ETF_TOKEN")
            self.assertEqual(cfg.get("account_id"), "ETF_MQ_ACC")
            self.assertEqual(cfg.get("_runtime_account_source"), "myquant.accounts.stock_etf")


if __name__ == "__main__":
    unittest.main()
