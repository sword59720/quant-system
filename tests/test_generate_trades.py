#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import tempfile
import unittest

from scripts.generate_trades import build_market_orders


class TestGenerateTrades(unittest.TestCase):
    def test_build_market_orders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = f"{tmpdir}/targets.json"
            pos_file = f"{tmpdir}/positions.json"
            out_file = f"{tmpdir}/trades.json"

            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "capital": 14000,
                        "targets": [
                            {"symbol": "159915", "target_weight": 0.5},
                            {"symbol": "511010", "target_weight": 0.2},
                        ],
                    },
                    f,
                )
            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump({"positions": [{"symbol": "159915", "weight": 0.2}]}, f)

            out = build_market_orders("stock", target_file, pos_file, out_file, total_capital=20000)
            self.assertEqual(out["market"], "stock")
            self.assertEqual(len(out["orders"]), 2)
            self.assertTrue(any(x["symbol"] == "159915" and x["action"] == "BUY" for x in out["orders"]))
            self.assertTrue(any(x["symbol"] == "511010" and x["action"] == "BUY" for x in out["orders"]))

    def test_build_crypto_contract_orders_flip_and_adjust(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target_file = f"{tmpdir}/crypto_targets.json"
            pos_file = f"{tmpdir}/crypto_positions.json"
            out_file = f"{tmpdir}/crypto_trades.json"

            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "capital": 6000,
                        "contract_mode": True,
                        "targets": [
                            {"symbol": "BTC/USDT", "target_weight": -0.3},
                            {"symbol": "ETH/USDT", "target_weight": 0.2},
                        ],
                    },
                    f,
                )
            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "positions": [
                            {"symbol": "BTC/USDT", "weight": 0.1},
                            {"symbol": "ETH/USDT", "weight": -0.1},
                        ]
                    },
                    f,
                )

            out = build_market_orders("crypto", target_file, pos_file, out_file, total_capital=20000)
            actions = {(x["symbol"], x["action"]): x for x in out["orders"]}

            self.assertIn(("BTC/USDT", "CLOSE_LONG"), actions)
            self.assertIn(("BTC/USDT", "OPEN_SHORT"), actions)
            self.assertIn(("ETH/USDT", "CLOSE_SHORT"), actions)
            self.assertIn(("ETH/USDT", "OPEN_LONG"), actions)
            self.assertAlmostEqual(actions[("BTC/USDT", "CLOSE_LONG")]["delta_weight"], -0.1)
            self.assertAlmostEqual(actions[("BTC/USDT", "OPEN_SHORT")]["delta_weight"], -0.3)


if __name__ == "__main__":
    unittest.main()
