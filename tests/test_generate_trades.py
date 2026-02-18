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


if __name__ == "__main__":
    unittest.main()
