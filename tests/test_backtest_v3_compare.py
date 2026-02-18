#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from scripts.backtest_v3 import build_stock_compare


class TestBacktestV3Compare(unittest.TestCase):
    def test_compare_both_mode(self):
        stock_result = {
            "mode": "global_momentum_both",
            "production": {
                "periods": 100,
                "annual_return": 0.12,
                "excess_annual_return_vs_alloc": 0.05,
                "max_drawdown": -0.10,
                "sharpe": 0.9,
                "final_nav": 1.8,
            },
            "research": {
                "periods": 100,
                "annual_return": 0.10,
                "excess_annual_return_vs_alloc": 0.03,
                "max_drawdown": -0.14,
                "sharpe": 0.7,
                "final_nav": 1.6,
            },
        }
        out = build_stock_compare(stock_result)
        self.assertTrue(out["available"])
        self.assertTrue(out["periods_equal"])
        self.assertEqual(out["preferred_model"], "production")
        self.assertGreater(out["deltas_production_minus_research"]["annual_return"], 0.0)

    def test_compare_not_available(self):
        out = build_stock_compare({"mode": "global_momentum_production_aligned"})
        self.assertFalse(out["available"])


if __name__ == "__main__":
    unittest.main()
