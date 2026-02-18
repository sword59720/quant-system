#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import unittest

import pandas as pd

from core.signal import liquidity_score, max_drawdown_score, momentum_score, normalize_rank, volatility_score


class TestSignalCore(unittest.TestCase):
    def test_momentum_score(self):
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107])
        score = momentum_score(close, lb_short=2, lb_long=4, w_short=0.6, w_long=0.4)
        expected = (107 / 106 - 1) * 0.6 + (107 / 104 - 1) * 0.4
        self.assertAlmostEqual(score, expected, places=12)

    def test_volatility_and_drawdown(self):
        close = pd.Series([100, 98, 99, 101, 96, 97, 98, 99, 95, 96])
        vol = volatility_score(close, lb=5)
        dd = max_drawdown_score(close, lb=8)
        self.assertIsNotNone(vol)
        self.assertGreater(vol, 0.0)
        self.assertIsNotNone(dd)
        self.assertLessEqual(dd, 0.0)

    def test_liquidity_score(self):
        amount = pd.Series([1_000_000, 1_500_000, 2_000_000, 3_000_000, 4_000_000])
        liq = liquidity_score(amount, lb=3)
        self.assertAlmostEqual(liq, (2_000_000 + 3_000_000 + 4_000_000) / 3, places=6)

    def test_normalize_rank_desc(self):
        out = normalize_rank({"A": 3, "B": 2, "C": 1}, ascending=False)
        self.assertAlmostEqual(out["A"], 1.0)
        self.assertAlmostEqual(out["B"], 0.5)
        self.assertAlmostEqual(out["C"], 0.0)

    def test_normalize_rank_asc(self):
        out = normalize_rank({"A": 3, "B": 2, "C": 1}, ascending=True)
        self.assertAlmostEqual(out["C"], 1.0)
        self.assertAlmostEqual(out["B"], 0.5)
        self.assertAlmostEqual(out["A"], 0.0)
        self.assertTrue(math.isclose(sum(out.values()), 1.5, rel_tol=1e-9))


if __name__ == "__main__":
    unittest.main()
