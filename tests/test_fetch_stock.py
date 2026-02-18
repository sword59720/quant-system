#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest

import pandas as pd

from scripts.fetch_stock_data import has_valid_cache


class TestFetchStock(unittest.TestCase):
    def test_has_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = f"{tmpdir}/510300.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range(end=pd.Timestamp.today(), periods=220, freq="D"),
                    "close": list(range(220)),
                }
            ).to_csv(p, index=False)
            self.assertTrue(has_valid_cache(p))

    def test_has_invalid_cache_when_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = f"{tmpdir}/510300.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=40), periods=220, freq="D"),
                    "close": list(range(220)),
                }
            ).to_csv(p, index=False)
            self.assertFalse(has_valid_cache(p, max_age_days=15))


if __name__ == "__main__":
    unittest.main()
