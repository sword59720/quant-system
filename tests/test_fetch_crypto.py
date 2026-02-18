#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts.fetch_crypto_data import fetch_htx_klines, has_valid_cache, to_htx_symbol


class TestFetchCrypto(unittest.TestCase):
    def test_to_htx_symbol(self):
        self.assertEqual(to_htx_symbol("BTC/USDT"), "btcusdt")
        self.assertEqual(to_htx_symbol("ETH/USDT"), "ethusdt")

    @mock.patch("scripts.fetch_crypto_data.requests.get")
    def test_fetch_htx_klines_parse(self, mock_get):
        mock_resp = mock.Mock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "status": "ok",
            "data": [
                {"id": 1700003600, "open": 2, "high": 3, "low": 1, "close": 2.5, "amount": 20},
                {"id": 1700000000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "amount": 10},
            ],
        }
        mock_get.return_value = mock_resp

        df = fetch_htx_klines("BTC/USDT", timeframe="4h", size=2)
        self.assertEqual(list(df.columns), ["date", "open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 2)
        self.assertTrue(df["date"].iloc[0] < df["date"].iloc[1])
        self.assertAlmostEqual(float(df["volume"].iloc[0]), 10.0)

    def test_has_valid_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/BTC_USDT.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range(end=pd.Timestamp.today(), periods=250, freq="4h"),
                    "close": list(range(250)),
                }
            ).to_csv(path, index=False)
            self.assertTrue(has_valid_cache(path))

    def test_has_invalid_cache_when_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/BTC_USDT.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=15), periods=250, freq="4h"),
                    "close": list(range(250)),
                }
            ).to_csv(path, index=False)
            self.assertFalse(has_valid_cache(path, max_age_days=5))


if __name__ == "__main__":
    unittest.main()
