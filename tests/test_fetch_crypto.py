#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import unittest
from unittest import mock

import pandas as pd

from scripts.fetch_crypto_data import (
    fetch_htx_klines,
    has_valid_cache,
    http_get_with_proxy_policy,
    is_dns_resolution_error,
    is_proxy_connection_error,
    should_soft_fail_on_dns,
    to_htx_symbol,
)


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

    def test_is_dns_resolution_error_true_for_name_resolution(self):
        e = RuntimeError("HTTPSConnectionPool(host='api.huobi.pro'): Failed to resolve")
        self.assertTrue(is_dns_resolution_error(e))

    def test_is_dns_resolution_error_false_for_http_status(self):
        e = RuntimeError("404 Client Error: Not Found for url")
        self.assertFalse(is_dns_resolution_error(e))

    def test_is_proxy_connection_error_true_for_proxyerror(self):
        e = RuntimeError("ProxyError: Unable to connect to proxy")
        self.assertTrue(is_proxy_connection_error(e))

    @mock.patch("scripts.fetch_crypto_data.requests.Session")
    @mock.patch("scripts.fetch_crypto_data.requests.get")
    def test_http_get_with_proxy_policy_auto_retry_direct_on_proxy_error(self, mock_get, mock_session_cls):
        mock_get.side_effect = RuntimeError("ProxyError: Unable to connect to proxy")
        mock_session = mock.Mock()
        mock_resp = mock.Mock()
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp
        mock_session_cls.return_value = mock_session

        r = http_get_with_proxy_policy(
            "https://api.huobi.pro/market/history/kline",
            params={"symbol": "btcusdt"},
            timeout=5,
            proxy_mode="auto",
            proxy_auto_bypass_on_error=True,
        )
        self.assertIs(r, mock_resp)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_session.get.call_count, 1)
        self.assertFalse(mock_session.trust_env)
        self.assertEqual(mock_session.close.call_count, 1)

    @mock.patch("scripts.fetch_crypto_data.requests.get")
    def test_http_get_with_proxy_policy_direct_mode(self, mock_get):
        mock_get.return_value = mock.Mock()
        with mock.patch("scripts.fetch_crypto_data.requests.Session") as mock_session_cls:
            mock_session = mock.Mock()
            mock_session.get.return_value = mock.Mock()
            mock_session_cls.return_value = mock_session

            http_get_with_proxy_policy(
                "https://api.huobi.pro/market/history/kline",
                params={"symbol": "ethusdt"},
                timeout=5,
                proxy_mode="direct",
                proxy_auto_bypass_on_error=True,
            )

            self.assertEqual(mock_get.call_count, 0)
            self.assertEqual(mock_session.get.call_count, 1)
            self.assertFalse(mock_session.trust_env)
            self.assertEqual(mock_session.close.call_count, 1)

    def test_should_soft_fail_on_dns(self):
        self.assertTrue(
            should_soft_fail_on_dns(
                total_fail=3,
                dns_fail=3,
                ready_count=2,
                enabled=True,
                min_ready_symbols=2,
            )
        )
        self.assertFalse(
            should_soft_fail_on_dns(
                total_fail=3,
                dns_fail=2,
                ready_count=2,
                enabled=True,
                min_ready_symbols=2,
            )
        )
        self.assertFalse(
            should_soft_fail_on_dns(
                total_fail=1,
                dns_fail=1,
                ready_count=0,
                enabled=True,
                min_ready_symbols=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
