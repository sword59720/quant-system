#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

import pandas as pd

from scripts.run_crypto import build_crypto_targets


class TestRunCrypto(unittest.TestCase):
    def _base_cfg(self, tmpdir):
        runtime = {
            "enabled": True,
            "env": "paper",
            "total_capital": 20000,
            "paths": {"data_dir": tmpdir, "output_dir": tmpdir},
        }
        crypto = {
            "enabled": True,
            "capital_alloc_pct": 0.3,
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "signal": {
                "momentum_lookback_bars": [30, 90],
                "momentum_weights": [0.6, 0.4],
                "top_n": 1,
                "factor_weights": {"momentum": 0.6, "low_vol": 0.25, "drawdown": 0.15},
            },
            "defense": {"use_usdt_defense": True, "risk_off_threshold_pct": -3.0},
        }
        risk = {"position_limits": {"crypto_single_max_pct": 0.30}}
        return runtime, crypto, risk

    def _write_crypto_csv(self, base_dir, symbol, close_values):
        d = os.path.join(base_dir, "crypto")
        os.makedirs(d, exist_ok=True)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=len(close_values), freq="4h"),
                "close": close_values,
            }
        )
        df.to_csv(os.path.join(d, f"{symbol.replace('/', '_')}.csv"), index=False)

    def test_build_targets_risk_on(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            uptrend_1 = [100 + i * 0.8 for i in range(220)]
            uptrend_2 = [100 + i * 0.4 for i in range(220)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", uptrend_1)
            self._write_crypto_csv(tmpdir, "ETH/USDT", uptrend_2)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertEqual(out["market"], "crypto")
            self.assertFalse(out["risk_off"])
            self.assertEqual(len(out["targets"]), 1)
            self.assertLessEqual(out["targets"][0]["target_weight"], 0.30)

    def test_build_targets_risk_off_to_usdt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            downtrend_1 = [200 - i * 0.9 for i in range(220)]
            downtrend_2 = [220 - i * 1.0 for i in range(220)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", downtrend_1)
            self._write_crypto_csv(tmpdir, "ETH/USDT", downtrend_2)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertTrue(out["risk_off"])
            self.assertEqual(out["targets"], [{"symbol": "USDT", "target_weight": 0.3}])


if __name__ == "__main__":
    unittest.main()
