#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.run_crypto import build_crypto_targets, maybe_auto_execute_live


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
            "trade": {"market_type": "spot", "allow_short": False},
            "signal": {
                "momentum_lookback_bars": [30, 90],
                "momentum_weights": [0.6, 0.4],
                "top_n": 1,
                "short_on_risk_off": False,
                "factor_weights": {"momentum": 0.6, "low_vol": 0.25, "drawdown": 0.15},
            },
            "defense": {"use_usdt_defense": True, "risk_off_threshold_pct": -3.0},
            "spot_model": {
                "name": "spot_test_model",
                "signal": {"top_n": 1, "momentum_lookback_bars": [30, 90]},
                "defense": {"risk_off_threshold_pct": -3.0, "use_usdt_defense": True},
            },
            "contract_model": {
                "name": "contract_test_model",
                "signal": {"top_n": 1, "momentum_lookback_bars": [20, 60], "short_on_risk_off": False},
                "defense": {"risk_off_threshold_pct": -2.0, "use_usdt_defense": False},
            },
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
            self.assertEqual(out["model_profile"], "spot_model")
            self.assertEqual(out["model_name"], "spot_test_model")
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
            self.assertEqual(out["market_type"], "spot")
            self.assertEqual(out["model_profile"], "spot_model")

    def test_build_targets_swap_risk_off_flat(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            crypto["trade"]["market_type"] = "swap"
            crypto["trade"]["allow_short"] = False

            downtrend_1 = [200 - i * 0.9 for i in range(220)]
            downtrend_2 = [220 - i * 1.0 for i in range(220)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", downtrend_1)
            self._write_crypto_csv(tmpdir, "ETH/USDT", downtrend_2)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertTrue(out["risk_off"])
            self.assertTrue(out["contract_mode"])
            self.assertEqual(out["model_profile"], "contract_model")
            self.assertEqual(out["model_name"], "contract_test_model")
            self.assertEqual(out["targets"], [])

    def test_build_targets_swap_risk_off_short(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            crypto["trade"]["market_type"] = "swap"
            crypto["trade"]["allow_short"] = True
            crypto["contract_model"]["signal"]["short_on_risk_off"] = True
            crypto["contract_model"]["signal"]["short_top_n"] = 1

            downtrend_1 = [200 - i * 0.9 for i in range(220)]
            downtrend_2 = [220 - i * 1.0 for i in range(220)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", downtrend_1)
            self._write_crypto_csv(tmpdir, "ETH/USDT", downtrend_2)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertTrue(out["risk_off"])
            self.assertEqual(len(out["targets"]), 1)
            self.assertLess(out["targets"][0]["target_weight"], 0.0)

    def test_build_targets_contract_advanced_rmm_long(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            crypto["trade"]["market_type"] = "swap"
            crypto["trade"]["allow_short"] = False
            crypto["contract_model"] = {
                "name": "contract_rmm_test",
                "signal": {
                    "engine": "advanced_rmm",
                    "top_n": 1,
                    "short_top_n": 1,
                    "momentum_lookback_bars": [30, 90],
                    "momentum_weights": [1.0, 0.0],
                    "ma_window_bars": 90,
                    "vol_window_bars": 20,
                    "risk_managed": {"enabled": True, "target_vol_annual": 0.2, "vol_lookback_bars": 20},
                    "max_exposure_pct": 0.9,
                },
                "defense": {"risk_off_threshold_pct": -3.0, "use_usdt_defense": False},
            }

            uptrend_1 = [100 + i * 1.0 for i in range(260)]
            uptrend_2 = [100 + i * 0.4 for i in range(260)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", uptrend_1)
            self._write_crypto_csv(tmpdir, "ETH/USDT", uptrend_2)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertEqual(out["engine"], "advanced_rmm")
            self.assertFalse(out["risk_off"])
            self.assertGreaterEqual(len(out["targets"]), 1)
            self.assertGreater(out["targets"][0]["target_weight"], 0.0)

    def test_build_targets_contract_advanced_ls_rmm_bi_directional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            crypto["trade"]["market_type"] = "swap"
            crypto["trade"]["allow_short"] = True
            crypto["contract_model"] = {
                "name": "contract_ls_test",
                "signal": {
                    "engine": "advanced_ls_rmm",
                    "top_n": 1,
                    "short_top_n": 1,
                    "momentum_lookback_bars": [30, 90],
                    "momentum_weights": [1.0, 0.0],
                    "ma_window_bars": 90,
                    "vol_window_bars": 20,
                    "momentum_threshold_pct": 0.2,
                    "ls_long_alloc_pct": 0.3,
                    "ls_short_alloc_pct": 0.3,
                    "ls_short_requires_risk_off": False,
                    "risk_managed": {"enabled": True, "target_vol_annual": 0.2, "vol_lookback_bars": 20},
                    "max_exposure_pct": 1.0,
                },
                "defense": {"risk_off_threshold_pct": -3.0, "use_usdt_defense": False},
            }

            uptrend = [100 + i * 1.0 for i in range(260)]
            downtrend = [300 - i * 1.0 for i in range(260)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", uptrend)
            self._write_crypto_csv(tmpdir, "ETH/USDT", downtrend)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertEqual(out["engine"], "advanced_ls_rmm")
            ws = {x["symbol"]: x["target_weight"] for x in out["targets"]}
            self.assertGreater(ws.get("BTC/USDT", 0.0), 0.0)
            self.assertLess(ws.get("ETH/USDT", 0.0), 0.0)

    def test_build_targets_contract_dynamic_budget_risk_off_bias(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime, crypto, risk = self._base_cfg(tmpdir)
            crypto["trade"]["market_type"] = "swap"
            crypto["trade"]["allow_short"] = True
            crypto["contract_model"] = {
                "name": "contract_ls_dynamic_test",
                "signal": {
                    "engine": "advanced_ls_rmm",
                    "top_n": 2,
                    "short_top_n": 2,
                    "momentum_lookback_bars": [30, 90],
                    "momentum_weights": [1.0, 0.0],
                    "ma_window_bars": 90,
                    "vol_window_bars": 20,
                    "momentum_threshold_pct": 0.2,
                    "ls_long_alloc_pct": 0.3,
                    "ls_short_alloc_pct": 0.3,
                    "ls_short_requires_risk_off": True,
                    "ls_dynamic_budget_enabled": True,
                    "ls_regime_momentum_threshold_pct": 0.2,
                    "ls_regime_trend_breadth": 0.5,
                    "ls_regime_up_long_alloc_pct": 0.8,
                    "ls_regime_up_short_alloc_pct": 0.1,
                    "ls_regime_neutral_long_alloc_pct": 0.4,
                    "ls_regime_neutral_short_alloc_pct": 0.4,
                    "ls_regime_down_long_alloc_pct": 0.1,
                    "ls_regime_down_short_alloc_pct": 0.8,
                    "risk_managed": {"enabled": True, "target_vol_annual": 0.2, "vol_lookback_bars": 20},
                    "max_exposure_pct": 1.0,
                },
                "defense": {"risk_off_threshold_pct": -1.0, "use_usdt_defense": False},
            }

            downtrend_1 = [600 - i * 1.2 for i in range(260)]
            downtrend_2 = [700 - i * 1.4 for i in range(260)]
            self._write_crypto_csv(tmpdir, "BTC/USDT", downtrend_1)
            self._write_crypto_csv(tmpdir, "ETH/USDT", downtrend_2)

            out = build_crypto_targets(runtime, crypto, risk)
            self.assertEqual(out["engine"], "advanced_ls_rmm")
            self.assertTrue(out["ls_dynamic_budget_enabled"])
            self.assertEqual(out["regime_state"], "risk_off")
            self.assertAlmostEqual(out["short_budget_used"], 0.8, places=4)
            self.assertGreater(len(out["targets"]), 0)
            self.assertTrue(all(x["target_weight"] < 0.0 for x in out["targets"]))

    def test_auto_execute_skips_when_not_live(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = {
                "env": "paper",
                "total_capital": 20000,
                "paths": {"output_dir": tmpdir},
            }
            crypto = {"execution": {"auto_place_order": True, "min_order_notional": 10}}
            out = maybe_auto_execute_live(runtime, crypto, os.path.join(tmpdir, "target.json"))
            self.assertTrue(out["enabled"])
            self.assertEqual(out["reason"], "env_not_live")
            self.assertFalse(out["executed"])

    def test_auto_execute_live_filters_small_orders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = {
                "env": "live",
                "total_capital": 20000,
                "paths": {"output_dir": tmpdir},
            }
            crypto = {"execution": {"auto_place_order": True, "min_order_notional": 100}}
            target_file = os.path.join(tmpdir, "target.json")

            with patch("scripts.generate_trades.build_market_orders") as mock_build, patch(
                "scripts.execute_trades.execute_trades"
            ) as mock_execute:
                mock_build.return_value = {
                    "orders": [
                        {"symbol": "BTC/USDT", "action": "OPEN_SHORT", "amount_quote": 120},
                        {"symbol": "ETH/USDT", "action": "OPEN_SHORT", "amount_quote": 50},
                    ]
                }
                mock_execute.return_value = True

                out = maybe_auto_execute_live(runtime, crypto, target_file)

            self.assertTrue(out["enabled"])
            self.assertEqual(out["reason"], "ok")
            self.assertTrue(out["executed"])
            self.assertEqual(out["orders_total"], 2)
            self.assertEqual(out["orders_after_filter"], 1)
            self.assertEqual(out["dropped_small_orders"], 1)
            mock_execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
