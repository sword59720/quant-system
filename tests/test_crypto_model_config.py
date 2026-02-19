#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from core.crypto_model import resolve_crypto_model_cfg


class TestCryptoModelConfig(unittest.TestCase):
    def test_select_spot_model(self):
        cfg = {
            "trade": {"market_type": "spot"},
            "signal": {"top_n": 1},
            "defense": {"risk_off_threshold_pct": -3.0},
            "spot_model": {"name": "spot_v1", "signal": {"top_n": 2}},
            "contract_model": {"name": "contract_v1", "signal": {"top_n": 3}},
        }
        out = resolve_crypto_model_cfg(cfg)
        self.assertEqual(out["profile_key"], "spot_model")
        self.assertEqual(out["profile_name"], "spot_v1")
        self.assertEqual(out["signal"]["top_n"], 2)
        self.assertFalse(out["contract_mode"])

    def test_select_contract_model(self):
        cfg = {
            "trade": {"market_type": "swap"},
            "signal": {"top_n": 1},
            "defense": {"risk_off_threshold_pct": -3.0},
            "spot_model": {"name": "spot_v1", "signal": {"top_n": 2}},
            "contract_model": {"name": "contract_v1", "signal": {"top_n": 3}},
        }
        out = resolve_crypto_model_cfg(cfg)
        self.assertEqual(out["profile_key"], "contract_model")
        self.assertEqual(out["profile_name"], "contract_v1")
        self.assertEqual(out["signal"]["top_n"], 3)
        self.assertTrue(out["contract_mode"])


if __name__ == "__main__":
    unittest.main()
