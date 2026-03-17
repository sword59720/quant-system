import unittest

import pandas as pd

from scripts.stock_etf.run_stock_etf import (
    _build_defensive_weights,
    _compute_defensive_allocation_ratio,
    _compute_dual_budget_multiplier,
    _resolve_defensive_rotation_cfg,
    _resolve_dual_budget_cfg,
)


class TestStockEtfV2Helpers(unittest.TestCase):
    def test_dual_budget_boosts_strong_risk_on_and_cuts_drawdown(self):
        close = pd.Series(
            [100, 101, 103, 105, 107, 110, 113, 116, 118, 121, 124, 126, 129, 132, 135],
            dtype=float,
        )
        cfg = _resolve_dual_budget_cfg(
            {
                "dual_layer_budget": {
                    "enabled": True,
                    "risk_on_base_mult": 1.10,
                    "neutral_base_mult": 0.80,
                    "risk_off_base_mult": 0.35,
                    "strong_trend_threshold": 0.05,
                    "weak_trend_threshold": 0.0,
                    "strong_trend_mult": 1.08,
                    "weak_trend_mult": 0.92,
                    "strong_breadth_threshold": 0.60,
                    "weak_breadth_threshold": 0.40,
                    "strong_breadth_mult": 1.05,
                    "weak_breadth_mult": 0.90,
                    "drawdown_window": 10,
                    "mild_drawdown_trigger": -0.04,
                    "hard_drawdown_trigger": -0.08,
                    "mild_drawdown_mult": 0.92,
                    "hard_drawdown_mult": 0.72,
                    "recovery_window": 5,
                    "recovery_threshold": 0.04,
                    "recovery_mult": 1.03,
                    "total_mult_min": 0.30,
                    "total_mult_max": 1.25,
                }
            }
        )
        mult, meta = _compute_dual_budget_multiplier(
            benchmark_close=close,
            regime_state="risk_on",
            trend_strength=0.07,
            breadth=0.75,
            cfg=cfg,
        )
        self.assertGreater(mult, 1.0)
        self.assertEqual(meta["regime_state"], "risk_on")

        close_dd = pd.Series([100, 102, 101, 99, 95, 92, 90, 89, 88, 87, 86, 85], dtype=float)
        mult_dd, meta_dd = _compute_dual_budget_multiplier(
            benchmark_close=close_dd,
            regime_state="neutral",
            trend_strength=-0.02,
            breadth=0.30,
            cfg=cfg,
        )
        self.assertLess(mult_dd, 1.0)
        self.assertLess(meta_dd["drawdown_mult"], 1.0)

    def test_defensive_rotation_prefers_positive_momentum_symbol(self):
        cfg = _resolve_defensive_rotation_cfg(
            {
                "defensive_rotation": {
                    "enabled": True,
                    "symbols": ["518880", "511010", "511990"],
                    "top_n": 1,
                    "momentum_windows": [3, 5, 8],
                    "momentum_weights": [0.5, 0.3, 0.2],
                    "vol_window": 3,
                    "trend_ma_window": 4,
                    "min_momentum": -0.01,
                    "trend_filter": True,
                }
            },
            "518880",
        )
        closes = {
            "518880": pd.Series([1.00, 1.01, 1.03, 1.05, 1.08, 1.10, 1.13, 1.15, 1.18, 1.20], dtype=float),
            "511010": pd.Series([1.00, 1.00, 0.999, 0.998, 0.999, 1.000, 1.000, 1.001, 1.002, 1.002], dtype=float),
            "511990": pd.Series([1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00], dtype=float),
        }
        weights, meta = _build_defensive_weights(closes, cfg)
        self.assertIn("518880", weights)
        self.assertEqual(meta["reason"], "momentum_rotation")
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_adaptive_defensive_allocation_cuts_ratio_in_strong_market(self):
        benchmark_close = pd.Series(
            [100, 102, 104, 106, 109, 111, 114, 117, 120, 124, 127, 131],
            dtype=float,
        )
        cfg = _resolve_defensive_rotation_cfg(
            {
                "defensive_rotation": {
                    "enabled": True,
                    "symbols": ["518880", "511010"],
                    "adaptive_allocation": {
                        "enabled": True,
                        "base_ratio": 0.12,
                        "min_ratio": 0.05,
                        "max_ratio": 0.40,
                        "neutral_add": 0.08,
                        "trend_ref": 0.08,
                        "trend_weight": 0.10,
                        "breadth_ref": 0.60,
                        "breadth_weight": 0.16,
                        "drawdown_window": 10,
                        "drawdown_ref": -0.08,
                        "drawdown_weight": 0.16,
                        "recovery_window": 5,
                        "recovery_ref": 0.05,
                        "recovery_weight": 0.08,
                        "calm_add": -0.02,
                        "stress_add": 0.06,
                    },
                }
            },
            "518880",
        )
        ratio, meta = _compute_defensive_allocation_ratio(
            benchmark_close=benchmark_close,
            regime_state="risk_on",
            trend_strength=0.10,
            breadth=0.82,
            phase2_state="calm",
            defensive_cfg=cfg,
        )
        self.assertLess(ratio, 0.12)
        self.assertGreaterEqual(ratio, cfg["adaptive_min_ratio"])
        self.assertLess(meta["target_ratio"], 0.12)

    def test_adaptive_defensive_allocation_raises_ratio_in_weak_market(self):
        benchmark_close = pd.Series(
            [100, 101, 99, 97, 95, 94, 92, 90, 89, 88, 87, 86],
            dtype=float,
        )
        cfg = _resolve_defensive_rotation_cfg(
            {
                "defensive_rotation": {
                    "enabled": True,
                    "symbols": ["518880", "511010"],
                    "adaptive_allocation": {
                        "enabled": True,
                        "base_ratio": 0.12,
                        "min_ratio": 0.05,
                        "max_ratio": 0.40,
                        "neutral_add": 0.08,
                        "trend_ref": 0.08,
                        "trend_weight": 0.10,
                        "breadth_ref": 0.60,
                        "breadth_weight": 0.16,
                        "drawdown_window": 10,
                        "drawdown_ref": -0.08,
                        "drawdown_weight": 0.16,
                        "recovery_window": 5,
                        "recovery_ref": 0.05,
                        "recovery_weight": 0.08,
                        "calm_add": -0.02,
                        "stress_add": 0.06,
                    },
                }
            },
            "518880",
        )
        ratio, meta = _compute_defensive_allocation_ratio(
            benchmark_close=benchmark_close,
            regime_state="neutral",
            trend_strength=-0.06,
            breadth=0.25,
            phase2_state="stress",
            defensive_cfg=cfg,
        )
        self.assertGreater(ratio, 0.20)
        self.assertLessEqual(ratio, cfg["adaptive_max_ratio"])
        self.assertGreater(meta["drawdown_term"], 0.0)


if __name__ == "__main__":
    unittest.main()
