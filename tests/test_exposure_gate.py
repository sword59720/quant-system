#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from core.exposure_gate import evaluate_exposure_gate


class TestExposureGate(unittest.TestCase):
    def test_gate_disabled(self):
        alloc, meta = evaluate_exposure_gate(
            strategy_ret_hist=[0.001] * 200,
            benchmark_ret_alloc_hist=[0.001] * 200,
            base_alloc_pct=0.7,
            gate_cfg={"enabled": False},
        )
        self.assertAlmostEqual(alloc, 0.7, places=8)
        self.assertEqual(meta["stage"], "disabled")

    def test_gate_pass_and_risk(self):
        cfg = {
            "enabled": True,
            "lookback_days": [63, 126],
            "cpcv": {
                "n_groups": 5,
                "test_groups": 2,
                "embargo_days": 1,
                "min_fold_days": 12,
            },
            "thresholds": {
                "warn_worst_min": -0.10,
                "risk_worst_min": -0.14,
            },
            "alloc_caps": {
                "pass": 0.70,
                "warn": 0.60,
                "risk": 0.50,
            },
        }

        alloc_good, meta_good = evaluate_exposure_gate(
            strategy_ret_hist=[0.002] * 220,
            benchmark_ret_alloc_hist=[0.001] * 220,
            base_alloc_pct=0.7,
            gate_cfg=cfg,
        )
        self.assertEqual(meta_good["stage"], "pass")
        self.assertAlmostEqual(alloc_good, 0.7, places=8)

        alloc_bad, meta_bad = evaluate_exposure_gate(
            strategy_ret_hist=[-0.0015] * 220,
            benchmark_ret_alloc_hist=[0.001] * 220,
            base_alloc_pct=0.7,
            gate_cfg=cfg,
        )
        self.assertEqual(meta_bad["stage"], "risk")
        self.assertAlmostEqual(alloc_bad, 0.5, places=8)


if __name__ == "__main__":
    unittest.main()
