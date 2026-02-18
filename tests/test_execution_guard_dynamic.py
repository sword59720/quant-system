#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from core.execution_guard import resolve_min_turnover


class TestExecutionGuardDynamic(unittest.TestCase):
    def test_disabled_uses_base(self):
        v, meta = resolve_min_turnover(
            base_min_turnover=0.03,
            alloc_pct=0.60,
            guard_cfg={"dynamic_min_turnover": {"enabled": False}},
        )
        self.assertAlmostEqual(v, 0.03, places=8)
        self.assertFalse(meta["enabled"])

    def test_dynamic_scales_with_alloc(self):
        v, meta = resolve_min_turnover(
            base_min_turnover=0.03,
            alloc_pct=0.60,
            guard_cfg={
                "dynamic_min_turnover": {
                    "enabled": True,
                    "alloc_ref": 0.70,
                    "min_multiplier": 0.60,
                    "max_multiplier": 1.40,
                    "floor": 0.018,
                    "ceil": 0.050,
                }
            },
        )
        self.assertTrue(meta["enabled"])
        self.assertAlmostEqual(v, 0.025714285714285714, places=8)
        self.assertAlmostEqual(meta["multiplier"], 0.8571428571428572, places=8)


if __name__ == "__main__":
    unittest.main()
