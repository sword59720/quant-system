#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from scripts.validate_stock_cpcv import (
    build_fold_indices,
    build_groups,
    ranking_score,
    summarize_folds,
)


class TestCPCVValidation(unittest.TestCase):
    def test_build_groups(self):
        groups = build_groups(100, 5)
        self.assertEqual(len(groups), 5)
        self.assertEqual(groups[0], (0, 20))
        self.assertEqual(groups[-1], (80, 100))

    def test_build_fold_indices_with_embargo(self):
        groups = build_groups(100, 5)
        idx = build_fold_indices(100, groups, (1, 3), embargo_days=2)
        self.assertTrue(min(idx) >= 22)
        self.assertTrue(max(idx) <= 77)
        self.assertGreater(len(idx), 0)

    def test_summarize_folds_gate(self):
        folds = [
            {"metrics": {"excess_ann": 0.04, "sharpe": 0.8, "max_drawdown": -0.15}},
            {"metrics": {"excess_ann": 0.03, "sharpe": 0.7, "max_drawdown": -0.18}},
            {"metrics": {"excess_ann": 0.01, "sharpe": 0.65, "max_drawdown": -0.17}},
        ]
        thresholds = {
            "excess_ann_mean_min": 0.02,
            "excess_ann_median_min": 0.015,
            "excess_ann_worst_min": -0.05,
            "excess_ann_win_rate_min": 0.5,
            "sharpe_median_min": 0.6,
            "max_drawdown_worst_min": -0.22,
        }
        out = summarize_folds(folds, thresholds)
        self.assertTrue(out["gate_passed"])
        self.assertGreater(out["excess_ann_mean"], 0.0)
        self.assertGreater(out["excess_ann_win_rate"], 0.5)

    def test_ranking_score_penalizes_tail(self):
        good = {
            "gate_passed": False,
            "excess_ann_mean": 0.05,
            "excess_ann_median": 0.05,
            "excess_ann_worst": -0.06,
            "excess_ann_win_rate": 0.7,
            "excess_ann_std": 0.05,
            "sharpe_median": 0.9,
            "max_drawdown_worst": -0.12,
            "up_excess_ann_mean": -0.6,
            "down_excess_ann_mean": 0.7,
            "down_excess_ann_worst": -0.08,
        }
        bad = dict(good)
        bad["excess_ann_worst"] = -0.35
        bad["down_excess_ann_worst"] = -0.30

        self.assertGreater(ranking_score(good, 20, 20), ranking_score(bad, 20, 20))


if __name__ == "__main__":
    unittest.main()
