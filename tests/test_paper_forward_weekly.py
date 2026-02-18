#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest

import yaml

from scripts.paper_forward_stock import run_paper_forward
from scripts.report_stock_weekly import build_weekly_summary


class TestPaperForwardWeekly(unittest.TestCase):
    def test_paper_forward_and_weekly_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open("config/runtime.yaml", "r", encoding="utf-8") as f:
                runtime = yaml.safe_load(f)
            with open("config/stock.yaml", "r", encoding="utf-8") as f:
                stock = yaml.safe_load(f)
            with open("config/risk.yaml", "r", encoding="utf-8") as f:
                risk = yaml.safe_load(f)
            runtime["paths"]["output_dir"] = tmpdir
            runtime["paths"]["data_dir"] = "./data"

            summary, history_file, latest_file = run_paper_forward(runtime, stock, risk)
            self.assertTrue(os.path.exists(history_file))
            self.assertTrue(os.path.exists(latest_file))
            self.assertIn("aggregate", summary)
            self.assertIn("latest", summary)
            self.assertGreater(summary["aggregate"]["periods"], 100)

            weekly_summary, _weekly_df = build_weekly_summary(history_file)
            latest_week = weekly_summary["latest_week"]
            self.assertIn("excess_ret_vs_alloc", latest_week)
            self.assertAlmostEqual(
                float(latest_week["excess_ret_vs_alloc"]),
                float(latest_week["strategy_ret"]) - float(latest_week["benchmark_ret_alloc"]),
                places=10,
            )


if __name__ == "__main__":
    unittest.main()
