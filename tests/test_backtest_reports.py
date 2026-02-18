#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from contextlib import redirect_stdout, redirect_stderr
import io
import unittest

from scripts import backtest_stock_global_model, backtest_stock_model_lab


class TestBacktestReports(unittest.TestCase):
    def test_global_model_backtest_report(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = backtest_stock_global_model.main()
        self.assertEqual(rc, 0)
        path = "./outputs/reports/stock_global_model_report.json"
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
        self.assertIn("out_of_sample", out)
        self.assertIn("strategy", out["out_of_sample"])
        self.assertIn("benchmark", out["out_of_sample"])

    def test_model_lab_report(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            rc = backtest_stock_model_lab.main()
        self.assertEqual(rc, 0)
        path = "./outputs/reports/stock_model_lab_report.json"
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            out = json.load(f)
        self.assertIn("recommendation", out)
        self.assertIn("model_comparison", out)
        self.assertGreaterEqual(len(out["model_comparison"]), 1)


if __name__ == "__main__":
    unittest.main()
