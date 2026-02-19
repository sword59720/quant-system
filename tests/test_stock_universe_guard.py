#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import unittest
from datetime import date, timedelta

from scripts.run_stock import apply_universe_change_guard, evaluate_backtest_gate


def _write_stock_csv(path, rows=40, start_price=1.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    start = date(2025, 1, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write("date,close\n")
        for i in range(rows):
            d = start + timedelta(days=i)
            px = start_price + i * 0.01
            f.write(f"{d.isoformat()},{px:.4f}\n")


class TestStockUniverseGuard(unittest.TestCase):
    def _runtime(self, tmpdir):
        return {
            "enabled": True,
            "paths": {
                "data_dir": os.path.join(tmpdir, "data"),
                "output_dir": os.path.join(tmpdir, "outputs"),
            },
        }

    def _stock(self, min_ready_ratio=0.9):
        return {
            "mode": "global_momentum",
            "benchmark_symbol": "510300",
            "defensive_symbol": "511010",
            "universe": ["510300", "159915", "511010"],
            "global_model": {
                "momentum_lb": 20,
                "ma_window": 20,
                "vol_window": 10,
                "warmup_min_days": 20,
            },
            "validation": {
                "universe_change_guard": {
                    "enabled": True,
                    "small_change_ratio": 0.10,
                    "medium_change_ratio": 0.50,
                    "min_history_rows": 20,
                    "min_ready_ratio": min_ready_ratio,
                    "run_backtest_on_medium_large": False,
                    "run_cpcv_on_large": False,
                }
            },
        }

    def test_guard_init_then_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            stock = self._stock()
            risk = {}
            data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
            for s in ["510300", "159915", "511010"]:
                _write_stock_csv(os.path.join(data_dir, f"{s}.csv"), rows=40, start_price=1.0)

            r1 = apply_universe_change_guard(runtime, stock, risk)
            self.assertTrue(r1["enabled"])
            self.assertEqual(r1["level"], "init")
            self.assertEqual(r1["status"], "pass")
            self.assertFalse(r1["alert_required"])

            r2 = apply_universe_change_guard(runtime, stock, risk)
            self.assertEqual(r2["level"], "none")
            self.assertEqual(r2["change_count"], 0)
            self.assertEqual(r2["status"], "pass")

    def test_guard_medium_change_with_missing_file_alert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = self._runtime(tmpdir)
            stock = self._stock(min_ready_ratio=0.70)
            risk = {}
            data_dir = os.path.join(runtime["paths"]["data_dir"], "stock")
            for s in ["510300", "159915", "511010"]:
                _write_stock_csv(os.path.join(data_dir, f"{s}.csv"), rows=40, start_price=1.0)

            _ = apply_universe_change_guard(runtime, stock, risk)

            stock2 = self._stock(min_ready_ratio=0.70)
            stock2["universe"] = ["510300", "159915", "511010", "512100"]  # 1/3 -> medium
            r = apply_universe_change_guard(runtime, stock2, risk)

            self.assertEqual(r["level"], "medium")
            self.assertIn("missing_data_files", r["issues"])
            self.assertTrue(r["alert_required"])
            self.assertTrue(os.path.exists(r["report_file"]))

    def test_backtest_gate_fail(self):
        baseline = {
            "annual_return": 0.20,
            "sharpe": 1.20,
            "max_drawdown": -0.10,
        }
        current = {
            "annual_return": 0.15,
            "sharpe": 0.80,
            "max_drawdown": -0.18,
        }
        gate = evaluate_backtest_gate(
            current=current,
            baseline=baseline,
            gate_cfg={
                "annual_return_drop_max": 0.02,
                "sharpe_drop_max": 0.20,
                "max_drawdown_widen_max": 0.03,
            },
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["reason"], "gate_failed")


if __name__ == "__main__":
    unittest.main()
