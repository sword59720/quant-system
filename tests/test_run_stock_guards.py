#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest

from scripts.run_stock import apply_execution_guard, apply_risk_overlay


class TestRunStockGuards(unittest.TestCase):
    def _runtime(self, tmpdir):
        return {
            "paths": {
                "output_dir": tmpdir,
            }
        }

    def _stock_cfg(self, monitor_file, sticky_mode=True):
        return {
            "capital_alloc_pct": 0.7,
            "defensive_symbol": "511010",
            "global_model": {
                "defensive_bypass_single_max": True,
                "execution_guard": {
                    "enabled": True,
                    "min_rebalance_days": 20,
                    "min_turnover": 0.05,
                    "force_rebalance_on_regime_change": True,
                },
                "risk_overlay": {
                    "enabled": True,
                    "monitor_file": monitor_file,
                    "trigger_excess_20d_vs_alloc": -0.02,
                    "trigger_strategy_drawdown": -0.12,
                    "release_excess_20d_vs_alloc": 0.01,
                    "release_strategy_drawdown": -0.06,
                    "min_defense_days": 10,
                    "sticky_mode": sticky_mode,
                },
            },
        }

    def test_execution_guard_min_rebalance_days_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "state")
            os.makedirs(state_dir, exist_ok=True)
            state_file = os.path.join(state_dir, "stock_signal_state.json")
            prev_targets = [{"symbol": "159915", "target_weight": 0.5}, {"symbol": "511010", "target_weight": 0.2}]
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_targets": prev_targets,
                        "last_regime_on": True,
                        "last_rebalance_date": "2026-02-13",
                    },
                    f,
                )

            out = {
                "market_date": "2026-02-13",
                "regime_on": True,
                "signal_reason": "risk_on",
                "targets": [{"symbol": "510300", "target_weight": 0.5}, {"symbol": "511010", "target_weight": 0.2}],
            }
            runtime = self._runtime(tmpdir)
            stock = self._stock_cfg(monitor_file=os.path.join(tmpdir, "monitor.json"))
            guarded = apply_execution_guard(runtime, stock, out)

            self.assertEqual(guarded["execution_guard"]["action"], "hold")
            self.assertEqual(guarded["execution_guard"]["reason"], "min_rebalance_days")
            self.assertEqual(guarded["targets"], prev_targets)

    def test_execution_guard_turnover_hold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "state")
            os.makedirs(state_dir, exist_ok=True)
            state_file = os.path.join(state_dir, "stock_signal_state.json")
            prev_targets = [{"symbol": "159915", "target_weight": 0.5}, {"symbol": "511010", "target_weight": 0.2}]
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_targets": prev_targets,
                        "last_regime_on": True,
                        "last_rebalance_date": "2026-01-01",
                    },
                    f,
                )

            out = {
                "market_date": "2026-02-13",
                "regime_on": True,
                "signal_reason": "risk_on",
                "targets": [{"symbol": "159915", "target_weight": 0.49}, {"symbol": "511010", "target_weight": 0.21}],
            }
            runtime = self._runtime(tmpdir)
            stock = self._stock_cfg(monitor_file=os.path.join(tmpdir, "monitor.json"))
            guarded = apply_execution_guard(runtime, stock, out)

            self.assertEqual(guarded["execution_guard"]["action"], "hold")
            self.assertEqual(guarded["execution_guard"]["reason"], "turnover_below_threshold")
            self.assertEqual(guarded["targets"], prev_targets)

    def test_risk_overlay_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor_file = os.path.join(tmpdir, "monitor.json")
            with open(monitor_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rolling": {"excess_return_20d_vs_alloc": -0.03},
                        "latest": {"strategy_dd": -0.05},
                    },
                    f,
                )

            runtime = self._runtime(tmpdir)
            stock = self._stock_cfg(monitor_file=monitor_file, sticky_mode=True)
            out = {
                "market_date": "2026-02-17",
                "alloc_pct_effective": 0.7,
                "defensive_symbol": "511010",
                "targets": [{"symbol": "159915", "target_weight": 0.5}, {"symbol": "511010", "target_weight": 0.2}],
            }
            x = apply_risk_overlay(runtime, stock, out)
            self.assertTrue(x["risk_overlay"]["triggered"])
            self.assertTrue(x["risk_overlay"]["state_active"])
            self.assertEqual(x["targets"], [{"symbol": "511010", "target_weight": 0.7}])
            self.assertTrue(x["force_rebalance"])
            self.assertEqual(x["force_rebalance_reason"], "risk_overlay_active")

    def test_risk_overlay_release_sticky(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "state")
            os.makedirs(state_dir, exist_ok=True)
            state_file = os.path.join(state_dir, "stock_risk_overlay_state.json")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({"active": True, "activated_date": "2026-01-01"}, f)

            monitor_file = os.path.join(tmpdir, "monitor.json")
            with open(monitor_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rolling": {"excess_return_20d_vs_alloc": 0.02},
                        "latest": {"strategy_dd": -0.03},
                    },
                    f,
                )

            runtime = self._runtime(tmpdir)
            stock = self._stock_cfg(monitor_file=monitor_file, sticky_mode=True)
            out = {
                "market_date": "2026-02-17",
                "alloc_pct_effective": 0.7,
                "defensive_symbol": "511010",
                "targets": [{"symbol": "511010", "target_weight": 0.7}],
            }
            x = apply_risk_overlay(runtime, stock, out)
            self.assertTrue(x["risk_overlay"]["released"])
            self.assertFalse(x["risk_overlay"]["state_active"])
            self.assertTrue(x["force_rebalance"])
            self.assertEqual(x["force_rebalance_reason"], "risk_overlay_released")

    def test_risk_overlay_non_sticky_clears_immediately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = os.path.join(tmpdir, "state")
            os.makedirs(state_dir, exist_ok=True)
            state_file = os.path.join(state_dir, "stock_risk_overlay_state.json")
            with open(state_file, "w", encoding="utf-8") as f:
                json.dump({"active": True, "activated_date": "2026-02-01"}, f)

            monitor_file = os.path.join(tmpdir, "monitor.json")
            with open(monitor_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "rolling": {"excess_return_20d_vs_alloc": 0.00},
                        "latest": {"strategy_dd": -0.05},
                    },
                    f,
                )

            runtime = self._runtime(tmpdir)
            stock = self._stock_cfg(monitor_file=monitor_file, sticky_mode=False)
            out = {
                "market_date": "2026-02-17",
                "alloc_pct_effective": 0.7,
                "defensive_symbol": "511010",
                "targets": [{"symbol": "159915", "target_weight": 0.5}, {"symbol": "511010", "target_weight": 0.2}],
            }
            x = apply_risk_overlay(runtime, stock, out)
            self.assertTrue(x["risk_overlay"]["released"])
            self.assertFalse(x["risk_overlay"]["state_active"])
            self.assertEqual(x["risk_overlay"]["release_reasons"], ["trigger_cleared_non_sticky"])


if __name__ == "__main__":
    unittest.main()
