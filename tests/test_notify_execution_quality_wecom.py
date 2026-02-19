#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from scripts.notify_execution_quality_wecom import (
    build_alert_message,
    evaluate_execution_quality_alert,
)


class TestNotifyExecutionQualityWecom(unittest.TestCase):
    def _alert_cfg(self, min_orders=1):
        return {
            "enabled": True,
            "min_orders": min_orders,
            "thresholds": {
                "success_rate_min": 0.95,
                "fill_rate_min": 0.90,
                "reject_rate_max": 0.05,
                "p95_execution_latency_ms_max": 2000.0,
                "p95_abs_slippage_bps_max": 20.0,
            },
        }

    def _report(self, **summary_overrides):
        summary = {
            "orders_total": 10,
            "success_rate": 0.99,
            "fill_rate": 0.95,
            "reject_rate": 0.01,
            "p95_execution_latency_ms": 1000.0,
            "latency_samples": 10,
            "p95_abs_slippage_bps": 10.0,
            "slippage_samples": 10,
        }
        summary.update(summary_overrides)
        return {
            "date": "2026-02-19",
            "summary": summary,
        }

    def test_no_breach(self):
        out = evaluate_execution_quality_alert(self._report(), self._alert_cfg())
        self.assertTrue(out["min_orders_reached"])
        self.assertFalse(out["should_alert"])
        self.assertEqual(len(out["breaches"]), 0)

    def test_breach_detected(self):
        report = self._report(
            success_rate=0.80,
            fill_rate=0.70,
            reject_rate=0.20,
            p95_execution_latency_ms=2500.0,
            p95_abs_slippage_bps=25.0,
        )
        out = evaluate_execution_quality_alert(report, self._alert_cfg())
        metrics = {x["metric"] for x in out["breaches"]}
        self.assertTrue(out["should_alert"])
        self.assertIn("success_rate", metrics)
        self.assertIn("fill_rate", metrics)
        self.assertIn("reject_rate", metrics)
        self.assertIn("p95_execution_latency_ms", metrics)
        self.assertIn("p95_abs_slippage_bps", metrics)

    def test_min_orders_gate(self):
        report = self._report(orders_total=1, success_rate=0.50)
        out = evaluate_execution_quality_alert(report, self._alert_cfg(min_orders=3))
        self.assertFalse(out["min_orders_reached"])
        self.assertFalse(out["should_alert"])
        self.assertGreaterEqual(len(out["breaches"]), 1)

    def test_build_alert_message_contains_context(self):
        report = self._report(success_rate=0.80)
        out = evaluate_execution_quality_alert(report, self._alert_cfg())
        msg = build_alert_message(out, "./outputs/reports/execution_quality_daily.json")
        self.assertIn("日期:", msg)
        self.assertIn("触发阈值:", msg)
        self.assertIn("成功率", msg)
        self.assertIn("报告:", msg)


if __name__ == "__main__":
    unittest.main()
