#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import tempfile
import unittest

from scripts.report_execution_quality import generate_execution_quality_report


class TestReportExecutionQuality(unittest.TestCase):
    def test_generate_from_order_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orders_dir = os.path.join(tmpdir, "orders")
            reports_dir = os.path.join(tmpdir, "reports")
            os.makedirs(orders_dir, exist_ok=True)
            os.makedirs(reports_dir, exist_ok=True)

            rec = {
                "ts": "2026-02-19T10:00:00",
                "market": "stock",
                "order_results": [
                    {
                        "symbol": "510300",
                        "status": "filled",
                        "latency_ms": 120.0,
                        "slippage_bps": 4.0,
                    },
                    {
                        "symbol": "510300",
                        "status": "submitted",
                        "latency_ms": 180.0,
                        "slippage_bps": -2.0,
                    },
                    {
                        "symbol": "511010",
                        "status": "rejected",
                        "latency_ms": 150.0,
                        "slippage_bps": 1.0,
                    },
                ],
            }
            with open(os.path.join(orders_dir, "execution_record_20260219_100000.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            # Different date should be ignored.
            rec2 = dict(rec)
            rec2["ts"] = "2026-02-18T10:00:00"
            with open(os.path.join(orders_dir, "execution_record_20260218_100000.json"), "w", encoding="utf-8") as f:
                json.dump(rec2, f, ensure_ascii=False, indent=2)

            out_file = os.path.join(reports_dir, "execution_quality_daily.json")
            out = generate_execution_quality_report(
                report_date="2026-02-19",
                output_file=out_file,
                orders_dir=orders_dir,
            )

            self.assertTrue(os.path.exists(out_file))
            self.assertEqual(out["summary"]["orders_total"], 3)
            self.assertAlmostEqual(out["summary"]["success_rate"], 2.0 / 3.0, places=8)
            self.assertAlmostEqual(out["summary"]["fill_rate"], 1.0 / 3.0, places=8)
            self.assertAlmostEqual(out["summary"]["reject_rate"], 1.0 / 3.0, places=8)
            self.assertEqual(out["summary"]["latency_samples"], 3)
            self.assertEqual(out["summary"]["slippage_samples"], 3)
            self.assertIn("stock", out["by_market"])
            self.assertEqual(out["by_market"]["stock"]["orders_total"], 3)
            self.assertGreaterEqual(len(out["by_symbol"]), 2)

    def test_generate_from_legacy_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            orders_dir = os.path.join(tmpdir, "orders")
            reports_dir = os.path.join(tmpdir, "reports")
            os.makedirs(orders_dir, exist_ok=True)
            os.makedirs(reports_dir, exist_ok=True)

            rec = {
                "ts": "2026-02-19T11:00:00",
                "market": "crypto",
                "executed": [
                    {
                        "symbol": "BTC/USDT",
                        "status": "filled",
                    }
                ],
                "failed_details": [
                    {
                        "symbol": "ETH/USDT",
                        "error": "network timeout",
                    }
                ],
            }
            with open(os.path.join(orders_dir, "execution_record_20260219_110000.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)

            out_file = os.path.join(reports_dir, "execution_quality_daily.json")
            out = generate_execution_quality_report(
                report_date="2026-02-19",
                output_file=out_file,
                orders_dir=orders_dir,
            )

            self.assertEqual(out["summary"]["orders_total"], 2)
            self.assertAlmostEqual(out["summary"]["success_rate"], 0.5, places=8)
            self.assertAlmostEqual(out["summary"]["fill_rate"], 0.5, places=8)
            self.assertAlmostEqual(out["summary"]["reject_rate"], 0.5, places=8)
            self.assertEqual(out["summary"]["latency_samples"], 0)
            self.assertEqual(out["summary"]["slippage_samples"], 0)
            self.assertIn("crypto", out["by_market"])
            self.assertEqual(out["by_market"]["crypto"]["orders_total"], 2)


if __name__ == "__main__":
    unittest.main()
