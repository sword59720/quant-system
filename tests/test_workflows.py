#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from core.workflows import (
    build_crypto_trade_steps,
    build_run_all_steps,
    build_stock_trade_steps,
)


class TestWorkflows(unittest.TestCase):
    def test_run_all_etf_default_flow(self):
        steps = build_run_all_steps(
            "/tmp/python",
            stock_etf_enabled=True,
            stock_single_enabled=False,
            crypto_enabled=True,
            skip_fetch=False,
            skip_reports=False,
        )
        got = [s.command[1] for s in steps]
        self.assertEqual(
            got,
            [
                "scripts/stock_etf/fetch_stock_etf_data.py",
                "scripts/crypto/fetch_crypto_data.py",
                "scripts/stock_etf/run_stock_etf.py",
                "scripts/crypto/run_crypto.py",
                "scripts/stock_etf/paper_forward_stock.py",
                "scripts/stock_etf/report_stock_weekly.py",
                "scripts/report_execution_quality.py",
                "scripts/notify_execution_quality_wecom.py",
            ],
        )

    def test_run_all_both_skip_fetch_and_reports(self):
        steps = build_run_all_steps(
            "/tmp/python",
            stock_etf_enabled=True,
            stock_single_enabled=True,
            crypto_enabled=True,
            skip_fetch=True,
            skip_reports=True,
        )
        got = [s.command[1] for s in steps]
        self.assertEqual(
            got,
            [
                "scripts/stock_etf/run_stock_etf.py",
                "scripts/stock_single/run_stock_single.py",
                "scripts/crypto/run_crypto.py",
            ],
        )

    def test_run_all_single_module_only(self):
        steps = build_run_all_steps(
            "/tmp/python",
            stock_etf_enabled=False,
            stock_single_enabled=False,
            crypto_enabled=True,
            skip_fetch=False,
            skip_reports=True,
        )
        self.assertEqual(
            [s.command[1] for s in steps],
            [
                "scripts/crypto/fetch_crypto_data.py",
                "scripts/crypto/run_crypto.py",
            ],
        )

    def test_stock_trade_step_options(self):
        steps = build_stock_trade_steps(
            "/tmp/python",
            dry_run=True,
            skip_fetch=True,
            skip_calc=True,
            yes=True,
        )
        self.assertEqual(steps[0].command, ("/tmp/python", "scripts/stock_etf/generate_trades_stock_etf.py"))
        self.assertEqual(
            steps[1].command,
            ("/tmp/python", "scripts/stock_etf/execute_trades_stock_etf.py", "--dry-run", "--yes"),
        )
        self.assertFalse(steps[2].fatal)
        self.assertFalse(steps[3].fatal)

    def test_crypto_trade_execute_file(self):
        steps = build_crypto_trade_steps("/tmp/python", dry_run=False, yes=False)
        execute_step = [s for s in steps if s.command[1] == "scripts/crypto/execute_trades_crypto.py"][0]
        self.assertEqual(
            execute_step.command,
            (
                "/tmp/python",
                "scripts/crypto/execute_trades_crypto.py",
                "--file",
                "./outputs/orders/crypto_trades.json",
            ),
        )


if __name__ == "__main__":
    unittest.main()
