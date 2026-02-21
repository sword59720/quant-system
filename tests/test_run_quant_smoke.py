#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from unittest.mock import patch

import scripts.run_quant as run_quant


class TestRunQuantSmoke(unittest.TestCase):
    def test_run_quant_skip_fetch(self):
        with patch("scripts.run_quant.run_steps", return_value=0) as run_steps_mock:
            with patch("sys.argv", ["run_quant.py", "--skip-fetch", "--skip-reports"]):
                rc = run_quant.main()
        self.assertEqual(rc, 0)
        steps = run_steps_mock.call_args.args[0]
        self.assertEqual(
            [step.command[1] for step in steps],
            [
                "scripts/stock_etf/run_stock_etf.py",
                "scripts/stock_single/run_stock_single.py",
                "scripts/crypto/run_crypto.py",
            ],
        )


if __name__ == "__main__":
    unittest.main()
