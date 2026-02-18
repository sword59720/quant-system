#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import unittest


class TestRunAllSmoke(unittest.TestCase):
    def test_run_all_skip_fetch(self):
        env = os.environ.copy()
        env["QS_SKIP_FETCH"] = "1"
        env["QS_SKIP_REPORTS"] = "1"
        p = subprocess.run([sys.executable, "scripts/run_all.py"], env=env, check=False, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
