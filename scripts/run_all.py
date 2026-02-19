#!/usr/bin/env python3

import os
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import EXIT_DISABLED

py = sys.executable

skip_fetch = os.getenv("QS_SKIP_FETCH", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
skip_reports = os.getenv("QS_SKIP_REPORTS", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}

cmds = []
if not skip_fetch:
    cmds.extend(
        [
            [py, "scripts/fetch_stock_data.py"],
            [py, "scripts/fetch_crypto_data.py"],
        ]
    )
cmds.extend(
    [
        [py, "scripts/run_stock.py"],
        [py, "scripts/run_crypto.py"],
    ]
)
if not skip_reports:
    cmds.extend(
        [
            [py, "scripts/paper_forward_stock.py"],
            [py, "scripts/report_stock_weekly.py"],
            [py, "scripts/report_execution_quality.py"],
            [py, "scripts/notify_execution_quality_wecom.py"],
        ]
    )

for c in cmds:
    print(f"[run] {' '.join(c)}")
    r = subprocess.run(c)
    if r.returncode == EXIT_DISABLED:
        print(f"[run] skipped(disabled): {' '.join(c)}")
        continue
    if r.returncode != 0:
        print(f"[error] failed: {' '.join(c)}")
        sys.exit(r.returncode)

print("[run] all done")
