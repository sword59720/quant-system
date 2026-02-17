#!/usr/bin/env python3

import subprocess
import sys

py = sys.executable

cmds = [
    [py, "scripts/fetch_stock_data.py"],
    [py, "scripts/fetch_crypto_data.py"],
    [py, "scripts/run_stock.py"],
    [py, "scripts/run_crypto.py"],
]

for c in cmds:
    print(f"[run] {' '.join(c)}")
    r = subprocess.run(c)
    if r.returncode != 0:
        print(f"[error] failed: {' '.join(c)}")
        sys.exit(r.returncode)

print("[run] all done")
