#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import time

EXCHANGES = ["binance", "okx", "bybit"]


def check(name: str):
    try:
        ex_class = getattr(ccxt, name)
        ex = ex_class({"enableRateLimit": True, "timeout": 10000})
        t0 = time.time()
        ex.load_markets()
        dt = (time.time() - t0) * 1000
        return True, f"OK ({dt:.0f} ms, markets={len(ex.markets)})"
    except Exception as e:
        return False, f"FAIL ({e})"


def main():
    print("[exchange-check] start")
    for name in EXCHANGES:
        ok, msg = check(name)
        print(f"- {name}: {msg}")
    print("[exchange-check] done")


if __name__ == "__main__":
    main()
