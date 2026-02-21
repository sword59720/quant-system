#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.exit_codes import EXIT_DISABLED, EXIT_OK
from scripts.stock_single.build_pool import main as build_pool_main
from scripts.stock_single.eval_hourly import main as eval_hourly_main
from scripts.stock_single.risk_check_fast import main as risk_check_fast_main


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-stock pipeline tasks.")
    parser.add_argument(
        "--task",
        choices=["full", "pool", "hourly", "risk"],
        default="full",
        help="full=pool+risk+hourly, pool=build pool, hourly=score to signals, risk=5m fast risk check",
    )
    return parser.parse_args()


def run_task(task: str) -> int:
    if task == "pool":
        return build_pool_main()
    if task == "hourly":
        return eval_hourly_main()
    if task == "risk":
        return risk_check_fast_main()

    # full
    rc = build_pool_main()
    if rc != EXIT_OK:
        return rc
    rc = risk_check_fast_main()
    if rc not in {EXIT_OK, EXIT_DISABLED}:
        return rc
    return eval_hourly_main()


def main():
    args = parse_args()
    return run_task(args.task)


if __name__ == "__main__":
    raise SystemExit(main())
