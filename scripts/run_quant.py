#!/usr/bin/env python3

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.exit_codes import EXIT_DISABLED
from core.pipeline import env_true, resolve_python_executable, run_steps
from core.workflows import build_run_all_steps


def parse_bool(x):
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {x} (expected true|false)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all pipelines with per-module switches (stock_etf / stock_single / crypto)."
    )
    parser.add_argument(
        "--stock-etf",
        type=parse_bool,
        default=env_true("QS_RUN_STOCK_ETF", "1"),
        metavar="true|false",
        help="enable ETF stock module",
    )
    parser.add_argument(
        "--stock-single",
        type=parse_bool,
        default=env_true("QS_RUN_STOCK_SINGLE", "1"),
        metavar="true|false",
        help="enable single-stock module",
    )
    parser.add_argument(
        "--crypto",
        type=parse_bool,
        default=env_true("QS_RUN_CRYPTO", "1"),
        metavar="true|false",
        help="enable crypto module",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        default=env_true("QS_SKIP_FETCH"),
        help="skip market data fetching",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        default=env_true("QS_SKIP_REPORTS"),
        help="skip report generation",
    )
    return parser.parse_args()


def build_cmds(
    stock_etf: bool,
    stock_single: bool,
    crypto: bool,
    skip_fetch: bool,
    skip_reports: bool,
    *,
    python_executable=None,
) -> list[list[str]]:
    py = python_executable or resolve_python_executable(prefer_venv=False)
    return [
        list(step.command)
        for step in build_run_all_steps(
            py,
            stock_etf_enabled=stock_etf,
            stock_single_enabled=stock_single,
            crypto_enabled=crypto,
            skip_fetch=skip_fetch,
            skip_reports=skip_reports,
        )
    ]


def main():
    args = parse_args()
    if not (args.stock_etf or args.stock_single or args.crypto):
        print("[error] all modules disabled; enable at least one of --stock-etf/--stock-single/--crypto")
        return 2

    py = resolve_python_executable(prefer_venv=False)
    steps = build_run_all_steps(
        py,
        stock_etf_enabled=args.stock_etf,
        stock_single_enabled=args.stock_single,
        crypto_enabled=args.crypto,
        skip_fetch=args.skip_fetch,
        skip_reports=args.skip_reports,
    )
    print(
        f"[run] stock_etf={args.stock_etf} stock_single={args.stock_single} crypto={args.crypto} "
        f"skip_fetch={args.skip_fetch} skip_reports={args.skip_reports}"
    )
    rc = run_steps(steps, default_allow_exit_codes=(EXIT_DISABLED,))
    if rc == 0:
        print("[run] all done")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
