#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键执行 Crypto 交易流程
拉取数据 -> 策略计算 -> 生成指令 -> 执行交易
"""

import sys
import argparse
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.pipeline import resolve_python_executable, run_steps
from core.workflows import build_crypto_trade_steps


def parse_args():
    parser = argparse.ArgumentParser(description="一键执行 Crypto 交易")
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="模拟模式（不实际下单）"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="跳过数据拉取"
    )
    parser.add_argument(
        "--skip-calc",
        action="store_true",
        help="跳过策略计算"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="跳过确认"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    py = resolve_python_executable(prefer_venv=True)

    print("\n" + "="*60)
    print("Crypto 量化交易流程")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {'模拟' if args.dry_run else '实盘'}")
    print("="*60)

    steps = build_crypto_trade_steps(
        py,
        dry_run=args.dry_run,
        skip_fetch=args.skip_fetch,
        skip_calc=args.skip_calc,
        yes=args.yes,
    )
    rc = run_steps(steps)
    if rc == 0:
        print("\n" + "="*60)
        print("Crypto 流程执行完毕")
        print("="*60)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
