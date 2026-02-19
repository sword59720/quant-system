#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键执行交易流程
整合策略计算 + 生成指令 + 执行交易
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime

def run_command(cmd: list, description: str) -> bool:
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"❌ {description} 失败")
        return False
    print(f"✅ {description} 完成")
    return True

def main():
    parser = argparse.ArgumentParser(description="一键执行交易")
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
    
    args = parser.parse_args()
    
    venv_python = "./.venv/bin/python"
    
    print("\n" + "="*60)
    print("🚀 quant-system 交易执行流程")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {'模拟' if args.dry_run else '实盘'}")
    print("="*60)
    
    # Step 1: 拉取数据
    if not args.skip_fetch:
        if not run_command(
            [venv_python, "scripts/fetch_stock_data.py"],
            "Step 1/4: 拉取股票数据"
        ):
            return False
    else:
        print("\n⏭️ 跳过数据拉取")
    
    # Step 2: 策略计算
    if not args.skip_calc:
        if not run_command(
            [venv_python, "scripts/run_stock.py"],
            "Step 2/4: 计算策略信号"
        ):
            return False
    else:
        print("\n⏭️ 跳过策略计算")
    
    # Step 3: 生成交易指令
    if not run_command(
        [venv_python, "scripts/generate_trades.py"],
        "Step 3/4: 生成交易指令"
    ):
        return False
    
    # Step 4: 执行交易
    execute_args = [venv_python, "scripts/execute_trades.py"]
    if args.dry_run:
        execute_args.append("--dry-run")
    if args.yes:
        execute_args.append("--yes")
    
    if not run_command(
        execute_args,
        f"Step 4/4: 执行交易 ({'模拟' if args.dry_run else '实盘'})"
    ):
        return False

    run_command(
        [venv_python, "scripts/report_execution_quality.py"],
        "附加步骤: 生成成交质量日报"
    )
    run_command(
        [venv_python, "scripts/notify_execution_quality_wecom.py"],
        "附加步骤: 成交质量阈值告警检查"
    )
    
    print("\n" + "="*60)
    print("✨ 全部流程执行完毕")
    print("="*60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
