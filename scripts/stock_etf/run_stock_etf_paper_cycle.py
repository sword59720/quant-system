#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def run_step(cmd:list[str], title:str)->int:
    print(f'[paper-cycle] {title}')
    print(f"[paper-cycle] $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(ROOT)).returncode

def resolve_python()->str:
    venv_py = ROOT / '.venv' / 'bin' / 'python'
    if venv_py.exists(): return str(venv_py)
    return sys.executable

def main()->int:
    p=argparse.ArgumentParser(description='Run stock_etf paper follow-up cycle')
    p.add_argument('--skip-notify', action='store_true')
    p.add_argument('--notify-dry-run', action='store_true')
    args=p.parse_args()
    py=resolve_python()
    steps=[([py,'scripts/stock_etf/run_stock_etf.py'],'运行 ETF 策略，生成目标仓位'),([py,'scripts/stock_etf/generate_trades_stock_etf.py'],'根据纸面持仓生成调仓指令')]
    if not args.skip_notify:
        notify=[py,'scripts/stock_etf/notify_stock_trades_wecom.py']
        if args.notify_dry_run: notify.append('--dry-run')
        steps.append((notify,'发送飞书调仓通知'))
    for cmd,title in steps:
        rc=run_step(cmd,title)
        if rc!=0:
            print(f'[paper-cycle] failed(exit={rc}): {title}')
            return rc
    print('[paper-cycle] done')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
