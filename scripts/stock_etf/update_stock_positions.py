#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json, os
from datetime import datetime
from typing import Any

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or str(x).strip() == '':
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def read_positions_from_csv(path: str) -> list[dict]:
    rows=[]
    with open(path,'r',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            symbol=str(row.get('symbol','')).strip()
            if not symbol: continue
            item={'symbol':symbol,'weight':round(safe_float(row.get('weight'),0.0),6)}
            q=row.get('quantity')
            if q not in (None,''):
                try: item['quantity']=int(float(q))
                except Exception: item['quantity']=str(q).strip()
            rows.append(item)
    return rows

def normalize_positions(items:list[dict]) -> list[dict]:
    merged={}
    for item in items:
        symbol=str(item.get('symbol','')).strip()
        if not symbol: continue
        merged.setdefault(symbol, {'symbol':symbol,'weight':0.0})
        merged[symbol]['weight'] += safe_float(item.get('weight'),0.0)
        if item.get('quantity') is not None:
            merged[symbol]['quantity']=item.get('quantity')
    out=[]
    for symbol in sorted(merged):
        row=merged[symbol]
        row['weight']=round(float(row.get('weight',0.0)),6)
        if abs(row['weight'])<1e-8: continue
        out.append(row)
    return out

def build_snapshot(positions:list[dict], source:str) -> dict:
    total_weight=round(sum(float(x.get('weight',0.0) or 0.0) for x in positions),6)
    cash_weight=round(max(0.0,1.0-total_weight),6)
    return {'ts':datetime.now().isoformat(),'source':source,'positions':positions,'summary':{'position_count':len(positions),'total_weight':total_weight,'cash_weight_est':cash_weight}}

def parse_set_items(raw_items:list[str]) -> list[dict]:
    rows=[]
    for raw in raw_items:
        parts=str(raw or '').strip().split(':')
        if len(parts)<2: raise ValueError(f'invalid --set format: {raw}')
        item={'symbol':parts[0].strip(),'weight':round(safe_float(parts[1],0.0),6)}
        if len(parts)>=3 and parts[2].strip()!='':
            try: item['quantity']=int(float(parts[2]))
            except Exception: item['quantity']=parts[2].strip()
        rows.append(item)
    return rows

def main() -> int:
    p=argparse.ArgumentParser(description='更新 ETF 纸面持仓快照')
    p.add_argument('--file', default='./outputs/state/stock_positions.json')
    p.add_argument('--from-csv', default='')
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--show', action='store_true')
    args=p.parse_args()
    if args.show:
        print(json.dumps(load_json(args.file, default={}), ensure_ascii=False, indent=2)); return 0
    if args.from_csv:
        rows=read_positions_from_csv(args.from_csv); source=f'csv:{args.from_csv}'
    elif args.set:
        rows=parse_set_items(args.set); source='cli:set'
    else:
        raise SystemExit('必须提供 --from-csv 或至少一个 --set')
    snapshot=build_snapshot(normalize_positions(rows), source)
    ensure_dir(os.path.dirname(args.file))
    with open(args.file,'w',encoding='utf-8') as f: json.dump(snapshot,f,ensure_ascii=False,indent=2)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2)); return 0

if __name__ == '__main__':
    raise SystemExit(main())
