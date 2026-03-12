#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_UNIVERSE = os.path.join(PROJECT_ROOT, "data", "stock_single", "universe.csv")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "data", "stock_single", "universe_resume.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Create resume universe file from a start symbol.")
    parser.add_argument("--input", default=DEFAULT_UNIVERSE, help="full universe csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="resume universe csv path")
    parser.add_argument("--start-symbol", default="601166.SH", help="start symbol (inclusive)")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    if "symbol" not in df.columns:
        raise RuntimeError(f"missing required column 'symbol': {args.input}")

    symbol_series = df["symbol"].astype(str).str.upper()
    start_symbol = str(args.start_symbol).strip().upper()
    matches = df.index[symbol_series == start_symbol]
    if len(matches) == 0:
        raise RuntimeError(f"start symbol not found: {start_symbol}")

    start_idx = int(matches[0])
    remaining_df = df.iloc[start_idx:].copy()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    remaining_df.to_csv(args.output, index=False)
    print(f"Resume universe saved to {args.output}, contains {len(remaining_df)} symbols.")


if __name__ == "__main__":
    raise SystemExit(main())
