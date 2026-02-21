#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import shutil
from datetime import datetime

import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compute_file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_history_frame(history_file):
    df = pd.read_csv(history_file)
    if "date" not in df.columns:
        raise RuntimeError(f"history missing date column: {history_file}")
    df["date"] = pd.to_datetime(df["date"])
    if "next_date" in df.columns:
        df["next_date"] = pd.to_datetime(df["next_date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def build_history_fingerprint(history_df, history_file=None):
    out = {
        "rows": int(len(history_df)),
        "date_start": None,
        "date_end": None,
        "next_date_end": None,
    }
    if len(history_df) > 0:
        out["date_start"] = history_df["date"].iloc[0].date().isoformat()
        out["date_end"] = history_df["date"].iloc[-1].date().isoformat()
        if "next_date" in history_df.columns:
            out["next_date_end"] = history_df["next_date"].iloc[-1].date().isoformat()

    if history_file and os.path.exists(history_file):
        out["history_file"] = history_file
        out["history_size_bytes"] = int(os.path.getsize(history_file))
        out["history_sha256"] = compute_file_sha256(history_file)
    return out


def get_snapshot_paths(runtime, snapshot_name="stock_backtest_snapshot"):
    report_dir = os.path.join(runtime["paths"]["output_dir"], "reports")
    snapshot_file = os.path.join(report_dir, f"{snapshot_name}.csv")
    meta_file = os.path.join(report_dir, f"{snapshot_name}_meta.json")
    return report_dir, snapshot_file, meta_file


def save_history_snapshot(history_file, runtime, snapshot_name="stock_backtest_snapshot", source="unknown"):
    report_dir, snapshot_file, meta_file = get_snapshot_paths(runtime, snapshot_name=snapshot_name)
    ensure_dir(report_dir)
    shutil.copy2(history_file, snapshot_file)
    history_df = load_history_frame(snapshot_file)
    fingerprint = build_history_fingerprint(history_df, history_file=snapshot_file)
    meta = {
        "ts": datetime.now().isoformat(),
        "source": source,
        "snapshot_name": snapshot_name,
        "snapshot_file": snapshot_file,
        "meta_file": meta_file,
        "fingerprint": fingerprint,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return snapshot_file, meta_file, meta, history_df


def load_history_snapshot(runtime, snapshot_name="stock_backtest_snapshot"):
    _report_dir, snapshot_file, meta_file = get_snapshot_paths(runtime, snapshot_name=snapshot_name)
    if not os.path.exists(snapshot_file):
        return None

    history_df = load_history_frame(snapshot_file)
    meta = None
    if os.path.exists(meta_file):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None

    if not isinstance(meta, dict):
        meta = {
            "ts": datetime.now().isoformat(),
            "source": "recovered_without_meta",
            "snapshot_name": snapshot_name,
            "snapshot_file": snapshot_file,
            "meta_file": meta_file,
            "fingerprint": build_history_fingerprint(history_df, history_file=snapshot_file),
        }
    else:
        meta.setdefault("snapshot_name", snapshot_name)
        meta.setdefault("snapshot_file", snapshot_file)
        meta.setdefault("meta_file", meta_file)
        meta["fingerprint"] = build_history_fingerprint(history_df, history_file=snapshot_file)

    return snapshot_file, meta_file, meta, history_df
