#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "git_head_short": "abf14ca",
    "files_sha256": {
        "config/stock.yaml": "473a834990e61eb65149fdb303f80ca38a703242cd452fdf8e0cbe7534eec797",
        "scripts/stock_etf/paper_forward_stock.py": "7ccdd1c2619ebc451a62ad714f57d450266127b8fceebbaf575513fa51e011be",
        "scripts/stock_etf/run_stock_etf.py": "a28b01463395eb396ff94183d40d2c50a38cad4ee09c0442265608ab51246c81",
        "data/stock/159915.csv": "7ef073b66a770ab53c3096bba2b717c69a90f8a457c0e64c70ad4efb563aa039",
        "data/stock/510300.csv": "f0daeedf50cf58876a8a52e5cc0c3486ce8fea11eb013a808af4bae38bf68a02",
        "data/stock/513100.csv": "bfa0be190bde7c209e4f3d6d0f0d9c4419fcd05650e7bfcaaaa3d505daf20c1c",
        "data/stock/518880.csv": "d1e757b8fd2c05502d097c17053db3cd79a485ff1f3855dd1caec85a346106f7",
    },
    "backtest_report": {
        "path": "outputs/reports/backtest_stock_etf_report.json",
        "periods": 2646,
        "annual_return": 0.19157398305107232,
        "sharpe": 1.4781506594338178,
        "max_drawdown": -0.1370610063965746,
        "date_start": "2015-03-27",
        "date_end": "2026-02-11",
        "history_sha256": "08d8ac787c514af6f61d70a4f7329446a7b714135e514e269d5de6a57265a2bf",
    },
}


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout.strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def almost_equal(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def collect_actual(repo_root: Path) -> dict:
    actual = {"git_head_short": None, "files_sha256": {}, "backtest_report": {}}

    try:
        actual["git_head_short"] = run(["git", "rev-parse", "--short", "HEAD"])
    except Exception as e:
        actual["git_head_short"] = f"<error:{e}>"

    for rel in EXPECTED["files_sha256"]:
        p = repo_root / rel
        if p.exists():
            actual["files_sha256"][rel] = file_sha256(p)
        else:
            actual["files_sha256"][rel] = "<missing>"

    rpt_path = repo_root / EXPECTED["backtest_report"]["path"]
    if rpt_path.exists():
        try:
            r = json.loads(rpt_path.read_text(encoding="utf-8"))
            s = r.get("stock", {})
            fp = s.get("data_fingerprint", {})
            w = (r.get("oos_windows", {}).get("windows", {}) or {}).get("full_period", {})
            actual["backtest_report"] = {
                "path": EXPECTED["backtest_report"]["path"],
                "periods": s.get("periods"),
                "annual_return": s.get("annual_return"),
                "sharpe": s.get("sharpe"),
                "max_drawdown": s.get("max_drawdown"),
                "date_start": w.get("date_start"),
                "date_end": w.get("date_end"),
                "history_sha256": fp.get("history_sha256"),
            }
        except Exception as e:
            actual["backtest_report"] = {"error": str(e)}
    else:
        actual["backtest_report"] = {"error": "missing report file"}

    return actual


def compare(actual: dict) -> list[str]:
    diffs = []

    if actual["git_head_short"] != EXPECTED["git_head_short"]:
        diffs.append(
            f"git_head_short mismatch: actual={actual['git_head_short']} expected={EXPECTED['git_head_short']}"
        )

    for rel, exp in EXPECTED["files_sha256"].items():
        act = actual["files_sha256"].get(rel)
        if act != exp:
            diffs.append(f"file sha mismatch: {rel} actual={act} expected={exp}")

    ar = actual.get("backtest_report", {})
    er = EXPECTED["backtest_report"]
    for k in ["path", "periods", "date_start", "date_end", "history_sha256"]:
        if ar.get(k) != er.get(k):
            diffs.append(f"report {k} mismatch: actual={ar.get(k)} expected={er.get(k)}")
    for k in ["annual_return", "sharpe", "max_drawdown"]:
        if (k not in ar) or (ar.get(k) is None):
            diffs.append(f"report {k} missing")
            continue
        if not almost_equal(float(ar[k]), float(er[k])):
            diffs.append(f"report {k} mismatch: actual={ar[k]} expected={er[k]}")

    return diffs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Raspberry Pi run consistency against local baseline fingerprint.")
    parser.add_argument(
        "--run-backtest",
        action="store_true",
        help="run stock_etf backtest before consistency check",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    if args.run_backtest:
        print("[check] running backtest_stock_etf.py ...", flush=True)
        p = subprocess.run([sys.executable, "scripts/stock_etf/backtest_stock_etf.py"], cwd=repo_root)
        if p.returncode != 0:
            print(f"[check] backtest failed with code {p.returncode}")
            return p.returncode

    actual = collect_actual(repo_root)
    diffs = compare(actual)

    print(json.dumps({"expected": EXPECTED, "actual": actual}, ensure_ascii=False, indent=2))
    if diffs:
        print("\n[check] mismatch found:")
        for d in diffs:
            print(f"- {d}")
        return 1

    print("\n[check] PASS: all fingerprints match expected baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
