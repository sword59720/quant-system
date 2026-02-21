#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable workflow step builders for ETF / single-stock / crypto pipelines."""

from __future__ import annotations

from core.pipeline import CommandStep, python_step


def build_run_all_steps(
    python_executable: str,
    *,
    stock_etf_enabled: bool = True,
    stock_single_enabled: bool = True,
    crypto_enabled: bool = True,
    skip_fetch: bool = False,
    skip_reports: bool = False,
) -> list[CommandStep]:
    steps: list[CommandStep] = []

    if not skip_fetch:
        if stock_etf_enabled:
            steps.append(
                python_step(
                    python_executable,
                    "scripts/stock_etf/fetch_stock_etf_data.py",
                    "Fetch ETF stock data",
                )
            )
        if stock_single_enabled:
            steps.append(
                python_step(
                    python_executable,
                    "scripts/stock_single/fetch_stock_single_data.py",
                    "Fetch single-stock data",
                )
            )
        if crypto_enabled:
            steps.append(
                python_step(
                    python_executable,
                    "scripts/crypto/fetch_crypto_data.py",
                    "Fetch crypto data",
                )
            )

    if stock_etf_enabled:
        steps.append(
            python_step(
                python_executable,
                "scripts/stock_etf/run_stock_etf.py",
                "Run ETF stock model",
            )
        )
    if stock_single_enabled:
        steps.append(
            python_step(
                python_executable,
                "scripts/stock_single/run_stock_single.py",
                "Run single-stock model",
            )
        )
    if crypto_enabled:
        steps.append(
            python_step(
                python_executable,
                "scripts/crypto/run_crypto.py",
                "Run crypto model",
            )
        )

    if not skip_reports:
        if stock_etf_enabled:
            steps.append(
                python_step(
                    python_executable,
                    "scripts/stock_etf/paper_forward_stock.py",
                    "ETF paper-forward evaluation",
                )
            )
            steps.append(
                python_step(
                    python_executable,
                    "scripts/stock_etf/report_stock_weekly.py",
                    "Build stock weekly report",
                )
            )
        if stock_etf_enabled or stock_single_enabled or crypto_enabled:
            steps.append(
                python_step(
                    python_executable,
                    "scripts/report_execution_quality.py",
                    "Build execution quality report",
                )
            )
            steps.append(
                python_step(
                    python_executable,
                    "scripts/notify_execution_quality_wecom.py",
                    "Notify execution quality alert",
                )
            )
    return steps


def build_stock_trade_steps(
    python_executable: str,
    *,
    dry_run: bool = False,
    skip_fetch: bool = False,
    skip_calc: bool = False,
    yes: bool = False,
) -> list[CommandStep]:
    steps: list[CommandStep] = []
    if not skip_fetch:
        steps.append(
            python_step(
                python_executable,
                "scripts/stock_etf/fetch_stock_etf_data.py",
                "Fetch stock market data",
            )
        )
    if not skip_calc:
        steps.append(
            python_step(
                python_executable,
                "scripts/stock_etf/run_stock_etf.py",
                "Run stock signal model",
            )
        )
    steps.append(
        python_step(
            python_executable,
            "scripts/stock_etf/generate_trades_stock_etf.py",
            "Generate trade instructions",
        )
    )

    execute_args = []
    if dry_run:
        execute_args.append("--dry-run")
    if yes:
        execute_args.append("--yes")
    steps.append(
        python_step(
            python_executable,
            "scripts/stock_etf/execute_trades_stock_etf.py",
            "Execute stock trades",
            args=execute_args,
        )
    )
    steps.append(
        python_step(
            python_executable,
            "scripts/report_execution_quality.py",
            "Build execution quality report",
            fatal=False,
        )
    )
    steps.append(
        python_step(
            python_executable,
            "scripts/notify_execution_quality_wecom.py",
            "Notify execution quality alert",
            fatal=False,
        )
    )
    return steps


def build_crypto_trade_steps(
    python_executable: str,
    *,
    dry_run: bool = False,
    skip_fetch: bool = False,
    skip_calc: bool = False,
    yes: bool = False,
) -> list[CommandStep]:
    steps: list[CommandStep] = []
    if not skip_fetch:
        steps.append(
            python_step(
                python_executable,
                "scripts/crypto/fetch_crypto_data.py",
                "Fetch crypto market data",
            )
        )
    if not skip_calc:
        steps.append(
            python_step(
                python_executable,
                "scripts/crypto/run_crypto.py",
                "Run crypto signal model",
            )
        )
    steps.append(
        python_step(
            python_executable,
            "scripts/crypto/generate_trades_crypto.py",
            "Generate trade instructions",
        )
    )

    execute_args = ["--file", "./outputs/orders/crypto_trades.json"]
    if dry_run:
        execute_args.append("--dry-run")
    if yes:
        execute_args.append("--yes")
    steps.append(
        python_step(
            python_executable,
            "scripts/crypto/execute_trades_crypto.py",
            "Execute crypto trades",
            args=execute_args,
        )
    )
    steps.append(
        python_step(
            python_executable,
            "scripts/report_execution_quality.py",
            "Build execution quality report",
            fatal=False,
        )
    )
    steps.append(
        python_step(
            python_executable,
            "scripts/notify_execution_quality_wecom.py",
            "Notify execution quality alert",
            fatal=False,
        )
    )
    return steps
