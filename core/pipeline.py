#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared command-pipeline helpers for script orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


TRUE_SET = {"1", "true", "TRUE", "yes", "YES"}


def env_true(key: str, default: str = "0") -> bool:
    return os.getenv(key, default).strip() in TRUE_SET


def resolve_python_executable(
    *,
    prefer_venv: bool = True,
    venv_python: str = "./.venv/bin/python",
    env_key: str = "QS_PYTHON",
) -> str:
    explicit = str(os.getenv(env_key, "")).strip()
    if explicit:
        return explicit
    if prefer_venv and os.path.exists(venv_python):
        return venv_python
    return sys.executable


@dataclass(frozen=True)
class CommandStep:
    command: tuple[str, ...]
    description: str
    allow_exit_codes: tuple[int, ...] = ()
    fatal: bool = True


def python_step(
    python_executable: str,
    script_path: str,
    description: str,
    *,
    args: Sequence[str] = (),
    allow_exit_codes: Sequence[int] = (),
    fatal: bool = True,
) -> CommandStep:
    return CommandStep(
        command=tuple([python_executable, script_path, *list(args)]),
        description=description,
        allow_exit_codes=tuple(allow_exit_codes),
        fatal=fatal,
    )


def run_steps(
    steps: Iterable[CommandStep],
    *,
    default_allow_exit_codes: Sequence[int] = (),
) -> int:
    step_list = list(steps)
    for idx, step in enumerate(step_list, start=1):
        cmd_str = " ".join(step.command)
        print(f"[run] step {idx}/{len(step_list)}: {step.description}")
        print(f"[run] {cmd_str}")
        rc = subprocess.run(list(step.command)).returncode
        allowed = set(step.allow_exit_codes) | set(default_allow_exit_codes)
        if rc == 0:
            continue
        if rc in allowed:
            print(f"[run] skipped(exit={rc}): {cmd_str}")
            continue
        if step.fatal:
            print(f"[error] failed(exit={rc}): {cmd_str}")
            return rc
        print(f"[warn] non-fatal failed(exit={rc}): {cmd_str}")
    return 0
