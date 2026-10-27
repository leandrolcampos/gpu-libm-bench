# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Execution and diagnostics for native compilation tools."""

import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path


def executable(path: Path) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(
            f"Required executable is missing or not executable: {path}"
        )
    return path


def run_tool(command: Sequence[str | Path]) -> str:
    arguments = [str(argument) for argument in command]
    invocation = shlex.join(arguments)
    try:
        result = subprocess.run(
            arguments, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Cannot execute {invocation}: {error}") from error
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {invocation}\n{output}"
        )
    return output
