# ===----------------------------------------------------------------------=== #
# Copyright 2026 Leandro Lacerda Campos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Execution and diagnostics for native compilation tools."""

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def executable(path: Path) -> Path:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(
            f"required executable is missing or not executable: {path}"
        )
    return path


def run_tool(
    command: Sequence[str | Path],
    *,
    timeout: float | None = 30,
    cwd: Path | None = None,
    stream: bool = False,
) -> str:
    if stream and timeout is not None:
        raise ValueError("streamed commands require timeout=None")

    arguments = [str(argument) for argument in command]
    invocation = shlex.join(arguments)
    location = f" in {cwd}" if cwd is not None else ""

    try:
        if stream:
            with subprocess.Popen(
                arguments,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as process:
                assert process.stdout is not None
                parts = []
                for line in process.stdout:
                    print(line, end="", file=sys.stderr, flush=True)
                    parts.append(line)

                returncode = process.wait(timeout=timeout)
                output = "".join(parts).strip()
        else:
            result = subprocess.run(
                arguments,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            returncode = result.returncode
            output = "\n".join(
                part.strip() for part in (result.stdout, result.stderr) if part
            )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"cannot execute {invocation}{location}: {error}") from error

    if returncode:
        raise RuntimeError(
            f"command failed ({returncode}){location}: {invocation}\n{output}"
        )
    return output
