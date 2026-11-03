# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Configuration and incremental builds of native CMake projects."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._commands import run_tool


def project_directory() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class CMakeBuild:
    source_directory: Path
    build_directory: Path
    arguments: tuple[str, ...] = ()
    toolchain: tuple[str, ...] = ()
    toolchain_files: tuple[Path, ...] = ()

    def configure(self) -> None:
        source = self.source_directory.expanduser().resolve()
        directory = self.build_directory.expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        cache = self._cache()
        if cache.get("CMAKE_HOME_DIRECTORY", str(source)) != str(source):
            raise ValueError(
                f"build directory belongs to another CMake project: {directory}"
            )

        state = self._read_state()
        changed = (state is None and (directory / "CMakeCache.txt").exists()) or (
            state is not None and state["identity"] != self._identity()
        )
        needs_clean = changed or (state is not None and state["needs_clean"])

        # Persist invalidation before invoking tools, including failed configurations.
        self._write_state({"identity": self._identity(), "needs_clean": needs_clean})

        run_tool(
            [
                "cmake",
                "-S",
                source,
                "-B",
                directory,
                "-G",
                "Ninja",
                *(["--fresh"] if changed else []),
                *self.arguments,
            ],
            timeout=None,
            cwd=directory,
            stream=True,
        )

        self._write_state({"identity": self._identity(), "needs_clean": needs_clean})

    def build(self) -> None:
        directory = self.build_directory.expanduser().resolve()
        state = self._read_state()

        run_tool(
            [
                "cmake",
                "--build",
                directory,
                *(
                    ["--clean-first"]
                    if state is not None and state["needs_clean"]
                    else []
                ),
            ],
            timeout=None,
            cwd=directory,
            stream=True,
        )

        self._write_state({"identity": self._identity(), "needs_clean": False})

    def _cache(self) -> dict[str, str]:
        path = self.build_directory.expanduser().resolve() / "CMakeCache.txt"
        if not path.is_file():
            return {}

        return dict(
            re.findall(
                r"^([^/#\r\n][^:=\r\n]*):[^=\r\n]*=(.*)$",
                path.read_text(),
                re.MULTILINE,
            )
        )

    def _identity(self) -> dict[str, Any]:
        cache = self._cache()
        paths = set(self.toolchain_files)
        for name in (
            "CMAKE_CXX_COMPILER",
            "CMAKE_LINKER",
            "CMAKE_MAKE_PROGRAM",
            "CMAKE_TOOLCHAIN_FILE",
        ):
            if cache.get(name):
                paths.add(Path(cache[name]))

        files = []
        for path in sorted(paths):
            resolved = path.expanduser().resolve()
            stat = resolved.stat() if resolved.is_file() else None
            files.append(
                [
                    str(resolved),
                    stat.st_size if stat else None,
                    stat.st_mtime_ns if stat else None,
                ]
            )

        return {
            "toolchain": list(self.toolchain),
            "files": files,
            "configuration": [
                argument
                for argument in self.arguments
                if argument.split("=", 1)[0]
                in {
                    "-DCMAKE_CXX_COMPILER",
                    "-DCMAKE_CXX_COMPILER_TARGET",
                    "-DCMAKE_TOOLCHAIN_FILE",
                }
            ],
            "environment": {
                name: os.environ.get(name)
                for name in ("CC", "CXX", "CXXFLAGS", "LDFLAGS")
            },
        }

    def _read_state(self) -> dict[str, Any] | None:
        path = (
            self.build_directory.expanduser().resolve() / ".gpu-libm-bench-build.json"
        )
        return json.loads(path.read_text()) if path.is_file() else None

    def _write_state(self, state: dict[str, Any]) -> None:
        path = (
            self.build_directory.expanduser().resolve() / ".gpu-libm-bench-build.json"
        )
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)
