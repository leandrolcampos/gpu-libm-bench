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

"""Update the generated binding definitions for math library images."""

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from gpu_libm_bench.configs import load_function_signature_config

from .libraries import ResolvedLibrary

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:[ \t]+[A-Za-z_][A-Za-z0-9_]*|[ \t]*\*)*")


def update_library_bindings(library: ResolvedLibrary, build_directory: Path) -> None:
    destination = build_directory / "generated" / "LibraryBindings.def"
    if not library.bindings:
        destination.unlink(missing_ok=True)
        return

    contents = _render_library_bindings(library).encode("utf-8")

    try:
        if destination.read_bytes() == contents:
            return
    except FileNotFoundError:
        pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / destination.name
        temporary.write_bytes(contents)
        temporary.replace(destination)


def _render_library_bindings(library: ResolvedLibrary) -> str:
    lines = ["// Generated from library bindings and function signatures. Do not edit."]
    binding_selectors: set[str] = set()

    for function_name, function_symbol in library.bindings:
        if not _IDENTIFIER.fullmatch(function_name) or not _IDENTIFIER.fullmatch(
            function_symbol
        ):
            raise ValueError(
                f"library {library.name!r}: function name {function_name!r} and symbol "
                f"{function_symbol!r} must be C identifiers"
            )

        binding_selector = f"LIBM_BIND_{function_name.upper()}"
        if binding_selector in binding_selectors:
            raise ValueError(
                f"library {library.name!r}: duplicate selector {binding_selector}"
            )
        binding_selectors.add(binding_selector)

        signature = load_function_signature_config(function_name)
        return_type = signature.return_type.strip()
        arguments = [argument.type.strip() for argument in signature.arguments]
        if any(not _TYPE.fullmatch(value) for value in (return_type, *arguments)):
            raise ValueError(f"function {function_name!r}: invalid C type in signature")

        lines.extend(
            (
                "",
                f"#if defined({binding_selector})",
                f"#define LIBM_FUNCTION_NAME {function_name}",
                f"#define LIBM_FUNCTION_SYMBOL {function_symbol}",
                f"#define LIBM_FUNCTION_RETURN_TYPE {return_type}",
            )
        )
        if len(arguments) == 1:
            lines.append(f"#define LIBM_FUNCTION_ARG_TYPE {arguments[0]}")
        else:
            lines.extend(
                f"#define LIBM_FUNCTION_ARG{index}_TYPE {argument}"
                for index, argument in enumerate(arguments, start=1)
            )
        lines.append("#endif")

    return "\n".join(lines) + "\n"
