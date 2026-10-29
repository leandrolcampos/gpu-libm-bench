# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel identities and launch argument contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .configs import FunctionConfig, SignatureConfig

_SCALAR_DTYPES = {
    "_Float16": "float16",
    "float": "float32",
    "double": "float64",
    "int": "int32",
}


class KernelKind(StrEnum):
    EVALUATE = "evaluate"


@dataclass(frozen=True, slots=True)
class KernelRequest:
    library: str
    function: str
    kind: KernelKind

    @property
    def symbol(self) -> str:
        return f"{self.function}{self.kind.value.capitalize()}"


@dataclass(frozen=True, slots=True)
class KernelArgument:
    name: str
    dtype: str
    is_pointer: bool


@dataclass(frozen=True, slots=True)
class KernelSpec:
    arguments: tuple[KernelArgument, ...]

    @staticmethod
    def from_config(config: "FunctionConfig", *, kind: KernelKind) -> "KernelSpec":
        if kind is KernelKind.EVALUATE:
            return _evaluate_spec(config.signature)
        raise ValueError(f"unsupported kernel kind: {kind!r}")


def _evaluate_spec(signature: "SignatureConfig") -> KernelSpec:
    arguments = []
    for argument in signature.arguments:
        type_name = argument.type.strip()
        if argument.role == "output":
            if not type_name.endswith("*"):
                raise ValueError(
                    f"output argument {argument.name!r} must have a pointer type"
                )
            type_name = type_name[:-1].strip()

        arguments.append(KernelArgument(argument.name, _scalar_dtype(type_name), True))

    return_type = signature.return_type.strip()
    if return_type != "void":
        arguments.append(KernelArgument("output", _scalar_dtype(return_type), True))

    arguments.append(KernelArgument("count", "uint64", False))
    return KernelSpec(tuple(arguments))


def _scalar_dtype(type_name: str) -> str:
    try:
        return _SCALAR_DTYPES[type_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported scalar type in kernel spec: {type_name!r}"
        ) from error
