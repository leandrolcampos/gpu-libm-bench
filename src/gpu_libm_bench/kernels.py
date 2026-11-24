# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Kernel identities and launch argument contracts."""

from dataclasses import dataclass, field
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
_FLOAT_STORAGE_DTYPES = {
    "float16": "uint16",
    "float32": "uint32",
    "float64": "uint64",
}


class KernelKind(StrEnum):
    EVALUATION = "evaluation"
    LATENCY = "latency"
    THROUGHPUT = "throughput"

    @property
    def unroll_factors(self) -> tuple[int, ...]:
        return (1, 2, 4, 8) if self is KernelKind.THROUGHPUT else ()


@dataclass(frozen=True, slots=True)
class KernelRequest:
    library: str
    function: str
    kind: KernelKind
    unroll_factor: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.kind.unroll_factors:
            if (
                type(self.unroll_factor) is not int
                or self.unroll_factor not in self.kind.unroll_factors
            ):
                raise ValueError("throughput unroll_factor must be 1, 2, 4, or 8")
        elif self.unroll_factor is not None:
            raise ValueError("unroll_factor is only supported for throughput kernels")

    @property
    def symbol(self) -> str:
        name = f"{self.function}{self.kind.value.capitalize()}"
        if self.unroll_factor is not None:
            return f"{name}{self.unroll_factor}"
        return name


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
        if kind is KernelKind.EVALUATION:
            return _evaluation_spec(config.signature)
        if kind is KernelKind.LATENCY:
            return _latency_spec(config.signature)
        if kind is KernelKind.THROUGHPUT:
            return _throughput_spec(config.signature)
        raise ValueError(f"unsupported kernel kind: {kind!r}")


def _evaluation_spec(signature: "SignatureConfig") -> KernelSpec:
    output_dtype, arguments = _process_signature(signature)

    return KernelSpec(
        arguments
        + (
            KernelArgument("output", output_dtype, True),
            KernelArgument("count", "uint64", False),
        )
    )


def _latency_spec(signature: "SignatureConfig") -> KernelSpec:
    output_dtype, arguments = _process_signature(signature)
    storage_dtype = _FLOAT_STORAGE_DTYPES[output_dtype]

    return KernelSpec(
        (
            KernelArgument("pass_count", "uint32", False),
            KernelArgument("chain_length", "uint32", False),
            KernelArgument("mask", storage_dtype, True),
            KernelArgument("elapsed_ticks", "uint64", True),
            KernelArgument("output", storage_dtype, True),
        )
        + arguments
    )


def _throughput_spec(signature: "SignatureConfig") -> KernelSpec:
    output_dtype, arguments = _process_signature(
        signature, require_matching_first=False
    )
    if output_dtype not in _FLOAT_STORAGE_DTYPES:
        raise ValueError("throughput kernels require a floating-point return value")

    return KernelSpec(
        (
            KernelArgument("pass_count", "uint32", False),
            KernelArgument("mask", "uint32", False),
        )
        + arguments
        + (
            KernelArgument("checksums", _FLOAT_STORAGE_DTYPES[output_dtype], True),
            KernelArgument("count", "uint64", False),
        )
    )


def _process_signature(
    signature: "SignatureConfig", *, require_matching_first: bool = True
) -> tuple[str, tuple[KernelArgument, ...]]:
    # Supports powers, exponentials, logarithms, and trigonometric functions,
    # except the sincos family (sincos, sincosf, sincosf16).
    if not signature.arguments:
        raise ValueError("signature must declare at least one argument")

    first = signature.arguments[0]
    first_type = first.type.strip()
    if require_matching_first and (
        first.role != "input" or first_type not in {"_Float16", "float", "double"}
    ):
        raise ValueError(
            "first signature argument must be a scalar _Float16, float, or double input"
        )

    return_type = signature.return_type.strip()
    if return_type == "void" or any(
        argument.role != "input" for argument in signature.arguments
    ):
        raise ValueError("kernels require a return value and scalar input arguments")
    if require_matching_first and return_type != first_type:
        raise ValueError("return type must match the first input type")

    return (
        _scalar_dtype(return_type),
        tuple(
            KernelArgument(argument.name, _scalar_dtype(argument.type.strip()), True)
            for argument in signature.arguments
        ),
    )


def _scalar_dtype(type_name: str) -> str:
    try:
        return _SCALAR_DTYPES[type_name]
    except KeyError as error:
        raise ValueError(
            f"unsupported scalar type in kernel spec: {type_name!r}"
        ) from error
