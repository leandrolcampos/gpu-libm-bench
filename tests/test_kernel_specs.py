# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Tests for kernel launch contracts derived from function signatures."""

import pytest

from gpu_libm_bench.configs import ArgumentConfig, FunctionConfig, SignatureConfig
from gpu_libm_bench.kernels import KernelArgument, KernelKind, KernelSpec


@pytest.mark.parametrize(
    ("name", "signature", "expected"),
    [
        (
            "expf",
            SignatureConfig("float", [ArgumentConfig("x", "float", "input")]),
            (
                KernelArgument("x", "float32", True),
                KernelArgument("output", "float32", True),
                KernelArgument("count", "uint64", False),
            ),
        ),
        (
            "pow",
            SignatureConfig(
                "double",
                [
                    ArgumentConfig("x", "double", "input"),
                    ArgumentConfig("y", "double", "input"),
                ],
            ),
            (
                KernelArgument("x", "float64", True),
                KernelArgument("y", "float64", True),
                KernelArgument("output", "float64", True),
                KernelArgument("count", "uint64", False),
            ),
        ),
        (
            "ldexp",
            SignatureConfig(
                "double",
                [
                    ArgumentConfig("x", "double", "input"),
                    ArgumentConfig("exponent", "int", "input"),
                ],
            ),
            (
                KernelArgument("x", "float64", True),
                KernelArgument("exponent", "int32", True),
                KernelArgument("output", "float64", True),
                KernelArgument("count", "uint64", False),
            ),
        ),
        (
            "expf16",
            SignatureConfig("_Float16", [ArgumentConfig("x", "_Float16", "input")]),
            (
                KernelArgument("x", "float16", True),
                KernelArgument("output", "float16", True),
                KernelArgument("count", "uint64", False),
            ),
        ),
        (
            "sincos",
            SignatureConfig(
                "void",
                [
                    ArgumentConfig("x", "double", "input"),
                    ArgumentConfig("sin", "double*", "output"),
                    ArgumentConfig("cos", "double*", "output"),
                ],
            ),
            (
                KernelArgument("x", "float64", True),
                KernelArgument("sin", "float64", True),
                KernelArgument("cos", "float64", True),
                KernelArgument("count", "uint64", False),
            ),
        ),
    ],
)
def test_evaluation_contract_follows_function_signature(
    name: str, signature: SignatureConfig, expected: tuple[KernelArgument, ...]
) -> None:
    function = FunctionConfig(name, signature, domains={})
    spec = KernelSpec.from_config(function, kind=KernelKind.EVALUATE)
    assert spec.arguments == expected


@pytest.mark.parametrize(
    ("signature", "message"),
    [
        (
            SignatureConfig("float", [ArgumentConfig("x", "long double", "input")]),
            "unsupported scalar type",
        ),
        (
            SignatureConfig(
                "void",
                [
                    ArgumentConfig("x", "float", "input"),
                    ArgumentConfig("output", "float", "output"),
                ],
            ),
            "must have a pointer type",
        ),
    ],
)
def test_unsupported_evaluation_signature_is_rejected(
    signature: SignatureConfig, message: str
) -> None:
    function = FunctionConfig("example", signature, domains={})
    with pytest.raises(ValueError, match=message):
        KernelSpec.from_config(function, kind=KernelKind.EVALUATE)
