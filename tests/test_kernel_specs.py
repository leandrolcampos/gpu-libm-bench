# Copyright 2026 Leandro Lacerda Campos
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
            "powi",
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
        (SignatureConfig("float", []), "at least one argument"),
        (
            SignatureConfig("int", [ArgumentConfig("x", "int", "input")]),
            "first signature argument",
        ),
        (
            SignatureConfig("float", [ArgumentConfig("x", "double", "input")]),
            "return type must match",
        ),
        (
            SignatureConfig(
                "void",
                [
                    ArgumentConfig("x", "double", "input"),
                    ArgumentConfig("sin", "double*", "output"),
                    ArgumentConfig("cos", "double*", "output"),
                ],
            ),
            "return value and scalar input arguments",
        ),
    ],
)
@pytest.mark.parametrize("kind", [KernelKind.EVALUATE, KernelKind.LATENCY])
def test_invalid_signature_is_rejected_for_both_wrappers(
    signature: SignatureConfig, message: str, kind: KernelKind
) -> None:
    function = FunctionConfig("example", signature, domains={})
    with pytest.raises(ValueError, match=message):
        KernelSpec.from_config(function, kind=kind)


@pytest.mark.parametrize(
    ("signature", "storage_dtype", "inputs"),
    [
        (
            SignatureConfig("float", [ArgumentConfig("x", "float", "input")]),
            "uint32",
            (KernelArgument("x", "float32", True),),
        ),
        (
            SignatureConfig(
                "double",
                [
                    ArgumentConfig("x", "double", "input"),
                    ArgumentConfig("exponent", "int", "input"),
                ],
            ),
            "uint64",
            (
                KernelArgument("x", "float64", True),
                KernelArgument("exponent", "int32", True),
            ),
        ),
        (
            SignatureConfig(
                "_Float16",
                [
                    ArgumentConfig("x", "_Float16", "input"),
                    ArgumentConfig("y", "_Float16", "input"),
                ],
            ),
            "uint16",
            (
                KernelArgument("x", "float16", True),
                KernelArgument("y", "float16", True),
            ),
        ),
    ],
)
def test_latency_contract_follows_function_signature(
    signature: SignatureConfig, storage_dtype: str, inputs: tuple[KernelArgument, ...]
) -> None:
    function = FunctionConfig("example", signature, domains={})

    assert (
        KernelSpec.from_config(function, kind=KernelKind.LATENCY).arguments
        == (
            KernelArgument("pass_count", "uint32", False),
            KernelArgument("chain_length", "uint32", False),
            KernelArgument("mask", storage_dtype, True),
            KernelArgument("elapsed_ticks", "uint64", True),
            KernelArgument("output", inputs[0].dtype, True),
        )
        + inputs
    )
