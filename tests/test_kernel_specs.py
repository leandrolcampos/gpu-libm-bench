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

"""Tests for kernel identities and launch argument contracts."""

import pytest

from gpu_libm_bench.configs import ArgumentConfig, FunctionConfig, SignatureConfig
from gpu_libm_bench.kernels import KernelArgument, KernelKind, KernelRequest, KernelSpec


def test_throughput_request_validates_unroll_factor_and_uses_it_in_symbol() -> None:
    kind = KernelKind.THROUGHPUT
    request = KernelRequest("libdevice", "expf", kind, unroll_factor=4)
    assert request.symbol == "expfThroughput4"

    with pytest.raises(ValueError, match="unroll_factor"):
        KernelRequest("libdevice", "expf", kind, unroll_factor=3)


def test_evaluation_spec_uses_signature_for_input_and_output_pointers() -> None:
    signature = SignatureConfig(
        "double",
        [
            ArgumentConfig("x", "double", "input"),
            ArgumentConfig("exponent", "int", "input"),
        ],
    )
    function = FunctionConfig("powi", signature, domains={})
    spec = KernelSpec.from_config(function, kind=KernelKind.EVALUATION)

    assert spec.arguments == (
        KernelArgument("x", "float64", True),
        KernelArgument("exponent", "int32", True),
        KernelArgument("output", "float64", True),
        KernelArgument("count", "uint64", False),
    )


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
def test_latency_spec_uses_float_storage_for_mask_and_output(
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
            KernelArgument("output", storage_dtype, True),
        )
        + inputs
    )


def test_throughput_spec_uses_return_type_for_checksum_storage() -> None:
    signature = SignatureConfig(
        "double",
        [
            ArgumentConfig("x", "float", "input"),
            ArgumentConfig("exponent", "int", "input"),
        ],
    )
    function = FunctionConfig("example", signature, domains={})
    assert KernelSpec.from_config(function, kind=KernelKind.THROUGHPUT).arguments == (
        KernelArgument("pass_count", "uint32", False),
        KernelArgument("mask", "uint32", False),
        KernelArgument("x", "float32", True),
        KernelArgument("exponent", "int32", True),
        KernelArgument("checksums", "uint64", True),
        KernelArgument("count", "uint64", False),
    )


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
@pytest.mark.parametrize("kind", [KernelKind.EVALUATION, KernelKind.LATENCY])
def test_evaluation_and_latency_specs_reject_unsupported_signatures(
    signature: SignatureConfig, message: str, kind: KernelKind
) -> None:
    function = FunctionConfig("example", signature, domains={})
    with pytest.raises(ValueError, match=message):
        KernelSpec.from_config(function, kind=kind)
