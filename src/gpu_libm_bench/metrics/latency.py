# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Dependent-call latency measurement protocols."""

from operator import index

import numpy as np

from gpu_libm_bench.devices import (
    DeviceBuffer,
    DeviceContext,
    DeviceKernel,
    LaunchConfig,
)

_PASS_COUNT = np.uint32(2)


class LatencyBenchmark:
    def __init__(
        self, context: DeviceContext, kernel: DeviceKernel, *, chain_length: int
    ) -> None:
        try:
            length = index(chain_length)
        except TypeError as error:
            raise TypeError("chain_length must be an integer") from error

        if not 1 <= length <= np.iinfo(np.uint32).max:
            raise ValueError("chain_length must be between 1 and 4294967295")

        self._context = context
        self._kernel = kernel
        self._chain_length = np.uint32(length)
        self._mask = context.create_buffer(1, dtype=np.uint32)
        context.copy_to_device(self._mask, np.zeros(1, dtype=np.uint32))

    def run_chain(
        self,
        x: DeviceBuffer[np.float32],
        elapsed_ticks: DeviceBuffer[np.uint64],
        output: DeviceBuffer[np.float32],
    ) -> None:
        for name, buffer in (
            ("x", x),
            ("elapsed_ticks", elapsed_ticks),
            ("output", output),
        ):
            if not isinstance(buffer, DeviceBuffer):
                raise TypeError(f"{name} must be a DeviceBuffer")
            if buffer.size < 1:
                raise ValueError(f"{name} must have capacity for at least one element")

        self._context.launch(
            self._kernel,
            _PASS_COUNT,
            self._chain_length,
            self._mask,
            elapsed_ticks,
            output,
            x,
            config=LaunchConfig(),
        )
