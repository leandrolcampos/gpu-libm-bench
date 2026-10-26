# Copyright 2026 Leandro A. Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Python interface to the CUDA backend."""

from typing import TypedDict

class DeviceProperties(TypedDict):
    device_id: int
    name: str
    architecture: str
    total_memory_bytes: int

def device_count() -> int: ...
def get_device_info(device_id: int = 0) -> DeviceProperties: ...
