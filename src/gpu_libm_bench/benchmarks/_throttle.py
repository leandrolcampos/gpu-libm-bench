# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Recovery policy for detected throttling during benchmarks."""


def recovery_delay_s(delay_s: float, consecutive_discards: int) -> float:
    if consecutive_discards >= 4:
        return delay_s + min(1.5 * delay_s, 0.5)
    return delay_s
