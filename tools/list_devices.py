# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""GPU device inventory for supported backends."""

import argparse

from rich.console import Console
from rich.table import Table

from gpu_libm_bench.compiler.backends import BackendNotFoundError
from gpu_libm_bench.devices import DeviceContext


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List visible devices across all supported GPU backends."
    )
    parser.parse_args()

    console = Console()
    errors = Console(stderr=True)
    table = Table(title="GPU devices")
    table.add_column("Backend")
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Architecture")
    table.add_column("Memory (MiB)", justify="right")
    failed = False

    for backend in DeviceContext.supported_backends():
        try:
            for device_id in range(DeviceContext.device_count(backend=backend)):
                device_info = DeviceContext.get_device_info(device_id, backend=backend)
                table.add_row(
                    device_info.backend,
                    str(device_info.id),
                    device_info.name,
                    device_info.architecture,
                    f"{device_info.total_memory_bytes / 2**20:,.2f}",
                )
        except BackendNotFoundError:
            continue
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            errors.print(f"{backend}: {error}", style="red", markup=False)
            failed = True

    if table.row_count:
        console.print(table)
    elif not failed:
        console.print("No GPU devices found.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
