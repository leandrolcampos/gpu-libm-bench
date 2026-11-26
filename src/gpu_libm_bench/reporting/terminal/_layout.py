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

"""Layout primitives for terminal reports."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.segment import Segment
from rich.table import Table
from rich.text import Text


@dataclass(frozen=True, slots=True)
class Indented:
    renderable: RenderableType
    top: int = 0
    bottom: int = 0
    left: int = 2

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # Stream complete lines; Rich Padding crops them to the terminal width.
        for _ in range(self.top):
            yield Segment.line()

        inner_options = options.update_width(max(1, options.max_width - self.left))
        for line in Segment.split_lines(console.render(self.renderable, inner_options)):
            if line:
                yield Segment(" " * self.left)
                yield from line
            yield Segment.line()

        for _ in range(self.bottom):
            yield Segment.line()


def build_domain_report(
    header: tuple[str, ...],
    domains: Mapping[str, Sequence[tuple[str, ...]]],
    generator_descriptions: Mapping[str, str],
) -> Group:
    widths = [cell_len(label) for label in header]
    for rows in domains.values():
        for row in rows:
            for index, value in enumerate(row):
                widths[index] = max(widths[index], cell_len(value))

    sections = []
    for domain, rows in domains.items():
        table = Table(box=box.SQUARE, width=sum(widths) + 3 * len(header) + 1)
        for index, (label, width) in enumerate(zip(header, widths, strict=True)):
            table.add_column(
                Text(label, justify="center"),
                width=width,
                justify="left" if index == 0 else "right",
                no_wrap=True,
                overflow="fold",
            )
        for row in rows:
            table.add_row(*(Text(value) for value in row))
        sections.append(
            Indented(
                Group(
                    Text(f"Domain: {domain}", style="bold"),
                    Text(generator_descriptions[domain]),
                    table,
                ),
                top=1,
            )
        )
    return Group(*sections)


def format_number(value: float | None) -> str:
    return f"{value:.5g}" if value is not None else "N/A"
