# Copyright 2026 Leandro Lacerda Campos
# SPDX-License-Identifier: Apache-2.0

"""Layout primitives for terminal reports."""

from dataclasses import dataclass

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.segment import Segment


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
