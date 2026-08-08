from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .utils import get_colors_for_values


@dataclass
class GridSpec:
    """Describes one axis of the grid (row / column / hue).

    Attributes:
        key:       Column name in the (already aggregated) data frame.
        values:    Ordered list of values; one row / column / line per entry.
        labels:    Optional pretty labels matched 1:1 with `values`. Defaults
                   to `str(value)`.
    """

    key: str
    values: Sequence[Any]
    labels: Sequence[str] | None = None

    def label(self, i: int) -> str:
        if self.labels is not None:
            return self.labels[i]
        return str(self.values[i])

    def __len__(self):
        return len(self.values)


@dataclass
class HueSpec(GridSpec):
    """Describes one axis of the grid (row / column / hue).

    Attributes:
        key:       Column name in the (already aggregated) data frame.
        values:    Ordered list of values; one row / column / line per entry.
        labels:    Optional pretty labels matched 1:1 with `values`. Defaults
                   to `str(value)`.
        colors:    Optional explicit colors. Either a sequence aligned with
                   `values`, or a mapping {value: color}. If None, the hue
                   axis falls back to a numeric colormap (see `cmap_name`).
    """

    colors: Sequence[Any] | Mapping[Any, Any] | None = None
    linestyles: Sequence[Literal["-", "--", "-.", ":"]] | None = None
    markerstyles: Sequence[Literal["o", "x", "v"]] | None = None

    def color(self, i: int) -> Any | None:
        if self.colors is None:
            return None
        if isinstance(self.colors, Mapping):
            return self.colors[self.values[i]]
        return self.colors[i]

    def get_colors(self, cmap_name: str = "viridis") -> list[Any]:
        """Return a list of colors aligned with `values`.

        Uses explicit `colors` if provided, otherwise falls back to
        `get_cmap_for_values` (which requires numeric values).
        """
        if self.colors is not None:
            return [self.color(i) for i in range(len(self.values))]
        return get_colors_for_values(list(self.values), cmap_name=cmap_name)

    def get_linestyles(self) -> list[Literal["-", "--", "-.", ":"]]:
        if self.linestyles is not None:
            return [self.linestyles[i] for i in range(len(self.values))]
        return ["-" for _ in range(len(self.values))]
