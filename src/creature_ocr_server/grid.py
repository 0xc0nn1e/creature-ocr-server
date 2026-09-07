"""Where every cell of the sheet sits, as fractions of the cropped page.

The sheet is a fixed 8 x 11 table, so which field a value belongs to follows
from where it sits on the page rather than from anything an engine says about
it. That keeps the mapping geometric - a value can only come from the cell it
was written in - and keeps a change of paper format a coordinate edit in
config.py. (3.1, 5.2-2, 6.5)

Fractions rather than pixels, so nothing here needs to know what resolution the
page was rendered at. That is also why this module imports no image library:
the desktop pipeline's grid.py cuts cells out of a decoded page and needs
PyMuPDF for it, but the server is only ever asked which cell a point falls in
and which band a cell occupies, and both are arithmetic. Adding a dependency
here to share one file would be the wrong trade.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterator

from . import config

# One cell of the table: the printed row number, and the field key.
Cell = tuple[int, str]

# (left, top, right, bottom) as fractions of the page, the frame an engine
# reports a box in and the frame a cell is addressed in.
Band = tuple[float, float, float, float]


def cell_band(row: int, field: str) -> Band:
    """Return the (left, top, right, bottom) fraction band of one cell."""
    if not 1 <= row <= config.ROWS_PER_PAGE:
        raise ValueError(
            f"row {row} is outside 1..{config.ROWS_PER_PAGE}"
        )
    try:
        left, right = config.CELL_COLUMNS[field]
    except KeyError:
        known = ", ".join(config.CELL_COLUMNS)
        raise ValueError(f"unknown field {field!r}: expected one of {known}") from None
    return left, config.CELL_ROW_EDGES[row - 1], right, config.CELL_ROW_EDGES[row]


def iter_cells() -> Iterator[Cell]:
    """Yield every cell of a page, row by row, fields in xlsx column order."""
    for row in range(1, config.ROWS_PER_PAGE + 1):
        for field, _ in config.OCR_FIELDS:
            yield row, field


def cell_at(x: float, y: float) -> Cell | None:
    """Return the cell a point of the page falls in, or None if it falls out.

    Both coordinates are fractions of the page, the same frame the OCR engine
    reports its boxes in. Everything outside the table - the printed No column,
    the header, a mark in the margin - is outside every cell and is dropped by
    the caller rather than guessed into a field. (5.2-2)
    """
    row = bisect.bisect_right(config.CELL_ROW_EDGES, y)
    if not 1 <= row <= config.ROWS_PER_PAGE:
        return None
    for field, (left, right) in config.CELL_COLUMNS.items():
        if left <= x < right:
            return row, field
    return None
