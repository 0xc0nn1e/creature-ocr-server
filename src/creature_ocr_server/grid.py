"""Where every cell of the sheet sits, as fractions of the cropped page.

The sheet is a fixed 8 x 11 table, so which field a value belongs to follows
from where it sits on the page rather than from anything an engine says about
it. That keeps the mapping geometric - a value can only come from the cell it
was written in - and keeps a change of paper format a coordinate edit in
config.py. (3.1, 5.2-2, 6.5)

Two frames, not one. The plain names address the crop /v1/ocr is sent; the `_v2`
names address the composite /v2/ocr is sent, which is that same crop pasted at
the top of a taller canvas with the 小学校 / 年 / 組 strip below it. Both come out
of one pair of private helpers, so the two can never drift into disagreeing about
what a row is - and the v1 numbers are left exactly where they were, which is
what keeps a v1 client's cache good. (3.2)

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


def _band(
    row: int, field: str, edges: tuple[float, ...], columns: dict[str, tuple]
) -> Band:
    """One cell's band, in whichever of the two frames is passed in."""
    if not 1 <= row <= config.ROWS_PER_PAGE:
        raise ValueError(
            f"row {row} is outside 1..{config.ROWS_PER_PAGE}"
        )
    try:
        left, right = columns[field]
    except KeyError:
        known = ", ".join(columns)
        raise ValueError(f"unknown field {field!r}: expected one of {known}") from None
    return left, edges[row - 1], right, edges[row]


def _at(
    x: float, y: float, edges: tuple[float, ...], columns: dict[str, tuple]
) -> Cell | None:
    """The cell a point falls in, in whichever frame is passed in."""
    row = bisect.bisect_right(edges, y)
    if not 1 <= row <= config.ROWS_PER_PAGE:
        return None
    for field, (left, right) in columns.items():
        if left <= x < right:
            return row, field
    return None


def cell_band(row: int, field: str) -> Band:
    """Return the (left, top, right, bottom) fraction band of one cell."""
    return _band(row, field, config.CELL_ROW_EDGES, config.CELL_COLUMNS)


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
    return _at(x, y, config.CELL_ROW_EDGES, config.CELL_COLUMNS)


def cell_band_v2(row: int, field: str) -> Band:
    """The same cell, addressed in the composite /v2/ocr is sent."""
    return _band(row, field, config.CELL_ROW_EDGES_V2, config.CELL_COLUMNS_V2)


def cell_at_v2(x: float, y: float) -> Cell | None:
    """The cell a point of the composite falls in, or None.

    The strip below the table falls outside every row, so a box from it lands
    here as None and is picked up by header_at_v2 instead. Neither function ever
    claims a point the other one wants. (5.2-2)
    """
    return _at(x, y, config.CELL_ROW_EDGES_V2, config.CELL_COLUMNS_V2)


def header_band_v2(field: str) -> Band:
    """Where one of the three page-level values is written on the composite."""
    try:
        return config.HEADER_BANDS_V2[field]
    except KeyError:
        known = ", ".join(config.HEADER_BANDS_V2)
        raise ValueError(f"unknown field {field!r}: expected one of {known}") from None


def header_at_v2(x: float, y: float) -> str | None:
    """The page-level field a point of the strip falls in, or None.

    Rectangles rather than a row-and-column lookup, because the strip is three
    boxes on one line and not a table. The bands stop short of the printed
    labels, so 年 and 組 themselves fall outside all three and are dropped rather
    than returned as the answer they label.
    """
    for field, (left, top, right, bottom) in config.HEADER_BANDS_V2.items():
        if left <= x < right and top <= y < bottom:
            return field
    return None
