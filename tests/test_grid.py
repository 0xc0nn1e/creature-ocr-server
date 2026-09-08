"""The cell grid: which band a cell occupies and which cell a point falls in.

CellBandTest, CellAtTest and IterCellsTest are the desktop pipeline's, lifted
unchanged. Everything of its tests/test_grid.py that is not here needs PyMuPDF,
because it is about cutting cells out of a decoded page - which this server
never does, and which is why it has no image library at all. NoImageLibraryTest
at the end asserts that, because it is an invariant and not a coincidence.
"""

import importlib
import sys
import unittest

from creature_ocr_server import config
from creature_ocr_server.grid import (
    cell_at,
    cell_at_v2,
    cell_band,
    cell_band_v2,
    header_at_v2,
    header_band_v2,
    iter_cells,
)

FIELDS = [key for key, _ in config.OCR_FIELDS]


class CellBandTest(unittest.TestCase):
    def test_a_cell_spans_its_own_column_and_row(self):
        left, top, right, bottom = cell_band(3, "symbol")

        self.assertEqual((left, right), config.CELL_COLUMNS["symbol"])
        self.assertEqual(top, config.CELL_ROW_EDGES[2])
        self.assertEqual(bottom, config.CELL_ROW_EDGES[3])

    def test_the_eight_rows_are_ordered_and_do_not_overlap(self):
        bands = [cell_band(row, "bug_name") for row in range(1, 9)]

        for upper, lower in zip(bands, bands[1:]):
            self.assertEqual(upper[3], lower[1])
            self.assertLess(upper[1], upper[3])

    def test_the_double_rule_belongs_to_neither_neighbour(self):
        left_of_rule = cell_band(1, "what_doing")[2]
        right_of_rule = cell_band(1, "found_month")[0]

        self.assertLess(left_of_rule, right_of_rule)

    def test_an_out_of_range_row_is_rejected(self):
        for row in (0, 9):
            with self.assertRaises(ValueError) as caught:
                cell_band(row, "bug_name")

            self.assertIn("row", str(caught.exception))

    def test_an_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            cell_band(1, "student_name")

        self.assertIn("student_name", str(caught.exception))

    def test_the_no_column_is_not_a_field(self):
        self.assertNotIn("no", config.CELL_COLUMNS)


class CellAtTest(unittest.TestCase):
    """Finding the cell a point of the page falls in."""

    def centre_of(self, row, field):
        left, top, right, bottom = cell_band(row, field)
        return (left + right) / 2, (top + bottom) / 2

    def test_the_centre_of_every_cell_finds_that_cell(self):
        for row, field in iter_cells():
            self.assertEqual(cell_at(*self.centre_of(row, field)), (row, field))

    def test_a_point_above_the_table_belongs_to_no_cell(self):
        self.assertIsNone(cell_at(0.5, config.CELL_ROW_EDGES[0] - 0.001))

    def test_a_point_below_the_table_belongs_to_no_cell(self):
        self.assertIsNone(cell_at(0.5, config.CELL_ROW_EDGES[-1] + 0.001))

    def test_the_printed_no_column_belongs_to_no_cell(self):
        _, top, _, bottom = cell_band(1, "bug_name")

        self.assertIsNone(cell_at(0.05, (top + bottom) / 2))

    def test_the_double_rule_belongs_to_no_cell(self):
        _, top, _, bottom = cell_band(1, "what_doing")
        gap = (config.CELL_COLUMNS["what_doing"][1] + config.CELL_COLUMNS["found_month"][0]) / 2

        self.assertIsNone(cell_at(gap, (top + bottom) / 2))

    def test_a_row_edge_belongs_to_the_row_below_it(self):
        self.assertEqual(cell_at(0.1, config.CELL_ROW_EDGES[1])[0], 2)

    def test_a_column_edge_belongs_to_the_field_on_its_right(self):
        edge = config.CELL_COLUMNS["symbol"][0]
        _, top, _, bottom = cell_band(1, "symbol")

        self.assertEqual(cell_at(edge, (top + bottom) / 2), (1, "symbol"))


class IterCellsTest(unittest.TestCase):
    def test_it_yields_every_row_and_field_exactly_once(self):
        cells = list(iter_cells())

        self.assertEqual(len(cells), 88)
        self.assertEqual(len(set(cells)), 88)
        self.assertEqual({row for row, _ in cells}, set(range(1, 9)))
        self.assertEqual({field for _, field in cells}, set(FIELDS))

    def test_it_yields_the_fields_in_xlsx_column_order(self):
        first_row = [field for row, field in iter_cells() if row == 1]

        self.assertEqual(first_row, FIELDS)


class CompositeGridTest(unittest.TestCase):
    """The v2 frame: the same table, addressed on a bigger canvas."""

    def test_the_table_is_the_v1_table_scaled_by_the_canvas(self):
        # The composite is the v1 crop pasted at (0, 0) and not resampled, so
        # every fraction is the v1 fraction times one ratio per axis. Measured
        # off .idea/rebuild_生き物_page01.png: the printed rules of both images
        # sit at the same pixel.
        down = config.V2_TABLE_PIXELS[1] / config.V2_CANVAS_PIXELS[1]
        across = config.V2_TABLE_PIXELS[0] / config.V2_CANVAS_PIXELS[0]

        for edge, moved in zip(config.CELL_ROW_EDGES, config.CELL_ROW_EDGES_V2):
            self.assertAlmostEqual(moved, edge * down)
        for field, (left, right) in config.CELL_COLUMNS.items():
            self.assertAlmostEqual(config.CELL_COLUMNS_V2[field][0], left * across)
            self.assertAlmostEqual(config.CELL_COLUMNS_V2[field][1], right * across)

    def test_the_v1_frame_did_not_move(self):
        # The one thing this whole version exists to leave alone: a v1 client's
        # cache is keyed on a digest of these numbers.
        self.assertEqual(config.CELL_ROW_EDGES[0], 0.0364)
        self.assertEqual(config.CELL_ROW_EDGES[-1], 0.9764)
        self.assertEqual(config.CELL_COLUMNS["bug_name"], (0.0704, 0.2424))

    def test_the_centre_of_every_cell_finds_that_cell(self):
        for row in range(1, config.ROWS_PER_PAGE + 1):
            for field in FIELDS:
                left, top, right, bottom = cell_band_v2(row, field)
                found = cell_at_v2((left + right) / 2, (top + bottom) / 2)
                self.assertEqual(found, (row, field))

    def test_the_strip_belongs_to_no_cell(self):
        for field in config.HEADER_BANDS_V2:
            left, top, right, bottom = header_band_v2(field)
            self.assertIsNone(cell_at_v2((left + right) / 2, (top + bottom) / 2))

    def test_the_table_belongs_to_no_header_field(self):
        for row in range(1, config.ROWS_PER_PAGE + 1):
            left, top, right, bottom = cell_band_v2(row, "bug_name")
            self.assertIsNone(header_at_v2((left + right) / 2, (top + bottom) / 2))


class HeaderGridTest(unittest.TestCase):
    """The strip: three boxes on one line, and the labels between them."""

    def test_the_centre_of_every_band_finds_that_field(self):
        for field in config.HEADER_BANDS_V2:
            left, top, right, bottom = header_band_v2(field)
            self.assertEqual(header_at_v2((left + right) / 2, (top + bottom) / 2), field)

    def test_the_printed_labels_belong_to_no_field(self):
        # A band that reached over its own label would hand a coordinate engine
        # the label back as the answer. Measured label positions.
        for left, right in ((0.1272, 0.1660), (0.2064, 0.2196), (0.2538, 0.2669)):
            middle = (left + right) / 2
            self.assertIsNone(header_at_v2(middle, 0.94))

    def test_the_bands_do_not_overlap(self):
        spans = sorted(
            (left, right) for left, _, right, _ in config.HEADER_BANDS_V2.values()
        )
        for (_, ends), (starts, _) in zip(spans, spans[1:]):
            self.assertLessEqual(ends, starts)

    def test_a_point_above_the_strip_belongs_to_no_field(self):
        self.assertIsNone(header_at_v2(0.05, 0.5))

    def test_an_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            header_band_v2("name")


class NoImageLibraryTest(unittest.TestCase):
    """The grid is arithmetic, and importing it must stay free.

    The desktop pipeline's grid.py imports PyMuPDF at module level because it
    cuts cells out of a decoded page. Sharing that file here would put a
    decoder, and its wheels, into a container that only ever does fractions -
    so the two files are deliberately different, and this is the difference.
    """

    def loaded(self, prefixes):
        return {name for name in sys.modules if name.startswith(prefixes)}

    def imported_by_the_grid(self, prefixes):
        """What importing the grid pulls in, on a machine that has the lot.

        These are borrowed out of sys.modules for the duration rather than
        subtracted from what was there before, because another test module may
        legitimately have one of them loaded already: the nemotron extra brings
        a decoder, and test_nemotron imports it at module scope, which discovery
        does before any of this runs. A delta measured against that is empty
        whatever the grid imports, so the invariant would stop being checked on
        exactly the machines that have the libraries to break it. Emptied, a
        grid that reaches for one has to load it again, and it shows.

        The same objects go back afterwards, so nothing is re-imported and no
        module ends up with two copies of itself.
        """
        borrowed = {name: sys.modules.pop(name) for name in self.loaded(prefixes)}
        self.addCleanup(sys.modules.update, borrowed)
        for name in [m for m in sys.modules if m.startswith("creature_ocr_server")]:
            del sys.modules[name]
        self.addCleanup(importlib.import_module, "creature_ocr_server.grid")

        importlib.import_module("creature_ocr_server.grid")

        return sorted(self.loaded(prefixes))

    def test_importing_the_grid_pulls_in_no_decoder(self):
        self.assertEqual(self.imported_by_the_grid(("pymupdf", "fitz", "PIL")), [])

    def test_importing_the_grid_pulls_in_no_vendor_sdk(self):
        # 4.2: a vendor SDK is named inside an engine module and nowhere else,
        # so the package installs and this suite passes with none of them.
        self.assertEqual(self.imported_by_the_grid(("google",)), [])


if __name__ == "__main__":
    unittest.main()
