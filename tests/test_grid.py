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
from creature_ocr_server.grid import cell_at, cell_band, iter_cells

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


class NoImageLibraryTest(unittest.TestCase):
    """The grid is arithmetic, and importing it must stay free.

    The desktop pipeline's grid.py imports PyMuPDF at module level because it
    cuts cells out of a decoded page. Sharing that file here would put a
    decoder, and its wheels, into a container that only ever does fractions -
    so the two files are deliberately different, and this is the difference.
    """

    def test_importing_the_grid_pulls_in_no_decoder(self):
        for name in [m for m in sys.modules if m.startswith("creature_ocr_server")]:
            del sys.modules[name]
        self.addCleanup(importlib.import_module, "creature_ocr_server.grid")

        importlib.import_module("creature_ocr_server.grid")

        loaded = [m for m in sys.modules if m.startswith(("pymupdf", "fitz", "PIL"))]
        self.assertEqual(loaded, [])

    def test_importing_the_grid_pulls_in_no_vendor_sdk(self):
        # 4.2: a vendor SDK is named inside an engine module and nowhere else,
        # so the package installs and this suite passes with none of them.
        for name in [m for m in sys.modules if m.startswith("creature_ocr_server")]:
            del sys.modules[name]
        self.addCleanup(importlib.import_module, "creature_ocr_server.grid")

        importlib.import_module("creature_ocr_server.grid")

        loaded = [m for m in sys.modules if m.startswith("google")]
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
