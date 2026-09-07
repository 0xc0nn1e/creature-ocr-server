"""The two halves of the split pipeline still agree about what a value is.

This server owns the checks now, and the desktop app keeps its own copy of them
because an operator editing a cell needs the marks to follow the keystroke. Two
copies of the same rules in two repositories is the cost of that, and this is
what stops it turning into two different rules: every check is run on both sides
over the same inputs and the answers have to match, character for character.

It needs the desktop checkout, so it skips where there is none - a container, a
build machine, anywhere but a developer's desk. **A skip is not a pass.** Run it
on a machine that has both before shipping a change to ocr.py or config.py in
either repository, because it is the only thing that will notice.

Point CREATURE_OCR_SRC at the desktop package's src directory to override where
it looks; by default it expects the two checkouts side by side.
"""

import logging
import os
import sys
import unittest
from pathlib import Path

from creature_ocr_server import config as server_config
from creature_ocr_server import ocr as server_ocr
from creature_ocr_server.grid import cell_band
from creature_ocr_server.ocr import Reading, TextBox, Usage


def desktop():
    """The desktop package's ocr and config, or a skip with a reason.

    Imported here rather than at module scope so the skip is a skip and not a
    collection error, and so a machine with no desktop checkout still runs the
    rest of this suite.
    """
    src = os.environ.get("CREATURE_OCR_SRC", "")
    root = Path(src) if src else Path(__file__).resolve().parents[2] / "creature-ocr"
    package = root if root.name == "src" else root / "src"
    if not (package / "creature_ocr" / "ocr.py").is_file():
        raise unittest.SkipTest(
            f"no desktop checkout at {package}; set CREATURE_OCR_SRC. "
            "This skip is not a pass - run it where both checkouts exist."
        )
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))
    from creature_ocr import config, ocr

    return ocr, config


# Every case the desktop repository already treats as load-bearing, plus the
# shapes its own CleanTest was written around. Written out rather than
# generated: a generated table drifts with the code it is meant to police.
CASES = [
    ("bug_name", "シオカラトンボ"),
    ("bug_name", "ハラビロオマキリ"),
    ("bug_name", ""),
    ("bug_name", "  カブトムシ  "),
    ("bug_name", "ストロ윤인していた"),
    ("bug_name", "⁸"),
    ("bug_name", "ℊ"),
    ("notice", "いけの上に\nとまった"),
    ("notice", "車でひかれて\r\nしんじゃった"),
    ("notice", "ストローを出していた"),
    ("where", "しめった地面"),
    ("where", "駐車場の花だん"),
    ("symbol", "き"),
    ("symbol", "あ"),
    ("symbol", "ん"),
    ("symbol", "㋖"),
    ("symbol", ""),
    ("what_doing", "④"),
    ("what_doing", "4"),
    ("what_doing", "10"),
    ("what_doing", "⑩"),
    ("what_doing", "11"),
    ("what_doing", "z"),
    ("found_month", "7"),
    ("found_month", "７"),
    ("found_month", "④"),
    ("found_month", "山"),
    ("found_month", "7月"),
    ("found_day", "8"),
    ("found_day", "８"),
    ("found_day", "３１"),
    ("location_town", "芝公園"),
    ("location_chome", "4"),
    ("location_chome", "山"),
    ("location_chome", "３"),
    ("location_name", "芝公園"),
    ("map_symbol", "G"),
    ("map_symbol", "Ⓖ"),
    ("map_symbol", "ⓖ"),
    ("map_symbol", "Ｇ"),
    ("map_symbol", "g"),
    ("map_symbol", "더"),
    ("map_symbol", "1"),
]


def hush(case):
    """Both pipelines complain about the rejections these cases feed them."""
    for name in ("creature_ocr.ocr", "creature_ocr_server.ocr"):
        logger = logging.getLogger(name)
        previous = logger.level
        logger.setLevel(logging.CRITICAL)
        case.addCleanup(logger.setLevel, previous)


FILLED_ROW = {
    "bug_name": "ショウリョウバッタ",
    "symbol": "き",
    "where": "しめった地面",
    "what_doing": "④",
    "found_month": "7",
    "found_day": "8",
    "location_town": "芝公園",
    "location_chome": "4",
    "location_name": "芝公園",
    "map_symbol": "G",
    "notice": "ストローを出していた",
}


def placed(number, unsure=(), **overrides):
    """One row of boxes as plain tuples, so both sides can build their own.

    The coordinates come from this server's grid. That is the point rather than
    an oversight: after the split the grid is single-source here, and a page
    that both sides read from the same boxes is exactly the comparison worth
    making.
    """
    values = FILLED_ROW | overrides
    boxes = []
    for field, text in values.items():
        if not text:
            continue
        left, top, right, bottom = cell_band(number, field)
        boxes.append((text, left, top, right, bottom, field in unsure))
    return boxes


class ValueCheckParityTest(unittest.TestCase):
    """Every pure check, run on both sides over the same input."""

    def setUp(self):
        self.ocr, self.config = desktop()
        hush(self)

    def test_clean_agrees(self):
        for field, raw in CASES:
            with self.subTest(field=field, raw=raw):
                self.assertEqual(
                    server_ocr.clean(field, raw), self.ocr.clean(field, raw)
                )

    def test_check_returns_the_value_clean_returns(self):
        # The split is a refactor and not a change: what check hands back as a
        # value is what the desktop pipeline returns, or the split moved a
        # value somewhere.
        for field, raw in CASES:
            with self.subTest(field=field, raw=raw):
                self.assertEqual(
                    server_ocr.check(field, raw)[0], self.ocr.clean(field, raw)
                )

    def test_fold_agrees(self):
        for field, raw in CASES:
            with self.subTest(field=field, raw=raw):
                self.assertEqual(
                    server_ocr.fold(field, raw), self.ocr.fold(field, raw)
                )

    def test_names_a_choice_agrees(self):
        for field, raw in CASES:
            with self.subTest(field=field, raw=raw):
                self.assertEqual(
                    server_ocr.names_a_choice(field, raw),
                    self.ocr.names_a_choice(field, raw),
                )

    def test_the_script_whitelist_agrees(self):
        for code in range(0x0, 0xFFFF, 7):
            character = chr(code)
            with self.subTest(code=hex(code)):
                self.assertEqual(
                    server_ocr.in_allowed_scripts(character),
                    self.ocr.in_allowed_scripts(character),
                )

    def test_row_gaps_agrees(self):
        rows = [
            {},
            dict(FILLED_ROW),
            dict(FILLED_ROW, what_doing=""),
            {field: "" for field in FILLED_ROW},
            {"bug_name": "カブトムシ"},
        ]
        for row in rows:
            with self.subTest(row=sorted(row)):
                self.assertEqual(
                    server_ocr.row_gaps(row), self.ocr.row_gaps(row)
                )

    def test_paired_field_problems_agrees(self):
        rows = [
            dict(FILLED_ROW, no="1"),
            dict(FILLED_ROW, no="2", map_symbol="B"),
            dict(FILLED_ROW, no="3", symbol="あ", where="しめった地面"),
            dict(FILLED_ROW, no="4", where=""),
        ]
        for row in rows:
            with self.subTest(no=row["no"]):
                self.assertEqual(
                    server_ocr.paired_field_problems(row),
                    self.ocr.paired_field_problems(row),
                )

    def test_cell_problems_agrees(self):
        rows = [
            dict(FILLED_ROW, no="1"),
            dict(FILLED_ROW, no="2", map_symbol="B"),
            dict(FILLED_ROW, no="3", notice=""),
            {"no": "4", **{field: "" for field in FILLED_ROW}},
        ]
        for row in rows:
            with self.subTest(no=row["no"]):
                self.assertEqual(
                    server_ocr.cell_problems(row), self.ocr.cell_problems(row)
                )

    def test_page_confidence_agrees(self):
        cases = [(0, 0, 0, 0, 0), (11, 0, 0, 0, 0), (11, 1, 0, 0, 0), (10, 0, 0, 1, 2)]
        for counts in cases:
            with self.subTest(counts=counts):
                self.assertEqual(
                    server_ocr.page_confidence(*counts),
                    self.ocr.page_confidence(*counts),
                )


def desktop_root():
    """The desktop checkout, wherever desktop() found its package."""
    src = os.environ.get("CREATURE_OCR_SRC", "")
    root = Path(src) if src else Path(__file__).resolve().parents[2] / "creature-ocr"
    return root.parent if root.name == "src" else root


def desktop_grid_png(directory):
    """A ruled page PNG on disk, built by the desktop repository's own helper.

    read_page_report takes a path because it caches beside the page; this server
    takes bytes because it has neither a path nor anywhere to cache. Loading the
    fixture builder straight off that checkout by file path, rather than putting
    its tests directory on sys.path, keeps its test_grid from shadowing this
    repository's - the two have the same name - and keeps PyMuPDF a dependency
    of the machine rather than of this package.
    """
    import importlib.util

    path = desktop_root() / "tests" / "test_grid.py"
    if not path.is_file():
        raise unittest.SkipTest(f"no desktop test helpers at {path}")
    spec = importlib.util.spec_from_file_location("desktop_test_grid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    png = directory / "page01.png"
    module.make_grid_png(png)
    return png


class WholePageParityTest(unittest.TestCase):
    """The same boxes, read by both pipelines, produce the same page.

    Not the same values one at a time - that is above - but the same finished
    rows and the same quality counts, which is what a client actually receives.
    """

    def setUp(self):
        import tempfile

        self.ocr, self.config = desktop()
        hush(self)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.png = desktop_grid_png(Path(self._tmp.name))

    def both(self, placements):
        """Read one page of boxes on each side and hand back both answers."""

        class ServerEngine(server_ocr.OCREngine):
            name = "parity"

            def recognize(self, image, mime_type="image/png"):
                return Reading(
                    [TextBox(*p) for p in placements], Usage(), []
                )

        outer = self.ocr

        class DesktopEngine(outer.OCREngine):
            name = "parity"

            def recognize(self, image, mime_type="image/png"):
                return [outer.TextBox(*p) for p in placements]

        mine = server_ocr.read_page_image(self.png.read_bytes(), ServerEngine())
        theirs, report = self.ocr.read_page_report(
            self.png, DesktopEngine(), use_cache=False
        )
        return mine, (theirs, report)

    def assertSamePage(self, placements):
        mine, (rows, report) = self.both(placements)

        self.assertEqual(mine.rows, rows)
        self.assertEqual(mine.report.values, report.values)
        self.assertEqual(mine.report.rejected, report.rejected)
        self.assertEqual(mine.report.disagreements, report.disagreements)
        self.assertEqual(mine.report.gaps, report.gaps)
        self.assertEqual(mine.report.unsure, report.unsure)
        self.assertEqual(mine.report.cells, report.cells)
        self.assertEqual(mine.report.score, report.score)
        self.assertEqual(mine.report.percent, report.percent)

    def test_a_blank_page(self):
        self.assertSamePage([])

    def test_a_clean_row(self):
        self.assertSamePage(placed(1))

    def test_a_rejected_value(self):
        self.assertSamePage(placed(2, location_chome="山"))

    def test_a_hole_in_a_filled_row(self):
        self.assertSamePage(placed(3, what_doing=""))

    def test_a_disagreeing_pair(self):
        self.assertSamePage(placed(4, map_symbol="B"))

    def test_a_strained_reading(self):
        self.assertSamePage(placed(5, where="駐車場の花だん", unsure=["where"]))

    def test_a_wrapped_cell(self):
        self.assertSamePage(placed(6, notice="車でひかれて\r\nしんじゃった"))

    def test_several_rows_at_once(self):
        self.assertSamePage(
            placed(1) + placed(2, location_chome="山") + placed(3, what_doing="")
        )


class SheetDefinitionParityTest(unittest.TestCase):
    """Both sides describe the same piece of paper."""

    def setUp(self):
        self.ocr, self.config = desktop()

    def test_the_columns_are_the_same_in_the_same_order(self):
        self.assertEqual(server_config.OCR_FIELDS, self.config.OCR_FIELDS)

    def test_the_grid_is_the_same(self):
        self.assertEqual(server_config.CELL_ROW_EDGES, self.config.CELL_ROW_EDGES)
        self.assertEqual(server_config.CELL_COLUMNS, self.config.CELL_COLUMNS)
        self.assertEqual(server_config.ROWS_PER_PAGE, self.config.ROWS_PER_PAGE)

    def test_every_table_a_check_reads_is_the_same(self):
        named = (
            "OCR_FIELD_CHOICES",
            "OCR_FIELD_SPELLINGS",
            "OCR_FIELD_CHARACTERS",
            "OCR_ALLOWED_SCRIPTS",
            "OCR_CHARACTER_FOLDS",
            "OCR_CIRCLED_DIGIT_FOLDS",
            "OCR_NUMBER_ONLY_FIELDS",
            "OCR_PAIRED_FIELDS",
            "BUG_NAME_HINTS",
        )
        for name in named:
            with self.subTest(table=name):
                self.assertEqual(
                    getattr(server_config, name), getattr(self.config, name)
                )

    def test_determinism_is_set_the_same_way(self):
        # 5.2-4 is a promise about the pipeline, not about one half of it.
        self.assertEqual(
            server_config.GEMINI_TEMPERATURE, self.config.GEMINI_TEMPERATURE
        )
        self.assertEqual(server_config.GEMINI_TOP_P, self.config.GEMINI_TOP_P)

    def test_the_retry_policy_is_the_same(self):
        self.assertEqual(server_config.OCR_RETRIES, self.config.OCR_RETRIES)
        self.assertEqual(
            server_config.OCR_BACKOFF_SECONDS, self.config.OCR_BACKOFF_SECONDS
        )


if __name__ == "__main__":
    unittest.main()
