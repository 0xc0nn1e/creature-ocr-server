"""Reading a page: the cell assignment, the value checks and the retry policy.

No network and no vendor SDK. The engine is a fake and every case here is
arithmetic over text, so the suite passes on a machine that has never heard of
Vertex AI - which is the point of keeping the SDKs in extras. (4.2)

Most of this file is the desktop pipeline's tests/test_ocr.py, carried over
without edits beyond the import path. That is deliberate and it is a check in
its own right: a case that had to be reworded to pass here is a difference
between the two implementations, and a difference is what tests/test_parity.py
exists to prevent. The classes that are new or rewritten are the ones the
server genuinely changed - what a call reports, what the deadline does, and a
page arriving as bytes instead of as a path.
"""

import json
import logging
import string
import time
import unicodedata
import unittest
from dataclasses import replace
from unittest.mock import patch

from creature_ocr_server import config
from creature_ocr_server.grid import cell_band
from creature_ocr_server.ocr import (
    CELL_DISAGREEMENT,
    CELL_MISSING,
    CELL_REJECTED,
    CELL_UNSURE,
    OCREngine,
    OCRError,
    OCRTimeout,
    PageReport,
    Reading,
    TextBox,
    Usage,
    assign_to_cells,
    cell_problems,
    check,
    checks_fingerprint,
    clean,
    fold,
    in_allowed_scripts,
    page_cells,
    page_confidence,
    paired_field_problems,
    read_page_image,
    recognize_with_retry,
    row_gaps,
)

PAGE = b"a whole page of PNG"

FIELDS = [key for key, _ in config.OCR_FIELDS]


def quieten(case):
    """Silence the pipeline's warnings for one test.

    Most cases below feed a page holding one or two boxes, which read_page_image
    is right to call a row full of holes and which would otherwise bury the test
    output. assertLogs sets its own level, so a case that asserts on the
    complaint still sees it.
    """
    logger = logging.getLogger("creature_ocr_server.ocr")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    case.addCleanup(logger.setLevel, previous)


class RecordingEngine(OCREngine):
    """Answers with a fixed reading and counts how often it was asked.

    As close to the desktop pipeline's double as the contract allows. The one
    difference is the one the server deliberately made: recognize returns what
    the call cost rather than adding it to an attribute, so `cost` here is what
    each call reports and not a running total.
    """

    name = "recording"

    def __init__(self, boxes=(), error=None, cost=None, notes=()):
        self.boxes = list(boxes)
        self.error = error
        self.cost = cost if cost is not None else Usage()
        self.notes = list(notes)
        self.calls = 0
        self.images = []
        self.mime_types = []

    def recognize(self, image, mime_type="image/png"):
        self.calls += 1
        self.images.append(image)
        self.mime_types.append(mime_type)
        if self.error is not None:
            raise self.error
        return Reading(list(self.boxes), self.cost, list(self.notes))


def box(row, field, text, part=0, of=1, line=0, lines=1):
    """A TextBox sitting in one cell, optionally one of several side by side."""
    left, top, right, bottom = cell_band(row, field)
    width = (right - left) / of
    height = (bottom - top) / lines
    return TextBox(
        text,
        left + part * width,
        top + line * height,
        left + (part + 1) * width,
        top + (line + 1) * height,
    )


# One row of a real sheet, so a test that cares what a page scores is scored
# the way a page is. A row holding one value out of eleven is ten gaps, which is
# correct for the sheet and useless as a fixture.
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


def filled_row(number, unsure=(), **overrides):
    """Boxes for one completely filled row, with any field replaced.

    `unsure` names the columns the engine is to report as badly read.
    """
    values = FILLED_ROW | overrides
    return [
        replace(box(number, field, text), unsure=field in unsure)
        for field, text in values.items()
        if text
    ]


def middle_of(field):
    """The horizontal centre of a column, read off the config not memorised."""
    left, right = config.CELL_COLUMNS[field]
    return (left + right) / 2


def between(first, second):
    """The centre of the printed rule between two columns."""
    return (config.CELL_COLUMNS[first][1] + config.CELL_COLUMNS[second][0]) / 2


class Recorder:
    """Collects the delays a retry would have slept for."""

    def __init__(self):
        self.delays = []

    def __call__(self, delay):
        self.delays.append(delay)


def disagreeing_pair():
    """A symbol letter and a wording that names a different one.

    Worked out from the configured words rather than written down, so the case
    keeps testing a disagreement after the sheet's wording lists change.
    """
    for letter, words in config.SYMBOL_WORDS.items():
        for other, theirs in config.SYMBOL_WORDS.items():
            if other == letter or not theirs:
                continue
            if not any(word in theirs[0] for word in words):
                return letter, theirs[0]
    raise unittest.SkipTest("no two symbol wordings disagree")


class RecognizeWithRetryTest(unittest.TestCase):
    def test_a_successful_call_is_not_retried(self):
        engine = RecordingEngine([box(1, "symbol", "き")])
        sleep = Recorder()

        found = recognize_with_retry(engine, PAGE, sleep=sleep)

        self.assertEqual([b.text for b in found.boxes], ["き"])
        self.assertEqual(engine.calls, 1)
        self.assertEqual(sleep.delays, [])

    def test_it_retries_three_times_before_giving_up(self):
        engine = RecordingEngine(error=OCRError("down"))
        sleep = Recorder()

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError):
                recognize_with_retry(engine, PAGE, sleep=sleep)

        # 6.4 asks for three retries, so four calls in total.
        self.assertEqual(engine.calls, 4)

    def test_the_backoff_grows_exponentially(self):
        engine = RecordingEngine(error=OCRError("down"))
        sleep = Recorder()

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError):
                recognize_with_retry(engine, PAGE, backoff=1.0, sleep=sleep)

        self.assertEqual(sleep.delays, [1.0, 2.0, 4.0])

    def test_an_unexpected_error_is_wrapped_too(self):
        engine = RecordingEngine(error=RuntimeError("boom"))
        sleep = Recorder()

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError):
                recognize_with_retry(engine, PAGE, sleep=sleep)

    def test_the_mime_type_reaches_the_engine(self):
        # The engine is told what it is looking at, and what it is told is what
        # the bytes turned out to be rather than what the client claimed. (6.2)
        engine = RecordingEngine()

        recognize_with_retry(engine, PAGE, "image/jpeg", sleep=Recorder())

        self.assertEqual(engine.mime_types, ["image/jpeg"])


class DeadlineTest(unittest.TestCase):
    """A request has a budget, and an attempt that cannot fit is not started.

    6.4's three retries against a page that measures up to 68 seconds is a
    request that can legitimately run for about 280 seconds. Something has to
    own that number or every client and proxy in front of this server picks its
    own, and the slow pages die where nobody is looking. (6.1, 6.4)
    """

    def test_a_budget_already_spent_stops_before_the_first_call(self):
        engine = RecordingEngine([box(1, "symbol", "き")])

        with self.assertRaises(OCRTimeout):
            recognize_with_retry(
                engine, PAGE, deadline=time.monotonic() - 1, sleep=Recorder()
            )

        # Not called at all: starting a call that cannot finish spends the
        # money and still returns nothing.
        self.assertEqual(engine.calls, 0)

    def test_a_timeout_is_not_reported_as_an_engine_failure(self):
        # Both are OCRError so a caller that does not care can catch one thing,
        # but a server answering 502 for a deadline it set itself would blame
        # the engine for its own setting.
        engine = RecordingEngine()

        with self.assertRaises(OCRTimeout) as raised:
            recognize_with_retry(
                engine, PAGE, deadline=time.monotonic() - 1, sleep=Recorder()
            )

        self.assertIsInstance(raised.exception, OCRError)

    def test_what_the_earlier_attempts_cost_survives_the_timeout(self):
        engine = RecordingEngine(error=OCRError("down", Usage(prompt_tokens=11)))
        deadline = time.monotonic() + 0.05

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError) as raised:
                recognize_with_retry(
                    engine,
                    PAGE,
                    backoff=0.05,
                    deadline=deadline,
                    sleep=lambda delay: time.sleep(delay),
                )

        self.assertGreaterEqual(raised.exception.usage.calls, 1)
        self.assertGreaterEqual(raised.exception.usage.prompt_tokens, 11)

    def test_no_deadline_means_no_budget(self):
        engine = RecordingEngine([box(1, "symbol", "き")])

        found = recognize_with_retry(engine, PAGE, deadline=None, sleep=Recorder())

        self.assertEqual(len(found.boxes), 1)


class EngineReadingOrderTest(unittest.TestCase):
    """The engine's order is authoritative; nothing here may re-sort it.

    Rebuilding reading order from bounding boxes cannot be done safely: a line
    whose right end dips below the start of the next line reads correctly on
    paper but sorts wrongly by any top-edge rule, and the corruption is silent.
    """

    def test_the_order_the_engine_gave_is_the_order_that_comes_out(self):
        boxes = [
            box(1, "notice", "ストロー", part=0, of=3),
            box(1, "notice", "を出", part=1, of=3),
            box(1, "notice", "していた", part=2, of=3),
        ]

        self.assertEqual(assign_to_cells(boxes)[(1, "notice")], "ストローを出していた")

    def test_a_line_that_dips_below_the_next_one_is_not_reordered(self):
        left, top, right, bottom = cell_band(1, "notice")
        width, height = right - left, bottom - top
        # "ストローを出" runs across the top but its right end drops below the
        # start of "していた" on the line under it. Reading order is unchanged.
        boxes = [
            TextBox("ストロー", left, top, left + width / 3, top + height * 0.45),
            TextBox(
                "を出",
                left + width * 0.55,
                top + height * 0.5,
                left + width * 0.8,
                top + height * 0.95,
            ),
            TextBox(
                "していた",
                left,
                top + height * 0.48,
                left + width / 3,
                top + height * 0.93,
            ),
        ]

        self.assertEqual(assign_to_cells(boxes)[(1, "notice")], "ストローを出していた")

    def test_boxes_are_not_sorted_by_position_within_a_cell(self):
        # Right-to-left in space, but this is the order the engine reported.
        boxes = [
            box(1, "bug_name", "ショウ", part=1, of=2),
            box(1, "bug_name", "リョウ", part=0, of=2),
        ]

        self.assertEqual(assign_to_cells(boxes)[(1, "bug_name")], "ショウリョウ")


class AssignToCellsTest(unittest.TestCase):
    def test_it_answers_for_every_cell_of_the_page(self):
        values = assign_to_cells([])

        self.assertEqual(len(values), 88)
        self.assertEqual(set(values.values()), {""})

    def test_a_box_lands_in_the_cell_its_centre_falls_in(self):
        values = assign_to_cells([box(3, "where", "しめった地面")])

        self.assertEqual(values[(3, "where")], "しめった地面")
        self.assertEqual(values[(3, "bug_name")], "")
        self.assertEqual(values[(2, "where")], "")

    def test_two_lines_of_one_cell_are_joined_as_the_engine_gave_them(self):
        values = assign_to_cells(
            [
                box(1, "notice", "ストローを出", line=0, lines=2),
                box(1, "notice", "していた", line=1, lines=2),
            ]
        )

        self.assertEqual(values[(1, "notice")], "ストローを出していた")

    def test_they_are_joined_without_a_separator(self):
        values = assign_to_cells(
            [
                box(1, "bug_name", "ショウ", part=0, of=2),
                box(1, "bug_name", "リョウバッタ", part=1, of=2),
            ]
        )

        self.assertEqual(values[(1, "bug_name")], "ショウリョウバッタ")

    def test_a_cell_written_on_two_lines_is_joined_not_lost(self):
        # A row-returning engine reports a wrapped cell as one string with the
        # break in it. The break is outside every allowed script, so leaving it
        # alone made clean empty the whole cell - it cost two 気が付いたこと
        # values on one page of five.
        boxes = [box(2, "notice", "いけの上に\nとまった")]

        values = assign_to_cells(boxes)

        self.assertEqual(values[(2, "notice")], "いけの上にとまった")

    def test_a_wrapped_cell_survives_the_character_check(self):
        engine = RecordingEngine(
            filled_row(2, notice="車でひかれて\r\nしんじゃった")
        )

        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            rows = read_page_image(PAGE, engine).rows

        self.assertEqual(rows[1]["notice"], "車でひかれてしんじゃった")

    def test_a_box_outside_the_table_is_dropped(self):
        # The printed No column sits left of every field.
        outside = TextBox("1", 0.045, 0.09, 0.060, 0.11)

        values = assign_to_cells([outside])

        self.assertEqual(set(values.values()), {""})

    def test_a_box_in_the_double_rule_belongs_to_neither_neighbour(self):
        # Coordinates derived from the config: the gap moves whenever the crop
        # is re-measured, and a test that memorised it would quietly start
        # asserting something else.
        centre = between("what_doing", "found_month")
        gap = TextBox("x", centre - 0.0002, 0.09, centre + 0.0002, 0.11)

        values = assign_to_cells([gap])

        self.assertEqual(values[(1, "what_doing")], "")
        self.assertEqual(values[(1, "found_month")], "")

    def test_a_box_is_never_counted_in_two_cells(self):
        values = assign_to_cells([box(4, "location_town", "芝公園")])

        filled = [cell for cell, text in values.items() if text]

        self.assertEqual(filled, [(4, "location_town")])

    def test_no_value_is_carried_between_rows(self):
        boxes = [box(row, "bug_name", f"virus{row}") for row in (2, 5)]

        values = assign_to_cells(boxes)

        self.assertEqual(values[(2, "bug_name")], "virus2")
        self.assertEqual(values[(5, "bug_name")], "virus5")
        self.assertEqual(values[(3, "bug_name")], "")


class CleanTest(unittest.TestCase):
    """3.1 fixes the characters a field can hold; anything else is a misread."""

    def test_japanese_free_text_passes_through(self):
        self.assertEqual(clean("bug_name", "ショウリョウバッタ"), "ショウリョウバッタ")
        self.assertEqual(clean("where", "しめった地面"), "しめった地面")
        self.assertEqual(clean("location_name", "芝公園"), "芝公園")
        self.assertEqual(clean("notice", "ストローを出していた"), "ストローを出していた")

    def test_a_value_mixing_in_hangul_is_rejected_whole(self):
        # Document OCR returns Korean for handwritten kana; the sheet has none.
        # Deleting just the Hangul would leave ストロしていた, a reading the
        # engine never gave that looks like an ordinary answer. 5.2-2 forbids
        # manufacturing one, so the whole value goes.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("notice", "ストロ윤인していた"), "")

    def test_it_returns_the_engines_value_or_nothing(self):
        """The whole contract, over accepted and rejected input alike.

        The earlier version of this test only listed inputs that get rejected,
        so it passed no matter what the accepted path did with them.
        """
        cases = [
            # accepted, unchanged
            ("bug_name", "ショウリョウバッタ"),
            ("where", "しめった地面"),
            ("what_doing", "④"),
            ("notice", "ストローを出していた"),
            ("found_day", "8"),
            ("map_symbol", "G"),
            # accepted, folded
            ("found_month", "７"),
            ("map_symbol", "Ⓖ"),
            ("map_symbol", "ⓖ"),
            ("map_symbol", "Ｇ"),
            # whitespace only
            ("found_day", " 8 "),
            # rejected
            ("found_month", "⑧"),
            ("found_month", "⁸"),
            ("found_month", "④"),
            ("found_month", "/8"),
            ("map_symbol", "ℊ"),
            ("map_symbol", "더"),
            ("location_chome", "山"),
            ("notice", "ストロ윤인していた"),
            ("bug_name", "ショウリョウバッタ법"),
        ]
        # The rejected half of the list logs; capture it so the run stays quiet.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            for field, raw in cases:
                self._assert_original_or_empty(field, raw)

    def _assert_original_or_empty(self, field, raw):
        """Allowed answers come from the Unicode database, not from config.

        Deriving them from OCR_CHARACTER_FOLDS would make this vacuous: a table
        saying Ⓖ maps to H would define its own output as correct. NFKC is the
        independent oracle for "the same character written another way" — the
        production code does not use it, precisely because it folds more than
        this sheet allows, but every fold the table does make has to agree
        with it.
        """
        with self.subTest(field=field, raw=raw):
            stripped = raw.strip()
            # Every column folds now, not only the four with a character set,
            # so the NFKC form is a permitted answer for all of them.
            permitted = {"", stripped, unicodedata.normalize("NFKC", stripped)}

            self.assertIn(
                clean(field, raw),
                permitted,
                "clean must return the engine's value, a width or ring variant "
                "of it, or nothing",
            )

    def test_a_circled_digit_is_not_folded_into_a_month(self):
        # ④ is what 何してた legitimately holds one column away. Folding it
        # would turn a column bleed into a month nobody wrote.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("found_month", "④"), "")
            self.assertEqual(clean("found_day", "⑧"), "")

    def test_a_superscript_digit_is_not_folded_into_a_month(self):
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("found_month", "⁸"), "")

    def test_a_decorative_letter_is_not_folded_into_the_map_symbol(self):
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("map_symbol", "ℊ"), "")

    def test_every_fold_lands_on_the_same_character(self):
        # Checked against the Unicode database rather than against the table,
        # so a mapping such as Ⓖ -> H cannot define itself as correct.
        every = {**config.OCR_CHARACTER_FOLDS, **config.OCR_CIRCLED_DIGIT_FOLDS}
        for source, target in every.items():
            with self.subTest(source=source, target=target):
                decomposed = unicodedata.normalize("NFKC", source)
                if decomposed != source:
                    self.assertEqual(decomposed, target)

    def test_a_ring_unicode_cannot_decompose_lands_where_its_twin_does(self):
        # The dingbat rings have no decomposition at all, so the database
        # cannot be asked directly. Each one is checked against the ordinary
        # circled digit of the same value, which the case above did check.
        plain = config.OCR_CIRCLED_DIGIT_FOLDS
        for first in (0x2776, 0x2780, 0x278A):
            for offset in range(10):
                dingbat = chr(first + offset)
                with self.subTest(dingbat=dingbat):
                    self.assertEqual(
                        plain[dingbat], plain[chr(0x2460 + offset)]
                    )

    def test_the_fold_tables_hold_only_width_and_ring_variants(self):
        every = {**config.OCR_CHARACTER_FOLDS, **config.OCR_CIRCLED_DIGIT_FOLDS}
        for source in every:
            with self.subTest(source=source):
                name = unicodedata.name(source)
                self.assertTrue(
                    name.startswith("FULLWIDTH ") or "CIRCLED " in name,
                    f"{name} is not a width or ring variant",
                )

    def test_the_number_columns_are_the_only_ones_that_keep_a_ring(self):
        # ④ is what 何してた holds one column over. Folding it there would let a
        # column bleed reach 見つけた月 as a month nobody wrote, so the circled
        # digits live in their own table and the number columns do not take it.
        for circled in "①②③④⑤⑥⑦⑧⑨":
            self.assertNotIn(circled, config.OCR_CHARACTER_FOLDS)
            self.assertIn(circled, config.OCR_CIRCLED_DIGIT_FOLDS)

        self.assertEqual(fold("found_month", "④"), "④")
        self.assertEqual(fold("what_doing", "④"), "4")

    def test_every_number_only_column_is_a_real_field_with_a_digit_set(self):
        for field in config.OCR_NUMBER_ONLY_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(
                    config.OCR_FIELD_CHARACTERS.get(field), string.digits
                )

    def test_a_digit_field_accepts_digits(self):
        self.assertEqual(clean("found_month", "7"), "7")
        self.assertEqual(clean("found_day", "8"), "8")
        self.assertEqual(clean("location_chome", "4"), "4")

    def test_a_digit_field_rejects_a_value_carrying_a_stray_mark(self):
        # '/8' is a stray stroke plus a digit. Keeping the 8 would be a guess:
        # the engine may equally have mangled an 18.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("found_month", "/8"), "")

    def test_a_digit_field_rejects_a_misread_kanji(self):
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("location_chome", "山"), "")

    def test_a_full_width_digit_becomes_ascii(self):
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("found_month", "７"), "7")
            self.assertEqual(clean("map_symbol", "Ｇ"), "G")

    def test_the_map_symbol_keeps_only_a_latin_letter(self):
        self.assertEqual(clean("map_symbol", "G"), "G")
        self.assertEqual(clean("map_symbol", "g"), "g")

    def test_a_circled_letter_is_the_letter(self):
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("map_symbol", "Ⓖ"), "G")
            self.assertEqual(clean("map_symbol", "ⓖ"), "g")

    def test_the_map_symbol_rejects_a_misread_kana(self):
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("map_symbol", "の"), "")
            self.assertEqual(clean("map_symbol", "더"), "")

    def test_every_constrained_field_is_a_real_field(self):
        keys = {field for field, _ in config.OCR_FIELDS}

        self.assertLessEqual(set(config.OCR_FIELD_CHARACTERS), keys)

    def test_a_printed_answer_is_kept(self):
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("symbol", "き"), "き")
            self.assertEqual(clean("what_doing", "④"), "4")
            self.assertEqual(clean("map_symbol", "G"), "G")

    def test_the_ring_comes_off_a_printed_answer(self):
        # The ring is the sheet saying "pick this one", not part of the answer,
        # so the xlsx gets the character inside it. 何してた's answers are
        # printed ①-⑩ and reach the operator as 1-10.
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("what_doing", "④"), "4")
            self.assertEqual(clean("what_doing", "⑩"), "10")
            self.assertEqual(clean("what_doing", "4"), "4")
            # A ring the handbook does not print, but a model might return.
            self.assertEqual(clean("what_doing", "❹"), "4")

    def test_the_ring_comes_off_japanese_in_a_free_column(self):
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("bug_name", "㋐ゲハ"), "アゲハ")
            self.assertEqual(clean("notice", "㊂びきいた"), "三びきいた")

    def test_a_value_that_is_not_a_printed_answer_is_dropped(self):
        # Each of these is a perfectly ordinary character of its own column's
        # character set, and none of them is an answer the sheet offers. 記号
        # runs あ to け, 何してた ① to ⑩, マップ記号 A to N.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("symbol", "し"), "")
            self.assertEqual(clean("what_doing", "⑪"), "")
            self.assertEqual(clean("map_symbol", "Z"), "")

    def test_a_column_answered_off_a_list_takes_one_answer_not_two(self):
        # Two circled kana in 記号 is one of them bleeding in from a neighbour
        # or a row, and joining them is a value nobody wrote.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("symbol", "きく"), "")

    def test_the_printed_answers_are_matched_without_letter_case(self):
        # The handbook prints its map symbols as capitals, so a lowercase
        # reading of one names the same symbol. What comes back is still the
        # engine's own value: the list says what counts as an answer, not how
        # it is spelled.
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("map_symbol", "g"), "g")
            self.assertEqual(clean("map_symbol", "ⓖ"), "g")

    def test_every_printed_answer_survives_its_own_column(self):
        # The tables have to agree: a character set that rejects one of its
        # column's own printed answers would empty every cell holding it. What
        # comes back is the answer with its ring taken off, which is the same
        # answer - ① is the first one whether or not it is still in a circle.
        for field, choices in config.OCR_FIELD_CHOICES.items():
            for answer in choices:
                with self.subTest(field=field, answer=answer):
                    self.assertEqual(clean(field, answer), fold(field, answer))
                    self.assertNotEqual(clean(field, answer), "")

    def test_a_numbered_answer_without_its_ring_is_the_same_answer(self):
        # Measured, not assumed: 4 of the 13 何してた cells in output/ came back
        # as a bare 4. Blanking those would lose a value that was read
        # correctly, which costs 5.1 more than the odd spelling does.
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("what_doing", "4"), "4")
            self.assertEqual(clean("what_doing", "10"), "10")

    def test_a_numbered_answer_off_the_end_of_the_list_is_still_dropped(self):
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("what_doing", "0"), "")
            self.assertEqual(clean("what_doing", "11"), "")

    def test_a_ringed_digit_is_not_accepted_by_a_month(self):
        # The spelling table is read field by field. 何してた sits one column
        # from 見つけた月, so the day a fold makes 4 and ④ interchangeable
        # everywhere is the day a column bleed becomes a month.
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            self.assertEqual(clean("found_month", "④"), "")

    def test_every_column_answered_off_a_list_is_a_real_field(self):
        keys = {field for field, _ in config.OCR_FIELDS}

        self.assertLessEqual(set(config.OCR_FIELD_CHOICES), keys)

    def test_every_other_spelling_names_an_answer_of_its_own_column(self):
        for field, spellings in config.OCR_FIELD_SPELLINGS.items():
            with self.subTest(field=field):
                self.assertIn(field, config.OCR_FIELD_CHOICES)
                self.assertLessEqual(
                    set(spellings.values()), set(config.OCR_FIELD_CHOICES[field])
                )


class CheckTest(unittest.TestCase):
    """check answers with the value and the reason; clean logs the reason.

    The split exists so a finding can be returned rather than logged: the
    caller is on another machine, and logging is not a way to send anybody
    anything. What check returns as a value is what clean returns, which is what
    the desktop pipeline returns - CleanTest below is the whole of that, lifted
    unchanged. (7.2)
    """

    def test_a_value_that_stands_has_no_reason(self):
        self.assertEqual(check("found_month", "7"), ("7", ""))

    def test_a_character_set_rejection_says_so(self):
        value, reason = check("location_chome", "山")

        self.assertEqual(value, "")
        self.assertIn("outside the field character set", reason)
        # The value it threw away, so an operator can see what was read.
        self.assertIn("山", reason)

    def test_an_answer_off_the_printed_list_says_so(self):
        value, reason = check("symbol", "ん")

        self.assertEqual(value, "")
        self.assertIn("not one of the printed answers", reason)

    def test_clean_logs_exactly_what_check_returned(self):
        with self.assertLogs("creature_ocr_server.ocr", level="WARNING") as logs:
            clean("location_chome", "山", where="row 2: ")

        _, reason = check("location_chome", "山")
        self.assertIn(f"row 2: location_chome: {reason}", "".join(logs.output))

    def test_a_value_that_stands_logs_nothing(self):
        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            clean("bug_name", "シオカラトンボ")


class ChecksFingerprintTest(unittest.TestCase):
    """What a client keys its cache on when the checks are somebody else's.

    The desktop pipeline caches an engine's boxes, before any check has run, so
    re-tuning a character set there invalidates nothing and needs to invalidate
    nothing. A client of this server caches finished rows, so a change here has
    to be visible to it or it will serve rows read under the old rules forever.
    (3.2, 4.2, 7.2)
    """

    def test_it_is_stable_across_calls(self):
        self.assertEqual(checks_fingerprint(), checks_fingerprint())

    def test_it_is_short_enough_to_read(self):
        self.assertEqual(len(checks_fingerprint()), 8)

    def test_widening_a_character_set_changes_it(self):
        before = checks_fingerprint()
        widened = dict(config.OCR_FIELD_CHARACTERS)
        widened["found_month"] = string.digits + "x"

        with patch.object(config, "OCR_FIELD_CHARACTERS", widened):
            after = checks_fingerprint()

        self.assertNotEqual(before, after)

    def test_another_printed_answer_changes_it(self):
        before = checks_fingerprint()
        listed = dict(config.MAP_SYMBOL_CHOICES)
        listed["O"] = "新しい記号"
        choices = dict(config.OCR_FIELD_CHOICES) | {"map_symbol": listed}

        with patch.object(config, "OCR_FIELD_CHOICES", choices):
            after = checks_fingerprint()

        self.assertNotEqual(before, after)

    def test_it_does_not_move_when_only_the_prompt_would(self):
        # The bug-name hints steer the model and no check reads them, so they
        # belong to the prompt fingerprint and not to this one. Counting them
        # twice would re-read every cached page for a reading aid.
        before = checks_fingerprint()

        with patch.object(config, "BUG_NAME_HINTS", ("カブトムシ",)):
            after = checks_fingerprint()

        self.assertEqual(before, after)


class SheetDefinitionTest(unittest.TestCase):
    """The definition GET /v1/sheet publishes, so a client needs no copy."""

    def test_it_serialises(self):
        json.dumps(config.sheet_definition(), ensure_ascii=False)

    def test_it_names_every_column_in_printed_order(self):
        listed = [field["key"] for field in config.sheet_definition()["fields"]]

        self.assertEqual(listed, FIELDS)

    def test_it_carries_the_headers_as_printed(self):
        listed = config.sheet_definition()["fields"]

        self.assertEqual(dict(config.OCR_FIELDS)[listed[0]["key"]], listed[0]["header"])

    def test_the_same_definition_serialises_the_same_way_twice(self):
        # It is hashed, so ordering has to be a promise rather than an accident.
        first = json.dumps(config.sheet_definition(), sort_keys=True)
        second = json.dumps(config.sheet_definition(), sort_keys=True)

        self.assertEqual(first, second)


class PairedFieldTest(unittest.TestCase):
    """3.1 pairs a printed answer with the child's own words for it.

    Read together the two catch a misread neither shows alone. Reported only:
    which of the two is wrong is not knowable from the page. (3.1, 5.2-2)
    """

    def test_the_sheets_own_example_row_agrees(self):
        # 記号 き is 地面で and the printed example answers it しめった地面;
        # 場所の名前 芝公園 is a 大きな公園, マップ記号 G. Neither pair is an
        # equality, which is why the check is on words and not on the value.
        example = {
            "no": "1",
            "symbol": "き",
            "where": "しめった地面",
            "location_name": "芝公園",
            "map_symbol": "G",
        }

        self.assertEqual(paired_field_problems(example), [])

    def test_a_wording_naming_another_category_is_reported(self):
        row = {"no": "3", "location_name": "芝公園", "map_symbol": "B"}

        problems = paired_field_problems(row)

        self.assertEqual(len(problems), 1)
        self.assertIn("map_symbol B", problems[0])
        self.assertIn("芝公園", problems[0])
        self.assertIn("G", problems[0])

    def test_a_wording_naming_no_category_is_not_a_disagreement(self):
        # Most wordings name none. Complaining about those would bury the ones
        # that do under a warning on every filled row.
        row = {"no": "2", "location_name": "みなと図書室", "map_symbol": "B"}

        self.assertEqual(paired_field_problems(row), [])

    def test_a_catch_all_answer_agrees_with_anything(self):
        # け is その他: any wording at all is a legitimate answer for it.
        row = {"no": "4", "symbol": "け", "where": "池のそば"}

        self.assertEqual(paired_field_problems(row), [])

    def test_a_column_that_came_back_blank_is_not_a_disagreement(self):
        self.assertEqual(paired_field_problems({"no": "5", "symbol": "き"}), [])
        self.assertEqual(
            paired_field_problems({"no": "5", "where": "しめった地面"}), []
        )

    def test_nothing_is_rewritten(self):
        row = {"no": "6", "symbol": "く", "where": "高い木の上"}

        paired_field_problems(row)

        self.assertEqual(row["symbol"], "く")
        self.assertEqual(row["where"], "高い木の上")

    def test_every_paired_column_is_a_real_field(self):
        keys = {field for field, _ in config.OCR_FIELDS}

        for letter_field, wording_field, words in config.OCR_PAIRED_FIELDS:
            with self.subTest(pair=(letter_field, wording_field)):
                self.assertIn(letter_field, keys)
                self.assertIn(wording_field, keys)
                # Every letter the words are listed under has to be an answer
                # the column actually offers, or the check silently does
                # nothing for it.
                self.assertEqual(
                    set(words), set(config.OCR_FIELD_CHOICES[letter_field])
                )


class RowGapsTest(unittest.TestCase):
    """A hole in a filled row is where the engine most likely missed. (7.2)"""

    def test_a_row_the_engine_read_nothing_in_has_no_gaps(self):
        # Most sheets stop at three of eight rows. An unused row is not a
        # finding, and 5.2-1 makes every column but `no` optional.
        self.assertEqual(row_gaps({field: "" for field in FIELDS}), [])

    def test_a_completely_filled_row_has_no_gaps(self):
        self.assertEqual(row_gaps(dict(FILLED_ROW)), [])

    def test_a_hole_in_a_filled_row_is_reported(self):
        # The real case: four 何してた cells came back empty across one run and
        # nothing else in the pipeline could see them.
        read = FILLED_ROW | {"what_doing": ""}

        self.assertEqual(row_gaps(read), ["what_doing"])

    def test_the_gaps_come_in_the_order_the_columns_are_printed_in(self):
        read = FILLED_ROW | {"notice": "", "symbol": ""}

        self.assertEqual(row_gaps(read), ["symbol", "notice"])


class PageConfidenceTest(unittest.TestCase):
    """The one per-page quality number a batch produces. (7.2)"""

    def test_a_page_nothing_was_wrong_with_is_whole(self):
        self.assertEqual(page_confidence(20, 0, 0), 1.0)

    def test_a_page_that_offered_no_values_has_no_score(self):
        # Never 1.0. A page the engine returned nothing for is either a blank
        # sheet or a filled one it failed on, nothing here can tell which, and
        # the second is the page in the batch most needing to be opened. The
        # number an operator scans by must not be at its best on it.
        self.assertIsNone(page_confidence(0, 0, 0))

    def test_a_rejected_value_costs_its_share(self):
        self.assertAlmostEqual(page_confidence(10, 1, 0), 0.9)

    def test_a_disagreement_costs_the_same_as_a_rejection(self):
        self.assertAlmostEqual(page_confidence(10, 0, 1), 0.9)
        self.assertAlmostEqual(page_confidence(10, 1, 1), 0.8)

    def test_it_never_goes_below_zero(self):
        # A row can disagree twice over while holding two values, so the two
        # counts are not bounded by each other.
        self.assertEqual(page_confidence(2, 2, 4), 0.0)


class CellProblemsTest(unittest.TestCase):
    """Which cells of a finished row want an eye, worked out from the row. (7.2)

    From the row and nothing else, so a page reopened tomorrow is marked exactly
    like one just read, and a cell an operator has corrected stops being marked
    without anyone having to remember it was.
    """

    def filled(self, **values):
        row = {field: "x" for field, _ in config.OCR_FIELDS}
        row["no"] = "1"
        row.update(values)
        return row

    def test_a_blank_row_has_no_problems(self):
        # `no` is there whether the child wrote anything or not, so a row that
        # is only a row number is blank - not eleven missing values.
        blank = {"no": "2"} | {field: "" for field, _ in config.OCR_FIELDS}

        self.assertEqual(cell_problems(blank), {})

    def test_a_hole_in_a_filled_row_is_marked_missing(self):
        row = self.filled(notice="")

        self.assertEqual(cell_problems(row), {"notice": CELL_MISSING})

    def test_a_row_with_nothing_wrong_is_not_marked(self):
        self.assertEqual(cell_problems(self.filled()), {})

    def test_a_disagreement_marks_both_halves_of_the_pair(self):
        # Which of the two is the misread is not knowable from the page, so
        # marking one of them would be the guess 5.2-2 forbids.
        letter, wording = disagreeing_pair()

        problems = cell_problems(self.filled(symbol=letter, where=wording))

        self.assertEqual(problems["symbol"], CELL_DISAGREEMENT)
        self.assertEqual(problems["where"], CELL_DISAGREEMENT)

    def test_filling_a_hole_in_clears_its_mark(self):
        # What makes it safe to recompute on every keystroke.
        row = self.filled(notice="")
        self.assertIn("notice", cell_problems(row))

        row["notice"] = "きれいだった"

        self.assertEqual(cell_problems(row), {})


class PageReportTest(unittest.TestCase):
    """The same counts as an object, so a front end can show them too. (7.2)"""

    def setUp(self):
        quieten(self)

    def test_it_reads_as_the_log_line(self):
        self.assertEqual(
            str(PageReport(32, 7, 0, 1, 0)),
            "confidence 76% (32 value(s), 7 rejected, 0 disagreement(s), "
            "1 gap(s), 0 unsure)",
        )

    def test_it_carries_the_cells_only_the_run_can_know_about(self):
        # A rejected value looks like a blank one once it has been emptied, and
        # an unsure reading looks like any other value, so neither can be found
        # again from the finished row.
        engine = RecordingEngine([box(1, "found_month", "abc")])

        result = read_page_image(PAGE, engine)

        self.assertEqual(result.report.cells[(1, "found_month")], CELL_REJECTED)

    def test_a_page_with_no_score_says_so_in_words(self):
        report = PageReport(0, 0, 0, 0, 0)

        self.assertIsNone(report.score)
        # Empty rather than a percentage: a column has no room to explain, and
        # 0% would be a lie about a page that may simply be blank.
        self.assertEqual(report.percent, "")
        self.assertIn("none, the page produced no values", str(report))


class PageCellsTest(unittest.TestCase):
    """Both halves of what is wrong with a page, in one table.

    A client used to work the recomputable half out for itself. It still has to
    - an operator editing a cell needs the marks to follow the keystroke - but
    for a page just read, the server has both halves and sending only one would
    make the client guess at the other.
    """

    def setUp(self):
        quieten(self)

    def test_a_hole_in_a_filled_row_is_marked(self):
        result = read_page_image(PAGE, RecordingEngine(filled_row(1, notice="")))

        marks = page_cells(result.rows, result.report)

        self.assertEqual(marks[(1, "notice")], CELL_MISSING)

    def test_a_rejected_cell_beats_the_hole_it_leaves_behind(self):
        # It came back and was thrown away. Calling that a gap would lose the
        # one fact worth knowing about it.
        boxes = filled_row(1, location_chome="山")
        result = read_page_image(PAGE, RecordingEngine(boxes))

        marks = page_cells(result.rows, result.report)

        self.assertEqual(marks[(1, "location_chome")], CELL_REJECTED)

    def test_an_unsure_cell_is_marked(self):
        boxes = filled_row(1, where="駐車場の花だん", unsure=["where"])

        result = read_page_image(PAGE, RecordingEngine(boxes))

        marks = page_cells(result.rows, result.report)

        self.assertEqual(marks[(1, "where")], CELL_UNSURE)

    def test_both_halves_of_a_disagreeing_pair_are_marked(self):
        letter, wording = disagreeing_pair()
        boxes = filled_row(1, symbol=letter, where=wording)

        result = read_page_image(PAGE, RecordingEngine(boxes))

        marks = page_cells(result.rows, result.report)
        self.assertEqual(marks[(1, "symbol")], CELL_DISAGREEMENT)
        self.assertEqual(marks[(1, "where")], CELL_DISAGREEMENT)

    def test_a_clean_page_has_nothing_marked(self):
        result = read_page_image(PAGE, RecordingEngine(filled_row(1)))

        self.assertEqual(page_cells(result.rows, result.report), {})


class ReadPageImageTest(unittest.TestCase):
    """The desktop pipeline's ReadPageTest, with a page of bytes for a path."""

    def setUp(self):
        quieten(self)

    def test_it_calls_the_engine_once_for_the_whole_page(self):
        engine = RecordingEngine()

        read_page_image(PAGE, engine)

        self.assertEqual(engine.calls, 1)

    def test_it_sends_the_page_image_itself(self):
        engine = RecordingEngine()

        read_page_image(PAGE, engine)

        self.assertEqual(engine.images, [PAGE])

    def test_it_returns_one_object_per_row(self):
        result = read_page_image(PAGE, RecordingEngine())

        self.assertEqual(len(result.rows), config.ROWS_PER_PAGE)

    def test_the_report_counts_what_the_page_found(self):
        engine = RecordingEngine([box(1, "bug_name", "カブトムシ")])

        result = read_page_image(PAGE, engine)

        self.assertEqual(len(result.rows), config.ROWS_PER_PAGE)
        self.assertEqual((result.report.values, result.report.rejected), (1, 0))

    def test_the_row_number_is_not_ocr_d(self):
        result = read_page_image(PAGE, RecordingEngine())

        self.assertEqual(
            [row["no"] for row in result.rows], [str(n) for n in range(1, 9)]
        )

    def test_every_row_carries_every_field(self):
        result = read_page_image(PAGE, RecordingEngine())

        for row in result.rows:
            self.assertEqual(set(row), {"no", *FIELDS})

    def test_a_blank_page_gives_eight_empty_rows(self):
        result = read_page_image(PAGE, RecordingEngine())

        for row in result.rows:
            self.assertEqual({row[field] for field in FIELDS}, {""})

    def test_a_value_reaches_the_row_it_was_written_in(self):
        engine = RecordingEngine(
            [box(3, "bug_name", "ハラビロカマキリ"), box(3, "found_day", "7")]
        )

        rows = read_page_image(PAGE, engine).rows

        self.assertEqual(rows[2]["bug_name"], "ハラビロカマキリ")
        self.assertEqual(rows[2]["found_day"], "7")
        self.assertEqual(rows[1]["bug_name"], "")

    def test_a_misread_outside_the_field_character_set_is_dropped(self):
        engine = RecordingEngine(
            [box(2, "location_chome", "山"), box(2, "map_symbol", "더")]
        )

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            rows = read_page_image(PAGE, engine).rows

        self.assertEqual(rows[1]["location_chome"], "")
        self.assertEqual(rows[1]["map_symbol"], "")

    def test_a_circled_map_letter_reaches_the_row_as_the_letter(self):
        engine = RecordingEngine(filled_row(4, map_symbol="Ⓖ"))

        with self.assertNoLogs("creature_ocr_server.ocr", level="WARNING"):
            rows = read_page_image(PAGE, engine).rows

        self.assertEqual(rows[3]["map_symbol"], "G")

    def test_a_page_that_fails_every_retry_is_raised_not_half_returned(self):
        engine = RecordingEngine(error=OCRError("down"))

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError):
                read_page_image(PAGE, engine, sleep=lambda delay: None)


class FindingsTest(unittest.TestCase):
    """What the client writes into ocr_error.txt, and what shape it arrives in.

    The desktop pipeline collected these off its own logger. Here they have to
    be returned, because the machine that writes the error file is not this one.
    Each line names the row and the column and carries no page in front of it -
    the client prepends that, because only the client knows what file the image
    came from. A batch of a thousand sheets whose findings all read the same is
    the exact failure that file exists to prevent. (7.2)
    """

    def setUp(self):
        quieten(self)

    def test_a_rejection_names_the_row_and_the_column(self):
        engine = RecordingEngine(filled_row(2, location_chome="山"))

        result = read_page_image(PAGE, engine)

        self.assertEqual(len(result.findings), 1)
        self.assertTrue(result.findings[0].startswith("row 2: location_chome: "))

    def test_no_finding_carries_a_page_in_front_of_it(self):
        boxes = filled_row(2, location_chome="山", what_doing="")
        engine = RecordingEngine(boxes)

        result = read_page_image(PAGE, engine)

        for finding in result.findings:
            self.assertTrue(finding.startswith("row "), finding)

    def test_a_hole_in_a_filled_row_is_reported(self):
        # The four bare-4 何してた cells that a whole run lost in silence.
        engine = RecordingEngine(filled_row(2, what_doing=""))

        result = read_page_image(PAGE, engine)

        self.assertIn(
            "row 2: nothing came back for what_doing, but the row is filled in",
            result.findings,
        )

    def test_an_unused_row_is_not_reported_as_a_hole(self):
        # Most sheets stop at three of eight rows. Complaining about the other
        # five would bury every real finding.
        result = read_page_image(PAGE, RecordingEngine(filled_row(1)))

        self.assertEqual(result.findings, [])

    def test_a_strained_reading_is_reported_and_kept(self):
        # 花だん where the paper says 花ばたけ. Ordinary Japanese, passes every
        # other check, so the engine saying it strained is the only signal. (7.2)
        boxes = filled_row(2, where="駐車場の花だん", unsure=["where"])

        result = read_page_image(PAGE, RecordingEngine(boxes))

        self.assertIn(
            "row 2: where: the engine could not read '駐車場の花だん' cleanly",
            result.findings,
        )
        self.assertEqual(result.rows[1]["where"], "駐車場の花だん")

    def test_a_disagreeing_pair_is_reported_and_left_alone(self):
        letter, wording = disagreeing_pair()
        boxes = filled_row(1, symbol=letter, where=wording)

        result = read_page_image(PAGE, RecordingEngine(boxes))

        self.assertTrue(any("does not agree with" in f for f in result.findings))
        self.assertEqual(result.rows[0]["symbol"], letter)
        self.assertEqual(result.rows[0]["where"], wording)

    def test_the_engine_speaks_first(self):
        # A note about the response as a whole belongs above the notes about
        # one cell of it, because it is what explains them.
        short = "the page came back without row(s) [8]"
        engine = RecordingEngine(filled_row(2, what_doing=""), notes=[short])

        result = read_page_image(PAGE, engine)

        self.assertEqual(result.findings[0], short)

    def test_the_engine_notes_also_reach_this_log(self):
        # Returned to the caller and left behind here: a batch that went wrong
        # is diagnosed on this machine, and the client may be long gone.
        engine = RecordingEngine(notes=["dropped a second row numbered 3"])

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING") as logs:
            read_page_image(PAGE, engine)

        self.assertIn("dropped a second row numbered 3", "".join(logs.output))

    def test_a_clean_page_complains_about_nothing(self):
        result = read_page_image(PAGE, RecordingEngine(filled_row(1)))

        self.assertEqual(result.findings, [])
        self.assertEqual(result.report.percent, "100%")


class UsageTest(unittest.TestCase):
    """What a run cost has to be readable off the log. (6.1, 7.1)"""

    def test_nothing_spent_says_so(self):
        self.assertEqual(str(Usage()), "no calls")

    def test_an_engine_that_reports_no_tokens_still_reports_time(self):
        self.assertEqual(str(Usage(calls=1, seconds=0.83)), "1 call in 0.8s")

    def test_it_counts_calls_in_the_plural(self):
        self.assertIn("5 calls", str(Usage(calls=5, seconds=21.0)))

    def test_tokens_are_shown_when_the_engine_reports_them(self):
        spent = str(Usage(calls=1, seconds=4.2, prompt_tokens=1234, output_tokens=567))

        self.assertIn("1,234 in", spent)
        self.assertIn("567 out", spent)

    def test_thinking_is_called_out_separately(self):
        # It is billed as output and counted there, but it is also the budget
        # that vanishes when a page comes back empty.
        spent = str(
            Usage(
                calls=1,
                seconds=4.2,
                prompt_tokens=10,
                output_tokens=900,
                thought_tokens=300,
            )
        )

        self.assertIn("900 out", spent)
        self.assertIn("300", spent)
        self.assertIn("thinking", spent)

    def test_tokens_are_never_hidden_by_an_uncounted_call(self):
        # What the engine reports before recognize_with_retry counts the call.
        # An earlier version short-circuited on calls == 0 and printed
        # "no calls" over a real token count.
        spent = str(Usage(prompt_tokens=1234, output_tokens=867))

        self.assertIn("1,234 in", spent)
        self.assertNotEqual(spent, "no calls")

    def test_totals_add_up(self):
        total = Usage(1, 2.0, 10, 20, 5) + Usage(2, 3.0, 30, 40, 5)

        self.assertEqual(total, Usage(3, 5.0, 40, 60, 10))

    def test_subtracting_gives_what_one_page_spent(self):
        before = Usage(4, 8.0, 100, 200, 0)
        after = before + Usage(1, 2.0, 10, 20, 0)

        self.assertEqual(after - before, Usage(1, 2.0, 10, 20, 0))


class ConfidenceLogTest(unittest.TestCase):
    """Every page logs a confidence, whether or not anything was wrong with it."""

    def setUp(self):
        quieten(self)

    def test_every_page_logs_one(self):
        # Every page, not only a bad one: a number that appears only when
        # something went wrong cannot be scanned down the log of a batch.
        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, RecordingEngine(filled_row(1)))

        self.assertRegex("".join(logs.output), r"confidence 100% \(11 value")

    def test_a_rejected_value_shows_up_in_it(self):
        engine = RecordingEngine(filled_row(2, location_chome="山"))

        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, engine)

        # One fault, not two: the cell is a rejection, and a rejection is not
        # also counted as a gap.
        self.assertIn(
            "confidence 91% (11 value(s), 1 rejected, 0 disagreement(s), "
            "0 gap(s), 0 unsure)",
            "".join(logs.output),
        )

    def test_a_hole_is_scored(self):
        engine = RecordingEngine(filled_row(2, what_doing=""))

        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, engine)

        line = "".join(logs.output)
        self.assertIn("confidence 91% (10 value(s), 0 rejected", line)
        self.assertIn("1 gap(s)", line)

    def test_a_cell_that_was_rejected_is_not_also_counted_unsure(self):
        boxes = filled_row(2, location_chome="山", unsure=FIELDS)

        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, RecordingEngine(boxes))

        # Eleven boxes, all flagged, one of them rejected: ten unsure, not
        # eleven. One cell, one fault.
        self.assertIn(
            "1 rejected, 0 disagreement(s), 0 gap(s), 10 unsure",
            "".join(logs.output),
        )

    def test_a_page_that_came_back_with_nothing_is_never_reported_as_perfect(self):
        # The page most worth opening must not read like the best page in the
        # batch. It has no score at all, and the line says why in words.
        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, RecordingEngine())

        line = "".join(logs.output)
        self.assertIn("confidence none, the page produced no values", line)
        self.assertIn("(0 value(s)", line)
        self.assertNotIn("100%", line)


class UsageRecordingTest(unittest.TestCase):
    """Every call is counted, including the ones that failed.

    Rewritten for this server's contract. The desktop pipeline reads a running
    total off the engine; here the total comes back from the call, because one
    engine answers many requests at once and an attribute would mix them.
    """

    def setUp(self):
        quieten(self)

    def test_one_page_counts_one_call(self):
        found = recognize_with_retry(RecordingEngine(), PAGE, sleep=Recorder())

        self.assertEqual(found.usage.calls, 1)

    def test_what_the_engine_reported_is_added_to_what_the_call_cost(self):
        engine = RecordingEngine(cost=Usage(prompt_tokens=1877, output_tokens=2254))

        found = recognize_with_retry(engine, PAGE, sleep=Recorder())

        self.assertEqual(found.usage.calls, 1)
        self.assertEqual(found.usage.prompt_tokens, 1877)
        self.assertEqual(found.usage.output_tokens, 2254)

    def test_a_retried_call_is_still_a_billed_call(self):
        # Three retries after the first attempt is four requests, and the bill
        # says four whether or not any of them worked. (6.4)
        engine = RecordingEngine(error=OCRError("down"))

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError) as raised:
                read_page_image(PAGE, engine, sleep=lambda delay: None)

        self.assertEqual(raised.exception.usage.calls, config.OCR_RETRIES + 1)

    def test_the_tokens_an_unusable_response_cost_are_not_lost(self):
        # It was sent, so it was billed. An engine that knows says so on the
        # exception; four failed attempts is four times what one of them cost.
        engine = RecordingEngine(error=OCRError("not json", Usage(prompt_tokens=1877)))

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError) as raised:
                read_page_image(PAGE, engine, sleep=lambda delay: None)

        self.assertEqual(raised.exception.usage.prompt_tokens, 1877 * 4)

    def test_an_engine_that_says_nothing_about_cost_still_counts_its_calls(self):
        engine = RecordingEngine(error=RuntimeError("boom"))

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError) as raised:
                recognize_with_retry(engine, PAGE, sleep=lambda delay: None)

        self.assertEqual(raised.exception.usage.calls, config.OCR_RETRIES + 1)
        self.assertEqual(raised.exception.usage.prompt_tokens, 0)

    def test_the_backoff_is_not_counted_as_call_time(self):
        # The clock stops when the call does. An earlier version slept inside
        # the except block, so the wait landed in the timing: at the default
        # backoff that is 1 + 2 + 4 seconds of doing nothing, reported as
        # engine latency. (6.1, 6.4)
        engine = RecordingEngine(error=OCRError("down"))
        waiting = 0.05

        with self.assertLogs("creature_ocr_server.ocr", level="WARNING"):
            with self.assertRaises(OCRError) as raised:
                recognize_with_retry(
                    engine,
                    PAGE,
                    retries=2,
                    backoff=0.01,
                    sleep=lambda delay: time.sleep(waiting),
                )

        self.assertEqual(raised.exception.usage.calls, 3)
        self.assertLess(raised.exception.usage.seconds, waiting)

    def test_the_time_a_call_took_is_recorded(self):
        found = recognize_with_retry(RecordingEngine(), PAGE, sleep=Recorder())

        self.assertGreaterEqual(found.usage.seconds, 0.0)

    def test_the_page_log_says_what_the_page_cost(self):
        engine = RecordingEngine([box(1, "symbol", "き")])

        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, engine)

        self.assertIn("1 call in", "".join(logs.output))

    def test_the_page_log_times_the_whole_page_not_just_the_call(self):
        # 6.1 budgets the page. Placing the boxes and checking them are not
        # free, and the gap between the two figures is the only place that
        # overhead shows up.
        engine = RecordingEngine([box(1, "symbol", "き")])

        with self.assertLogs("creature_ocr_server.ocr", level="INFO") as logs:
            read_page_image(PAGE, engine)

        self.assertRegex("".join(logs.output), r"in \d+\.\d\ds \(1 call in")


    def test_an_engine_keeps_nothing_between_calls(self):
        # The whole reason recognize returns what it cost. One engine answers
        # every request that names its model, so an engine that accumulated
        # into an attribute would report one caller's tokens to another - and
        # there is no attribute left to accumulate into.
        engine = RecordingEngine(cost=Usage(prompt_tokens=5))

        first = recognize_with_retry(engine, PAGE, sleep=Recorder())
        second = recognize_with_retry(engine, PAGE, sleep=Recorder())

        self.assertEqual(first.usage.prompt_tokens, 5)
        self.assertEqual(second.usage.prompt_tokens, 5)
        self.assertFalse(hasattr(OCREngine, "usage"))


if __name__ == "__main__":
    unittest.main()
