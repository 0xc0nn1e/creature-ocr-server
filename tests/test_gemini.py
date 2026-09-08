"""Gemini engine tests.

No network and no cloud SDK: the engine is exercised through the one loader
function that imports its SDK, so the suite passes with google-genai not
installed. The prompt and the schema are plain data and are tested with no
engine at all.

Carried over from the desktop repository. What changed is what this server
changed: recognize hands back a Reading instead of a list and a running total,
text_boxes returns what it complained about instead of logging it, and the
registry can say whether an engine is usable without building one.
"""

import json
import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from creature_ocr_server import config, engines
from creature_ocr_server.engines.gemini import (
    GeminiEngine,
    build_prompt,
    build_prompt_v2,
    grid_fingerprint,
    parse_page_v2,
    parse_rows,
    prompt_fingerprint,
    response_schema,
    response_schema_v2,
    text_boxes,
    text_boxes_v2,
    tokens,
)
from creature_ocr_server.grid import cell_at_v2, header_at_v2
from creature_ocr_server.ocr import (
    OCRError,
    Usage,
    assign_to_cells,
    assign_to_header,
    checks_fingerprint,
)

PAGE = b"a whole page of PNG"

FIELDS = [key for key, _ in config.OCR_FIELDS]


class FakeApiError(Exception):
    """Stands in for google.genai.errors.APIError."""


def fake_usage(prompt=0, answer=0, thoughts=0):
    """usage_metadata as the SDK shapes it."""
    return SimpleNamespace(
        prompt_token_count=prompt,
        candidates_token_count=answer,
        thoughts_token_count=thoughts,
        total_token_count=prompt + answer + thoughts,
    )


def fake_response(rows, text=None, finish_reason="STOP", usage=None):
    """A response carrying the JSON a page would come back as."""
    return SimpleNamespace(
        text=json.dumps(rows) if text is None else text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        usage_metadata=usage,
    )


def fake_sdk(response=None):
    """A Gen AI SDK triple whose client answers with one response."""
    genai = MagicMock()
    client = genai.Client.return_value
    client.models.generate_content.return_value = response or fake_response([])
    return genai, MagicMock(), FakeApiError


def row(number, **fields):
    """One row object as the model is asked to return it."""
    return {"no": number, **fields}


def boxes(rows):
    """The positioned text a response turns into."""
    found, _ = text_boxes(rows)
    return found


def notes(rows):
    """What the engine had to work around in a response.

    Returned rather than logged, because the desktop pipeline collected these
    off its own logger into ocr_error.txt and this one cannot: the operator who
    needs to know a row was dropped is on another machine. (6.4, 7.2)
    """
    _, found = text_boxes(rows)
    return found


def quieten(case):
    """Silence the engine for one test, for anything that still logs."""
    logger = logging.getLogger("creature_ocr_server.engines.gemini")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    case.addCleanup(logger.setLevel, previous)


class ResponseSchemaTest(unittest.TestCase):
    """The schema is what 5.2-1 is actually enforced by."""

    def setUp(self):
        self.item = response_schema()["items"]

    def test_no_is_the_only_required_field(self):
        # 5.2-1: everything else optional, so a blank cell can stay blank
        # instead of being invented to satisfy the schema.
        self.assertEqual(self.item["required"], ["no"])

    def test_it_asks_for_every_field_and_no_others(self):
        self.assertEqual(
            sorted(set(self.item["properties"]) - {"no", "unsure"}), sorted(FIELDS)
        )

    def test_it_asks_which_cells_were_hard_to_read(self):
        # The only place a strained reading can be reported from: by the time
        # it is text it looks exactly like a confident one, and 花だん for
        # 花ばたけ passes every check downstream. (7.2)
        unsure = self.item["properties"]["unsure"]

        self.assertEqual(unsure["type"], "array")
        self.assertEqual(unsure["items"]["type"], "string")

    def test_only_a_real_column_can_be_called_unsure(self):
        self.assertEqual(self.item["properties"]["unsure"]["items"]["enum"], FIELDS)

    def test_every_field_but_the_row_number_is_a_string(self):
        for field in FIELDS:
            self.assertEqual(self.item["properties"][field]["type"], "string")

    def test_the_row_number_is_an_integer_bounded_by_the_sheet(self):
        number = self.item["properties"]["no"]

        self.assertEqual(number["type"], "integer")
        self.assertEqual(number["minimum"], 1)
        self.assertEqual(number["maximum"], config.ROWS_PER_PAGE)

    def test_the_keys_are_ordered_the_way_the_paper_is(self):
        # unsure last: it is a judgement about the row, made after reading it.
        self.assertEqual(
            self.item["propertyOrdering"], ["no", *FIELDS, "unsure"]
        )

    def test_a_page_comes_back_as_a_list_of_rows(self):
        self.assertEqual(response_schema()["type"], "array")


class BuildPromptTest(unittest.TestCase):
    """The prompt is where 5.2-2 lives, and it is built from config."""

    def setUp(self):
        self.prompt = build_prompt()

    def test_it_names_every_column_by_its_printed_heading(self):
        for key, header in config.OCR_FIELDS:
            self.assertIn(header, self.prompt)
            self.assertIn(key, self.prompt)

    def test_the_columns_come_in_the_order_they_are_printed_in(self):
        found = [self.prompt.index(header) for _, header in config.OCR_FIELDS]

        self.assertEqual(found, sorted(found))

    def test_it_explains_the_printed_row_number_column(self):
        # Without this the model numbers rows by its own counter, and placing a
        # row by the number it reports stops meaning anything. (5.1)
        self.assertIn("leftmost printed column", self.prompt)
        self.assertIn("`no`", self.prompt)

    def test_it_states_the_rules_that_5_2_2_asks_for(self):
        for rule in (
            "Never take a value from another row",
            "blank on the paper is an empty string",
            "cannot read is an empty string",
            "Never guess",
            "handwriting is the same",
        ):
            self.assertIn(rule, self.prompt)

    def test_it_forbids_translating_what_it_reads(self):
        self.assertIn("Do not translate", self.prompt)

    def test_it_carries_the_character_sets_from_the_config(self):
        for field in config.OCR_FIELD_CHARACTERS:
            self.assertIn(field, self.prompt)

    def test_it_carries_the_printed_answer_lists_from_the_config(self):
        for field, choices in config.OCR_FIELD_CHOICES.items():
            with self.subTest(field=field):
                self.assertIn(field, self.prompt)
                for answer, meaning in choices.items():
                    self.assertIn(answer, self.prompt)
                    # The meaning is the point: a single kana is hard to read
                    # alone and easy beside the words that explain it.
                    self.assertIn(meaning, self.prompt)

    def test_it_carries_the_other_spellings_the_checks_accept(self):
        # The prompt must not ask for less than ocr.clean accepts: told only
        # the printed list, the model reads 何してた's un-ringed number, obeys
        # "anything else is an empty string", and returns nothing. That cost
        # three cells on one page of five.
        for field, spellings in config.OCR_FIELD_SPELLINGS.items():
            for spelling in spellings:
                with self.subTest(field=field, spelling=spelling):
                    self.assertIn(spelling, self.prompt)

    def test_it_forbids_copying_a_meaning_into_a_cell(self):
        # The meanings and the species list are reading aids. Left unguarded
        # they are a vocabulary the model can write a cell out of.
        self.assertIn("Never\ncopy a meaning into a cell", self.prompt)
        self.assertIn("full name of a real place", self.prompt)

    def test_it_says_which_columns_answer_the_same_thing_twice(self):
        for letter_field, wording_field, _ in config.OCR_PAIRED_FIELDS:
            with self.subTest(pair=(letter_field, wording_field)):
                self.assertIn(letter_field, self.prompt)
                self.assertIn(wording_field, self.prompt)

    def test_it_forbids_correcting_one_paired_column_to_match_the_other(self):
        # The pairing is a reading aid. Left as "make them agree" it would
        # invent the value 5.2-2 forbids in whichever column reads worse.
        self.assertIn("return both as they stand and", self.prompt)
        self.assertIn("correct neither", self.prompt)

    def test_it_carries_last_years_names(self):
        for name in config.BUG_NAME_HINTS:
            self.assertIn(name, self.prompt)

    def test_it_forbids_replacing_handwriting_with_a_listed_name(self):
        # 7.2 caught Gemini "correcting" ハラビロオマキリ to ハラビロカマキリ
        # with no list in front of it. Handing it a species list makes that
        # easier, so the list arrives with the rule against it attached.
        self.assertIn("differs from a listed name by even one character", self.prompt)

    def test_it_warns_off_the_printed_example_row(self):
        # The crop cuts through the sheet's worked example, so the bottom of it
        # sits directly above row 1 - carrying a value for every column, all of
        # them on the lists above.
        self.assertIn("例", self.prompt)

    def test_it_asks_for_every_row_of_this_sheet(self):
        self.assertIn(str(config.ROWS_PER_PAGE), self.prompt)


class TextBoxesTest(unittest.TestCase):
    """Turning rows into boxes, and refusing to lose a page over one row."""

    def test_a_value_lands_in_its_own_cell(self):
        placed = assign_to_cells(boxes([row(2, bug_name="ショウリョウバッタ")]))

        self.assertEqual(placed[(2, "bug_name")], "ショウリョウバッタ")

    def test_a_row_is_placed_by_its_number_not_by_its_position(self):
        found = assign_to_cells(boxes([row(5, symbol="き"), row(2, symbol="ち")]))

        self.assertEqual(found[(5, "symbol")], "き")
        self.assertEqual(found[(2, "symbol")], "ち")

    def test_the_boxes_come_back_in_the_order_the_grid_reads_in(self):
        found = boxes([row(5, symbol="き"), row(2, symbol="ち")])

        self.assertEqual([b.text for b in found], ["ち", "き"])

    def test_a_flagged_cell_comes_back_flagged(self):
        found = boxes([row(1, where="駐車場の花だん", unsure=["where"])])

        self.assertEqual([b.text for b in found], ["駐車場の花だん"])
        self.assertTrue(found[0].unsure)

    def test_a_cell_nobody_flagged_is_not_flagged(self):
        found = boxes([row(1, where="しめった地面", symbol="き")])

        self.assertEqual([b.unsure for b in found], [False, False])

    def test_only_the_flagged_column_of_a_row_is_flagged(self):
        found = boxes([row(1, symbol="き", where="地面", unsure=["where"])])

        self.assertEqual(
            {b.text: b.unsure for b in found}, {"き": False, "地面": True}
        )

    def test_a_flag_on_a_column_this_sheet_does_not_have_is_dropped(self):
        found, said = text_boxes([row(1, symbol="き", unsure=["weather"])])

        self.assertEqual([b.unsure for b in found], [False])
        self.assertTrue(any("weather" in note for note in said))

    def test_a_flag_that_is_not_a_list_is_dropped_not_raised(self):
        # One malformed row must not cost the other seven. (5.1)
        found, said = text_boxes([row(1, symbol="き", unsure="where")])

        self.assertEqual([b.text for b in found], ["き"])
        self.assertTrue(any("unsure is not a list" in note for note in said))

    def test_an_empty_value_takes_no_box(self):
        self.assertEqual(boxes([row(1, bug_name="", symbol="  ")]), [])

    def test_a_missing_field_takes_no_box(self):
        self.assertEqual(boxes([row(1)]), [])

    def test_a_null_field_takes_no_box(self):
        self.assertEqual(boxes([row(1, notice=None)]), [])

    def test_a_number_is_read_as_the_text_it_stands_for(self):
        self.assertEqual([b.text for b in boxes([row(1, found_month=7)])], ["7"])

    def test_the_row_number_is_never_written_into_a_cell(self):
        found = assign_to_cells(boxes([row(3)]))

        self.assertEqual(set(found.values()), {""})

    def test_a_field_the_sheet_does_not_have_is_ignored(self):
        self.assertEqual(boxes([row(1, invented="なにか")]), [])

    def test_a_row_numbered_off_the_sheet_is_dropped_not_raised(self):
        found, said = text_boxes([row(99, symbol="き"), row(1, symbol="ち")])

        self.assertEqual([b.text for b in found], ["ち"])
        self.assertTrue(any("dropped row 99" in note for note in said))

    def test_a_row_numbered_with_something_that_is_not_a_number_is_dropped(self):
        found, said = text_boxes([row("no.3", symbol="き")])

        self.assertEqual(found, [])
        self.assertTrue(any("is not a row number" in note for note in said))

    def test_a_second_row_claiming_a_number_is_dropped_not_merged(self):
        # Cells are joined, so keeping both would concatenate two rows into one
        # value nobody wrote and blank the other row. (5.1)
        found, said = text_boxes(
            [
                row(3, bug_name="ショウリョウバッタ"),
                row(3, bug_name="ハラビロカマキリ"),
            ]
        )

        self.assertEqual(assign_to_cells(found)[(3, "bug_name")], "ショウリョウバッタ")
        self.assertIn("dropped a second row numbered 3", said)

    def test_a_row_that_is_not_an_object_is_dropped(self):
        found, said = text_boxes(["き"])

        self.assertEqual(found, [])
        self.assertTrue(any("not an object" in note for note in said))

    def test_a_short_answer_is_reported(self):
        self.assertTrue(any("without row(s)" in note for note in notes([row(1)])))

    def test_a_row_out_of_position_is_reported_but_kept(self):
        found, said = text_boxes([row(4, symbol="き")])

        self.assertEqual([b.text for b in found], ["き"])
        self.assertIn("row 4 came back in position 1", said)

    def test_a_clean_page_says_nothing(self):
        rows = range(1, config.ROWS_PER_PAGE + 1)
        full = [row(number, symbol="き") for number in rows]

        self.assertEqual(notes(full), [])


class ParseRowsTest(unittest.TestCase):
    """A page that did not arrive is a retry, unlike a row that cannot sit."""

    def test_it_reads_the_rows_out_of_the_response(self):
        self.assertEqual(parse_rows(fake_response([row(1)])), [row(1)])

    def test_a_wrapped_array_is_accepted_anyway(self):
        self.assertEqual(
            parse_rows(fake_response({"rows": [row(1)]})), [row(1)]
        )

    def test_an_empty_response_names_the_finish_reason(self):
        with self.assertRaises(OCRError) as caught:
            parse_rows(fake_response([], text="", finish_reason="MAX_TOKENS"))

        self.assertIn("MAX_TOKENS", str(caught.exception))

    def test_a_response_that_is_not_json_is_an_ocr_error(self):
        with self.assertRaises(OCRError):
            parse_rows(fake_response([], text="not json at all"))

    def test_a_response_that_is_not_rows_is_an_ocr_error(self):
        with self.assertRaises(OCRError):
            parse_rows(fake_response({"unexpected": True}))


class TokensTest(unittest.TestCase):
    """Reading the meter off a response, defensively. (6.1, 7.1)"""

    def test_it_reads_what_the_call_was_billed_for(self):
        spent = tokens(fake_response([], usage=fake_usage(prompt=1234, answer=567)))

        self.assertEqual(spent.prompt_tokens, 1234)
        self.assertEqual(spent.output_tokens, 567)

    def test_thinking_counts_as_output_and_is_also_kept_apart(self):
        spent = tokens(
            fake_response([], usage=fake_usage(prompt=10, answer=600, thoughts=300))
        )

        self.assertEqual(spent.output_tokens, 900)
        self.assertEqual(spent.thought_tokens, 300)

    def test_a_response_with_no_meter_reports_nothing(self):
        self.assertEqual(tokens(fake_response([])), Usage())

    def test_a_meter_with_missing_counts_reports_zero_not_none(self):
        spent = tokens(fake_response([], usage=SimpleNamespace()))

        self.assertEqual(spent, Usage())


class GeminiEngineTest(unittest.TestCase):
    """The only tests that touch the SDK, and they replace it wholesale."""

    def setUp(self):
        quieten(self)

    def build(self, sdk, **kwargs):
        """Build the engine with the SDK replaced.

        The credential setting is emptied for the duration, so a machine whose
        own environment names a service account still runs these cases down the
        Application Default Credentials path instead of reaching for a real key.
        """
        settings = {"project": "p", "location": "us-central1", "model": "m"}
        settings.update(kwargs)
        with patch.dict(os.environ, {config.GOOGLE_ENV_CREDENTIALS: ""}):
            with patch("creature_ocr_server.engines.gemini._load_sdk", return_value=sdk):
                return GeminiEngine(**settings)

    def test_it_reports_its_engine_name(self):
        self.assertEqual(self.build(fake_sdk()).name, "gemini")

    def test_it_reports_the_model_it_was_built_with(self):
        # The response cache keys on this, so swapping models while comparing
        # engines cannot serve back the previous model's reading. (4.2)
        self.assertIn("m", self.build(fake_sdk(), model="m").settings)

    def test_a_moved_grid_is_another_reading(self):
        # A box from this engine is a cell address, not a measurement, so a
        # cached one is only meaningful under the grid it was written for.
        # Re-measuring the crop moved four of the eleven columns into a
        # neighbour, which would have put 見つけた日 into 見つけた月. (4.2)
        before = self.build(fake_sdk()).settings
        moved = dict(config.CELL_COLUMNS)
        moved["symbol"] = (0.30, 0.34)

        with patch.object(config, "CELL_COLUMNS", moved):
            after = self.build(fake_sdk()).settings

        self.assertNotEqual(before, after)

    def test_a_moved_row_edge_is_another_reading(self):
        edges = tuple(e + 0.01 for e in config.CELL_ROW_EDGES)
        before = self.build(fake_sdk()).settings

        with patch.object(config, "CELL_ROW_EDGES", edges):
            after = self.build(fake_sdk()).settings

        self.assertNotEqual(before, after)

    def test_the_same_grid_is_the_same_reading(self):
        self.assertEqual(grid_fingerprint(), grid_fingerprint())

    def test_another_prompt_is_another_reading(self):
        # What this engine returns is the answer to a question. Change the
        # question - another printed list, another rule - and the cached answer
        # was given to a question nobody is asking any more. (3.2, 4.2)
        before = self.build(fake_sdk()).settings

        with patch.object(config, "BUG_NAME_HINTS", ("シオカラトンボ",)):
            after = self.build(fake_sdk()).settings

        self.assertNotEqual(before, after)

    def test_the_same_prompt_is_the_same_reading(self):
        self.assertEqual(prompt_fingerprint(), prompt_fingerprint())

    def test_another_model_is_another_reading(self):
        first = self.build(fake_sdk(), model="one").settings
        second = self.build(fake_sdk(), model="two").settings

        self.assertNotEqual(first, second)

    def test_it_talks_to_vertex_not_to_the_developer_api(self):
        sdk = fake_sdk()

        self.build(sdk, location="asia-northeast1")

        sdk[0].Client.assert_called_once_with(
            vertexai=True,
            project="p",
            location="asia-northeast1",
            credentials=None,
        )

    def test_an_empty_credential_setting_leaves_the_client_on_adc(self):
        # None is the SDK's own default: an empty setting has to reach Vertex
        # exactly the way this engine did before there was a setting. (6.3)
        sdk = fake_sdk()

        self.build(sdk)

        self.assertIsNone(sdk[0].Client.call_args.kwargs["credentials"])

    def test_the_service_account_env_names_reaches_the_client(self):
        sdk = fake_sdk()
        key = MagicMock()

        with patch("creature_ocr_server.engines.auth.credentials", return_value=key):
            self.build(sdk)

        self.assertIs(sdk[0].Client.call_args.kwargs["credentials"], key)

    def test_who_authenticated_is_not_part_of_the_cache_key(self):
        # settings is written into the cache file under every page, so a key
        # in it would put a private key in the output directory. Nor does who
        # signed in change what the model read, so changing it must not throw
        # the cached readings away and bill the batch again. (3.2, 6.2)
        plain = self.build(fake_sdk()).settings

        with patch(
            "creature_ocr_server.engines.auth.credentials", return_value=MagicMock()
        ):
            with_key = self.build(fake_sdk()).settings

        self.assertEqual(plain, with_key)

    def test_it_pins_the_generation_parameters_5_2_4_asks_for(self):
        sdk = fake_sdk()
        types = sdk[1]

        self.build(sdk)

        _, kwargs = types.GenerateContentConfig.call_args
        self.assertEqual(kwargs["temperature"], config.GEMINI_TEMPERATURE)
        self.assertEqual(kwargs["top_p"], config.GEMINI_TOP_P)

    def test_it_asks_for_json_in_the_shape_of_the_schema(self):
        sdk = fake_sdk()
        types = sdk[1]

        self.build(sdk)

        _, kwargs = types.GenerateContentConfig.call_args
        self.assertEqual(kwargs["response_mime_type"], "application/json")
        self.assertEqual(kwargs["response_schema"], response_schema())
        self.assertEqual(kwargs["system_instruction"], build_prompt())

    def test_it_sets_no_output_token_limit(self):
        # A model that thinks spends that budget on thinking and then returns
        # nothing, which is indistinguishable from a refusal.
        sdk = fake_sdk()

        self.build(sdk)

        _, kwargs = sdk[1].GenerateContentConfig.call_args
        self.assertNotIn("max_output_tokens", kwargs)

    def test_it_sends_the_page_as_inline_png_bytes(self):
        sdk = fake_sdk()
        types = sdk[1]

        self.build(sdk).recognize(PAGE)

        types.Part.from_bytes.assert_called_once_with(
            data=PAGE, mime_type="image/png"
        )

    def test_one_page_costs_exactly_one_request(self):
        sdk = fake_sdk()
        client = sdk[0].Client.return_value

        self.build(sdk).recognize(PAGE)

        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_the_request_carries_the_model_and_the_config(self):
        sdk = fake_sdk()
        client = sdk[0].Client.return_value

        engine = self.build(sdk, model="gemini-test")
        engine.recognize(PAGE)

        _, kwargs = client.models.generate_content.call_args
        self.assertEqual(kwargs["model"], "gemini-test")
        self.assertIs(
            kwargs["config"], sdk[1].GenerateContentConfig.return_value
        )

    def test_it_returns_the_positioned_text_of_the_page(self):
        sdk = fake_sdk(fake_response([row(2, symbol="き")]))

        found = self.build(sdk).recognize(PAGE)

        self.assertEqual([b.text for b in found.boxes], ["き"])
        self.assertEqual(assign_to_cells(found.boxes)[(2, "symbol")], "き")

    def test_it_reports_what_the_call_cost(self):
        # Returned rather than added to the engine: one engine answers every
        # request that names its model, and an attribute would mix one
        # caller's tokens into another caller's answer. (6.1, 7.1)
        sdk = fake_sdk(
            fake_response([row(1, symbol="き")], usage=fake_usage(1234, 567, 300))
        )

        found = self.build(sdk).recognize(PAGE)

        self.assertEqual(found.usage.prompt_tokens, 1234)
        self.assertEqual(found.usage.output_tokens, 867)
        self.assertEqual(found.usage.thought_tokens, 300)

    def test_it_keeps_nothing_between_calls(self):
        sdk = fake_sdk(
            fake_response([row(1, symbol="き")], usage=fake_usage(1234, 567, 300))
        )
        engine = self.build(sdk)

        first = engine.recognize(PAGE)
        second = engine.recognize(PAGE)

        self.assertEqual(first.usage, second.usage)

    def test_a_page_that_came_back_unusable_was_still_billed(self):
        # It was charged for whether or not it could be parsed, so hiding it
        # would understate the run. The tokens ride on the failure, because
        # there is no engine attribute left to leave them on. (6.1, 7.1)
        sdk = fake_sdk(
            fake_response([], text="not json", usage=fake_usage(1234, 5))
        )
        engine = self.build(sdk)

        with self.assertRaises(OCRError) as raised:
            engine.recognize(PAGE)

        self.assertEqual(raised.exception.usage.prompt_tokens, 1234)

    def test_what_the_engine_worked_around_comes_back_with_the_page(self):
        sdk = fake_sdk(fake_response([row(4, symbol="き")]))

        found = self.build(sdk).recognize(PAGE)

        self.assertIn("row 4 came back in position 1", found.notes)

    def test_a_clean_page_carries_no_notes(self):
        full = [
            row(number, symbol="き") for number in range(1, config.ROWS_PER_PAGE + 1)
        ]
        sdk = fake_sdk(fake_response(full))

        found = self.build(sdk).recognize(PAGE)

        self.assertEqual(found.notes, [])

    def test_widening_a_character_set_is_another_reading(self):
        # A client caches finished rows now, not boxes, so a change to what a
        # check accepts has to be visible to it or it serves rows read under
        # the old rules forever. (3.2, 7.2)
        before = self.build(fake_sdk()).settings
        widened = dict(config.OCR_FIELD_CHARACTERS)
        widened["found_month"] = "0123456789x"

        with patch.object(config, "OCR_FIELD_CHARACTERS", widened):
            after = self.build(fake_sdk()).settings

        self.assertNotEqual(before, after)

    def test_the_published_settings_are_what_the_engine_reads_under(self):
        # GET /v1/engines answers with the classmethod, before anything is
        # built. If the two ever differ, every client keys its cache on a
        # reader that did not read the page. (3.2, 4.2)
        engine = self.build(fake_sdk(), model="gemini-test")

        self.assertEqual(engine.settings, GeminiEngine.settings_for("gemini-test"))

    def test_the_settings_can_be_answered_without_building_anything(self):
        with patch(
            "creature_ocr_server.engines.gemini._load_sdk",
            side_effect=AssertionError("the SDK must not be loaded"),
        ):
            settings = GeminiEngine.settings_for("gemini-test")

        self.assertIn("gemini-test", settings)

    def test_a_vendor_error_becomes_an_ocr_error(self):
        sdk = fake_sdk()
        client = sdk[0].Client.return_value
        client.models.generate_content.side_effect = FakeApiError("refused")

        with self.assertRaises(OCRError):
            self.build(sdk).recognize(PAGE)

    def test_a_missing_project_is_rejected_before_any_call(self):
        self.assert_missing_setting("project", config.GEMINI_ENV_PROJECT)

    def test_a_missing_location_is_rejected_before_any_call(self):
        self.assert_missing_setting("location", config.GEMINI_ENV_LOCATION)

    def test_a_missing_model_is_rejected_before_any_call(self):
        self.assert_missing_setting("model", config.GEMINI_ENV_MODEL)

    def assert_missing_setting(self, argument, key):
        sdk = fake_sdk()

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                self.build(sdk, **{argument: None})

        self.assertIn(key, str(caught.exception))
        sdk[0].Client.assert_not_called()

    def test_the_settings_come_from_the_environment(self):
        sdk = fake_sdk()
        environment = {
            config.GEMINI_ENV_PROJECT: "from-env",
            config.GEMINI_ENV_LOCATION: "global",
            config.GEMINI_ENV_MODEL: "gemini-from-env",
        }

        with patch.dict(os.environ, environment, clear=True):
            with patch(
                "creature_ocr_server.engines.gemini._load_sdk", return_value=sdk
            ):
                engine = GeminiEngine()

        sdk[0].Client.assert_called_once_with(
            vertexai=True,
            project="from-env",
            location="global",
            credentials=None,
        )
        self.assertIn("gemini-from-env", engine.settings)

    def test_a_missing_sdk_says_which_extra_installs_it(self):
        with patch(
            "creature_ocr_server.engines.gemini._load_sdk", side_effect=ImportError
        ):
            with self.assertRaises(ValueError) as caught:
                GeminiEngine(project="p", location="us-central1", model="m")

        self.assertIn("[gemini]", str(caught.exception))


class EngineRegistryTest(unittest.TestCase):
    """Choosing an engine, and choosing it the same way from every endpoint."""

    def setUp(self):
        self.addCleanup(engines.forget)

    def test_the_argument_wins(self):
        with patch.dict(os.environ, {config.OCR_ENGINE_ENV: "gemini"}):
            self.assertEqual(engines.resolve("gemini"), "gemini")

    def test_the_environment_is_used_when_nothing_is_asked_for(self):
        with patch.dict(engines.ENGINES, {"other": GeminiEngine}, clear=False):
            with patch.dict(os.environ, {config.OCR_ENGINE_ENV: "other"}):
                self.assertEqual(engines.resolve(), "other")

    def test_a_blank_environment_value_counts_as_unset(self):
        # load_env copies a bare OCR_ENGINE= line out as an empty string, which
        # is a setting nobody meant to make.
        with patch.dict(os.environ, {config.OCR_ENGINE_ENV: ""}):
            self.assertEqual(engines.resolve(), config.DEFAULT_OCR_ENGINE)

    def test_an_unknown_name_lists_what_there_is(self):
        with self.assertRaises(ValueError) as caught:
            engines.resolve("nope")

        self.assertIn("nope", str(caught.exception))
        for name in engines.ENGINES:
            self.assertIn(name, str(caught.exception))


class EngineStatusTest(unittest.TestCase):
    """Readiness has to be answerable without building anything.

    An endpoint that reported it by constructing every engine would make an
    informational request parse a key and, on a first call, exchange an OAuth
    token - per backend, for backends nobody is using. (4.2, 6.3)
    """

    def test_it_builds_nothing(self):
        with patch(
            "creature_ocr_server.engines.gemini._load_sdk",
            side_effect=AssertionError("the SDK must not be loaded"),
        ):
            with patch(
                "creature_ocr_server.engines.auth.credentials",
                side_effect=AssertionError("no credential may be read"),
            ):
                engines.status("gemini")

    def test_a_missing_setting_is_named(self):
        with patch.dict(os.environ, {}, clear=True):
            ready, detail = engines.status("gemini")

        self.assertFalse(ready)
        self.assertIn(config.GEMINI_ENV_PROJECT, detail)

    def test_a_fully_configured_engine_is_ready(self):
        environment = {
            config.GEMINI_ENV_PROJECT: "p",
            config.GEMINI_ENV_LOCATION: "global",
            config.GEMINI_ENV_MODEL: "m",
        }
        with patch.dict(os.environ, environment):
            with patch("importlib.util.find_spec", return_value=object()):
                ready, detail = engines.status("gemini")

        self.assertTrue(ready)
        self.assertEqual(detail, "")

    def test_a_missing_extra_says_which_one_installs_it(self):
        environment = {
            config.GEMINI_ENV_PROJECT: "p",
            config.GEMINI_ENV_LOCATION: "global",
            config.GEMINI_ENV_MODEL: "m",
        }
        with patch.dict(os.environ, environment):
            with patch("importlib.util.find_spec", return_value=None):
                ready, detail = engines.status("gemini")

        self.assertFalse(ready)
        self.assertIn("[gemini]", detail)


class ModelChoiceTest(unittest.TestCase):
    """The allow-list, which is a memory bound as much as a menu."""

    def setUp(self):
        self.addCleanup(engines.forget)

    def test_no_model_asked_for_is_no_model(self):
        self.assertEqual(engines.choose_model("gemini", None), "")

    def test_a_listed_model_is_accepted(self):
        listed = GeminiEngine.models()[0]

        self.assertEqual(engines.choose_model("gemini", listed), listed)

    def test_an_unlisted_model_is_refused_with_the_list(self):
        with self.assertRaises(ValueError) as caught:
            engines.choose_model("gemini", "gemini-made-up")

        self.assertIn("gemini-made-up", str(caught.exception))
        self.assertIn(GeminiEngine.models()[0], str(caught.exception))

    def test_the_list_can_be_set_from_the_environment(self):
        with patch.dict(os.environ, {config.GEMINI_ENV_MODELS: "a, b ,c"}):
            self.assertEqual(GeminiEngine.models(), ("a", "b", "c"))

    def test_an_empty_environment_list_leaves_the_configured_one(self):
        with patch.dict(os.environ, {config.GEMINI_ENV_MODELS: "  "}):
            self.assertEqual(GeminiEngine.models(), config.GEMINI_MODELS)

    def test_an_engine_with_no_models_refuses_one(self):
        class Processor(GeminiEngine):
            name = "processor"

            @classmethod
            def models(cls):
                return ()

        with patch.dict(engines.ENGINES, {"processor": Processor}):
            with self.assertRaises(ValueError) as caught:
                engines.choose_model("processor", "anything")

        self.assertIn("takes no model", str(caught.exception))


class SharedEngineTest(unittest.TestCase):
    """One engine per model, built once and answering every request for it."""

    def setUp(self):
        self.addCleanup(engines.forget)
        engines.forget()

    def build_counting(self):
        """A registry entry that records how often it was constructed."""
        built = []

        class Counted(GeminiEngine):
            name = "counted"

            def __init__(self, model=None):
                built.append(model)
                self.settings = f"counted {model}"
                self.cache_name = model or "counted"

            def recognize(self, image, mime_type="image/png"):
                raise AssertionError("not called")

        return Counted, built

    def test_the_same_model_is_built_once(self):
        engine, built = self.build_counting()
        listed = GeminiEngine.models()[0]

        with patch.dict(engines.ENGINES, {"counted": engine}):
            first = engines.shared("counted", listed)
            second = engines.shared("counted", listed)

        self.assertIs(first, second)
        self.assertEqual(built, [listed])

    def test_another_model_is_another_engine(self):
        engine, built = self.build_counting()
        one, two = GeminiEngine.models()[:2]

        with patch.dict(engines.ENGINES, {"counted": engine}):
            first = engines.shared("counted", one)
            second = engines.shared("counted", two)

        self.assertIsNot(first, second)
        self.assertEqual(built, [one, two])

    def test_an_unlisted_model_is_never_built(self):
        engine, built = self.build_counting()

        with patch.dict(engines.ENGINES, {"counted": engine}):
            with self.assertRaises(ValueError):
                engines.shared("counted", "gemini-made-up")

        self.assertEqual(built, [])


if __name__ == "__main__":
    unittest.main()


class CompositeSheetTest(unittest.TestCase):
    """The v2 sheet: the same table, plus the strip below it."""

    def test_the_schema_asks_for_the_strip_beside_the_rows_not_inside_them(self):
        schema = response_schema_v2()

        self.assertEqual(schema["type"], "object")
        for field, _ in config.HEADER_FIELDS:
            self.assertEqual(schema["properties"][field], {"type": "string"})
            self.assertNotIn(field, schema["properties"]["rows"]["items"]["properties"])

    def test_the_rows_half_is_the_v1_schema_unchanged(self):
        self.assertEqual(response_schema_v2()["properties"]["rows"], response_schema())

    def test_only_the_rows_are_required(self):
        # A strip nobody filled in has to be able to come back empty rather than
        # invented, the same rule 5.2-1 sets for a blank cell.
        self.assertEqual(response_schema_v2()["required"], ["rows"])

    def test_the_prompt_is_the_v1_prompt_and_then_the_strip(self):
        self.assertTrue(build_prompt_v2().startswith(build_prompt()))
        for _, header in config.HEADER_FIELDS:
            self.assertIn(header, build_prompt_v2())

    def test_the_prompt_says_a_label_is_never_the_answer(self):
        # The one mistake the strip invites: 年 is printed beside the box, not
        # written in it.
        self.assertIn("the label", build_prompt_v2())

    def test_a_page_round_trips_into_the_fields_it_was_asked_for(self):
        boxes, notes = text_boxes_v2(
            {
                "school": "芝",
                "grade": "4",
                "school_class": "2",
                "rows": [{"no": 1, "bug_name": "カナブン"}],
            }
        )

        self.assertEqual(notes, [f"the page came back without row(s) {list(range(2, 9))}"])
        self.assertEqual(
            assign_to_header(boxes, header_at_v2),
            {"school": "芝", "grade": "4", "school_class": "2"},
        )
        placed = assign_to_cells(boxes, cell_at_v2)
        self.assertEqual(placed[(1, "bug_name")], "カナブン")

    def test_an_empty_strip_produces_no_boxes_for_it(self):
        boxes, _ = text_boxes_v2({"school": "", "rows": []})

        self.assertEqual(assign_to_header(boxes, header_at_v2)["school"], "")

    def test_a_bare_array_is_read_as_a_page_with_no_strip(self):
        # A model that ignored the object wrapper still gives up its eight rows
        # rather than costing the whole page.
        found = parse_page_v2(SimpleNamespace(text='[{"no": 1}]'))

        self.assertEqual(found, {"rows": [{"no": 1}]})

    def test_a_page_that_is_not_an_object_raises(self):
        with self.assertRaises(OCRError):
            parse_page_v2(SimpleNamespace(text='"a string"'))

    def test_a_response_with_no_rows_raises(self):
        with self.assertRaises(OCRError):
            text_boxes_v2({"school": "芝"})

    def test_the_two_sheets_are_cached_under_different_names(self):
        self.assertNotEqual(
            GeminiEngine.settings_for("gemini-3.7-flash"),
            GeminiEngine.settings_for("gemini-3.7-flash", sheet=config.SHEET_V2),
        )
        self.assertNotEqual(
            GeminiEngine.cache_name_for("gemini-3.7-flash"),
            GeminiEngine.cache_name_for("gemini-3.7-flash", sheet=config.SHEET_V2),
        )

    def test_v1_settings_are_spelled_exactly_as_they_always_were(self):
        # Every page a client has already paid to read is filed under this
        # string. It must not gain so much as a word.
        self.assertEqual(
            GeminiEngine.settings_for("gemini-3.7-flash"),
            f"gemini-3.7-flash temperature=0.0 top_p=0.1 "
            f"grid={grid_fingerprint()} prompt={prompt_fingerprint()} "
            f"checks={checks_fingerprint()}",
        )

    def test_an_unknown_sheet_is_refused(self):
        with self.assertRaises(ValueError):
            GeminiEngine(project="p", location="l", model="m", sheet="v3")
