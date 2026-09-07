"""Document AI engine tests.

No network and no cloud SDK: the engine is exercised through the one loader
function that imports its SDK, so the suite passes with google-cloud-documentai
not installed.

Carried over from the desktop repository. What changed is what this server
changed: recognize hands back a Reading instead of a list, text_boxes returns
what it dropped instead of logging it, and the registry can say whether this
engine is usable and what a client should key its cache on without building one.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from creature_ocr_server import config, engines
from creature_ocr_server.engines.documentai import DocumentAIEngine, text_boxes
from creature_ocr_server.ocr import OCRError, Usage, assign_to_cells, checks_fingerprint

PAGE = b"a whole page of PNG"

ENVIRONMENT = {
    config.DOCUMENTAI_ENV_PROJECT: "p",
    config.DOCUMENTAI_ENV_LOCATION: "us",
    config.DOCUMENTAI_ENV_PROCESSOR_ID: "x",
}


class FakeApiError(Exception):
    """Stands in for google.api_core.exceptions.GoogleAPICallError."""


def _entry(content, left, top, right, bottom, start, end):
    """One token or symbol as Document AI shapes it."""
    return SimpleNamespace(
        layout=SimpleNamespace(
            text_anchor=SimpleNamespace(
                text_segments=[SimpleNamespace(start_index=start, end_index=end)]
            ),
            bounding_poly=SimpleNamespace(
                normalized_vertices=[
                    SimpleNamespace(x=left, y=top),
                    SimpleNamespace(x=right, y=top),
                    SimpleNamespace(x=right, y=bottom),
                    SimpleNamespace(x=left, y=bottom),
                ]
            ),
        )
    )


def _entries(items):
    text, entries, cursor = "", [], 0
    for content, left, top, right, bottom in items:
        start, cursor = cursor, cursor + len(content)
        text += content
        entries.append(_entry(content, left, top, right, bottom, start, cursor))
    return text, entries


def fake_document(tokens, symbols=()):
    """A response document built from (text, l, t, r, b) tuples."""
    text, token_entries = _entries(tokens)
    symbol_text, symbol_entries = _entries(symbols)
    # Symbols index into the same document text, so build it from whichever
    # side carries the characters.
    return SimpleNamespace(
        text=symbol_text if symbols else text,
        pages=[SimpleNamespace(tokens=token_entries, symbols=symbol_entries)],
    )


def fake_sdk(document=None):
    """A Document AI SDK triple whose client answers with one document."""
    documentai = MagicMock()
    client = documentai.DocumentProcessorServiceClient.return_value
    client.processor_path.return_value = "projects/p/locations/us/processors/x"
    client.process_document.return_value.document = document or fake_document([])
    return documentai, MagicMock(), FakeApiError


def boxes(document):
    """Just the boxes, for a case that is not about what was dropped."""
    return text_boxes(document)[0]


def notes(document):
    """Just the complaints."""
    return text_boxes(document)[1]


def middle_of(field):
    """The horizontal centre of a column, for placing a box in one."""
    left, right = config.CELL_COLUMNS[field]
    return (left + right) / 2


class TextBoxesTest(unittest.TestCase):
    """Parsing a Document AI response into positioned text."""

    def test_it_resolves_each_token_against_the_document_text(self):
        document = fake_document(
            [
                ("ショウリョウバッタ", 0.10, 0.05, 0.20, 0.12),
                ("き", 0.24, 0.05, 0.26, 0.12),
            ]
        )

        found = boxes(document)

        self.assertEqual([b.text for b in found], ["ショウリョウバッタ", "き"])

    def test_it_keeps_the_normalised_bounding_box(self):
        document = fake_document([("き", 0.24, 0.05, 0.26, 0.12)])

        found = boxes(document)[0]

        self.assertEqual(
            (found.left, found.top, found.right, found.bottom),
            (0.24, 0.05, 0.26, 0.12),
        )

    def test_a_whitespace_only_token_is_dropped(self):
        document = fake_document([(" \n", 0.24, 0.05, 0.26, 0.12)])

        self.assertEqual(boxes(document), [])

    def test_a_token_with_no_position_is_dropped(self):
        document = fake_document([("き", 0.24, 0.05, 0.26, 0.12)])
        document.pages[0].tokens[0].layout.bounding_poly.normalized_vertices = []

        self.assertEqual(boxes(document), [])

    def test_what_was_dropped_is_said_rather_than_logged(self):
        # The operator who has to know is on another machine, so a drop comes
        # back in the reading instead of going to this container's log. (7.2)
        document = fake_document([("き", 0.24, 0.05, 0.26, 0.12)])
        document.pages[0].tokens[0].layout.bounding_poly.normalized_vertices = []

        self.assertIn("き", notes(document)[0])

    def test_an_empty_page_gives_no_boxes(self):
        self.assertEqual(boxes(fake_document([])), [])
        self.assertEqual(notes(fake_document([])), [])

    def test_symbols_are_preferred_over_tokens(self):
        # A word box straddling the month/day rule against the two characters
        # that make it up. Per character, each lands in its own cell.
        document = fake_document(
            tokens=[("78", 0.515, 0.05, 0.563, 0.08)],
            symbols=[("7", 0.515, 0.05, 0.535, 0.08), ("8", 0.545, 0.05, 0.563, 0.08)],
        )

        self.assertEqual([b.text for b in boxes(document)], ["7", "8"])

    def test_tokens_are_used_when_the_processor_returns_no_symbols(self):
        document = fake_document(tokens=[("78", 0.515, 0.05, 0.563, 0.08)])

        self.assertEqual([b.text for b in boxes(document)], ["78"])

    def test_a_word_across_a_rule_reaches_both_cells_as_symbols(self):
        month, day = middle_of("found_month"), middle_of("found_day")
        document = fake_document(
            tokens=[("78", month - 0.01, 0.09, day + 0.01, 0.12)],
            symbols=[
                ("7", month - 0.01, 0.09, month + 0.01, 0.12),
                ("8", day - 0.01, 0.09, day + 0.01, 0.12),
            ],
        )

        values = assign_to_cells(boxes(document))

        self.assertEqual(values[(1, "found_month")], "7")
        self.assertEqual(values[(1, "found_day")], "8")

    def test_tokens_come_out_in_document_order(self):
        document = fake_document(
            [
                ("ストロー", 0.86, 0.05, 0.90, 0.08),
                ("を出", 0.91, 0.05, 0.94, 0.08),
                ("していた", 0.86, 0.09, 0.90, 0.12),
            ]
        )
        # The processor may list them in any order; the anchors say what the
        # reading order is, and that is what has to come out.
        document.pages[0].tokens.reverse()

        found = boxes(document)

        self.assertEqual([b.text for b in found], ["ストロー", "を出", "していた"])


class DocumentAIEngineTest(unittest.TestCase):
    """The only tests that touch the SDK, and they replace it wholesale."""

    def build(self, sdk, **kwargs):
        """Build the engine with the SDK replaced.

        The credential setting is emptied for the duration, so a machine whose
        own environment names a service account still runs these cases down the
        Application Default Credentials path instead of reaching for a real key.
        """
        settings = {"project": "p", "location": "us", "processor_id": "x"}
        settings.update(kwargs)
        with patch.dict(os.environ, {config.GOOGLE_ENV_CREDENTIALS: ""}):
            with patch(
                "creature_ocr_server.engines.documentai._load_sdk", return_value=sdk
            ):
                return DocumentAIEngine(**settings)

    def test_it_reports_its_engine_name(self):
        self.assertEqual(self.build(fake_sdk()).name, "documentai")

    def test_it_reports_the_processor_it_was_built_with(self):
        # A client's response cache keys on this. Without it, repointing the
        # server at another processor would silently reuse the old one's
        # reading. (3.2, 4.2)
        settings = self.build(fake_sdk(), location="eu").settings

        for value in ("p", "eu", "x"):
            self.assertIn(value, settings)

    def test_another_processor_is_another_reading(self):
        first = self.build(fake_sdk(), processor_id="one").settings
        second = self.build(fake_sdk(), processor_id="two").settings

        self.assertNotEqual(first, second)

    def test_another_region_is_another_reading(self):
        first = self.build(fake_sdk(), location="us").settings
        second = self.build(fake_sdk(), location="eu").settings

        self.assertNotEqual(first, second)

    def test_another_project_is_another_reading(self):
        first = self.build(fake_sdk(), project="one").settings
        second = self.build(fake_sdk(), project="two").settings

        self.assertNotEqual(first, second)

    def test_the_value_checks_are_part_of_the_cache_key(self):
        # A client of this server caches finished rows, not boxes, so widening
        # a character set with nothing to say so would serve rows read under
        # the old rules for as long as that cache lives. (3.2, 4.2)
        self.assertIn(checks_fingerprint(), self.build(fake_sdk()).settings)

    def test_the_grid_is_not_part_of_the_cache_key(self):
        # These boxes are measurements of where the ink was, so re-reading them
        # under a new grid is the point of caching boxes rather than rows. That
        # is the difference from Gemini, whose boxes are cell addresses. (4.2)
        self.assertNotIn("grid=", self.build(fake_sdk()).settings)
        self.assertNotIn("prompt=", self.build(fake_sdk()).settings)

    def test_the_processor_names_the_cache_directory(self):
        self.assertEqual(
            self.build(fake_sdk(), processor_id="abc").cache_name, "documentai-abc"
        )

    def test_it_asks_for_per_character_boxes_and_japanese(self):
        sdk = fake_sdk()
        documentai = sdk[0]

        self.build(sdk)

        _, kwargs = documentai.OcrConfig.call_args
        self.assertTrue(kwargs["enable_symbol"])
        documentai.OcrConfig.Hints.assert_called_once_with(
            language_hints=list(config.DOCUMENTAI_LANGUAGE_HINTS)
        )

    def test_the_request_carries_the_ocr_options(self):
        sdk = fake_sdk()
        documentai = sdk[0]

        self.build(sdk).recognize(PAGE)

        _, kwargs = documentai.ProcessRequest.call_args
        self.assertIs(kwargs["process_options"], documentai.ProcessOptions.return_value)

    def test_it_sends_the_page_as_a_raw_document(self):
        sdk = fake_sdk()
        documentai = sdk[0]

        self.build(sdk).recognize(PAGE)

        documentai.RawDocument.assert_called_once_with(
            content=PAGE, mime_type="image/png"
        )
        documentai.ProcessRequest.assert_called_once()

    def test_one_page_costs_exactly_one_request(self):
        sdk = fake_sdk()
        client = sdk[0].DocumentProcessorServiceClient.return_value

        self.build(sdk).recognize(PAGE)

        self.assertEqual(client.process_document.call_count, 1)

    def test_it_returns_the_positioned_text_of_the_page(self):
        sdk = fake_sdk(fake_document([("き", 0.24, 0.05, 0.26, 0.12)]))

        found = self.build(sdk).recognize(PAGE)

        self.assertEqual([b.text for b in found.boxes], ["き"])
        self.assertEqual(found.boxes[0].left, 0.24)

    def test_this_engine_does_not_claim_tokens_it_was_not_told(self):
        # Document AI bills per page and reports no token count. Zero is a true
        # statement about it; recognize_with_retry fills in the call and the
        # time. (6.1, 7.1)
        found = self.build(fake_sdk()).recognize(PAGE)

        self.assertEqual(found.usage, Usage())

    def test_it_builds_the_regional_endpoint_from_the_location(self):
        sdk = fake_sdk()
        client_options = sdk[1]

        self.build(sdk, location="eu")

        client_options.assert_called_once_with(
            api_endpoint="eu-documentai.googleapis.com"
        )

    def test_it_asks_the_sdk_for_the_processor_path(self):
        sdk = fake_sdk()
        client = sdk[0].DocumentProcessorServiceClient.return_value

        self.build(sdk, location="eu")

        client.processor_path.assert_called_once_with("p", "eu", "x")

    def test_an_empty_credential_setting_leaves_the_client_on_adc(self):
        # None is the SDK's own default: an empty setting has to reach Document
        # AI exactly the way this engine did before there was a setting. (6.3)
        sdk = fake_sdk()

        self.build(sdk)

        built = sdk[0].DocumentProcessorServiceClient.call_args
        self.assertIsNone(built.kwargs["credentials"])
        self.assertIs(built.kwargs["client_options"], sdk[1].return_value)

    def test_the_service_account_the_environment_names_reaches_the_client(self):
        sdk = fake_sdk()
        key = MagicMock()

        with patch("creature_ocr_server.engines.auth.credentials", return_value=key):
            self.build(sdk)

        built = sdk[0].DocumentProcessorServiceClient.call_args
        self.assertIs(built.kwargs["credentials"], key)

    def test_who_authenticated_is_not_part_of_the_cache_key(self):
        # settings is published by GET /v1/engines and written into a client's
        # cache file, so a key in it would put a private key in somebody's
        # output directory. Nor does who signed in change what the processor
        # read, so changing it must not throw the cached readings away and bill
        # the batch again. (3.2, 6.2, 6.3)
        plain = self.build(fake_sdk()).settings

        with patch(
            "creature_ocr_server.engines.auth.credentials", return_value=MagicMock()
        ):
            with_key = self.build(fake_sdk()).settings

        self.assertEqual(plain, with_key)

    def test_a_google_api_error_becomes_an_ocr_error(self):
        sdk = fake_sdk()
        engine = self.build(sdk)
        client = sdk[0].DocumentProcessorServiceClient.return_value
        client.process_document.side_effect = FakeApiError("quota")

        with self.assertRaises(OCRError):
            engine.recognize(PAGE)

    def test_a_missing_project_is_rejected_before_any_call(self):
        sdk = fake_sdk()

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "creature_ocr_server.engines.documentai._load_sdk", return_value=sdk
            ):
                with self.assertRaises(ValueError) as caught:
                    DocumentAIEngine(processor_id="x")

        self.assertIn(config.DOCUMENTAI_ENV_PROJECT, str(caught.exception))
        sdk[0].DocumentProcessorServiceClient.assert_not_called()

    def test_a_missing_processor_id_is_rejected_before_any_call(self):
        sdk = fake_sdk()

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "creature_ocr_server.engines.documentai._load_sdk", return_value=sdk
            ):
                with self.assertRaises(ValueError) as caught:
                    DocumentAIEngine(project="p", location="us")

        self.assertIn(config.DOCUMENTAI_ENV_PROCESSOR_ID, str(caught.exception))
        sdk[0].DocumentProcessorServiceClient.assert_not_called()

    def test_a_missing_region_is_rejected_rather_than_guessed(self):
        sdk = fake_sdk()

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "creature_ocr_server.engines.documentai._load_sdk", return_value=sdk
            ):
                with self.assertRaises(ValueError) as caught:
                    DocumentAIEngine(project="p", processor_id="x")

        self.assertIn(config.DOCUMENTAI_ENV_LOCATION, str(caught.exception))
        sdk[0].DocumentProcessorServiceClient.assert_not_called()


class PublishedSettingsTest(unittest.TestCase):
    """What GET /v1/engines can say about this engine without building one.

    A client arrives with nothing configured and has to be told what to key its
    cache on. Working that out must cost no credential, no SDK and no socket,
    which is the whole reason these are classmethods. (3.2, 4.2)
    """

    def test_the_settings_are_answerable_from_the_environment_alone(self):
        with patch.dict(os.environ, ENVIRONMENT):
            with patch(
                "creature_ocr_server.engines.documentai._load_sdk",
                side_effect=AssertionError("the SDK must not be loaded"),
            ):
                with patch(
                    "creature_ocr_server.engines.auth.credentials",
                    side_effect=AssertionError("no credential may be read"),
                ):
                    settings = DocumentAIEngine.settings_for("")

        self.assertIn("p/us/x", settings)

    def test_the_cache_directory_is_answerable_the_same_way(self):
        with patch.dict(os.environ, ENVIRONMENT):
            self.assertEqual(
                DocumentAIEngine.cache_name_for(""), "documentai-x"
            )

    def test_an_unconfigured_server_still_answers(self):
        # /v1/engines is what a client calls to find out it is not configured,
        # so it cannot be the thing that needs configuration. (6.5)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(DocumentAIEngine.cache_name_for(""), "documentai")
            self.assertIn("hints=ja", DocumentAIEngine.settings_for(""))

    def test_it_offers_no_model(self):
        # The reader is a processor, not a model id, so asking for one is a
        # mistake worth saying so about rather than a setting to ignore. (6.5)
        self.assertEqual(DocumentAIEngine.models(), ())
        self.assertEqual(DocumentAIEngine.default_model(), "")


class RegistryTest(unittest.TestCase):
    """This engine is reachable and reportable the way every engine is."""

    def test_it_is_in_the_registry_under_its_own_name(self):
        self.assertIs(engines.ENGINES["documentai"], DocumentAIEngine)

    def test_a_request_may_not_name_a_model_for_it(self):
        with self.assertRaises(ValueError) as caught:
            engines.choose_model("documentai", "gemini-3.8-flash")

        self.assertIn("takes no model", str(caught.exception))

    def test_readiness_is_reported_without_building_anything(self):
        with patch.dict(os.environ, ENVIRONMENT):
            with patch(
                "creature_ocr_server.engines.documentai._load_sdk",
                side_effect=AssertionError("the SDK must not be loaded"),
            ):
                with patch(
                    "creature_ocr_server.engines.auth.credentials",
                    side_effect=AssertionError("no credential may be read"),
                ):
                    engines.status("documentai")

    def test_a_missing_setting_is_named(self):
        with patch.dict(os.environ, {}, clear=True):
            ready, detail = engines.status("documentai")

        self.assertFalse(ready)
        self.assertIn(config.DOCUMENTAI_ENV_PROJECT, detail)

    def test_a_missing_sdk_says_which_extra_installs_it(self):
        with patch.dict(os.environ, ENVIRONMENT):
            with patch("importlib.util.find_spec", return_value=None):
                ready, detail = engines.status("documentai")

        self.assertFalse(ready)
        self.assertIn("[documentai]", detail)

    def test_the_whole_module_name_is_what_is_looked_for(self):
        # Not its first segment. `google` is a namespace package that
        # google-auth alone puts on the path, so a check that asked about that
        # segment would report this engine ready anywhere [gemini] is
        # installed without [documentai], and GET /v1/engines would offer an
        # engine that cannot be built. Unpatched on purpose: patching find_spec
        # is what let this pass while the check was looking at the wrong name.
        with patch.dict(os.environ, ENVIRONMENT):
            with patch.object(DocumentAIEngine, "sdk", "google.cloud.not_an_sdk"):
                ready, detail = engines.status("documentai")

        self.assertFalse(ready)
        self.assertIn("[documentai]", detail)


if __name__ == "__main__":
    unittest.main()
