"""LM Studio engine tests.

No network: the engine is exercised through the one function that opens a
socket, the way nemotron is, and nothing here needs a base URL that answers or a
key. The reading half is Gemini's own - the prompt, the schema, the row parsing
and the cell addressing - so it is not tested twice here. What is tested is what
this engine adds: an OpenAI-shaped request that names no model, an OpenAI-shaped
answer, and a cache key that can name a URL but cannot name the model behind it.
"""

import json
import logging
import os
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from creature_ocr_server import config, engines
from creature_ocr_server.engines import lmstudio
from creature_ocr_server.engines.gemini import (
    build_prompt,
    build_prompt_v2,
    grid_fingerprint,
    prompt_fingerprint,
    response_schema,
    response_schema_v2,
)
from creature_ocr_server.engines.lmstudio import (
    LMStudioEngine,
    answered_by,
    content,
    parse_page_v2,
    parse_rows,
    response_format,
    tokens,
    unfence,
)
from creature_ocr_server.grid import cell_at_v2, header_at_v2
from creature_ocr_server.ocr import (
    OCRError,
    Usage,
    assign_to_cells,
    assign_to_header,
    checks_fingerprint,
    header_fingerprint,
)

PAGE = b"a whole page of PNG"
BASE_URL = "http://127.0.0.1:4321/v1"
URL = BASE_URL + "/chat/completions"
KEY = "lm-studio-not-a-real-key"
MODEL = "qwen2.5-vl-7b"


def answer(rows=None, text=None, model=MODEL, finish_reason="stop", usage=None):
    """One chat completion, as the OpenAI-compatible API shapes it."""
    if text is None:
        text = json.dumps([] if rows is None else rows)
    body = {
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def row(number, **fields):
    """One row object as the model is asked to return it."""
    return {"no": number, **fields}


def quieten(case):
    """Silence the module's own logging for a case that is not testing it."""
    logger = logging.getLogger("creature_ocr_server.engines.lmstudio")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    case.addCleanup(logger.setLevel, previous)


def http_error(code, body=b"no"):
    """An HTTPError with a readable body, as urllib raises one."""
    return urllib.error.HTTPError(URL, code, "refused", {}, BytesIO(body))


class FakeResponse:
    """What urlopen hands back, as far as _post is concerned."""

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *anything):
        return False


class EndpointTest(unittest.TestCase):
    """A base URL, not a whole URL - which is the opposite of nemotron's."""

    def test_the_chat_path_is_appended_to_a_base_url(self):
        self.assertEqual(lmstudio._url(BASE_URL), URL)

    def test_a_trailing_slash_is_taken_off_first(self):
        self.assertEqual(lmstudio._url(BASE_URL + "/"), URL)
        self.assertEqual(lmstudio._url("http://127.0.0.1:4321/"), URL)

    def test_a_host_with_no_path_is_given_the_openai_surface_too(self):
        # The failure this exists for: LM Studio answers a request that misses
        # its endpoint with a 200 and a complaint, so a base URL written down as
        # a host and a port would read as a model that returned nothing.
        self.assertEqual(lmstudio._url("http://127.0.0.1:4321"), URL)

    def test_a_path_the_operator_did_write_is_left_alone(self):
        # LM Studio's own native API, or a proxy mounted on a prefix.
        self.assertEqual(
            lmstudio._url("http://127.0.0.1:4321/api/v0"),
            "http://127.0.0.1:4321/api/v0/chat/completions",
        )

    def test_a_url_that_already_names_the_path_is_used_as_it_stands(self):
        # An operator who pasted the whole thing out of another client's
        # configuration should not end up posting to /chat/completions twice.
        self.assertEqual(lmstudio._url(URL), URL)

    def test_nothing_configured_means_the_conventional_local_address(self):
        # The one setting in this package with a default. A wrong guess here
        # cannot bill anybody or read a deployment nobody chose.
        self.assertTrue(lmstudio._url("").startswith(config.LMSTUDIO_BASE_URL))
        self.assertTrue(lmstudio._url(None).endswith(lmstudio.CHAT_PATH))


class TimeoutTest(unittest.TestCase):
    """One attempt's budget, because the machine it runs on is unknown."""

    def test_nothing_set_means_the_default(self):
        self.assertEqual(lmstudio._timeout(""), config.LMSTUDIO_TIMEOUT_SECONDS)

    def test_a_number_is_taken_as_seconds(self):
        self.assertEqual(lmstudio._timeout("600"), 600.0)

    def test_something_that_is_not_a_number_names_the_setting(self):
        with self.assertRaises(ValueError) as caught:
            lmstudio._timeout("ages")

        self.assertIn(config.LMSTUDIO_ENV_TIMEOUT, str(caught.exception))

    def test_zero_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            lmstudio._timeout("0")

        self.assertIn(config.LMSTUDIO_ENV_TIMEOUT, str(caught.exception))


class ResponseFormatTest(unittest.TestCase):
    """Gemini's schema, in the envelope this API asks for it in."""

    def test_it_asks_for_the_schema_by_name(self):
        found = response_format()

        self.assertEqual(found["type"], "json_schema")
        self.assertEqual(found["json_schema"]["name"], lmstudio.SCHEMA_NAME)

    def test_it_is_gemini_s_schema_without_vertex_s_own_keyword(self):
        # propertyOrdering is a Vertex extension. LM Studio turns a schema into
        # a grammar, and an unknown keyword is at best ignored.
        schema = response_format()["json_schema"]["schema"]
        expected = dict(response_schema())
        expected["items"] = {
            key: value
            for key, value in response_schema()["items"].items()
            if key != "propertyOrdering"
        }

        self.assertEqual(schema, expected)

    def test_the_keyword_is_gone_at_every_depth(self):
        found = json.dumps(response_format(config.SHEET_V2))

        self.assertNotIn("propertyOrdering", found)

    def test_the_v2_format_asks_for_the_page_object(self):
        schema = response_format(config.SHEET_V2)["json_schema"]["schema"]

        self.assertEqual(schema["type"], response_schema_v2()["type"])
        self.assertEqual(schema["required"], ["rows"])

    def test_the_schema_gemini_sends_is_left_alone(self):
        # _plain copies rather than edits: the other engine still needs the key.
        response_format()

        self.assertIn("propertyOrdering", response_schema()["items"])


class ContentTest(unittest.TestCase):
    """An answer that is not an answer says so, and 6.4 retries it."""

    def test_the_one_choice_s_text_comes_back(self):
        self.assertEqual(content(answer(text="[]")), "[]")

    def test_an_answer_that_is_not_an_object_raises(self):
        with self.assertRaises(OCRError) as caught:
            content(["not", "an", "object"])

        self.assertIn("not an object", str(caught.exception))

    def test_an_answer_with_no_choices_raises(self):
        with self.assertRaises(OCRError):
            content({"model": MODEL, "choices": []})

    def test_an_answer_without_a_completion_names_the_setting_to_check(self):
        # What a request that missed the endpoint looks like from in here: LM
        # Studio returns 200 and a complaint, so this is the only place the
        # cause can be named.
        with self.assertRaises(OCRError) as caught:
            content({"model": MODEL, "choices": []})

        self.assertIn(config.LMSTUDIO_ENV_BASE_URL, str(caught.exception))

    def test_a_server_s_own_complaint_is_carried(self):
        with self.assertRaises(OCRError) as caught:
            content({"error": "no model loaded"})

        self.assertIn("no model loaded", str(caught.exception))

    def test_an_empty_answer_says_why_it_was_empty(self):
        # The two ways it happens - the output budget ran out, or the model
        # cannot see an image at all - are settings problems and look like
        # nothing otherwise.
        with self.assertRaises(OCRError) as caught:
            content(answer(text="", finish_reason="length"))

        self.assertIn("length", str(caught.exception))

    def test_an_answer_with_no_reason_still_says_something(self):
        with self.assertRaises(OCRError) as caught:
            content(answer(text="", finish_reason=None))

        self.assertIn("no reason given", str(caught.exception))

    def test_a_malformed_choice_raises(self):
        with self.assertRaises(OCRError):
            content({"choices": ["not an object"]})


class UnfenceTest(unittest.TestCase):
    """A local model wraps its JSON in Markdown often enough to allow for it."""

    def test_plain_json_is_left_alone(self):
        self.assertEqual(unfence('  [{"no": 1}]  '), '[{"no": 1}]')

    def test_a_fence_with_a_language_tag_is_taken_off(self):
        self.assertEqual(unfence('```json\n[{"no": 1}]\n```'), '[{"no": 1}]')

    def test_a_bare_fence_is_taken_off_too(self):
        self.assertEqual(unfence('```\n[]\n```'), "[]")

    def test_an_unclosed_fence_still_gives_up_its_json(self):
        self.assertEqual(unfence('```json\n[]'), "[]")


class ParseTest(unittest.TestCase):
    """Everything here raises: it is a whole page that did not arrive."""

    def test_the_rows_come_back(self):
        self.assertEqual(parse_rows(answer([row(1)])), [{"no": 1}])

    def test_a_fenced_page_is_read_rather_than_thrown_away(self):
        found = parse_rows(answer(text='```json\n[{"no": 2}]\n```'))

        self.assertEqual(found, [{"no": 2}])

    def test_a_wrapped_array_is_accepted(self):
        # The schema asks for a bare array; a page that is otherwise perfectly
        # good is not worth throwing away over the wrapper.
        self.assertEqual(parse_rows(answer(text='{"rows": [{"no": 1}]}')), [{"no": 1}])

    def test_something_that_is_not_json_raises(self):
        with self.assertRaises(OCRError) as caught:
            parse_rows(answer(text="here are your rows!"))

        self.assertIn("not JSON", str(caught.exception))

    def test_json_that_is_not_rows_raises(self):
        with self.assertRaises(OCRError):
            parse_rows(answer(text='"a string"'))

    def test_a_v2_bare_array_is_read_as_a_page_with_no_strip(self):
        self.assertEqual(
            parse_page_v2(answer(text='[{"no": 1}]')), {"rows": [{"no": 1}]}
        )

    def test_a_v2_page_that_is_not_an_object_raises(self):
        with self.assertRaises(OCRError):
            parse_page_v2(answer(text="3"))


class TokensTest(unittest.TestCase):
    """Nobody is billed for a local call and the numbers still matter. (7.1)"""

    def test_what_the_server_reported(self):
        found = tokens(answer(usage={"prompt_tokens": 1200, "completion_tokens": 300}))

        self.assertEqual(found, Usage(prompt_tokens=1200, output_tokens=300))

    def test_reasoning_tokens_are_kept_and_not_counted_twice(self):
        # completion_tokens already includes them, which is not what Vertex
        # does, so unlike Gemini's they are not added on again.
        found = tokens(
            answer(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 300,
                    "completion_tokens_details": {"reasoning_tokens": 120},
                }
            )
        )

        self.assertEqual(found.output_tokens, 300)
        self.assertEqual(found.thought_tokens, 120)

    def test_a_page_that_was_read_is_not_thrown_away_for_a_missing_meter(self):
        self.assertEqual(tokens(answer()), Usage())

    def test_a_number_that_is_not_one_counts_as_none(self):
        found = tokens(answer(usage={"prompt_tokens": "lots", "completion_tokens": 2}))

        self.assertEqual(found, Usage(output_tokens=2))


class AnsweredByTest(unittest.TestCase):
    """The only record of which model read a page."""

    def test_the_server_says_which_model_it_used(self):
        self.assertEqual(answered_by(answer()), MODEL)

    def test_an_answer_that_says_nothing_is_not_an_error(self):
        self.assertEqual(answered_by({"choices": []}), "")

    def test_an_answer_that_is_not_an_object_is_not_read(self):
        # Read before the answer is known to be shaped like one, so a list here
        # must not become an AttributeError that loses the real message.
        self.assertEqual(answered_by([]), "")


class LMStudioEngineTest(unittest.TestCase):
    """The engine, with the one function that opens a socket replaced."""

    def setUp(self):
        quieten(self)
        self.sent = []

    def build(self, environment=None, **kwargs):
        settings = {"base_url": BASE_URL}
        settings.update(kwargs)
        with patch.dict(os.environ, environment or {}, clear=True):
            return LMStudioEngine(**settings)

    def fake_post(self, response=None, error=None):
        def post(url, payload, key, timeout):
            self.sent.append(
                {"url": url, "payload": payload, "key": key, "timeout": timeout}
            )
            if error is not None:
                raise error
            return response if response is not None else answer()

        return post

    def recognize(self, response=None, error=None, environment=None, **kwargs):
        engine = self.build(environment, **kwargs)
        with patch.object(lmstudio, "_post", self.fake_post(response, error)):
            return engine.recognize(PAGE)

    def payload(self, **kwargs):
        self.recognize(**kwargs)
        return self.sent[0]["payload"]

    def test_it_is_named_for_the_engine_not_the_vendor(self):
        self.assertEqual(LMStudioEngine.name, "lmstudio")

    def test_it_posts_to_the_chat_endpoint_under_the_base_url(self):
        self.recognize()

        self.assertEqual(self.sent[0]["url"], URL)

    def test_it_names_no_model(self):
        # The whole point: whichever model LM Studio has loaded reads the page,
        # and this server does not choose. (6.5)
        self.assertEqual(self.payload()["model"], "")

    def test_a_named_model_is_sent_when_an_operator_insists(self):
        # The escape hatch for a build that refuses a request naming none.
        found = self.payload(environment={config.LMSTUDIO_ENV_MODEL: MODEL})

        self.assertEqual(found["model"], MODEL)

    def test_it_asks_gemini_s_question(self):
        # 4.1's comparison is a comparison because the question is the same one.
        messages = self.payload()["messages"]

        self.assertEqual(messages[0], {"role": "system", "content": build_prompt()})

    def test_it_sends_the_page_as_a_base64_data_uri(self):
        part = self.payload()["messages"][1]["content"][0]

        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_it_pins_the_generation_parameters_5_2_4_asks_for(self):
        found = self.payload()

        self.assertEqual(found["temperature"], config.LMSTUDIO_TEMPERATURE)
        self.assertEqual(found["top_p"], config.LMSTUDIO_TOP_P)

    def test_it_asks_for_the_page_as_structured_json(self):
        self.assertEqual(self.payload()["response_format"], response_format())

    def test_it_does_not_stream(self):
        # The answer is read in one piece; a stream would arrive as a body that
        # is not JSON at all.
        self.assertFalse(self.payload()["stream"])

    def test_it_sets_no_output_token_limit(self):
        # A model that thinks spends that budget on thinking and then returns
        # nothing at all, which looks exactly like a refusal.
        self.assertNotIn("max_tokens", self.payload())

    def test_one_page_is_one_call(self):
        self.recognize()

        self.assertEqual(len(self.sent), 1)

    def test_no_key_is_sent_when_none_is_configured(self):
        # A local server checks nothing by default.
        self.recognize()

        self.assertEqual(self.sent[0]["key"], "")

    def test_a_key_is_taken_from_the_environment_when_there_is_one(self):
        self.recognize(environment={config.LMSTUDIO_ENV_API_KEY: KEY})

        self.assertEqual(self.sent[0]["key"], KEY)

    def test_the_timeout_comes_from_the_environment(self):
        self.recognize(environment={config.LMSTUDIO_ENV_TIMEOUT: "600"})

        self.assertEqual(self.sent[0]["timeout"], 600.0)

    def test_a_timeout_that_is_not_a_number_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.build({config.LMSTUDIO_ENV_TIMEOUT: "ages"})

        self.assertIn(config.LMSTUDIO_ENV_TIMEOUT, str(caught.exception))

    def test_the_base_url_comes_from_the_environment(self):
        with patch.dict(
            os.environ, {config.LMSTUDIO_ENV_BASE_URL: "http://box:9999"}, clear=True
        ):
            engine = LMStudioEngine()

        self.assertIn("http://box:9999", engine.settings)

    def test_a_row_lands_in_the_cell_it_was_written_in(self):
        found = self.recognize(response=answer([row(3, bug_name="カブトムシ")]))

        self.assertEqual(assign_to_cells(found.boxes)[(3, "bug_name")], "カブトムシ")

    def test_a_row_it_could_not_place_is_said_rather_than_raised(self):
        # One bad row must not cost the other seven. (5.1, 6.4, 7.2)
        found = self.recognize(response=answer([row(99, bug_name="カナブン")]))

        self.assertTrue(any("99" in note for note in found.notes))

    def test_the_log_says_which_model_answered(self):
        # The published cache key cannot name it, so this line is the only
        # record of which model read a page.
        with self.assertLogs(lmstudio.logger, "INFO") as caught:
            found = self.recognize()

        self.assertIn(MODEL, caught.output[0])
        # And it stays out of the reading. Those notes are the caller's error
        # file, where a line on every page would be noise rather than a record.
        self.assertFalse(any(MODEL in note for note in found.notes))

    def test_the_reading_carries_what_the_call_reported(self):
        found = self.recognize(
            response=answer(usage={"prompt_tokens": 9, "completion_tokens": 5})
        )

        self.assertEqual(found.usage, Usage(prompt_tokens=9, output_tokens=5))

    def test_a_page_that_came_back_unusable_was_still_paid_for(self):
        # Hiding what an unusable answer cost would understate the run. (6.1)
        with self.assertRaises(OCRError) as caught:
            self.recognize(
                response=answer(
                    text="no rows for you", usage={"prompt_tokens": 9}
                )
            )

        self.assertEqual(caught.exception.usage, Usage(prompt_tokens=9))

    def test_a_refusal_becomes_an_ocr_error(self):
        with self.assertRaises(OCRError):
            self.recognize(error=OCRError("lmstudio refused a page: HTTP 500"))

    def test_nothing_is_sent_and_nothing_is_built_for_an_unknown_sheet(self):
        with self.assertRaises(ValueError):
            self.build(sheet="v3")


class CompositeSheetTest(unittest.TestCase):
    """The v2 sheet: the same table, plus the strip below it."""

    def setUp(self):
        quieten(self)
        self.sent = []

    def build(self):
        with patch.dict(os.environ, {}, clear=True):
            return LMStudioEngine(base_url=BASE_URL, sheet=config.SHEET_V2)

    def recognize(self, response):
        engine = self.build()

        def post(url, payload, key, timeout):
            self.sent.append(payload)
            return response

        with patch.object(lmstudio, "_post", post):
            return engine.recognize(PAGE)

    def test_it_asks_the_v2_question_with_the_v2_schema(self):
        self.recognize(answer())

        self.assertEqual(self.sent[0]["messages"][0]["content"], build_prompt_v2())
        self.assertEqual(
            self.sent[0]["response_format"], response_format(config.SHEET_V2)
        )

    def test_a_page_round_trips_into_the_fields_it_was_asked_for(self):
        found = self.recognize(
            answer(
                text=json.dumps(
                    {
                        "school": "芝",
                        "grade": "4",
                        "school_class": "2",
                        "rows": [row(1, bug_name="カナブン")],
                    }
                )
            )
        )

        self.assertEqual(
            assign_to_header(found.boxes, header_at_v2),
            {"school": "芝", "grade": "4", "school_class": "2"},
        )
        self.assertEqual(
            assign_to_cells(found.boxes, cell_at_v2)[(1, "bug_name")], "カナブン"
        )

    def test_the_two_sheets_are_cached_under_different_names(self):
        self.assertNotEqual(
            LMStudioEngine.settings_for("", config.SHEET_V1),
            LMStudioEngine.settings_for("", config.SHEET_V2),
        )
        self.assertEqual(LMStudioEngine.cache_name_for(""), "lmstudio")
        self.assertEqual(
            LMStudioEngine.cache_name_for("", config.SHEET_V2), "lmstudio-v2"
        )

    def test_the_v2_key_says_which_paper_and_which_header(self):
        found = LMStudioEngine.settings_for("", config.SHEET_V2, BASE_URL)

        self.assertIn("sheet=v2", found)
        self.assertIn(header_fingerprint(), found)


class PublishedSettingsTest(unittest.TestCase):
    """What GET /v1/engines can say about this engine without building one."""

    def test_the_settings_are_answerable_with_nothing_configured(self):
        # /v1/engines is what a client calls to find out it is not configured,
        # so it cannot be the thing that needs configuration. (6.5)
        with patch.dict(os.environ, {}, clear=True):
            found = LMStudioEngine.settings_for("")

        self.assertIn(config.LMSTUDIO_BASE_URL, found)

    def test_it_names_the_url_the_page_will_be_posted_to(self):
        with patch.dict(
            os.environ, {config.LMSTUDIO_ENV_BASE_URL: BASE_URL}, clear=True
        ):
            self.assertIn(URL, LMStudioEngine.settings_for(""))

    def test_the_api_key_is_not_part_of_the_cache_key(self):
        # settings is published by GET /v1/engines and written into a client's
        # cache file beside every page. (6.2, 6.3)
        with patch.dict(
            os.environ,
            {config.LMSTUDIO_ENV_API_KEY: KEY, config.LMSTUDIO_ENV_BASE_URL: BASE_URL},
            clear=True,
        ):
            engine = LMStudioEngine()

        self.assertNotIn(KEY, engine.settings)
        self.assertNotIn(KEY, LMStudioEngine.settings_for(""))

    def test_the_grid_and_the_prompt_are_part_of_the_cache_key(self):
        # A box from here is a cell address rather than a measurement, and what
        # comes back is the answer to a question. Both are Gemini's reasons.
        found = LMStudioEngine.settings_for("", config.SHEET_V1, BASE_URL)

        self.assertIn(f"grid={grid_fingerprint()}", found)
        self.assertIn(f"prompt={prompt_fingerprint()}", found)

    def test_the_value_checks_are_part_of_the_cache_key(self):
        self.assertIn(
            checks_fingerprint(),
            LMStudioEngine.settings_for("", config.SHEET_V1, BASE_URL),
        )

    def test_the_published_settings_are_what_the_engine_reads_under(self):
        with patch.dict(os.environ, {}, clear=True):
            engine = LMStudioEngine(base_url=BASE_URL)

        self.assertEqual(
            engine.settings, LMStudioEngine.settings_for("", config.SHEET_V1, BASE_URL)
        )

    def test_v1_settings_are_spelled_this_way(self):
        # Every page a client has already read is filed under this string.
        self.assertEqual(
            LMStudioEngine.settings_for("", config.SHEET_V1, BASE_URL),
            f"lmstudio {URL} temperature=0.0 top_p=0.1 "
            f"grid={grid_fingerprint()} prompt={prompt_fingerprint()} "
            f"checks={checks_fingerprint()}",
        )

    def test_the_cache_directory_is_the_engine_s_own_name(self):
        # There is no model id to name it by, and a URL would make a directory
        # name out of a port number.
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(LMStudioEngine.cache_name_for(""), "lmstudio")

    def test_it_offers_no_model(self):
        # Whatever LM Studio has loaded is the reader. (6.5)
        self.assertEqual(LMStudioEngine.models(), ())
        self.assertEqual(LMStudioEngine.default_model(), "")


class RegistryTest(unittest.TestCase):
    """This engine is reachable and reportable the way every engine is."""

    def test_it_is_in_the_registry_under_its_own_name(self):
        self.assertIs(engines.ENGINES["lmstudio"], LMStudioEngine)

    def test_a_request_may_not_name_a_model_for_it(self):
        with self.assertRaises(ValueError) as caught:
            engines.choose_model("lmstudio", MODEL)

        self.assertIn("takes no model", str(caught.exception))

    def test_it_is_usable_with_nothing_set_and_nothing_installed(self):
        # There is no SDK to find and no setting that has to be filled in. What
        # status cannot say is whether LM Studio is running: that needs a
        # socket, and it opens none. (4.2, 6.5)
        self.assertEqual(LMStudioEngine.sdk, "")
        with patch.dict(os.environ, {}, clear=True):
            with patch("importlib.util.find_spec", return_value=None):
                with patch.object(
                    lmstudio, "_post", side_effect=AssertionError("nothing may be sent")
                ):
                    ready, detail = engines.status("lmstudio")

        self.assertTrue(ready)
        self.assertEqual(detail, "")

    def test_it_names_no_extra_because_there_is_nothing_to_install(self):
        # Naming one would send an operator to an extra that does not exist.
        self.assertEqual(LMStudioEngine.extra, "")


class PostTest(unittest.TestCase):
    """Every way the request can fail becomes the error 6.4 retries."""

    def setUp(self):
        quieten(self)

    def post(self, error=None, body=b"{}", key=""):
        self.sent = []

        def urlopen(request, timeout=None):
            self.sent.append(request)
            if error is not None:
                raise error
            return FakeResponse(body)

        with patch.object(lmstudio.urllib.request, "urlopen", urlopen):
            return lmstudio._post(URL, {"model": ""}, key, 1.0)

    def test_an_answer_comes_back_parsed(self):
        self.assertEqual(self.post(body=b'{"model": "m"}'), {"model": "m"})

    def test_no_authorization_header_is_sent_without_a_key(self):
        self.post()

        self.assertIsNone(self.sent[0].get_header("Authorization"))

    def test_a_key_is_sent_as_a_bearer_token(self):
        self.post(key=KEY)

        self.assertEqual(self.sent[0].get_header("Authorization"), f"Bearer {KEY}")

    def test_a_refusal_about_the_model_says_what_to_do_about_it(self):
        # This engine's characteristic failure: a build that insists on being
        # told a model when the whole point is to leave that to LM Studio.
        with self.assertRaises(OCRError) as caught:
            self.post(http_error(404, b'{"error": "model not found"}'))

        self.assertIn(config.LMSTUDIO_ENV_MODEL, str(caught.exception))

    def test_another_refusal_carries_the_status_and_the_server_s_words(self):
        with self.assertRaises(OCRError) as caught:
            self.post(http_error(500, b"context overflow"))

        message = str(caught.exception)

        self.assertIn("500", message)
        self.assertIn("context overflow", message)
        self.assertNotIn(config.LMSTUDIO_ENV_MODEL, message)

    def test_a_refusal_body_is_bounded(self):
        with self.assertRaises(OCRError) as caught:
            self.post(http_error(500, b"x" * 10_000))

        self.assertLess(len(str(caught.exception)), lmstudio.ERROR_BODY_CHARS + 200)

    def test_a_server_that_is_not_running_names_the_setting_to_check(self):
        # The failure this actually produces: LM Studio is closed, or its local
        # server was never started.
        with self.assertRaises(OCRError) as caught:
            self.post(urllib.error.URLError("connection refused"))

        message = str(caught.exception)

        self.assertIn("connection refused", message)
        self.assertIn(URL, message)
        self.assertIn(config.LMSTUDIO_ENV_BASE_URL, message)

    def test_a_timeout_is_an_ocr_error(self):
        with self.assertRaises(OCRError):
            self.post(TimeoutError("timed out"))

    def test_a_body_that_is_not_json_is_an_ocr_error(self):
        with self.assertRaises(OCRError) as caught:
            self.post(body=b"<html>404</html>")

        self.assertIn("not JSON", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
