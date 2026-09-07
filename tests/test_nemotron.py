"""NVIDIA NIM nemotron-ocr-v2 engine tests.

No network: the engine is exercised through the one function that opens a
socket, the way the other two engines are exercised through the one function
that imports their SDK. Nothing here needs an API key or an endpoint.

Carried over from the desktop repository. What changed is what this server
changed: recognize hands back a Reading instead of a list, text_boxes returns
what it dropped instead of logging it, and the registry can say whether this
engine is usable and what a client should key its cache on without building one.

The re-encoding cases need pymupdf, which is the nemotron extra and is not part
of the base install - the engine reads an ordinary page without it. They skip
where it is absent, which is the same rule the other engines' SDKs follow: the
whole suite has to pass on a machine with no extra installed at all. (4.2)
"""

import hashlib
import logging
import os
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from creature_ocr_server import config, engines
from creature_ocr_server.engines import nemotron
from creature_ocr_server.engines.nemotron import NemotronEngine, shrink, text_boxes
from creature_ocr_server.ocr import OCRError, Usage, assign_to_cells, checks_fingerprint

try:
    import pymupdf
except ImportError:  # pragma: no cover - depends on what is installed
    pymupdf = None

ENDPOINT = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2"
KEY = "nvapi-not-a-real-key"
ENVIRONMENT = {config.NEMOTRON_ENV_API_KEY: KEY}
CONFIGURED = {
    config.NEMOTRON_ENV_API_KEY: KEY,
    config.NEMOTRON_ENV_ENDPOINT: ENDPOINT,
}

needs_imaging = unittest.skipUnless(
    pymupdf is not None, 'pymupdf is not installed: pip install -e ".[nemotron]"'
)


def noisy_png(width=160, height=120, alpha=False):
    """A page PNG that PNG cannot compress, which is what a scan looks like.

    The ruled fixture the rest of the suite uses is flat colour, so PNG beats
    JPEG on it at every quality and nothing can ever be made smaller. A real
    300 DPI scan of paper is noise, and noise is the case shrink exists for.
    The bytes are a hash stream rather than random ones, so the case is the same
    every run.

    The alpha channel is optional and, when asked for, fully opaque: plenty of
    renderers write one whether the paper needed it or not, and that is the
    page this fixture stands for.
    """
    channels = 4 if alpha else 3
    wanted = width * height * channels
    samples = bytearray()
    counter = 0
    while len(samples) < wanted:
        samples += hashlib.sha256(counter.to_bytes(4, "big")).digest()
        counter += 1
    samples = bytearray(samples[:wanted])
    if alpha:
        samples[3::4] = b"\xff" * (width * height)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, width, height, bytes(samples), alpha)
    return pixmap.tobytes("png")


def transparent_png(width=32, height=24):
    """A page whose paper was never filled in, black ink on nothing.

    MuPDF premultiplies on the way in, so by the time shrink holds a pixmap
    every transparent pixel reads black and the colour the client saw behind
    the ink is gone. This is the page that cannot be re-encoded honestly.
    """
    samples = bytearray()
    for y in range(height):
        for x in range(width):
            ink = y == height // 2 and width // 4 <= x < width * 3 // 4
            samples += b"\x00\x00\x00\xff" if ink else b"\xff\xff\xff\x00"
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, width, height, bytes(samples), True)
    return pixmap.tobytes("png")


def quieten(case):
    """Silence the module's own logging for a case that is not testing it."""
    logger = logging.getLogger("creature_ocr_server.engines.nemotron")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    case.addCleanup(logger.setLevel, previous)


def detection(text, left, top, right, bottom, confidence=0.99):
    """One text detection as the API shapes it: a quad, not a rectangle."""
    return {
        "text_prediction": {"text": text, "confidence": confidence},
        "bounding_box": {
            "points": [
                {"x": left, "y": top},
                {"x": right, "y": top},
                {"x": right, "y": bottom},
                {"x": left, "y": bottom},
            ]
        },
    }


def answer(*detections):
    """A whole response for one image."""
    return {
        "model": "nvidia/nemotron-ocr-v2",
        "data": [{"index": 0, "text_detections": list(detections)}],
        "usage": {"images_size_mb": 0.41},
    }


def boxes(response):
    """Just the boxes, for a case that is not about what was dropped."""
    return text_boxes(response)[0]


def notes(response):
    """Just the complaints."""
    return text_boxes(response)[1]


def cell_of(row, field):
    """Where a value for one cell has to sit to be assigned to it."""
    left, right = config.CELL_COLUMNS[field]
    top = config.CELL_ROW_EDGES[row - 1]
    bottom = config.CELL_ROW_EDGES[row]
    return (
        (left + right) / 2 - 0.001,
        (top + bottom) / 2 - 0.001,
        (left + right) / 2 + 0.001,
        (top + bottom) / 2 + 0.001,
    )


def http_error(code, body=b"no"):
    """An HTTPError with a readable body, as urllib raises one."""
    return urllib.error.HTTPError(ENDPOINT, code, "refused", {}, BytesIO(body))


class TextBoxesTest(unittest.TestCase):
    """The response is already in page fractions, so nothing is scaled."""

    def setUp(self):
        quieten(self)

    def test_a_quad_becomes_the_box_that_contains_it(self):
        box = boxes(answer(detection("き", 0.1, 0.2, 0.3, 0.4)))[0]

        self.assertEqual(
            (box.left, box.top, box.right, box.bottom), (0.1, 0.2, 0.3, 0.4)
        )

    def test_a_tilted_quad_is_squared_off_rather_than_dropped(self):
        # A scanned sheet is never quite straight, and the grid is axis aligned.
        tilted = {
            "text_prediction": {"text": "き", "confidence": 0.9},
            "bounding_box": {
                "points": [
                    {"x": 0.10, "y": 0.21},
                    {"x": 0.31, "y": 0.20},
                    {"x": 0.30, "y": 0.39},
                    {"x": 0.11, "y": 0.40},
                ]
            },
        }

        box = boxes(answer(tilted))[0]

        self.assertEqual(
            (box.left, box.top, box.right, box.bottom), (0.10, 0.20, 0.31, 0.40)
        )

    def test_a_value_lands_in_the_cell_it_was_written_in(self):
        found = boxes(answer(detection("カブトムシ", *cell_of(3, "bug_name"))))

        self.assertEqual(assign_to_cells(found)[(3, "bug_name")], "カブトムシ")

    def test_the_engine_order_is_kept(self):
        # 4.2: the caller never re-sorts what an engine returned.
        found = boxes(
            answer(
                detection("second", 0.5, 0.5, 0.6, 0.6),
                detection("first", 0.1, 0.1, 0.2, 0.2),
            )
        )

        self.assertEqual([box.text for box in found], ["second", "first"])

    def test_a_low_confidence_reading_is_kept_and_marked(self):
        # 5.2-2 reports rather than rewrites: the value stays for a person.
        low = config.NEMOTRON_UNSURE_BELOW - 0.01
        box = boxes(answer(detection("き", 0.1, 0.2, 0.3, 0.4, low)))[0]

        self.assertEqual(box.text, "き")
        self.assertTrue(box.unsure)

    def test_a_confident_reading_is_not_marked(self):
        high = config.NEMOTRON_UNSURE_BELOW + 0.01
        box = boxes(answer(detection("き", 0.1, 0.2, 0.3, 0.4, high)))[0]

        self.assertFalse(box.unsure)

    def test_a_reading_with_no_confidence_is_not_marked(self):
        bare = {
            "text_prediction": {"text": "き"},
            "bounding_box": {"points": [{"x": 0.1, "y": 0.2}, {"x": 0.3, "y": 0.4}]},
        }

        self.assertFalse(boxes(answer(bare))[0].unsure)

    def test_whitespace_is_not_a_reading(self):
        self.assertEqual(boxes(answer(detection("   ", 0.1, 0.2, 0.3, 0.4))), [])

    def test_a_detection_with_no_position_is_dropped_not_guessed_at(self):
        # 5.2-2 would rather lose a value than invent where it sat.
        homeless = {"text_prediction": {"text": "き", "confidence": 0.9}}

        self.assertEqual(boxes(answer(homeless)), [])

    def test_what_was_dropped_is_said_rather_than_logged(self):
        # The operator who has to know is on another machine. (7.2)
        homeless = {"text_prediction": {"text": "き", "confidence": 0.9}}

        self.assertIn("き", notes(answer(homeless))[0])

    def test_a_position_that_cannot_be_read_is_dropped(self):
        broken = {
            "text_prediction": {"text": "き", "confidence": 0.9},
            "bounding_box": {"points": [{"x": "left", "y": 0.2}]},
        }

        self.assertEqual(boxes(answer(broken)), [])

    def test_one_bad_detection_does_not_cost_the_others(self):
        found = boxes(
            answer(
                {"text_prediction": {"text": "き"}},
                detection("ち", 0.1, 0.2, 0.3, 0.4),
            )
        )

        self.assertEqual([box.text for box in found], ["ち"])

    def test_a_page_with_no_text_is_not_an_error(self):
        self.assertEqual(boxes(answer()), [])

    def test_an_answer_with_no_data_raises(self):
        # The whole page is missing, which 6.4 retries.
        with self.assertRaises(OCRError):
            text_boxes({"model": "x", "data": []})

    def test_an_answer_without_detections_raises(self):
        with self.assertRaises(OCRError):
            text_boxes({"data": [{"index": 0}]})

    def test_an_answer_that_is_not_an_object_raises(self):
        with self.assertRaises(OCRError):
            text_boxes([])


@needs_imaging
class ShrinkTest(unittest.TestCase):
    """Only reachable when NEMOTRON_MAX_BYTES asks for it. (5.2-3)"""

    def setUp(self):
        quieten(self)
        self.image = noisy_png()

    def test_a_page_that_already_fits_is_not_touched(self):
        room = nemotron._encoded_size(self.image) + 1

        body, mime = shrink(self.image, room)

        self.assertIs(body, self.image)
        self.assertEqual(mime, "image/png")

    def test_a_page_that_does_not_fit_comes_back_smaller_and_as_jpeg(self):
        limit = nemotron._encoded_size(self.image) // 2

        body, mime = shrink(self.image, limit)

        self.assertEqual(mime, "image/jpeg")
        self.assertLessEqual(nemotron._encoded_size(body), limit)

    def test_an_opaque_alpha_channel_is_dropped_rather_than_refused(self):
        # JPEG has no alpha channel and pymupdf refuses the encode rather than
        # dropping one. A bare ValueError out of here is not an OCRError, so
        # 6.4 would retry a local encoding that cannot start working, spend the
        # backoff and the slot on it, and end in a 502 saying nothing.
        page = noisy_png(alpha=True)
        limit = nemotron._encoded_size(page) // 2

        body, mime = shrink(page, limit)

        self.assertEqual(mime, "image/jpeg")
        self.assertLessEqual(nemotron._encoded_size(body), limit)
        # And the page is still the page. Dropping an opaque alpha channel
        # keeps every colour sample; dropping a transparent one would not, and
        # a blackened page still encodes small enough to pass the two above.
        after = pymupdf.Pixmap(body)
        self.assertEqual((after.width, after.height), (160, 120))
        self.assertGreater(len(set(after.samples)), 1)

    def test_a_transparent_page_is_refused_rather_than_quietly_blackened(self):
        # MuPDF premultiplies, so what was behind the ink is gone before this
        # function sees it and a dropped channel means a black page. A black
        # page reads back as no text at all, which is indistinguishable from a
        # sheet nobody wrote on - the one reading 5.2-2 will not invent.
        page = transparent_png()

        with self.assertRaises(OCRError) as caught:
            shrink(page, 4)

        self.assertIn("transparent", str(caught.exception))
        self.assertIn(config.NEMOTRON_ENV_MAX_BYTES, str(caught.exception))

    def test_a_limit_nothing_fits_raises_rather_than_sending_the_smallest(self):
        # Quietly sending a worse page than was asked for would move 5.1's
        # accuracy with nothing in the run to say why.
        with self.assertRaises(OCRError) as caught:
            shrink(self.image, 500)

        self.assertIn(config.NEMOTRON_ENV_MAX_BYTES, str(caught.exception))


class ShrinkWithoutTheExtraTest(unittest.TestCase):
    """The one place this engine needs a library, and it says which. (4.2)"""

    def setUp(self):
        quieten(self)

    def test_it_names_the_extra_rather_than_failing_as_an_import(self):
        with patch.object(nemotron, "_load_imaging", side_effect=ImportError("no")):
            with self.assertRaises(OCRError) as caught:
                shrink(b"a page that does not fit", 4)

        self.assertIn("[nemotron]", str(caught.exception))

    def test_a_page_that_already_fits_never_asks_for_it(self):
        # Which is why this engine is usable with no extra installed at all.
        with patch.object(
            nemotron, "_load_imaging", side_effect=AssertionError("not needed")
        ):
            body, mime = shrink(b"a page", 1_000_000)

        self.assertEqual(mime, "image/png")


class NemotronEngineTest(unittest.TestCase):
    """The engine, with the one function that opens a socket replaced."""

    def setUp(self):
        quieten(self)
        self.sent = []

    def build(self, **kwargs):
        settings = {"endpoint": ENDPOINT}
        settings.update(kwargs)
        with patch.dict(os.environ, ENVIRONMENT, clear=True):
            return NemotronEngine(**settings)

    def fake_post(self, response=None, error=None):
        def post(url, payload, key, timeout):
            self.sent.append(
                {"url": url, "payload": payload, "key": key, "timeout": timeout}
            )
            if error is not None:
                raise error
            return response if response is not None else answer()

        return post

    def recognize(self, image=b"a page", response=None, error=None, **kwargs):
        engine = self.build(**kwargs)
        with patch.object(nemotron, "_post", self.fake_post(response, error)):
            return engine.recognize(image)

    def test_it_is_named_for_the_engine_not_the_vendor(self):
        self.assertEqual(NemotronEngine.name, "nemotron")

    def test_the_variant_names_the_cache_directory(self):
        # 4.1 compares builds, and the comparison is paid for once. (3.2)
        self.assertEqual(self.build().cache_name, "nemotron-v2_multilingual")

    def test_another_variant_is_another_cache_directory(self):
        self.assertNotEqual(
            self.build(variant="v2_english").cache_name, self.build().cache_name
        )

    def test_the_settings_say_which_deployment_read_the_page(self):
        settings = self.build().settings

        self.assertIn(ENDPOINT, settings)
        self.assertIn("v2_multilingual", settings)
        self.assertIn(f"merge={config.NEMOTRON_MERGE_LEVEL}", settings)

    def test_the_api_key_is_not_part_of_the_cache_key(self):
        # settings is published by GET /v1/engines and written into a client's
        # cache file, so a key in it would hand a credential to every client
        # that asks what this server can do. (6.2, 6.3)
        self.assertNotIn(KEY, self.build().settings)

    def test_a_size_limit_is_part_of_the_cache_key(self):
        # It changes the pixels the engine is shown, so a reading taken under it
        # is not a reading taken without it.
        self.assertNotEqual(
            self.build(max_bytes="180000").settings, self.build().settings
        )

    def test_the_value_checks_are_part_of_the_cache_key(self):
        # A client of this server caches finished rows, not boxes. (3.2, 4.2)
        self.assertIn(checks_fingerprint(), self.build().settings)

    def test_the_grid_is_not_part_of_the_cache_key(self):
        # These boxes are measurements, so moving a column is a free re-read,
        # the way it is for Document AI.
        before = self.build().settings
        with patch.dict(config.CELL_COLUMNS, {"bug_name": (0.4, 0.5)}):
            self.assertEqual(self.build().settings, before)

    def test_a_missing_endpoint_stops_before_anything_is_sent(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(nemotron, "_post") as post:
                with self.assertRaises(ValueError) as caught:
                    NemotronEngine()

        self.assertIn(config.NEMOTRON_ENV_ENDPOINT, str(caught.exception))
        post.assert_not_called()

    def test_a_missing_api_key_stops_before_anything_is_sent(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(nemotron, "_post") as post:
                with self.assertRaises(ValueError) as caught:
                    NemotronEngine(endpoint=ENDPOINT)

        self.assertIn(config.NEMOTRON_ENV_API_KEY, str(caught.exception))
        post.assert_not_called()

    def test_a_size_limit_that_is_not_a_number_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.build(max_bytes="lots")

        self.assertIn(config.NEMOTRON_ENV_MAX_BYTES, str(caught.exception))

    def test_the_endpoint_comes_from_the_environment(self):
        environment = dict(ENVIRONMENT)
        environment[config.NEMOTRON_ENV_ENDPOINT] = "https://from-env"
        with patch.dict(os.environ, environment, clear=True):
            engine = NemotronEngine()

        self.assertIn("https://from-env", engine.settings)

    def test_the_variant_comes_from_the_environment(self):
        environment = dict(ENVIRONMENT)
        environment[config.NEMOTRON_ENV_VARIANT] = "v2_english"
        with patch.dict(os.environ, environment, clear=True):
            engine = NemotronEngine(endpoint=ENDPOINT)

        self.assertEqual(engine.cache_name, "nemotron-v2_english")

    def test_it_posts_to_the_url_it_was_configured_with(self):
        # Nothing is appended. NVIDIA's hosted deployment answers on the model's
        # own URL and a container answers on /v1/ocr, so there is no suffix that
        # is right for both, and a guessed one turns a working setting into a
        # 404 that a half-megabyte body reports as a dropped connection.
        self.recognize()

        self.assertEqual(self.sent[0]["url"], ENDPOINT)

    def test_a_container_endpoint_is_used_as_given_too(self):
        self.recognize(endpoint="http://localhost:8000/v1/ocr")

        self.assertEqual(self.sent[0]["url"], "http://localhost:8000/v1/ocr")

    def test_a_trailing_slash_is_taken_off(self):
        self.recognize(endpoint=ENDPOINT + "/")

        self.assertEqual(self.sent[0]["url"], ENDPOINT)

    def test_it_sends_the_page_as_a_base64_data_uri(self):
        self.recognize(image=b"a page")

        url = self.sent[0]["payload"]["input"][0]["url"]

        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertEqual(self.sent[0]["payload"]["input"][0]["type"], "image_url")

    def test_it_asks_for_the_finest_grouping(self):
        # A merged paragraph would cross the printed rules, and a box is placed
        # by its centre, so a row's answers would land in one column. (6.2)
        self.recognize()

        self.assertEqual(self.sent[0]["payload"]["merge_levels"], ["word"])

    def test_it_authenticates_with_the_key_from_the_environment(self):
        self.recognize()

        self.assertEqual(self.sent[0]["key"], KEY)

    def test_one_page_is_one_call(self):
        self.recognize()

        self.assertEqual(len(self.sent), 1)

    def test_the_reading_comes_back_as_boxes(self):
        found = self.recognize(response=answer(detection("き", 0.1, 0.2, 0.3, 0.4)))

        self.assertEqual([box.text for box in found.boxes], ["き"])

    def test_a_page_under_the_limit_is_sent_untouched(self):
        self.recognize(image=b"a page", max_bytes="100000")

        url = self.sent[0]["payload"]["input"][0]["url"]

        self.assertIn("image/png", url)

    def test_a_refusal_becomes_an_ocr_error(self):
        with self.assertRaises(OCRError):
            self.recognize(error=OCRError("nemotron refused a page: HTTP 500"))

    def test_this_engine_does_not_claim_tokens_it_was_not_told(self):
        # Document AI leaves them at zero for the same reason: a true statement
        # rather than a missing measurement. recognize_with_retry fills in the
        # call and the time. (6.1, 7.1)
        found = self.recognize()

        self.assertEqual(found.usage, Usage())


class PostTest(unittest.TestCase):
    """Every way the request can fail becomes the error 6.4 retries."""

    def setUp(self):
        quieten(self)

    def post(self, error):
        with patch.object(nemotron.urllib.request, "urlopen", side_effect=error):
            return nemotron._post(ENDPOINT, {}, KEY, 1.0)

    def test_a_page_that_is_too_large_says_what_to_do_about_it(self):
        with self.assertRaises(OCRError) as caught:
            self.post(http_error(413, b"payload too large"))

        message = str(caught.exception)

        self.assertIn(config.NEMOTRON_ENV_MAX_BYTES, message)
        self.assertIn(config.NEMOTRON_ENV_ENDPOINT, message)

    def test_another_refusal_carries_the_status_and_the_server_s_words(self):
        with self.assertRaises(OCRError) as caught:
            self.post(http_error(422, b"invalid image url"))

        message = str(caught.exception)

        self.assertIn("422", message)
        self.assertIn("invalid image url", message)

    def test_a_refusal_body_is_bounded(self):
        with self.assertRaises(OCRError) as caught:
            self.post(http_error(500, b"x" * 10_000))

        self.assertLess(len(str(caught.exception)), nemotron.ERROR_BODY_CHARS + 200)

    def test_an_endpoint_that_cannot_be_reached_names_the_setting_to_check(self):
        # The failure this actually produces in the field: a wrong URL, whose
        # 404 arrives as a connection dropped part way through a large upload.
        with self.assertRaises(OCRError) as caught:
            self.post(urllib.error.URLError("no route to host"))

        message = str(caught.exception)

        self.assertIn("no route to host", message)
        self.assertIn(ENDPOINT, message)
        self.assertIn(config.NEMOTRON_ENV_ENDPOINT, message)

    def test_a_timeout_is_an_ocr_error(self):
        with self.assertRaises(OCRError):
            self.post(TimeoutError("timed out"))


@needs_imaging
class ShrinkResolutionTest(unittest.TestCase):
    """Re-encoding spends quality, and only quality. (5.2-3)"""

    def setUp(self):
        quieten(self)
        self.image = noisy_png(width=320, height=240)

    def test_the_page_keeps_every_pixel_it_had(self):
        # Opening the PNG as a document and rendering the page resamples it to
        # whatever the file claims for its own DPI, which is a resolution loss
        # the setting never asked for and the cache key does not record.
        before = pymupdf.Pixmap(self.image)

        body, _ = shrink(self.image, nemotron._encoded_size(self.image) // 2)
        after = pymupdf.Pixmap(body)

        self.assertEqual((after.width, after.height), (before.width, before.height))


class MalformedAnswerTest(unittest.TestCase):
    """An answer that is not an answer says so. (6.4)"""

    def setUp(self):
        quieten(self)

    def build(self):
        with patch.dict(os.environ, ENVIRONMENT, clear=True):
            return NemotronEngine(endpoint=ENDPOINT)

    def test_a_response_that_is_not_an_object_is_an_ocr_error(self):
        # Reading a field off the answer before checking its shape would raise
        # AttributeError instead, and lose the message that says what was wrong.
        engine = self.build()

        with patch.object(nemotron, "_post", return_value=["not", "an", "object"]):
            with self.assertRaises(OCRError) as caught:
                engine.recognize(b"a page")

        self.assertIn("not an object", str(caught.exception))

    def test_a_response_with_no_data_is_an_ocr_error(self):
        engine = self.build()

        with patch.object(nemotron, "_post", return_value={"model": "x"}):
            with self.assertRaises(OCRError):
                engine.recognize(b"a page")


class PublishedSettingsTest(unittest.TestCase):
    """What GET /v1/engines can say about this engine without building one.

    For this engine that also means without the API key, which a built engine
    demands and which must never reach the published string anyway. (3.2, 6.3)
    """

    def test_the_settings_are_answerable_without_a_credential(self):
        with patch.dict(os.environ, {config.NEMOTRON_ENV_ENDPOINT: ENDPOINT}):
            with patch(
                "creature_ocr_server.engines.auth.api_key",
                side_effect=AssertionError("no credential may be read"),
            ):
                settings = NemotronEngine.settings_for("")

        self.assertIn(ENDPOINT, settings)
        self.assertNotIn(KEY, settings)

    def test_the_cache_directory_is_answerable_the_same_way(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                NemotronEngine.cache_name_for(""), "nemotron-v2_multilingual"
            )

    def test_an_unconfigured_server_still_answers(self):
        # /v1/engines is what a client calls to find out it is not configured,
        # so it cannot be the thing that needs configuration. (6.5)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("nemotron-ocr-v2", NemotronEngine.settings_for(""))

    def test_it_offers_no_model(self):
        # The deployment the URL points at is the reader. (6.5)
        self.assertEqual(NemotronEngine.models(), ())
        self.assertEqual(NemotronEngine.default_model(), "")


class RegistryTest(unittest.TestCase):
    """This engine is reachable and reportable the way every engine is."""

    def test_it_is_in_the_registry_under_its_own_name(self):
        self.assertIs(engines.ENGINES["nemotron"], NemotronEngine)

    def test_a_request_may_not_name_a_model_for_it(self):
        with self.assertRaises(ValueError) as caught:
            engines.choose_model("nemotron", "v2_english")

        self.assertIn("takes no model", str(caught.exception))

    def test_readiness_is_reported_without_opening_a_socket(self):
        with patch.dict(os.environ, CONFIGURED):
            with patch.object(
                nemotron, "_post", side_effect=AssertionError("nothing may be sent")
            ):
                ready, _ = engines.status("nemotron")

        self.assertTrue(ready)

    def test_a_missing_endpoint_is_named(self):
        with patch.dict(os.environ, {}, clear=True):
            ready, detail = engines.status("nemotron")

        self.assertFalse(ready)
        self.assertIn(config.NEMOTRON_ENV_ENDPOINT, detail)

    def test_a_missing_api_key_is_named(self):
        with patch.dict(
            os.environ, {config.NEMOTRON_ENV_ENDPOINT: ENDPOINT}, clear=True
        ):
            ready, detail = engines.status("nemotron")

        self.assertFalse(ready)
        self.assertIn(config.NEMOTRON_ENV_API_KEY, detail)

    def test_it_is_usable_with_no_extra_installed(self):
        # There is no SDK to find, and the one library shrink can want is only
        # reached when NEMOTRON_MAX_BYTES asks for it. (4.2)
        self.assertEqual(NemotronEngine.sdk, "")
        with patch.dict(os.environ, CONFIGURED):
            with patch("importlib.util.find_spec", return_value=None):
                ready, _ = engines.status("nemotron")

        self.assertTrue(ready)


if __name__ == "__main__":
    unittest.main()
