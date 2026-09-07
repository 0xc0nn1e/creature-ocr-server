"""The HTTP surface.

No SDK, no network and no credential: the registry is patched to hold a double
that answers with whatever boxes a case wants. Every status code this server
can produce has a case here, because a status code is the whole of what a
client sees when something goes wrong.
"""

import logging
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from creature_ocr_server import config, engines, ocr
from creature_ocr_server.api import security
from creature_ocr_server.api.app import app
from creature_ocr_server.ocr import OCREngine, OCRError, OCRTimeout, Reading, Usage

from test_ocr import FIELDS, box, filled_row

PNG = config.PNG_MAGIC + b"a page that says it is a PNG"


class FakeEngine(OCREngine):
    """A registry entry that needs nothing and answers with fixed boxes."""

    name = "fake"
    boxes = ()
    error = None
    cost = Usage(calls=0, prompt_tokens=11, output_tokens=22, thought_tokens=3)
    notes = ()

    def __init__(self, model=None):
        self.model = model or "fake-1"
        self.settings = f"fake {self.model} checks={ocr.checks_fingerprint()}"
        self.cache_name = self.model

    @classmethod
    def models(cls):
        return ("fake-1", "fake-2")

    @classmethod
    def default_model(cls):
        return "fake-1"

    @classmethod
    def settings_for(cls, model):
        return f"fake {model} checks={ocr.checks_fingerprint()}"

    @classmethod
    def cache_name_for(cls, model):
        return model

    def recognize(self, image, mime_type="image/png"):
        if type(self).error is not None:
            raise type(self).error
        return Reading(list(type(self).boxes), type(self).cost, list(type(self).notes))


def answering(boxes=(), error=None, notes=(), **attributes):
    """A registry holding one FakeEngine subclass set up for this case."""
    engine = type(
        "Answering",
        (FakeEngine,),
        {"boxes": tuple(boxes), "error": error, "notes": tuple(notes), **attributes},
    )
    return {"fake": engine}


class ApiTestCase(unittest.TestCase):
    """One client, one patched registry, nothing left behind."""

    def setUp(self):
        self.addCleanup(engines.forget)
        engines.forget()
        # These cases feed deliberately partial pages and start a server with
        # no key, both of which this package is right to say something about
        # and neither of which belongs in the test output.
        package = logging.getLogger("creature_ocr_server")
        previous = package.level
        package.setLevel(logging.CRITICAL)
        self.addCleanup(package.setLevel, previous)

    def client(self, registry=None, environment=None):
        """A TestClient whose registry holds only the double this case wants."""
        registry = registry if registry is not None else answering()
        self.enterContext(patch.dict(engines.ENGINES, registry, clear=True))
        self.enterContext(
            patch.dict(
                "os.environ",
                {config.OCR_ENGINE_ENV: "fake", **(environment or {})},
            )
        )
        return self.enterContext(TestClient(app))

    def read(self, client, image=PNG, headers=None, **form):
        """POST one page."""
        return client.post(
            "/v1/ocr",
            files={"image": ("page01.png", image, "image/png")},
            data=form,
            headers=headers,
        )


class HealthTest(ApiTestCase):
    """It has to answer on a server that is not configured at all.

    A container healthcheck calls it, and a healthcheck that failed while the
    engine was unconfigured - or while a four-minute page was in flight - would
    restart the process instead of reporting the problem. (6.1)
    """

    def test_it_says_it_is_alive(self):
        found = self.client().get("/v1/health")

        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.json()["status"], "ok")

    def test_it_needs_no_key(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        self.assertEqual(client.get("/v1/health").status_code, 200)

    def test_it_answers_when_the_engine_cannot_be_built(self):
        # A server whose cloud is having a bad morning should still come up and
        # say so, rather than refuse to start and leave nothing to ask.
        registry = answering()
        registry["fake"].__init__ = lambda self, model=None: (_ for _ in ()).throw(
            RuntimeError("the credential is unreadable")
        )

        self.assertEqual(self.client(registry).get("/v1/health").status_code, 200)


class EnginesTest(ApiTestCase):
    """What a client is told so it needs no configuration of its own."""

    def test_it_lists_what_there_is(self):
        found = self.client().get("/v1/engines").json()

        self.assertEqual([engine["name"] for engine in found], ["fake"])

    def test_it_publishes_the_settings_a_client_keys_its_cache_on(self):
        # The client looks in its cache before it asks for a page, so it has to
        # be able to build the key without having asked - and it cannot work
        # this string out for itself any more. (3.2, 4.2)
        found = self.client().get("/v1/engines").json()[0]

        listed = {model["name"]: model["settings"] for model in found["models"]}
        self.assertEqual(listed["fake-1"], FakeEngine.settings_for("fake-1"))
        self.assertEqual(listed["fake-2"], FakeEngine.settings_for("fake-2"))

    def test_it_names_the_cache_directory_per_model(self):
        found = self.client().get("/v1/engines").json()[0]

        self.assertEqual(
            [model["cache_name"] for model in found["models"]], ["fake-1", "fake-2"]
        )

    def test_an_engine_with_no_models_publishes_its_own_cache_key(self):
        # documentai is read by a processor and nemotron by a deployment, so
        # neither lists a model - and without these two fields their key
        # existed nowhere a client could read it before asking. Looking in the
        # cache first is the one thing this endpoint is for. (3.2)
        registry = answering(
            models=classmethod(lambda cls: ()),
            default_model=classmethod(lambda cls: ""),
            settings_for=classmethod(lambda cls, model: "processor-7 checks=abc"),
            cache_name_for=classmethod(lambda cls, model: "fake-processor-7"),
        )
        found = self.client(registry).get("/v1/engines").json()[0]

        self.assertEqual(found["models"], [])
        self.assertEqual(found["settings"], "processor-7 checks=abc")
        self.assertEqual(found["cache_name"], "fake-processor-7")

    def test_an_engine_with_models_says_nothing_at_its_own_level(self):
        # There the key belongs to the model and ModelInfo already carries it.
        # One built from a blank model id would name a reader that does not
        # exist, which is worse than saying nothing.
        found = self.client().get("/v1/engines").json()[0]

        self.assertTrue(found["models"])
        self.assertEqual(found["settings"], "")
        self.assertEqual(found["cache_name"], "")

    def test_an_unusable_engine_says_why_rather_than_disappearing(self):
        registry = answering(requires=("NOT_SET_ANYWHERE",))
        found = self.client(registry).get("/v1/engines").json()[0]

        self.assertFalse(found["ready"])
        self.assertIn("NOT_SET_ANYWHERE", found["detail"])

    def test_a_ready_engine_says_so_with_no_detail(self):
        found = self.client().get("/v1/engines").json()[0]

        self.assertTrue(found["ready"])
        self.assertEqual(found["detail"], "")


class SheetTest(ApiTestCase):
    """The paper, so both halves of the pipeline describe the same one."""

    def test_it_carries_the_fingerprint_the_settings_quote(self):
        found = self.client().get("/v1/sheet").json()

        self.assertEqual(found["fingerprint"], ocr.checks_fingerprint())

    def test_it_carries_the_columns_in_printed_order(self):
        found = self.client().get("/v1/sheet").json()

        self.assertEqual([f["key"] for f in found["sheet"]["fields"]], FIELDS)

    def test_it_is_what_the_fingerprint_is_taken_over(self):
        found = self.client().get("/v1/sheet").json()

        self.assertEqual(found["sheet"], config.sheet_definition())


class AuthTest(ApiTestCase):
    """A server holding a cloud credential should not answer just anyone."""

    def test_no_key_configured_means_no_check(self):
        self.assertEqual(self.client().get("/v1/engines").status_code, 200)

    def test_a_configured_key_is_required(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        self.assertEqual(client.get("/v1/engines").status_code, 401)

    def test_the_right_key_gets_in(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={security.HEADER: "secret"})

        self.assertEqual(found.status_code, 200)

    def test_a_wrong_key_is_refused(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={security.HEADER: "guess"})

        self.assertEqual(found.status_code, 401)

    def test_reading_a_page_needs_the_key_too(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        self.assertEqual(self.read(client).status_code, 401)

    def test_the_refusal_says_nothing_about_the_key(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={security.HEADER: "guess"})

        self.assertNotIn("secret", found.text)
        self.assertNotIn("guess", found.text)

    def test_a_bearer_token_carries_the_same_key(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={"Authorization": "Bearer secret"})

        self.assertEqual(found.status_code, 200)

    def test_the_bearer_scheme_is_read_without_case(self):
        # RFC 7235 says the scheme is case insensitive, and a generated client
        # is as likely to write one as the other.
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={"Authorization": "bearer secret"})

        self.assertEqual(found.status_code, 200)

    def test_a_wrong_bearer_token_is_refused(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={"Authorization": "Bearer guess"})

        self.assertEqual(found.status_code, 401)

    def test_another_scheme_is_not_a_key(self):
        # Basic base64("secret") is not the key, and must not be read as one.
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines", headers={"Authorization": "Basic c2VjcmV0"})

        self.assertEqual(found.status_code, 401)

    def test_reading_a_page_takes_a_bearer_token_too(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = self.read(client, headers={"Authorization": "Bearer secret"})

        self.assertEqual(found.status_code, 200)

    def test_the_refusal_says_which_headers_are_read(self):
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        found = client.get("/v1/engines")

        self.assertIn(security.HEADER, found.json()["detail"])
        self.assertIn(security.SCHEME, found.json()["detail"])
        # A 401 that accepts Bearer is supposed to say so. It names the scheme
        # and never the secret.
        self.assertEqual(found.headers["WWW-Authenticate"], security.SCHEME)

    def test_a_header_that_is_not_ascii_is_a_refusal_not_a_crash(self):
        # A header arrives as whatever the client sent, decoded latin-1, and
        # hmac.compare_digest refuses a str that is not ASCII. Comparing text
        # made this a 500 that an unauthenticated request could ask for at
        # will, which on a server bound to anything but the loopback address is
        # a crash anyone can reach. (6.2)
        client = self.client(environment={config.SERVER_ENV_API_KEY: "secret"})

        # As bytes, because httpx will not encode a non-ASCII str into a
        # header at all - which is exactly why this arrives from a real client
        # rather than from a well behaved library.
        for headers in (
            {security.HEADER.encode(): b"\xff"},
            {b"Authorization": b"Bearer \xff"},
        ):
            with self.subTest(headers=headers):
                self.assertEqual(
                    client.get("/v1/engines", headers=headers).status_code, 401
                )


class ReadPageTest(ApiTestCase):
    """The shape of an answer, which is the whole contract with the client."""

    def test_a_page_comes_back_as_rows(self):
        found = self.read(self.client(answering(filled_row(1))))

        self.assertEqual(found.status_code, 200)
        self.assertEqual(len(found.json()["rows"]), config.ROWS_PER_PAGE)

    def test_every_row_carries_every_column(self):
        found = self.read(self.client()).json()

        for row in found["rows"]:
            self.assertEqual(set(row), {"no", *FIELDS})

    def test_the_row_number_is_a_string(self):
        # It is a string in the desktop pipeline, where it goes straight into
        # the xlsx. A helpful round trip that made it an integer would be a
        # silent change in a client nobody edited.
        found = self.read(self.client()).json()

        self.assertEqual(
            [row["no"] for row in found["rows"]],
            [str(number) for number in range(1, config.ROWS_PER_PAGE + 1)],
        )

    def test_a_blank_page_still_returns_every_row(self):
        # 5.1 allows no row to be dropped, blank or not.
        found = self.read(self.client()).json()

        self.assertEqual(len(found["rows"]), config.ROWS_PER_PAGE)
        self.assertEqual({row["bug_name"] for row in found["rows"]}, {""})

    def test_it_says_which_engine_and_model_read_the_page(self):
        found = self.read(self.client(), model="fake-2").json()

        self.assertEqual(found["engine"], "fake")
        self.assertEqual(found["model"], "fake-2")
        self.assertEqual(found["settings"], FakeEngine.settings_for("fake-2"))
        self.assertEqual(found["cache_name"], "fake-2")

    def test_the_settings_match_what_the_engines_endpoint_published(self):
        client = self.client()
        listed = client.get("/v1/engines").json()[0]
        published = {m["name"]: m["settings"] for m in listed["models"]}

        found = self.read(client, model="fake-1").json()

        self.assertEqual(found["settings"], published["fake-1"])

    def test_the_sheet_fingerprint_rides_along(self):
        found = self.read(self.client()).json()

        self.assertEqual(found["sheet_fingerprint"], ocr.checks_fingerprint())

    def test_the_usage_can_be_read_straight_back_into_the_desktop_type(self):
        found = self.read(self.client()).json()

        spent = Usage(**found["usage"])
        self.assertEqual(spent.calls, 1)
        self.assertEqual(spent.prompt_tokens, 11)

    def test_the_request_id_comes_back_in_the_body_and_the_header(self):
        found = self.read(self.client())

        self.assertEqual(found.json()["request_id"], found.headers["X-Request-ID"])

    def test_a_request_id_the_client_chose_is_kept(self):
        client = self.client()

        found = client.post(
            "/v1/ocr",
            files={"image": ("page01.png", PNG, "image/png")},
            headers={"X-Request-ID": "batch-7-page-3"},
        )

        self.assertEqual(found.json()["request_id"], "batch-7-page-3")


class ReportTest(ApiTestCase):
    """The per-page quality numbers, and the cells worth looking at. (7.2)"""

    def test_a_clean_page_scores_and_complains_about_nothing(self):
        found = self.read(self.client(answering(filled_row(1)))).json()

        self.assertEqual(found["report"]["percent"], "100%")
        self.assertEqual(found["report"]["cells"], [])
        self.assertEqual(found["findings"], [])

    def test_a_page_that_offered_nothing_has_no_score(self):
        # Not 1.0 and not 0%: either the sheet was blank or the engine read a
        # filled sheet and returned nothing, and nothing here can tell those
        # apart. The page most worth opening must not look like the best one.
        found = self.read(self.client()).json()

        self.assertIsNone(found["report"]["score"])
        self.assertEqual(found["report"]["percent"], "")

    def test_a_rejected_cell_is_marked_and_explained(self):
        boxes = filled_row(2, location_chome="山")
        found = self.read(self.client(answering(boxes))).json()

        self.assertEqual(found["report"]["rejected"], 1)
        self.assertIn(
            {"row": 2, "field": "location_chome", "problem": ocr.CELL_REJECTED},
            found["report"]["cells"],
        )
        self.assertTrue(
            any("location_chome" in note for note in found["findings"])
        )

    def test_a_hole_in_a_filled_row_is_marked(self):
        # The recomputable half of the marks, sent as well: the client can work
        # it out but should not have to guess whether it agrees with us.
        boxes = filled_row(3, notice="")
        found = self.read(self.client(answering(boxes))).json()

        self.assertIn(
            {"row": 3, "field": "notice", "problem": ocr.CELL_MISSING},
            found["report"]["cells"],
        )

    def test_a_cell_the_engine_strained_over_is_marked(self):
        boxes = filled_row(4, where="駐車場の花だん", unsure=["where"])
        found = self.read(self.client(answering(boxes))).json()

        self.assertEqual(found["report"]["unsure"], 1)
        self.assertIn(
            {"row": 4, "field": "where", "problem": ocr.CELL_UNSURE},
            found["report"]["cells"],
        )

    def test_a_finding_carries_no_page_in_front_of_it(self):
        # The client prepends the filename, because only the client knows it.
        # Doing it at both ends, or at neither, makes a batch log useless. (7.2)
        boxes = filled_row(2, location_chome="山")
        found = self.read(self.client(answering(boxes))).json()

        for finding in found["findings"]:
            self.assertTrue(finding.startswith("row "), finding)

    def test_what_the_engine_worked_around_reaches_the_client(self):
        registry = answering(filled_row(1), notes=["dropped a second row numbered 3"])
        found = self.read(self.client(registry)).json()

        self.assertEqual(found["findings"][0], "dropped a second row numbered 3")


class RefusalTest(ApiTestCase):
    """Every way a request can be turned away, and the code that says which."""

    def test_no_image_at_all_is_a_validation_error(self):
        found = self.client().post("/v1/ocr", data={"engine": "fake"})

        self.assertEqual(found.status_code, 422)

    def test_something_that_is_not_a_png_is_refused(self):
        # The declared content type is a claim; the magic bytes are a fact, and
        # the type is passed on to the model. (6.2)
        found = self.read(self.client(), image=b"\xff\xd8\xff a JPEG really")

        self.assertEqual(found.status_code, 415)

    def test_a_page_over_the_limit_is_refused(self):
        client = self.client(environment={config.SERVER_ENV_MAX_IMAGE_BYTES: "64"})

        found = self.read(client, image=config.PNG_MAGIC + b"x" * 512)

        self.assertEqual(found.status_code, 413)

    def test_an_unknown_engine_lists_what_there_is(self):
        # A name no registry holds. Not a real engine's: this case is about a
        # name that is not registered, and reading as though documentai were
        # still unimplemented would age badly.
        found = self.read(self.client(), engine="no-such-engine")

        self.assertEqual(found.status_code, 400)
        self.assertIn("fake", found.json()["detail"])

    def test_a_model_off_the_allow_list_is_refused(self):
        # It is a memory bound as much as a menu: an engine is built once per
        # model and kept, so a free-form string would grow an SDK client a
        # request at a time.
        found = self.read(self.client(), model="fake-99")

        self.assertEqual(found.status_code, 400)
        self.assertIn("fake-99", found.json()["detail"])

    def test_an_unconfigured_engine_says_which_setting_is_missing(self):
        registry = answering(requires=("NOT_SET_ANYWHERE",))

        found = self.read(self.client(registry))

        self.assertEqual(found.status_code, 503)
        self.assertIn("NOT_SET_ANYWHERE", found.json()["detail"])

    def test_an_engine_that_gives_up_is_a_bad_gateway(self):
        registry = answering(error=OCRError("Gemini refused a page"))

        found = self.read(self.client(registry))

        self.assertEqual(found.status_code, 502)

    def test_the_vendor_message_never_reaches_the_client(self):
        # A Vertex error routinely carries the project id and sometimes the
        # service account address. The rule that keeps a credential out of an
        # engine's settings and out of a log applies to an error body. (6.2, 6.3)
        secret = "projects/very-private-project-1234"
        registry = answering(error=OCRError(f"Gemini refused a page: {secret}"))

        found = self.read(self.client(registry))

        self.assertEqual(found.status_code, 502)
        self.assertNotIn(secret, found.text)
        # The id is how the operator finds the real message in the server log,
        # which is the only place it now exists.
        self.assertIn(found.headers["X-Request-ID"], found.json()["detail"])

    def test_running_out_of_time_is_not_reported_as_an_engine_failure(self):
        # A budget the server set itself is not the engine being broken, and
        # answering 502 would send an operator to look at Vertex. (6.1, 6.4)
        client = self.client(
            answering(filled_row(1)),
            environment={config.SERVER_ENV_REQUEST_DEADLINE: "0"},
        )

        found = self.read(client)

        self.assertEqual(found.status_code, 504)
        self.assertIn(found.headers["X-Request-ID"], found.json()["detail"])


class DeterminismTest(ApiTestCase):
    """5.2-4 is a promise about numbers no request may move."""

    def test_temperature_is_not_a_request_parameter(self):
        # The moment a caller can send one, re-running the same image stops
        # being guaranteed to give the same rows - silently, and only for
        # whichever pages were read that day.
        found = self.read(self.client(), temperature="1.0")

        self.assertEqual(found.status_code, 200)
        self.assertIn("checks=", found.json()["settings"])

    def test_the_same_page_twice_gives_the_same_rows(self):
        client = self.client(answering(filled_row(1)))

        first = self.read(client).json()["rows"]
        second = self.read(client).json()["rows"]

        self.assertEqual(first, second)


class SharedEngineTest(ApiTestCase):
    """One engine per model, however many requests name it."""

    def test_a_second_request_reuses_the_engine(self):
        built = []
        registry = answering(filled_row(1))
        original = registry["fake"].__init__

        def counted(self, model=None):
            built.append(model)
            original(self, model)

        registry["fake"].__init__ = counted
        client = self.client(registry)

        self.read(client)
        self.read(client)

        # One build, not three: the startup warm-up and both requests all want
        # the same engine, because none of them named a model.
        self.assertEqual(built, [None])

    def test_another_model_is_another_engine(self):
        built = []
        registry = answering(filled_row(1))
        original = registry["fake"].__init__

        def counted(self, model=None):
            built.append(model)
            original(self, model)

        registry["fake"].__init__ = counted
        client = self.client(registry)

        self.read(client, model="fake-1")
        self.read(client, model="fake-2")

        self.assertEqual([model for model in built if model], ["fake-1", "fake-2"])


    def test_the_default_engine_is_warmed_at_startup(self):
        # google.auth refreshes a token without a lock, so one fetch before
        # anybody asks makes a race between the first few requests a
        # non-question rather than a rare one.
        built = []
        registry = answering()
        original = registry["fake"].__init__

        def counted(self, model=None):
            built.append(model)
            original(self, model)

        registry["fake"].__init__ = counted

        self.client(registry)

        self.assertEqual(built, [None])


if __name__ == "__main__":
    unittest.main()
