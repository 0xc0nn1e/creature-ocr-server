"""pyClaudeApi engine tests.

No network: the request goes through lmstudio._post, which is replaced here as
test_lmstudio replaces it. The request and the parsing are lmstudio's and are
tested there; what is tested here is what this engine changes - its own
settings, a demanded URL and key, and a cache key that names it.
"""

import json
import logging
import os
import unittest
from unittest.mock import patch

from creature_ocr_server import config, engines
from creature_ocr_server.engines import lmstudio
from creature_ocr_server.engines.gemini import build_prompt, build_prompt_v2
from creature_ocr_server.engines.pyclaudeapi import PyClaudeAPIEngine

PAGE = b"a whole page of PNG"
BASE_URL = "http://127.0.0.1:8010/v1"
URL = BASE_URL + "/chat/completions"
KEY = "pyclaudeapi-not-a-real-key"
CONFIGURED = {
    config.PYCLAUDEAPI_ENV_BASE_URL: BASE_URL,
    config.PYCLAUDEAPI_ENV_API_KEY: KEY,
}


def answer(rows=None):
    """One chat completion, as pyClaudeApi shapes it."""
    return {
        "model": "claude-cli",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": json.dumps(rows or [])},
                "finish_reason": "stop",
            }
        ],
    }


def quieten(case):
    """Keep the engine's log lines out of the test output."""
    logger = logging.getLogger("creature_ocr_server")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    case.addCleanup(logger.setLevel, previous)


class PyClaudeAPIEngineTest(unittest.TestCase):
    """The engine, with the one function that opens a socket replaced."""

    def setUp(self):
        quieten(self)
        self.sent = []

    def build(self, environment=None, **kwargs):
        with patch.dict(os.environ, CONFIGURED if environment is None else environment, clear=True):
            return PyClaudeAPIEngine(**kwargs)

    def recognize(self, **kwargs):
        engine = self.build(**kwargs)

        def post(url, payload, key, timeout):
            self.sent.append({"url": url, "payload": payload, "key": key, "timeout": timeout})
            return answer()

        with patch.object(lmstudio, "_post", post):
            return engine.recognize(PAGE)

    def test_it_is_registered_under_its_own_name(self):
        self.assertIs(engines.ENGINES["pyclaudeapi"], PyClaudeAPIEngine)

    def test_it_posts_to_the_chat_endpoint_with_its_own_key(self):
        self.recognize()

        self.assertEqual(self.sent[0]["url"], URL)
        self.assertEqual(self.sent[0]["key"], KEY)

    def test_it_asks_gemini_s_question_and_names_no_model(self):
        self.recognize()
        payload = self.sent[0]["payload"]

        self.assertEqual(payload["model"], "")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": build_prompt()})

    def test_the_v2_sheet_asks_the_v2_question(self):
        self.recognize(sheet=config.SHEET_V2)

        self.assertEqual(
            self.sent[0]["payload"]["messages"][0]["content"], build_prompt_v2()
        )

    def test_lmstudio_s_settings_do_not_configure_it(self):
        environment = dict(CONFIGURED)
        environment[config.LMSTUDIO_ENV_MODEL] = "qwen2.5-vl-7b"
        environment[config.LMSTUDIO_ENV_TIMEOUT] = "5"
        self.recognize(environment=environment)

        self.assertEqual(self.sent[0]["payload"]["model"], "")
        self.assertEqual(self.sent[0]["timeout"], config.PYCLAUDEAPI_TIMEOUT_SECONDS)

    def test_the_timeout_is_its_own_setting(self):
        environment = dict(CONFIGURED)
        environment[config.PYCLAUDEAPI_ENV_TIMEOUT] = "90"
        self.recognize(environment=environment)

        self.assertEqual(self.sent[0]["timeout"], 90.0)

    def test_a_timeout_that_is_not_a_number_names_the_setting(self):
        environment = dict(CONFIGURED)
        environment[config.PYCLAUDEAPI_ENV_TIMEOUT] = "soon"

        with self.assertRaisesRegex(ValueError, config.PYCLAUDEAPI_ENV_TIMEOUT):
            self.build(environment)

    def test_a_timeout_that_is_not_finite_names_the_setting(self):
        # float() takes all three, and a socket takes none of them.
        for value in ("inf", "nan", "1e999"):
            environment = dict(CONFIGURED)
            environment[config.PYCLAUDEAPI_ENV_TIMEOUT] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, config.PYCLAUDEAPI_ENV_TIMEOUT):
                    self.build(environment)

    def test_a_missing_url_is_refused_rather_than_guessed(self):
        environment = {config.PYCLAUDEAPI_ENV_API_KEY: KEY}

        with self.assertRaisesRegex(ValueError, config.PYCLAUDEAPI_ENV_BASE_URL):
            self.build(environment)

    def test_a_url_that_is_not_http_is_refused_rather_than_guessed(self):
        # Only slashes would fall back to LM Studio's address; no scheme, or a
        # URL urllib cannot split, would report ready and fail every page.
        for url in ("/", " // ", "127.0.0.1:8010/v1", "http://[::1:8010/v1"):
            environment = dict(CONFIGURED)
            environment[config.PYCLAUDEAPI_ENV_BASE_URL] = url
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, config.PYCLAUDEAPI_ENV_BASE_URL):
                    self.build(environment)

    def test_a_url_that_is_not_http_publishes_what_an_unset_one_does(self):
        # GET /v1/engines asks every engine at once, so this must not raise.
        unset = PyClaudeAPIEngine.settings_for(base_url="")
        for url in ("/", "127.0.0.1:8010/v1", "http://[::1:8010/v1"):
            environment = {config.PYCLAUDEAPI_ENV_BASE_URL: url}
            with self.subTest(url=url), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(PyClaudeAPIEngine.settings_for(), unset)

    def test_a_missing_key_is_refused(self):
        environment = {config.PYCLAUDEAPI_ENV_BASE_URL: BASE_URL}

        with self.assertRaisesRegex(ValueError, config.PYCLAUDEAPI_ENV_API_KEY):
            self.build(environment)

    def test_a_key_a_header_cannot_carry_is_refused_without_repeating_it(self):
        # http.client would quote it in an error, and each retry would log that.
        for key in ("sk-secret\nsecond-line", "sk-secret-秘"):
            environment = dict(CONFIGURED)
            environment[config.PYCLAUDEAPI_ENV_API_KEY] = key
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as caught:
                    self.build(environment)

                self.assertIn(config.PYCLAUDEAPI_ENV_API_KEY, str(caught.exception))
                self.assertNotIn("sk-secret", str(caught.exception))

    def test_status_names_what_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            ready, detail = engines.status("pyclaudeapi")

        self.assertFalse(ready)
        self.assertIn(config.PYCLAUDEAPI_ENV_BASE_URL, detail)

    def test_the_key_is_never_published(self):
        engine = self.build()

        self.assertNotIn(KEY, engine.settings)
        self.assertNotIn(KEY, engine.cache_name)

    def test_the_cache_key_names_the_engine_and_the_url(self):
        engine = self.build()

        self.assertTrue(engine.settings.startswith(f"pyclaudeapi {URL} "))
        self.assertEqual(engine.cache_name, "pyclaudeapi")
        with patch.dict(os.environ, CONFIGURED, clear=True):
            self.assertEqual(PyClaudeAPIEngine.settings_for(), engine.settings)

    def test_an_unconfigured_key_does_not_borrow_lm_studio_s_address(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = PyClaudeAPIEngine.settings_for()

        self.assertNotIn(config.LMSTUDIO_BASE_URL, settings)

    def test_each_sheet_has_its_own_cache_key(self):
        v2 = self.build(sheet=config.SHEET_V2)

        self.assertIn(f"sheet={config.SHEET_V2}", v2.settings)
        self.assertEqual(v2.cache_name, f"pyclaudeapi-{config.SHEET_V2}")


if __name__ == "__main__":
    unittest.main()
