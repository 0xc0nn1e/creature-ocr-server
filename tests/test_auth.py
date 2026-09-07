"""Credential selection tests.

No cloud SDK and no real key: google.oauth2 is reached through the one loader
function that imports it, so the suite passes with google-auth not installed,
and every key here is a dict this file made up.
"""

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from creature_ocr_server import config
from creature_ocr_server.engines import auth

# Shaped like a service account key, worth nothing. The private key is a
# recognisable fake: no test gets far enough to parse it, and a grep for it in
# the output directory should never find anything.
KEY = {
    "type": "service_account",
    "project_id": "sheets-ocr",
    "private_key_id": "0" * 40,
    "private_key": "-----BEGIN PRIVATE KEY-----\nNOTAREALKEY==\n-----END PRIVATE KEY-----\n",
    "client_email": "ocr-runner@sheets-ocr.iam.gserviceaccount.com",
    "token_uri": "https://oauth2.googleapis.com/token",
}


def fake_sdk(email=KEY["client_email"]):
    """A google.oauth2.service_account stand-in that loads any key."""
    service_account = MagicMock()
    built = service_account.Credentials.from_service_account_info.return_value
    built.service_account_email = email
    return service_account


def load(setting, sdk=None):
    """Ask for credentials with the SDK replaced, never reading the real one."""
    sdk = sdk if sdk is not None else fake_sdk()
    with patch("creature_ocr_server.engines.auth._load_sdk", return_value=sdk):
        return auth.credentials(setting)


class NoSettingTest(unittest.TestCase):
    """An empty setting is the Application Default Credentials path."""

    def test_an_unset_variable_leaves_the_sdk_to_resolve_adc(self):
        sdk = fake_sdk()

        with patch.dict(os.environ, {}, clear=True):
            found = load(None, sdk)

        self.assertIsNone(found)
        sdk.Credentials.from_service_account_info.assert_not_called()

    def test_an_empty_variable_counts_as_unset(self):
        # A bare GOOGLE_APPLICATION_CREDENTIALS= line in .env is a setting
        # nobody meant to make, and must not become an error.
        environment = {config.GOOGLE_ENV_CREDENTIALS: ""}

        with patch.dict(os.environ, environment, clear=True):
            self.assertIsNone(load(None))

    def test_whitespace_only_counts_as_unset(self):
        self.assertIsNone(load("   "))

    def test_the_adc_path_never_imports_google_auth(self):
        # The package has to keep working with no cloud SDK installed at all,
        # and ADC needs nothing from this module. (4.2)
        with patch(
            "creature_ocr_server.engines.auth._load_sdk", side_effect=AssertionError
        ):
            self.assertIsNone(auth.credentials(""))


class InlineKeyTest(unittest.TestCase):
    """The key pasted into .env."""

    def test_a_pasted_key_is_loaded_with_the_platform_scope(self):
        sdk = fake_sdk()

        found = load(json.dumps(KEY), sdk)

        sdk.Credentials.from_service_account_info.assert_called_once_with(
            KEY, scopes=auth.SCOPES
        )
        self.assertIs(found, sdk.Credentials.from_service_account_info.return_value)

    def test_a_key_pasted_across_lines_is_loaded(self):
        # What config.load_env hands back when an operator pastes the file as
        # the console downloaded it.
        found = load(json.dumps(KEY, indent=2))

        self.assertIsNotNone(found)

    def test_the_setting_is_read_from_the_environment(self):
        sdk = fake_sdk()
        environment = {config.GOOGLE_ENV_CREDENTIALS: json.dumps(KEY)}

        with patch.dict(os.environ, environment, clear=True):
            load(None, sdk)

        sdk.Credentials.from_service_account_info.assert_called_once_with(
            KEY, scopes=auth.SCOPES
        )

    def test_something_that_opens_a_brace_but_is_not_json_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            load('{"type": "service_account"')

        self.assertIn(config.GOOGLE_ENV_CREDENTIALS, str(caught.exception))
        self.assertIn("not valid JSON", str(caught.exception))


class KeyFileTest(unittest.TestCase):
    """The key left in its own file, which is what this variable means to
    every other Google library."""

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path = self.tmp / "ocr-runner.json"
        self.path.write_text(json.dumps(KEY), encoding="utf-8")

    def test_a_path_is_read_and_loaded_with_the_platform_scope(self):
        sdk = fake_sdk()

        load(str(self.path), sdk)

        sdk.Credentials.from_service_account_info.assert_called_once_with(
            KEY, scopes=auth.SCOPES
        )

    def test_a_path_that_is_not_there_names_the_file_and_the_setting(self):
        missing = self.tmp / "gone.json"

        with self.assertRaises(ValueError) as caught:
            load(str(missing))

        self.assertIn(config.GOOGLE_ENV_CREDENTIALS, str(caught.exception))
        self.assertIn(str(missing), str(caught.exception))

    def test_a_directory_is_refused_rather_than_read(self):
        with self.assertRaises(ValueError):
            load(str(self.tmp))

    def test_a_file_that_is_not_json_is_refused(self):
        self.path.write_text("not a key at all", encoding="utf-8")

        with self.assertRaises(ValueError) as caught:
            load(str(self.path))

        self.assertIn("not valid JSON", str(caught.exception))


class RefusalTest(unittest.TestCase):
    """A setting that is present but unusable stops the run."""

    def test_a_key_the_loader_rejects_becomes_a_setting_error(self):
        # google-auth and the cryptography library under it raise several
        # unrelated types for a key that will not load. All of them have to
        # reach the operator as one sentence about .env.
        for broken in (ValueError("no key"), KeyError("client_email"), TypeError):
            with self.subTest(broken=broken):
                sdk = fake_sdk()
                sdk.Credentials.from_service_account_info.side_effect = broken

                with self.assertRaises(ValueError) as caught:
                    load(json.dumps(KEY), sdk)

                self.assertIn(config.GOOGLE_ENV_CREDENTIALS, str(caught.exception))

    def test_a_bad_setting_never_falls_back_to_adc(self):
        # Returning None here would authenticate as whoever is logged in on the
        # machine, read the batch as them and bill them, without saying so.
        sdk = fake_sdk()
        sdk.Credentials.from_service_account_info.side_effect = ValueError("no")

        with self.assertRaises(ValueError):
            load(json.dumps(KEY), sdk)

    def test_a_missing_google_auth_says_which_extra_installs_it(self):
        with patch(
            "creature_ocr_server.engines.auth._load_sdk", side_effect=ImportError
        ):
            with self.assertRaises(ValueError) as caught:
                auth.credentials(json.dumps(KEY))

        self.assertIn("[gemini]", str(caught.exception))
        self.assertIn("[documentai]", str(caught.exception))


class LoggingTest(unittest.TestCase):
    """What a run is allowed to say about how it authenticated."""

    def test_the_private_key_never_reaches_the_log(self):
        with self.assertLogs("creature_ocr_server.engines.auth", logging.INFO) as caught:
            load(json.dumps(KEY))

        written = "\n".join(caught.output)
        self.assertNotIn("NOTAREALKEY", written)
        self.assertNotIn("PRIVATE KEY", written)

    def test_it_says_which_service_account_is_being_billed(self):
        with self.assertLogs("creature_ocr_server.engines.auth", logging.INFO) as caught:
            load(json.dumps(KEY))

        self.assertIn(KEY["client_email"], "\n".join(caught.output))

    def test_it_says_so_when_it_falls_back_to_adc(self):
        with self.assertLogs("creature_ocr_server.engines.auth", logging.INFO) as caught:
            auth.credentials("")

        self.assertIn("application default credentials", "\n".join(caught.output))


if __name__ == "__main__":
    unittest.main()
