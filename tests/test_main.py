"""What `python -m creature_ocr_server` binds, and how it reads its settings.

One rule, and it is the same one engines.resolve and auth.credentials state: an
empty value counts as unset. os.environ.get answers with its default only when
the name is absent, and a .env copied from the template sets all three of these
to nothing at all - so without the fallback, HOST= binds every interface rather
than the loopback address, and PORT= and LOG_LEVEL= stop the server starting.

uvicorn.run is replaced rather than called: nothing here opens a socket. (6.5)
"""

import os
import unittest
from unittest.mock import patch

from creature_ocr_server.__main__ import main

SETTINGS = ("HOST", "PORT", "LOG_LEVEL")


class RunnerTest(unittest.TestCase):
    def run_with(self, environment):
        cleared = {name: "" for name in SETTINGS}
        cleared.update(environment)
        with patch.dict(os.environ, cleared, clear=False):
            for name in SETTINGS:
                if name not in environment:
                    del os.environ[name]
            with patch("uvicorn.run") as run:
                main()
        return run.call_args.kwargs

    def test_nothing_set_binds_the_loopback_address(self):
        found = self.run_with({})

        self.assertEqual(found["host"], "127.0.0.1")
        self.assertEqual(found["port"], 8000)
        self.assertEqual(found["log_level"], "info")

    def test_an_empty_host_is_not_every_interface(self):
        # The failure this test exists for. uvicorn binds "" to 0.0.0.0, so a
        # bare HOST= line would publish a server holding a cloud credential on
        # every interface, which is the opposite of what it documents.
        self.assertEqual(self.run_with({"HOST": ""})["host"], "127.0.0.1")

    def test_an_empty_port_still_starts(self):
        # int("") raises, so this one does not misbehave: it refuses to start.
        self.assertEqual(self.run_with({"PORT": ""})["port"], 8000)

    def test_an_empty_log_level_still_starts(self):
        # uvicorn has no "" level, so this one refuses to start as well.
        self.assertEqual(self.run_with({"LOG_LEVEL": ""})["log_level"], "info")

    def test_a_value_that_was_meant_is_used(self):
        found = self.run_with({"HOST": "0.0.0.0", "PORT": "7488", "LOG_LEVEL": "DEBUG"})

        self.assertEqual(found["host"], "0.0.0.0")
        self.assertEqual(found["port"], 7488)
        self.assertEqual(found["log_level"], "debug")

    def test_surrounding_space_is_not_a_value(self):
        self.assertEqual(self.run_with({"PORT": "  7488  "})["port"], 7488)


if __name__ == "__main__":
    unittest.main()
