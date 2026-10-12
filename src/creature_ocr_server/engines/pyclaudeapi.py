"""pyClaudeApi, a local server that reads a page with the Claude Code CLI.

pyClaudeApi answers on LM Studio's OpenAI-compatible surface, so on the wire
this engine is lmstudio: the same request, Gemini's prompt and schema, the same
parsing of the answer. Everything that sends and reads is inherited from
LMStudioEngine. What differs is only how it is configured:

- Its own settings, so lmstudio and pyclaudeapi can be configured side by side
  and a client keeps their readings in different cache directories.
- No default URL. LM Studio has a conventional port; pyClaudeApi does not, and
  guessing would point at this server's own 8000.
- The key is demanded, because pyClaudeApi refuses a request without one, and
  it is read through auth.api_key like nemotron's: never logged and never in
  the published settings. (6.3)
- No model. Claude Code chooses the model, so the request names none and the
  cache key cannot name one either - the same limit lmstudio states.

There is no SDK and nothing to install: it is one HTTP request. (4.2, 6.5)
"""

from __future__ import annotations

import math
import os
import urllib.parse

from .. import config
from ..ocr import checks_fingerprint, header_fingerprint
from . import auth
from .gemini import (
    build_prompt,
    build_prompt_v2,
    grid_fingerprint,
    grid_fingerprint_v2,
    prompt_fingerprint,
    prompt_fingerprint_v2,
)
from .lmstudio import LMStudioEngine, _url, response_format


def _base_url(setting: str | None) -> str:
    """The base URL as set, or "" if it is not an http or https URL.

    Checked before lmstudio's _url sees it. _url reads a value of only slashes
    as LM Studio's address and would send the page and the key there; a URL
    urllib cannot split would raise inside settings_for and take GET
    /v1/engines down for every engine; and one with no scheme would report
    ready and then fail every page.
    """
    text = str(setting or "").strip()
    try:
        parts = urllib.parse.urlsplit(text)
    except ValueError:
        return ""
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return ""
    return text


def _timeout(setting: str | float | None) -> float:
    """How long one attempt may take, or the default."""
    text = str(setting or "").strip()
    if not text:
        return config.PYCLAUDEAPI_TIMEOUT_SECONDS
    try:
        seconds = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{config.PYCLAUDEAPI_ENV_TIMEOUT} is not a number: {text!r}"
        ) from exc
    if not math.isfinite(seconds):
        # float() takes "inf" and "nan", and a socket takes neither.
        raise ValueError(
            f"{config.PYCLAUDEAPI_ENV_TIMEOUT} is not a finite number: {text!r}"
        )
    if seconds <= 0:
        raise ValueError(
            f"{config.PYCLAUDEAPI_ENV_TIMEOUT} must be above zero, not {seconds}"
        )
    return seconds


class PyClaudeAPIEngine(LMStudioEngine):
    """One Claude Code CLI read per page, through a pyClaudeApi server."""

    name = "pyclaudeapi"
    requires = (config.PYCLAUDEAPI_ENV_BASE_URL, config.PYCLAUDEAPI_ENV_API_KEY)
    sdk = ""
    extra = ""

    @classmethod
    def settings_for(
        cls,
        model: str = "",
        sheet: str = config.SHEET_V1,
        base_url: str | None = None,
    ) -> str:
        """What a client's response cache keys this engine's readings on.

        lmstudio's key with this engine's name and URL. A URL that is unset or
        unusable stays empty rather than falling back to LM Studio's address,
        so a server that is not configured does not publish a key for a reader
        it never had. Temperature and top_p are left out: they are sent, but
        the CLI does not take them, so they say nothing about what read the
        page.
        """
        if base_url is None:
            base_url = os.environ.get(config.PYCLAUDEAPI_ENV_BASE_URL, "")
        base = _base_url(base_url)
        url = _url(base) if base else ""
        if sheet == config.SHEET_V1:
            return (
                f"pyclaudeapi {url} "
                f"grid={grid_fingerprint()} prompt={prompt_fingerprint()} "
                f"checks={checks_fingerprint()}"
            )
        return (
            f"pyclaudeapi {url} sheet={sheet} "
            f"grid={grid_fingerprint_v2()} prompt={prompt_fingerprint_v2()} "
            f"checks={checks_fingerprint()}+{header_fingerprint()}"
        )

    def __init__(
        self,
        base_url: str | None = None,
        sheet: str = config.SHEET_V1,
        timeout: str | float | None = None,
    ) -> None:
        if sheet not in config.SHEETS:
            raise ValueError(
                f"unknown sheet {sheet!r}: expected one of "
                f"{', '.join(config.SHEETS)}"
            )
        self._sheet = sheet
        if base_url is None:
            base_url = os.environ.get(config.PYCLAUDEAPI_ENV_BASE_URL, "")
        if timeout is None:
            timeout = os.environ.get(config.PYCLAUDEAPI_ENV_TIMEOUT, "")
        # Demanded rather than defaulted: see the note at the top.
        if not str(base_url).strip():
            raise ValueError(
                f"{config.PYCLAUDEAPI_ENV_BASE_URL} is not set: see .env.example"
            )
        base = _base_url(base_url)
        if not base:
            # The value is not repeated: the 503 carries this sentence to the
            # client, and a value in the wrong setting can be anything.
            raise ValueError(
                f"{config.PYCLAUDEAPI_ENV_BASE_URL} is not an http or https URL: "
                "see .env.example"
            )
        self._url = _url(base)
        self._timeout = _timeout(timeout)
        # Claude Code chooses the model; the request names none.
        self._model = ""
        # Read last and kept out of everything below. (6.3)
        self._key = auth.api_key(config.PYCLAUDEAPI_ENV_API_KEY)
        if not (self._key.isascii() and self._key.isprintable()):
            # Refused by name alone. http.client refuses a header like this
            # with an error that quotes it, and lmstudio's _post would put that
            # into the log line of every retry. (6.3)
            raise ValueError(
                f"{config.PYCLAUDEAPI_ENV_API_KEY} holds a character a header "
                "cannot carry: see .env.example"
            )
        v1 = sheet == config.SHEET_V1
        self._prompt = build_prompt() if v1 else build_prompt_v2()
        self._format = response_format(sheet)
        self.settings = self.settings_for("", sheet, self._url)
        self.cache_name = self.cache_name_for("", sheet)
