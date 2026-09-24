"""LM Studio, over its OpenAI-compatible server. One call per page, over HTTP.

The engine that names no model. Every other backend here is chosen by an id -
a Vertex model, a Document AI processor, a NIM deployment - and this one is
chosen by whatever the operator has loaded in LM Studio. So the request carries
no model id, `models()` is empty, and a request that names one is refused the
way it is for a processor. Switching model is something somebody does in LM
Studio, not something a client asks for and not something this server decides.

What that costs is one thing and it is worth stating plainly: the published
cache key cannot name the model. `status()` and `settings` must stay pure - GET
/v1/engines answers them without loading a credential or opening a socket - and
the only way to find out what LM Studio has loaded is to ask LM Studio. So the
key names the base URL, and a client that has already read a page will keep
serving that reading after the model behind the same URL has been changed. What
this engine can do about it is say which model answered, and it logs that per
page beside the request id, because otherwise nothing anywhere would record it.
It is not put into the reading's notes: those are a client's error file, and a
line on every page - including every page that was read perfectly - is noise
there rather than a record. (7.2)

Read like Gemini rather than like nemotron: the prompt, the response schema, the
row parsing and the cell addressing are Gemini's own, imported from that module.
That is deliberate and it is what makes 4.1's comparison a comparison - two
models answering the same question, with the same prompt fingerprint in both
cache keys to prove it was the same question. Nothing flows the other way:
gemini.py does not know this engine exists, so adding a backend is still one
module, one registry entry and one test. Importing it is free, because its SDK
is named inside _load_sdk and nothing here calls that.

There is no vendor SDK and nothing to install: this is one POST of JSON, which
the standard library does. _post below is the single place this package opens a
socket to LM Studio - the seam a test replaces, exactly as the others replace
their SDK - and the engine reports itself usable with a base install. (4.2, 6.5)

LMSTUDIO_BASE_URL is a base URL and /chat/completions is appended to it, which
is the opposite of what NEMOTRON_ENDPOINT does. The reason is that there the
suffix is genuinely unknowable - NVIDIA's hosted deployment answers on the
model's own URL and a NIM container answers on /v1/ocr - while LM Studio serves
one fixed OpenAI-compatible surface that every client in existence is configured
with as a base. A URL that already ends in the path is used as it stands, and a
setting that is only a host and a port gets /v1 as well, because that is a host
somebody wrote down rather than a path they chose.

A local server usually checks no credential, so LMSTUDIO_API_KEY is optional and
the Authorization header is only sent when it is set. Where it is set the same
rule holds as everywhere else: it is never logged and never enters the published
settings string. (6.3)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from .. import config
from ..ocr import (
    OCREngine,
    OCRError,
    Reading,
    Usage,
    checks_fingerprint,
    header_fingerprint,
)
from .gemini import (
    build_prompt,
    build_prompt_v2,
    grid_fingerprint,
    grid_fingerprint_v2,
    prompt_fingerprint,
    prompt_fingerprint_v2,
    response_schema,
    response_schema_v2,
    text_boxes,
    text_boxes_v2,
)

logger = logging.getLogger(__name__)

# How much of a refusal's body is quoted back. The same bound nemotron uses, for
# the same reason: enough to carry the server's own message, short enough that a
# page of HTML from something that is not LM Studio does not fill the log.
ERROR_BODY_CHARS = 500

# What is appended to the base URL, and what a base URL that already ends in it
# is recognised by. See the note at the top for why this engine appends and
# nemotron does not.
CHAT_PATH = "/chat/completions"

# Where LM Studio's OpenAI-compatible surface lives, added to a setting that
# names a host and nothing else. Every other client is configured with the /v1 in
# it, so an operator who leaves it out has written down a host rather than made a
# choice - and LM Studio answers a request that misses its endpoint with a 200
# and a complaint rather than a refusal, which arrives here as a page that came
# back empty and reads like a model that returned nothing.
API_PREFIX = "/v1"

# The name the structured-output schema is sent under. LM Studio requires one
# and does nothing with it.
SCHEMA_NAME = "survey_page"

# What a refusal that complains about the model means and what to do about it.
# Its own constant because it is this engine's characteristic failure: the whole
# point here is to leave the model to LM Studio, and a build that insists on
# being told one refuses every page until an operator names it.
MODEL_HINT = (
    "LM Studio did not accept the model this request named. The field is left "
    f"as {config.LMSTUDIO_ENV_MODEL} says, which is empty by default so that "
    "whatever model is loaded reads the page. A build that insists on being "
    f"told needs {config.LMSTUDIO_ENV_MODEL} set to the id LM Studio lists at "
    "GET /v1/models"
)


def _url(base: str | None) -> str:
    """The whole URL a page is posted to, from the base URL a client configures.

    Empty means the conventional address, which is the one setting in this
    package that has a default: LM Studio serves 127.0.0.1:1234 out of the box
    and an operator who has not changed it has nothing to fill in. A wrong guess
    here cannot bill anybody or read a deployment nobody chose.
    """
    url = str(base or "").strip().rstrip("/") or config.LMSTUDIO_BASE_URL.rstrip("/")
    if url.endswith(CHAT_PATH):
        return url
    if not urllib.parse.urlsplit(url).path:
        # A host and a port and nothing else. Anything the operator did write -
        # /v1, LM Studio's own /api/v0, a path a reverse proxy is mounted on - is
        # left exactly as it stands.
        url += API_PREFIX
    return url + CHAT_PATH


def _timeout(setting: str | float | None) -> float:
    """How long one attempt may take, or the default of two minutes."""
    text = str(setting or "").strip()
    if not text:
        return config.LMSTUDIO_TIMEOUT_SECONDS
    try:
        seconds = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{config.LMSTUDIO_ENV_TIMEOUT} is not a number: {text!r}"
        ) from exc
    if seconds <= 0:
        raise ValueError(
            f"{config.LMSTUDIO_ENV_TIMEOUT} must be above zero, not {seconds}"
        )
    return seconds


def _post(url: str, payload: dict, key: str, timeout: float) -> dict:
    """POST one request and return the parsed answer.

    The one place this package opens a socket to LM Studio, so the rest of the
    module is testable without a network and a test can substitute the whole
    transport the way the other engines substitute an SDK. (4.2)

    Every failure becomes OCRError, which is what 6.4's retry is written
    against: a page that could not be asked about, not a page that is blank.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        # Only when there is one. A local server checks nothing by default, and
        # sending an empty bearer token is a way to be refused by the one that
        # does check.
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _refusal(exc)
        if exc.code in (400, 404) and "model" in detail.lower():
            raise OCRError(
                f"lmstudio refused a page: HTTP {exc.code} {detail}. {MODEL_HINT}"
            ) from exc
        raise OCRError(f"lmstudio refused a page: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # The failure this engine actually produces in the field: the server is
        # not running, or it is running with a model that cannot see an image.
        # A page is a large body, so a server that rejects the request before
        # reading it closes the connection mid-upload and arrives here rather
        # than as the status code it really was.
        raise OCRError(
            f"lmstudio could not be reached at {url}: {exc}. Check "
            f"{config.LMSTUDIO_ENV_BASE_URL}, and that LM Studio's local server "
            "is running with a model that can read an image"
        ) from exc
    except ValueError as exc:
        raise OCRError(
            f"lmstudio answered with something that is not JSON: {exc}"
        ) from exc


def _refusal(exc: urllib.error.HTTPError) -> str:
    """The server's own words, bounded, or the status line if it had none."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a body that will not read is not the error
        return exc.reason or ""
    return body.strip()[:ERROR_BODY_CHARS]


def answered_by(answer: dict) -> str:
    """Which model LM Studio actually used, as it reports it.

    The request names no model and the published cache key cannot either, so the
    log line this feeds is the only record of which model read a page. Read
    before the answer is known to be shaped like an answer, so it is guarded
    rather than assumed.
    """
    if not isinstance(answer, dict):
        return ""
    return str(answer.get("model") or "").strip()


def tokens(answer: dict) -> Usage:
    """What the call was billed for, or nothing if the server did not say.

    Nobody is billed for a local call, and the numbers are still worth carrying:
    they are how 7.1 compares what a page costs to ask of one model against
    another. Read defensively for the same reason Gemini's are - usage is
    reporting, not the answer, and a page that was read must not be thrown away
    because the meter was missing.

    Reasoning tokens are kept separately as well as counted as output, which is
    the convention completion_tokens already follows, so unlike Gemini's they
    are not added on again.
    """
    meta = answer.get("usage") if isinstance(answer, dict) else None
    if not isinstance(meta, dict):
        return Usage()
    details = meta.get("completion_tokens_details")
    thoughts = 0
    if isinstance(details, dict):
        thoughts = _count(details.get("reasoning_tokens"))
    return Usage(
        prompt_tokens=_count(meta.get("prompt_tokens")),
        output_tokens=_count(meta.get("completion_tokens")),
        thought_tokens=thoughts,
    )


def _count(value: object) -> int:
    """One reported number, or zero for anything that is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def response_format(sheet: str = config.SHEET_V1) -> dict:
    """How the page's JSON shape is asked for over the OpenAI-compatible API.

    The schema is Gemini's, so the two engines are asked for the same thing;
    only the envelope differs. propertyOrdering is dropped because it is
    Vertex's own extension: LM Studio turns a schema into a grammar and an
    unknown keyword is at best ignored and at worst refused.
    """
    schema = response_schema() if sheet == config.SHEET_V1 else response_schema_v2()
    return {
        "type": "json_schema",
        "json_schema": {"name": SCHEMA_NAME, "schema": _plain(schema)},
    }


def _plain(schema: object) -> object:
    """The same schema with Vertex's own keyword taken out, at every depth."""
    if isinstance(schema, dict):
        return {
            key: _plain(value)
            for key, value in schema.items()
            if key != "propertyOrdering"
        }
    if isinstance(schema, list):
        return [_plain(item) for item in schema]
    return schema


def _finish_reason(choice: dict) -> str:
    """Why a choice carried no text: a stop, a token budget, a refusal."""
    return str(choice.get("finish_reason") or "") or "no reason given"


def content(answer: dict) -> str:
    """The one choice's text, or raise because the page did not come back.

    An answer that is not shaped like an answer costs the whole page, so it
    raises and 6.4 retries it. An empty answer is reported with the finish
    reason attached, because the ways it happens - the model ran out of output
    tokens, or it cannot see an image at all - are settings problems and look
    like nothing otherwise.
    """
    if not isinstance(answer, dict):
        raise OCRError(
            f"lmstudio answered with {type(answer).__name__}, not an object"
        )
    choices = answer.get("choices")
    if not isinstance(choices, list) or not choices:
        detail = answer.get("error")
        raise OCRError(
            "lmstudio answered without a completion for the page"
            + (f": {detail}" if detail else "")
            + f". Check {config.LMSTUDIO_ENV_BASE_URL} reaches LM Studio's "
            "OpenAI-compatible server: a request that misses its endpoint is "
            "answered with 200 and a complaint rather than refused"
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise OCRError("lmstudio answered with a malformed choice for the page")
    message = first.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise OCRError(f"lmstudio returned no text ({_finish_reason(first)})")
    return text


def unfence(text: str) -> str:
    """The JSON inside a Markdown fence, or the text exactly as it came.

    A local model asked for JSON returns it wrapped in ```json often enough that
    refusing it would cost whole pages that were read perfectly well. Gemini
    needs nothing of the kind - Vertex honours response_mime_type - so this step
    lives here rather than in the parsing the two engines share.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline != -1:
        # Drop the language tag on the opening fence, if it had one.
        body = body[newline + 1 :]
    end = body.rfind("```")
    return (body[:end] if end != -1 else body).strip()


def _page(answer: dict) -> object:
    """Whatever JSON the model answered with, or say why there is none."""
    text = content(answer)
    try:
        return json.loads(unfence(text))
    except ValueError as exc:
        raise OCRError(
            f"lmstudio returned something that is not JSON: {exc}"
        ) from exc


def parse_rows(answer: dict) -> list:
    """Read the rows out of an answer, or say why there are none.

    Everything here raises: it is a whole page that did not arrive, so 6.4's
    retry is the right answer, unlike a single unplaceable row - which
    gemini.text_boxes drops into the reading's notes.
    """
    found = _page(answer)
    if isinstance(found, dict):
        # The schema asks for a bare array; accept a wrapped one anyway rather
        # than throw away a page that is otherwise perfectly good.
        found = found.get("rows")
    if not isinstance(found, list):
        raise OCRError("lmstudio returned no list of rows")
    return found


def parse_page_v2(answer: dict) -> dict:
    """Read a whole v2 page - the strip and the rows - out of an answer.

    A bare array is accepted as a page with an empty strip, so a model that
    ignored the object wrapper still gives up its rows instead of costing the
    whole page.
    """
    found = _page(answer)
    if isinstance(found, list):
        return {"rows": found}
    if not isinstance(found, dict):
        raise OCRError("lmstudio returned no page object")
    return found


class LMStudioEngine(OCREngine):
    """One structured-extraction call per page, against a local LM Studio."""

    name = "lmstudio"
    # Nothing has to be set: the base URL has a conventional default and the
    # key is optional, so GET /v1/engines reports this engine usable on a server
    # that was never configured. What it cannot report is whether LM Studio is
    # actually running - that would need a socket, and status() opens none.
    requires = ()
    # No SDK and no extra. There is nothing to install for this engine at all,
    # so naming an extra here would send an operator to `pip install -e
    # ".[lmstudio]"`, which does not exist. (4.2, 6.5)
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

        A classmethod, so GET /v1/engines can publish it without opening a
        socket - which for this engine is the whole difficulty, and the reason
        the string names a URL where the other prompt engine names a model. See
        the note at the top: changing the model behind this URL does not change
        this string, and a client's cached page will outlive the change.

        The grid and the prompt are in it for the same reason they are in
        Gemini's: a box from here is a cell address rather than a measurement,
        and what comes back is the answer to a question. The checks, because a
        client of this server caches finished rows. (3.2, 4.2, 5.2-4)
        """
        # None means "ask the environment", which is what a published key wants;
        # a URL passed in is taken as given, so the string describes the engine
        # that actually read the page.
        if base_url is None:
            base_url = os.environ.get(config.LMSTUDIO_ENV_BASE_URL, "")
        url = _url(base_url)
        if sheet == config.SHEET_V1:
            return (
                f"lmstudio {url} "
                f"temperature={config.LMSTUDIO_TEMPERATURE} "
                f"top_p={config.LMSTUDIO_TOP_P} "
                f"grid={grid_fingerprint()} prompt={prompt_fingerprint()} "
                f"checks={checks_fingerprint()}"
            )
        return (
            f"lmstudio {url} sheet={sheet} "
            f"temperature={config.LMSTUDIO_TEMPERATURE} "
            f"top_p={config.LMSTUDIO_TOP_P} "
            f"grid={grid_fingerprint_v2()} prompt={prompt_fingerprint_v2()} "
            f"checks={checks_fingerprint()}+{header_fingerprint()}"
        )

    @classmethod
    def cache_name_for(cls, model: str = "", sheet: str = config.SHEET_V1) -> str:
        """The directory a client keeps these readings under. (3.2, 4.1)

        The engine's own name: there is no model id to name it by, and the URL
        would make a directory name out of a port number. The sheet joins it for
        anything but v1, because two readings of one page are two readings.
        """
        return cls.name if sheet == config.SHEET_V1 else f"{cls.name}-{sheet}"

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
            base_url = os.environ.get(config.LMSTUDIO_ENV_BASE_URL, "")
        if timeout is None:
            timeout = os.environ.get(config.LMSTUDIO_ENV_TIMEOUT, "")
        self._url = _url(base_url)
        self._timeout = _timeout(timeout)
        # Empty on purpose, and empty by default: the model loaded in LM Studio
        # is the reader. This is the escape hatch for a build that refuses a
        # request which does not name one, not a way for this server to choose.
        self._model = os.environ.get(config.LMSTUDIO_ENV_MODEL, "").strip()
        # Read last of the settings and kept out of everything below: it is not
        # in settings, which GET /v1/engines publishes, and it is never logged.
        # Read straight from the environment rather than through auth.api_key,
        # which demands the setting - and a local server usually wants none. (6.3)
        self._key = os.environ.get(config.LMSTUDIO_ENV_API_KEY, "").strip()
        # Built once and shared by every request this engine serves, which is
        # safe because none of it changes between calls.
        v1 = sheet == config.SHEET_V1
        self._prompt = build_prompt() if v1 else build_prompt_v2()
        self._format = response_format(sheet)
        # Worked out by the same functions GET /v1/engines answers with, so what
        # a client was told to key its cache on is what actually read the page.
        self.settings = self.settings_for("", sheet, self._url)
        self.cache_name = self.cache_name_for("", sheet)

    def recognize(self, image: bytes, mime_type: str = "image/png") -> Reading:
        payload = {
            # Empty unless an operator has named one. See the note above.
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:{};base64,{}".format(
                                    mime_type,
                                    base64.b64encode(image).decode("ascii"),
                                )
                            },
                        }
                    ],
                },
            ],
            # 5.2-4's determinism, pinned here as it is for Gemini and never a
            # request parameter. No output token limit: a model that thinks
            # spends that budget on thinking and then returns nothing at all.
            "temperature": config.LMSTUDIO_TEMPERATURE,
            "top_p": config.LMSTUDIO_TOP_P,
            "stream": False,
            "response_format": self._format,
        }
        answer = _post(self._url, payload, self._key, self._timeout)
        # Both read before the answer is parsed, and the usage attached to the
        # failure if it cannot be: a page that came back unusable still cost the
        # time, and which model produced it is the first thing anyone will ask.
        spent = tokens(answer)
        answered = answered_by(answer)
        if answered:
            # The only record of which model read this page. Said here rather
            # than handed back in the reading: see the note at the top.
            logger.info("the model LM Studio had loaded is %s", answered)
        try:
            if self._sheet == config.SHEET_V1:
                boxes, found = text_boxes(parse_rows(answer))
            else:
                boxes, found = text_boxes_v2(parse_page_v2(answer))
        except OCRError as exc:
            exc.usage = spent
            raise
        return Reading(boxes, spent, found)
