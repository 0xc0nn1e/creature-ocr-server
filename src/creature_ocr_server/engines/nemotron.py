"""NVIDIA NIM, using the nemotron-ocr-v2 model. One call per page, over HTTP.

Like Document AI and unlike Gemini, this model takes no instructions and reports
a bounding box for everything it reads, already normalised to the image it was
given. That is the frame the cell grid uses, so one page-wide call fills a value
per cell without paying per cell, and a reading survives a change to the grid:
what it says is where the ink was, not which cell it belonged to. (3.2, 4.2)

There is no vendor SDK, so unlike the other two engines there is nothing to
import for an ordinary page. The request is one POST of JSON, which the standard
library does, and _post below is the single place this package opens a socket to
NVIDIA - the seam a test replaces, exactly as the others replace their SDK.

The one thing that does need a library is shrink, and only when
NEMOTRON_MAX_BYTES asks for it, so pymupdf is named inside _load_imaging rather
than at the top of this module. That is what keeps the package importable and
the whole test suite passing with the nemotron extra not installed, which is the
same rule every other engine here follows. (4.2, 6.5)

NEMOTRON_ENDPOINT is the whole URL the page is posted to, and nothing is
appended to it. NVIDIA's hosted deployment answers on the model's own URL, while
a NIM run as a container answers on /v1/ocr under its host, so there is no
common suffix to add: guessing one turns a working setting into a 404, and a 404
on a request whose body is half a megabyte arrives as a connection dropped
mid-upload rather than as a status code.

Two things about this model are worth knowing before its numbers are read. Its
model card claims printed text and does not claim handwriting, and every sheet
here is handwritten. And v2_multilingual, the build that reads Japanese, is
documented as line-level, so asking for merge_level=word may still come back
with a box per line; a box is placed by its centre, so one line box spanning the
row would put that row's answers into a single column. Both are questions for
7.2's benchmark rather than for this module. (4.1, 7.2)

The API key is neither read nor written here: auth.api_key is the one place that
reads a credential, and it never reaches OCREngine.settings - which
GET /v1/engines publishes - or a log line. (6.3)
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request

from .. import config
from ..ocr import OCREngine, OCRError, Reading, TextBox, Usage, checks_fingerprint
from . import auth

logger = logging.getLogger(__name__)

# How much of a refusal's body is quoted back. Enough to carry NVIDIA's own
# message, short enough that a page of HTML from a proxy does not fill the log.
ERROR_BODY_CHARS = 500

# What a 413 means here and what to do about it. Its own constant because it is
# the failure this engine is most likely to hit: a 300 DPI page is around half a
# megabyte once base64 has grown it by a third, and NVIDIA's hosted endpoints
# document a much smaller cap on an inline image. The client renders the page,
# so the first way out belongs to whoever runs it. (5.2-3, 6.4)
TOO_LARGE_HINT = (
    "the endpoint refused the page as too large. Either have the client render "
    f"it at a lower DPI, or set {config.NEMOTRON_ENV_MAX_BYTES} to have the "
    f"page re-encoded smaller before it is sent, or point "
    f"{config.NEMOTRON_ENV_ENDPOINT} at a NIM you run yourself"
)


def _load_imaging():
    """Import the imaging library shrink re-encodes a page with.

    Named here only, and only when called, so this engine costs nothing to
    import and a server that never sets NEMOTRON_MAX_BYTES never needs the
    extra at all. The desktop pipeline imports pymupdf at module scope because
    it renders the PDF itself; this server is handed a cropped page and only
    touches an image when an operator has asked for one to be made smaller.
    (4.2, 6.5)
    """
    import pymupdf

    return pymupdf


def _post(url: str, payload: dict, key: str, timeout: float) -> dict:
    """POST one request and return the parsed answer.

    The one place this package opens a socket to NVIDIA, so the rest of the
    module is testable without a network and a test can substitute the whole
    transport the way the other engines substitute an SDK. (4.2)

    Every failure becomes OCRError, which is what 6.4's retry is written
    against: a page that could not be asked about, not a page that is blank.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _refusal(exc)
        if exc.code == 413:
            raise OCRError(f"nemotron refused a page: {TOO_LARGE_HINT}") from exc
        raise OCRError(f"nemotron refused a page: HTTP {exc.code} {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # A page is a large body, and a server that rejects the request before
        # reading it closes the connection mid-upload: that arrives here as a
        # broken connection rather than as the status code it really was. The
        # setting most likely to be wrong is named, because a wrong URL is the
        # one cause this message cannot otherwise distinguish.
        raise OCRError(
            f"nemotron could not be reached at {url}: {exc}. Check "
            f"{config.NEMOTRON_ENV_ENDPOINT} is the whole URL the page is "
            "posted to"
        ) from exc
    except ValueError as exc:
        raise OCRError(
            f"nemotron answered with something that is not JSON: {exc}"
        ) from exc


def _refusal(exc: urllib.error.HTTPError) -> str:
    """The server's own words, bounded, or the status line if it had none."""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - a body that will not read is not the error
        return exc.reason or ""
    return body.strip()[:ERROR_BODY_CHARS]


def _detections(answer: dict) -> list:
    """The one image's detections, or raise because the page did not come back.

    An answer that is not shaped like an answer costs the whole page, so it
    raises and 6.4 retries it. A page that genuinely held no text is a valid
    empty list and is not an error.
    """
    if not isinstance(answer, dict):
        raise OCRError(f"nemotron answered with {type(answer).__name__}, not an object")
    data = answer.get("data")
    if not isinstance(data, list) or not data:
        raise OCRError("nemotron answered with no data for the page")
    first = data[0]
    if not isinstance(first, dict):
        raise OCRError("nemotron answered with a malformed entry for the page")
    found = first.get("text_detections")
    if not isinstance(found, list):
        raise OCRError("nemotron answered without text detections for the page")
    return found


def text_boxes(answer: dict) -> tuple[list[TextBox], list[str]]:
    """Turn one answer into boxes, in the order the engine gave them.

    The points are a quadrilateral in fractions of the page, which is the frame
    TextBox already uses, so they are reduced to the box that contains them and
    nothing is scaled. A detection with no position is dropped rather than
    guessed at: 5.2-2 would rather lose a value than invent where it sat.

    The confidence becomes unsure rather than a reason to drop the value. This
    is the only engine that reports one, and 5.2-2 reports rather than rewrites:
    the value stays, marked, for a person to look at.

    What was dropped comes back as notes rather than going to a debug log, the
    way Gemini's do: the operator who has to know is on another machine. (7.2)
    """
    boxes: list[TextBox] = []
    notes: list[str] = []
    for detection in _detections(answer):
        if not isinstance(detection, dict):
            notes.append("dropped a detection that is not an object")
            continue
        prediction = detection.get("text_prediction") or {}
        text = str(prediction.get("text", "")).strip()
        if not text:
            continue
        points = (detection.get("bounding_box") or {}).get("points")
        if not points:
            notes.append(f"dropped {text!r}: the engine gave it no position")
            continue
        try:
            xs = [float(point["x"]) for point in points]
            ys = [float(point["y"]) for point in points]
        except (KeyError, TypeError, ValueError):
            notes.append(f"dropped {text!r}: its position could not be read")
            continue
        confidence = prediction.get("confidence")
        unsure = (
            isinstance(confidence, (int, float))
            and confidence < config.NEMOTRON_UNSURE_BELOW
        )
        boxes.append(TextBox(text, min(xs), min(ys), max(xs), max(ys), unsure=unsure))
    return boxes, notes


def shrink(image: bytes, limit: int) -> tuple[bytes, str]:
    """Re-encode a page until its base64 fits in limit bytes.

    Only reached when NEMOTRON_MAX_BYTES is set, because 5.2-3 forbids throwing
    image quality away: greyscale is what makes faint pencil readable, and JPEG
    spends exactly that. An operator who sets it has decided that a re-encoded
    reading beats no reading, and the setting is in the published cache key so
    the choice is recorded beside every page it changed.

    Raises rather than sending the smallest attempt anyway. Quietly sending a
    worse page than was asked for would move 5.1's accuracy with nothing in the
    run to say why.
    """
    smallest = _encoded_size(image)
    if smallest <= limit:
        # Nothing to do. recognize only calls this when the page is over the
        # limit, but a caller that has not checked should not be handed a JPEG
        # that is larger than the PNG it started with.
        return image, "image/png"

    try:
        pymupdf = _load_imaging()
    except ImportError as exc:
        raise OCRError(
            f"{config.NEMOTRON_ENV_MAX_BYTES} asks for the page to be "
            're-encoded, but pymupdf is not installed: pip install -e ".[nemotron]"'
        ) from exc

    # The image straight into a pixmap, not through a document. Opening it as a
    # page and rendering that page resamples it to whatever the PNG claims for
    # its own DPI - on a 3367x1442 crop, 2526x1082 before any quality is even
    # considered - and 5.2-3 is the reason this function only spends what the
    # setting says it spends.
    pixmap = pymupdf.Pixmap(image)
    if pixmap.alpha:
        # JPEG has no alpha channel and pymupdf refuses the encode rather than
        # dropping one, so an RGBA page - which plenty of renderers write
        # whether the paper needed it or not - would arrive as a bare
        # ValueError, be retried three times over, and end as a 502 that names
        # nothing. Dropping the channel is the answer, but only while nothing
        # is actually transparent. (6.4)
        #
        # MuPDF premultiplies on the way in, so a transparent pixel reads
        # (0, 0, 0) whatever colour it was on the client's screen, and no
        # set_alpha option gets that back. Dropping the channel then would send
        # a black page, and a black page comes back as an empty reading - which
        # is indistinguishable from a sheet nobody wrote on. So a page with any
        # transparency in it is refused and said out loud: 5.2-2 would rather
        # lose the page than invent what was on it. (5.2-2)
        #
        # One 19 MB slice on a full-size crop, about four milliseconds, against
        # the four JPEG encodes below.
        opaque = b"\xff" * (pixmap.width * pixmap.height)
        if pixmap.samples[pixmap.n - 1 :: pixmap.n] != opaque:
            raise OCRError(
                "the page is partly transparent and JPEG cannot carry that: "
                "what was behind the ink is already lost by the time it is "
                "decoded. Have the client flatten the page onto its paper "
                f"colour, or clear {config.NEMOTRON_ENV_MAX_BYTES} so the page "
                "is sent exactly as it arrived"
            )
        pixmap = pymupdf.Pixmap(pixmap, 0)
    for quality in config.NEMOTRON_JPEG_QUALITIES:
        candidate = pixmap.tobytes("jpeg", jpg_quality=quality)
        size = _encoded_size(candidate)
        smallest = min(smallest, size)
        if size <= limit:
            logger.info(
                "re-encoded the page at jpeg quality %d to fit %d bytes",
                quality,
                limit,
            )
            return candidate, "image/jpeg"
    raise OCRError(
        f"the page will not fit {config.NEMOTRON_ENV_MAX_BYTES}={limit}: the "
        f"smallest encoding tried is {smallest} base64 bytes. Have the client "
        "render the page at a lower DPI, or raise the limit"
    )


def _encoded_size(image: bytes) -> int:
    """How big the image is once base64 has grown it, which is what is sent."""
    return (len(image) + 2) // 3 * 4


class NemotronEngine(OCREngine):
    """One nemotron-ocr-v2 call per page."""

    name = "nemotron"
    # What must be set before this engine can be built. There is no SDK to
    # check for: an ordinary page needs nothing installed, and the one library
    # shrink wants is only reached when NEMOTRON_MAX_BYTES asks for it, so a
    # server without the extra is still a usable nemotron server and
    # GET /v1/engines should say so. (4.2, 6.5)
    requires = (config.NEMOTRON_ENV_ENDPOINT, config.NEMOTRON_ENV_API_KEY)
    sdk = ""
    extra = "nemotron"

    @classmethod
    def settings_for(
        cls,
        model: str = "",
        variant: str | None = None,
        endpoint: str | None = None,
        max_bytes: int | None = None,
    ) -> str:
        """What a client's response cache keys this engine's readings on.

        A classmethod, so GET /v1/engines can publish it without reading a
        credential or opening a socket - which for this engine also means
        without the API key, and the key would never be allowed into it
        anyway. (3.2, 4.2, 6.3)

        The endpoint rides along because two deployments of the same variant can
        read differently; the merge level and the unsure threshold because both
        change what comes back; the size limit because a re-encoded page is a
        different page. The checks, because a client of this server caches
        finished rows rather than boxes.

        No grid and no prompt fingerprint: these boxes are measurements, so
        re-reading them under a new grid is free. (4.2)
        """
        # None means "ask the environment", which is what the endpoint wants.
        # An endpoint or a limit is then taken as given even when it is empty or
        # zero, so the string describes the engine that actually read the page
        # rather than a setting it was built without. The variant is the one
        # exception: it also names the cache directory, and an empty one would
        # publish a key no engine can produce, so it falls back either way.
        variant = variant or cls.default_variant()
        if endpoint is None:
            endpoint = os.environ.get(config.NEMOTRON_ENV_ENDPOINT, "")
        endpoint = _url(endpoint)
        if max_bytes is None:
            max_bytes = _max_bytes(os.environ.get(config.NEMOTRON_ENV_MAX_BYTES, ""))
        settings = (
            f"nemotron-ocr-v2 {variant} "
            f"merge={config.NEMOTRON_MERGE_LEVEL} "
            f"unsure_below={config.NEMOTRON_UNSURE_BELOW} "
            f"endpoint={endpoint} "
            f"checks={checks_fingerprint()}"
        )
        if max_bytes:
            settings += f" max_bytes={max_bytes}"
        return settings

    @classmethod
    def default_variant(cls) -> str:
        """Which build the endpoint is understood to point at.

        The request carries no variant field - the deployment is the variant -
        so this steers nothing at all. It names the cache directory and goes
        into the settings string, which is the whole of its job. (3.2, 4.1)
        """
        return (
            os.environ.get(config.NEMOTRON_ENV_VARIANT, "").strip()
            or config.NEMOTRON_VARIANT
        )

    @classmethod
    def cache_name_for(cls, model: str = "", variant: str = "") -> str:
        """The directory a client keeps this variant's readings under.

        Pointing the server at the English build to compare the two then keeps
        both sets of readings instead of overwriting one. (3.2, 4.1)
        """
        return f"{cls.name}-{variant or cls.default_variant()}"

    def __init__(
        self,
        endpoint: str | None = None,
        variant: str | None = None,
        max_bytes: str | int | None = None,
    ) -> None:
        endpoint = endpoint or os.environ.get(config.NEMOTRON_ENV_ENDPOINT, "")
        variant = variant or self.default_variant()
        if max_bytes is None:
            max_bytes = os.environ.get(config.NEMOTRON_ENV_MAX_BYTES, "")
        # Checked before anything is sent, so a misconfigured server says so at
        # the first request instead of collecting an authentication failure per
        # page. The endpoint is demanded rather than defaulted: NVIDIA serves
        # this model from a per-model URL and the same NIM can be run locally,
        # so a guess would either fail every call or read a deployment nobody
        # chose.
        if not str(endpoint).strip():
            raise ValueError(
                f"{config.NEMOTRON_ENV_ENDPOINT} is not set: see .env.example"
            )
        self._url = _url(endpoint)
        self._limit = _max_bytes(max_bytes)
        # The credential is fetched last of the settings and kept out of
        # everything below: it is not in settings, which GET /v1/engines
        # publishes and a client writes to disk beside every page, and it is
        # never logged. (6.3)
        self._key = auth.api_key(config.NEMOTRON_ENV_API_KEY)
        # Worked out by the same functions GET /v1/engines answers with, so what
        # a client was told to key its cache on is what actually read the page.
        self.settings = self.settings_for("", variant, self._url, self._limit)
        self.cache_name = self.cache_name_for("", variant)

    def recognize(self, image: bytes, mime_type: str = "image/png") -> Reading:
        notes: list[str] = []
        if self._limit and _encoded_size(image) > self._limit:
            image, mime_type = shrink(image, self._limit)
            notes.append(f"the page was re-encoded as {mime_type} to fit the limit")
        payload = {
            "input": [
                {
                    "type": "image_url",
                    "url": "data:{};base64,{}".format(
                        mime_type, base64.b64encode(image).decode("ascii")
                    ),
                }
            ],
            "merge_levels": [config.NEMOTRON_MERGE_LEVEL],
        }
        answer = _post(self._url, payload, self._key, config.NEMOTRON_TIMEOUT_SECONDS)
        boxes, found = text_boxes(answer)
        # After text_boxes, which is what checks the answer is shaped like one:
        # reading a field off it first would turn a malformed answer into an
        # AttributeError and lose the message that says what was wrong with it.
        #
        # This API bills per request rather than per token, so the usage is
        # empty and recognize_with_retry fills in the call and the time. The
        # size the answer reports is logged rather than counted: it is a fact
        # about one page, not a running total, and Usage is shared with two
        # engines that have no such number. (6.1, 7.1)
        logger.debug("nemotron reported %s", answer.get("usage") or {})
        return Reading(boxes, Usage(), notes + found)


def _url(endpoint: str) -> str:
    """The endpoint as configured. Only a trailing slash is taken off.

    A path that ends in one is the same resource and NVIDIA's gateway does not
    agree. Nothing else is appended or removed: see the note at the top.
    """
    return str(endpoint or "").strip().rstrip("/")


def _max_bytes(setting: str | int | None) -> int:
    """The base64 size a page is held to, or 0 for the default of not caring."""
    text = str(setting or "").strip()
    if not text:
        return 0
    try:
        limit = int(text)
    except ValueError as exc:
        raise ValueError(
            f"{config.NEMOTRON_ENV_MAX_BYTES} is not a number: {text!r}"
        ) from exc
    if limit <= 0:
        raise ValueError(
            f"{config.NEMOTRON_ENV_MAX_BYTES} must be above zero, not {limit}"
        )
    return limit
