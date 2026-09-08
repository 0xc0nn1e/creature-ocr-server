"""Google Document AI, using a Document OCR processor. One call per page.

The processor takes no instructions - there is no prompt to tell it the sheet
has 88 cells - but it reports a bounding box for every symbol it reads, in
coordinates normalised to the image it was given. That is the same frame the
cell grid uses, so one page-wide call yields a value per cell without paying for
a call per cell. (3.2, 4.2)

The difference from Gemini is the whole reason both engines exist. A box from
here says where the ink actually was, so it survives a change to the cell grid
and owes nothing to a prompt; a box from Gemini is a cell address this server
wrote. That is why the settings string below carries neither a grid nor a prompt
fingerprint, and why it still carries the checks one: a client of this server
caches finished rows, so widening a character set with nothing to say so would
serve rows read under the old rules for as long as that cache lives. (3.2, 4.2)

Two things differ from the desktop pipeline's copy of this file, and both come
from being a server. recognize returns a Reading rather than a list, so one
engine can answer many requests at once and what a call cost travels with it.
And what it had to drop comes back in that Reading's notes rather than going to
a debug log, because the operator who needs to know is on another machine.
(6.4, 7.2)

Credentials are whichever of the two ways the environment chose, and are neither
read nor written here: auth.credentials is the one place that decides. (6.3)
"""

from __future__ import annotations

import logging
import os

from .. import config
from ..ocr import (
    OCREngine,
    OCRError,
    Reading,
    TextBox,
    Usage,
    checks_fingerprint,
    header_fingerprint,
)
from . import auth

logger = logging.getLogger(__name__)


def _load_sdk():
    """Import the Document AI SDK.

    The one place the SDK is named, so the rest of the package imports and the
    tests run with google-cloud-documentai absent, and so a test can substitute
    the whole SDK without a project or a network. (4.2)
    """
    from google.api_core.client_options import ClientOptions
    from google.api_core.exceptions import GoogleAPICallError
    from google.cloud import documentai

    return documentai, ClientOptions, GoogleAPICallError


def _anchor_text(text: str, anchor) -> str:
    """Resolve a layout's text anchor into the substring it points at."""
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in anchor.text_segments
    )


def _anchor_start(token) -> int:
    """Where a token begins in document.text, i.e. its place in reading order."""
    segments = token.layout.text_anchor.text_segments
    return int(segments[0].start_index) if segments else 0


def text_boxes(document) -> tuple[list[TextBox], list[str]]:
    """Turn a Document AI response into positioned text, and say what was lost.

    Symbols, one box per character, in preference to tokens. A token is a whole
    word and its box can straddle a printed rule: on a real sheet 5 of 64 did,
    which merged 見つけた月 and 見つけた日 into a single "78" and left the day
    empty. A single character always falls inside one cell. Tokens remain the
    fallback for a processor that does not return symbols.

    Vertices are already normalised to the page, so no resolution has to be
    threaded through. Emitted in reading order, which is the order document.text
    is written in: entries are sorted by where their anchor points into it. That
    is the processor's own layout analysis, and the caller relies on it rather
    than trying to rebuild the order out of bounding boxes.

    A symbol with no position is dropped rather than guessed at - 5.2-2 would
    rather lose a value than invent where it sat - and the drop is a note rather
    than a log line, for the same reason Gemini's are. (5.2-2, 7.2)
    """
    boxes: list[TextBox] = []
    notes: list[str] = []
    for page in document.pages:
        found = list(page.symbols) or list(page.tokens)
        for entry in sorted(found, key=_anchor_start):
            layout = entry.layout
            content = _anchor_text(document.text, layout.text_anchor).strip()
            if not content:
                continue
            vertices = layout.bounding_poly.normalized_vertices
            if not vertices:
                notes.append(f"dropped {content!r}: the engine gave it no position")
                continue
            xs = [vertex.x for vertex in vertices]
            ys = [vertex.y for vertex in vertices]
            boxes.append(TextBox(content, min(xs), min(ys), max(xs), max(ys)))
    return boxes, notes


class DocumentAIEngine(OCREngine):
    """One Document OCR call per page."""

    name = "documentai"
    # What must be set before this engine can be built, and what installs the
    # SDK it needs. Declared rather than known by the registry, so adding an
    # engine stays one module: GET /v1/engines reports readiness by reading
    # these, without loading a credential or opening a socket. (4.2, 6.5)
    requires = (
        config.DOCUMENTAI_ENV_PROJECT,
        config.DOCUMENTAI_ENV_LOCATION,
        config.DOCUMENTAI_ENV_PROCESSOR_ID,
    )
    sdk = "google.cloud.documentai"
    extra = "documentai"

    @classmethod
    def settings_for(
        cls,
        model: str = "",
        sheet: str = config.SHEET_V1,
        project: str = "",
        location: str = "",
        processor_id: str = "",
    ) -> str:
        """What a client's response cache keys this engine's readings on.

        A classmethod, so it can be answered without building anything: no
        credential, no SDK, no socket. GET /v1/engines publishes it, and the
        client keys on what it is told. (3.2, 4.2)

        The processor, its region and its project, because a retrained or
        relocated processor reads the same page differently and serving the old
        one's answer back would be a wrong reading rather than a stale one. The
        language hints because they are sent with every request and come from
        configuration. The checks because a client of this server caches
        finished rows rather than boxes.

        No grid and no prompt fingerprint, unlike Gemini: these boxes are
        measurements of where the ink was, so re-reading them under a new grid
        is not merely safe, it is the point. (4.2)

        The arguments are for the engine that was actually built with them; an
        empty one falls back to the environment, which is what the endpoint
        wants.
        """
        project = project or os.environ.get(config.DOCUMENTAI_ENV_PROJECT, "")
        location = location or os.environ.get(config.DOCUMENTAI_ENV_LOCATION, "")
        processor_id = processor_id or os.environ.get(
            config.DOCUMENTAI_ENV_PROCESSOR_ID, ""
        )
        # The sheet joins it for anything but v1, and only for anything but v1:
        # the v1 string has to keep coming out exactly as it did, or every page
        # a client has already filed under it is thrown away. This engine's boxes
        # are measurements, so the frame they are read in is not part of its
        # identity - but which image was sent is, and that is what the word says.
        settings = (
            f"{project}/{location}/{processor_id} "
            f"hints={','.join(config.DOCUMENTAI_LANGUAGE_HINTS)} "
            f"checks={checks_fingerprint()}"
        )
        if sheet == config.SHEET_V1:
            return settings
        return f"{settings}+{header_fingerprint()} sheet={sheet}"

    @classmethod
    def cache_name_for(
        cls,
        model: str = "",
        sheet: str = config.SHEET_V1,
        processor_id: str = "",
    ) -> str:
        """The directory a client keeps this processor's readings under.

        The processor is the closest thing this engine has to a model, so it
        names the directory: pointing the server at another processor to compare
        the two keeps both sets of readings instead of overwriting one. (3.2, 4.1)
        """
        processor_id = processor_id or os.environ.get(
            config.DOCUMENTAI_ENV_PROCESSOR_ID, ""
        )
        named = f"{cls.name}-{processor_id}" if processor_id else cls.name
        return named if sheet == config.SHEET_V1 else f"{named}-{sheet}"

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        processor_id: str | None = None,
        sheet: str = config.SHEET_V1,
    ) -> None:
        project = project or os.environ.get(config.DOCUMENTAI_ENV_PROJECT, "")
        location = location or os.environ.get(config.DOCUMENTAI_ENV_LOCATION, "")
        processor_id = processor_id or os.environ.get(
            config.DOCUMENTAI_ENV_PROCESSOR_ID, ""
        )
        # Checked before the SDK is even loaded, so a misconfigured server says
        # so at the first request instead of failing its way through a batch.
        # The region is demanded rather than defaulted: guessing it wrong makes
        # every call come back NOT_FOUND, which reads as a broken processor
        # rather than as a wrong setting.
        for name, value in (
            (config.DOCUMENTAI_ENV_PROJECT, project),
            (config.DOCUMENTAI_ENV_LOCATION, location),
            (config.DOCUMENTAI_ENV_PROCESSOR_ID, processor_id),
        ):
            if not value:
                raise ValueError(f"{name} is not set: see .env.example")

        try:
            documentai, client_options, api_error = _load_sdk()
        except ImportError as exc:
            raise ValueError(
                "google-cloud-documentai is not installed: "
                'pip install -e ".[documentai]"'
            ) from exc

        self._sdk = documentai
        self._api_error = api_error
        # The endpoint is region specific and must match the region the
        # processor was created in, or every call comes back NOT_FOUND.
        #
        # Credentials are asked for after the SDK loaded, so a machine with no
        # SDK at all is told to install the extra rather than told about a key.
        # None is the SDK's own default and means Application Default
        # Credentials, so an empty setting leaves this call as it always was.
        self._client = documentai.DocumentProcessorServiceClient(
            credentials=auth.credentials(),
            client_options=client_options(
                api_endpoint=f"{location}-documentai.googleapis.com"
            ),
        )
        self._processor = self._client.processor_path(project, location, processor_id)
        # Worked out by the same functions GET /v1/engines answers with, so what
        # a client was told to key its cache on is what actually read the page.
        self.settings = self.settings_for("", sheet, project, location, processor_id)
        self.cache_name = self.cache_name_for("", sheet, processor_id)
        # enable_symbol asks for per-character boxes, which is what keeps a word
        # written across a printed rule from landing wholly in one cell. The
        # language hint stops the processor reading handwritten kana as Korean.
        self._options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_symbol=True,
                hints=documentai.OcrConfig.Hints(
                    language_hints=list(config.DOCUMENTAI_LANGUAGE_HINTS)
                ),
            )
        )

    def recognize(self, image: bytes, mime_type: str = "image/png") -> Reading:
        request = self._sdk.ProcessRequest(
            name=self._processor,
            raw_document=self._sdk.RawDocument(content=image, mime_type=mime_type),
            process_options=self._options,
        )
        try:
            response = self._client.process_document(request=request)
        except self._api_error as exc:
            raise OCRError(f"Document AI refused a page: {exc}") from exc
        boxes, notes = text_boxes(response.document)
        # This API bills per page rather than per token and reports no count, so
        # the usage is empty and recognize_with_retry fills in the call and the
        # time. Zero tokens here is a true statement about Document AI, not a
        # measurement that went missing. (6.1, 7.1)
        return Reading(boxes, Usage(), notes)
