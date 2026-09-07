"""Gemini on Vertex AI, one structured-extraction call per page.

Where a Document OCR processor takes no instructions, an LLM can be told what
the sheet is: the prompt names the eleven columns in the order they are printed
and states 5.2-2's rules, and the response schema makes `no` the only required
field, so a blank cell can come back blank instead of being filled to satisfy
the schema. That is what 5.2-1 and 5.2-2 ask for and what a Document OCR
processor cannot be given. (4.1, 5.2-1, 5.2-2)

What comes back is rows, not positioned text, so each value is emitted as a
TextBox covering the cell it was asked for. Those coordinates are a cell address
written in the frame ocr.assign_to_cells reads, not a measurement of where the
model saw something: the model is never asked for a bounding box, because the
grid already knows where every cell is to four decimal places. Handing the rows
back through the same geometry keeps OCREngine to one response shape, and it
puts every engine through the same ocr.check, which is what makes the 4.1
comparison a like-for-like one.

A row the model numbers wrongly is dropped here, never raised. Anything raised
out of recognize is retried three times and then costs the whole page - eight
rows against 5.1's 0% row-drop target - so one bad row must not become one lost
sheet. What was dropped and why is returned in the reading's notes rather than
only logged, because the operator who needs to know is on another machine.
(5.1, 6.4, 7.2)

Two things differ from the desktop pipeline's copy of this file, and both come
from being a server. recognize returns what the call cost instead of adding it
to the engine, so one engine can answer many requests at once. And settings_for
is a module-level function rather than something only a built engine knows, so
GET /v1/engines can publish the cache key for every allowed model without
loading a credential or opening a socket.

Credentials are whichever of the two ways the environment chose, and are neither
read nor written here: auth.credentials is the one place that decides. (6.3)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import string
from collections.abc import Iterable

from .. import config
from ..grid import Cell, cell_band, iter_cells
from ..ocr import (
    OCREngine,
    OCRError,
    Reading,
    TextBox,
    Usage,
    checks_fingerprint,
)
from . import auth

logger = logging.getLogger(__name__)


def _load_sdk():
    """Import the Gen AI SDK.

    The one place the SDK is named, so the rest of the package imports and the
    tests run with google-genai absent, and so a test can substitute the whole
    SDK without a project or a network. (4.2)
    """
    from google import genai
    from google.genai import errors, types

    return genai, types, errors.APIError


def response_schema() -> dict:
    """The JSON shape one page comes back in.

    A plain dict rather than the SDK's Schema type, so the schema stays data
    that can be read and asserted on with no SDK installed, and so nothing
    outside _load_sdk names the vendor. Built from OCR_FIELDS, so a change of
    paper format is still only a config edit. (6.5)
    """
    fields = [key for key, _ in config.OCR_FIELDS]
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                # An integer, so there is no "3." or "no.3" to parse. The range
                # is the sheet's own, not a constant.
                "no": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": config.ROWS_PER_PAGE,
                },
                **{key: {"type": "string"} for key in fields},
                # Which of this row's cells the model had to strain to read.
                # The only place a strained reading can come from: it looks
                # exactly like a confident one by the time it is text, and
                # 花だん for 花ばたけ passes every check this pipeline has.
                # Constrained to the real column names, so an invented one
                # cannot arrive; parsing checks again anyway. (7.2)
                "unsure": {
                    "type": "array",
                    "items": {"type": "string", "enum": fields},
                },
            },
            # 5.2-1: no is the only required field. Leaving every other one
            # optional is what takes the pressure off an empty cell - a model
            # that must produce a value will invent one.
            "required": ["no"],
            # Vertex honours this to fix the order keys are emitted in.
            # OCR_FIELDS is already the paper's left-to-right order, so this
            # makes the model read across the row the way the row is written.
            # unsure comes last: it is a judgement about the row, so it should
            # be made after the row has been read, not before.
            "propertyOrdering": ["no", *fields, "unsure"],
        },
    }


def grid_fingerprint() -> str:
    """A short digest of the cell grid this engine addresses its boxes in.

    Part of the cache identity, because a box from here is a cell address
    rather than a measurement: move a column and a cached box is re-read into
    the wrong field. Four of the eleven columns changed field the last time the
    crop was re-measured, which would have moved 見つけた日 into 見つけた月
    silently.

    Document AI needs nothing of the kind - its boxes say where the ink
    actually was, so re-reading them under a new grid is not only safe but the
    point of caching boxes rather than rows. The difference between the two
    engines is exactly this, so it belongs here and not in the cache. (4.2)
    """
    layout = repr(
        (
            config.ROWS_PER_PAGE,
            config.CELL_ROW_EDGES,
            tuple(config.CELL_COLUMNS.items()),
        )
    )
    return hashlib.sha256(layout.encode()).hexdigest()[:8]


def prompt_fingerprint() -> str:
    """A short digest of the instructions a page was read under.

    Part of the cache identity for the same reason the grid is. What this engine
    returns is the answer to a question, so changing the question - another
    printed list, another rule - changes the answer, and a cached reading taken
    under the old prompt is no longer this engine's reading of that page.
    Without it, adding the handbook's answer lists would have gone on serving
    readings taken before they existed and 7.2 would have measured the prompt
    that was replaced. (3.2, 4.2, 7.2)

    Document AI needs nothing of the kind: it takes no instructions at all.
    """
    return hashlib.sha256(build_prompt().encode()).hexdigest()[:8]


def _character_rule(field: str, allowed: str) -> str:
    """Say in words what OCR_FIELD_CHARACTERS says as a set."""
    if allowed == string.digits:
        return f"- {field}: digits 0-9 only."
    if allowed == string.ascii_letters:
        return f"- {field}: a single Latin letter only."
    return f"- {field}: only these characters: {allowed}"


def _choice_rule(
    field: str, choices: dict[str, str], spellings: dict[str, str]
) -> str:
    """Say in words what OCR_FIELD_CHOICES and OCR_FIELD_SPELLINGS say as tables.

    The other spellings have to be named here too, or the prompt asks for less
    than ocr.clean accepts and the model obeys the prompt. Told only the printed
    list and the rule above it - anything else is an empty string - a model
    reading 何してた's number written with no ring round it returns nothing at
    all. That is measured, not imagined: three cells on one page of five came
    back empty that way, all three a bare handwritten 4.

    The sentence is worded for the ring, because 何してた is the only column
    OCR_FIELD_SPELLINGS has an entry for. Give another column one and this has
    to be reworded with it - and that costs a re-read of every cached page,
    because the prompt fingerprint is part of the cache key.
    """
    rule = f"- {field}: exactly one of {' '.join(choices)}, and nothing else."
    if spellings:
        rule += (
            " A number written with no ring round it is still one of those"
            " answers: return it as the child wrote it, one of"
            f" {' '.join(spellings)}."
        )
    return rule


def _choice_legend(field: str, header: str, choices: dict[str, str]) -> str:
    """The printed list a column is answered off, answers and meanings.

    The meanings are the point of sending it: a single handwritten お is hard,
    葉っぱ in the column beside it is not, and the model can only use the one
    to read the other if it knows what お stands for.
    """
    listed = "\n".join(f"  {answer}  {meaning}" for answer, meaning in choices.items())
    return f"{field}, the column headed {header}:\n{listed}"


def build_prompt() -> str:
    """The instructions sent with every page.

    Assembled from OCR_FIELDS and OCR_FIELD_CHARACTERS rather than written out,
    so a new paper format needs no edit here. (6.5)

    English instructions around the sheet's own Japanese column headings: the
    headings are what is printed on the paper and have to appear exactly as
    printed, and the surrounding prose follows the repository's language rule.
    The risk that carries - an English instruction inviting the model to
    translate what it reads - is answered by the copy-exactly rule below.

    5.2-5 suggests sending the header band and the data band as separate
    images. Stage 1 has already cut the header off, so there is no header band
    left to send; naming the columns in order does the same job.

    The same reasoning is why the survey handbook's printed lists are written
    out here rather than attached as the photographs they were read off. What
    the model needs is the answers and what they mean, and a page of clean text
    says that without also sending a creased booklet on a desk. The lists come
    from config, so a new paper format is still only a config edit. (6.5)
    """
    rows = config.ROWS_PER_PAGE
    columns = "\n".join(
        f"- {key}: the column headed {header}" for key, header in config.OCR_FIELDS
    )
    characters = "\n".join(
        _character_rule(field, allowed)
        for field, allowed in config.OCR_FIELD_CHARACTERS.items()
    )
    headers = dict(config.OCR_FIELDS)
    choices = "\n".join(
        _choice_rule(field, printed, config.OCR_FIELD_SPELLINGS.get(field, {}))
        for field, printed in config.OCR_FIELD_CHOICES.items()
    )
    legends = "\n\n".join(
        _choice_legend(field, headers[field], printed)
        for field, printed in config.OCR_FIELD_CHOICES.items()
    )
    pairs = "\n".join(
        f"- {letter_field} is the printed answer and {wording_field} is the "
        "child's own words for the same thing."
        for letter_field, wording_field, _ in config.OCR_PAIRED_FIELDS
    )
    recorded = "、".join(config.BUG_NAME_HINTS)
    return f"""\
You are transcribing one scanned page of a handwritten Japanese wildlife survey
sheet. The page is a printed table with exactly {rows} data rows.

The leftmost printed column holds the row number, 1 to {rows}. Return it as
`no`. It is not something to transcribe; it tells you, and tells the reader of
your answer, which printed row a value was written in. Read it off the paper
rather than counting the rows you have answered so far.

The columns to transcribe, left to right, are:
{columns}

Return one object per row, for every row 1 to {rows}, including rows that are
entirely empty.

Rules:
- Never take a value from another row. Read each row only from that row.
- A cell that is blank on the paper is an empty string. Do not fill it.
- A cell you cannot read is an empty string. Never guess, and never write a
  value merely because it would be plausible.
- Write the same value as a neighbouring row only when you can see that the
  handwriting is the same. Do not copy a value down a column.
- Copy the Japanese exactly as it is written. Do not translate it, do not
  romanise it, do not convert between kana and kanji, and do not correct
  spelling or grammar.

Some columns can only hold certain characters. If what is written does not fit,
return an empty string rather than the nearest character that does:
{characters}

Some columns are not written freely at all: the child copies one answer off a
printed list. Return the listed answer exactly as it is spelled here, and an
empty string when what is on the paper is not one of them:
{choices}

Never draw a ring a number does not have, and never take off one it does. The
ring is on the paper or it is not.

Those lists are printed in the survey handbook the sheet is filled in from.
What each answer means:

{legends}

The meanings are there to read the handwriting by, and for nothing else. Never
copy a meaning into a cell, never replace what a child wrote with the wording
used here, and never expand what they wrote into the full name of a real place.
A cell holds what is written in it.

Two of those columns are answered twice, once as the printed answer and once in
the child's own words:
{pairs}
Read each of the two with the other one in mind - a single kana is hard to read
alone and easy beside the words that explain it. Then write down what is
actually on the paper. If the two disagree, return both as they stand and
correct neither: you cannot tell which one the child got wrong, and a corrected
value is an invented one.

The same survey ran last year and recorded these names, most common first:
{recorded}

Use them only to settle handwriting you can already almost read. Handwriting
that differs from a listed name by even one character is returned as the
handwriting, not as the listed name. A name that is not on the list is written
down as it is read, and a blank cell stays blank.

The table's first printed row is a worked example labelled 例 instead of a
number, and the bottom of it can still be visible above row 1. It is printed,
not handwritten, and it is not one of the {rows} rows. Never return it, and
never let any of its values into row 1.

Some of this handwriting is a child's, and some of it runs together into a
shape you can only partly resolve. In `unsure`, list the columns of that row
whose handwriting you could not read cleanly: strokes that run into one
another, a character you had to choose a reading of, a word you settled by what
would make sense rather than by what you could see.

Still return your best reading of those cells. `unsure` does not replace the
value and it is not permission to guess - a cell you cannot read at all is
still an empty string, and everything above still holds. It marks a cell for a
person to look at, and it is the only way anyone can be told: by the time a
strained reading is text it looks exactly like a confident one.

Leave `unsure` out of a row you read cleanly. It is for the cells that gave you
trouble, so listing every column says nothing at all.
"""


def _row_number(entry: dict, position: int, notes: list[str]) -> int | None:
    """Which printed row this object is, or None if it does not say.

    Placing a row by the number it reports rather than by where it sits in the
    response is what stops a reordered or short answer from shifting every
    value one row up. Every rejection here is a note and one lost row; none of
    them raises, because a raised exception costs the other seven. (5.1)
    """
    raw = entry.get("no")
    try:
        # float first, so both 3 and a JSON 3.0 arrive as 3.
        number = int(float(raw))
    except (TypeError, ValueError):
        notes.append(f"dropped a row: {raw!r} is not a row number")
        return None
    if not 1 <= number <= config.ROWS_PER_PAGE:
        notes.append(
            f"dropped row {number}: the sheet has {config.ROWS_PER_PAGE} rows"
        )
        return None
    if number != position:
        notes.append(f"row {number} came back in position {position}")
    return number


def _unsure_fields(entry: dict, number: int, notes: list[str]) -> set[str]:
    """Which of a row's columns the model said it could not read cleanly.

    Read the same way the values are: by looking each name up against
    OCR_FIELDS, never by trusting what the model listed. The schema already
    constrains it, but a name that is not a column of this sheet cannot mark a
    cell, and a mark on a column that does not exist is worth a word rather
    than a silent drop.
    """
    listed = entry.get("unsure") or []
    if not isinstance(listed, list):
        notes.append(f"row {number}: unsure is not a list: {listed!r}")
        return set()
    fields = {field for field, _ in config.OCR_FIELDS}
    found = set()
    for name in listed:
        if name in fields:
            found.add(name)
        else:
            notes.append(
                f"row {number}: unsure names no column of this sheet: {name!r}"
            )
    return found


def _by_cell(
    rows: Iterable[object], notes: list[str]
) -> tuple[dict[Cell, str], set[Cell]]:
    """Sort the response into cells, dropping anything that cannot be placed.

    Returns what each cell holds and which cells the model flagged as badly
    read. Both come out of the same pass, because a flag is only meaningful
    for a row that was placed at all.

    Fields are taken by looking up each OCR_FIELDS key, never by walking the
    keys the model sent, so an invented column cannot reach a cell. A second
    row claiming a number already used is dropped rather than merged: values
    landing in one cell are joined downstream, so keeping both would
    concatenate two rows into one value nobody wrote and blank the other row.
    """
    values: dict[Cell, str] = {}
    unsure: set[Cell] = set()
    seen: set[int] = set()
    for position, entry in enumerate(rows, start=1):
        if not isinstance(entry, dict):
            notes.append(f"dropped a row that is not an object: {entry!r}")
            continue
        number = _row_number(entry, position, notes)
        if number is None:
            continue
        if number in seen:
            notes.append(f"dropped a second row numbered {number}")
            continue
        seen.add(number)
        flagged = _unsure_fields(entry, number, notes)
        for field, _ in config.OCR_FIELDS:
            value = entry.get(field)
            text = "" if value is None else str(value).strip()
            if text:
                values[(number, field)] = text
                if field in flagged:
                    unsure.add((number, field))
    missing = sorted(set(range(1, config.ROWS_PER_PAGE + 1)) - seen)
    if missing:
        notes.append(f"the page came back without row(s) {missing}")
    return values, unsure


def text_boxes(rows: Iterable[object]) -> tuple[list[TextBox], list[str]]:
    """Turn a Gemini response into positioned text, and say what went wrong.

    One box per cell that holds something, covering that cell's own band. The
    box is the address the value was asked for; the model is not asked where it
    saw anything. A box the model flagged carries that flag, so a strained
    reading reaches the caller's error file. (7.2)

    Emitted in the grid's own order, whatever order the model answered in, so
    the reading order recognize promises does not depend on the model.

    The notes come back rather than going to a log, because the desktop
    pipeline collected them off its own logger and this one cannot: a row the
    model never sent is exactly the kind of thing the operator has to be told,
    and the operator is not on this machine. read_page_image logs them here as
    well as forwarding them. (6.4, 7.2)
    """
    notes: list[str] = []
    values, unsure = _by_cell(rows, notes)
    boxes = []
    for cell in iter_cells():
        text = values.get(cell)
        if text:
            boxes.append(TextBox(text, *cell_band(*cell), unsure=cell in unsure))
    return boxes, notes


def _finish_reason(response) -> str:
    """Why a response carried no text: a safety block, or the token budget."""
    reasons = [
        str(getattr(candidate, "finish_reason", ""))
        for candidate in getattr(response, "candidates", None) or []
    ]
    return ", ".join(reason for reason in reasons if reason) or "no reason given"


def tokens(response) -> Usage:
    """What the call was billed for, or nothing if the SDK did not say.

    Read defensively: usage_metadata is reporting, not the answer, and a page
    that was read must not be thrown away because the meter was missing.

    Thinking counts as output for billing, so it goes into output_tokens, and
    is kept separately as well - it is the only way to see the budget that
    disappears when a thinking model returns no text at all. (6.1, 7.1)
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return Usage()
    thoughts = getattr(meta, "thoughts_token_count", None) or 0
    answer = getattr(meta, "candidates_token_count", None) or 0
    return Usage(
        prompt_tokens=getattr(meta, "prompt_token_count", None) or 0,
        output_tokens=answer + thoughts,
        thought_tokens=thoughts,
    )


def parse_rows(response) -> list:
    """Read the rows out of a response, or say why there are none.

    An empty response is reported with the finish reason attached, because the
    two ways it happens - the answer was blocked, or it ran out of output
    tokens - are settings problems and look like nothing at all otherwise.
    Everything here raises: it is a whole page that did not arrive, so 6.4's
    retry is the right answer, unlike a single unplaceable row.
    """
    text = getattr(response, "text", None)
    if not text:
        raise OCRError(f"Gemini returned no text ({_finish_reason(response)})")
    try:
        found = json.loads(text)
    except ValueError as exc:
        raise OCRError(f"Gemini returned something that is not JSON: {exc}") from exc
    if isinstance(found, dict):
        # The schema asks for a bare array; accept a wrapped one anyway rather
        # than throw away a page that is otherwise perfectly good.
        found = found.get("rows")
    if not isinstance(found, list):
        raise OCRError("Gemini returned no list of rows")
    return found


class GeminiEngine(OCREngine):
    """One structured-extraction call per page."""

    name = "gemini"
    # What must be set before this engine can be built, and what installs the
    # SDK it needs. Declared rather than known by the registry, so adding an
    # engine stays one module: GET /v1/engines reports readiness by reading
    # these, without loading a credential or opening a socket. (4.2, 6.5)
    requires = (
        config.GEMINI_ENV_PROJECT,
        config.GEMINI_ENV_LOCATION,
        config.GEMINI_ENV_MODEL,
    )
    sdk = "google.genai"
    extra = "gemini"

    @classmethod
    def models(cls) -> tuple[str, ...]:
        """Which models a request may name, best first.

        An allow-list and not merely a menu: an engine is built once per model
        and kept, so a free-form model string would grow this server an
        unbounded number of SDK clients. Overridable from the environment,
        because Vertex retires model ids on Google's own schedule and an
        operator should not have to wait for a release. (4.1, 6.5)
        """
        listed = os.environ.get(config.GEMINI_ENV_MODELS, "")
        if listed.strip():
            return tuple(name.strip() for name in listed.split(",") if name.strip())
        return config.GEMINI_MODELS

    @classmethod
    def default_model(cls) -> str:
        """What GEMINI_MODEL names, or nothing at all.

        Deliberately not defaulted to a model id: Vertex retires them on
        Google's own schedule, so a default written here would eventually
        either fail every call or, worse, quietly measure a model nobody
        chose. Empty means the server is not configured yet, which is what
        GET /v1/engines reports. (6.5)
        """
        return os.environ.get(config.GEMINI_ENV_MODEL, "").strip()

    @classmethod
    def cache_name_for(cls, model: str) -> str:
        """The directory a client keeps this model's readings under. (3.2, 4.1)"""
        return model

    @classmethod
    def settings_for(
        cls,
        model: str,
        temperature: float = config.GEMINI_TEMPERATURE,
        top_p: float = config.GEMINI_TOP_P,
    ) -> str:
        """What a client's response cache keys this engine's readings on.

        A classmethod, so it can be answered without building anything: no
        credential, no SDK, no socket. That matters because the client cannot
        work this string out for itself any more - the prompt, the grid and the
        value checks that go into it are all this server's - so GET /v1/engines
        publishes it for every allowed model and the client keys on what it is
        told. (3.2, 4.2)

        Five things move it, and each of them changes what a page says:

        - the model, obviously;
        - the generation settings, because 5.2-4's determinism is a promise
          about those two numbers;
        - the grid, because a box from here is a cell address rather than a
          measurement, so moving a column re-reads a cached value into the
          wrong field - four of the eleven columns changed field the last time
          the crop was re-measured;
        - the prompt, because what this engine returns is the answer to a
          question, and a different question is a different answer;
        - the checks, because a client of this server caches finished rows, so
          widening a character set with nothing to say so would serve rows read
          under the old rules for as long as that cache lives.

        (3.2, 4.2, 5.2-4, 7.2)
        """
        return (
            f"{model} temperature={temperature} top_p={top_p} "
            f"grid={grid_fingerprint()} prompt={prompt_fingerprint()} "
            f"checks={checks_fingerprint()}"
        )

    def __init__(
        self,
        project: str | None = None,
        location: str | None = None,
        model: str | None = None,
        temperature: float = config.GEMINI_TEMPERATURE,
        top_p: float = config.GEMINI_TOP_P,
    ) -> None:
        project = project or os.environ.get(config.GEMINI_ENV_PROJECT, "")
        location = location or os.environ.get(config.GEMINI_ENV_LOCATION, "")
        model = model or os.environ.get(config.GEMINI_ENV_MODEL, "")
        # The model arrives as an argument. The desktop pipeline sets an
        # environment variable and lets the constructor read it back, which is
        # correct for one run on one machine and would be a race here: a global
        # write under a threadpool can build an engine labelled with another
        # request's model, which then caches its readings under the wrong name.
        #
        # Checked before the SDK is loaded, so a misconfigured server says so at
        # the first request instead of failing its way through a batch. The
        # model is demanded rather than defaulted: Vertex model ids are
        # versioned and retired on Google's own schedule, so a default written
        # here would eventually either fail every call or, worse, quietly
        # measure a model nobody chose.
        for name, value in (
            (config.GEMINI_ENV_PROJECT, project),
            (config.GEMINI_ENV_LOCATION, location),
            (config.GEMINI_ENV_MODEL, model),
        ):
            if not value:
                raise ValueError(f"{name} is not set: see .env.example")

        try:
            genai, types, api_error = _load_sdk()
        except ImportError as exc:
            raise ValueError(
                'google-genai is not installed: pip install -e ".[gemini]"'
            ) from exc

        self._types = types
        self._api_error = api_error
        self._model = model
        # Worked out by the same function GET /v1/engines answers with, so what
        # a client was told to key its cache on is what actually read the page.
        self.settings = self.settings_for(model, temperature, top_p)
        # The model names the cache directory a client keeps, so the comparison
        # 4.1 asks for is paid for once: switching models keeps what the
        # previous one read instead of overwriting it. (3.2, 4.1)
        self.cache_name = model
        # Credentials are asked for after the SDK loaded, so a machine with no
        # SDK at all is told to install the extra rather than told about a key.
        # None is the SDK's own default and means Application Default
        # Credentials, so an empty setting leaves this call as it always was.
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=auth.credentials(),
        )
        # Built once, and shared by every request this engine serves. No output
        # token limit: a model that thinks spends that budget on thinking and
        # then returns nothing at all, which looks exactly like a refusal.
        self._config = types.GenerateContentConfig(
            system_instruction=build_prompt(),
            response_mime_type="application/json",
            response_schema=response_schema(),
            temperature=temperature,
            top_p=top_p,
        )

    def recognize(self, image: bytes, mime_type: str = "image/png") -> Reading:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    self._types.Part.from_bytes(data=image, mime_type=mime_type)
                ],
                config=self._config,
            )
        except self._api_error as exc:
            raise OCRError(f"Gemini refused a page: {exc}") from exc
        # Read before the response is parsed, and attached to the failure if it
        # cannot be: a page that came back unusable was still charged for, and
        # hiding that would understate the run. (6.1, 7.1)
        spent = tokens(response)
        try:
            rows = parse_rows(response)
        except OCRError as exc:
            exc.usage = spent
            raise
        boxes, notes = text_boxes(rows)
        return Reading(boxes, spent, notes)
