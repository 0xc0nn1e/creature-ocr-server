"""Read one cropped page image into the rows of the survey sheet.

One engine call per page. The engine reports what it read and where on the page
it read it, and the field a value belongs to is decided here, from the cell its
box sits in. So the cell mapping stays geometric - a value can only come from
the cell it was written in - without paying for one call per cell. (3.2, 5.2-2)

This is the desktop pipeline's ocr.py with the file handling taken out. There is
no response cache and no error log: a request carries one page in and one answer
out, so what the desktop version wrote to disk this version returns. Everything
between those two edges - the cell assignment, the character sets, the printed
answer lists, the paired-column check and the page confidence - is the same
code, and tests/test_parity.py holds it to that. (3.2, 6.4, 7.2)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from . import config
from .grid import Cell, cell_at, iter_cells

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """A page the engine could not be asked about, not a page that is blank.

    It carries what the failed attempts cost. An engine bills for a response it
    sent whether or not that response could be used, so a page that burned four
    calls and produced nothing still has to be able to say so - otherwise the
    only calls a run could account for would be the ones that worked, and the
    bill would be a surprise. (6.1, 7.1)
    """

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage if usage is not None else Usage()


class OCRTimeout(OCRError):
    """The budget for this page ran out before another attempt could start.

    Its own type because it is a different answer to the caller: a page that
    failed is a page the engine could not read, and a page that ran out of time
    is one nobody finished asking about. Mixing the two would report a healthy
    engine as broken on the day somebody set the deadline too low. (6.1, 6.4)
    """


@dataclass(frozen=True)
class TextBox:
    """One piece of text and where it sits, as fractions of the page.

    The same frame the cell grid uses, so a box can be placed in a cell without
    knowing the render resolution.
    """

    text: str
    left: float
    top: float
    right: float
    bottom: float
    # The engine read this, and said it could not read it cleanly. Cramped
    # handwriting that runs together, a character it had to choose a reading
    # of. The text is still its best reading and still goes to the xlsx: this
    # only marks the cell for a person to look at, because nothing else in the
    # pipeline can tell a confident reading from a strained one. A character
    # set cannot - 花だん and 徃門小学校 are both perfectly ordinary Japanese.
    #
    # Last with a default, so every TextBox(text, l, t, r, b) still builds and
    # every cache written before this field existed still loads. (5.2-2, 7.2)
    unsure: bool = False

    @property
    def centre(self) -> tuple[float, float]:
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2


@dataclass(frozen=True)
class Usage:
    """What engine calls have cost so far: how many, how long, how many tokens.

    6.1 sets a per-page latency budget and 7.1 puts 1,000 sheets through this,
    so both numbers have to be visible from an ordinary run rather than worked
    out afterwards from a bill. Tokens stay zero for an engine that bills per
    request and reports none, which is a true statement about Document AI
    rather than a missing measurement.

    Frozen, so a single instance is safe as the class-level default on
    OCREngine: adding to it binds a new value on the engine rather than
    mutating one that every engine shares.
    """

    calls: int = 0
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    # Billed as output, and counted in output_tokens as well. Kept apart
    # because it is the budget that disappears silently: a thinking model that
    # spends it all returns no text at all, which reads like a refusal.
    thought_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.calls + other.calls,
            self.seconds + other.seconds,
            self.prompt_tokens + other.prompt_tokens,
            self.output_tokens + other.output_tokens,
            self.thought_tokens + other.thought_tokens,
        )

    def __sub__(self, other: Usage) -> Usage:
        """What was spent between two readings of a running total."""
        return Usage(
            self.calls - other.calls,
            self.seconds - other.seconds,
            self.prompt_tokens - other.prompt_tokens,
            self.output_tokens - other.output_tokens,
            self.thought_tokens - other.thought_tokens,
        )

    def __str__(self) -> str:
        """Say only what is known, and never hide a number that is there."""
        parts = []
        if self.calls:
            plural = "" if self.calls == 1 else "s"
            parts.append(f"{self.calls} call{plural} in {self.seconds:.1f}s")
        if self.prompt_tokens or self.output_tokens:
            counted = f"{self.prompt_tokens:,} in / {self.output_tokens:,} out tokens"
            if self.thought_tokens:
                counted += f" ({self.thought_tokens:,} of the output was thinking)"
            parts.append(counted)
        return ", ".join(parts) or "no calls"


@dataclass(frozen=True)
class Reading:
    """What one engine call produced: the text, what it cost, what it noticed.

    Returned rather than recorded on the engine. The desktop pipeline adds each
    call into an OCREngine.usage running total, which is a sensible thing for a
    batch with a beginning and an end; a server has neither. One engine is built
    per model here and shared by every request that names it, and an engine that
    accumulated into an attribute would mix one caller's tokens into another
    caller's answer. Nothing here is mutable and nothing outside the call can
    reach it, which is what makes that sharing safe. (6.1, 7.1)

    `notes` is what the engine found wrong with a response and could still work
    around: a row it had to drop, a row that came back in the wrong position, a
    row the model never sent at all. The desktop pipeline logs those and
    collects them into ocr_error.txt. Here they have to be handed back, because
    the only machine that can write that file is somewhere else. (6.4, 7.2)
    """

    boxes: list[TextBox] = dataclass_field(default_factory=list)
    usage: Usage = Usage()
    notes: list[str] = dataclass_field(default_factory=list)


class OCREngine(ABC):
    """One OCR backend.

    Engine SDKs are imported inside an implementation only, so the package
    works with none of them installed and swapping engine stays a configuration
    change rather than a code change. (4.2, 6.5)

    An engine is built once and then answers every request that names it, so an
    implementation must hold nothing that changes from one call to the next.
    That is the whole reason recognize returns what the call cost instead of
    adding it to an attribute, and it is why there is no `usage` here to add to.
    """

    name: str
    # What must be set in the environment before this engine can be built, and
    # the vendor module it imports with the pip extra that installs it.
    # Declared by the engine rather than known by the registry, so adding a
    # backend stays one module and GET /v1/engines can report whether it is
    # usable without loading a credential or opening a socket. Empty means
    # nothing is needed, which is what a test double wants. (4.2, 6.5)
    requires: tuple[str, ...] = ()
    sdk: str = ""
    extra: str = ""
    # What this engine was set up with, for a client's response cache to key
    # on: a reading is only reusable while it still comes from the same reader.
    # Every engine has settings that arrive from the environment and change what
    # comes back - a model id, a processor id - so every engine fills this in.
    # The default is for a test double, not for a real engine.
    #
    # A client cannot work this string out for itself any more, because the
    # prompt and the cell grid that go into it are this server's. So it has to
    # be publishable without building anything: GET /v1/engines reports it per
    # model, and the client keys its cache on what it is told. (3.2, 4.2)
    settings: str = ""
    # The directory a client should keep this engine's readings under, one per
    # model, so that swapping models keeps both sets rather than overwriting
    # the older one. Whatever names the reader best: the model id for Gemini,
    # the processor for Document AI. Empty falls back to the engine name, which
    # is what a test double wants. (3.2, 4.1)
    cache_name: str = ""

    @classmethod
    def models(cls) -> tuple[str, ...]:
        """Which models a request may name for this engine, best first.

        Empty for an engine whose reader is not a model at all - a Document AI
        processor, a NIM deployment - and asking one of those for a model is a
        mistake worth saying so about rather than a setting to ignore. (6.5)
        """
        return ()

    @classmethod
    def default_model(cls) -> str:
        """The model a request that names none is served with.

        Published so a client can show what it will get without asking for
        anything, and empty for an engine whose reader is not a model. (6.5)
        """
        return ""

    @classmethod
    def settings_for(cls, model: str) -> str:
        """What a client should key its cache on for this engine and model.

        Answerable without building anything, because GET /v1/engines answers
        it for every allowed model and must not need a credential to do so. See
        the note on `settings` above for why the client cannot work it out. (3.2)
        """
        return ""

    @classmethod
    def cache_name_for(cls, model: str) -> str:
        """The directory a client should keep this model's readings under."""
        return cls.name

    @abstractmethod
    def recognize(self, image: bytes, mime_type: str = "image/png") -> Reading:
        """Read one whole page and return what was found, with positions.

        **In reading order.** The caller keeps that order as given and never
        re-sorts it: an engine has done real layout analysis over the whole
        page, and a rule reconstructed from bounding boxes alone cannot match
        it. Guessing the order back from geometry scrambles a line whose right
        end dips below the start of the next one, which ordinary cramped
        handwriting does, and it does so silently.

        An empty page returns an empty reading. Never guess at text that is not
        there, and never report a box outside the page. (5.2-2)

        A page that could not be read at all raises OCRError, carrying what the
        attempt cost. A page that is merely blank is not that. (6.1, 6.4)
        """


def recognize_with_retry(
    engine: OCREngine,
    image: bytes,
    mime_type: str = "image/png",
    retries: int = config.OCR_RETRIES,
    backoff: float = config.OCR_BACKOFF_SECONDS,
    deadline: float | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Reading:
    """Call the engine, retrying a failure with exponential backoff. (6.4)

    Kept out of the engines so every engine gets the same policy, and out of the
    request handler so a second caller cannot get a different one. sleep can be
    replaced so tests do not wait out the backoff; it is looked up when the call
    is made rather than bound as a default, so patching time.sleep works too.

    What comes back carries everything the attempts cost, the failed ones
    included: a call that was sent was billed, and a total that counted only the
    calls that worked would understate the run. An engine that raises says what
    its attempt cost through OCRError.usage. (6.1, 7.1)

    `deadline` is the time.monotonic() instant this page has to be finished by,
    and it is checked before an attempt starts rather than during one. Cutting a
    call off halfway throws away the whole of it and leaves the caller nothing
    for the money; refusing to start one that cannot finish leaves them a clear
    answer with the earlier failures still accounted for. None means no budget,
    which is what a test wants. (6.1, 6.4)
    """
    sleep = sleep or time.sleep
    spent = Usage()
    for attempt in range(retries + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise OCRTimeout(
                f"{engine.name} ran out of time after {attempt} attempt(s)",
                spent,
            )
        started = time.monotonic()
        try:
            reading = engine.recognize(image, mime_type)
        except Exception as exc:
            # Counted before anything is decided about the failure: it was
            # sent, so it was billed, and it is why the request took as long as
            # it did. An engine that knows what its own attempt cost says so on
            # the exception; one that does not adds nothing.
            spent = spent + Usage(calls=1, seconds=time.monotonic() - started)
            spent = spent + getattr(exc, "usage", Usage())
            if attempt == retries:
                raise OCRError(
                    f"{engine.name} gave up after {retries} retries (6.4)", spent
                ) from exc
            delay = backoff * 2**attempt
            logger.warning(
                "%s failed (%s), retrying in %.1fs", engine.name, exc, delay
            )
        else:
            spent = spent + Usage(calls=1, seconds=time.monotonic() - started)
            return Reading(reading.boxes, spent + reading.usage, reading.notes)
        # Deliberately after the clock has stopped: waiting is not call time.
        # At the default backoff three retries wait 1 + 2 + 4 seconds, which
        # would swamp the latency the log is there to show. (6.1)
        sleep(delay)


def assign_to_cells(boxes: Iterable[TextBox]) -> dict[Cell, str]:
    """Place every box in the cell its centre falls in and join each cell's text.

    Assignment goes by the centre point, so a character that overlaps a printed
    rule still lands in exactly one cell and can never be counted twice.
    Anything outside the table — the printed No column, the header, a mark in
    the margin — belongs to no cell and is dropped rather than guessed into a
    field. A cell nothing landed in stays an empty string. (5.2-1, 5.2-2)

    The boxes of a cell are joined in the order the engine gave them, which is
    its reading order, and without a separator: Japanese is written without
    spaces, so inserting one would corrupt every multi-token value. A box whose
    own text carries a line break is joined the same way, for the same reason.
    """
    grouped: dict[Cell, list[TextBox]] = {cell: [] for cell in iter_cells()}
    for box in boxes:
        cell = cell_at(*box.centre)
        if cell is None:
            logger.debug("dropped %r: it falls outside the table", box.text)
            continue
        grouped[cell].append(box)
    return {
        cell: "".join(unwrap(box.text) for box in found)
        for cell, found in grouped.items()
    }


def unsure_cells(boxes: Iterable[TextBox]) -> set[Cell]:
    """The cells the engine said it could not read cleanly.

    Placed by the same centre-point rule assign_to_cells uses, so a box is
    marked in exactly the cell its text was counted in and nowhere else. (7.2)
    """
    found = set()
    for box in boxes:
        if not box.unsure:
            continue
        cell = cell_at(*box.centre)
        if cell is not None:
            found.add(cell)
    return found


def unwrap(text: str) -> str:
    """One box's text with the paper's line breaks taken out.

    A cell the child wrote on two lines comes back from a row-returning engine
    as a single string with the break still in it, because that engine reports
    a cell rather than a stroke. The break is the paper's layout, not a
    character anybody wrote, and joining a wrapped cell without a separator is
    already what assign_to_cells does when the break arrives as two boxes
    instead.

    Left in, it is outside every range OCR_ALLOWED_SCRIPTS allows, so clean
    empties the whole cell: `いけの上に\\nとまった` reached the xlsx as nothing
    at all the first time a page came back wrapped. Two 気が付いたこと values
    went that way on one page of five. (5.2-2)
    """
    return "".join(text.splitlines())


def in_allowed_scripts(character: str) -> bool:
    """True when a character belongs to a script the sheet can be written in."""
    code = ord(character)
    return any(low <= code <= high for low, high in config.OCR_ALLOWED_SCRIPTS)


# Merged once. clean runs per cell, and 7.1's thousand sheets is 88,000 of them.
_EVERY_FOLD = {**config.OCR_CHARACTER_FOLDS, **config.OCR_CIRCLED_DIGIT_FOLDS}


def fold(field: str, text: str) -> str:
    """Rewrite the spellings config lists as the same answer, and nothing else.

    A ring drawn round a character is the sheet's way of saying "this one" and
    not part of the answer, so ④ reaches the xlsx as 4 and Ⓖ as G. The three
    columns whose whole answer is a number are the exception, and take only the
    width folds: there a circled digit is a neighbouring column's answer that
    has bled in, and folding it would make it look like a month somebody wrote.
    (3.1, 5.2-2)
    """
    folds = (
        config.OCR_CHARACTER_FOLDS
        if field in config.OCR_NUMBER_ONLY_FIELDS
        else _EVERY_FOLD
    )
    return "".join(folds.get(character, character) for character in text)


def check(field: str, text: str) -> tuple[str, str]:
    """The value this field can hold, and why it could not hold what came.

    Detection, not repair. A value is kept whole or rejected whole; characters
    are never picked out of the middle of one. Deleting the Hangul out of
    ストロ윤인していた would leave ストロしていた — a reading the engine never
    gave, that looks like an ordinary Japanese answer and would be believed by
    whoever checks the xlsx. The Hangul version at least announces that the
    cell was misread. 5.2-2 is explicit: unreadable is an empty string, never a
    guess, and a manufactured value is the worst of both.

    Four fields have a fixed character set from 3.1 — the month, the day, the
    chome number, the map letter. The rest are free Japanese handwriting, and
    there the test is the script whitelist 6.2 asks for.

    Every field folds the listed set of equivalent forms first — ７ to 7, Ⓖ to
    G, ④ to 4 — and nothing else. Unicode NFKC would have been the obvious tool
    and is the wrong one: it also turns ⁸ into 8 and ℊ into g, and it would fold
    a circled digit into the three columns whose whole answer is a number, where
    a ④ is a neighbouring column bleeding in rather than a month. See fold.

    Three more columns hold one of a printed list rather than free writing
    (OCR_FIELD_CHOICES), and those are checked as a whole value once the
    characters pass: 記号 is one of あ to け, 何してた one of ① to ⑩, マップ記号
    one of A to N. A value off the list is not a reading of that column at all,
    so it goes the same way as a character outside the set. What counts as one
    of them is names_a_choice's business, not this function's; either way the
    value that comes back is the engine's own.

    Whatever survives is either the value the engine gave, one of those listed
    folds of it, or an empty string. Never anything else.

    It comes back as (value, reason). The reason is empty when the value stood,
    and otherwise is the sentence the desktop pipeline logs when it did not -
    clean, just below, is that pipeline's signature kept over the top of this
    one. A return value rather than a log line because the caller is on another
    machine now, and logging is not a way to send anybody anything. (7.2)
    """
    candidate = fold(field, text.strip())
    allowed = config.OCR_FIELD_CHARACTERS.get(field)
    if allowed is not None:
        acceptable = all(c in allowed for c in candidate)
    else:
        acceptable = all(in_allowed_scripts(c) for c in candidate)
    if not acceptable:
        return "", f"rejected {candidate!r}, outside the field character set"
    choices = config.OCR_FIELD_CHOICES.get(field)
    if candidate and choices is not None and not names_a_choice(field, candidate):
        return "", (
            f"rejected {candidate!r}, not one of the printed answers "
            f"{' '.join(choices)}"
        )
    return candidate, ""


def clean(field: str, text: str, where: str = "") -> str:
    """The value check's answer, with the reason logged instead of returned.

    check is the whole of the logic; this is the desktop pipeline's signature
    kept over the top of it, so the value that comes back is identical on both
    sides of the split and tests/test_parity.py can compare the two directly.

    `where` is prefixed to anything logged. A rejection with no page and no row
    on it is unreadable in a batch of a thousand sheets and useless in the error
    file, and this function knows the field and nothing else, so the caller says
    where it was.
    """
    value, reason = check(field, text)
    if reason:
        # Loud on purpose: the engine read something the paper cannot contain,
        # and the run log is where that has to be visible.
        logger.warning("%s%s: %s", where, field, reason)
    return value


def names_a_choice(field: str, value: str) -> bool:
    """True when the value names one of a column's printed answers.

    A column answered off a list is checked as a whole value, and against the
    printed spellings plus the OCR_FIELD_SPELLINGS that stand for them. Letter
    case is ignored on top of that, because the handbook prints its map symbols
    as capitals and a lowercase reading of one names the same symbol.

    None of that rewrites the value: this answers whether the engine read an
    answer, not which one, and clean still returns what the engine gave. So a
    何してた read as 4 stays 4 and one read as ④ stays ④ - both are the fourth
    answer, and neither was written by this code. (5.2-2)
    """
    named = set(config.OCR_FIELD_CHOICES.get(field, ()))
    named |= set(config.OCR_FIELD_SPELLINGS.get(field, ()))
    return value.upper() in {spelling.upper() for spelling in named}


def paired_field_problems(row: dict[str, str]) -> list[str]:
    """Where a row's printed letter and its wording of the same thing disagree.

    3.1 pairs 記号 with どこで and マップ記号 with 場所の名前: the letter is the
    category, the column beside it is the child's own words for it. Read
    together they catch a misread neither one shows on its own - 場所の名前
    芝公園 against マップ記号 B is one of the two read wrongly, and no character
    set can see it.

    Reported, never repaired. Which of the two is the misread is not knowable
    from the page, so rewriting either would manufacture the value 5.2-2
    forbids. A wording that names no category at all is not a disagreement -
    most of them name none - so only a wording that clearly names a different
    one is returned. (3.1, 5.2-2)
    """
    return [message for _, _, message in _disagreements(row)]


def _disagreements(row: dict[str, str]) -> list[tuple[str, str, str]]:
    """Each disagreement as (letter column, wording column, what to say).

    The body of paired_field_problems, with the two columns kept rather than
    thrown away, so a front end can mark the cells the sentence is about
    without parsing the sentence back apart.
    """
    problems = []
    for letter_field, wording_field, words in config.OCR_PAIRED_FIELDS:
        letter = row.get(letter_field, "").upper()
        wording = row.get(wording_field, "")
        expected = words.get(letter)
        if not wording or not expected:
            continue
        if any(word in wording for word in expected):
            continue
        named = sorted(
            other
            for other, theirs in words.items()
            if any(word in wording for word in theirs)
        )
        if named:
            problems.append(
                (
                    letter_field,
                    wording_field,
                    f"row {row.get('no', '?')}: {letter_field} {letter} "
                    f"({config.OCR_FIELD_CHOICES[letter_field][letter]}) does not "
                    f"agree with {wording_field} {wording!r}, which reads like "
                    f"{'/'.join(named)}",
                )
            )
    return problems


def row_gaps(read: dict[str, str]) -> list[str]:
    """The columns the engine returned nothing for, in a row it read something in.

    Takes what the engine said, before clean has been near it, so a gap and a
    rejection stay different things: a rejected cell did come back and is
    reported as a rejection, and counting it again here would fault one cell
    twice. Every cell of a filled row is exactly one of kept, rejected, or
    this.

    A row with nothing in it at all is one of the sheet's unused rows - most
    sheets stop at three of eight - and says nothing, so it has no gaps. A row
    that is filled in except for one cell is the interesting case: that hole is
    where the engine most likely missed something, and nothing else in the
    pipeline can see it. A rejection is loud; 4 came back as an empty 何してた
    four times in one run and made no sound at all.

    A hint and not a defect. 5.2-1 makes every column but `no` optional, so a
    child who left 気が付いたこと blank produces one of these too, and it is
    still worth an operator's eye. (5.2-1, 7.2)
    """
    if not any(read.values()):
        return []
    return [field for field, _ in config.OCR_FIELDS if not read.get(field)]


# What can be wrong with one cell. MISSING and DISAGREEMENT are worked out from
# a finished row, so a page reopened tomorrow is marked exactly like one just
# read. REJECTED and UNSURE need what the engine said, which only the run that
# called it has, so they are carried on PageReport and refine MISSING where they
# are known. (7.2)
CELL_MISSING = "missing"
CELL_DISAGREEMENT = "disagreement"
CELL_REJECTED = "rejected"
CELL_UNSURE = "unsure"


def cell_problems(row: dict[str, str]) -> dict[str, str]:
    """Which cells of one finished row want an operator's eye, and why.

    Takes the row and nothing else - no engine, no cache, no run - so the same
    marks appear whether the rows came from a run just finished, from
    ocr_preview.json, or from a page an operator has been editing. A cell that
    has been corrected stops being marked because the corrected row no longer
    has the fault, which is what makes this safe to recompute on every
    keystroke. (5.2-1, 7.2)
    """
    # Only the OCR fields. row_gaps asks whether the row carries anything at
    # all, and `no` comes from the row's position on the paper rather than from
    # the child, so handing it the whole row would make every blank row look
    # like eight missing values.
    read = {field: row.get(field, "") for field, _ in config.OCR_FIELDS}
    problems = {field: CELL_MISSING for field in row_gaps(read)}
    for letter_field, wording_field, _ in _disagreements(row):
        # Both halves of the pair: which of the two is the misread is not
        # knowable, so marking one of them would be a guess. (5.2-2)
        problems[letter_field] = CELL_DISAGREEMENT
        problems[wording_field] = CELL_DISAGREEMENT
    return problems


def page_confidence(
    values: int,
    rejected: int,
    disagreements: int,
    gaps: int = 0,
    unsure: int = 0,
) -> float | None:
    """How much of what the engine returned for a page survived the checks.

    Not the engine's own confidence: neither engine reports one that means
    anything about a cell of this sheet. This is the share of the values it
    offered that are still standing after 3.1's character sets, the printed
    answer lists and the paired-column check - which is what an operator
    actually needs from a page in a batch, namely which sheet to open and look
    at and which to leave alone. (7.2)

    A gap counts against the score and into what is being scored, so the
    denominator is every cell of a filled row rather than only the ones that
    came back. Without that, a page whose only fault is four holes reads 100%
    and is skipped - which is how four missed 何してた cells survived a whole
    run unremarked. An `unsure` cell is one the engine read and said it read
    badly, which no character set can see: 花だん where the paper says
    花ばたけ, and 徃門小学校 where it says 御門小学校, are both perfectly
    ordinary Japanese.

    The faults never overlap: `values` and `gaps` add up to the cells of the
    filled rows, a rejection is one of the values, and a cell already rejected
    is not counted unsure on top. One cell, one fault.

    A page that offered no values at all has no score, and says so: None, not
    1.0. Scoring it whole is the one answer that is always wrong. Either the
    sheet was blank, or the engine read a filled sheet and returned nothing -
    the worst outcome a page has, and the page most needing to be opened - and
    nothing here can tell the two apart. A number an operator scans a batch by
    must never be at its best on the page it fails on.
    """
    cells = values + gaps
    if cells <= 0:
        return None
    faults = rejected + disagreements + gaps + unsure
    return max(0.0, 1.0 - faults / cells)


@dataclass(frozen=True)
class PageReport:
    """The per-page quality counts read_page already works out. (7.2)

    Kept as an object rather than only a log line so a front end can put the
    same numbers in a column: an operator scanning a batch for the sheet to open
    should not have to read them back out of the log. The wording of `__str__`
    is the log line, so the two never drift apart.
    """

    values: int
    rejected: int
    disagreements: int
    gaps: int
    unsure: int
    # (row number, field) -> CELL_REJECTED or CELL_UNSURE, for the cells
    # cell_problems cannot see from the finished row: a rejected value is
    # indistinguishable from a blank one once it has been emptied, and an
    # unsure reading looks like any other value.
    cells: dict[tuple[int, str], str] = dataclass_field(default_factory=dict)

    @property
    def score(self) -> float | None:
        """The page's confidence, or None when it offered no values at all."""
        return page_confidence(
            self.values, self.rejected, self.disagreements, self.gaps, self.unsure
        )

    @property
    def percent(self) -> str:
        """The score as an operator reads it. Empty when there is no score."""
        score = self.score
        return "" if score is None else f"{100 * score:.0f}%"

    def __str__(self) -> str:
        # A page that returned nothing says so in words rather than a
        # percentage: 0% would read as a bad page, and this one may be blank.
        told = self.percent or "none, the page produced no values"
        return (
            f"confidence {told} ({self.values} value(s), {self.rejected} rejected, "
            f"{self.disagreements} disagreement(s), {self.gaps} gap(s), "
            f"{self.unsure} unsure)"
        )


def page_cells(
    rows: Iterable[dict[str, str]], report: PageReport
) -> dict[Cell, str]:
    """Every cell of a finished page worth an operator's eye, and why.

    The two halves put together. cell_problems reads a finished row and finds
    what is visible in it afterwards - a hole in a filled row, a printed letter
    that disagrees with the words beside it. PageReport.cells carries the two
    that are only knowable while the page is being read: a value the checks
    threw away looks exactly like a blank one once it has been emptied, and a
    reading the engine struggled with looks exactly like a confident one.

    The report wins where the two meet, which is what its own comment asks for:
    a cell that came back and was rejected is a rejection, not a gap. (7.2)
    """
    problems: dict[Cell, str] = {}
    for row in rows:
        number = int(row["no"])
        for field, problem in cell_problems(row).items():
            problems[(number, field)] = problem
    problems.update(report.cells)
    return problems


def checks_fingerprint() -> str:
    """A short digest of the definition every value is checked against.

    Part of an engine's settings, and so part of what a client's response cache
    is keyed on. The desktop pipeline caches an engine's boxes, before any check
    has been near them, so re-tuning a character set there costs nothing and
    invalidates nothing. A client of this server caches finished rows instead,
    and a finished row is the answer to the model and the checks together - so
    widening a character set here, with nothing to say so, would go on serving
    rows read under the old one for as long as that cache lives. (3.2, 4.2, 7.2)

    Taken over sheet_definition rather than over this module, because what a
    check does is stable and what it is given is not: the columns, the printed
    answer lists, the folds and the script ranges are the part that gets edited.
    Canonical JSON with sorted keys, so the digest is a promise about content
    rather than about the order a dict happens to be built in.
    """
    canonical = json.dumps(
        config.sheet_definition(), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class PageResult:
    """One page, read: its rows, its quality counts, and what to complain about.

    `findings` is what the desktop pipeline's ocr_error.txt is made of. Each
    line names the row and the column and nothing else - `row 3: notice: ...` -
    and carries no page in front of it, because this server was handed an image
    and does not know what file it came from. Whoever writes the error file adds
    that. Getting this wrong in either direction produces a batch log of a
    thousand indistinguishable lines, which is the exact thing that file exists
    to prevent. (7.2)
    """

    rows: list[dict[str, str]]
    report: PageReport
    findings: list[str] = dataclass_field(default_factory=list)
    # What the page cost, failed attempts included. Carried here rather than
    # left on the engine, because the engine is shared by every request that
    # names its model and holds nothing that changes between calls. (6.1, 7.1)
    usage: Usage = Usage()


def read_page_image(
    image: bytes,
    engine: OCREngine,
    mime_type: str = "image/png",
    retries: int = config.OCR_RETRIES,
    backoff: float = config.OCR_BACKOFF_SECONDS,
    deadline: float | None = None,
    sleep: Callable[[float], None] | None = None,
) -> PageResult:
    """Read one cropped page into its rows and its quality counts.

    One engine call for the whole page. Always ROWS_PER_PAGE rows, filled or
    not: 5.1 allows no row to be dropped. `no` comes from the row index and is
    never OCRed, so a misread digit cannot lose a record. A blank or unreadable
    cell is an empty string, and so is anything that does not fit the field's
    character set. (5.1, 5.2-1, 5.2-2)

    If the call fails every retry this raises rather than returning half a
    sheet, so the caller can answer for that one page and leave the rest of
    their batch running. (6.4)

    Nothing is cached. The desktop pipeline keeps an engine's boxes beside the
    page they came from; this server has neither the page's identity nor
    anywhere to keep it, and the client that has both still does. What comes
    back carries the engine's settings with it, which is what lets that client
    key a cache on the reader rather than on the request. (3.2)

    Every page reports a confidence: how much of what the engine offered
    survived the checks. It is the only per-page quality number there is, and it
    is what says which sheet to go and look at. (7.2)
    """
    started = time.monotonic()
    reading = recognize_with_retry(
        engine,
        image,
        mime_type,
        retries=retries,
        backoff=backoff,
        deadline=deadline,
        sleep=sleep,
    )
    boxes = reading.boxes
    # The engine's own findings come first: they are about the response as a
    # whole, and everything after them is about one cell of it.
    findings = list(reading.notes)
    for message in findings:
        logger.warning("%s", message)

    def note(message: str) -> None:
        """Say it to the caller and to this server's log, in that order.

        Both, and not one or the other. The caller needs it because their error
        file is the only place an operator looks; this log needs it because a
        batch that went wrong is diagnosed here, and a message that only ever
        left over the wire leaves nothing behind when the client is gone.
        """
        findings.append(message)
        logger.warning("%s", message)

    values = assign_to_cells(boxes)
    unsure = unsure_cells(boxes)
    rows = []
    rejected = 0
    gaps = 0
    strained = 0
    faulty: dict[Cell, str] = {}
    for number in range(1, config.ROWS_PER_PAGE + 1):
        read = {
            field: values[(number, field)].strip() for field, _ in config.OCR_FIELDS
        }
        row = {"no": str(number)}
        for field, raw in read.items():
            row[field], reason = check(field, raw)
            # A value the engine did offer and the checks threw away. Counted
            # here rather than inside check, which answers about one value and
            # has nothing to say about the page.
            if raw and not row[field]:
                rejected += 1
                faulty[(number, field)] = CELL_REJECTED
                note(f"row {number}: {field}: {reason}")
            elif row[field] and (number, field) in unsure:
                # Only a value that survived: one the checks already threw out
                # is a rejection, and faulting the same cell twice would make
                # the score mean nothing.
                strained += 1
                faulty[(number, field)] = CELL_UNSURE
                note(
                    f"row {number}: {field}: the engine could not read "
                    f"{row[field]!r} cleanly"
                )
        rows.append(row)
        # From `read` and not from `row`, so a cell the checks emptied is a
        # rejection and not also a gap. One fault per cell.
        empty = row_gaps(read)
        gaps += len(empty)
        if empty:
            note(
                f"row {number}: nothing came back for {', '.join(empty)}, "
                "but the row is filled in"
            )
    problems = [problem for row in rows for problem in paired_field_problems(row)]
    for problem in problems:
        note(problem)
    # The whole page, not just the call: placing the boxes and checking them
    # cost time too, and 6.1 budgets the page rather than the request. Two
    # figures, so the gap between them - this server's own overhead - stays
    # visible.
    logger.info(
        "%d text box(es) in %.2fs (%s)",
        len(boxes),
        time.monotonic() - started,
        reading.usage,
    )
    offered = sum(1 for text in values.values() if text.strip())
    report = PageReport(offered, rejected, len(problems), gaps, strained, faulty)
    logger.info("%s", report)
    return PageResult(rows, report, findings, reading.usage)
