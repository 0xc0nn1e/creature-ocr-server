"""The wire contract, written down so it cannot drift by accident.

Two shapes here are pinned rather than chosen, and both would be easy to
"improve" into a bug:

`rows[].no` is a **string**. It is a string in the desktop pipeline, where it
comes from the row's position on the paper and is written straight into the
xlsx, and a helpful JSON round trip that made it an integer would be a silent
behaviour change in a client nobody edited.

`report.cells[].row` is an **integer**, and is deliberately not called `no` for
that reason: two fields with one name and two types is a mistake somebody makes
exactly once, quietly.

`usage` carries the field names of ocr.Usage exactly, so a client can do
Usage(**usage) and keep adding pages up with the code it already has.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Health(BaseModel):
    """Liveness. It touches no engine and needs no configuration to answer."""

    status: str
    version: str


class ModelInfo(BaseModel):
    """One model an engine will answer for, and how to file what it reads.

    `settings` is the whole point of this endpoint. A client used to build that
    string itself, but the prompt, the grid and the value checks that go into it
    are this server's now, so it has to be told - and it has to be told before
    it asks for a page, because that is when it looks in its cache. (3.2, 4.2)
    """

    name: str
    settings: str
    cache_name: str


class EngineInfo(BaseModel):
    """One backend: whether it can be used, and what it will answer for.

    `settings` and `cache_name` are filled only for an engine whose reader is
    not a model - a Document AI processor, a NIM deployment. Those engines list
    no models, so without these two fields their cache key existed nowhere a
    client could read it before asking, and a client that cannot build the key
    cannot look in its cache first. That is the one thing GET /v1/engines is
    for. (3.2, 4.2)

    They stay empty for an engine that does list models, because there the key
    belongs to the model and ModelInfo already carries it. Filling them from a
    blank model id would publish a string describing a reader that does not
    exist, which is worse than publishing nothing.
    """

    name: str
    ready: bool
    detail: str = ""
    default_model: str = ""
    settings: str = ""
    cache_name: str = ""
    models: list[ModelInfo] = Field(default_factory=list)


class SheetInfo(BaseModel):
    """The paper, as the value checks see it, plus a digest of it.

    Published so both halves of the split pipeline can describe the same sheet
    without either one guessing. The fingerprint is for noticing drift, and for
    nothing else: a client that refused to run on a mismatch would let one
    setting here stop every desktop install in the field. (6.5)
    """

    fingerprint: str
    sheet: dict


class CellProblem(BaseModel):
    """One cell worth an operator's eye, and what is wrong with it. (7.2)"""

    row: int
    field: str
    problem: str


class ReportInfo(BaseModel):
    """How much of what the engine offered survived the checks. (7.2)"""

    values: int
    rejected: int
    disagreements: int
    gaps: int
    unsure: int
    # None when the page offered no values at all. Not 1.0: either the sheet
    # was blank or the engine read a filled sheet and returned nothing, and a
    # number an operator scans a batch by must never be at its best on the page
    # it failed on.
    score: float | None = None
    percent: str = ""
    cells: list[CellProblem] = Field(default_factory=list)


class UsageInfo(BaseModel):
    """What the request cost, failed attempts included. (6.1, 7.1)"""

    calls: int
    seconds: float
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int


class PageResponse(BaseModel):
    """One page, read.

    `findings` is what a client writes into its own error file. Each line names
    the row and the column and carries no page in front of it, because this
    server was handed an image and does not know what file it came from - the
    client prepends that. Do not "fix" this at either end without fixing the
    other: a batch of a thousand sheets whose findings all read the same is the
    exact failure that file exists to prevent. (7.2)
    """

    request_id: str
    engine: str
    model: str = ""
    settings: str
    cache_name: str
    sheet_fingerprint: str
    rows: list[dict[str, str]]
    report: ReportInfo
    findings: list[str] = Field(default_factory=list)
    usage: UsageInfo
