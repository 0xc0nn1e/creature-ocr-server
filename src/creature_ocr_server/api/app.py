"""The endpoints.

Four of them, and only one costs anything. /v1/health touches no engine so a
container healthcheck cannot kill a process in the middle of a four-minute
page. /v1/engines and /v1/sheet are pure reads of configuration, which is what
lets a client arrive with nothing configured and find out what it can ask for.
/v1/ocr is the one that spends money.

The reading itself is a blocking SDK call, so the route is a plain `def` and
FastAPI runs it in a worker thread. Two things guard it. A semaphore caps how
many are in flight, because the binding constraint is Vertex quota rather than
this machine, and lowering anyio's own thread limiter instead would put
/v1/health in the same queue. And every request carries a deadline: a page
measures 18 to 68 seconds and 6.4 allows three retries, so a genuinely bad one
can legitimately run for about 280 seconds, and something has to own that
number or every client and proxy in front of this server will pick its own.
(6.1, 6.4)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .. import __version__, config, engines, ocr
from ..ocr import OCRError, OCRTimeout
from . import limits, security
from .schemas import (
    CellProblem,
    EngineInfo,
    Health,
    ModelInfo,
    PageResponse,
    ReportInfo,
    SheetInfo,
    UsageInfo,
)

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# A survey page must not reach this container's filesystem. See the note on
# keep_uploads_in_memory; done at import so it is in force before the first
# request rather than after the first startup event.
limits.keep_uploads_in_memory()

# How many engine calls may be in flight. Built lazily on the running loop the
# first time it is wanted, so importing this module starts nothing.
_slots: asyncio.Semaphore | None = None


def configure_logging() -> None:
    """Put this package's log where whoever runs the container can read it.

    uvicorn configures its own loggers and nothing else, so without this every
    line this server has to say about a batch - the per-page confidence, the
    findings, the request id a 502 has to be matched by - goes nowhere at all,
    and a warning escapes only through logging's last-resort handler, with no
    timestamp and no logger name on it. That is not a cosmetic loss: the
    vendor's own error message is deliberately kept out of the response body,
    so this log is the only place it exists. (6.2, 7.2)

    The level is left alone if something has already set it, so a test that
    silences the package stays silent through a startup.
    """
    package = logging.getLogger(__name__.split(".")[0])
    if any(getattr(handler, "_ours", False) for handler in package.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler._ours = True
    package.addHandler(handler)
    if package.level == logging.NOTSET:
        wanted = os.environ.get(config.SERVER_ENV_LOG_LEVEL, "").strip().upper()
        package.setLevel(wanted or logging.INFO)
    package.propagate = False


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Say what this server is, and warm what it can before anyone asks."""
    global _slots
    _slots = asyncio.Semaphore(limits.max_concurrency())
    configure_logging()
    security.announce()
    logger.info(
        "sheet %s, up to %d call(s) in flight, %.0fs per page",
        ocr.checks_fingerprint(),
        limits.max_concurrency(),
        limits.request_deadline(),
    )
    name = engines.resolve()
    ready, detail = engines.status(name)
    if ready:
        # One credential fetch now rather than a race between the first few
        # requests: google.auth refreshes a token without a lock, and a warm
        # credential makes that a non-question rather than a rare one. A
        # failure here is logged and nothing more - a server that will not
        # start because a cloud is having a bad morning is worse than one that
        # answers /v1/health and says why /v1/ocr is failing.
        try:
            await run_in_threadpool(engines.shared, name)
            logger.info("%s is ready", name)
        except Exception as exc:
            logger.warning("%s could not be built yet: %s", name, exc)
    else:
        logger.warning("%s is not usable: %s", name, detail)
    yield
    engines.forget()


app = FastAPI(
    title="creature-ocr-server",
    version=__version__,
    summary="Reads one cropped survey page and returns its rows.",
    lifespan=lifespan,
)


@app.middleware("http")
async def tag_and_weigh(request: Request, call_next):
    """Give every request an id, and refuse a body nobody meant to send.

    Both have to happen here. The id has to exist before anything can log
    against it, and the size has to be checked before the body is read, because
    by the time a route runs Starlette has already read and parsed it. Uvicorn
    imposes no body limit of its own.

    A chunked request declares no length, so this cannot be the only check; the
    route weighs what it actually received as well.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id
    declared = request.headers.get("content-length")
    limit = limits.max_image_bytes()
    if declared and declared.isdigit() and int(declared) > limit:
        return JSONResponse(
            status_code=413,
            content={"detail": f"the request body is larger than {limit} bytes"},
            headers={REQUEST_ID_HEADER: request_id},
        )
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@contextlib.asynccontextmanager
async def a_slot(request_id: str):
    """Wait for a place in the queue, or say so rather than waiting forever.

    Acquired before the work is handed to a thread, so the cap is on calls to
    the engine and not on threads. A request that cannot get in within the
    queue timeout is told to come back: waiting is right, and waiting five
    minutes and concluding the server has hung is not.
    """
    waited = limits.queue_timeout()
    try:
        await asyncio.wait_for(_slots.acquire(), timeout=waited)
    except (TimeoutError, asyncio.TimeoutError):
        logger.warning("%s: no free slot after %.0fs", request_id, waited)
        raise HTTPException(
            status_code=429,
            detail="the server is busy; retry shortly",
            headers={"Retry-After": str(int(waited))},
        ) from None
    try:
        yield
    finally:
        _slots.release()


@app.get("/v1/health", response_model=Health)
async def health() -> Health:
    """Alive. Deliberately answers on a server with nothing configured.

    It touches no engine, reads no credential and takes no slot, so a container
    healthcheck cannot be starved by a queue of pages or kill a process in the
    middle of one. What is or is not usable is /v1/engines' question.
    """
    return Health(status="ok", version=__version__)


@app.get(
    "/v1/engines",
    response_model=list[EngineInfo],
    dependencies=[Depends(security.require_key)],
)
async def list_engines() -> list[EngineInfo]:
    """What this server can read a page with, and how to file what it reads.

    A pure read of configuration: nothing here builds an engine, loads a
    credential or opens a socket, so a client may call it at startup without
    paying for the privilege.

    The settings string per model is the load-bearing part. A client caches
    finished rows now, and it looks in that cache before it asks for a page, so
    it has to be able to key the cache without having asked - which it cannot
    do for itself, because the prompt, the grid and the value checks are all
    here. (3.2, 4.2)
    """
    found = []
    for name in sorted(engines.ENGINES):
        engine = engines.ENGINES[name]
        ready, detail = engines.status(name)
        found.append(
            EngineInfo(
                name=name,
                ready=ready,
                detail=detail,
                default_model=engine.default_model(),
                models=[
                    ModelInfo(
                        name=model,
                        settings=engine.settings_for(model),
                        cache_name=engine.cache_name_for(model),
                    )
                    for model in engine.models()
                ],
            )
        )
    return found


@app.get(
    "/v1/sheet",
    response_model=SheetInfo,
    dependencies=[Depends(security.require_key)],
)
async def sheet() -> SheetInfo:
    """The paper, as the value checks see it.

    Published so both halves of the split pipeline can describe the same sheet
    from one source rather than from two copies that drift. The fingerprint is
    for noticing that they have; it is not a gate, because a server-side edit
    that stopped every desktop install in the field would be the worse failure.
    (6.5)
    """
    return SheetInfo(
        fingerprint=ocr.checks_fingerprint(), sheet=config.sheet_definition()
    )


@app.post(
    "/v1/ocr",
    response_model=PageResponse,
    dependencies=[Depends(security.require_key)],
)
async def read_page(
    request: Request,
    image: UploadFile = File(description="One cropped page, PNG."),
    engine: str | None = Form(default=None, description="Which backend reads it."),
    model: str | None = Form(default=None, description="Which model, if it has one."),
) -> PageResponse:
    """Read one cropped page and return its rows.

    Async, so the concurrency slot is taken on the event loop and the blocking
    engine call is handed to a worker thread inside it. Capping threads instead
    would be a global setting and would put /v1/health in the same queue.

    The image is expected to have had its personal-information band cut off
    already: that is stage 1's job and it happens on the machine that holds the
    scan. Nothing here can check that, and nothing here should pretend to. (6.2)
    """
    request_id = request.state.request_id
    data = await image.read()
    limit = limits.max_image_bytes()
    if len(data) > limit:
        # The middleware catches a declared length; this catches a chunked
        # request, which declares none.
        raise HTTPException(413, f"the page is larger than {limit} bytes")
    if not limits.looks_like_png(data):
        raise HTTPException(415, "the page must be a PNG")

    try:
        name = engines.resolve(engine)
        chosen = engines.choose_model(name, model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None

    ready, detail = engines.status(name)
    if not ready:
        raise HTTPException(503, detail)
    try:
        built = await run_in_threadpool(engines.shared, name, chosen or None)
    except ValueError as exc:
        # A setting that passed the pure check and still could not be used - an
        # unreadable key, most likely. It names a setting and no secret.
        raise HTTPException(503, str(exc)) from None

    logger.info(
        "%s: reading a %d byte page with %s%s",
        request_id,
        len(data),
        name,
        f" ({chosen})" if chosen else "",
    )
    started = time.monotonic()
    async with a_slot(request_id):
        try:
            result = await run_in_threadpool(
                ocr.read_page_image,
                data,
                built,
                "image/png",
                deadline=started + limits.request_deadline(),
            )
        except OCRTimeout as exc:
            logger.error("%s: %s", request_id, exc)
            raise HTTPException(
                504, f"the page did not finish in time (request {request_id})"
            ) from None
        except OCRError as exc:
            # The vendor's own message stays in this log. It routinely carries
            # the project id and sometimes the service account address, and the
            # rule that keeps a credential out of settings and out of a log
            # applies to an error body too. (6.2, 6.3)
            logger.error("%s: %s", request_id, exc)
            raise HTTPException(
                502, f"{name} could not read this page (request {request_id})"
            ) from None

    logger.info(
        "%s: %s in %.2fs (%s)",
        request_id,
        result.report,
        time.monotonic() - started,
        result.usage,
    )
    marks = ocr.page_cells(result.rows, result.report)
    return PageResponse(
        request_id=request_id,
        engine=name,
        model=chosen,
        settings=built.settings,
        cache_name=built.cache_name or name,
        sheet_fingerprint=ocr.checks_fingerprint(),
        rows=result.rows,
        report=ReportInfo(
            values=result.report.values,
            rejected=result.report.rejected,
            disagreements=result.report.disagreements,
            gaps=result.report.gaps,
            unsure=result.report.unsure,
            score=result.report.score,
            percent=result.report.percent,
            cells=[
                CellProblem(row=row, field=field, problem=problem)
                for (row, field), problem in sorted(marks.items())
            ],
        ),
        findings=result.findings,
        usage=UsageInfo(
            calls=result.usage.calls,
            seconds=result.usage.seconds,
            prompt_tokens=result.usage.prompt_tokens,
            output_tokens=result.usage.output_tokens,
            thought_tokens=result.usage.thought_tokens,
        ),
    )
