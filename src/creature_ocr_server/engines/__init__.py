"""OCR engine implementations, and the one place an engine is chosen.

One module per backend. A module here is the only place its vendor SDK is
imported, so the rest of the package runs with none of them installed, the test
suite passes on a machine that has never had a cloud account, and adding a
backend is one file plus one entry below. (4.2, 6.5)

Importing the modules below is therefore free: each names its SDK inside its own
loader, which nothing calls until an engine is actually built. That is what lets
this registry hold every engine without making every SDK a dependency. nemotron
has no SDK to name - it is one HTTP request - so it costs nothing either way,
and the one library it can want is named inside the function that wants it.

Two things here that the desktop pipeline's registry does not have, and both
come from being a server:

`status` answers whether an engine could be built without building it. The
desktop version finds out by constructing one and catching the failure, which
costs a key parse and, on a first call, an OAuth round trip - acceptable once at
the start of a run, wrong for an endpoint a client polls.

`shared` keeps one engine per model rather than building one per request.
Building loads a credential and constructs an SDK client, and a fresh credential
has no cached token, so per-request construction would pay a token exchange for
every page. What makes the sharing safe is that engines hold nothing that
changes between calls: recognize returns what it cost instead of recording it.
"""

from __future__ import annotations

import importlib.util
import os
import threading

from .. import config
from ..ocr import OCREngine
from .documentai import DocumentAIEngine
from .gemini import GeminiEngine
from .nemotron import NemotronEngine

ENGINES: dict[str, type[OCREngine]] = {
    DocumentAIEngine.name: DocumentAIEngine,
    GeminiEngine.name: GeminiEngine,
    NemotronEngine.name: NemotronEngine,
}

# Built engines, keyed by (engine, model, sheet). Bounded by each engine's own
# model allow-list, by the two sheets config names, and by nothing else - which
# is the second reason that allow-list exists: without it a caller could name any
# string and grow this dictionary an SDK client at a time until the process ran
# out of memory.
#
# The sheet is part of the key because it is part of what an engine is: the
# prompt, the response schema and the frame all differ, and so does the settings
# string a client keys its cache on. One engine per (model, sheet) keeps the
# invariant that an engine holds nothing that changes between calls. (3.2, 4.2)
_BUILT: dict[tuple[str, str, str], OCREngine] = {}

# Held while an engine is built, so two cold requests for the same model do not
# each construct a client and each pay for a token exchange. Not held while a
# page is read: that is the forty seconds this server exists to spend, and
# serialising it would make the concurrency limit meaningless.
_LOCK = threading.Lock()


def resolve(name: str | None = None) -> str:
    """The engine asked for, or the one the environment names, or the default.

    The single place an engine is chosen, so switching one stays a setting
    rather than an edit, and so every endpoint chooses the same way. (4.2, 6.5)

    An empty environment value counts as unset: a bare OCR_ENGINE= line is a
    setting nobody meant to make.
    """
    chosen = name or os.environ.get(config.OCR_ENGINE_ENV) or config.DEFAULT_OCR_ENGINE
    if chosen not in ENGINES:
        known = ", ".join(sorted(ENGINES))
        raise ValueError(f"unknown OCR engine {chosen!r}: expected one of {known}")
    return chosen


def _installed(sdk: str) -> bool:
    """Whether the vendor module an engine names is importable.

    The whole dotted name, never its first segment. `google` is a namespace
    package that google-auth alone puts on the path, so asking about that
    segment would call google-cloud-documentai installed anywhere [gemini] is
    installed without it - a base install, or a plain `.[gemini]` - and GET
    /v1/engines would offer an engine that cannot be built. Only the parent
    packages are imported, which for a namespace package is no code at all, so
    this stays the pure check status promises. (4.2)

    A missing parent raises rather than answering None, which is the same
    answer: the module is not there.
    """
    try:
        return importlib.util.find_spec(sdk) is not None
    except ModuleNotFoundError:
        return False


def status(name: str) -> tuple[bool, str]:
    """Whether this engine could be built, and if not, why not.

    A pure check. It reads the environment and asks whether a module is
    importable; it loads no credential, constructs no client and sends nothing.
    An endpoint that reported readiness by trying to build every engine would
    make an informational request cost a key parse and a token exchange per
    backend, for backends nobody is using. The sentence it gives back is the
    one the constructor would have raised. (4.2, 6.3)
    """
    engine = ENGINES[name]
    for key in engine.requires:
        if not os.environ.get(key, "").strip():
            return False, f"{key} is not set: see .env.example"
    if engine.sdk and not _installed(engine.sdk):
        return False, (
            f"{engine.sdk} is not installed: "
            f'pip install -e ".[{engine.extra}]"'
        )
    return True, ""


def choose_model(name: str, model: str | None = None) -> str:
    """The model this request will be served with, or a ValueError saying why.

    An engine whose reader is not a model at all - a Document AI processor, a
    NIM deployment - offers none, and asking it for one is a mistake rather
    than a setting to be quietly dropped: a box that silently does nothing is
    worse than one that says it has nothing to offer. (6.5)

    A model that is not on the list is refused rather than tried. That is what
    stops an unknown string from becoming another SDK client kept forever, and
    what keeps the settings string GET /v1/engines published true of the engine
    that actually reads the page. (3.2, 4.2)
    """
    offered = ENGINES[name].models()
    if not model:
        return ""
    if not offered:
        raise ValueError(f"{name} takes no model: it is not chosen by model id")
    if model not in offered:
        raise ValueError(
            f"unknown model {model!r} for {name}: expected one of "
            f"{', '.join(offered)}"
        )
    return model


def choose_sheet(sheet: str | None = None) -> str:
    """The sheet this request is about, or a ValueError naming the two there are.

    A separate function for the same reason resolve() is one: which paper a
    request means is decided in one place, so /v1/ocr and /v2/ocr cannot come to
    disagree about what the word means.
    """
    chosen = sheet or config.SHEET_V1
    if chosen not in config.SHEETS:
        raise ValueError(
            f"unknown sheet {chosen!r}: expected one of {', '.join(config.SHEETS)}"
        )
    return chosen


def build_engine(
    name: str | None = None,
    model: str | None = None,
    sheet: str | None = None,
) -> OCREngine:
    """Build one engine. The model is an argument, never an environment write.

    The desktop pipeline sets GEMINI_MODEL and lets the constructor read it
    back, with a comment explaining that this is where every engine reads its
    settings from. That is correct for one run on one machine and would be a
    race here: a process-global write under a threadpool can hand one request's
    model to another request's engine, which then labels its readings with the
    wrong cache name and poisons a client's cache with a different model's
    answers. So it is passed in, and an engine with no model concept is refused
    one above rather than sent something it ignores. (4.2)
    """
    chosen = resolve(name)
    wanted = choose_model(chosen, model)
    paper = choose_sheet(sheet)
    engine = ENGINES[chosen]
    arguments = {}
    if wanted:
        arguments["model"] = wanted
    # The sheet is passed only when it is not v1, so a v1 engine is built by
    # exactly the call that always built it. An engine written before there was
    # a second sheet therefore keeps serving the first one untouched, and only a
    # v2 request reaches it with a word it does not know.
    if paper != config.SHEET_V1:
        arguments["sheet"] = paper
    return engine(**arguments)


def shared(
    name: str | None = None,
    model: str | None = None,
    sheet: str | None = None,
) -> OCREngine:
    """The engine for this name and model, built once and kept.

    Per-request construction would also work, and is the escape hatch if this
    ever becomes awkward: building the prompt and the two fingerprints is sub
    millisecond, and parsing a key is a few milliseconds. What it would cost is
    an OAuth token exchange per page, because a freshly loaded credential has
    no cached token - under one percent of a forty-second call, and still a
    round trip paid for nothing.
    """
    chosen = resolve(name)
    wanted = choose_model(chosen, model)
    paper = choose_sheet(sheet)
    key = (chosen, wanted, paper)
    found = _BUILT.get(key)
    if found is not None:
        return found
    with _LOCK:
        # Checked again inside the lock: two cold requests for the same model
        # both arrive here, and only one of them should build a client.
        found = _BUILT.get(key)
        if found is None:
            found = build_engine(chosen, wanted or None, paper)
            _BUILT[key] = found
        return found


def forget() -> None:
    """Drop every built engine. For a test, and for a settings reload."""
    with _LOCK:
        _BUILT.clear()
