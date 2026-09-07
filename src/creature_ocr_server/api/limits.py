"""What a request may weigh, and what it must actually be.

Three small things that all have to happen outside the route handler, because
by the time a handler runs the body has already been read and parsed.
"""

from __future__ import annotations

import os

from starlette.formparsers import MultiPartParser

from .. import config


def _number(key: str, fallback: float) -> float:
    """A setting from the environment, or the value that ships. Never a crash.

    A server that refuses to start because someone typed a word into a numeric
    setting is a server that is down for a typo; the shipped value is right for
    almost everybody, so an unreadable one falls back to it.
    """
    raw = os.environ.get(key, "").strip()
    if not raw:
        return fallback
    try:
        return type(fallback)(raw)
    except ValueError:
        return fallback


def max_image_bytes() -> int:
    """The largest page this server will accept."""
    return int(
        _number(config.SERVER_ENV_MAX_IMAGE_BYTES, config.SERVER_MAX_IMAGE_BYTES)
    )


def max_concurrency() -> int:
    """How many engine calls may be in flight at once. (6.1)"""
    wanted = _number(
        config.SERVER_ENV_MAX_CONCURRENCY, config.SERVER_MAX_CONCURRENCY
    )
    return max(1, int(wanted))


def request_deadline() -> float:
    """The whole budget for one page, in seconds. (6.1, 6.4)"""
    return _number(
        config.SERVER_ENV_REQUEST_DEADLINE, config.SERVER_REQUEST_DEADLINE_SECONDS
    )


def queue_timeout() -> float:
    """How long a request waits for a slot before it is turned away."""
    return _number(
        config.SERVER_ENV_QUEUE_TIMEOUT, config.SERVER_QUEUE_TIMEOUT_SECONDS
    )


def keep_uploads_in_memory() -> None:
    """Stop Starlette spilling a survey page onto this container's disk.

    MultiPartParser spools a part larger than one megabyte into a real
    temporary file, and a 300 DPI page of this sheet is half a megabyte to one
    and a half. These are personal-information-removed pages so it is not a
    breach, but "the OCR server writes survey sheet images to disk" is a
    sentence nobody should have to defend when avoiding it is one assignment.
    (6.2, 7.3)

    Called at import of the app rather than here, so that reading this module
    has no side effect. The container also runs with a read-only root
    filesystem and a tmpfs on /tmp, so a mistake lands in memory anyway - two
    defences, because this one is a private attribute of somebody else's
    library.
    """
    if hasattr(MultiPartParser, "spool_max_size"):
        MultiPartParser.spool_max_size = max_image_bytes() + 1


def looks_like_png(image: bytes) -> bool:
    """Whether these bytes are a PNG, whatever the client called them.

    The declared content type is a claim; this is a fact. It matters because
    the type is passed on to the engine, and telling a model that a JPEG is a
    PNG is telling it something untrue about what it is looking at. Stage 1
    only ever writes PNG, so anything else is a mistake worth naming rather
    than a format to support. (5.2-3, 6.2)
    """
    return image.startswith(config.PNG_MAGIC)
