"""Who may ask this server to read a page.

The server holds a cloud credential and spends money on every request, so an
open port is an open account. One shared key in a header is the whole of it,
which is the right size for a service a desktop application on the same network
talks to; anything more belongs to whatever fronts it.

Header only, never a query string: a query string is written into every access
log and every proxy log it passes through, and a key in a log is a key. The
comparison is constant time, the key is never logged, and it is never put into
an engine's settings - which is published by GET /v1/engines and written into a
client's cache file. (6.2, 6.3)
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, status

from .. import config

logger = logging.getLogger(__name__)

HEADER = "X-API-Key"


def configured_key() -> str:
    """The key this server expects, or empty for a server that expects none."""
    return os.environ.get(config.SERVER_ENV_API_KEY, "").strip()


def announce() -> None:
    """Say once, at startup, whether anything is guarding this server.

    An open server that says nothing is the failure that actually happens: the
    key is left out of the environment by accident, everything works, and
    nobody finds out until the bill does. (6.3)
    """
    if configured_key():
        logger.info("requests must carry a %s header", HEADER)
    else:
        logger.warning(
            "no API key configured: this server will answer anyone who can "
            "reach it. Set %s to require a %s header.",
            config.SERVER_ENV_API_KEY,
            HEADER,
        )


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    """Refuse a request that does not carry the configured key.

    No key configured means no check, so a developer can start the server and
    use it without inventing one. announce() above is what stops that being a
    silent state.
    """
    expected = configured_key()
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        # The same answer either way, and nothing about the key in it.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{HEADER} is missing or wrong",
        )
