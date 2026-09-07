"""Who may ask this server to read a page.

The server holds a cloud credential and spends money on every request, so an
open port is an open account. One shared key in a header is the whole of it,
which is the right size for a service a desktop application on the same network
talks to; anything more belongs to whatever fronts it.

Two headers carry it, `X-API-Key` and `Authorization: Bearer`, and they carry
the same one key. Bearer is what an HTTP client library, a gateway and a
generated SDK all reach for by default; the plain header is what a curl in a
README wants. Neither is a different credential, so there is still one secret
and one setting to rotate.

Header only, never a query string: a query string is written into every access
log and every proxy log it passes through, and a key in a log is a key. The
comparison is constant time and is made over bytes rather than text, because a
header arrives as whatever the client sent and hmac.compare_digest refuses a
str that is not ASCII - which would turn a wrong key into a 500 that an
unauthenticated request could ask for at will. The key is never logged, and it
is never put into an engine's settings - which is published by GET /v1/engines
and written into a client's cache file. (6.2, 6.3)
"""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import Header, HTTPException, status

from .. import config

logger = logging.getLogger(__name__)

HEADER = "X-API-Key"
BEARER = "Authorization"
SCHEME = "Bearer"


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


def _bearer(header: str | None) -> str | None:
    """The token out of an Authorization header, or None if it holds no Bearer.

    The scheme is compared without case, which RFC 7235 requires of it, and the
    token is taken as the rest of the line: a key is not a place to be clever
    about whitespace, and something that is not a Bearer at all is not a
    failed key but a header this server has no opinion about.
    """
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != SCHEME.lower():
        return None
    return token.strip()


def _matches(offered: str | None, expected: str) -> bool:
    """Whether an offered key is the configured one, in constant time.

    Over bytes, because a header is whatever the client sent: Starlette decodes
    one as latin-1, so any byte above 0x7f arrives as a str that
    hmac.compare_digest refuses outright. Comparing text there turns a wrong
    key into a 500, which is a crash an unauthenticated request can ask for
    whenever it likes. (6.2)
    """
    if offered is None:
        return False
    return hmac.compare_digest(offered.encode("utf-8"), expected.encode("utf-8"))


def require_key(
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> None:
    """Refuse a request that does not carry the configured key.

    Either header will do and both hold the same key, so a client picks the one
    its HTTP library makes easy rather than the one this server preferred.

    No key configured means no check, so a developer can start the server and
    use it without inventing one. announce() above is what stops that being a
    silent state.
    """
    expected = configured_key()
    if not expected:
        return
    # Both are weighed, never one and then the other on a condition, so what
    # comes back says nothing about which header was tried.
    offered = (_matches(x_api_key, expected), _matches(_bearer(authorization), expected))
    if not any(offered):
        # The same answer to every way of getting it wrong, and nothing about
        # the key in it. WWW-Authenticate because a 401 that accepts Bearer is
        # supposed to say so; it names the scheme, never the secret.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"{HEADER} or {BEARER}: {SCHEME} is missing or wrong",
            headers={"WWW-Authenticate": SCHEME},
        )
