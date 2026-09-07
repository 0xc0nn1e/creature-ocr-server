"""Run the server: python -m creature_ocr_server.

One worker, deliberately. Engines are built once per model and kept in the
process, so a second worker means a second set of SDK clients and a second
credential refresh for work that is entirely spent waiting on Vertex. Scale by
raising SERVER_MAX_CONCURRENCY, or by running more containers if the quota
allows it. (6.1)
"""

from __future__ import annotations

import os


def main() -> int:
    import uvicorn

    # An empty value counts as unset, the same rule engines.resolve and
    # auth.credentials state: a bare HOST= line in a .env copied from the
    # template is a setting nobody meant to make. get() only answers with its
    # default when the name is absent, so all three are read and then fallen
    # back on, which is not a nicety here - empty LOG_LEVEL and empty PORT
    # both stop the server from starting at all, and an empty HOST is worse
    # than that: it binds every interface rather than the loopback address
    # this line documents, on a server that holds a cloud credential.
    uvicorn.run(
        "creature_ocr_server.api.app:app",
        host=os.environ.get("HOST", "").strip() or "127.0.0.1",
        port=int(os.environ.get("PORT", "").strip() or "8000"),
        log_level=os.environ.get("LOG_LEVEL", "").strip().lower() or "info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
