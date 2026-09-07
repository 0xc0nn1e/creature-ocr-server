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

    uvicorn.run(
        "creature_ocr_server.api.app:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
