# syntax=docker/dockerfile:1

# Debian slim rather than alpine. Alpine is musl, and grpcio - which arrives
# with google-cloud-documentai the day that engine is written - has no musl
# wheels, so it would compile from source for half an hour on every build.
#
# 3.12 rather than the newest release, because google-genai is pure Python and
# will run anywhere while the Google dependency chain underneath the other
# engines is not, and because the desktop repository asks for >=3.11: this
# keeps the two in step without living on the edge of what protobuf supports.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copied by name, never `COPY . .`. There is a real service account key in the
# sibling checkout and a .env beside it, and the surest way to keep a
# credential out of a layer is never to offer the build one. .dockerignore is
# the second defence, not the only one. (6.2, 6.3)
COPY pyproject.toml ./
COPY src ./src

# [documentai,gemini]. All three engines are implemented and two are installed.
# documentai brings grpcio and protobuf with it - some 25 MB on top of a 117 MB
# site-packages, most of that grpcio - for an engine that stays unusable until
# its region and processor id are set. That is paid on purpose: pointing the
# server at a processor is then a setting rather than a rebuild. Until it is
# configured GET /v1/engines reports it not ready and says which setting is
# missing, which is what the extras pattern is for. (4.2)
#
# nemotron needs nothing installed to read a page, so this image can serve it
# as it stands: its extra is pymupdf, wanted only where NEMOTRON_MAX_BYTES asks
# for a page to be re-encoded smaller.

# The source goes away with the build that consumed it. What runs is the copy
# in site-packages, and a second copy in /app that is not the one running is a
# trap for whoever next opens a shell in here to work out what is going on.
RUN pip install --no-cache-dir ".[documentai,gemini]" \
 && rm -rf /app/src /app/build /app/pyproject.toml

# Nothing here needs to write anywhere. Run the container with a read-only root
# filesystem and a tmpfs on /tmp as compose.yaml does, so an upload that
# somehow escaped the in-memory limit lands in RAM rather than on a disk. (6.2)
RUN useradd --create-home --uid 10001 reader
USER reader

EXPOSE 8000

# python rather than curl, which slim does not have. It calls /v1/health, which
# touches no engine on purpose: a healthcheck that queued behind a four-minute
# page would restart the process in the middle of one. (6.1)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD \
    ["python", "-c", "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=3).status == 200 else 1)"]

# One worker. Engines are built once per model and kept in the process, so a
# second worker means a second set of SDK clients and a second credential
# refresh for work that is entirely spent waiting on Vertex. Scale with
# SERVER_MAX_CONCURRENCY, or with more containers if the quota allows. (6.1)
CMD ["uvicorn", "creature_ocr_server.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
