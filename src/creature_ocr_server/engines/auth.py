"""How the engines authenticate. Every credential this package reads. (6.3)

Two ways to Google, and the environment picks. They live together because
"where does a credential come from" should have one answer, and because the two
rules at the end of this docstring are the same for both.

Left empty, GOOGLE_APPLICATION_CREDENTIALS means Application Default
Credentials: the identity the machine already has, which on a workstation is
whoever ran `gcloud auth application-default login` and on a cloud host is the
attached service account.

Filled in, it names a service account instead: either a path to a key file or
the key JSON itself. For a container the path form is the one to use, with the
key mounted read-only from outside the image, so rotating it is a restart
rather than a rebuild and the key never enters a layer anybody can copy.

Both engines come through here, so the choice is one setting rather than one per
engine, and google.oauth2 is named in one place the way each vendor SDK is named
in one place. Nothing is imported until an engine is actually built. (4.2, 6.5)

The credential never leaves this module. It is not part of OCREngine.settings,
which is published by GET /v1/engines and written into a client's cache file, so
no private key can reach either. It is never logged either: what goes into the
log is the service account's email address, which is not a secret and is the one
thing worth knowing about a batch that has already been billed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .. import config

logger = logging.getLogger(__name__)

# Document AI and Vertex AI are both reached with the one platform scope. ADC
# has it applied for it; a key loaded by hand has to ask for it.
SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def _load_sdk():
    """Import the service account credential loader.

    google-auth arrives with either engine's SDK, so this imports exactly when
    an engine does. Named here only, and only when called, so the package still
    imports and the tests still run with no cloud SDK installed at all. (4.2)
    """
    from google.oauth2 import service_account

    return service_account


def _refuse(reason: str) -> str:
    """One wording for every way the setting can be unusable.

    Every one of them is the same problem to an operator - what the setting
    names is not a key this can load - and has the same two ways out, so they
    get the same sentence with the detail swapped in.
    """
    return (
        f"{config.GOOGLE_ENV_CREDENTIALS} {reason}. Correct it, or leave "
        "it empty to authenticate with application default credentials instead. "
        "See .env.example"
    )


def _key_text(setting: str) -> tuple[str, str]:
    """The key JSON the setting stands for, and what to call it in an error.

    The two forms end up in one place on purpose. A path is read here rather
    than handed to from_service_account_file so that a missing or unreadable
    file is reported against the setting, instead of surfacing out of google-auth
    several frames from anything that mentions a setting.
    """
    if setting.startswith("{"):
        return setting, "the key in the environment"
    path = Path(setting)
    if not path.is_file():
        raise ValueError(_refuse(f"points at {path}, which is not a file"))
    try:
        return path.read_text(encoding="utf-8"), str(path)
    except OSError as exc:
        raise ValueError(_refuse(f"points at {path}, which cannot be read")) from exc


def credentials(setting: str | None = None):
    """The service account the environment names, or None to let the SDK resolve ADC.

    None is not a failure: it is the default both clients already have, and it
    is what keeps the ADC path exactly as it was.

    A setting that is present but unusable raises ValueError rather than falling
    back to ADC. Falling back would authenticate as somebody else without saying
    so, read the batch under an identity nobody chose, and bill it to them.
    """
    if setting is None:
        setting = os.environ.get(config.GOOGLE_ENV_CREDENTIALS, "")
    setting = setting.strip()
    if not setting:
        # An empty value counts as unset, the same way a bare OCR_ENGINE= line
        # does in build_engine: it is a setting nobody meant to make.
        logger.info("authenticating with application default credentials")
        return None

    try:
        service_account = _load_sdk()
    except ImportError as exc:
        raise ValueError(
            "google-auth is not installed: "
            'pip install -e ".[gemini]" or pip install -e ".[documentai]"'
        ) from exc

    text, what = _key_text(setting)
    try:
        info = json.loads(text)
    except ValueError as exc:
        raise ValueError(_refuse(f"({what}) is not valid JSON")) from exc
    try:
        found = service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )
    except Exception as exc:
        # Broad on purpose. A file that is JSON but not a service account key, a
        # private key that will not parse and a field that is missing come back
        # as unrelated exception types from google-auth and from the cryptography
        # library underneath it, and every one of them means the same one thing.
        # The cause is chained, so the detail is still there for whoever wants it.
        raise ValueError(_refuse(f"({what}) is not a usable key: {exc}")) from exc

    logger.info(
        "authenticating as the service account %s", found.service_account_email
    )
    return found
