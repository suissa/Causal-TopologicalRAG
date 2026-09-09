from __future__ import annotations

import hashlib
from pathlib import Path

PREREGISTRATION_VERSION = 1
PREREGISTRATION_RELATIVE_PATH = "docs/PREREGISTRATION.md"
FROZEN_PREREGISTRATION_SHA256 = "3e3270eb987a7b5ae403a287c1ae9de0edd6e684e2f92044d6eef88dac3cd467"


def _repository_root() -> Path:
    # src/ctrag/benchmarks/protocol.py -> repository root
    return Path(__file__).resolve().parents[3]


def preregistration_manifest() -> dict[str, object]:
    """Return the exact preregistration identity bound to an experiment run.

    Confirmatory benchmark runs are intentionally tied to the checked-out
    preregistration document. If the document is unavailable or has changed
    without a protocol-version update, the benchmark must fail instead of
    silently producing a result bundle under a different scientific protocol.
    """
    path = _repository_root() / PREREGISTRATION_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"preregistration document not found: {path}; "
            "confirmatory benchmark runs require the frozen protocol"
        )
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FROZEN_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "preregistration fingerprint changed; classify the change as a "
            "protocol amendment, bump PREREGISTRATION_VERSION and freeze a new hash "
            f"before confirmatory runs (expected {FROZEN_PREREGISTRATION_SHA256}, got {digest})"
        )
    return {
        "version": PREREGISTRATION_VERSION,
        "path": PREREGISTRATION_RELATIVE_PATH,
        "sha256": digest,
        "bytes": len(payload),
    }
