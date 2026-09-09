from __future__ import annotations

import hashlib
from pathlib import Path

PREREGISTRATION_VERSION = 1
PREREGISTRATION_RELATIVE_PATH = "docs/PREREGISTRATION.md"


def _repository_root() -> Path:
    # src/ctrag/benchmarks/protocol.py -> repository root
    return Path(__file__).resolve().parents[3]


def preregistration_manifest() -> dict[str, object]:
    """Return the exact preregistration identity bound to an experiment run.

    Confirmatory benchmark runs are intentionally tied to the checked-out
    preregistration document. If the document is unavailable, the benchmark
    must fail instead of silently producing an unregistered result bundle.
    """
    path = _repository_root() / PREREGISTRATION_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"preregistration document not found: {path}; "
            "confirmatory benchmark runs require the frozen protocol"
        )
    payload = path.read_bytes()
    return {
        "version": PREREGISTRATION_VERSION,
        "path": PREREGISTRATION_RELATIVE_PATH,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
