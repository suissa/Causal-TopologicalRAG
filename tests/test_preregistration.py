from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ctrag.benchmarks.protocol import (
    FROZEN_PREREGISTRATION_SHA256,
    PREREGISTRATION_RELATIVE_PATH,
    PREREGISTRATION_VERSION,
    preregistration_manifest,
)
from ctrag.benchmarks.runner import Config, run


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_frozen_preregistration_hash_matches_committed_document() -> None:
    path = _repo_root() / PREREGISTRATION_RELATIVE_PATH
    payload = path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == FROZEN_PREREGISTRATION_SHA256


def test_preregistration_manifest_is_explicit_and_versioned() -> None:
    manifest = preregistration_manifest()
    assert manifest == {
        "version": PREREGISTRATION_VERSION,
        "path": PREREGISTRATION_RELATIVE_PATH,
        "sha256": FROZEN_PREREGISTRATION_SHA256,
        "bytes": (_repo_root() / PREREGISTRATION_RELATIVE_PATH).stat().st_size,
    }


def test_benchmark_result_bundle_records_preregistration_fingerprint(tmp_path) -> None:
    report = run(Config(seeds=(7,), ks=(1,), traces=1), tmp_path)
    assert report["config"]["preregistration"]["version"] == PREREGISTRATION_VERSION
    assert report["config"]["preregistration"]["path"] == PREREGISTRATION_RELATIVE_PATH
    assert report["config"]["preregistration"]["sha256"] == FROZEN_PREREGISTRATION_SHA256


def test_modified_preregistration_is_rejected(monkeypatch, tmp_path) -> None:
    from ctrag.benchmarks import protocol

    fake_repo = tmp_path / "repo"
    doc = fake_repo / PREREGISTRATION_RELATIVE_PATH
    doc.parent.mkdir(parents=True)
    doc.write_text("modified after freeze\n", encoding="utf-8")
    monkeypatch.setattr(protocol, "_repository_root", lambda: fake_repo)

    with pytest.raises(RuntimeError, match="preregistration fingerprint changed"):
        protocol.preregistration_manifest()
