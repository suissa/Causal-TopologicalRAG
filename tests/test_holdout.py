from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ctrag.benchmarks import holdout


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_holdout_spec_is_frozen_and_splits_do_not_overlap() -> None:
    spec_path = _repo_root() / holdout.HOLDOUT_SPEC_RELATIVE_PATH
    assert hashlib.sha256(spec_path.read_bytes()).hexdigest() == holdout.HOLDOUT_SPEC_SHA256

    train = set(holdout.split_seeds("train"))
    dev = set(holdout.split_seeds("dev"))
    test = set(holdout.split_seeds("test"))
    assert train
    assert dev
    assert test
    assert train.isdisjoint(dev)
    assert train.isdisjoint(test)
    assert dev.isdisjoint(test)


def test_final_test_dataset_fingerprint_is_frozen_without_retrieval() -> None:
    assert holdout.split_dataset_sha256("test") == holdout.FROZEN_TEST_DATASET_SHA256


def test_final_test_is_sealed_by_default(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="final test is sealed"):
        holdout.run_split("test", tmp_path / "sealed", ks=(1,))


def test_development_split_rejects_unblind_flags(tmp_path) -> None:
    with pytest.raises(ValueError, match="valid only for the test split"):
        holdout.run_split("dev", tmp_path / "dev", ks=(1,), unblind_final_test=True)


def test_dev_run_writes_fingerprinted_bundle_without_test_results(tmp_path) -> None:
    result = holdout.run_split("dev", tmp_path / "dev", ks=(1,))
    bundle = result["bundle"]
    assert bundle["split"] == "dev"
    assert bundle["evaluation_status"] == "development"
    assert bundle["holdout_spec"]["sha256"] == holdout.HOLDOUT_SPEC_SHA256
    assert bundle["preregistration"]["sha256"]
    assert bundle["dataset_sha256"] == holdout.split_dataset_sha256("dev")
    assert "results.json" in bundle["files"]
    assert (tmp_path / "dev" / "holdout-bundle.json").is_file()
    assert not (tmp_path / "dev" / "UNBLINDED.json").exists()


def test_existing_unblind_sentinel_forces_replication(monkeypatch, tmp_path) -> None:
    fake_repo = tmp_path / "repo"
    sentinel = fake_repo / holdout.UNBLINDED_SENTINEL_RELATIVE_PATH
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text('{"status":"unblinded"}\n', encoding="utf-8")
    monkeypatch.setattr(holdout, "_repo_root", lambda: fake_repo)

    # Avoid needing the fake repo to contain the real holdout spec for this policy-only check.
    monkeypatch.setattr(holdout, "load_holdout_spec", lambda: {
        "splits": {"test": {"seeds": [1009]}},
        "traces_per_seed": 1,
        "datasets": ["failure_recovery", "branching"],
    })
    with pytest.raises(RuntimeError, match="must pass --replication"):
        holdout.run_split("test", tmp_path / "result", ks=(1,))
