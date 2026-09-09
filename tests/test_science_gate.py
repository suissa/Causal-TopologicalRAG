from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctrag.science_gate import validate_ledger, validate_science_gate


def test_current_science_gate_passes_and_final_test_is_still_sealed() -> None:
    result = validate_science_gate()
    assert result["science_changes"] >= 1
    assert result["final_test_status"] == "sealed"
    assert len(result["final_test_sha256"]) == 64


def test_correctness_bug_requires_regression_test_file(tmp_path) -> None:
    ledger = {
        "schema_version": 1,
        "policy_version": 1,
        "entries": [{
            "id": "SCI-999",
            "classification": "correctness_bug",
            "code_changed": True,
            "issue": 999,
            "title": "missing regression",
            "description": "fixture",
            "pre_fix": {"tests": "failed"},
            "post_fix": {"tests": "passed"},
            "regression_tests": ["tests/does-not-exist.py::test_missing"],
        }],
    }
    with pytest.raises(RuntimeError, match="regression test file not found"):
        validate_ledger(ledger, root=tmp_path, changelog="SCI-999")


def test_negative_result_cannot_be_reclassified_with_code_change(tmp_path) -> None:
    ledger = {
        "schema_version": 1,
        "policy_version": 1,
        "entries": [{
            "id": "SCI-998",
            "classification": "negative_result",
            "code_changed": True,
            "title": "CT-RAG lost",
            "description": "A genuine unfavorable experiment is still a result.",
        }],
    }
    with pytest.raises(RuntimeError, match="negative result cannot be accompanied by a code change"):
        validate_ledger(ledger, root=tmp_path, changelog="SCI-998")


def test_science_change_id_must_exist_in_human_changelog(tmp_path) -> None:
    test_file = tmp_path / "tests" / "test_fix.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_fix(): pass\n", encoding="utf-8")
    ledger = {
        "schema_version": 1,
        "policy_version": 1,
        "entries": [{
            "id": "SCI-997",
            "classification": "correctness_bug",
            "code_changed": True,
            "issue": 997,
            "title": "fixture",
            "description": "fixture",
            "pre_fix": {},
            "post_fix": {},
            "regression_tests": ["tests/test_fix.py::test_fix"],
        }],
    }
    with pytest.raises(RuntimeError, match="missing from CHANGELOG_SCIENCE.md"):
        validate_ledger(ledger, root=tmp_path, changelog="")
