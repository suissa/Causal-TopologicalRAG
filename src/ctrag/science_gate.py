from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmarks.holdout import (
    FROZEN_TEST_DATASET_SHA256,
    UNBLINDED_SENTINEL_RELATIVE_PATH,
    load_holdout_spec,
    split_dataset_sha256,
)
from .benchmarks.protocol import preregistration_manifest

ALLOWED_CLASSIFICATIONS = {
    "correctness_bug",
    "modeling_assumption",
    "dataset_artifact",
    "negative_result",
    "protocol_amendment",
    "exploratory_analysis",
}
LEDGER_RELATIVE_PATH = "research/science-changes.json"
CHANGELOG_RELATIVE_PATH = "CHANGELOG_SCIENCE.md"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_ledger(ledger: dict[str, Any], *, root: Path, changelog: str) -> None:
    _require(ledger.get("schema_version") == 1, "unsupported science ledger schema")
    _require(ledger.get("policy_version") == 1, "unsupported science policy version")
    entries = ledger.get("entries")
    _require(isinstance(entries, list), "science ledger entries must be a list")

    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        _require(isinstance(entry, dict), f"science entry #{position} must be an object")
        change_id = entry.get("id")
        _require(isinstance(change_id, str) and change_id.startswith("SCI-"), "invalid science change id")
        _require(change_id not in seen, f"duplicate science change id: {change_id}")
        seen.add(change_id)
        _require(change_id in changelog, f"{change_id} missing from {CHANGELOG_RELATIVE_PATH}")

        classification = entry.get("classification")
        _require(
            classification in ALLOWED_CLASSIFICATIONS,
            f"{change_id} has unsupported classification: {classification}",
        )
        _require(isinstance(entry.get("title"), str) and entry["title"].strip(), f"{change_id} missing title")
        _require(
            isinstance(entry.get("description"), str) and entry["description"].strip(),
            f"{change_id} missing description",
        )

        if classification == "correctness_bug":
            _require(entry.get("code_changed") is True, f"{change_id} correctness bug must record code_changed=true")
            _require(isinstance(entry.get("issue"), int), f"{change_id} correctness bug must link an issue")
            _require(isinstance(entry.get("pre_fix"), dict), f"{change_id} missing pre_fix evidence")
            _require(isinstance(entry.get("post_fix"), dict), f"{change_id} missing post_fix evidence")
            tests = entry.get("regression_tests")
            _require(isinstance(tests, list) and tests, f"{change_id} must list regression tests")
            for test_ref in tests:
                _require(isinstance(test_ref, str) and "::" in test_ref, f"{change_id} invalid regression-test ref")
                path = root / test_ref.split("::", 1)[0]
                _require(path.is_file(), f"{change_id} regression test file not found: {path}")

        if classification == "negative_result":
            _require(
                entry.get("code_changed") is False,
                f"{change_id} negative result cannot be accompanied by a code change; "
                "open a separate correctness defect if one exists",
            )


def validate_science_gate(root: Path | None = None) -> dict[str, Any]:
    root = root or _repo_root()
    ledger_path = root / LEDGER_RELATIVE_PATH
    changelog_path = root / CHANGELOG_RELATIVE_PATH
    _require(ledger_path.is_file(), f"missing {LEDGER_RELATIVE_PATH}")
    _require(changelog_path.is_file(), f"missing {CHANGELOG_RELATIVE_PATH}")

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    changelog = changelog_path.read_text(encoding="utf-8")
    validate_ledger(ledger, root=root, changelog=changelog)

    prereg = preregistration_manifest()
    holdout = load_holdout_spec()
    test_digest = split_dataset_sha256("test")
    _require(test_digest == FROZEN_TEST_DATASET_SHA256, "frozen final-test fingerprint mismatch")

    sentinel_path = root / UNBLINDED_SENTINEL_RELATIVE_PATH
    final_test_status = "sealed"
    if sentinel_path.exists():
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        _require(sentinel.get("status") == "unblinded", "invalid final-test unblind sentinel")
        _require(
            sentinel.get("dataset_sha256") == FROZEN_TEST_DATASET_SHA256,
            "unblind sentinel does not match the frozen final test",
        )
        final_test_status = "unblinded-replication-only"

    return {
        "science_changes": len(ledger["entries"]),
        "preregistration_sha256": prereg["sha256"],
        "holdout_protocol": holdout["protocol"],
        "final_test_sha256": test_digest,
        "final_test_status": final_test_status,
    }


def main() -> None:
    result = validate_science_gate()
    print(
        "Science gate passed: "
        f"changes={result['science_changes']} "
        f"final_test={result['final_test_status']} "
        f"test_sha256={result['final_test_sha256']}"
    )


if __name__ == "__main__":
    main()
