from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .datasets import generate
from .protocol import preregistration_manifest
from .runner import Config, run

HOLDOUT_SPEC_RELATIVE_PATH = "research/holdout/manifest-v1.json"
HOLDOUT_SPEC_SHA256 = "f0e3def445bd5c4d8c3cbc9c66e64f3a752c7bc782d15a8332bede15671e5137"
HOLDOUT_PROTOCOL_VERSION = 1
# Filled after the generator-derived test manifest is independently fingerprinted.
FROZEN_TEST_DATASET_SHA256: str | None = None
UNBLINDED_SENTINEL_RELATIVE_PATH = "research/holdout/UNBLINDED.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def load_holdout_spec() -> dict[str, Any]:
    path = _repo_root() / HOLDOUT_SPEC_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"holdout manifest not found: {path}")
    digest = _sha256_file(path)
    if digest != HOLDOUT_SPEC_SHA256:
        raise RuntimeError(
            "holdout split specification changed without a protocol-version/hash update "
            f"(expected {HOLDOUT_SPEC_SHA256}, got {digest})"
        )
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != HOLDOUT_PROTOCOL_VERSION:
        raise RuntimeError("holdout schema/protocol version mismatch")
    return spec


def split_seeds(split: str) -> tuple[int, ...]:
    spec = load_holdout_spec()
    try:
        seeds = tuple(int(seed) for seed in spec["splits"][split]["seeds"])
    except KeyError as error:
        raise ValueError(f"unknown holdout split: {split}") from error
    if not seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"invalid seed set for split {split}")
    return seeds


def split_dataset_manifest(split: str) -> list[dict[str, Any]]:
    spec = load_holdout_spec()
    seeds = split_seeds(split)
    traces = int(spec["traces_per_seed"])
    manifests: list[dict[str, Any]] = []
    for dataset_name in spec["datasets"]:
        for seed in seeds:
            manifests.append(generate(str(dataset_name), seed, traces).manifest())
    return manifests


def split_dataset_sha256(split: str) -> str:
    payload = _canonical_json(split_dataset_manifest(split)).encode("utf-8")
    digest = _sha256_bytes(payload)
    if split == "test" and FROZEN_TEST_DATASET_SHA256 is not None:
        if digest != FROZEN_TEST_DATASET_SHA256:
            raise RuntimeError(
                "final-test dataset fingerprint changed; the frozen holdout is invalid "
                f"(expected {FROZEN_TEST_DATASET_SHA256}, got {digest})"
            )
    return digest


def _bundle_manifest(output: Path, *, split: str, status: str) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output.iterdir()):
        if not path.is_file() or path.name in {"holdout-bundle.json", "UNBLINDED.json"}:
            continue
        files[path.name] = {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
    return {
        "schema_version": 1,
        "holdout_protocol_version": HOLDOUT_PROTOCOL_VERSION,
        "split": split,
        "evaluation_status": status,
        "holdout_spec": {
            "path": HOLDOUT_SPEC_RELATIVE_PATH,
            "sha256": HOLDOUT_SPEC_SHA256,
        },
        "dataset_sha256": split_dataset_sha256(split),
        "preregistration": preregistration_manifest(),
        "files": files,
    }


def _write_bundle(output: Path, *, split: str, status: str) -> dict[str, Any]:
    bundle = _bundle_manifest(output, split=split, status=status)
    text = _canonical_json(bundle)
    bundle_path = output / "holdout-bundle.json"
    bundle_path.write_text(text, encoding="utf-8", newline="\n")
    bundle["bundle_sha256"] = _sha256_bytes(text.encode("utf-8"))
    return bundle


def run_split(
    split: str,
    output: Path,
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10),
    dimensions: int = 256,
    max_hops: int = 8,
    hop_decay: float = 0.7,
    unblind_final_test: bool = False,
    replication: bool = False,
) -> dict[str, Any]:
    spec = load_holdout_spec()
    if split not in spec["splits"]:
        raise ValueError(f"unknown holdout split: {split}")

    sentinel = _repo_root() / UNBLINDED_SENTINEL_RELATIVE_PATH
    if split == "test":
        if sentinel.exists():
            if not replication:
                raise RuntimeError(
                    "final test has already been unblinded; subsequent runs must pass --replication"
                )
            status = "replication"
        else:
            if replication:
                raise RuntimeError("cannot mark the first final-test run as replication")
            if not unblind_final_test:
                raise RuntimeError(
                    "final test is sealed; pass --unblind-final-test only for the one-shot frozen evaluation"
                )
            status = "pristine_holdout"
    else:
        if unblind_final_test or replication:
            raise ValueError("unblind/replication flags are valid only for the test split")
        status = "development"

    config = Config(
        seeds=split_seeds(split),
        ks=ks,
        traces=int(spec["traces_per_seed"]),
        dimensions=dimensions,
        max_hops=max_hops,
        hop_decay=hop_decay,
    )
    report = run(config, output)
    bundle = _write_bundle(output, split=split, status=status)

    if split == "test" and status == "pristine_holdout":
        sentinel_payload = {
            "schema_version": 1,
            "status": "unblinded",
            "holdout_spec_sha256": HOLDOUT_SPEC_SHA256,
            "dataset_sha256": bundle["dataset_sha256"],
            "result_bundle_sha256": bundle["bundle_sha256"],
            "instruction": (
                "Commit this file as research/holdout/UNBLINDED.json immediately after the "
                "first final evaluation; every later final-test run is a replication."
            ),
        }
        (output / "UNBLINDED.json").write_text(
            _canonical_json(sentinel_payload), encoding="utf-8", newline="\n"
        )

    return {"report": report, "bundle": bundle}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or fingerprint CT-RAG Phase 2 holdout splits")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fingerprint = subparsers.add_parser("fingerprint", help="Fingerprint a split without retrieval")
    fingerprint.add_argument("split", choices=("train", "dev", "test"))

    execute = subparsers.add_parser("run", help="Run one holdout split")
    execute.add_argument("split", choices=("train", "dev", "test"))
    execute.add_argument("--output", type=Path)
    execute.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    execute.add_argument("--dimensions", type=int, default=256)
    execute.add_argument("--max-hops", type=int, default=8)
    execute.add_argument("--hop-decay", type=float, default=0.7)
    execute.add_argument("--unblind-final-test", action="store_true")
    execute.add_argument("--replication", action="store_true")

    args = parser.parse_args()
    if args.command == "fingerprint":
        print(f"{args.split} dataset SHA-256: {split_dataset_sha256(args.split)}")
        return

    output = args.output or Path(f"benchmark-results/{args.split}")
    result = run_split(
        args.split,
        output,
        ks=tuple(args.ks),
        dimensions=args.dimensions,
        max_hops=args.max_hops,
        hop_decay=args.hop_decay,
        unblind_final_test=args.unblind_final_test,
        replication=args.replication,
    )
    print(
        f"Wrote {len(result['report']['results'])} observations for {args.split}; "
        f"bundle SHA-256 {result['bundle']['bundle_sha256']}"
    )


if __name__ == "__main__":
    main()
