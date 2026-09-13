"""Independent ground-truth SCM evidence benchmark (#38, #41).

The structural equations and labels are generated before retrieval.  Hidden
exogenous values and answer labels are never stored in MemoryNode metadata/text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ctrag.embedding import HashingEmbedder
from ctrag.models import CausalProvenance, Edge, EdgeKind, MemoryNode, QueryMode, RetrievalWeights
from ctrag.retriever import CTRetriever
from ctrag.topology import CausalTopology

from .protocol import preregistration_manifest

SCHEMA_VERSION = 1
GENERATOR = "ctrag-independent-scm-suite"
FAMILIES = ("linear_chain", "observed_confounder", "hidden_confounder")
SEEDS = (101, 202, 303, 404, 505)


@dataclass(frozen=True)
class SCMRecord:
    id: str
    family: str
    split: str
    x: int
    z: int | None
    y: float
    intervention: str
    factual_y: float
    counterfactual_y: float
    true_individual_effect: float


def generate_records(family: str, seed: int, n: int = 240) -> list[SCMRecord]:
    """Generate paired outcomes from fixed structural equations.

    linear_chain: X := 1[Ux>.5], Y := 2X + Uy
    observed_confounder: Z := 1[Uz>.5], X depends on Z, Y := 2X+3Z+Uy
    hidden_confounder: same equations but Z is withheld from the index/estimator.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown SCM family: {family}")
    rng = random.Random(seed)
    records = []
    for index in range(n):
        z = int(rng.random() > .5)
        ux, uy = rng.random(), rng.gauss(0, .2)
        natural_x = int(ux < (.8 if z else .2)) if family != "linear_chain" else int(ux > .5)
        intervention = "observational"
        x = natural_x
        if index % 3 == 1:
            intervention, x = "do_x_0", 0
        elif index % 3 == 2:
            intervention, x = "do_x_1", 1
        confound = 0.0 if family == "linear_chain" else 3.0 * z
        y0, y1 = confound + uy, 2.0 + confound + uy
        y = y1 if x else y0
        records.append(SCMRecord(
            id=f"{family}-{seed}-{index:04d}", family=family,
            split="train" if index < int(.6*n) else "dev" if index < int(.8*n) else "test",
            x=x, z=None if family == "hidden_confounder" else z, y=y,
            intervention=intervention, factual_y=y,
            counterfactual_y=y0 if x else y1, true_individual_effect=y1-y0,
        ))
    return records


def public_view(record: SCMRecord) -> dict:
    """Fields allowed in the retrieval index; answer/paired outcomes are excluded."""
    return {"id": record.id, "family": record.family, "split": record.split,
            "x": record.x, "z": record.z, "y": record.y,
            "intervention": record.intervention}


def assert_no_label_leakage(node: MemoryNode) -> None:
    serialized = (node.text + json.dumps(node.metadata, sort_keys=True)).casefold()
    forbidden = ("counterfactual_y", "true_individual_effect", "factual_y", "exogenous", "latent_z")
    leaked = [field for field in forbidden if field in serialized]
    if leaked:
        raise RuntimeError(f"SCM answer-label leakage: {leaked}")


def build_topology(records: list[SCMRecord], *, include_true_causal_edges: bool) -> CausalTopology:
    topology = CausalTopology()
    epoch = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i, record in enumerate(records):
        view = public_view(record)
        z_text = "z unobserved" if record.z is None else f"z {record.z}"
        node = MemoryNode(record.id,
            f"{record.family} {record.intervention} x {record.x} {z_text} outcome {record.y:.3f}",
            timestamp=epoch + timedelta(seconds=i), metadata=view)
        assert_no_label_leakage(node)
        topology.add_node(node)
    if include_true_causal_edges:
        # Link comparable do(0)/do(1) evidence within deterministic triplets.
        by_id = {r.id: r for r in records}
        for i in range(0, len(records)-2, 3):
            obs, do0, do1 = records[i:i+3]
            for target in (do0, do1):
                topology.add_edge(Edge(obs.id, target.id, EdgeKind.CAUSAL,
                    provenance=CausalProvenance.WORKFLOW, confidence=1.0))
    return topology


def _ate(rows: list[SCMRecord], *, intervention_only: bool, adjust_z: bool) -> float:
    eligible = [r for r in rows if (r.intervention != "observational") == intervention_only]
    if adjust_z and all(r.z is not None for r in eligible):
        effects = []
        for z in (0, 1):
            group = [r for r in eligible if r.z == z]
            y1, y0 = [r.y for r in group if r.x == 1], [r.y for r in group if r.x == 0]
            if y1 and y0:
                effects.append(statistics.mean(y1)-statistics.mean(y0))
        return statistics.mean(effects)
    y1, y0 = [r.y for r in eligible if r.x == 1], [r.y for r in eligible if r.x == 0]
    return statistics.mean(y1)-statistics.mean(y0)


def evaluate_family(family: str, seed: int) -> dict:
    records = generate_records(family, seed)
    train_dev = [r for r in records if r.split != "test"]
    dev = [r for r in records if r.split == "dev"]
    topology = build_topology(train_dev, include_true_causal_edges=True)
    retriever = CTRetriever(topology, embedder=HashingEmbedder(128))
    dense = RetrievalWeights(1, 0, 0, 0, 0, 0)
    paired_hits = {"dense": [], "ctrag_supplied_graph": []}
    for record in dev:
        if record.intervention != "observational":
            continue
        unit = int(record.id.rsplit("-", 1)[1]) // 3
        gold = {r.id for r in train_dev
                if int(r.id.rsplit("-", 1)[1]) // 3 == unit
                and r.intervention != "observational"}
        for arm, weights in (("dense", dense), ("ctrag_supplied_graph", None)):
            hits = retriever.search("intervention outcomes for this unit", mode=QueryMode.WHAT_NEXT,
                                    anchor_ids=[record.id], k=2, max_hops=2,
                                    weights=weights, exhaustive=True)
            paired_hits[arm].append(len(gold.intersection(h.node.id for h in hits))/max(1, len(gold)))
    observational_unadjusted = _ate(train_dev, intervention_only=False, adjust_z=False)
    observational_adjusted = _ate(train_dev, intervention_only=False, adjust_z=True)
    randomized_do = _ate(train_dev, intervention_only=True, adjust_z=False)
    # This simple evidence retriever does not identify unit exogenous noise; no
    # estimated counterfactual is emitted. Coverage=0 is the honest result.
    return {
        "family": family, "seed": seed, "n_train_dev": len(train_dev),
        "true_ate": 2.0,
        "estimators": {
            "observational_unadjusted": observational_unadjusted,
            "observational_adjusted_if_z_visible": observational_adjusted,
            "randomized_interventional_difference": randomized_do,
        },
        "absolute_error": {
            "observational_unadjusted": abs(observational_unadjusted-2.0),
            "observational_adjusted_if_z_visible": abs(observational_adjusted-2.0),
            "randomized_interventional_difference": abs(randomized_do-2.0),
        },
        "intervention_pair_recall_at_2": {arm: statistics.mean(values) for arm, values in paired_hits.items()},
        "counterfactual_estimation": {"coverage": 0.0, "mae": None,
            "reason": "CT-RAG retrieves evidence but has no identified SCM/exogenous-state estimator"},
        "causal_discovery": {"performed": False,
            "reason": "true causal edges are supplied to the retrieval arm; supplying is not discovery"},
    }


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    prereg = preregistration_manifest()
    rows = [evaluate_family(family, seed) for family in FAMILIES for seed in SEEDS]
    payload = {"schema_version": SCHEMA_VERSION, "generator": GENERATOR,
        "status": "exploratory_train_dev_only", "families": list(FAMILIES), "seeds": list(SEEDS),
        "information_contract": {
            "index_fields": ["id", "family", "split", "x", "z_when_observed", "y", "intervention"],
            "hidden_fields": ["exogenous noise", "latent z", "paired counterfactual outcome", "true individual effect"],
            "supplied_graph_arm_is_causal_discovery": False,
        }, "results": rows, "preregistration": prereg,
        "final_holdout": "sealed_not_loaded"}
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n"
    (output/"scm-results.json").write_text(encoded, encoding="utf-8", newline="\n")
    manifest = {"schema_version": 1, "artifact": "scm-results.json",
        "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "preregistration_sha256": prereg["sha256"], "final_holdout": "sealed_not_loaded"}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n",
        encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("scm-results"))
    args = parser.parse_args()
    result = run(args.output)
    print(f"SCM benchmark complete: {len(result['results'])} family/seed cells; final holdout sealed")


if __name__ == "__main__":
    main()
