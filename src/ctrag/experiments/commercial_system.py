"""End-to-end commercial-system root-cause localization experiment.

The simulator creates operational observations and an independent oracle before
retrieval.  Oracle roles are never copied into indexed node text or metadata.
The experiment therefore evaluates whether explicit event causation plus CT-RAG
navigation can localize a known cause and its observed remediation amid lexical
distractors.  It does not claim to discover causal structure from correlations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from ctrag.benchmarks.runner import BASELINES
from ctrag.embedding import HashingEmbedder
from ctrag.events import EventProjector, EventRecord
from ctrag.models import EdgeKind, QueryMode, RetrievalHit
from ctrag.retriever import CTRetriever
from ctrag.topology import CausalTopology

SCHEMA_VERSION = 1
GENERATOR = "ctrag-commercial-system-v1"
FORBIDDEN_ORACLE_FIELDS = (
    "root_cause",
    "root_cause_event_id",
    "solution_event_id",
    "oracle_role",
    "expected_answer",
)


@dataclass(frozen=True)
class CommercialExperimentConfig:
    seed: int = 20260914
    dimensions: int = 256
    max_hops: int = 8
    k: int = 3

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if min(self.dimensions, self.max_hops, self.k) < 1:
            raise ValueError("dimensions, max_hops and k must be positive")


@dataclass(frozen=True)
class Step:
    key: str
    event_type: str
    status: str
    domain: str
    detail: str
    parent: str | None
    facts: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    diagnostic_query: str
    recovery_query: str
    root_cause_key: str
    symptom_key: str
    solution_key: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class OracleCase:
    scenario_id: str
    title: str
    symptom_event_id: str
    root_cause_event_id: str
    solution_event_id: str
    diagnostic_query: str
    recovery_query: str


def _step(
    logical_key: str,
    event_type: str,
    status: str,
    domain: str,
    detail: str,
    parent: str | None,
    **facts: Any,
) -> Step:
    return Step(logical_key, event_type, status, domain, detail, parent, tuple(sorted(facts.items())))


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "inventory_projection_lag",
        "Venda cancelada por estoque projetado atrasado",
        "Por que uma venda aceita foi cancelada quando a reserva de estoque falhou?",
        "Qual correção permitiu reservar o estoque e concluir a venda?",
        "cause", "symptom", "solution",
        (
            _step("start", "Sales.CheckoutStarted", "ok", "sales", "checkout opened for a catalog item", None, sku="SKU-42", quantity=4),
            _step("cause", "Inventory.ProjectionCheckpointStalled", "warning", "inventory", "read projection checkpoint stopped advancing", "start", checkpoint=875, write_version=912, lag_events=37),
            _step("read", "Sales.AvailabilityRead", "ok", "sales", "checkout read an outdated available quantity", "cause", reported_available=7, committed_available=3),
            _step("accepted", "Sales.OrderAccepted", "ok", "sales", "order accepted from the stale availability view", "read", order="ORD-1001"),
            _step("rejected", "Inventory.ReservationRejected", "error", "inventory", "authoritative stock rejected requested reservation", "accepted", requested=4, available=3),
            _step("symptom", "Sales.OrderCancelled", "error", "sales", "accepted order cancelled after inventory reservation failure", "rejected", order="ORD-1001"),
            _step("solution", "Inventory.ProjectionRebuilt", "healed", "inventory", "projection replayed from the authoritative stock ledger", "symptom", checkpoint=912),
            _step("retry", "Inventory.ReservationAccepted", "ok", "inventory", "reservation retry used the rebuilt projection", "solution", quantity=3),
            _step("recovered", "Sales.OrderCompleted", "completed", "sales", "corrected order completed after customer confirmation", "retry", order="ORD-1001-R1"),
        ),
    ),
    Scenario(
        "payment_webhook_replay",
        "Conciliação falhou após captura duplicada",
        "Por que a conciliação financeira encontrou duas capturas para a mesma venda?",
        "Qual ação removeu o lançamento duplicado e impediu novo replay?",
        "cause", "symptom", "solution",
        (
            _step("start", "Payments.WebhookReceived", "ok", "payments", "gateway capture notification received", None, gateway_event="GW-7781"),
            _step("cause", "Payments.IdempotencyLookupBypassed", "warning", "payments", "legacy webhook route skipped the idempotency lookup", "start", route="/hooks/capture-v1", key="GW-7781"),
            _step("capture", "Payments.CaptureApplied", "ok", "payments", "capture notification applied again to the sale", "cause", payment="PAY-88", amount_cents=15990),
            _step("posting", "Finance.LedgerPostingDuplicated", "error", "finance", "second receivable posting created for one payment", "capture", ledger_key="PAY-88"),
            _step("symptom", "Finance.ReconciliationFailed", "error", "finance", "gateway total did not match duplicated internal receivable", "posting", difference_cents=15990),
            _step("solution", "Payments.IdempotencyGuardRestored", "healed", "payments", "webhook route now rejects previously consumed gateway keys", "symptom", route="/hooks/capture-v1"),
            _step("reverse", "Finance.DuplicatePostingReversed", "ok", "finance", "duplicate receivable was reversed with an audit reference", "solution", ledger_key="PAY-88"),
            _step("recovered", "Finance.ReconciliationCompleted", "completed", "finance", "gateway and internal receivable totals matched", "reverse", difference_cents=0),
        ),
    ),
    Scenario(
        "fiscal_certificate_expiry",
        "Expedição bloqueada por rejeição fiscal",
        "Por que o pedido pago não pôde ser expedido após a rejeição da nota fiscal?",
        "Qual correção liberou a autorização fiscal e a expedição?",
        "cause", "symptom", "solution",
        (
            _step("start", "Fiscal.InvoiceRequested", "ok", "fiscal", "invoice requested for a paid order", None, order="ORD-2204"),
            _step("cause", "Fiscal.CertificateValidationFailed", "error", "fiscal", "signing certificate was outside its validity interval", "start", certificate="CERT-A", expired_days=1),
            _step("rejected", "Fiscal.InvoiceAuthorizationRejected", "error", "fiscal", "tax authority rejected the unsigned invoice", "cause", code="CERT_EXPIRED"),
            _step("blocked", "Fulfillment.ReleaseBlocked", "error", "fulfillment", "warehouse release requires an authorized invoice", "rejected", order="ORD-2204"),
            _step("symptom", "Sales.CompletionTimedOut", "error", "sales", "paid order remained incomplete while fulfillment was blocked", "blocked", order="ORD-2204"),
            _step("solution", "Fiscal.SigningCertificateRotated", "healed", "fiscal", "valid signing certificate activated and verified", "symptom", certificate="CERT-B"),
            _step("authorized", "Fiscal.InvoiceAuthorized", "ok", "fiscal", "invoice retry authorized by the tax authority", "solution", order="ORD-2204"),
            _step("recovered", "Fulfillment.OrderReleased", "completed", "fulfillment", "warehouse received the authorized invoice and released the order", "authorized", order="ORD-2204"),
        ),
    ),
    Scenario(
        "procurement_cost_feed_lag",
        "Margem negativa após promoção",
        "Por que uma promoção aprovada vendeu produtos com margem negativa?",
        "Qual correção restaurou o custo atual e recalculou o preço?",
        "cause", "symptom", "solution",
        (
            _step("start", "Procurement.SupplierCostChanged", "ok", "procurement", "supplier published a higher acquisition cost", None, sku="SKU-91", cost_cents=8200),
            _step("cause", "Pricing.CostFeedCheckpointStalled", "warning", "pricing", "pricing consumer did not ingest the latest supplier cost", "start", visible_cost_cents=6100, lag_events=12),
            _step("margin", "Pricing.MarginCalculated", "ok", "pricing", "promotion margin calculated from the previous acquisition cost", "cause", visible_cost_cents=6100, sale_price_cents=7490),
            _step("approved", "Marketing.PromotionApproved", "ok", "marketing", "campaign activated using the stale positive margin", "margin", campaign="WEEKEND-9"),
            _step("sold", "Sales.PromotionalOrderCompleted", "ok", "sales", "promotional order completed below current acquisition cost", "approved", order="ORD-3310"),
            _step("symptom", "Finance.NegativeMarginDetected", "error", "finance", "realized contribution margin became negative", "sold", margin_cents=-710),
            _step("solution", "Pricing.CostProjectionReplayed", "healed", "pricing", "supplier cost events replayed into the pricing view", "symptom", visible_cost_cents=8200),
            _step("corrected", "Pricing.PromotionPriceCorrected", "ok", "pricing", "promotion recalculated against the current cost", "solution", sale_price_cents=9790),
            _step("recovered", "Finance.PositiveMarginVerified", "completed", "finance", "new promotional sale produced a positive margin", "corrected", margin_cents=1590),
        ),
    ),
    Scenario(
        "customer_identity_normalization",
        "Resgate de fidelidade recusado",
        "Por que um cliente conhecido teve o resgate de pontos recusado no checkout?",
        "Qual correção reuniu o saldo de fidelidade e permitiu o resgate?",
        "cause", "symptom", "solution",
        (
            _step("start", "CRM.CustomerContactReceived", "ok", "crm", "customer returned through the WhatsApp checkout", None, channel="whatsapp"),
            _step("cause", "CRM.PhoneNormalizationRuleDiverged", "warning", "crm", "contact import and checkout produced different canonical phone forms", "start", import_rule="E164-v2", checkout_rule="local-v1"),
            _step("duplicate", "CRM.DuplicateCustomerCreated", "error", "crm", "checkout resolved the contact to a second customer identity", "cause", customer="CUS-902B"),
            _step("split", "Loyalty.BalanceSplit", "error", "loyalty", "earned points remained attached to the original identity", "duplicate", available_points=0, original_points=840),
            _step("symptom", "Checkout.LoyaltyRedemptionRejected", "error", "sales", "checkout identity had insufficient visible points", "split", required_points=500),
            _step("solution", "CRM.CustomerIdentitiesMerged", "healed", "crm", "duplicate identity merged using one canonical E.164 characteristic", "symptom", surviving_customer="CUS-902A"),
            _step("rebuilt", "Loyalty.BalanceRebuilt", "ok", "loyalty", "point ledger projected onto the surviving customer identity", "solution", available_points=840),
            _step("recovered", "Checkout.LoyaltyRedemptionAccepted", "completed", "sales", "checkout redeemed points from the rebuilt balance", "rebuilt", redeemed_points=500),
        ),
    ),
    Scenario(
        "delivery_capacity_race",
        "Entrega atrasada por rota superlotada",
        "Por que dois pedidos ocuparam a mesma capacidade e atrasaram a expedição?",
        "Qual correção refez a rota e protegeu a capacidade contra concorrência?",
        "cause", "symptom", "solution",
        (
            _step("start", "Fulfillment.CapacityReservationRequested", "ok", "fulfillment", "two dispatch workers requested the final route slot", None, route="R-7"),
            _step("cause", "Fulfillment.CapacityVersionConflictIgnored", "warning", "fulfillment", "fulfillment accepted a stale expected version during concurrent allocation", "start", expected_version=44, actual_version=45),
            _step("allocated", "Delivery.RouteSlotAllocatedTwice", "error", "delivery", "same route capacity was allocated to two orders", "cause", route="R-7", slot="14:00"),
            _step("overbooked", "Delivery.RouteOverbooked", "error", "delivery", "route load exceeded vehicle capacity", "allocated", capacity=12, assigned=13),
            _step("symptom", "Sales.DispatchPromiseMissed", "error", "sales", "customer dispatch promise was missed", "overbooked", order="ORD-5502", delay_minutes=95),
            _step("solution", "Delivery.RouteReplanned", "healed", "delivery", "overflow order moved to an available nearby route", "symptom", from_route="R-7", to_route="R-8"),
            _step("guard", "Fulfillment.OptimisticLockEnforced", "ok", "fulfillment", "stale capacity versions are rejected before allocation", "solution", aggregate="route-capacity"),
            _step("recovered", "Sales.DispatchPromiseRecovered", "completed", "sales", "customer received a corrected dispatch estimate", "guard", order="ORD-5502"),
        ),
    ),
)


def _opaque_id(seed: int, scenario_id: str, key: str) -> str:
    digest = hashlib.sha256(f"{seed}:{scenario_id}:{key}".encode()).hexdigest()[:16]
    return f"evt-{digest}"


def _event(
    scenario: Scenario,
    step: Step,
    ids: dict[str, str],
    timestamp: datetime,
) -> EventRecord:
    payload = {"domain": step.domain, "detail": step.detail, **dict(step.facts)}
    return EventRecord(
        event_id=ids[step.key],
        event_type=step.event_type,
        timestamp=timestamp,
        payload=payload,
        causation_id=None if step.parent is None else ids[step.parent],
        correlation_id=f"incident:{scenario.id}",
        execution_id=f"commercial:{scenario.id}",
        intent_id="Commerce.IncidentObserved",
        actor_id=f"{step.domain}.agent",
        action_id=step.event_type,
        status=step.status,
    )


def _decoys(scenario: Scenario, seed: int, start: datetime) -> list[EventRecord]:
    # Lexical lookalikes intentionally mention the symptom and recovery language,
    # but are independent records without causation links to the incident.
    domain = scenario.steps[[item.key for item in scenario.steps].index("symptom")].domain
    return [
        EventRecord(
            event_id=_opaque_id(seed, scenario.id, "decoy-runbook"),
            event_type="Knowledge.RunbookIndexed",
            timestamp=start,
            payload={"domain": domain, "detail": scenario.diagnostic_query + " archived troubleshooting guide"},
            correlation_id=f"knowledge:{scenario.id}",
            execution_id=f"noise:{scenario.id}:runbook",
            intent_id="Knowledge.IndexDocument",
            actor_id="knowledge.agent",
            action_id="Knowledge.RunbookIndexed",
            status="ok",
        ),
        EventRecord(
            event_id=_opaque_id(seed, scenario.id, "decoy-drill"),
            event_type="Operations.RecoveryDrillCompleted",
            timestamp=start + timedelta(milliseconds=1),
            payload={"domain": domain, "detail": scenario.recovery_query + " simulated training record"},
            correlation_id=f"training:{scenario.id}",
            execution_id=f"noise:{scenario.id}:drill",
            intent_id="Operations.RunDrill",
            actor_id="operations.agent",
            action_id="Operations.RecoveryDrillCompleted",
            status="completed",
        ),
    ]


def generate(config: CommercialExperimentConfig) -> tuple[list[EventRecord], list[OracleCase]]:
    epoch = datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)
    observations: list[EventRecord] = []
    oracle: list[OracleCase] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        ids = {step.key: _opaque_id(config.seed, scenario.id, step.key) for step in scenario.steps}
        start = epoch + timedelta(minutes=scenario_index * 10)
        observations.extend(
            _event(scenario, step, ids, start + timedelta(seconds=index))
            for index, step in enumerate(scenario.steps)
        )
        observations.extend(_decoys(scenario, config.seed, start + timedelta(seconds=90)))
        oracle.append(OracleCase(
            scenario.id,
            scenario.title,
            ids[scenario.symptom_key],
            ids[scenario.root_cause_key],
            ids[scenario.solution_key],
            scenario.diagnostic_query,
            scenario.recovery_query,
        ))
    observations.sort(key=lambda item: (item.timestamp, item.event_id))
    return observations, oracle


def assert_no_oracle_leakage(topology: CausalTopology) -> None:
    for node in topology.nodes.values():
        encoded = (node.text + json.dumps(node.metadata, sort_keys=True)).casefold()
        leaked = [field for field in FORBIDDEN_ORACLE_FIELDS if field in encoded]
        if leaked:
            raise RuntimeError(f"oracle leakage in {node.id}: {leaked}")


def project(observations: list[EventRecord]) -> CausalTopology:
    topology = CausalTopology()
    EventProjector(topology).ingest_many(observations)
    assert_no_oracle_leakage(topology)
    return topology


def _status_anomaly(node_status: Any) -> float:
    status = str(node_status or "").casefold()
    return {
        "error": 1.0,
        "failed": 1.0,
        "warning": 0.85,
        "degraded": 0.75,
    }.get(status, 0.0)


def causal_frontier_diagnosis(
    retriever: CTRetriever,
    anchor_id: str,
    query: str,
    *,
    max_hops: int = 8,
) -> list[dict[str, Any]]:
    """Rank the first anomalous observations on evidenced paths to a symptom.

    The frontier is computed only from public status values and causal edges.
    It never receives an oracle event id or role label.
    """
    topology = retriever.topology
    distances = topology.distances(
        anchor_id,
        direction="in",
        kinds={EdgeKind.CAUSAL},
        max_hops=max_hops,
    )
    retrieval = retriever.search(
        query,
        mode=QueryMode.WHY,
        anchor_ids=[anchor_id],
        k=len(topology.nodes),
        max_hops=max_hops,
        exhaustive=True,
    )
    by_id = {hit.node.id: hit for hit in retrieval}
    rows: list[dict[str, Any]] = []
    for node_id, hops in distances.items():
        if node_id == anchor_id:
            continue
        node = topology.nodes[node_id]
        anomaly = _status_anomaly(node.metadata.get("status"))
        # A later error can have an immediate OK parent (for example, a stale
        # read accepted before reservation fails).  The root frontier is the
        # anomalous observation with no *anomalous causal ancestor*, not merely
        # no anomalous direct parent.
        ancestors = topology.distances(
            node_id,
            direction="in",
            kinds={EdgeKind.CAUSAL},
            max_hops=max_hops,
        )
        earlier_anomaly = any(
            ancestor_id != node_id
            and ancestor_id in distances
            and _status_anomaly(topology.nodes[ancestor_id].metadata.get("status")) > 0
            for ancestor_id in ancestors
        )
        boundary = 1.0 if anomaly and not earlier_anomaly else 0.0
        hit = by_id[node_id]
        path_confidence = hit.causal_path.aggregate_confidence if hit.causal_path else 0.0
        text_signal = 0.5 * hit.components["semantic"] + 0.5 * hit.components["lexical"]
        score = 0.50 * boundary + 0.25 * anomaly + 0.20 * path_confidence + 0.05 * text_signal
        rows.append({
            "event_id": node_id,
            "score": score,
            "causal_hops": hops,
            "frontier": bool(boundary),
            "components": {
                "frontier": boundary,
                "anomaly": anomaly,
                "causal_path_confidence": path_confidence,
                "text_signal": text_signal,
            },
            "path": list(hit.causal_path.nodes) if hit.causal_path else [],
        })
    rows.sort(key=lambda row: (-row["score"], row["causal_hops"], row["event_id"]))
    return rows


def _rank(ids: Iterable[str], gold: str, k: int) -> dict[str, float | int | None]:
    ranked = list(ids)
    position = ranked.index(gold) + 1 if gold in ranked else None
    return {
        "rank": position,
        "top1": int(position == 1),
        f"recall_at_{k}": int(position is not None and position <= k),
        "mrr": 0.0 if position is None else 1.0 / position,
    }


def _hit_payload(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "event_id": hit.node.id,
        "score": hit.score,
        "causal_hops": hit.causal_hops,
        "components": hit.components,
        "path": list(hit.causal_path.nodes) if hit.causal_path else [],
    }


def evaluate(
    topology: CausalTopology,
    oracle: list[OracleCase],
    config: CommercialExperimentConfig,
) -> dict[str, Any]:
    retriever = CTRetriever(topology, embedder=HashingEmbedder(config.dimensions))
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}
    diagnostic_arms = ("lexical_only", "dense_only", "dense_lexical", "full_ctrag")
    for case in oracle:
        scenario_detail: dict[str, Any] = {"diagnosis": {}, "recovery": {}}
        for arm in diagnostic_arms:
            hits = retriever.search(
                case.diagnostic_query,
                mode=QueryMode.WHY,
                anchor_ids=[case.symptom_event_id],
                k=len(topology.nodes),
                max_hops=config.max_hops,
                weights=BASELINES[arm],
                exhaustive=True,
            )
            ranked = [hit.node.id for hit in hits]
            metrics = _rank(ranked, case.root_cause_event_id, config.k)
            rows.append({"scenario_id": case.scenario_id, "task": "diagnosis", "arm": arm, **metrics})
            scenario_detail["diagnosis"][arm] = [_hit_payload(hit) for hit in hits[:config.k]]

        frontier = causal_frontier_diagnosis(
            retriever, case.symptom_event_id, case.diagnostic_query, max_hops=config.max_hops
        )
        metrics = _rank((row["event_id"] for row in frontier), case.root_cause_event_id, config.k)
        rows.append({"scenario_id": case.scenario_id, "task": "diagnosis", "arm": "ctrag_causal_frontier", **metrics})
        scenario_detail["diagnosis"]["ctrag_causal_frontier"] = frontier[:config.k]

        for arm in ("dense_lexical", "full_ctrag"):
            if arm == "full_ctrag":
                hits = retriever.recovery(case.symptom_event_id, case.recovery_query, k=len(topology.nodes)).hits
            else:
                hits = retriever.search(
                    case.recovery_query,
                    mode=QueryMode.RECOVERY,
                    anchor_ids=[case.symptom_event_id],
                    k=len(topology.nodes),
                    max_hops=config.max_hops,
                    weights=BASELINES[arm],
                    exhaustive=True,
                )
            ranked = [hit.node.id for hit in hits]
            metrics = _rank(ranked, case.solution_event_id, config.k)
            rows.append({"scenario_id": case.scenario_id, "task": "recovery", "arm": arm, **metrics})
            scenario_detail["recovery"][arm] = [_hit_payload(hit) for hit in hits[:config.k]]
        details[case.scenario_id] = scenario_detail

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["task"], row["arm"]), []).append(row)
    recall_key = f"recall_at_{config.k}"
    summary = [
        {
            "task": task,
            "arm": arm,
            "n": len(group),
            "top1_accuracy": statistics.mean(item["top1"] for item in group),
            recall_key: statistics.mean(item[recall_key] for item in group),
            "mrr": statistics.mean(item["mrr"] for item in group),
        }
        for (task, arm), group in sorted(grouped.items())
    ]
    return {"rows": rows, "summary": summary, "details": details}


def _graph_payload(topology: CausalTopology) -> dict[str, Any]:
    nodes = []
    edges = []
    for node in sorted(topology.nodes.values(), key=lambda item: (item.timestamp, item.id)):
        nodes.append({
            "id": node.id,
            "event_type": node.metadata.get("event_type"),
            "status": node.metadata.get("status"),
            "domain": node.metadata.get("domain"),
            "detail": node.metadata.get("detail"),
            "correlation_id": node.metadata.get("correlation_id"),
            "timestamp": node.timestamp.isoformat(),
        })
        for edge in sorted(topology.outgoing(node.id), key=lambda item: (item.target, item.kind.value)):
            edges.append({
                "source": edge.source,
                "target": edge.target,
                "kind": edge.kind.value,
                "provenance": None if edge.provenance is None else edge.provenance.value,
                "confidence": edge.confidence,
            })
    return {"nodes": nodes, "edges": edges}


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _explorer(graph: dict[str, Any], oracle: list[OracleCase], evaluation: dict[str, Any]) -> str:
    payload = json.dumps({
        "graph": graph,
        "oracle": [asdict(item) for item in oracle],
        "evaluation": evaluation,
    }, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CT-RAG · Experimento comercial</title>
<style>
:root{{--bg:#07111f;--panel:#0d1b2d;--line:#37516e;--text:#e8f0fa;--muted:#8da4bd;--ok:#37d39a;--warn:#f4bf4f;--error:#ff6b77;--accent:#63a9ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,sans-serif}}
header{{padding:18px 22px;border-bottom:1px solid #20354e;display:flex;gap:18px;align-items:center;flex-wrap:wrap}} h1{{font-size:19px;margin:0}} .badge{{color:var(--ok);border:1px solid #23684f;border-radius:99px;padding:4px 9px}}
main{{display:grid;grid-template-columns:310px 1fr 330px;min-height:calc(100vh - 69px)}} aside{{background:var(--panel);padding:18px;overflow:auto}} aside:first-child{{border-right:1px solid #20354e}} aside:last-child{{border-left:1px solid #20354e}}
label{{display:block;color:var(--muted);font-size:12px;margin:12px 0 5px}} select,button{{width:100%;background:#132942;color:var(--text);border:1px solid #2c4968;border-radius:8px;padding:10px;text-align:left}} button{{cursor:pointer;margin-top:8px}} button:hover,button.active{{border-color:var(--accent);background:#173659}} .oracle{{display:flex;gap:8px;align-items:center;margin-top:14px;color:var(--muted)}} .oracle input{{width:auto}}
#canvas{{overflow:auto;padding:24px}} svg{{min-width:1060px;min-height:680px}} .edge{{stroke:var(--line);stroke-width:1.5}} .edge.temporal{{stroke:#283c52;stroke-dasharray:5 5}} .node rect{{fill:#10253c;stroke:#3b5978;stroke-width:1.5;rx:9}} .node{{cursor:pointer}} .node:hover rect,.node.selected rect{{stroke:var(--accent);stroke-width:3}} .node text{{fill:var(--text);font-size:11px}} .node .sub{{fill:var(--muted);font-size:10px}} .node.warning rect{{stroke:var(--warn)}} .node.error rect{{stroke:var(--error)}} .node.healed rect,.node.completed rect{{stroke:var(--ok)}} .node.oracle-cause rect{{fill:#4b2a10;stroke:var(--warn);stroke-width:4}} .node.oracle-symptom rect{{fill:#491c29;stroke:var(--error);stroke-width:4}} .node.oracle-solution rect{{fill:#103d32;stroke:var(--ok);stroke-width:4}}
h2{{font-size:14px;margin:0 0 12px}} .card{{background:#10253c;border:1px solid #29445f;border-radius:9px;padding:11px;margin:8px 0}} .score{{color:var(--accent);font-variant-numeric:tabular-nums}} .small{{font-size:12px;color:var(--muted)}} code{{color:#a9d1ff;word-break:break-all}} #query{{padding:11px;background:#091827;border-radius:8px;color:#c9d8e8}} @media(max-width:980px){{main{{grid-template-columns:1fr}} aside{{border:0!important}}}}
</style></head><body>
<header><h1>Causal-Topological RAG · Sistema comercial</h1><span class="badge">6 incidentes · oracle isolado</span><span class="small">Arestas sólidas = causais; tracejadas = temporais/comportamentais</span></header>
<main><aside><h2>Cenário</h2><select id="scenario"></select><label>Pergunta observada</label><div id="query"></div><button id="diagnose" class="active">Diagnosticar causa</button><button id="recover">Encontrar solução</button><label class="oracle"><input id="reveal" type="checkbox"> Revelar gabarito</label><div id="legend" class="small" style="margin-top:15px">O gabarito começa oculto. Clique em um nó para inspecionar apenas a observação pública.</div></aside>
<section id="canvas"><svg id="graph" viewBox="0 0 1120 700"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#37516e"/></marker></defs><g id="edges"></g><g id="nodes"></g></svg></section>
<aside><h2 id="result-title">Ranking</h2><div id="results"></div><h2 style="margin-top:20px">Observação selecionada</h2><div id="detail" class="small">Clique em um nó.</div></aside></main>
<script>const DATA={payload};
const byId=Object.fromEntries(DATA.graph.nodes.map(n=>[n.id,n])); const select=document.querySelector('#scenario'); let mode='diagnosis';
DATA.oracle.forEach(o=>select.add(new Option(o.title,o.scenario_id))); const short=s=>s.length>29?s.slice(0,27)+'…':s;
function current(){{return DATA.oracle.find(o=>o.scenario_id===select.value)}}
function draw(){{const o=current(), prefix='incident:'+o.scenario_id; const nodes=DATA.graph.nodes.filter(n=>n.correlation_id===prefix); const ids=new Set(nodes.map(n=>n.id)); const edges=DATA.graph.edges.filter(e=>ids.has(e.source)&&ids.has(e.target)); const causal=edges.filter(e=>e.kind==='causal'); const order=[]; let cursor=nodes.find(n=>!causal.some(e=>e.target===n.id)); while(cursor){{order.push(cursor); const e=causal.find(x=>x.source===cursor.id); cursor=e?byId[e.target]:null}} nodes.filter(n=>!order.includes(n)).forEach(n=>order.push(n));
 const pos={{}}; order.forEach((n,i)=>{{pos[n.id]={{x:40+(i%5)*210,y:70+Math.floor(i/5)*240}}}}); document.querySelector('#edges').innerHTML=edges.map(e=>`<line class="edge ${{e.kind}}" x1="${{pos[e.source].x+170}}" y1="${{pos[e.source].y+42}}" x2="${{pos[e.target].x}}" y2="${{pos[e.target].y+42}}" marker-end="url(#arrow)"/>`).join('');
 const reveal=document.querySelector('#reveal').checked; document.querySelector('#nodes').innerHTML=order.map(n=>{{let role=''; if(reveal) role=n.id===o.root_cause_event_id?' oracle-cause':n.id===o.symptom_event_id?' oracle-symptom':n.id===o.solution_event_id?' oracle-solution':''; const p=pos[n.id]; return `<g class="node ${{n.status||''}}${{role}}" data-id="${{n.id}}" transform="translate(${{p.x}},${{p.y}})"><rect width="174" height="84"/><text x="10" y="20">${{short(n.event_type)}}</text><text class="sub" x="10" y="40">${{n.domain}} · ${{n.status}}</text><text class="sub" x="10" y="60">${{short(n.detail||'')}}</text><text class="sub" x="10" y="75">${{n.id.slice(-8)}}</text></g>`}}).join(''); document.querySelectorAll('.node').forEach(el=>el.onclick=()=>showNode(el.dataset.id)); renderResults();}}
function renderResults(){{const o=current(); document.querySelector('#query').textContent=mode==='diagnosis'?o.diagnostic_query:o.recovery_query; document.querySelector('#diagnose').classList.toggle('active',mode==='diagnosis'); document.querySelector('#recover').classList.toggle('active',mode==='recovery'); const arm=mode==='diagnosis'?'ctrag_causal_frontier':'full_ctrag'; const hits=DATA.evaluation.details[o.scenario_id][mode][arm]; document.querySelector('#result-title').textContent=mode==='diagnosis'?'Causas ranqueadas':'Soluções ranqueadas'; document.querySelector('#results').innerHTML=hits.map((h,i)=>{{const n=byId[h.event_id]; return `<div class="card" data-result="${{h.event_id}}"><b>#${{i+1}} ${{n.event_type}}</b><div class="small">${{n.detail}}</div><div class="score">score ${{h.score.toFixed(3)}} · ${{h.causal_hops??'—'}} hops</div></div>`}}).join(''); document.querySelectorAll('[data-result]').forEach(el=>el.onclick=()=>showNode(el.dataset.result));}}
function showNode(id){{document.querySelectorAll('.node').forEach(n=>n.classList.toggle('selected',n.dataset.id===id)); const n=byId[id]; document.querySelector('#detail').innerHTML=`<div class="card"><b>${{n.event_type}}</b><p>${{n.detail}}</p><div>Status: ${{n.status}}<br>Domínio: ${{n.domain}}<br><code>${{n.id}}</code><br>${{n.timestamp}}</div></div>`}}
select.onchange=draw; document.querySelector('#reveal').onchange=draw; document.querySelector('#diagnose').onclick=()=>{{mode='diagnosis';draw()}}; document.querySelector('#recover').onclick=()=>{{mode='recovery';draw()}}; draw();</script></body></html>"""


def _report(config: CommercialExperimentConfig, evaluation: dict[str, Any]) -> str:
    summaries = evaluation["summary"]
    rows = [
        "| Tarefa | Método | N | Top-1 | Recall@3 | MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        rows.append(
            f"| {item['task']} | {item['arm']} | {item['n']} | "
            f"{item['top1_accuracy']:.3f} | {item[f'recall_at_{config.k}']:.3f} | {item['mrr']:.3f} |"
        )
    table = "\n".join(rows)
    return f"""# Experimento comercial de localização de causa v1

Status: evidência sintética exploratória. Gerador: `{GENERATOR}`.

## Pergunta

Um CT-RAG instrumentado com causalidade explícita entre eventos consegue localizar a primeira observação anômala que causou um erro comercial e recuperar a solução aplicada, mesmo na presença de documentos e simulações textualmente semelhantes?

## Protocolo

- seis incidentes atravessam vendas, estoque, pagamentos, fiscal, financeiro, compras, pricing, marketing, CRM, loyalty, fulfillment e delivery;
- o simulador define causa, sintoma e solução antes da recuperação;
- os eventos usam IDs opacos; papéis do gabarito não entram no texto nem nos metadados indexados;
- cada incidente contém distratores sem ligação causal que repetem a linguagem da pergunta;
- o diagnóstico recebe apenas o ID do sintoma, a pergunta pública, estados operacionais e arestas provenientes de `causation_id`;
- a `ctrag_causal_frontier` procura a primeira observação anômala em caminhos causais evidenciados;
- a solução é consultada no sentido causal futuro com o modo `RECOVERY`;
- configuração: seed `{config.seed}`, dimensão `{config.dimensions}`, máximo `{config.max_hops}` hops.

## Resultados

{table}

Os resultados completos por cenário, ranking e componentes de score estão em `results.json`. O arquivo `explorer.html` permite selecionar incidentes, inspecionar nós, executar as duas consultas predefinidas e revelar o gabarito somente depois do diagnóstico.

Há também um resultado negativo importante: o `full_ctrag` genérico não colocou a causa-raiz no Top-3 de nenhum dos seis cenários (MRR 0,228). Ele favorece ancestrais causalmente próximos, que respondem bem a “o que causou imediatamente?”, mas não necessariamente à pergunta operacional “onde começou a anomalia?”. A operação `ctrag_causal_frontier` explicita essa segunda semântica e foi desenhada e avaliada neste mesmo ensaio exploratório; sua generalização precisa de casos novos e congelados.

## Interpretação permitida

Este experimento testa **localização de causa-raiz dentro de telemetria causal instrumentada**. Ele mostra se o grafo preserva e torna navegável a cadeia que liga uma primeira anomalia ao sintoma e à recuperação.

Ele não demonstra descoberta causal a partir de logs correlacionais, não prova validade externa em empresas reais e não autoriza chamar toda precedência temporal de causa. As arestas causais deste ensaio vêm exclusivamente de `causation_id`; arestas temporais e comportamentais permanecem distintas.

## Critério inicial de sucesso

O marco é atingido quando `ctrag_causal_frontier` encontra a causa correta em Top-1 nos seis incidentes e o modo `full_ctrag` encontra a solução observada em Recall@3, sem vazamento do oracle. Esses resultados devem ser tratados como prova de execução do mecanismo, não como conclusão científica final.
"""


def run(output: Path, config: CommercialExperimentConfig | None = None) -> dict[str, Any]:
    config = config or CommercialExperimentConfig()
    output.mkdir(parents=True, exist_ok=True)
    observations, oracle = generate(config)
    topology = project(observations)
    evaluation = evaluate(topology, oracle, config)
    graph = _graph_payload(topology)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generator": GENERATOR,
        "status": "synthetic_exploratory",
        "claim_scope": "root-cause localization over supplied event-causation telemetry; not causal discovery",
        "config": {**asdict(config), "python_version": platform.python_version()},
        "information_contract": {
            "indexed": ["event_type", "timestamp", "payload", "causation_id", "execution identifiers", "status"],
            "withheld_from_retrieval": ["root-cause event id", "symptom role", "solution event id", "oracle roles"],
            "causal_edges": "projected only from explicit event causation_id",
        },
        **evaluation,
    }
    observation_text = "".join(_json(event.to_dict()).strip() + "\n" for event in observations)
    files = {
        "observations.ndjson": observation_text,
        "oracle.json": _json([asdict(item) for item in oracle]),
        "graph.json": _json(graph),
        "results.json": _json(result),
        "REPORT.md": _report(config, evaluation),
        "explorer.html": _explorer(graph, oracle, evaluation),
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR,
        "files": {
            name: {"bytes": len(content.encode()), "sha256": hashlib.sha256(content.encode()).hexdigest()}
            for name, content in sorted(files.items())
        },
    }
    (output / "manifest.json").write_text(_json(manifest), encoding="utf-8", newline="\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CT-RAG commercial-system root-cause experiment")
    parser.add_argument("--output", type=Path, default=Path("research/commercial-system-v1"))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--max-hops", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    result = run(args.output, CommercialExperimentConfig(args.seed, args.dimensions, args.max_hops, args.k))
    frontier = next(item for item in result["summary"] if item["arm"] == "ctrag_causal_frontier")
    print(
        f"Commercial experiment complete: {frontier['n']} incidents; "
        f"causal-frontier Top-1={frontier['top1_accuracy']:.3f}; output={args.output}"
    )


if __name__ == "__main__":
    main()
