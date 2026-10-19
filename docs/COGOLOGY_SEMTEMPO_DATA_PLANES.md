# Cogology SemTempo: Metrics and Traces Data Planes

Status: proposed architecture contract  
Scope: WhatsApp-first conversational systems, initially commerce and delivery  
Implementation target: define stable event contracts before introducing storage-specific code.

## Purpose

Cogology SemTempo GraphDB combines temporal, semantic, behavioral, and topological projections without conflating conversational observations with operational facts. The first ingestion source may be only WhatsApp conversations. Operational systems can be connected later as independent evidence sources.

The system is organized around two independent data planes:

- **Metrics Data Plane (MDP):** observations and measurements derived from conversations and their media.
- **Traces Data Plane (TDP):** observed operational steps that reconstruct a business context from initiation to an explicit terminal condition.

The planes may be correlated by stable context identifiers, but neither plane silently overwrites or manufactures facts in the other.

## Architecture

    WhatsApp messages / media
              |
              v
      Immutable observations
              |
              v
      Normalization + extraction
              |
              +----> Metrics Data Plane
              |        - linguistic/semantic measurements
              |        - interaction and timing measurements
              |        - intent/context hypotheses
              |        - behavior and trajectory projections
              |
              +----> Occasioning/context projection
                           |
                           v
                 Temporal substrate
                 (versioned snapshots)
                           |
                           v
                 Topological projection
                           |
                           v
                 Cogology SemTempo GraphDB
                           ^
                           |
    Operational events ----+----> Traces Data Plane
    (payment, preparation, dispatch,
    delivery, delivery-code confirmation)

The EventStore or equivalent immutable source remains authoritative for recorded events. A persistent index (including a possible Persistent Treap) is a rebuildable temporal projection, not the source of historical truth. Graph/topology projections are also rebuildable.

## Plane contracts

### Metrics Data Plane

The MDP answers: **what was observable in the interaction, and how did the interaction evolve?**

Initial metric families:

- message counts, turn-taking, inter-message latency, session gaps, and channel/media type;
- language features such as lexical choices, semantic similarity/change, reformulations, corrections, confirmations, negations, ambiguity, and unresolved questions;
- explicit intent/entity extraction with source spans and extraction confidence;
- context creation, continuation, switching, reopening, abandonment, and completion hypotheses;
- audio transcription and measurable audio properties when available and consented to;
- recurrence and transition patterns over a BehaviorTrajectoryID.

Every derived item must retain provenance to the source message/media and the extractor version. Store observations separately from features, patterns, and hypotheses. A delay, wording choice, or audio characteristic is not by itself proof of an emotional or cognitive state.

### Traces Data Plane

The TDP answers: **what operational steps were observed in the lifecycle of this business context?**

For delivery, a trace may contain:

    context_opened -> order_created -> payment_confirmed
      -> preparation_completed -> courier_received
      -> customer_received -> delivery_code_confirmed -> context_closed

This is an illustrative lifecycle, not a claim that every integration exposes every event. Missing steps remain missing; the projector must not synthesize them from temporal adjacency.

For the initial WhatsApp-first commerce flow, the customer providing the delivery code to the courier is the terminal confirmation available to the system. The context closes only when that confirmation is recorded and validated according to the configured delivery protocol. Payment confirmation alone does not close a delivery context.

The terminal event should record at least the context/order identifier, event time, confirmation outcome, verification method, and source provenance. Do not store the raw delivery code in graph projections or metrics; store a verification result or a protected reference according to the security policy.

## Canonical records

### Observation

- observation_id: stable, idempotency-safe identifier
- source_kind: WhatsApp text, audio, image, or operational event
- source_ref: protected reference to the original artifact
- occurred_at: source event time, if known
- observed_at: ingestion/observation time
- context_id: conversation/business context when resolved
- actor_ref: pseudonymous actor reference
- payload_digest: integrity fingerprint; not a replacement for the source
- provenance: connector, message/event identifier, and ingestion version

### Derived measurement or hypothesis

- derived_id, subject_ref, context_id
- kind: metric, feature, pattern, intent hypothesis, state hypothesis, or transition hypothesis
- value: typed value with units where applicable
- valid_time: when it applies to the modeled world
- recorded_time: when the system produced/recorded it
- source_observation_ids: evidence references
- extractor and extractor_version
- confidence: calibrated or explicitly uncalibrated score
- epistemic_status: observed, derived, inferred, or hypothesized

### Operational trace event

- event_id, context_id, event_kind
- occurred_at, recorded_at
- actor_ref / system_ref
- causation_id only when explicit evidence supplies it
- correlation_id for shared context, never treated as causality
- source_provenance and validation outcome

## Occasioning and temporal identity

An **occasion** is a bounded, evolving context in which messages, actors, intentions, and operational events may be related. A conversation is a source stream; it is not automatically identical to a business occasion. One conversation may contain multiple occasions, and one occasion may span multiple conversations.

Recommended identifiers:

- ConversationID: source conversation/thread identity
- OccasionID: modeled context identity
- BehaviorID: one behavior instance or local behavioral unit
- BehaviorTrajectoryID: ordered sequence of behavior instances
- TraceID: operational execution/lifecycle identity
- TransitionID: one observed or hypothesized state/context transition

Use explicit links between these identifiers. Never infer identity merely because timestamps overlap or two messages are adjacent.

## Temporal and topological projections

Each projection must be reproducible from authoritative records and versioned inputs.

- Preserve **valid time** (when an event/claim applies) separately from **recorded time** (when the system learned it).
- Corrections append a new version or retraction; they do not silently rewrite historical observations.
- Persistent Treap is a candidate for efficient persistent snapshots/range queries, subject to workload benchmarks. It is not mandated as the authoritative store.
- Graph edges distinguish semantic, temporal, behavioral, correlation, execution, and causal relations.
- Temporal adjacency, shared context, semantic similarity, and correlation do **not** independently justify a causal edge.
- Attractors/topological scores are navigational projections, not facts about a person or the operational world.

## Cross-plane joins

Cross-plane relationships must be explicit and typed:

- same_context: evidence that two records refer to the same occasion/order;
- supports: an operational event validates a particular conversational outcome;
- precedes: temporal ordering only;
- correlates_with: association without causal assertion;
- caused_by: only when supported by explicit causation evidence or a separately documented causal inference with provenance and uncertainty.

Example: a WhatsApp message containing a delivery confirmation and a validated delivery-code event may be linked to the same OccasionID. This does not establish that the message caused the delivery event.

## Privacy and epistemic safeguards

- Minimize retained message/media content; prefer protected source references and derived measurements where feasible.
- Use pseudonymous actor identifiers in analytical projections and enforce tenant/context boundaries.
- Separate observable features from inferred cognitive or emotional hypotheses.
- State hypotheses as uncertain, versioned claims with supporting evidence; never persist them as unquestioned identity facts.
- Provide retention/deletion workflows that cover source artifacts and rebuildable projections.
- Never expose raw delivery codes, authentication secrets, or unnecessary personal data in graph labels, logs, or benchmark fixtures.

## Initial implementation sequence

1. Define and test the canonical Observation, DerivedMeasurement, and OperationalTraceEvent contracts.
2. Build a deterministic WhatsApp fixture with text, audio-transcription, correction, context-switch, and delivery-confirmation cases.
3. Implement idempotent observation ingestion and provenance-preserving MDP extraction.
4. Implement TDP lifecycle projection with delivery-code confirmation as the configured terminal event.
5. Add cross-plane linking without causal promotion.
6. Add versioned temporal snapshots; benchmark Persistent Treap against a simpler baseline before selecting it.
7. Add topology/GraphDB projection and replay-equivalence tests.

## Acceptance criteria

- Replaying the same source records produces identical IDs and projection contents.
- Out-of-order ingestion does not silently change event time or invent causality.
- Valid time and recorded time remain distinguishable.
- Payment does not close a delivery trace; only a validated terminal confirmation closes it in the configured delivery workflow.
- A missing delivery-code confirmation leaves the trace open or explicitly unresolved.
- Metrics derived from messages preserve source and extractor provenance.
- Cross-plane correlation does not become a causal edge without qualifying evidence.
- Rebuilding temporal and graph projections from authoritative records reproduces the same result.
- Fixtures contain synthetic/pseudonymous data and no real delivery codes or private customer content.
