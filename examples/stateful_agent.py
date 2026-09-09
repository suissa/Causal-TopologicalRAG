from datetime import datetime, timedelta, timezone

from ctrag import CausalTopology, CTRetriever, EventProjector, EventRecord, QueryMode


def main() -> None:
    topology = CausalTopology()
    projector = EventProjector(topology)
    now = datetime.now(timezone.utc)

    events = [
        EventRecord(
            event_id="e1",
            event_type="Checkout.PaymentAuthorized",
            timestamp=now,
            execution_id="checkout-42",
            intent_id="checkout",
            action_id="authorize-payment",
            status="ok",
        ),
        EventRecord(
            event_id="e2",
            event_type="Checkout.InventoryReservationFailed",
            timestamp=now + timedelta(seconds=1),
            causation_id="e1",
            execution_id="checkout-42",
            intent_id="checkout",
            action_id="reserve-inventory",
            status="error",
            payload={"reason": "stock_changed"},
        ),
        EventRecord(
            event_id="e3",
            event_type="Checkout.InventoryReplanned",
            timestamp=now + timedelta(seconds=2),
            causation_id="e2",
            execution_id="checkout-42",
            intent_id="checkout",
            action_id="heal-inventory",
            status="ok",
        ),
        EventRecord(
            event_id="e4",
            event_type="Checkout.Completed",
            timestamp=now + timedelta(seconds=3),
            causation_id="e3",
            execution_id="checkout-42",
            intent_id="checkout",
            action_id="complete-checkout",
            status="ok",
        ),
    ]

    for event in events:
        projector.ingest(event)

    topology.register_attractor("e4")
    retriever = CTRetriever(topology)

    print("\nWHY did inventory reservation fail?\n")
    for hit in retriever.search(
        "why did the inventory reservation fail?",
        mode=QueryMode.WHY,
        anchor_ids=["e2"],
        k=3,
    ):
        print(
            f"{hit.node.id:>2} score={hit.score:.3f} "
            f"causal_hops={hit.causal_hops} components={hit.components}"
        )

    print("\nBasin of checkout completion:")
    print(sorted(topology.basin("e4")))


if __name__ == "__main__":
    main()
