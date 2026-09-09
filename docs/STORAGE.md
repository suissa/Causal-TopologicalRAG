# CT-RAG Storage Architecture

CT-RAG separates **semantic/causal meaning** from **storage technology**. A database is an implementation detail of a projection; it must not change what a causal edge, basin, attractor, retrieval score input, or event means.

## Authority boundary

```text
Authoritative Event Source
          |
          | projection
          v
+---------------------------+
| CT-RAG retrieval topology |
| nodes / edges / basins    |
+---------------------------+
          |
          +--> vector index
          +--> topology store
          +--> terrain overlay
```

The event source is authoritative. SQLite, graph databases, vector databases and other CT-RAG stores are rebuildable retrieval projections.

A storage adapter MUST NOT promote temporal adjacency to causality, discard causal provenance, rewrite confidence, or merge observed/inferred/hypothesized causal evidence.

## Interfaces

`src/ctrag/storage.py` defines five storage contracts plus the semantic view consumed by retrieval.

### `TopologyView`

The read/navigation interface required by CT-RAG. It exposes nodes, neighborhoods, directed distances, causal-path evidence, basin membership and basin affinity.

`CausalTopology` is the in-memory reference implementation. A remote or database-backed topology may implement the same structural protocol without changing retrieval semantics.

### `MemoryStore`

```python
put_memory(node)
get_memory(node_id)
list_memories()
```

Persists `MemoryNode`, including timestamp, metadata and embedding.

### `VectorIndex`

```python
upsert_vector(node_id, vector)
delete_vector(node_id)
search_vector(vector, k=10)
```

`InMemoryVectorIndex` is the deterministic stdlib baseline. FAISS/Qdrant adapters can implement the same contract later.

### `TopologyStore`

```python
save_topology(topology)
load_topology()
```

A topology store must round-trip:

- memory nodes;
- every edge kind;
- causal provenance;
- causal confidence;
- edge weight;
- edge evidence;
- provenance metadata;
- explicit/discovered attractor descriptors.

### `EventSource`

```python
read_events()
```

This interface deliberately sits outside the retrieval projection stores. `ListEventSource` is a test/reference adapter. EventStoreDB or other authoritative event stores can implement it without making them topology databases.

### `TerrainStore`

```python
save_terrain(terrain)
load_terrain(topology)
```

The terrain remains a **non-authoritative navigation overlay**. Persisting reinforcement or erosion does not modify the historical topology edges.

## SQLite adapter

`SQLiteCTStore` is the first persistent local adapter and uses only Python's `sqlite3` module.

It stores four independent projection tables:

```text
memories
edges
attractors
terrain_state
```

The canonical model serializers (`MemoryNode.to_dict`, `Edge.to_dict`, `AttractorDescriptor.to_dict`) are used before JSON is persisted. Reload uses the matching `from_dict` contracts.

This keeps SQLite from defining CT-RAG semantics; it only stores the canonical representation.

## Example

```python
from ctrag import SQLiteCTStore

store = SQLiteCTStore("ctrag.sqlite")
store.save_topology(topology)
store.save_terrain(terrain)

reloaded_topology = store.load_topology()
reloaded_terrain = store.load_terrain(reloaded_topology)
```

A reloaded topology can be passed directly to `CTRetriever`.

## Conformance requirement

A persistent adapter is conformant only if the in-memory and reloaded representations produce equivalent retrieval semantics.

The SQLite conformance suite verifies:

1. node serialization equality;
2. edge serialization equality;
3. provenance/confidence/evidence preservation;
4. attractor descriptor equality;
5. exact terrain transition-count/influence restoration;
6. identical CT-RAG ranking, score components and causal hop counts before and after reload.

This is the key architectural rule:

> **Changing the database may change performance and scale, but must not change the meaning of the terrain.**

## Future adapters

The interfaces intentionally permit later experiments with:

- FAISS / Qdrant for `VectorIndex`;
- Neo4j / CozoDB for topology persistence/navigation;
- EventStoreDB for `EventSource`;
- DuckDB for experiment/analytics storage.

Those adapters require their own conformance tests against the reference in-memory semantics before they can be considered interchangeable.
