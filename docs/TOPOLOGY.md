# Topologia do Causal-Topological RAG

Este documento explica **o que a topologia representa no CT-RAG, como ela é construída, quais são as suas partes e como essas partes participam da recuperação de memória**.

A topologia não é apenas uma visualização do histórico. Ela é a estrutura navegável que permite ao CT-RAG responder perguntas que um RAG puramente vetorial não consegue representar diretamente, como:

- **por que** um estado aconteceu;
- **o que aconteceu antes** dele;
- **o que normalmente acontece depois**;
- qual trajetória levou a uma falha;
- qual trajetória histórica levou à recuperação;
- em qual região comportamental do sistema o estado se encontra;
- quais estados convergem para o mesmo resultado;
- onde duas trajetórias divergem.

Em termos simples:

> A topologia do CT-RAG transforma memórias independentes em um terreno navegável de estados e relações.

---

## 1. O que significa "topologia" no CT-RAG

No CT-RAG, a topologia é um **grafo direcionado e tipado**.

Os vértices representam estados ou memórias. As arestas representam relações explicitamente classificadas entre esses estados.

```text
MemoryNode ---- Edge ----> MemoryNode
```

Por exemplo:

```text
Payment.Authorized
        |
        | causal
        v
Inventory.ReservationFailed
        |
        | behavioral / temporal
        v
Inventory.RetryStarted
        |
        | causal
        v
Inventory.Recovered
```

Essa estrutura permite distinguir duas coisas que em um RAG comum costumam ser misturadas:

1. **similaridade de conteúdo**;
2. **relação estrutural entre acontecimentos**.

Dois eventos podem ter textos muito parecidos e não possuir nenhuma relação causal. Da mesma forma, dois eventos com textos muito diferentes podem fazer parte da mesma trajetória causal.

Por isso, no CT-RAG:

> semântica responde principalmente **"o que parece relacionado?"**;
>
> topologia responde principalmente **"como estes estados estão relacionados no sistema?"**.

---

## 2. A topologia é uma projeção, não o histórico autoritativo

Quando o CT-RAG é alimentado por Event Sourcing, o Event Store continua sendo a fonte de verdade.

A topologia é uma **projeção derivada para recuperação e navegação**.

```text
Authoritative Event History
          |
          | projection
          v
   CT-RAG Topology
          |
          | retrieval/navigation
          v
     Context for LLM
```

Isso é uma decisão arquitetural importante.

O CT-RAG não deve reescrever o significado do evento original nem transformar sua projeção em uma nova fonte autoritativa.

A topologia pode ser reconstruída, recalculada, persistida em outro storage ou receber novos índices sem alterar o histórico original.

---

# 3. As partes fundamentais da topologia

A implementação atual possui cinco conceitos estruturais principais:

```text
MemoryNode
Edge
CausalPath
AttractorDescriptor
Basin
```

Eles formam camadas diferentes do terreno.

---

## 3.1 MemoryNode — um ponto do terreno

`MemoryNode` representa uma memória, evento, estado ou observação recuperável.

Campos principais:

```python
MemoryNode(
    id=...,
    text=...,
    timestamp=...,
    metadata=...,
    embedding=...,
    is_attractor=...,
)
```

### `id`

É a identidade estável do estado dentro da topologia.

Quando a topologia é criada a partir de eventos, normalmente:

```text
MemoryNode.id = event_id
```

O ID deve ser único.

---

### `text`

É a representação textual usada pelos componentes semântico e lexical do retriever.

No projector atual, ele é construído a partir de informações como:

```text
event_type
status
payload
```

Exemplo:

```text
Inventory.ReservationFailed status=error sku=ABC stock=0
```

A topologia não depende apenas desse texto. Ele é um dos sinais do ranking.

---

### `timestamp`

Representa quando aquele estado/evento ocorreu.

O timestamp participa do sinal temporal, mas:

> timestamp nunca é interpretado automaticamente como causalidade.

`A aconteceu antes de B` não significa `A causou B`.

Essa é uma das invariantes centrais do CT-RAG.

---

### `metadata`

Armazena identidade e contexto estrutural do evento.

Exemplos usados atualmente:

```text
execution_id
intent_id
actor_id
action_id
correlation_id
causation_id
status
```

Esses dados permitem calcular afinidade comportamental e reconstruir trajetórias.

---

### `embedding`

É a representação vetorial usada para similaridade semântica.

A topologia pode funcionar com diferentes providers de embedding.

O embedding é uma característica do nó, não uma aresta causal.

---

## 3.2 Edge — uma relação tipada

Uma `Edge` conecta dois nós:

```text
source -> target
```

Sua identidade estrutural é:

```text
(source, target, kind, provenance)
```

Isso permite que dois nós tenham, por exemplo, uma relação temporal e uma relação causal ao mesmo tempo sem que essas relações sejam confundidas.

---

# 4. Os quatro tipos de aresta

O CT-RAG separa explicitamente quatro tipos de relação.

```python
EdgeKind.SEMANTIC
EdgeKind.CAUSAL
EdgeKind.TEMPORAL
EdgeKind.BEHAVIORAL
```

Essa separação é fundamental.

---

## 4.1 CAUSAL

Representa uma relação causal declarada ou inferida explicitamente.

```text
A --causal--> B
```

Significado:

> existe evidência de que A participa da explicação causal de B.

Toda aresta causal precisa possuir `provenance`.

Exemplo:

```python
Edge(
    source="payment-authorized",
    target="stock-failed",
    kind=EdgeKind.CAUSAL,
    provenance=CausalProvenance.EXECUTION,
    confidence=1.0,
)
```

Uma aresta causal pode carregar ainda:

```text
confidence
evidence
provenance_metadata
weight
```

---

## 4.2 TEMPORAL

Representa ordem observada.

```text
A --temporal--> B
```

Significado:

> B ocorreu depois de A dentro da sequência considerada.

Não significa que A causou B.

Por exemplo:

```text
User.LoggedIn
      |
      | temporal
      v
Weather.CacheRefreshed
```

A ordem pode ser verdadeira sem haver causalidade.

---

## 4.3 BEHAVIORAL

Representa continuidade dentro de uma mesma execução ou fluxo comportamental.

```text
A --behavioral--> B
```

Pode significar, por exemplo:

```text
mesma execution_id
mesmo fluxo de execução
mesma trajetória observada
```

Ela é importante porque nem toda continuidade de comportamento possui uma prova causal forte o suficiente para virar uma aresta `CAUSAL`.

---

## 4.4 SEMANTIC

Representa associação semântica explícita no grafo.

```text
A --semantic--> B
```

O tipo existe no modelo para relações semânticas materializadas.

Entretanto, na implementação atual, a similaridade semântica usada no ranking é normalmente calculada diretamente pelos embeddings e **não precisa criar automaticamente uma `SEMANTIC` edge para cada vizinhança vetorial**.

Isso evita transformar o grafo causal em uma cópia completa do índice vetorial.

---

# 5. Proveniência causal

Uma relação causal precisa responder não apenas:

> "qual é a confiança?"

mas também:

> "de onde veio essa afirmação?"

Por isso o CT-RAG possui `CausalProvenance`.

Atualmente:

| Proveniência | Significado | Fator atual |
| --- | --- | ---: |
| `execution` | evidência diretamente observada na execução | 1.00 |
| `workflow` | relação definida pelo workflow | 0.95 |
| `dependency` | dependência estrutural conhecida | 0.90 |
| `event` | `causation_id`/evidência explícita de evento | 0.90 |
| `inferred` | relação inferida | 0.60 |
| `hypothesized` | hipótese causal ainda fraca | 0.35 |

A razão dessa diferenciação é evitar que uma hipótese passe a ter a mesma autoridade de uma relação observada.

```text
observed execution causality
          !=
inferred causality
          !=
hypothesized causality
```

Essa distinção é preservada durante o retrieval.

---

# 6. Como a topologia é criada

Existem atualmente duas formas principais.

---

## 6.1 Construção programática direta

É possível montar a topologia manualmente:

```python
from ctrag import (
    CausalTopology,
    MemoryNode,
    Edge,
    EdgeKind,
    CausalProvenance,
)

topology = CausalTopology()

topology.add_node(MemoryNode(
    id="a",
    text="Payment authorized",
))

topology.add_node(MemoryNode(
    id="b",
    text="Inventory reservation failed",
))

topology.add_edge(Edge(
    source="a",
    target="b",
    kind=EdgeKind.CAUSAL,
    provenance=CausalProvenance.EXECUTION,
))
```

Essa forma é útil para:

- testes;
- datasets sintéticos;
- importação de grafos externos;
- projeções customizadas;
- experimentos de pesquisa.

---

## 6.2 Construção por Event Sourcing

O caminho mais importante para agentes stateful é o `EventProjector`.

```text
EventRecord
    |
    v
EventProjector
    |
    +--> MemoryNode
    +--> CAUSAL edges
    +--> TEMPORAL edges
    +--> BEHAVIORAL edges
    v
CausalTopology
```

Cada evento vira um `MemoryNode`.

---

# 7. Como um EventRecord vira topologia

O `EventRecord` atual possui:

```text
event_id
event_type
timestamp
payload
causation_id
correlation_id
execution_id
intent_id
actor_id
action_id
status
```

Considere:

```json
{
  "event_id": "e2",
  "event_type": "Inventory.ReservationFailed",
  "causation_id": "e1",
  "execution_id": "checkout-42",
  "intent_id": "checkout",
  "status": "error"
}
```

O projector cria aproximadamente:

```text
Node e2
  text:
    Inventory.ReservationFailed status=error

  metadata:
    causation_id=e1
    execution_id=checkout-42
    intent_id=checkout
    status=error
```

Se `e1` já existe na topologia, o projector cria:

```text
e1 --CAUSAL(EVENT)--> e2
```

Se `e1` for também o evento anterior da mesma execução:

```text
e1 --TEMPORAL--> e2
e1 --BEHAVIORAL--> e2
```

Portanto podem existir três relações entre os mesmos estados:

```text
                 CAUSAL
              +--------->
              |
e1 -----------+---------> e2
              |
              +--------->
             TEMPORAL
             BEHAVIORAL
```

Elas não são redundantes. Cada uma possui significado diferente.

---

# 8. Regra fundamental: sequência não cria causalidade

Considere uma execução:

```text
e1
 |
 v
e2
 |
 v
e3
```

O simples fato de os eventos terem ocorrido nessa ordem permite criar relações de sequência:

```text
e1 --TEMPORAL--> e2
e2 --TEMPORAL--> e3
```

E, dentro de uma mesma execução observada:

```text
e1 --BEHAVIORAL--> e2
e2 --BEHAVIORAL--> e3
```

Mas o CT-RAG **não cria** automaticamente:

```text
e1 --CAUSAL--> e2
```

Para isso é necessária evidência causal explícita ou uma inferência que permaneça identificada como `INFERRED`/`HYPOTHESIZED`.

Essa regra evita um erro comum:

```text
post hoc ergo propter hoc
```

ou seja, concluir causalidade apenas porque uma coisa ocorreu antes da outra.

---

# 9. Estrutura interna do grafo

`CausalTopology` mantém atualmente três estruturas fundamentais:

```python
nodes: dict[str, MemoryNode]
_out: dict[str, list[Edge]]
_in: dict[str, list[Edge]]
```

Conceitualmente:

```text
nodes
  id -> MemoryNode

_out
  source -> outgoing edges

_in
  target -> incoming edges
```

Isso permite navegar eficientemente nas duas direções.

Por exemplo, para `WHY` é importante atravessar o grafo para trás:

```text
cause <- cause <- current state
```

Para `WHAT_NEXT`, é importante atravessar para frente:

```text
current state -> consequence -> consequence
```

---

# 10. Neighborhood e distances

A topologia oferece navegação por vizinhança.

```python
topology.neighborhood(
    node_id,
    direction="in" | "out" | "both",
    kinds={...},
    max_hops=4,
)
```

E cálculo de distâncias em hops:

```python
topology.distances(...)
```

Exemplo:

```text
A -> B -> C -> D
```

Partindo de `A`:

```text
distance(A) = 0
distance(B) = 1
distance(C) = 2
distance(D) = 3
```

Essa distância estrutural é diferente de distância vetorial.

---

# 11. CausalPath — uma trajetória causal explicável

Uma consulta causal não deve retornar apenas um número.

Por isso o CT-RAG representa a melhor trajetória encontrada como `CausalPath`.

Exemplo:

```text
Payment.Authorized
      |
      v
Inventory.LockRequested
      |
      v
Inventory.ReservationFailed
```

Um `CausalPath` mantém:

```text
anchor_id
candidate_id
direction
nodes
edges
best_confidence
aggregate_confidence
evidence
provenances
```

Assim, o resultado consegue explicar:

> "Este estado foi recuperado porque existe esta trajetória causal ligando-o ao estado âncora."

---

# 12. Confiança de uma trajetória causal

Para cada aresta causal, a contribuição atual é:

```text
edge.confidence
× min(edge.weight, 1)
× provenance_factor
```

A confiança de uma trajetória é o produto das suas arestas.

```text
path_confidence = Π edge_factor
```

Isso faz trajetórias longas ou compostas por evidências fracas perderem confiança naturalmente.

---

## 12.1 Múltiplas trajetórias

Um mesmo estado pode estar conectado ao anchor por mais de um caminho.

```text
        B
       / \
A ----   ---- D
       \ /
        C
```

O CT-RAG preserva o melhor caminho para explicação e agrega as evidências dos caminhos encontrados usando noisy-OR:

```text
aggregate = 1 - Π(1 - confidence_i)
```

Portanto:

- `best_confidence` explica a melhor trajetória individual;
- `aggregate_confidence` representa a força combinada das múltiplas trajetórias.

---

# 13. Attractors — estados para os quais trajetórias convergem

Um `Attractor` representa um estado ou região terminal/relevante para o qual trajetórias do sistema convergem.

Exemplos conceituais:

```text
Order.Completed
Payment.Recovered
HumanIntervention.Required
Execution.Failed
Agent.GoalSatisfied
```

Um attractor não precisa significar "sucesso".

Ele significa:

> um estado estruturalmente importante como destino de trajetórias.

Atualmente attractors podem ser registrados explicitamente:

```python
topology.register_attractor(
    "recovered",
    confidence=1.0,
    origin="manual",
    metadata={"outcome": "recovery"},
)
```

O `AttractorDescriptor` contém:

```text
node_id
confidence
origin
metadata
```

Em fases posteriores, o mesmo contrato permite distinguishir attractors:

```text
manual
discovered
empirical
workflow-defined
```

sem misturá-los.

---

# 14. Basin — a região que converge para um attractor

Um basin é o conjunto de estados que consegue chegar a um attractor usando a topologia causal/comportamental considerada.

Formalmente, para um attractor `A`:

```text
B(A) = { x | x alcança A por relações CAUSAL/BEHAVIORAL }
```

Na implementação atual, o basin é encontrado fazendo uma travessia reversa a partir do attractor.

Exemplo:

```text
A -> B -> C -> Success
         \
          -> D -> Failure
```

Basin de `Success`:

```text
{A, B, C, Success}
```

Basin de `Failure`:

```text
{A, B, D, Failure}
```

Observe que:

```text
A e B pertencem aos dois basins
```

Isso é esperado. Antes do ponto de divergência, ambos os resultados ainda eram alcançáveis.

---

# 15. Basin membership

Para um estado qualquer:

```python
topology.basin_memberships(node_id)
```

retorna os attractors alcançáveis daquele estado.

Exemplo:

```text
checkout-started
      |
      v
payment-authorized
     / \
    /   \
   v     v
success failure
```

Antes da bifurcação:

```text
memberships(payment-authorized)
= {success, failure}
```

Depois da bifurcação:

```text
memberships(success-path-node)
= {success}
```

Essa informação é particularmente útil para detectar regiões de divergência.

---

# 16. Shared basin affinity

Dois estados podem ser estruturalmente relacionados por convergirem para o mesmo attractor, mesmo que seus textos sejam muito diferentes.

```text
X -------->
            Recovery
Y -------->
```

O CT-RAG calcula `shared_basin_affinity`.

O objeto retornado informa:

```text
left_id
right_id
shared_attractors
score
```

Portanto a pontuação não é opaca: é possível saber **qual attractor justificou a afinidade**.

Na implementação atual:

```text
shared basin score
= 0.6 × maior confiança dos attractors compartilhados
```

O fator `0.6` preserva o comportamento do benchmark original enquanto o descriptor permite evolução posterior.

---

# 17. Basin boundary

A boundary representa a fronteira de uma região de atração.

Um nó pertence à boundary quando está dentro do basin, mas possui conexão causal/comportamental com um estado fora dele.

Exemplo:

```text
             -> Success
           /
A -> B -> C
           \
             -> Failure
```

Para o basin de `Success`, `C` é um candidato natural a boundary porque toca uma trajetória que também leva para fora daquela região.

API:

```python
topology.basin_boundary(attractor_id)
```

Essa estrutura será útil para:

- localizar pontos de decisão;
- detectar divergências;
- comparar sucesso vs. falha;
- estudar transições entre regiões do terreno.

---

# 18. Neighboring basins

Dois basins são considerados vizinhos quando:

1. compartilham estados; ou
2. possuem conexão pela boundary.

API:

```python
topology.neighboring_basins(attractor_id)
```

Isso cria uma visão de nível superior:

```text
node graph
   |
   v
basin graph
```

Exemplo:

```text
[Normal Operation Basin]
          |
          v
[Retry Basin]
          |
          v
[Recovered Basin]

          or

[Human Intervention Basin]
```

O objetivo futuro é permitir navegar não apenas entre eventos, mas entre **regiões comportamentais do sistema**.

---

# 19. O que significa "terreno"

No CT-RAG, "terreno" é uma metáfora operacional para a combinação de:

```text
nós
+ relações direcionadas
+ distâncias
+ confiança causal
+ trajetórias
+ attractors
+ basins
+ boundaries
+ frequência/força de transição futura
```

A ideia é que o histórico de execução deixe de ser visto como uma lista plana:

```text
e1
e2
e3
e4
e5
```

E passe a ser entendido como uma estrutura navegável:

```text
                    Recovery
                   /
Start -> Validate -> Retry
   \               \
    \               Failure
     -> Alternate -> Human
```

É essa estrutura que dá origem à parte **Topological** do nome CT-RAG.

---

# 20. Como a topologia participa do retrieval

O pipeline conceitual do CT-RAG é:

```text
Query
  |
  v
Semantic/Lexical search
  |
  v
Anchor state(s)
  |
  v
Topology expansion
  |
  +--> causal paths
  +--> temporal neighborhood
  +--> behavioral neighborhood
  +--> basin membership
  +--> attractors
  |
  v
Multi-signal reranking
  |
  v
Context
```

A topologia não substitui embeddings.

Ela adiciona informação que embeddings não codificam de maneira confiável.

---

# 21. Componentes atuais do score

O `CTRetriever` combina atualmente:

```text
semantic
lexical
causal
topological
temporal
behavioral
```

Conceitualmente:

```text
FinalScore(node)
 = ws * Semantic(node)
 + wl * Lexical(node)
 + wc * Causal(node)
 + wt * Topological(node)
 + wtime * Temporal(node)
 + wb * Behavioral(node)
```

Os pesos dependem do modo da consulta.

Isso é importante porque a mesma topologia deve ser interpretada de maneira diferente conforme a pergunta.

---

# 22. WHY muda a direção da navegação

Uma consulta:

```text
WHY did this fail?
```

procura principalmente ancestrais.

```text
cause -> cause -> current failure
```

Logo:

```text
direction = incoming
```

O retriever percorre causalidade para trás.

---

# 23. WHAT_NEXT muda a direção oposta

Uma consulta:

```text
WHAT_NEXT from this state?
```

procura descendentes.

```text
current state -> consequence -> next state
```

Logo:

```text
direction = outgoing
```

A mesma topologia suporta as duas perguntas porque as relações são direcionadas.

---

# 24. RECOVERY usa estrutura de trajetória

Para uma falha, recuperar semanticamente "documentos sobre erro" não é suficiente.

O objetivo é encontrar algo como:

```text
Failure
  |
  v
Retry
  |
  v
Fallback
  |
  v
Recovered
```

E comparar com trajetórias históricas semelhantes.

É justamente a existência das relações estruturais que permite representar essa sequência como uma trajetória recuperável.

---

# 25. COUNTERFACTUAL não significa causalidade inventada

O modo `COUNTERFACTUAL` pode usar o terreno para encontrar divergências observadas, por exemplo:

```text
              -> path A -> success
state X -----|
              -> path B -> failure
```

Mas a topologia, sozinha, não prova um contrafactual causal do tipo:

> "se B não tivesse ocorrido, A certamente teria acontecido".

O CT-RAG deve preservar a distinção entre:

```text
observed trajectory
inferred relation
hypothesis
counterfactual claim
```

A topologia oferece evidência estrutural para análise; ela não transforma associação observacional em identificação causal automaticamente.

---

# 26. Por que não usar apenas GraphRAG

O diferencial da topologia do CT-RAG não é simplesmente "usar um grafo".

O grafo é semântico em seu contrato.

Cada relação possui um significado específico:

```text
causal != temporal != behavioral != semantic
```

Além disso, causalidade possui:

```text
provenance
confidence
evidence
path aggregation
```

E o terreno possui:

```text
attractors
basins
boundaries
neighboring basins
```

Portanto a topologia é construída para representar **trajetórias de execução e dinâmica comportamental**, não apenas relações genéricas entre documentos.

---

# 27. Exemplo completo

Considere uma execução de checkout:

```text
E1 Checkout.Started
E2 Payment.Authorized
E3 Inventory.ReservationFailed
E4 Inventory.RetryStarted
E5 Inventory.Reserved
E6 Checkout.Completed
```

Com causation IDs:

```text
E1 -> E2
E2 -> E3
E3 -> E4
E4 -> E5
E5 -> E6
```

A projeção pode produzir:

```text
E1 --causal--> E2 --causal--> E3 --causal--> E4 --causal--> E5 --causal--> E6
 |             |             |             |             |
 +--temporal---+--temporal---+--temporal---+--temporal---+
 |             |             |             |             |
 +behavioral---+behavioral---+behavioral---+behavioral----+
```

Se `E6` for registrado como attractor:

```text
Attractor: Checkout.Completed
```

O basin de `E6` contém os estados que convergem para ele:

```text
{E1, E2, E3, E4, E5, E6}
```

Agora imagine outra execução:

```text
E3 -> F4 HumanIntervention.Required
```

com `F4` sendo outro attractor.

O estado equivalente a `E3` pode pertencer a dois basins:

```text
Checkout.Completed
HumanIntervention.Required
```

Isso revela que `E3` está em uma região de divergência importante do terreno.

Uma consulta de recuperação pode então procurar não apenas eventos parecidos com `E3`, mas:

> trajetórias que partiram desta região e terminaram no basin `Checkout.Completed`.

Essa é a diferença prática entre recuperar **memórias parecidas** e recuperar **experiência estruturalmente relevante**.

---

# 28. O que a topologia não significa

A topologia não deve ser interpretada como:

- uma prova automática de causalidade científica;
- um substituto para o Event Store;
- um grafo em que toda proximidade significa causa;
- uma representação que precisa duplicar todo o índice vetorial;
- um modelo estático para sempre.

Ela é uma projeção navegável, tipada e explicável do histórico e das relações conhecidas.

---

# 29. Estado atual e evolução planejada

Hoje a topologia já suporta:

- nós tipados como memória/estado;
- arestas causal, temporal, behavioral e semantic;
- provenance causal;
- confiança causal;
- múltiplos caminhos;
- path reconstruction;
- neighborhoods e hop distance;
- attractors explícitos;
- basin membership;
- shared-basin affinity;
- basin boundaries;
- neighboring basins.

As próximas fases ampliam esse terreno para incluir:

```text
transition frequency
reinforcement
erosion / decay
strongly connected components
sink discovery
recurrent-state discovery
empirical attractor discovery
basin stability
basin drift
persistent topology stores
```

Importante: reforço e erosão deverão modificar **influência de navegação**, não reescrever silenciosamente a evidência histórica original.

---

# 30. Limitação atual importante da projeção de eventos

No `EventProjector` atual, uma aresta criada por `causation_id` é materializada quando o evento causal referenciado já existe na topologia.

Ou seja, com ingestão fora de ordem:

```text
child arrives before parent
```

a relação causal ainda não é reconciliada automaticamente depois.

A issue de Event Sourcing/NDJSON do roadmap deve endurecer exatamente esse contrato, juntamente com replay idempotente, schema mapping e tratamento de duplicatas.

Portanto o modelo topológico já representa a relação corretamente, mas a camada de ingestão ainda está sendo fortalecida para reconstrução robusta de streams reais.

---

# 31. Resumo mental do modelo

Uma forma simples de pensar no CT-RAG é:

```text
MemoryNode
    = um lugar no terreno

Edge
    = uma relação conhecida entre lugares

CAUSAL edge
    = uma relação causal com provenance/evidência

CausalPath
    = uma explicação navegável entre dois estados

Attractor
    = um destino estrutural relevante

Basin
    = a região de estados que consegue convergir para esse destino

Boundary
    = onde uma região toca possibilidades externas

Neighboring Basin
    = outra região comportamental estruturalmente adjacente

Topology
    = o terreno inteiro navegável
```

E o CT-RAG usa esse terreno junto da recuperação semântica para responder não somente:

> **"qual memória se parece com esta pergunta?"**

mas também:

> **"qual memória pertence à trajetória, à causa e à região comportamental relevante para esta pergunta?"**
