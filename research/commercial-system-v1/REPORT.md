# Experimento comercial de localização de causa v1

Status: evidência sintética exploratória. Gerador: `ctrag-commercial-system-v1`.

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
- configuração: seed `20260914`, dimensão `256`, máximo `8` hops.

## Resultados

| Tarefa | Método | N | Top-1 | Recall@3 | MRR |
|---|---|---:|---:|---:|---:|
| diagnosis | ctrag_causal_frontier | 6 | 1.000 | 1.000 | 1.000 |
| diagnosis | dense_lexical | 6 | 0.000 | 0.000 | 0.043 |
| diagnosis | dense_only | 6 | 0.000 | 0.000 | 0.035 |
| diagnosis | full_ctrag | 6 | 0.000 | 0.000 | 0.228 |
| diagnosis | lexical_only | 6 | 0.000 | 0.000 | 0.054 |
| recovery | dense_lexical | 6 | 0.000 | 0.000 | 0.058 |
| recovery | full_ctrag | 6 | 1.000 | 1.000 | 1.000 |

Os resultados completos por cenário, ranking e componentes de score estão em `results.json`. O arquivo `explorer.html` permite selecionar incidentes, inspecionar nós, executar as duas consultas predefinidas e revelar o gabarito somente depois do diagnóstico.

Há também um resultado negativo importante: o `full_ctrag` genérico não colocou a causa-raiz no Top-3 de nenhum dos seis cenários (MRR 0,228). Ele favorece ancestrais causalmente próximos, que respondem bem a “o que causou imediatamente?”, mas não necessariamente à pergunta operacional “onde começou a anomalia?”. A operação `ctrag_causal_frontier` explicita essa segunda semântica e foi desenhada e avaliada neste mesmo ensaio exploratório; sua generalização precisa de casos novos e congelados.

## Interpretação permitida

Este experimento testa **localização de causa-raiz dentro de telemetria causal instrumentada**. Ele mostra se o grafo preserva e torna navegável a cadeia que liga uma primeira anomalia ao sintoma e à recuperação.

Ele não demonstra descoberta causal a partir de logs correlacionais, não prova validade externa em empresas reais e não autoriza chamar toda precedência temporal de causa. As arestas causais deste ensaio vêm exclusivamente de `causation_id`; arestas temporais e comportamentais permanecem distintas.

## Critério inicial de sucesso

O marco é atingido quando `ctrag_causal_frontier` encontra a causa correta em Top-1 nos seis incidentes e o modo `full_ctrag` encontra a solução observada em Recall@3, sem vazamento do oracle. Esses resultados devem ser tratados como prova de execução do mecanismo, não como conclusão científica final.
