### ML service (through kubectl port-forward, so every number includes the forward)

Persistent HTTP connections; requests retried after a dropped forward: 0 (the retry is timed from its own start).

| scenario | requests | p50 ms | p95 ms | p99 ms | throughput |
|---|---|---|---|---|---|
| 1 row, sequential | 500 | 12.7 | 34.6 | 77.1 | 60 req/s |
| 50 rows per request | 20 | 14.9 | 28.7 | 28.7 | 3099 rows/s |
| 1 row, 4 clients | 300 | 64.3 | 121.9 | 187.4 | 57 req/s |

### LLM service (claude-haiku-5-5, top-k from rag/config.yaml, 10 questions)

| metric | value |
|---|---|
| latency p50 / p95 / max (s) | 2.43 / 4.20 / 4.20 |
| of which retrieval, mean (s) | 0.443 |
| of which answer call, mean (s) | 2.16 |
| tokens in / out per request, mean | 1772 / 396 |
| cost per request (USD), mean | 0.00038 |

### Pods (kubectl top)

```text
NAME                           CPU(cores)   MEMORY(bytes)   
alertmanager-66b998d5c-qtwhq   3m           23Mi            
grafana-68d57f687f-rdkjc       21m          86Mi            
llm-service-6b7d487fc6-w2dtv   4m           494Mi           
ml-service-5cbd99cccf-sx4qb    5m           190Mi           
mlflow-84f444d98c-s8njl        5m           365Mi           
postgres-5f8c7f7cf7-m27tv      18m          50Mi            
prometheus-845cf6c8c-rl75j     4m           53Mi            
pushgateway-84769bddbb-lxgqp   1m           14Mi            
responder-6f695d66f8-xh6v5     1m           15Mi
```
