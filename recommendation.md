# Production Deployment Recommendations

## Current State

The system currently runs as a single-machine prototype with all components (LLM, vector DB, UI) on localhost. This document outlines recommendations for scaling this system to production.

## 1. Containerization

### Docker Compose Setup
- **App Container**: Streamlit application + Python backend
- **Ollama Container**: Dedicated GPU-enabled container for LLM inference
- **ChromaDB Container**: Separate persistent vector store service

```yaml
# Recommended docker-compose structure
services:
  app:
    build: .
    ports: ["8501:8501"]
    depends_on: [ollama, chromadb]
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
  chromadb:
    image: chromadb/chroma:latest
    volumes: ["chroma_data:/chroma/chroma"]
```

## 2. Model Serving

### Current → Production
| Aspect | Current | Recommended |
|--------|---------|-------------|
| LLM | llama3.2:3b (CPU) | llama3.1:8b or Mistral 7B (GPU) |
| Embedding | nomic-embed-text | Same (efficient enough) |
| Serving | Ollama single instance | vLLM or TGI with load balancing |
| Concurrency | Single user | Multi-user with request queuing |

### GPU Requirements
- Minimum: NVIDIA GPU with 8GB VRAM (for 7B models)
- Recommended: NVIDIA A10G or L4 (24GB VRAM) for production throughput

## 3. Vector Database Scaling

### Current → Production
| Aspect | Current | Recommended |
|--------|---------|-------------|
| Database | ChromaDB (local SQLite) | Weaviate, Pinecone, or Qdrant |
| Storage | Local filesystem | Cloud-managed or self-hosted cluster |
| Indexing | HNSW (default) | HNSW with tuned parameters |
| Replication | None | Multi-replica for availability |

### Data Pipeline
- Implement incremental ingestion (avoid full re-index)
- Add data versioning for Wikipedia article updates
- Schedule periodic re-ingestion (e.g., monthly)

## 4. API Layer

Replace Streamlit with a proper API backend:
- **FastAPI** for REST endpoints
- **WebSocket** support for streaming responses
- **Rate limiting** per user/API key
- **Authentication** via JWT or API keys

```
Client → API Gateway (nginx) → FastAPI → Ollama/vLLM
                                       → ChromaDB/Weaviate
```

## 5. Monitoring & Observability

- **Metrics**: Response latency, retrieval accuracy, token usage (Prometheus + Grafana)
- **Logging**: Structured logging with ELK stack
- **Tracing**: Request tracing with OpenTelemetry
- **Alerts**: Latency spikes, model errors, disk usage

## 6. Security

- Input sanitization (prompt injection prevention)
- Rate limiting to prevent abuse
- HTTPS for all communications
- No PII storage in vector database
- Model output filtering

## 7. Caching

- **Embedding cache**: Cache query embeddings for repeated queries
- **Response cache**: Cache full responses for identical queries (Redis)
- **Semantic cache**: Cache similar queries using embedding similarity threshold

## 8. Cost Optimization

- Use quantized models (GGUF Q4_K_M) for cost-effective inference
- Implement request batching for embedding generation
- Auto-scaling based on traffic patterns
- Spot/preemptible instances for non-critical workloads

## 9. Testing Strategy

- **Unit tests**: Chunking, query classification, prompt formatting
- **Integration tests**: Full RAG pipeline with known Q&A pairs
- **Load tests**: Concurrent user simulation
- **Evaluation**: RAGAS metrics (faithfulness, relevancy, context recall)

## 10. Deployment Architecture (Cloud)

```
                    ┌─────────────┐
                    │   CDN/LB    │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────┴─────┐ ┌───┴───┐ ┌─────┴─────┐
        │  FastAPI   │ │FastAPI│ │  FastAPI   │
        │ Instance 1 │ │Inst 2 │ │ Instance 3 │
        └─────┬─────┘ └───┬───┘ └─────┬─────┘
              │            │            │
        ┌─────┴────────────┴────────────┴─────┐
        │        Message Queue (Redis)         │
        └─────────────────┬───────────────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
        ┌─────┴─────┐ ┌──┴──┐ ┌─────┴─────┐
        │  vLLM GPU │ │vLLM │ │  vLLM GPU │
        │  Worker 1 │ │  2  │ │  Worker 3 │
        └───────────┘ └─────┘ └───────────┘
```
