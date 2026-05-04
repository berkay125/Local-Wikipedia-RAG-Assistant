"""
Embedding and Vector Storage Module.

Generates embeddings using Ollama's nomic-embed-text model
and stores them in ChromaDB with metadata.

Design Choice: Option B - Single vector store with metadata filtering.
Rationale:
  - Simpler to manage than two separate stores
  - Metadata filters (type=person / type=place) enable targeted retrieval
  - Mixed queries can search across all entries without merging results
  - Less resource usage and code complexity
"""
import json
import os
import sys
import time

import chromadb
import requests

from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    RAW_DATA_FILE,
)
from chunker import chunk_all_documents


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text using Ollama's embedding API.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def get_embeddings_batch(texts: list[str], batch_size: int = 50) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using Ollama's batch /api/embed endpoint.
    Sends multiple texts per request for much faster processing.
    """
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_end = min(i + batch_size, total)
        print(f"\r  Embedding batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
              f"(chunks {i+1}-{batch_end}/{total})...", end="", flush=True)

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
            timeout=300,
        )
        response.raise_for_status()
        batch_embeddings = response.json()["embeddings"]
        all_embeddings.extend(batch_embeddings)

    print()  # newline after progress
    return all_embeddings


def create_vector_store(chunks: list[dict] = None):
    """
    Create/populate the ChromaDB vector store with embedded chunks.

    Each document is stored with:
      - id: unique identifier
      - embedding: vector from nomic-embed-text
      - document: the chunk text
      - metadata: entity_name, entity_type, chunk_index, source_url
    """
    if chunks is None:
        chunks = chunk_all_documents()

    if not chunks:
        print("No chunks to embed. Run ingest.py first.")
        return

    # Initialize ChromaDB
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection if it exists (for re-ingestion)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  This may take a few minutes...\n")

    # Prepare data
    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "entity_name": c["entity_name"],
            "entity_type": c["entity_type"],
            "chunk_index": c["chunk_index"],
            "source_url": c["source_url"],
        }
        for c in chunks
    ]

    # Generate embeddings
    start_time = time.time()
    embeddings = get_embeddings_batch(texts)
    embed_time = time.time() - start_time

    # Add to ChromaDB
    print(f"  Adding {len(chunks)} chunks to ChromaDB...")

    # ChromaDB has a limit on batch size, so we add in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"\n{'='*50}")
    print(f"Vector store created successfully!")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Total chunks: {collection.count()}")
    print(f"  Embedding time: {embed_time:.1f}s")
    print(f"  Storage: {CHROMA_DB_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    create_vector_store()
