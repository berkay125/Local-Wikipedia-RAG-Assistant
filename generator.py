"""
Answer Generation Module.

Uses Ollama's local LLM to generate answers based on retrieved context.
Supports streaming responses for real-time output.
"""
import json
import requests

from config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    SYSTEM_PROMPT,
    ANSWER_PROMPT_TEMPLATE,
)
from retriever import retrieve


def generate_answer(query: str, context: str, stream: bool = False):
    """
    Generate an answer using Ollama LLM.

    Args:
        query: The user's question
        context: Retrieved context from vector store
        stream: If True, yields response tokens one by one

    Returns/Yields:
        If stream=False: returns the full answer string
        If stream=True: yields answer tokens as they are generated
    """
    prompt = ANSWER_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context if context else "No relevant context found.",
        question=query,
    )

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }

    if stream:
        return _stream_response(payload)
    else:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")


def _stream_response(payload: dict):
    """Stream response tokens from Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break


def ask(query: str, stream: bool = False):
    """
    Full RAG pipeline: retrieve context then generate answer.

    Args:
        query: The user's question
        stream: Whether to stream the response

    Returns:
        dict with answer, query_info, chunks, and context
    """
    # Step 1: Retrieve
    retrieval_result = retrieve(query)

    context = retrieval_result["context"]
    query_info = retrieval_result["query_info"]
    chunks = retrieval_result["chunks"]

    # Step 2: Generate
    if stream:
        return {
            "stream": generate_answer(query, context, stream=True),
            "query_info": query_info,
            "chunks": chunks,
            "context": context,
        }
    else:
        answer = generate_answer(query, context, stream=False)
        return {
            "answer": answer,
            "query_info": query_info,
            "chunks": chunks,
            "context": context,
        }


if __name__ == "__main__":
    print("Testing generation pipeline...\n")
    result = ask("Who was Albert Einstein and what is he known for?")
    print(f"Query type: {result['query_info']['query_type']}")
    print(f"Chunks used: {len(result['chunks'])}")
    print(f"\nAnswer:\n{result['answer']}")
