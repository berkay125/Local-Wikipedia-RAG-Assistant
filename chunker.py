"""
Text Chunking Module.

Splits large documents into smaller, overlapping chunks
that are suitable for embedding and retrieval.

Strategy: Sentence-aware fixed-size chunking with overlap.
- Target chunk size: ~500 characters
- Overlap: ~100 characters
- Splits on sentence boundaries to preserve meaning
"""
import json
import re

from config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DATA_FILE


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations and edge cases.
    """
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks, respecting sentence boundaries.

    Args:
        text: The full text to chunk
        chunk_size: Target size for each chunk (in characters)
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, split it by words
        if sentence_len > chunk_size:
            # Flush current chunk first
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_length = 0

            # Split long sentence into word-based chunks
            words = sentence.split()
            word_chunk = []
            word_chunk_len = 0
            for word in words:
                if word_chunk_len + len(word) + 1 > chunk_size and word_chunk:
                    chunks.append(" ".join(word_chunk))
                    # Keep overlap words
                    overlap_text = " ".join(word_chunk)
                    overlap_words = []
                    overlap_len = 0
                    for w in reversed(word_chunk):
                        if overlap_len + len(w) + 1 > overlap:
                            break
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    word_chunk = overlap_words
                    word_chunk_len = overlap_len
                word_chunk.append(word)
                word_chunk_len += len(word) + 1

            if word_chunk:
                current_chunk_sentences = [" ".join(word_chunk)]
                current_length = word_chunk_len
            continue

        # Check if adding this sentence would exceed chunk_size
        if current_length + sentence_len + 1 > chunk_size and current_chunk_sentences:
            # Save current chunk
            chunks.append(" ".join(current_chunk_sentences))

            # Create overlap: keep last sentences that fit within overlap size
            overlap_sentences = []
            overlap_length = 0
            for s in reversed(current_chunk_sentences):
                if overlap_length + len(s) + 1 > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_length += len(s) + 1

            current_chunk_sentences = overlap_sentences
            current_length = overlap_length

        current_chunk_sentences.append(sentence)
        current_length += sentence_len + 1

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks


def chunk_all_documents(raw_data: list[dict] = None) -> list[dict]:
    """
    Chunk all ingested documents.

    Returns a list of dicts with:
        - text: the chunk text
        - entity_name: name of the entity
        - entity_type: 'person' or 'place'
        - chunk_index: index of this chunk within the entity
        - source_url: Wikipedia URL
    """
    if raw_data is None:
        with open(RAW_DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

    all_chunks = []

    for entity in raw_data:
        name = entity["name"]
        entity_type = entity["type"]
        text = entity["text"]
        url = entity.get("url", "")

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "entity_name": name,
                "entity_type": entity_type,
                "chunk_index": i,
                "source_url": url,
            })

    print(f"Chunked {len(raw_data)} entities into {len(all_chunks)} chunks")
    return all_chunks


if __name__ == "__main__":
    chunks = chunk_all_documents()
    # Print some stats
    if chunks:
        lengths = [len(c["text"]) for c in chunks]
        print(f"  Min chunk length: {min(lengths)}")
        print(f"  Max chunk length: {max(lengths)}")
        print(f"  Avg chunk length: {sum(lengths) / len(lengths):.0f}")
        print(f"\nSample chunk (first):")
        print(f"  Entity: {chunks[0]['entity_name']} ({chunks[0]['entity_type']})")
        print(f"  Text: {chunks[0]['text'][:200]}...")
