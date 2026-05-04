# Product Requirements Document (PRD)
# Local Wikipedia RAG Assistant

## 1. Overview

Build a ChatGPT-style question-answering system that runs entirely on localhost. The system uses Retrieval-Augmented Generation (RAG) to answer questions about famous people and places using data from Wikipedia.

## 2. Problem Statement

Users need a local, privacy-preserving AI assistant that can answer factual questions about well-known people and places without relying on external cloud APIs. The system should ground its answers in actual Wikipedia data to minimize hallucination.

## 3. Goals

- Ingest Wikipedia articles for at least 20 people and 20 places
- Process and store data locally using vector embeddings
- Retrieve relevant information based on user queries
- Generate accurate, context-grounded answers using a local LLM
- Provide a user-friendly chat interface

## 4. Non-Goals

- Real-time Wikipedia updates
- Multi-language support
- User authentication
- Cloud deployment

## 5. Technical Architecture

### 5.1 Data Pipeline

```
Wikipedia API → Raw Text → Chunking → Embedding → ChromaDB
```

1. **Ingestion**: Fetch Wikipedia articles via MediaWiki API (plain text extraction)
2. **Chunking**: Split into ~500 character chunks with 100 char overlap, sentence-aware
3. **Embedding**: Generate vectors using `nomic-embed-text` via Ollama
4. **Storage**: Store in ChromaDB with metadata (entity_name, entity_type)

### 5.2 Query Pipeline

```
User Query → Query Router → Vector Search → Context Assembly → LLM → Answer
```

1. **Query Classification**: Keyword-based routing (person/place/both/general)
2. **Retrieval**: Top-5 similar chunks from ChromaDB with metadata filtering
3. **Generation**: Ollama `llama3.2:3b` with grounded prompt template
4. **Display**: Streamlit chat UI with streaming responses

### 5.3 Vector Store Design

**Option B**: Single ChromaDB collection with metadata filtering.

- Each chunk has `entity_type` metadata ("person" or "place")
- Query router determines filter based on query classification
- Mixed queries search without filter

### 5.4 Technology Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.10+ |
| LLM Runtime | Ollama |
| LLM Model | llama3.2:3b |
| Embedding Model | nomic-embed-text |
| Vector Database | ChromaDB |
| Backend Storage | SQLite (via ChromaDB) |
| UI Framework | Streamlit |
| Data Source | Wikipedia MediaWiki API |

## 6. Data Requirements

### People (20)
Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Mahatma Gandhi, Nelson Mandela, Cleopatra, Isaac Newton, Wolfgang Amadeus Mozart, Ludwig van Beethoven, Pablo Picasso, Napoleon, Martin Luther King Jr., Steve Jobs

### Places (20)
Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Petra, Stonehenge, Angkor Wat, Christ the Redeemer, Great Barrier Reef, Niagara Falls, Amazon Rainforest, Sahara, Venice, Santorini

## 7. User Interface Requirements

- Chat-style message interface
- Streaming response display
- Source chunk viewer (expandable)
- Example questions for quick access
- Chat history within session
- Clear/reset functionality

## 8. Success Criteria

- System runs fully on localhost with no external API calls
- Retrieves relevant context for queries about configured entities
- Generates accurate, grounded answers
- Returns "I don't know" for out-of-scope questions
- Responds within reasonable time (< 30s per query)
