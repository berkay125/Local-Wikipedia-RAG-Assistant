# Youtube Demo Link
https://youtu.be/T8XsXvRmkuU

# 📚 Local Wikipedia RAG Assistant

A ChatGPT-style question-answering system that runs **entirely on your local machine**. It uses Retrieval-Augmented Generation (RAG) to answer questions about famous people and places using Wikipedia data, a local LLM (via Ollama), and a local vector database (ChromaDB).

## 🏗️ Architecture

```
Wikipedia → Ingest → Chunk → Embed → ChromaDB
                                         ↑
User Query → Route → Retrieve → Context → LLM → Answer
```

**Key Design Decisions:**
- **Vector Store**: Single ChromaDB collection with metadata filtering (Option B)
- **Chunking**: Sentence-aware, 500-char chunks with 100-char overlap
- **Query Routing**: Keyword-based classification (person/place/both/general)
- **No external APIs**: Everything runs on localhost

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** installed and running

## 🚀 Setup Instructions

### 1. Install Ollama

Download and install from: https://ollama.com/download

### 2. Pull Required Models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### 3. Start Ollama (if not auto-started)

```bash
ollama serve
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Ingest Wikipedia Data

```bash
python ingest.py
```

This fetches articles for 20 people and 20 places from Wikipedia.

### 6. Create Vector Store

```bash
python embedder.py
```

This generates embeddings and stores them in ChromaDB. May take a few minutes.

### 7. Start the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## 💬 Example Queries

### People
- "Who was Albert Einstein and what is he known for?"
- "What did Marie Curie discover?"
- "Why is Nikola Tesla famous?"
- "Compare Lionel Messi and Cristiano Ronaldo"
- "What is Frida Kahlo known for?"

### Places
- "Where is the Eiffel Tower located?"
- "Why is the Great Wall of China important?"
- "What is Machu Picchu?"
- "What was the Colosseum used for?"
- "Where is Mount Everest?"

### Mixed
- "Which famous place is located in Turkey?"
- "Which person is associated with electricity?"
- "Compare the Eiffel Tower and the Statue of Liberty"

### Failure Cases (should return "I don't know")
- "Who is the president of Mars?"
- "Tell me about John Doe"

## 📁 Project Structure

```
├── README.md              # This file
├── product_prd.md         # Product Requirements Document
├── recommendation.md      # Production deployment recommendations
├── requirements.txt       # Python dependencies
├── config.py              # Configuration constants
├── ingest.py              # Wikipedia data ingestion
├── chunker.py             # Text chunking logic
├── embedder.py            # Embedding generation + ChromaDB storage
├── retriever.py           # Query routing + context retrieval
├── generator.py           # LLM answer generation
├── app.py                 # Streamlit chat interface
├── data/                  # Ingested Wikipedia data (JSON)
└── chroma_db/             # ChromaDB vector database files
```

## 🔧 Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LLM_MODEL` | `llama3.2:3b` | Ollama LLM model |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `TOP_K` | 5 | Number of chunks to retrieve |

## ⚠️ Technical Constraints

- **No external LLM APIs** (OpenAI, Claude, etc.)
- All processing runs on localhost
- Models served via Ollama
- Vector storage via ChromaDB (SQLite backend)
