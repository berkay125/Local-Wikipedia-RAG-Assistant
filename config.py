"""
Configuration constants for the Local Wikipedia RAG Assistant.
"""
import os

# --- Ollama Settings ---
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Chunking Settings ---
CHUNK_SIZE = 1500         # characters per chunk
CHUNK_OVERLAP = 200       # overlap between consecutive chunks

# --- Retrieval Settings ---
TOP_K = 5                 # number of chunks to retrieve

# --- ChromaDB Settings ---
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
COLLECTION_NAME = "wikipedia_rag"

# --- Data Settings ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "wikipedia_raw.json")

# --- Entity Lists ---
PEOPLE = [
    # Required (10)
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Additional (10)
    "Mahatma Gandhi",
    "Nelson Mandela",
    "Cleopatra",
    "Isaac Newton",
    "Wolfgang Amadeus Mozart",
    "Ludwig van Beethoven",
    "Pablo Picasso",
    "Napoleon",
    "Martin Luther King Jr.",
    "Steve Jobs",
]

PLACES = [
    # Required (10)
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",   # Wikipedia title: "Giza pyramid complex"
    "Mount Everest",
    # Additional (10)
    "Petra",
    "Stonehenge",
    "Angkor Wat",
    "Christ the Redeemer (statue)",
    "Great Barrier Reef",
    "Niagara Falls",
    "Amazon rainforest",
    "Sahara",
    "Venice",
    "Santorini",
]

# Wikipedia title overrides for entities whose page title differs from the common name
WIKIPEDIA_TITLE_MAP = {
    "Pyramids of Giza": "Giza pyramid complex",
    "Martin Luther King Jr.": "Martin Luther King Jr.",
    "Christ the Redeemer (statue)": "Christ the Redeemer (statue)",
}

# --- Prompt Template ---
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.
Rules:
1. Use ONLY the information from the context below to answer.
2. If the answer is not in the context, say "I don't know based on the available information."
3. Be concise but informative.
4. If comparing two entities, use information from the context about both.
5. Cite which entity the information comes from when relevant."""

ANSWER_PROMPT_TEMPLATE = """{system_prompt}

Context:
{context}

Question: {question}

Answer:"""
