"""
Streamlit Chat Interface for the Local Wikipedia RAG Assistant.

Features:
  - Chat-style UI with message history
  - Streaming responses
  - Source chunk display (expandable)
  - Query classification info
  - Response timing
  - System reset capability
"""
import time
import streamlit as st
from generator import ask
from retriever import classify_query

# --- Page Config ---
st.set_page_config(
    page_title="Wikipedia RAG Assistant",
    page_icon="📚",
    layout="wide",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-chip {
        display: inline-block;
        background-color: #262730;
        border: 1px solid #4a4a5a;
        border-radius: 16px;
        padding: 2px 10px;
        margin: 2px 4px;
        font-size: 0.8em;
        color: #fafafa;
    }
    .timing-info {
        color: #888;
        font-size: 0.8em;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    st.markdown("**Model:** `llama3.2:3b`")
    st.markdown("**Embeddings:** `nomic-embed-text`")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Retrieval:** Top-5 chunks")
    st.markdown("---")

    show_sources = st.checkbox("Show source chunks", value=True)
    show_query_info = st.checkbox("Show query analysis", value=False)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    examples = [
        "Who was Albert Einstein?",
        "What did Marie Curie discover?",
        "Where is the Eiffel Tower?",
        "Compare Messi and Ronaldo",
        "What is Machu Picchu?",
        "Which famous place is in Turkey?",
        "Who is the president of Mars?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_query = ex
            st.rerun()

# --- Main UI ---
st.title("📚 Wikipedia RAG Assistant")
st.caption("A local ChatGPT-style system powered by Ollama + ChromaDB")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and show_sources and "chunks" in message:
            with st.expander(f"📄 Sources ({len(message['chunks'])} chunks)"):
                for i, chunk in enumerate(message["chunks"]):
                    meta = chunk["metadata"]
                    st.markdown(f"**{i+1}. {meta['entity_name']}** ({meta['entity_type']})")
                    st.caption(f"Distance: {chunk['distance']:.4f}")
                    st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                    st.markdown("---")

        if message["role"] == "assistant" and "timing" in message:
            st.caption(f"⏱️ {message['timing']}")

# Handle pending query from sidebar examples
if "pending_query" in st.session_state:
    query = st.session_state.pending_query
    del st.session_state.pending_query
else:
    query = st.chat_input("Ask a question about famous people or places...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        start_time = time.time()

        # Show query analysis if enabled
        if show_query_info:
            q_info = classify_query(query)
            st.caption(
                f"🔍 Query type: **{q_info['query_type']}** | "
                f"People: {q_info['matched_people'] or 'none'} | "
                f"Places: {q_info['matched_places'] or 'none'}"
            )

        try:
            result = ask(query, stream=True)
            chunks = result["chunks"]
            query_info = result["query_info"]

            # Stream the response
            response_text = st.write_stream(result["stream"])

            elapsed = time.time() - start_time
            timing = f"Response generated in {elapsed:.1f}s"
            st.caption(f"⏱️ {timing}")

            # Show sources
            if show_sources and chunks:
                with st.expander(f"📄 Sources ({len(chunks)} chunks)"):
                    for i, chunk in enumerate(chunks):
                        meta = chunk["metadata"]
                        st.markdown(f"**{i+1}. {meta['entity_name']}** ({meta['entity_type']})")
                        st.caption(f"Distance: {chunk['distance']:.4f}")
                        st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
                        st.markdown("---")

            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "chunks": chunks,
                "timing": timing,
            })

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.error(error_msg)
            if "Connection" in str(e) or "refused" in str(e):
                st.warning("Make sure Ollama is running: `ollama serve`")
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
