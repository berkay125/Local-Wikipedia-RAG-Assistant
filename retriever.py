"""
Retrieval Module.

Handles query routing and context retrieval from the vector store.

Query Routing Strategy: Keyword-based
  - Check if query mentions known person names -> filter by type=person
  - Check if query mentions known place names -> filter by type=place
  - If both or neither found -> search across all entities
"""
import chromadb
import requests

from config import (
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    TOP_K,
    PEOPLE,
    PLACES,
)


def get_query_embedding(query: str) -> list[float]:
    """Generate an embedding for the user query."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": query},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def classify_query(query: str) -> dict:
    """
    Determine what type of entity the query is about.
    Returns dict with query_type, matched_people, matched_places.
    """
    query_lower = query.lower()
    matched_people = []
    matched_places = []

    for person in PEOPLE:
        name_parts = person.lower().split()
        if person.lower() in query_lower:
            matched_people.append(person)
        elif len(name_parts) > 1 and name_parts[-1] in query_lower:
            matched_people.append(person)

    for place in PLACES:
        place_clean = place.lower().replace("(statue)", "").strip()
        if place_clean in query_lower:
            matched_places.append(place)

    has_people = len(matched_people) > 0
    has_places = len(matched_places) > 0

    if has_people and has_places:
        query_type = "both"
    elif has_people:
        query_type = "person"
    elif has_places:
        query_type = "place"
    else:
        person_keywords = ["who", "born", "died", "famous for", "known for",
                          "discovered", "invented", "scientist", "artist", "player"]
        place_keywords = ["where", "located", "built", "visit", "tower", "wall",
                         "mountain", "canyon", "place", "monument", "landmark"]

        person_score = sum(1 for kw in person_keywords if kw in query_lower)
        place_score = sum(1 for kw in place_keywords if kw in query_lower)

        if person_score > place_score:
            query_type = "person"
        elif place_score > person_score:
            query_type = "place"
        else:
            query_type = "general"

    return {
        "query_type": query_type,
        "matched_people": matched_people,
        "matched_places": matched_places,
    }


def retrieve(query: str, top_k: int = TOP_K) -> dict:
    """
    Retrieve relevant chunks from the vector store based on the user query.
    Returns dict with chunks, query_info, and formatted context string.
    """
    query_info = classify_query(query)
    query_type = query_info["query_type"]

    query_embedding = get_query_embedding(query)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    where_filter = None
    if query_type == "person":
        where_filter = {"entity_type": "person"}
    elif query_type == "place":
        where_filter = {"entity_type": "place"}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    chunks = []
    if results and results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

    context_parts = []
    for chunk in chunks:
        entity = chunk["metadata"]["entity_name"]
        etype = chunk["metadata"]["entity_type"]
        context_parts.append(f"[Source: {entity} ({etype})]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    return {"chunks": chunks, "query_info": query_info, "context": context}


if __name__ == "__main__":
    test_queries = [
        "Who was Albert Einstein?",
        "Where is the Eiffel Tower?",
        "Compare Messi and Ronaldo",
    ]
    for q in test_queries:
        info = classify_query(q)
        print(f"Query: {q}")
        print(f"  Type: {info['query_type']}, People: {info['matched_people']}, Places: {info['matched_places']}")
