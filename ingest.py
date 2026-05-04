"""
Wikipedia Data Ingestion Module.

Fetches Wikipedia articles for configured people and places,
cleans the text, and stores it as a local JSON file.
Uses requests + BeautifulSoup (no external Wikipedia API wrapper).
"""
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

from config import PEOPLE, PLACES, DATA_DIR, RAW_DATA_FILE, WIKIPEDIA_TITLE_MAP


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

# User-Agent required by Wikipedia API policy
HEADERS = {
    "User-Agent": "LocalWikipediaRAG/1.0 (educational project; Python/requests)"
}


def get_wikipedia_text(title: str, max_retries: int = 3) -> dict | None:
    """
    Fetch the plain text content of a Wikipedia page using the MediaWiki API.
    Returns a dict with title, text, and url, or None if not found.
    Includes retry logic for rate limiting (429 errors).
    """
    # Use the title map if an override exists
    api_title = WIKIPEDIA_TITLE_MAP.get(title, title)

    params = {
        "action": "query",
        "titles": api_title,
        "prop": "extracts",
        "explaintext": True,       # plain text, no HTML
        "format": "json",
        "redirects": 1,            # follow redirects
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(WIKIPEDIA_API_URL, params=params, headers=HEADERS, timeout=30)

            # Handle rate limiting with exponential backoff
            if response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"    [RATE LIMITED] Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id == "-1":
                    print(f"  [WARNING] Page not found: {title}")
                    return None

                text = page_data.get("extract", "")
                if not text:
                    print(f"  [WARNING] No text extracted for: {title}")
                    return None

                # Clean up text
                text = clean_text(text)

                return {
                    "name": title,
                    "wikipedia_title": page_data.get("title", api_title),
                    "text": text,
                    "url": f"https://en.wikipedia.org/wiki/{api_title.replace(' ', '_')}",
                }

        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"    [RETRY] Error: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [ERROR] Failed to fetch {title} after {max_retries} attempts: {e}")
                return None
    return None


def clean_text(text: str) -> str:
    """Clean Wikipedia extracted text by removing non-content sections."""
    # First, remove entire sections that are not useful for Q&A
    # These sections appear as == Section Name == in Wikipedia extracts
    # We need to remove everything from these headers to the next header or end
    sections_to_remove = [
        "References", "See also", "Bibliography", "External links",
        "Further reading", "Notes", "Sources", "Citations",
        "Cited sources", "Works cited", "Selected bibliography",
        "Selected works", "Footnotes", "General and cited references",
        "General references", "Works", "Publications", "Discography",
        "Filmography", "Videography", "Awards and nominations",
    ]

    for section in sections_to_remove:
        # Match section header and everything until next section or end
        # Handles == Section ==, === Section ===, etc.
        pattern = rf"={2,}\s*{re.escape(section)}\s*={2,}.*?(?=\n={2,}[^=]|\Z)"
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove remaining == Section headers == but keep the text after them
    text = re.sub(r"={2,}\s*", "", text)
    # Remove reference markers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Remove excess whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
    return text.strip()


def ingest_all():
    """Fetch all configured entities from Wikipedia and save to JSON."""
    os.makedirs(DATA_DIR, exist_ok=True)

    all_data = []
    total = len(PEOPLE) + len(PLACES)

    print(f"Starting ingestion of {total} entities...\n")

    # Ingest people
    print("--- Ingesting People ---")
    for i, person in enumerate(PEOPLE, 1):
        print(f"  [{i}/{len(PEOPLE)}] Fetching: {person}")
        result = get_wikipedia_text(person)
        if result:
            result["type"] = "person"
            all_data.append(result)
            print(f"    [OK] Got {len(result['text'])} characters")
        else:
            print(f"    [FAIL] Failed")
        time.sleep(2)  # Be polite to Wikipedia

    # Ingest places
    print("\n--- Ingesting Places ---")
    for i, place in enumerate(PLACES, 1):
        print(f"  [{i}/{len(PLACES)}] Fetching: {place}")
        result = get_wikipedia_text(place)
        if result:
            result["type"] = "place"
            all_data.append(result)
            print(f"    [OK] Got {len(result['text'])} characters")
        else:
            print(f"    [FAIL] Failed")
        time.sleep(2)

    # Save to JSON
    with open(RAW_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Ingestion complete!")
    print(f"  Total entities fetched: {len(all_data)}/{total}")
    print(f"  Saved to: {RAW_DATA_FILE}")
    print(f"{'='*50}")

    return all_data


if __name__ == "__main__":
    ingest_all()
