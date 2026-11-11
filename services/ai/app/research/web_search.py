
from ddgs import DDGS


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return source metadata and snippets."""

    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10.")

    results = DDGS().text(
        query,
        max_results=max_results,
    )

    sources = []

    for item in results or []:
        title = item.get("title", "")
        url = item.get("href", "")
        snippet = item.get("body", "")

        if not title or not url.startswith(("https://", "http://")):
            continue

        sources.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return sources
