"""Step 3 — Tools.  One thin wrapper around Tavily."""

from tavily import TavilyClient


def web_search(query: str) -> str:
    results = TavilyClient().search(query, max_results=5)["results"]
    return "\n".join(f"- {r['title']}: {r['content'][:200]}" for r in results)
