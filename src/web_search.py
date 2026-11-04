from typing import List, Dict, Any, Optional
from loguru import logger
from ddgs import DDGS

class WebSearchFallback:
    """
    Automated Web Search Fallback for Corrective RAG (CRAG).
    Triggers when document retrieval confidence is low or ambiguous.
    """

    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, Any]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        logger.info(f"CRAG Web Fallback executing query: '{clean_query[:60]}'...")
        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(clean_query, max_results=self.max_results))
                formatted = []
                for idx, r in enumerate(raw_results, start=1):
                    formatted.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                        "source_id": f"Web_{idx}"
                    })
                logger.info(f"CRAG Web Fallback retrieved {len(formatted)} web snippets.")
                return formatted
        except Exception as e:
            logger.warning(f"DuckDuckGo search encounter error: {e}. Returning empty web evidence.")
            return []
