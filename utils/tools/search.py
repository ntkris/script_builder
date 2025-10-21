"""Search utilities for web research"""

from typing import List, Optional
import os
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Result from a search query"""
    query: str
    title: str
    url: str
    text: str = ""
    highlights: List[str] = []
    score: float = 0.0
    published_date: Optional[str] = None
    author: Optional[str] = None


def search_exa(
    query: str,
    num_results: int = 5,
    max_characters: int = 2000,
    start_published_date: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[SearchResult]:
    """
    Search using Exa API and return normalized results.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
        max_characters: Maximum characters for text content (default: 2000)
        start_published_date: Optional start date for filtering results (ISO format: YYYY-MM-DD)
        api_key: Optional API key (defaults to EXA_API_KEY env var)

    Returns:
        List of SearchResult objects

    Example:
        from utils import search_exa

        results = search_exa("machine learning trends", num_results=5, start_published_date="2024-01-01")
        for result in results:
            print(f"{result.title}: {result.url}")
            print(f"Text: {result.text[:200]}...")
    """
    try:
        from exa_py import Exa
    except ImportError:
        raise ImportError("Exa package not installed. Install with: pip install exa-py")

    # Get API key
    if api_key is None:
        api_key = os.getenv("EXA_API_KEY")

    if not api_key:
        raise ValueError("EXA_API_KEY not found in environment variables or provided as argument")

    # Initialize Exa client
    exa = Exa(api_key=api_key)

    try:
        # Build search parameters
        search_params = {
            "query": query,
            "num_results": num_results,
            "text": {"max_characters": max_characters},
            "highlights": True
        }

        # Add date filter if provided
        if start_published_date:
            search_params["start_published_date"] = start_published_date

        # Search with text content and highlights
        response = exa.search_and_contents(**search_params)

        # Convert to normalized SearchResult objects
        search_results = []
        for result in response.results:
            # Handle score - may be None
            score = 0.0
            if hasattr(result, 'score') and result.score is not None:
                score = float(result.score)

            search_results.append(SearchResult(
                query=query,
                title=result.title,
                url=result.url,
                text=result.text if hasattr(result, 'text') and result.text else "",
                highlights=result.highlights if hasattr(result, 'highlights') and result.highlights else [],
                score=score,
                published_date=result.published_date if hasattr(result, 'published_date') else None,
                author=result.author if hasattr(result, 'author') else None
            ))

        return search_results

    except Exception as e:
        print(f"⚠️  Exa search failed for query '{query}': {e}")
        return []
