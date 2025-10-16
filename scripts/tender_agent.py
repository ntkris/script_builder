#!/usr/bin/env python3
"""Simplified tender discovery pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from dotenv import load_dotenv

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (  # noqa: E402
    AIRequest,
    GeminiModel,
    Provider,
    TokenTracker,
    call_gemini,
    search_exa,
)

DEFAULT_MODEL = GeminiModel.GEMINI_2_5_FLASH.value


@dataclass
class TenderCandidate:
    """A potential tender discovered from search results."""

    query: str
    title: str
    url: str
    text: str
    published_date: str | None


@dataclass
class TenderRecord:
    """Structured tender information for CSV export."""

    Title: str
    Description: str
    Location: str
    Weblink: str
    Posted_By: str
    Value_GBP: str
    Published_Date: str
    Deadline: str


def parse_json_block(raw_text: str):
    """Best-effort JSON parsing that tolerates extra prose from the model."""

    raw_text = raw_text.strip()
    if not raw_text:
        return None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Attempt to find the first JSON object or array in the text.
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = raw_text.find(start_char)
        end = raw_text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            snippet = raw_text[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                continue
    return None


def generate_queries(
    user_prompt: str,
    model: str,
    tracker: TokenTracker,
    max_queries: int,
) -> List[str]:
    """Use an LLM to turn the prompt into focused search queries."""

    system_message = (
        "You help a construction company discover public tenders. "
        f"Return a JSON array of between 5 and {max_queries} short web search queries. "
        "Combine the services, tender terminology, and geography implied by the user request."
    )

    request = AIRequest(
        provider=Provider.GOOGLE,
        model=model,
        max_tokens=512,
        step_name="Generate queries",
        system=system_message,
        messages=[
            {
                "role": "user",
                "content": (
                    "User request:\n" + user_prompt + "\n\n" "Respond with a JSON array of query strings."
                ),
            }
        ],
    )
    response = call_gemini(request, tracker)
    data = parse_json_block(response.content)
    if not isinstance(data, list):
        raise ValueError("Query generation failed: model response was not a JSON array")

    queries = []
    for item in data:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                queries.append(stripped)
        elif isinstance(item, dict):
            # Allow objects with a `query` key for resilience.
            value = item.get("query") if isinstance(item, dict) else None
            if isinstance(value, str) and value.strip():
                queries.append(value.strip())

        if len(queries) >= max_queries:
            break

    if not queries:
        raise ValueError("No queries produced from model response")
    return queries


def gather_candidates(
    queries: Sequence[str],
    results_per_query: int,
    exa_api_key: str | None,
) -> List[TenderCandidate]:
    """Search Exa for each query and collect potential tenders."""

    candidates: List[TenderCandidate] = []
    for query in queries:
        results = search_exa(
            query,
            num_results=results_per_query,
            max_characters=2000,
            api_key=exa_api_key,
        )
        for result in results:
            text = result.text or " ".join(result.highlights or [])
            candidates.append(
                TenderCandidate(
                    query=query,
                    title=result.title,
                    url=result.url,
                    text=text,
                    published_date=result.published_date,
                )
            )
    return candidates


def filter_candidates(
    user_prompt: str,
    candidates: Sequence[TenderCandidate],
    model: str,
    tracker: TokenTracker,
) -> List[TenderCandidate]:
    """Use an LLM to retain only highly relevant tender notices."""

    if not candidates:
        return []

    payload = {
        "user_prompt": user_prompt,
        "candidates": [
            {
                "index": idx,
                "query": candidate.query,
                "title": candidate.title,
                "url": candidate.url,
                "excerpt": candidate.text[:600],
            }
            for idx, candidate in enumerate(candidates)
        ],
    }

    request = AIRequest(
        provider=Provider.GOOGLE,
        model=model,
        max_tokens=512,
        step_name="Filter results",
        system=(
            "You evaluate search results and keep only those that are real tender or procurement "
            "notices matching the construction company request. Return JSON with a single key "
            "'keep_indexes' whose value is a list of integer indexes to keep."
        ),
        messages=[
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        ],
    )
    response = call_gemini(request, tracker)
    data = parse_json_block(response.content)

    indexes: Iterable[int]
    if isinstance(data, dict) and isinstance(data.get("keep_indexes"), list):
        indexes = [idx for idx in data["keep_indexes"] if isinstance(idx, int)]
    elif isinstance(data, list):
        indexes = [idx for idx in data if isinstance(idx, int)]
    else:
        indexes = []

    kept = [candidates[i] for i in indexes if 0 <= i < len(candidates)]
    return kept


def extract_record(
    user_prompt: str,
    candidate: TenderCandidate,
    model: str,
    tracker: TokenTracker,
) -> TenderRecord | None:
    """Use an LLM to convert a search result into structured tender data."""

    payload = {
        "user_prompt": user_prompt,
        "title": candidate.title,
        "url": candidate.url,
        "published_date": candidate.published_date,
        "content": candidate.text,
    }
    request = AIRequest(
        provider=Provider.GOOGLE,
        model=model,
        max_tokens=768,
        step_name="Extract tender",
        system=(
            "Extract tender details from the provided content. Respond with a JSON object "
            "containing exactly these keys: Title, Description, Location, Weblink, Posted_By, "
            "Value_GBP, Published_Date, Deadline. Use empty strings for unknown values."
        ),
        messages=[
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        ],
    )
    response = call_gemini(request, tracker)
    data = parse_json_block(response.content)
    if not isinstance(data, dict):
        return None

    def get_field(name: str) -> str:
        value = data.get(name)
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False)

    return TenderRecord(
        Title=get_field("Title") or candidate.title,
        Description=get_field("Description"),
        Location=get_field("Location"),
        Weblink=get_field("Weblink") or candidate.url,
        Posted_By=get_field("Posted_By"),
        Value_GBP=get_field("Value_GBP"),
        Published_Date=get_field("Published_Date") or (candidate.published_date or ""),
        Deadline=get_field("Deadline"),
    )


def extract_tenders(
    user_prompt: str,
    candidates: Sequence[TenderCandidate],
    model: str,
    tracker: TokenTracker,
) -> List[TenderRecord]:
    """Run structured extraction on each kept candidate."""

    records: List[TenderRecord] = []
    for candidate in candidates:
        record = extract_record(user_prompt, candidate, model, tracker)
        if record:
            records.append(record)
    return records


def export_csv(records: Sequence[TenderRecord], output_path: Path) -> None:
    """Write structured tender data to a CSV file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Title",
        "Description",
        "Location",
        "Weblink",
        "Posted_By",
        "Value_GBP",
        "Published_Date",
        "Deadline",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover tenders from a natural language prompt")
    parser.add_argument("prompt", help="Natural language description of desired tenders")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output CSV (defaults to output/tenders_<timestamp>.csv)",
    )
    parser.add_argument(
        "--query-model",
        default=DEFAULT_MODEL,
        help="Gemini model used for query generation",
    )
    parser.add_argument(
        "--filter-model",
        default=DEFAULT_MODEL,
        help="Gemini model used for filtering results",
    )
    parser.add_argument(
        "--extract-model",
        default=DEFAULT_MODEL,
        help="Gemini model used for structured extraction",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=10,
        help="Maximum number of queries to request from the LLM",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=5,
        help="Number of Exa results to request for each query",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    tracker = TokenTracker()

    queries = generate_queries(
        args.prompt,
        model=args.query_model,
        tracker=tracker,
        max_queries=max(args.max_queries, 1),
    )
    exa_api_key = None  # search_exa will read from the environment when None
    candidates = gather_candidates(queries, args.results_per_query, exa_api_key)
    kept_candidates = filter_candidates(args.prompt, candidates, args.filter_model, tracker)
    records = extract_tenders(args.prompt, kept_candidates, args.extract_model, tracker)

    if not args.output:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        args.output = Path("output") / f"tenders_{timestamp}.csv"

    export_csv(records, args.output)

    print(f"Generated {len(records)} tenders -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
