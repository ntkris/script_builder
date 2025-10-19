#!/usr/bin/env python3
"""Discover tenders aligned with a company's services and geography."""

import argparse
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (  # noqa: E402
    AIRequest,
    GeminiModel,
    Provider,
    TokenTracker,
    call_gemini,
    search_exa,
    SearchResult,
)


class SearchQuery(BaseModel):
    """Individual search query produced by Gemini."""

    query: str
    reasoning: str


class QueryPlan(BaseModel):
    """Container model for a batch of search queries."""

    queries: List[SearchQuery]


class TenderExtraction(BaseModel):
    """Structured representation of a candidate tender opportunity."""

    is_tender: bool = Field(description="True when the page is an actual tender or RFP")
    is_recent: bool = Field(description="True when published within the last 30 days")
    is_location_match: bool = Field(description="True when the tender matches the target geography")
    reason: str = Field(description="Short explanation supporting the decision")
    title: str = Field(description="Tender title or headline")
    description: str = Field(description="Concise summary of the opportunity")
    published_date: Optional[str] = Field(
        default=None,
        description="ISO 8601 date string for publication (YYYY-MM-DD)",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="ISO 8601 date or descriptive deadline string",
    )
    value_gbp: Optional[float] = Field(
        default=None,
        description="Tender value converted to GBP as a numeric value",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Three-letter currency code for the stated tender value",
    )
    location: List[str] = Field(
        default_factory=list,
        description="List of geographic areas mentioned for the tender",
    )
    original_value: Optional[float] = Field(
        default=None,
        description="Original numeric value before conversion, if available",
    )
    original_currency: Optional[str] = Field(
        default=None,
        description="Currency for the original tender value",
    )


def generate_search_queries(
    company_url: str,
    location: str,
    description: str,
    max_queries: int,
    tracker: TokenTracker,
) -> List[SearchQuery]:
    """Use Gemini to craft targeted procurement search queries."""

    prompt = f"""You are an expert procurement researcher generating search queries to find active public tenders.

Company website: {company_url}
Location focus: {location}
Services and capabilities: {description}

Goal: produce {max_queries} highly targeted search queries that can be sent to a search engine specialising in tenders. Mix national procurement portals, local government sites, framework notices, and relevant sector terms. Prioritise recent and open opportunities.

Guidelines:
- Embed the target location or nearby regions when helpful
- Incorporate the company's service keywords and synonyms
- Prefer queries likely to surface official procurement notices or RFP listings
- Vary phrasing and site focus so the queries cover different lead sources

Return only structured JSON matching the specified schema."""

    request = AIRequest(
        messages=[{"role": "user", "content": prompt}],
        model=GeminiModel.GEMINI_2_5_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=1200,
        json_mode=True,
        response_schema=QueryPlan,
        step_name="Generate Tender Queries",
    )

    response = call_gemini(request, tracker)

    try:
        plan = QueryPlan.model_validate_json(response.content)
        print(f"✅ Generated {len(plan.queries)} queries")
        return plan.queries
    except ValidationError as err:
        print(f"⚠️ Failed to parse structured queries: {err}")
    except ValueError as err:
        print(f"⚠️ Gemini response was not valid JSON: {err}")

    print("Using fallback tender queries.")
    fallback = [
        SearchQuery(
            query=f"{location} public tender {description}",
            reasoning="General tender search",
        ),
        SearchQuery(
            query=f"{location} procurement opportunity site:gov", reasoning="Government portals"
        ),
    ]
    return fallback[:max_queries]


def evaluate_search_result(
    result: SearchResult,
    company_url: str,
    location: str,
    description: str,
    tracker: TokenTracker,
) -> Optional[TenderExtraction]:
    """Determine whether a search result is a viable tender and extract key facts."""

    today = datetime.utcnow().strftime("%Y-%m-%d")
    highlights = "\n".join(result.highlights) if result.highlights else "None"

    prompt = f"""You are a procurement analyst vetting tenders for a business development team.

Company website: {company_url}
Company capabilities: {description}
Target geography: {location}
Current date: {today}

Evaluate the search result below. Decide if it represents an active tender, RFP, RFQ, contract notice, or framework opportunity that matches the company's services and location focus. Reject news articles, award notices, expired tenders, or unrelated opportunities.

Requirements:
- Only set is_tender to true for genuine, open procurement opportunities.
- Confirm the notice was published within the last 30 days relative to the current date.
- Confirm the tender applies to the target geography or explicitly mentions a relevant location.
- When values are expressed in other currencies, convert the best estimate to GBP (numeric) using reasonable contemporary rates.
- Use ISO 8601 format (YYYY-MM-DD) for dates when possible.
- Provide concise but information-rich descriptions suitable for a CRM entry.

Search result metadata:
Query: {result.query}
Title: {result.title}
URL: {result.url}
Published date (from search): {result.published_date or 'Unknown'}
Highlights: {highlights}

Extracted page text:
{result.text}

Return JSON that conforms to the provided schema."""

    request = AIRequest(
        messages=[{"role": "user", "content": prompt}],
        model=GeminiModel.GEMINI_2_5_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=1800,
        json_mode=True,
        response_schema=TenderExtraction,
        step_name="Evaluate Tender",
    )

    response = call_gemini(request, tracker)

    try:
        extraction = TenderExtraction.model_validate_json(response.content)
    except (ValidationError, ValueError) as err:
        print(f"⚠️ Failed to parse tender data for {result.url}: {err}")
        return None

    return extraction


def within_last_30_days(published_date: Optional[str]) -> bool:
    """Validate that a date string falls within the last 30 days."""

    if not published_date:
        return False

    try:
        parsed = datetime.fromisoformat(published_date.replace("Z", ""))
    except ValueError:
        return False

    cutoff = datetime.utcnow() - timedelta(days=30)
    return parsed >= cutoff


def build_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""

    parser = argparse.ArgumentParser(description="Discover fresh tenders tailored to a company.")
    parser.add_argument("url", help="Company website URL")
    parser.add_argument("location", help="Primary geographic focus for tenders")
    parser.add_argument(
        "text",
        help="Short description of the company's services and differentiators",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=6,
        help="Maximum number of search queries to generate",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=6,
        help="Number of Exa results to fetch per query",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for the resulting CSV file",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    load_dotenv()

    tracker = TokenTracker()

    queries = generate_search_queries(
        company_url=args.url,
        location=args.location,
        description=args.text,
        max_queries=args.max_queries,
        tracker=tracker,
    )

    seen_urls: Set[str] = set()
    qualifying_rows = []

    for idx, query in enumerate(queries, start=1):
        print(f"\n🔎 Running query {idx}/{len(queries)}: {query.query}")
        results = search_exa(query.query, num_results=args.results_per_query, max_characters=4000)

        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)

            extraction = evaluate_search_result(
                result=result,
                company_url=args.url,
                location=args.location,
                description=args.text,
                tracker=tracker,
            )

            if extraction is None:
                continue

            # Ensure all gating criteria are satisfied
            recent_by_model = extraction.is_recent
            recent_by_date = within_last_30_days(extraction.published_date)
            location_match = extraction.is_location_match

            if not extraction.is_tender:
                print(f"   ⏭️ Skipping (not a tender): {extraction.reason}")
                continue

            if not recent_by_model or not recent_by_date:
                print(f"   ⏭️ Skipping (not recent): {extraction.published_date}")
                continue

            if not location_match:
                print(f"   ⏭️ Skipping (wrong location): {extraction.reason}")
                continue

            qualifying_rows.append(
                {
                    "url": result.url,
                    "title": extraction.title,
                    "description": extraction.description,
                    "published_date": extraction.published_date or "",
                    "deadline": extraction.deadline or "",
                    "value_in_gbp": f"{extraction.value_gbp:.2f}" if extraction.value_gbp is not None else "",
                    "currency": extraction.original_currency
                    or extraction.currency
                    or ("GBP" if extraction.value_gbp else ""),
                    "location": "; ".join(extraction.location),
                }
            )

            print(f"   ✅ Added tender: {extraction.title}")

    if not qualifying_rows:
        print("\nNo qualifying tenders were found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"tenders_{timestamp}.csv"

    fieldnames = [
        "url",
        "title",
        "description",
        "published_date",
        "deadline",
        "value_in_gbp",
        "currency",
        "location",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(qualifying_rows)

    print(f"\n📄 Saved {len(qualifying_rows)} tenders to {output_path}")


if __name__ == "__main__":
    main()

