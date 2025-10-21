#!/usr/bin/env python3
"""Discover tenders aligned with a company's services and geography."""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Set
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from utils import (
    StepLogger,
    search_exa,
    SearchResult,
    save_json,
    extract,
    call_anthropic,
    AIRequest,
    AnthropicModel,
)
load_dotenv()

class SearchQuery(BaseModel):
    """Individual search query produced by Gemini."""

    query: str
    reasoning: str


class QueryPlan(BaseModel):
    """Container model for a batch of search queries."""

    queries: List[SearchQuery]


class PreFilterResult(BaseModel):
    """Quick classification of whether a search result is likely a tender page."""

    is_likely_tender: bool = Field(
        description="True if the page appears to be an active tender opportunity"
    )
    reason: str = Field(
        description="Brief explanation for the classification decision"
    )


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
    logger: StepLogger,
) -> List[SearchQuery]:
    """Use Claude Sonnet 4 to craft targeted procurement search queries."""

    prompt = f"""You are an expert procurement researcher generating search queries for Exa's semantic search engine to find active public tenders.

Company website: {company_url}
Location focus: {location}
Services and capabilities: {description}

Goal: Produce {max_queries} natural language search queries optimized for semantic search that will find ACTIVE, OPEN tender opportunities.

IMPORTANT - Query Format for Exa Semantic Search:
- Write natural language queries, NOT Google-style boolean operators
- NO "site:", "OR", quotation marks, or minus signs
- Focus on semantic meaning and intent
- Use descriptive phrases that would appear on tender pages
- Include recency signals like "new", "recent", "published this week/month"

Good Exa Query Examples:
✓ "Fire alarm installation tender opportunities published recently UK government"
✓ "New CCTV security system procurement contracts NHS education sector"
✓ "Active fire safety tender Scotland Wales local authorities"
✓ "Recent access control intruder alarm tender public sector buildings"
✓ "Fire door compartmentation contract notice UK healthcare"

Bad Query Examples (DO NOT USE):
✗ "site:contractsfinder.gov.uk \"fire alarm\" OR \"fire safety\" -award"
✗ "(tender OR procurement) AND (CCTV OR security) -awarded"
✗ "\"contract notice\" \"fire alarm\" -framework"

Target Sources to find (but reference naturally):
- contractsfinder.service.gov.uk (UK central government)
- find-tender.service.gov.uk (UK post-Brexit tenders)
- publiccontractsscotland.gov.uk (Scottish public sector)
- sell2wales.gov.wales (Welsh public sector)
- NHS procurement, education procurement, local authority tenders

What to Find:
- Active tender opportunities, RFPs, contract notices
- Recent procurement announcements
- Open bidding opportunities

What to Avoid (signal through natural language):
- Contract awards or concluded tenders
- Framework supplier lists
- Old or expired opportunities
- Preliminary market engagement only

Query Strategy:
- Cover different service areas (fire safety, security systems, access control, etc.)
- Vary by sector (NHS, education, local authorities, public buildings)
- Include geographic variants (UK-wide, Scotland, Wales, specific regions)
- Mix broad and specific service terms
- Emphasize recency and active status"""

    request = AIRequest(
        messages=[{"role": "user", "content": prompt}],
        model=AnthropicModel.CLAUDE_SONNET_4,
        max_tokens=2000,
        step_name="Generate Tender Queries",
        json_mode=True,
        response_schema=QueryPlan,
    )
    response = call_anthropic(request, logger)

    # Parse JSON response (already extracted by Anthropic provider)
    data = json.loads(response.content)
    plan = QueryPlan.model_validate(data)

    print(f"✅ Generated {len(plan.queries)} queries with Claude Sonnet 4")
    return plan.queries


def quick_prefilter(result: SearchResult, logger: StepLogger) -> PreFilterResult:
    """Fast Gemini Flash pre-filter to identify likely tender pages before full evaluation."""

    prompt = f"""You are a procurement analyst doing a quick first-pass filter on search results.

Classify whether this search result is likely an ACTIVE TENDER PAGE that should be evaluated in detail.

ACCEPT (is_likely_tender = true) if the page appears to be:
- An open tender, RFP, RFQ, or contract notice
- A live procurement opportunity
- An official procurement portal listing

REJECT (is_likely_tender = false) if the page appears to be:
- An award notice or contract already awarded
- A framework supplier list (existing framework, not a new tender)
- A tender aggregator or search portal (like bidstats.uk)
- A marketing page from a service provider company
- A preliminary market engagement notice (not the actual tender)
- A news article about procurement
- Expired or historical tender information

Search Result:
Title: {result.title}
URL: {result.url}
Snippet: {result.text[:500] if result.text else 'No text available'}

Provide a brief reason (1 sentence) for your decision."""

    try:
        prefilter = extract(
            text="",
            schema=PreFilterResult,
            prompt=prompt,
            logger=logger,
            step_name="Quick Pre-Filter",
        )
        return prefilter
    except (ValidationError, ValueError) as err:
        print(f"⚠️ Pre-filter failed for {result.url}: {err}")
        # Default to accepting on error (permissive)
        return PreFilterResult(
            is_likely_tender=True, reason="Pre-filter failed, passing through to full evaluation"
        )


def evaluate_search_result(
    result: SearchResult,
    company_url: str,
    location: str,
    description: str,
    logger: StepLogger,
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
Highlights: {highlights}"""

    text = f"Extracted page text:\n{result.text}"

    try:
        extraction = extract(
            text=text,
            schema=TenderExtraction,
            prompt=prompt,
            logger=logger,
            step_name="Evaluate Tender",
        )
        return extraction
    except (ValidationError, ValueError) as err:
        print(f"⚠️ Failed to parse tender data for {result.url}: {err}")
        return None


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
    parser.add_argument(
        "--url",
        default="https://www.kcsprojects.co.uk/",
        help="Company website URL (default: https://www.kcsprojects.co.uk/)",
    )
    parser.add_argument(
        "--location",
        default="United Kingdom",
        help="Primary geographic focus for tenders (default: United Kingdom)",
    )
    parser.add_argument(
        "--text",
        default="KCS Projects is a fire and security systems company specializing in comprehensive protection solutions for commercial and public sector buildings. Our core services include security system design and installation (intruder alarms, CCTV surveillance, access control systems, automated gates and gate automation, security fencing), fire safety services (fire alarm installation and maintenance, fire door installation, fire compartmentation, fire strategy development), and safeguarding services (safeguarding audits, safeguarding strategy, security audits, security strategy). We provide integrated fire and security solutions for education facilities, healthcare buildings, commercial properties, and public infrastructure across the UK.",
        help="Short description of the company's services and differentiators",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=8,
        help="Maximum number of search queries to generate (default: 8)",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=10,
        help="Number of Exa results to fetch per query (default: 10)",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    load_dotenv()

    logger = StepLogger("tender_finder")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Step 1: Generate search queries
    logger.step("Generate Search Queries", inputs={
        "company_url": args.url,
        "location": args.location,
        "max_queries": args.max_queries
    })

    queries = generate_search_queries(
        company_url=args.url,
        location=args.location,
        description=args.text,
        max_queries=args.max_queries,
        logger=logger,
    )

    # Log actual queries with reasoning for debugging
    logger.output({
        "queries": [q.model_dump() for q in queries]
    })

    # Step 2: Search and evaluate tenders
    logger.step("Search and Evaluate Tenders", inputs={
        "queries": [q.model_dump() for q in queries],
        "results_per_query": args.results_per_query,
        "target_location": args.location,
        "company_url": args.url
    })

    seen_urls: Set[str] = set()
    qualifying_rows = []
    all_evaluations = []  # Track decisions with reasons
    all_results = []

    # Date filter: only search results from last 30 days
    cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    for idx, query in enumerate(queries, start=1):
        print(f"\n🔎 Running query {idx}/{len(queries)}: {query.query}")
        results = search_exa(
            query.query,
            num_results=args.results_per_query,
            max_characters=4000,
            start_published_date=cutoff_date,
        )

        # Track all search results for cache (full results)
        for result in results:
            all_results.append({
                "query": query.query,
                "result": result.model_dump(),
            })

        for result in results:
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)

            # Stage 1: Quick pre-filter with Gemini Flash
            print(f"   🔍 Pre-filtering: {result.url}")
            prefilter = quick_prefilter(result, logger)

            if not prefilter.is_likely_tender:
                print(f"   ⏭️ Pre-filter rejected: {prefilter.reason}")
                all_evaluations.append({
                    "url": result.url,
                    "query": query.query,
                    "decision": "prefilter_rejected",
                    "reason": prefilter.reason,
                    "extraction": None,
                })
                continue

            print(f"   ✓ Pre-filter passed: {prefilter.reason}")

            # Stage 2: Detailed evaluation with Gemini Flash
            extraction = evaluate_search_result(
                result=result,
                company_url=args.url,
                location=args.location,
                description=args.text,
                logger=logger,
            )

            if extraction is None:
                all_evaluations.append({
                    "url": result.url,
                    "query": query.query,
                    "decision": "failed_extraction",
                    "reason": "Failed to parse extraction from LLM response",
                    "extraction": None,
                })
                continue

            # Ensure all gating criteria are satisfied
            recent_by_model = extraction.is_recent
            recent_by_date = within_last_30_days(extraction.published_date)
            location_match = extraction.is_location_match

            # Determine decision and reason
            if not extraction.is_tender:
                decision = "rejected_not_tender"
                decision_reason = extraction.reason
                print(f"   ⏭️ Skipping (not a tender): {extraction.reason}")
            elif not recent_by_model or not recent_by_date:
                decision = "rejected_not_recent"
                decision_reason = f"Published date: {extraction.published_date}, model says recent: {recent_by_model}"
                print(f"   ⏭️ Skipping (not recent): {extraction.published_date}")
            elif not location_match:
                decision = "rejected_wrong_location"
                decision_reason = extraction.reason
                print(f"   ⏭️ Skipping (wrong location): {extraction.reason}")
            else:
                decision = "accepted"
                decision_reason = f"Tender matches criteria: {extraction.reason}"
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

            # Log all evaluations with decisions and full extraction data
            all_evaluations.append({
                "url": result.url,
                "query": query.query,
                "decision": decision,
                "reason": decision_reason,
                "extraction": extraction.model_dump()
            })

        # Update progress after each query with decision breakdown
        decision_counts = {}
        for eval in all_evaluations:
            decision_counts[eval["decision"]] = decision_counts.get(eval["decision"], 0) + 1

        logger.update({
            "queries_completed": idx,
            "total_results": len(all_results),
            "total_evaluations": len(all_evaluations),
            "decision_breakdown": decision_counts
        })

    # Save interim search results to cache (too large for StepLogger)
    save_json(
        all_results,
        f"search_results_{timestamp}.json",
        output_dir="cache",
        description="All search results with full text",
    )

    # Log all evaluations with decisions and reasons for debugging
    logger.output({
        "evaluations": all_evaluations,  # Full extraction data with decisions
        "summary": {
            "total_results_fetched": len(all_results),
            "unique_urls_evaluated": len(all_evaluations),
            "prefilter_rejected": sum(1 for e in all_evaluations if e["decision"] == "prefilter_rejected"),
            "prefilter_passed": sum(1 for e in all_evaluations if e["decision"] != "prefilter_rejected"),
            "accepted": len(qualifying_rows),
            "rejected_not_tender": sum(1 for e in all_evaluations if e["decision"] == "rejected_not_tender"),
            "rejected_not_recent": sum(1 for e in all_evaluations if e["decision"] == "rejected_not_recent"),
            "rejected_wrong_location": sum(1 for e in all_evaluations if e["decision"] == "rejected_wrong_location"),
            "failed_extraction": sum(1 for e in all_evaluations if e["decision"] == "failed_extraction")
        }
    })

    if not qualifying_rows:
        print("\nNo qualifying tenders were found.")
        logger.finalize()
        return

    # Step 3: Save final CSV
    logger.step("Save Results", inputs={
        "qualifying_tenders": qualifying_rows  # Log actual tender data
    })

    # Always save final CSV to outputs directory
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
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

    logger.output({
        "csv_path": str(output_path),
        "tenders": qualifying_rows  # Log actual tenders saved to CSV
    })

    # Finalize logging
    logger.finalize()


if __name__ == "__main__":
    main()

