#!/usr/bin/env python3
"""
Research Agent: Uses Exa for search and Google Gemini for analysis.

Given a topic, generates search queries, fetches information, and writes a comprehensive report.
"""

import sys
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from dotenv import load_dotenv

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()
from utils import (
    call_gemini, AIRequest, GeminiModel, Provider,
    TokenTracker, save_json, search_exa, SearchResult
)


# Pydantic models for structured data
class SearchQuery(BaseModel):
    """A search query to execute"""
    query: str
    reasoning: str


class QueryPlan(BaseModel):
    """Plan of search queries to execute"""
    queries: List[SearchQuery]


class ResearchSection(BaseModel):
    """A section in the research report"""
    title: str
    content: str


class ResearchReport(BaseModel):
    """Final research report"""
    topic: str
    summary: str
    sections: List[ResearchSection]
    sources: List[str]
    timestamp: str


# Initialize
tracker = TokenTracker()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_search_queries(topic: str, num_queries: int = 5) -> List[SearchQuery]:
    """Generate search queries for the given topic using Gemini"""
    print(f"🤖 Generating {num_queries} search queries for: {topic}")

    prompt = f"""Generate {num_queries} diverse search queries to research the following topic comprehensively:

Topic: {topic}

Requirements:
- Create queries that cover different aspects of the topic
- Include queries for background, current state, trends, and implications
- Make queries specific and searchable
- Each query should explore a unique angle

Return your response as a JSON object with this structure:
{{
  "queries": [
    {{"query": "...", "reasoning": "..."}},
    ...
  ]
}}"""

    request = AIRequest(
        messages=[{"role": "user", "content": prompt}],
        model=GeminiModel.GEMINI_2_5_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=2000,
        step_name="Generate Search Queries"
    )

    response = call_gemini(request, tracker)

    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        queries = [SearchQuery(**q) for q in data["queries"]]
        print(f"✅ Generated {len(queries)} queries")
        return queries
    except Exception as e:
        print(f"⚠️  Failed to parse queries: {e}")
        print(f"Response: {response.content[:200]}...")
        # Fallback to simple queries
        return [
            SearchQuery(query=f"{topic} overview", reasoning="Get general overview"),
            SearchQuery(query=f"{topic} latest developments", reasoning="Find recent news"),
            SearchQuery(query=f"{topic} benefits and challenges", reasoning="Understand pros/cons"),
        ]


def synthesize_findings(topic: str, all_results: List[SearchResult]) -> ResearchReport:
    """Synthesize search results into a comprehensive report using Gemini"""
    print(f"🤖 Synthesizing findings into report...")

    # Prepare search results for context
    results_context = "\n\n".join([
        f"SOURCE: {r.title}\nURL: {r.url}\nCONTENT: {r.text}\n"
        for r in all_results
    ])

    prompt = f"""You are a research analyst. Synthesize the following search results into a comprehensive, well-structured research report about: {topic}

SEARCH RESULTS:
{results_context}

Create a detailed report with:
1. An executive summary (2-3 paragraphs)
2. 4-6 main sections covering different aspects of the topic
3. Each section should have a title and detailed content (2-4 paragraphs)
4. Synthesize information across sources, don't just list findings
5. Include insights, trends, and implications
6. Be objective and balanced

Return your response as a markdown string."""

    request = AIRequest(
        messages=[{"role": "user", "content": prompt}],
        model=GeminiModel.GEMINI_2_5_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=4000,
        temperature=0.7,
        step_name="Synthesize Research Report"
    )

    response = call_gemini(request, tracker)

    # Parse JSON response
    try:
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        data = json.loads(content)
        report = ResearchReport(
            topic=data.get("topic", topic),
            summary=data.get("summary", ""),
            sections=[ResearchSection(**s) for s in data.get("sections", [])],
            sources=data.get("sources", []),
            timestamp=timestamp
        )
        print(f"✅ Report synthesized with {len(report.sections)} sections")
        return report

    except Exception as e:
        print(f"⚠️  Failed to parse report: {e}")
        print(f"Response: {response.content[:200]}...")
        # Create a simple fallback report
        return ResearchReport(
            topic=topic,
            summary=response.content[:500],
            sections=[ResearchSection(title="Findings", content=response.content)],
            sources=[r.url for r in all_results[:10]],
            timestamp=timestamp
        )


def format_markdown_report(report: ResearchReport) -> str:
    """Format the research report as markdown"""
    md = f"# Research Report: {report.topic}\n\n"
    md += f"*Generated: {report.timestamp}*\n\n"
    md += "---\n\n"
    md += "## Executive Summary\n\n"
    md += f"{report.summary}\n\n"
    md += "---\n\n"

    for section in report.sections:
        md += f"## {section.title}\n\n"
        md += f"{section.content}\n\n"

    md += "---\n\n"
    md += "## Sources\n\n"
    for i, source in enumerate(report.sources, 1):
        md += f"{i}. {source}\n"

    return md


def main():
    """Main research agent workflow"""
    if len(sys.argv) < 2:
        print("Usage: python research_agent.py <topic>")
        print("Example: python research_agent.py 'artificial general intelligence'")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])

    print(f"\n{'='*60}")
    print(f"🔬 RESEARCH AGENT")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")

    # Step 1: Generate search queries
    queries = generate_search_queries(topic, num_queries=5)
    save_json(
        [q.model_dump() for q in queries],
        f"queries_{timestamp}.json",
        output_dir="cache",
        description="Search Queries"
    )

    # Step 2: Execute searches
    print(f"\n📡 Executing searches...")
    all_results = []
    for query in queries:
        print(f"\n  Query: {query.query}")
        print(f"  Reasoning: {query.reasoning}")
        print(f"🔍 Searching: {query.query}")
        results = search_exa(query.query, num_results=3, max_characters=2000)
        print(f"  ✅ Found {len(results)} results")
        all_results.extend(results)

    print(f"\n✅ Total search results collected: {len(all_results)}")
    save_json(
        [r.model_dump() for r in all_results],
        f"search_results_{timestamp}.json",
        output_dir="cache",
        description="Search Results"
    )

    # Step 3: Synthesize report
    print(f"\n📝 Creating research report...")
    report = synthesize_findings(topic, all_results)

    # Step 4: Save outputs
    # Save JSON version to cache
    save_json(
        report.model_dump(),
        f"report_{timestamp}.json",
        output_dir="cache",
        description="Research Report (JSON)"
    )

    # Save markdown version to outputs
    markdown = format_markdown_report(report)
    output_path = Path("outputs") / f"report_{timestamp}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)
    print(f"💾 Saved markdown report: {output_path}")

    # Step 5: Save token summary
    tracker.save_summary("research_agent", output_dir="cache")

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ RESEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Queries executed: {len(queries)}")
    print(f"Sources analyzed: {len(all_results)}")
    print(f"Report sections: {len(report.sections)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Print executive summary
    print("📊 EXECUTIVE SUMMARY:")
    print(f"{report.summary}\n")


if __name__ == "__main__":
    main()
