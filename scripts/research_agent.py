#!/usr/bin/env python3
"""
Research Agent: Uses Exa for search and Google Gemini for analysis.

Given a topic, generates search queries, fetches information, and writes a comprehensive report.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
from utils import (
    call_gemini, call_anthropic, AIRequest, GeminiModel, AnthropicModel, Provider,
    save_json, search_exa, SearchResult, extract
)
from utils.step_logger import StepLogger
from dotenv import load_dotenv
load_dotenv()




# Pydantic models for structured data
class SearchQuery(BaseModel):
    """A search query to execute"""
    query: str
    reasoning: str


class QueryPlan(BaseModel):
    """Plan of search queries to execute"""
    queries: List[SearchQuery]


class RelevanceFilter(BaseModel):
    """Filter to determine if a search result is relevant"""
    is_relevant: bool = Field(description="True if this source is relevant to the research topic")
    reason: str = Field(description="Brief explanation of why it is or isn't relevant")
    key_insights: List[str] = Field(default_factory=list, description="Key insights from this source if relevant")


def generate_search_queries(logger: StepLogger, topic: str, num_queries: int = 5) -> List[SearchQuery]:
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

    try:
        request = AIRequest(
            messages=[{"role": "user", "content": prompt}],
            model=GeminiModel.GEMINI_2_5_FLASH,
            provider=Provider.GOOGLE,
            max_tokens=2000,
            step_name="Generate Search Queries",
            json_mode=True,
            response_schema=QueryPlan,
        )

        response = call_gemini(request, logger)
        return QueryPlan.model_validate(json.loads(response.content)).queries
    except Exception as e:
        logger.fail(e)
        print(f"❌ Error generating search queries: {e}")
        return []

def filter_results(logger: StepLogger, topic: str, results: List[SearchResult]) -> List[SearchResult]:
    """Filter search results for relevance using Gemini structured extraction"""
    print(f"🔍 Filtering {len(results)} search results for relevance...")

    relevant_results = []
    for i, result in enumerate(results):
        try:
            text = f"Title: {result.title}\nURL: {result.url}\nContent: {result.text[:1000]}"

            relevance = extract(
                text=text,
                schema=RelevanceFilter,
                prompt=f"Determine if this search result is relevant to researching: {topic}",
                logger=logger,
                step_name=f"Filter Result {i+1}/{len(results)}"
            )

            if relevance.is_relevant:
                print(f"  ✅ Relevant: {result.title}")
                relevant_results.append(result)
            else:
                print(f"  ❌ Not relevant: {result.title} - {relevance.reason}")

        except Exception as e:
            print(f"  ⚠️  Error filtering {result.title}: {e}")
            # If filtering fails, include the result
            relevant_results.append(result)

    print(f"✅ Filtered to {len(relevant_results)} relevant results")
    return relevant_results


def synthesize_findings(logger: StepLogger, topic: str, relevant_results: List[SearchResult], timestamp: str) -> str:
    """Synthesize search results into a comprehensive markdown report using Claude"""
    print(f"🤖 Synthesizing findings into report with Claude...")

    # Prepare search results for context
    results_context = "\n\n".join([
        f"SOURCE {i+1}: {r.title}\nURL: {r.url}\nCONTENT:\n{r.text}\n"
        for i, r in enumerate(relevant_results)
    ])

    prompt = f"""You are a research analyst. Synthesize the following search results into a comprehensive, well-structured research report about:

# Research Topic: {topic}

## Search Results
{results_context}

## Your Task
Write a detailed, professional research report in markdown format with:

1. **Executive Summary** (2-3 paragraphs) - High-level overview of findings
2. **4-6 Main Sections** - Each covering a different aspect of the topic
   - Use descriptive section titles (## Heading)
   - 2-4 paragraphs per section
   - Synthesize information across sources
   - Include insights, trends, and implications
3. **Key Findings** - Bulleted list of the most important discoveries
4. **Sources** - Numbered list of all sources cited

Guidelines:
- Be objective and balanced
- Cite sources naturally in text when referencing specific information
- Use proper markdown formatting
- Focus on quality insights, not just summarizing sources
- Make it engaging and informative

Write the complete report now:"""

    request = AIRequest(
        messages=[{"role": "user", "content": prompt}],
        model=AnthropicModel.CLAUDE_SONNET_4,
        max_tokens=4000,
        temperature=0.7,
        step_name="Synthesize Research Report"
    )

    response = call_anthropic(request, logger)

    # Add metadata header
    markdown = f"# Research Report: {topic}\n\n"
    markdown += f"*Generated: {timestamp}*\n\n"
    markdown += "---\n\n"
    markdown += response.content

    print(f"✅ Report synthesized ({len(markdown)} characters)")
    return markdown


def main():
    """Main research agent workflow"""
    if len(sys.argv) < 2:
        print("Usage: python research_agent.py <topic>")
        print("Example: python research_agent.py 'artificial general intelligence'")
        sys.exit(1)

    topic = " ".join(sys.argv[1:])
    logger = StepLogger("research_agent")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"🔬 RESEARCH AGENT")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")

    # Step 1: Generate search queries
    logger.step("Generate Search Queries", inputs={"topic": topic, "num_queries": 5})
    try:
        queries = generate_search_queries(logger, topic, num_queries=5)
        save_json(
            [q.model_dump() for q in queries],
            f"queries_{timestamp}.json",
            output_dir="cache",
            description="Search Queries"
        )
        logger.output({"queries": [q.model_dump() for q in queries], "count": len(queries)})
    except Exception as e:
        logger.fail(e)
        print(f"❌ Failed to generate search queries: {e}")
        logger.finalize()
        return

    # Step 2: Execute searches
    logger.step("Execute Searches", inputs={"num_queries": len(queries)})
    try:
        print(f"\n📡 Executing searches...")
        all_results = []
        for i, query in enumerate(queries):
            print(f"\n  Query: {query.query}")
            print(f"  Reasoning: {query.reasoning}")
            print(f"🔍 Searching: {query.query}")
            results = search_exa(query.query, num_results=3, max_characters=2000)
            print(f"  ✅ Found {len(results)} results")
            all_results.extend(results)
            logger.update({"completed": i + 1, "total": len(queries), "results_so_far": len(all_results)})

        print(f"\n✅ Total search results collected: {len(all_results)}")
        save_json(
            [r.model_dump() for r in all_results],
            f"search_results_{timestamp}.json",
            output_dir="cache",
            description="Search Results"
        )
        logger.output({"total_results": len(all_results)})
    except Exception as e:
        logger.fail(e)
        print(f"❌ Failed to execute searches: {e}")
        logger.finalize()
        return

    # Step 3: Filter relevant results
    logger.step("Filter Results", inputs={"total_results": len(all_results)})
    try:
        relevant_results = filter_results(logger, topic, all_results)
        save_json(
            [r.model_dump() for r in relevant_results],
            f"relevant_results_{timestamp}.json",
            output_dir="cache",
            description="Filtered Relevant Results"
        )
        logger.output({"relevant_results": len(relevant_results), "filtered_out": len(all_results) - len(relevant_results)})
    except Exception as e:
        logger.fail(e)
        print(f"❌ Failed to filter results: {e}")
        logger.finalize()
        return

    # Step 4: Synthesize report with Claude
    logger.step("Synthesize Report", inputs={"sources": len(relevant_results)})
    try:
        print(f"\n📝 Creating research report with Claude...")
        markdown = synthesize_findings(logger, topic, relevant_results, timestamp)
        logger.output({"report_length": len(markdown)})
    except Exception as e:
        logger.fail(e)
        print(f"❌ Failed to synthesize report: {e}")
        logger.finalize()
        return

    # Step 5: Save markdown report
    logger.step("Save Report")
    try:
        output_path = Path("outputs") / f"report_{timestamp}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(markdown)
        print(f"💾 Saved report: {output_path}")
        logger.output({"output_path": str(output_path)})
    except Exception as e:
        logger.fail(e)
        print(f"❌ Failed to save report: {e}")
        logger.finalize()
        return

    # Finalize logger
    logger.finalize()

    # Print summary
    print(f"\n{'='*60}")
    print(f"✅ RESEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Topic: {topic}")
    print(f"Queries executed: {len(queries)}")
    print(f"Total sources: {len(all_results)}")
    print(f"Relevant sources: {len(relevant_results)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
