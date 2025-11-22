#!/usr/bin/env python3
"""
Drawing Comparison Agent

Uses Claude Agent SDK to systematically compare interior design drawings
with shop drawings to identify inconsistencies that could impact construction.

Use case: Construction company flagged inconsistencies between interior designer's
drawings and fabrication shop's drawings causing construction delays.
"""

import asyncio
import csv
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import List, Literal, Optional
from dotenv import load_dotenv
from utils.step_logger import StepLogger
from utils import save_json
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, tool, create_sdk_mcp_server

load_dotenv()

# Pydantic models for structured data
class Discrepancy(BaseModel):
    """A single discrepancy found between drawings"""
    id: str
    type: Literal["dimension", "material", "missing_element", "specification", "detail_reference", "other"]
    severity: Literal["critical", "moderate", "minor"]
    location: str  # Which elevation/detail
    description: str
    interior_design_value: Optional[str] = None
    shop_drawing_value: Optional[str] = None
    construction_impact: str
    recommendation: str


class ComparisonReport(BaseModel):
    """Complete comparison report"""
    project: str
    client: str
    timestamp: str
    critical_issues: List[Discrepancy]
    moderate_issues: List[Discrepancy]
    minor_issues: List[Discrepancy]
    summary: dict


# Custom tools for the agent
@tool(
    name="record_discrepancy",
    description="Record a discrepancy found between the interior design and shop drawings",
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique identifier for this discrepancy"},
            "type": {"type": "string", "enum": ["dimension", "material", "missing_element", "specification", "detail_reference", "other"]},
            "severity": {"type": "string", "enum": ["critical", "moderate", "minor"]},
            "location": {"type": "string", "description": "Which elevation/detail (e.g., 'Elevation A', 'Detail B')"},
            "description": {"type": "string", "description": "Clear description of the discrepancy"},
            "interior_design_value": {"type": "string", "description": "Value/spec from interior design drawing"},
            "shop_drawing_value": {"type": "string", "description": "Value/spec from shop drawing"},
            "construction_impact": {"type": "string", "description": "How this impacts construction"},
            "recommendation": {"type": "string", "description": "Recommended resolution"}
        },
        "required": ["id", "type", "severity", "location", "description", "construction_impact", "recommendation"]
    }
)
async def record_discrepancy(args):
    """Record a discrepancy - stores it in global list for later compilation"""
    global discrepancies_found

    discrepancy = Discrepancy(
        id=args["id"],
        type=args["type"],
        severity=args["severity"],
        location=args["location"],
        description=args["description"],
        interior_design_value=args.get("interior_design_value"),
        shop_drawing_value=args.get("shop_drawing_value"),
        construction_impact=args["construction_impact"],
        recommendation=args["recommendation"]
    )

    discrepancies_found.append(discrepancy)

    return {
        "content": [{
            "type": "text",
            "text": f"✅ Recorded {discrepancy.severity} discrepancy: {discrepancy.id} at {discrepancy.location}"
        }]
    }


@tool(
    name="get_discrepancy_count",
    description="Get current count of discrepancies found so far",
    input_schema={
        "type": "object",
        "properties": {},
        "required": []
    }
)
async def get_discrepancy_count(args):
    """Get count of discrepancies found so far"""
    global discrepancies_found

    critical = sum(1 for d in discrepancies_found if d.severity == "critical")
    moderate = sum(1 for d in discrepancies_found if d.severity == "moderate")
    minor = sum(1 for d in discrepancies_found if d.severity == "minor")

    return {
        "content": [{
            "type": "text",
            "text": f"Total: {len(discrepancies_found)} | Critical: {critical} | Moderate: {moderate} | Minor: {minor}"
        }]
    }


# Global storage for discrepancies (reset for each run)
discrepancies_found: List[Discrepancy] = []


async def run_comparison_agent(
    interior_pdf_path: Path,
    shop_pdf_path: Path,
    logger: StepLogger
) -> ComparisonReport:
    """Run the drawing comparison agent"""

    global discrepancies_found
    discrepancies_found = []  # Reset

    # Step 1: Read both PDFs
    logger.step("Load Drawing PDFs", inputs={
        "interior_design": str(interior_pdf_path),
        "shop_drawing": str(shop_pdf_path)
    })

    print(f"📄 Reading interior design drawing: {interior_pdf_path.name}")
    print(f"📄 Reading shop drawing: {shop_pdf_path.name}")

    # Note: We'll pass the PDF paths to Claude and let it read them
    # The PDFs are already available to Claude through the Read tool

    logger.output({
        "interior_design_size": f"{interior_pdf_path.stat().st_size / 1024:.1f} KB",
        "shop_drawing_size": f"{shop_pdf_path.stat().st_size / 1024:.1f} KB"
    })

    # Step 2: Create MCP server with custom tools
    logger.step("Initialize Agent with Custom Tools")

    print("🔧 Creating MCP server with drawing comparison tools...")
    server = create_sdk_mcp_server(
        name="drawing-comparison-tools",
        version="1.0.0",
        tools=[record_discrepancy, get_discrepancy_count]
    )

    # Configure agent with tools
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",  # Use Haiku 4.5 (faster, cheaper: $1/$5 vs Sonnet's $3/$15)
        mcp_servers={"drawing": server},
        allowed_tools=[
            "Read",  # Allow reading PDF files
            "mcp__drawing__record_discrepancy",
            "mcp__drawing__get_discrepancy_count"
        ],
        max_turns=40,  # Allow multiple back-and-forth for complete analysis
        max_buffer_size=10 * 1024 * 1024  # 10MB buffer for PDF handling (default is 1MB)
    )

    logger.output({
        "model": "claude-haiku-4-5",
        "tools_registered": ["record_discrepancy", "get_discrepancy_count"],
        "max_turns": 40,
        "max_buffer_size": "10MB"
    })

    # Step 3: Run agent comparison
    logger.step("Run Agent Comparison Analysis", inputs={
        "task": "Systematic comparison of drawings for construction discrepancies"
    })

    print("\n" + "="*60)
    print("🤖 Starting Agent Analysis...")
    print("="*60)

    prompt = f"""You are a construction quality control expert reviewing architectural drawings.

TASK: Compare two drawings for the same project (Daughter's Bedroom at 27 Summit 17A) and identify ALL discrepancies that could impact construction.

DRAWINGS TO COMPARE:
1. Interior Design Drawing: {interior_pdf_path}
2. Shop Drawing: {shop_pdf_path}

COMPARISON METHODOLOGY:
1. Read both PDFs thoroughly
2. For EACH elevation (A, B, C) systematically compare:
   - Overall dimensions
   - Component dimensions
   - Materials and finishes
   - Detail references (A, B, C, D, E, F, G, H)
   - Heights and levels
   - Specifications

3. For EACH detail callout (Details A-H):
   - Check if it exists in both drawings
   - Compare dimensions and specifications
   - Verify consistency

4. Use the record_discrepancy tool for EVERY inconsistency found:
   - CRITICAL: Dimension mismatches, missing structural elements, wrong materials
   - MODERATE: Finish discrepancies, minor dimension differences, unclear specs
   - MINOR: Notation differences, scale variations (if not impacting actual size)

5. Be thorough - construction delays are expensive. Check:
   - Marble skirting dimensions (75mm mentioned?)
   - Paneling specifications (18+18 HDHMR vs other materials)
   - Study unit dimensions and specifications
   - Architrave details and grooves
   - Cabinet and shutter specifications
   - All detail callouts match between drawings

6. After completing analysis, use get_discrepancy_count to verify your findings

IMPORTANT:
- Read the actual PDF files using the Read tool
- Be systematic - go elevation by elevation
- Record EVERY discrepancy, no matter how small
- Focus on construction impact
- Provide actionable recommendations

Begin your systematic analysis now."""

    messages = []
    token_usage = None
    total_cost = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        print("\n📨 Agent messages:\n")
        async for message in client.receive_response():
            messages.append(message)
            print(f"  {message}")

            # Extract token usage from ResultMessage
            if hasattr(message, 'usage') and message.usage:
                token_usage = message.usage
            if hasattr(message, 'total_cost_usd') and message.total_cost_usd:
                total_cost = message.total_cost_usd

    logger.output({
        "messages_received": len(messages),
        "discrepancies_found": len(discrepancies_found),
        "all_messages": [str(m) for m in messages],  # Full conversation as array
        "token_usage": token_usage,  # Token breakdown from SDK
        "total_cost_usd": total_cost
    })

    # Step 4: Compile report
    logger.step("Compile Discrepancy Report")

    print(f"\n✅ Analysis complete. Found {len(discrepancies_found)} discrepancies.")

    # Categorize by severity
    critical = [d for d in discrepancies_found if d.severity == "critical"]
    moderate = [d for d in discrepancies_found if d.severity == "moderate"]
    minor = [d for d in discrepancies_found if d.severity == "minor"]

    print(f"   Critical: {len(critical)}")
    print(f"   Moderate: {len(moderate)}")
    print(f"   Minor: {len(minor)}")

    report = ComparisonReport(
        project="27 Summit 17A - Daughter's Bedroom",
        client="Mr. & Mrs. Kannan",
        timestamp=datetime.now().isoformat(),
        critical_issues=critical,
        moderate_issues=moderate,
        minor_issues=minor,
        summary={
            "total_discrepancies": len(discrepancies_found),
            "critical_count": len(critical),
            "moderate_count": len(moderate),
            "minor_count": len(minor),
            "elevations_reviewed": ["A", "B", "C"],
            "details_reviewed": ["A", "B", "C", "D", "E", "F", "G", "H"]
        }
    )

    logger.output({
        "report_summary": report.summary,
        "critical_issues": [d.model_dump() for d in critical],
        "moderate_issues": [d.model_dump() for d in moderate],
        "minor_issues": [d.model_dump() for d in minor]
    })

    return report


def save_discrepancies_to_csv(report: ComparisonReport, output_path: Path):
    """Save discrepancies to CSV file"""

    # Collect all discrepancies
    all_discrepancies = []
    all_discrepancies.extend(report.critical_issues)
    all_discrepancies.extend(report.moderate_issues)
    all_discrepancies.extend(report.minor_issues)

    # CSV columns
    fieldnames = [
        "id",
        "severity",
        "type",
        "location",
        "description",
        "interior_design_value",
        "shop_drawing_value",
        "construction_impact",
        "recommendation"
    ]

    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for disc in all_discrepancies:
            writer.writerow(disc.model_dump())

    return len(all_discrepancies)


async def main():
    """Main execution"""
    logger = StepLogger("drawing_comparison_agent")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "="*60)
    print("🏗️  Drawing Comparison Agent")
    print("="*60)
    print("\nComparing interior design vs shop drawings")
    print("Client: Mr. & Mrs. Kannan")
    print("Project: 27 Summit 17A - Daughter's Bedroom")
    print("="*60 + "\n")

    try:
        # Define paths
        sample_id = "sample_2"
        interior_pdf = Path(f"inputs/drawings/{sample_id}/interior_design_drawing.pdf")
        shop_pdf = Path(f"inputs/drawings/{sample_id}/shop_drawing.pdf")

        # Run comparison
        report = await run_comparison_agent(interior_pdf, shop_pdf, logger)

        # Save report
        logger.step("Save Discrepancy Report")

        # Save JSON
        json_filename = f"drawing_discrepancies_{timestamp}.json"
        save_json(
            report.model_dump(),
            json_filename,
            output_dir="outputs",
            description="Drawing Comparison Discrepancy Report"
        )

        # Save CSV
        csv_filename = f"drawing_discrepancies_{timestamp}.csv"
        csv_path = Path("outputs") / csv_filename
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        discrepancy_count = save_discrepancies_to_csv(report, csv_path)

        print(f"💾 CSV saved: {csv_path}")

        logger.output({
            "json_saved_to": f"outputs/{json_filename}",
            "csv_saved_to": f"outputs/{csv_filename}",
            "total_issues": discrepancy_count
        })

        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"Total Discrepancies: {report.summary['total_discrepancies']}")
        print(f"  🔴 Critical: {report.summary['critical_count']}")
        print(f"  🟡 Moderate: {report.summary['moderate_count']}")
        print(f"  ⚪ Minor: {report.summary['minor_count']}")
        print(f"\n📁 JSON Report: outputs/{json_filename}")
        print(f"📁 CSV Report: outputs/{csv_filename}")
        print("="*60)

        # Print critical issues if any
        if report.critical_issues:
            print("\n🔴 CRITICAL ISSUES (require immediate attention):")
            for issue in report.critical_issues:
                print(f"\n  {issue.id} - {issue.location}")
                print(f"  {issue.description}")
                print(f"  Impact: {issue.construction_impact}")
                print(f"  Recommendation: {issue.recommendation}")

        # Finalize
        logger.finalize()
        print("\n✅ Analysis complete!")

    except Exception as e:
        logger.fail(e)
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
