#!/usr/bin/env python3
"""
Drawing Comparison Agent

Uses Claude Agent SDK to systematically compare design drawings with shop drawings
to identify inconsistencies that could impact construction.

Works for any drawing type: architectural, structural, MEP, interior design, etc.

Use case: Construction company flagged inconsistencies between design drawings and
fabrication shop's drawings causing construction delays.
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
    location: str  # Drawing section, elevation, detail, or sheet reference
    description: str
    design_drawing_value: Optional[str] = None  # Value from design drawing
    shop_drawing_value: Optional[str] = None     # Value from shop drawing
    construction_impact: str
    recommendation: str


class ComparisonReport(BaseModel):
    """Complete comparison report"""
    project_name: str
    client_name: Optional[str] = None
    drawing_type: Optional[str] = None  # e.g., "Interior Design", "MEP", "Structural"
    timestamp: str
    critical_issues: List[Discrepancy]
    moderate_issues: List[Discrepancy]
    minor_issues: List[Discrepancy]
    summary: dict


# Custom tools for the agent
@tool(
    name="record_discrepancy",
    description="Record a discrepancy found between the design drawing and shop drawing",
    input_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Unique identifier for this discrepancy"},
            "type": {"type": "string", "enum": ["dimension", "material", "missing_element", "specification", "detail_reference", "other"]},
            "severity": {"type": "string", "enum": ["critical", "moderate", "minor"]},
            "location": {"type": "string", "description": "Drawing section, elevation, detail, or sheet reference where discrepancy was found"},
            "description": {"type": "string", "description": "Clear description of the discrepancy"},
            "design_drawing_value": {"type": "string", "description": "Value/spec from design drawing"},
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
        design_drawing_value=args.get("design_drawing_value"),
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
    design_pdf_path: Path,
    shop_pdf_path: Path,
    project_name: str,
    client_name: Optional[str] = None,
    drawing_type: Optional[str] = None,
    logger: Optional[StepLogger] = None
) -> ComparisonReport:
    """Run the drawing comparison agent

    Args:
        design_pdf_path: Path to design drawing PDF
        shop_pdf_path: Path to shop drawing PDF
        project_name: Name of the project
        client_name: Client name (optional)
        drawing_type: Type of drawing (e.g., "Interior Design", "MEP", "Structural")
        logger: StepLogger instance for tracking
    """

    global discrepancies_found
    discrepancies_found = []  # Reset

    # Step 1: Read both PDFs
    logger.step("Load Drawing PDFs", inputs={
        "design_drawing": str(design_pdf_path),
        "shop_drawing": str(shop_pdf_path)
    })

    print(f"📄 Reading design drawing: {design_pdf_path.name}")
    print(f"📄 Reading shop drawing: {shop_pdf_path.name}")

    # Note: We'll pass the PDF paths to Claude and let it read them
    # The PDFs are already available to Claude through the Read tool

    logger.output({
        "design_drawing_size": f"{design_pdf_path.stat().st_size / 1024:.1f} KB",
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

    # Build project context
    project_context = f"Project: {project_name}"
    if client_name:
        project_context += f" | Client: {client_name}"
    if drawing_type:
        project_context += f" | Drawing Type: {drawing_type}"

    prompt = f"""You are a construction quality control expert reviewing technical drawings.

TASK: Systematically compare a design drawing with a shop drawing for the same project and identify ALL discrepancies that could impact construction.

{project_context}

DRAWINGS TO COMPARE:
1. Design Drawing: {design_pdf_path}
2. Shop Drawing: {shop_pdf_path}

COMPARISON METHODOLOGY:

1. DISCOVERY PHASE - First, read both PDFs thoroughly to understand:
   - What type of drawings these are (architectural, structural, MEP, interior, etc.)
   - The overall scope and what elements are shown
   - How the drawings are organized (sheets, elevations, sections, details, etc.)
   - Naming conventions and reference systems used
   - Units of measurement

2. SYSTEMATIC COMPARISON - Compare the drawings element by element:

   a) STRUCTURE & ORGANIZATION:
      - Are all sheets/sections from design drawing represented in shop drawing?
      - Are any elements missing or added?
      - Do reference systems align (detail callouts, grid lines, etc.)?

   b) DIMENSIONS & MEASUREMENTS:
      - Overall dimensions
      - Component dimensions
      - Heights, depths, widths
      - Spacing and clearances
      - Tolerances

   c) MATERIALS & SPECIFICATIONS:
      - Material types and grades
      - Finishes and surface treatments
      - Assembly methods
      - Hardware and fixtures
      - Product specifications

   d) DETAILS & CONNECTIONS:
      - Detail callouts exist in both drawings
      - Construction details match
      - Connection methods
      - Joint details
      - Edge conditions

   e) ANNOTATIONS & NOTES:
      - Critical notes present in both
      - Specification references
      - Installation instructions
      - Performance requirements

3. RECORD DISCREPANCIES using the record_discrepancy tool:

   SEVERITY GUIDELINES:
   - CRITICAL: Dimension mismatches >5%, missing structural/functional elements,
     wrong materials that affect performance, safety issues
   - MODERATE: Minor dimension differences (1-5%), finish discrepancies,
     unclear specifications, missing non-critical elements
   - MINOR: Notation differences, formatting variations, scale differences
     (if not affecting actual dimensions)

4. After completing analysis, use get_discrepancy_count to verify your findings

IMPORTANT PRINCIPLES:
- Let the drawings tell you what to compare - don't assume a structure
- Be systematic but adaptive to what's actually in the drawings
- Every discrepancy matters - construction delays are expensive
- Focus on construction impact and buildability
- Provide actionable recommendations
- If drawings use different terminology, note it but look for functional equivalence

Begin your systematic analysis now. Start by reading both PDFs to understand what you're working with."""

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
        project_name=project_name,
        client_name=client_name,
        drawing_type=drawing_type,
        timestamp=datetime.now().isoformat(),
        critical_issues=critical,
        moderate_issues=moderate,
        minor_issues=minor,
        summary={
            "total_discrepancies": len(discrepancies_found),
            "critical_count": len(critical),
            "moderate_count": len(moderate),
            "minor_count": len(minor)
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
        "design_drawing_value",
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

    # Project configuration
    sample_id = "sample_2"
    project_name = "27 Summit 17A - Daughter's Bedroom"
    client_name = "Mr. & Mrs. Kannan"
    drawing_type = "Interior Design"

    print("\n" + "="*60)
    print("🏗️  Drawing Comparison Agent")
    print("="*60)
    print(f"\nProject: {project_name}")
    print(f"Client: {client_name}")
    print(f"Drawing Type: {drawing_type}")
    print("="*60 + "\n")

    try:
        # Define paths
        design_pdf = Path(f"inputs/drawings/{sample_id}/interior_design_drawing.pdf")
        shop_pdf = Path(f"inputs/drawings/{sample_id}/shop_drawing.pdf")

        # Run comparison
        report = await run_comparison_agent(
            design_pdf,
            shop_pdf,
            project_name=project_name,
            client_name=client_name,
            drawing_type=drawing_type,
            logger=logger
        )

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
