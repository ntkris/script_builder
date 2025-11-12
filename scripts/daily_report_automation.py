#!/usr/bin/env python3
"""
Daily Report Automation Script

Hybrid approach:
- Deterministic navigation (fast, reliable)
- Optional vision verification (self-correcting)
"""

import sys
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import argparse
from utils import extract, call_anthropic, AIRequest, AnthropicModel
from utils.step_logger import StepLogger
from playwright.sync_api import sync_playwright, Page

load_dotenv()

# ============================================================================
# Configuration (FILL THESE IN)
# ============================================================================

APP_URL = "https://example.com/daily-reports"  # TODO: Replace with actual URL
USERNAME = "your_username"  # TODO: Move to .env as REPORT_APP_USERNAME
PASSWORD = "your_password"  # TODO: Move to .env as REPORT_APP_PASSWORD

# ============================================================================
# Data Schema
# ============================================================================

class DailyReport(BaseModel):
    """Schema for daily site report"""
    date: str = Field(description="Report date (YYYY-MM-DD)")
    site_name: str = Field(description="Site or project name")
    weather: str = Field(description="Weather conditions")
    crew_count: int = Field(description="Number of crew members present")
    work_completed: str = Field(description="Summary of work completed")
    materials_used: Optional[str] = Field(default=None, description="Materials used")
    equipment: Optional[str] = Field(default=None, description="Equipment used")
    safety_incidents: Optional[str] = Field(default=None, description="Safety incidents or concerns")
    delays: Optional[str] = Field(default=None, description="Delays or issues")
    notes: Optional[str] = Field(default=None, description="Additional notes")

# ============================================================================
# Extraction
# ============================================================================

def extract_report(file_path: Path, logger: StepLogger) -> DailyReport:
    """Extract structured data from daily report file"""
    logger.step("Extract Report Data", inputs={"file": str(file_path)})

    print(f"📄 Reading report: {file_path.name}")

    with open(file_path, 'r') as f:
        content = f.read()

    # Extract using Gemini
    report = extract(
        text=content,
        schema=DailyReport,
        prompt="Extract all daily report information from this text. Be thorough and accurate.",
        logger=logger,
        step_name="Extract Daily Report"
    )

    print(f"✅ Extracted: {report.site_name} - {report.date}")
    logger.output({"report": report.model_dump()})

    return report

# ============================================================================
# Browser Automation (Deterministic)
# ============================================================================

def login(page: Page, logger: StepLogger):
    """Log into the application"""
    logger.step("Login to Application", inputs={"url": APP_URL})

    print(f"🌐 Navigating to {APP_URL}")
    page.goto(APP_URL)
    page.wait_for_load_state("networkidle")

    # TODO: Replace with actual selectors
    print("🔑 Logging in...")
    page.fill("#username", USERNAME)  # Replace selector
    page.fill("#password", PASSWORD)  # Replace selector
    page.click("button[type='submit']")  # Replace selector

    page.wait_for_load_state("networkidle")
    print("✅ Logged in")
    logger.output({"status": "logged_in"})

def fill_and_submit_form(page: Page, report: DailyReport, logger: StepLogger):
    """Fill out the daily report form and submit"""
    logger.step("Fill Form", inputs={"report_date": report.date})

    # TODO: Replace with actual selectors for your form
    print("📝 Filling form...")

    # Navigate to new report page (if needed)
    # page.click("#new_report_button")
    # page.wait_for_load_state("networkidle")

    # Fill form fields
    page.fill("#date", report.date)  # Replace selector
    page.fill("#site_name", report.site_name)  # Replace selector
    page.fill("#weather", report.weather)  # Replace selector
    page.fill("#crew_count", str(report.crew_count))  # Replace selector
    page.fill("#work_completed", report.work_completed)  # Replace selector

    if report.materials_used:
        page.fill("#materials", report.materials_used)  # Replace selector

    if report.equipment:
        page.fill("#equipment", report.equipment)  # Replace selector

    if report.safety_incidents:
        page.fill("#safety", report.safety_incidents)  # Replace selector

    if report.delays:
        page.fill("#delays", report.delays)  # Replace selector

    if report.notes:
        page.fill("#notes", report.notes)  # Replace selector

    print("✅ Form filled")
    logger.output({"fields_filled": True})

    # Submit
    logger.step("Submit Form")
    print("🚀 Submitting...")
    page.click("#submit_button")  # Replace selector
    page.wait_for_load_state("networkidle")

    print("✅ Form submitted")
    logger.output({"submitted": True})

# ============================================================================
# Vision Verification (Optional - Agentic)
# ============================================================================

def verify_submission(page: Page, logger: StepLogger) -> bool:
    """Use Claude vision to verify successful submission"""
    logger.step("Verify Submission")

    print("👁️  Taking screenshot for verification...")
    screenshot = page.screenshot()

    request = AIRequest(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Look at this screenshot. Was the daily report successfully submitted? "
                               "Look for success messages, confirmation text, or error messages. "
                               "Answer with 'SUCCESS' if submitted successfully, or 'ERROR: <description>' if there was a problem."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot.decode("utf-8") if isinstance(screenshot, bytes)
                                   else screenshot
                        }
                    }
                ]
            }
        ],
        model=AnthropicModel.CLAUDE_SONNET_4,
        max_tokens=200,
        step_name="Vision Verification"
    )

    response = call_anthropic(request, logger)
    result = response.content.strip()

    success = result.upper().startswith("SUCCESS")

    if success:
        print("✅ Verification: Submission successful")
    else:
        print(f"⚠️  Verification: {result}")

    logger.output({"verification": result, "success": success})
    return success

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automate daily report entry")
    parser.add_argument("file", help="Daily report file in inputs/")
    parser.add_argument("--no-verify", action="store_true", help="Skip vision verification")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    args = parser.parse_args()

    logger = StepLogger("daily_report_automation")

    # Step 1: Load and extract report
    input_path = Path("inputs") / args.file
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)

    report = extract_report(input_path, logger)

    # Step 2: Browser automation
    logger.step("Launch Browser")
    print("🚀 Launching browser...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context()
        page = context.new_page()

        logger.output({"browser": "chromium", "headless": args.headless})

        try:
            # Login
            login(page, logger)

            # Fill and submit
            fill_and_submit_form(page, report, logger)

            # Optional verification
            if not args.no_verify:
                verify_submission(page, logger)

            # Keep browser open for a moment to see result
            if not args.headless:
                print("✅ Complete! (Browser will close in 3 seconds)")
                page.wait_for_timeout(3000)

        except Exception as e:
            logger.fail(e)
            print(f"❌ Error: {e}")

            # Take screenshot on error
            error_screenshot = Path("outputs") / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            page.screenshot(path=str(error_screenshot))
            print(f"📸 Error screenshot saved: {error_screenshot}")

            raise

        finally:
            browser.close()

    # Finalize
    logger.finalize()
    print("🎉 Daily report automation complete!")

if __name__ == "__main__":
    main()
