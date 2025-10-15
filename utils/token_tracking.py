"""Token usage tracking utilities for API calls"""
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage counters"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class StepTokenUsage(BaseModel):
    """Token usage for a specific processing step"""
    step_name: str
    usage: TokenUsage
    timestamp: str


class TokenConsumptionSummary(BaseModel):
    """Summary of all token usage in a session"""
    video_path: str
    user_prompt: str
    steps: List[StepTokenUsage] = Field(default_factory=list)
    total_usage: TokenUsage = Field(default_factory=TokenUsage)
    session_timestamp: str


class TokenTracker:
    """Track token usage across multiple API calls"""

    def __init__(self):
        self.steps: List[StepTokenUsage] = []

    def track(self, step_name: str, message_response: any) -> TokenUsage:
        """
        Track token usage from API response.

        Args:
            step_name: Name of the processing step
            message_response: API response object with usage data

        Returns:
            TokenUsage object with the usage data
        """
        try:
            usage_data = message_response.usage
            usage = TokenUsage(
                input_tokens=usage_data.input_tokens,
                output_tokens=usage_data.output_tokens,
                total_tokens=usage_data.input_tokens + usage_data.output_tokens
            )

            step_usage = StepTokenUsage(
                step_name=step_name,
                usage=usage,
                timestamp=datetime.now().isoformat()
            )

            self.steps.append(step_usage)

            print(f"📊 {step_name}: {usage.input_tokens}→{usage.output_tokens} tokens")
            return usage

        except Exception as e:
            print(f"⚠️ Failed to track tokens for {step_name}: {e}")
            return TokenUsage()

    def get_total_usage(self) -> TokenUsage:
        """Calculate total token usage across all steps"""
        total = TokenUsage()
        for step in self.steps:
            total.input_tokens += step.usage.input_tokens
            total.output_tokens += step.usage.output_tokens
            total.total_tokens += step.usage.total_tokens
        return total

    def save_summary(
        self,
        video_path: str,
        output_dir: str = "outputs",
        user_prompt: str = ""
    ) -> Optional[str]:
        """
        Save token consumption summary to JSON file.

        Args:
            video_path: Path to the video being processed
            output_dir: Directory to save summary (default: "outputs")
            user_prompt: Optional user prompt to include in summary

        Returns:
            Path to saved summary file, or None if failed
        """
        if not self.steps:
            return None

        total_usage = self.get_total_usage()

        summary = TokenConsumptionSummary(
            video_path=video_path,
            user_prompt=user_prompt,
            steps=self.steps,
            total_usage=total_usage,
            session_timestamp=datetime.now().isoformat()
        )

        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/token_usage_{video_name}_{timestamp}.json"

        Path(output_dir).mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary.model_dump(), f, indent=2, ensure_ascii=False)

        print(f"\n💰 Total Tokens: {total_usage.input_tokens}→{total_usage.output_tokens} ({total_usage.total_tokens})")
        print(f"💾 Token usage: {output_path}")

        return output_path
