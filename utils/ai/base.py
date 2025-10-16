"""Base interfaces for AI/LLM providers"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class Provider(str, Enum):
    """AI provider options"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    # Add more providers as needed


class AnthropicModel(str, Enum):
    """Anthropic Claude models"""
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
    CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class OpenAIModel(str, Enum):
    """OpenAI models"""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"


class GeminiModel(str, Enum):
    """Google Gemini models"""
    # Gemini 2.5 (Latest)
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"

    # Gemini 2.0
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"

    # Gemini 1.5 (Legacy)
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_1_5_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_1_5_PRO = "gemini-1.5-pro"


class ToolDefinition(BaseModel):
    """
    Tool definition for function calling.
    Compatible with both Anthropic and OpenAI formats.
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


class ToolCall(BaseModel):
    """
    Tool call from LLM response.
    Normalized across providers.
    """
    id: str
    name: str
    arguments: Dict[str, Any]


class AIRequest(BaseModel):
    """
    Universal request interface for LLM calls.
    Works across different providers (Anthropic, OpenAI, etc.)
    """
    messages: List[Dict[str, Any]]
    model: str  # Can use enum values or custom strings
    provider: Provider = Provider.ANTHROPIC
    max_tokens: int = 1024
    temperature: Optional[float] = None
    system: Optional[str] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[str] = None  # "auto", "required", or specific tool name
    step_name: str = "LLM Call"  # For tracking/logging purposes
    json_mode: bool = False  # Enable structured output
    response_schema: Optional[Any] = None  # Pydantic model for structured output


class AIResponse(BaseModel):
    """
    Universal response interface from LLM calls.
    Normalizes responses across different providers.
    """
    content: str
    model: str
    provider: Provider
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None  # "end_turn", "tool_use", "max_tokens", etc.
    raw_response: Optional[Any] = Field(default=None, exclude=True)  # Original provider response


# Token tracking classes

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

    def track(self, step_name: str, response: AIResponse) -> TokenUsage:
        """
        Track token usage from AIResponse.

        Args:
            step_name: Name of the processing step
            response: AIResponse object from call_anthropic or call_gemini

        Returns:
            TokenUsage object with the usage data
        """
        try:
            usage = TokenUsage(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens
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
