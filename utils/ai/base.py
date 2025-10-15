"""Base interfaces for AI/LLM providers"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class Provider(str, Enum):
    """AI provider options"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
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
