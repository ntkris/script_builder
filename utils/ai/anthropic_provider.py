"""Anthropic Claude provider implementation"""

from typing import Optional, List
import os
import anthropic
from .base import AIRequest, AIResponse, Provider, ToolCall
from ..token_tracking import TokenTracker


def call_anthropic(
    request: AIRequest,
    token_tracker: Optional[TokenTracker] = None,
    api_key: Optional[str] = None
) -> AIResponse:
    """
    Make a call to Anthropic Claude API with automatic token tracking.

    Args:
        request: AIRequest with parameters
        token_tracker: Optional TokenTracker for automatic tracking
        api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var)

    Returns:
        AIResponse with normalized response data

    Example:
        from utils import call_anthropic, AIRequest, AnthropicModel, TokenTracker

        tracker = TokenTracker()

        request = AIRequest(
            messages=[{"role": "user", "content": "Hello!"}],
            model=AnthropicModel.CLAUDE_SONNET_4,
            max_tokens=1000,
            step_name="Greeting"
        )
        response = call_anthropic(request, tracker)
        print(response.content)
    """
    # Initialize client with API key
    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables or provided as argument")

    client = anthropic.Anthropic(api_key=api_key)
    # Build API call parameters
    params = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": request.messages,
    }

    # Add optional parameters
    if request.temperature is not None:
        params["temperature"] = request.temperature

    if request.system is not None:
        params["system"] = request.system

    # Add tools if provided
    if request.tools:
        params["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            }
            for tool in request.tools
        ]

        if request.tool_choice:
            # Map generic tool_choice to Anthropic format
            if request.tool_choice == "required":
                params["tool_choice"] = {"type": "any"}
            elif request.tool_choice == "auto":
                params["tool_choice"] = {"type": "auto"}
            else:
                # Specific tool name
                params["tool_choice"] = {"type": "tool", "name": request.tool_choice}

    # Make the API call
    message = client.messages.create(**params)

    # Extract response content and tool calls
    content = ""
    tool_calls = []

    for block in message.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input
            ))

    # Calculate token usage
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    total_tokens = input_tokens + output_tokens

    # Automatic token tracking
    if token_tracker:
        token_tracker.track(request.step_name, message)

    # Return normalized response
    return AIResponse(
        content=content.strip(),
        model=message.model,
        provider=Provider.ANTHROPIC,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        tool_calls=tool_calls if tool_calls else None,
        finish_reason=message.stop_reason,
        raw_response=message
    )
