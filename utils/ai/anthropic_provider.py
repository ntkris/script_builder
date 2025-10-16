"""Anthropic Claude provider implementation"""

from typing import Optional, List, Type
import os
import json
import anthropic
import xmltodict
from pydantic import BaseModel
from .base import AIRequest, AIResponse, Provider, ToolCall, TokenTracker


def _generate_json_schema_prompt(schema) -> str:
    """Generate JSON schema description from Pydantic model for Claude"""
    import typing

    # Check if it's a list type
    schema_origin = typing.get_origin(schema)
    if schema_origin is list or schema_origin is List:
        item_type = typing.get_args(schema)[0]
        schema_dict = item_type.model_json_schema()
        return f"Please output a JSON array of objects matching this schema: {json.dumps(schema_dict, indent=2)}"
    else:
        schema_dict = schema.model_json_schema()
        return f"Please output a JSON object matching this schema: {json.dumps(schema_dict, indent=2)}"


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

    # Handle structured output with XML-wrapped JSON
    messages = request.messages.copy()
    system_prompt = request.system or ""

    if request.json_mode and request.response_schema:
        # Generate JSON schema prompt
        json_schema_prompt = _generate_json_schema_prompt(request.response_schema)

        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{json_schema_prompt}"
        else:
            system_prompt = json_schema_prompt

        # Prefill assistant response with opening XML tag
        # Claude will output JSON and close with </result>
        messages.append({"role": "assistant", "content": "<result>"})

    # Build API call parameters
    params = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
    }

    # Add optional parameters
    if request.temperature is not None:
        params["temperature"] = request.temperature

    if system_prompt:
        params["system"] = system_prompt

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

    # If json_mode is enabled, parse XML-wrapped JSON
    if request.json_mode and request.response_schema:
        try:
            # Claude outputs: JSON</result>
            # Reconstruct full XML by adding opening tag (closing tag already there)
            xml_content = "<result>" + content
            parsed = xmltodict.parse(xml_content)
            result_data = parsed["result"]

            # If xmltodict parsed it as a string, that's the JSON - use it directly
            # If it parsed it as a dict/list, convert back to JSON string
            if isinstance(result_data, str):
                content = result_data
            else:
                content = json.dumps(result_data)
        except Exception as e:
            # If XML parsing fails, return raw content with error indication
            print(f"⚠️  Warning: Failed to parse XML response: {e}")
            print(f"Raw content: {content}")
            import traceback
            traceback.print_exc()
            # Keep original content

    # Calculate token usage
    input_tokens = message.usage.input_tokens
    output_tokens = message.usage.output_tokens
    total_tokens = input_tokens + output_tokens

    # Return normalized response
    ai_response = AIResponse(
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

    # Automatic token tracking
    if token_tracker:
        token_tracker.track(request.step_name, ai_response)

    return ai_response
