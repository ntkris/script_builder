"""OpenAI provider implementation"""

from typing import Optional, Any, Dict
import os
import json
import copy
from openai import OpenAI
from .base import AIRequest, AIResponse, Provider, ToolCall
from ..step_logger import StepLogger


def _clean_schema_for_openai(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean Pydantic schema for OpenAI Structured Outputs compatibility.

    OpenAI strict mode requires:
    - additionalProperties: false at all object levels
    - required array must include ALL properties (not just non-optional ones)
    """
    # Deep copy to avoid modifying original
    schema = copy.deepcopy(schema)

    # Add additionalProperties: false to root
    if schema.get("type") == "object":
        schema["additionalProperties"] = False

        # OpenAI strict mode: ALL properties must be in required array
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())

    # Process properties recursively
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            if isinstance(prop_schema, dict):
                if prop_schema.get("type") == "object":
                    prop_schema["additionalProperties"] = False
                    # Ensure all nested properties are required
                    if "properties" in prop_schema:
                        prop_schema["required"] = list(prop_schema["properties"].keys())
                    # Recursively clean nested objects
                    schema["properties"][prop_name] = _clean_schema_for_openai(prop_schema)
                elif prop_schema.get("type") == "array":
                    # Handle array items
                    if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                        if prop_schema["items"].get("type") == "object":
                            prop_schema["items"]["additionalProperties"] = False
                            # Ensure all nested properties are required
                            if "properties" in prop_schema["items"]:
                                prop_schema["items"]["required"] = list(prop_schema["items"]["properties"].keys())
                        # Recursively clean nested objects
                        prop_schema["items"] = _clean_schema_for_openai(prop_schema["items"])

    # Handle definitions ($defs)
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            if isinstance(def_schema, dict):
                schema["$defs"][def_name] = _clean_schema_for_openai(def_schema)

    return schema


def call_openai(
    request: AIRequest,
    logger: Optional[StepLogger] = None,
    api_key: Optional[str] = None
) -> AIResponse:
    """
    Make a call to OpenAI API with automatic token tracking.

    Args:
        request: AIRequest with parameters
        logger: Optional StepLogger for automatic token tracking
        api_key: Optional API key (defaults to OPENAI_API_KEY env var)

    Returns:
        AIResponse with normalized response data

    Example:
        from utils import call_openai, AIRequest, OpenAIModel
        from utils.step_logger import StepLogger

        logger = StepLogger("my_script")

        request = AIRequest(
            messages=[{"role": "user", "content": "Hello!"}],
            model=OpenAIModel.GPT_5_MINI,
            max_tokens=1000,
            step_name="Greeting"
        )
        response = call_openai(request, logger)
        print(response.content)
    """
    # Initialize client with API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables or provided as argument")

    client = OpenAI(api_key=api_key)

    # Handle structured output with JSON mode
    messages = request.messages.copy()
    system_prompt = request.system

    # Convert messages to OpenAI format
    openai_messages = []

    # Add system message if provided
    if system_prompt:
        openai_messages.append({"role": "system", "content": system_prompt})

    # Convert user/assistant messages
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Handle multipart content (text + images)
        if isinstance(content, list):
            openai_content = []
            for part in content:
                if part.get("type") == "text":
                    openai_content.append({
                        "type": "text",
                        "text": part["text"]
                    })
                elif part.get("type") == "image":
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        # OpenAI vision format
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
                            }
                        })
            openai_messages.append({"role": role, "content": openai_content})
        else:
            # Simple text content
            openai_messages.append({"role": role, "content": content})

    # Build API call parameters
    params = {
        "model": request.model,
        "messages": openai_messages,
        "max_completion_tokens": request.max_tokens,
    }

    # Add GPT-5 specific parameters
    if request.reasoning_effort:
        params["reasoning_effort"] = request.reasoning_effort

    if request.verbosity:
        params["verbosity"] = request.verbosity

    # Handle JSON mode with Structured Outputs
    if request.json_mode:
        if request.response_schema:
            # Use Structured Outputs (new, recommended) - guarantees schema adherence
            schema_dict = request.response_schema.model_json_schema()

            # Clean schema for OpenAI (add additionalProperties: false)
            schema_dict = _clean_schema_for_openai(schema_dict)

            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema.__name__,
                    "schema": schema_dict,
                    "strict": True  # Enable strict mode for guaranteed adherence
                }
            }
        else:
            # Fallback to basic JSON mode if no schema provided
            params["response_format"] = {"type": "json_object"}

            # Add instruction to prompt for basic JSON mode
            last_message = openai_messages[-1]
            if isinstance(last_message["content"], list):
                for part in last_message["content"]:
                    if part.get("type") == "text":
                        part["text"] += "\n\nPlease respond with valid JSON only."
                        break
            else:
                last_message["content"] += "\n\nPlease respond with valid JSON only."

    # Handle tools if provided
    if request.tools:
        params["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": True  # Enable structured outputs for function calling
                }
            }
            for tool in request.tools
        ]

        # Disable parallel function calls (required for structured outputs)
        params["parallel_tool_calls"] = False

        if request.tool_choice:
            # Map generic tool_choice to OpenAI format
            if request.tool_choice == "required":
                params["tool_choice"] = "required"
            elif request.tool_choice == "auto":
                params["tool_choice"] = "auto"
            else:
                # Specific tool name
                params["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}

    # Make the API call
    response = client.chat.completions.create(**params)

    # Extract response content and tool calls
    message = response.choices[0].message
    content = message.content or ""
    tool_calls = []

    if message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments)
            ))

    # Calculate token usage
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
    total_tokens = response.usage.total_tokens if response.usage else input_tokens + output_tokens

    # Get finish reason
    finish_reason = response.choices[0].finish_reason if response.choices else None

    # Return normalized response
    ai_response = AIResponse(
        content=content.strip(),
        model=response.model,
        provider=Provider.OPENAI,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        tool_calls=tool_calls if tool_calls else None,
        finish_reason=finish_reason,
        raw_response=response
    )

    # Automatic token tracking with StepLogger
    if logger:
        logger.track(request.step_name, ai_response)

    return ai_response
