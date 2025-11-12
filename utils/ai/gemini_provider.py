"""Google Gemini provider implementation"""

from typing import Optional
import os
import base64
from google import genai
from google.genai import types
from .base import AIRequest, AIResponse, Provider, ToolCall
from ..step_logger import StepLogger


def call_gemini(
    request: AIRequest,
    logger: Optional[StepLogger] = None,
    api_key: Optional[str] = None
) -> AIResponse:
    """
    Make a call to Google Gemini API with automatic token tracking.

    Args:
        request: AIRequest with parameters
        logger: Optional StepLogger for automatic token tracking
        api_key: Optional API key (defaults to GEMINI_API_KEY env var)

    Returns:
        AIResponse with normalized response data

    Example:
        from utils import call_gemini, AIRequest, GeminiModel, Provider
        from utils.step_logger import StepLogger

        logger = StepLogger("my_script")

        request = AIRequest(
            messages=[{"role": "user", "content": "Hello!"}],
            model=GeminiModel.GEMINI_2_5_FLASH,
            max_tokens=1000,
            step_name="Greeting",
            provider=Provider.GOOGLE
        )
        response = call_gemini(request, logger)
        print(response.content)
    """
    # Initialize client with API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or provided as argument")

    client = genai.Client(api_key=api_key)

    # Build generation config
    config_params = {
        "max_output_tokens": request.max_tokens,
    }

    if request.temperature is not None:
        config_params["temperature"] = request.temperature

    if request.system is not None:
        config_params["system_instruction"] = request.system

    # Handle structured output (json_mode)
    if request.json_mode and request.response_schema:
        config_params["response_mime_type"] = "application/json"
        # Gemini accepts list[Type] syntax directly
        config_params["response_schema"] = request.response_schema

    # Handle tools if provided
    if request.tools:
        # Convert to Gemini function declarations
        function_declarations = []
        for tool in request.tools:
            function_declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            })
        tool = types.Tool(function_declarations=function_declarations)
        config_params["tools"] = [tool]

    config = types.GenerateContentConfig(**config_params)

    # Convert messages to Gemini format
    # Gemini uses "user" and "model" roles
    gemini_contents = []
    for msg in request.messages:
        role = msg["role"]
        content = msg["content"]

        # Map roles: user->user, assistant->model
        if role == "assistant":
            role = "model"

        # Handle multimodal content (list of parts with text and/or images)
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append(types.Part(text=part["text"]))
                elif part.get("type") == "image":
                    source = part.get("source", {})
                    if source.get("type") == "base64":
                        # Gemini expects raw bytes, not base64 string
                        image_bytes = base64.b64decode(source["data"])
                        parts.append(
                            types.Part(
                                inline_data=types.Blob(
                                    mime_type=source.get("media_type", "image/png"),
                                    data=image_bytes
                                )
                            )
                        )
            gemini_contents.append(
                types.Content(role=role, parts=parts)
            )
        else:
            # Simple text content
            gemini_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=content)]
                )
            )

    # Make the API call
    # For single user message with simple text, pass as string
    # For multi-turn or multimodal, pass as list of Contents
    if len(gemini_contents) == 1 and gemini_contents[0].role == "user" and isinstance(request.messages[0]["content"], str):
        contents = request.messages[0]["content"]
    else:
        contents = gemini_contents

    response = client.models.generate_content(
        model=request.model,
        contents=contents,
        config=config
    )

    # Extract response content and tool calls
    content = ""
    tool_calls = []

    if response.candidates:
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            # Empty response - get finish reason for debugging
            finish_reason = str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else "unknown"
            safety_ratings = []
            if hasattr(candidate, 'safety_ratings'):
                safety_ratings = [f"{r.category}: {r.probability}" for r in candidate.safety_ratings]

            error_msg = f"No response from Gemini - content is empty (finish_reason: {finish_reason}"
            if safety_ratings:
                error_msg += f", safety: {', '.join(safety_ratings)}"
            error_msg += ")"
            raise ValueError(error_msg)

        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                content += part.text
            elif hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=fc.name,  # Gemini doesn't provide IDs, use name
                    name=fc.name,
                    arguments=dict(fc.args)
                ))
    else:
        # No candidates at all
        raise ValueError("No response from Gemini - no candidates returned")

    # Calculate token usage
    input_tokens = response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else 0
    output_tokens = response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else 0
    total_tokens = input_tokens + output_tokens

    # Get finish reason
    finish_reason = None
    if response.candidates:
        finish_reason = str(response.candidates[0].finish_reason)

    # Return normalized response
    ai_response = AIResponse(
        content=content.strip(),
        model=request.model,
        provider=Provider.GOOGLE,
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
