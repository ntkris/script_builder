"""Structured data extraction from text using Gemini Flash models"""

from typing import Type, TypeVar, List, Union, Optional
from pydantic import BaseModel, ValidationError
import json

from ..ai.base import AIRequest, Provider, GeminiModel
from ..ai.gemini_provider import call_gemini
from ..step_logger import StepLogger

T = TypeVar('T', bound=BaseModel)


def extract(
    text: str,
    schema: Type[T],
    prompt: str = "Extract structured data from the following text:",
    logger: Optional[StepLogger] = None,
    return_list: bool = False,
    step_name: str = "Structured Extraction",
    model: GeminiModel = GeminiModel.GEMINI_2_5_FLASH
) -> Union[T, List[T]]:
    """
    Extract structured data from text using Gemini Flash models with native JSON mode.

    Args:
        text: The text to extract data from
        schema: Pydantic model defining the structure to extract
        prompt: Instructions for extraction (prepended to text)
        logger: Optional StepLogger for token tracking
        return_list: If True, expects list of schema objects
        step_name: Name for token tracking
        model: Gemini model to use (defaults to GEMINI_2_5_FLASH)

    Returns:
        Parsed Pydantic model instance(s) validated against schema

    Raises:
        ValidationError: If extracted data doesn't match schema
        ValueError: If JSON parsing fails

    Example:
        ```python
        from pydantic import BaseModel
        from utils import extract
        from utils.step_logger import StepLogger

        class Person(BaseModel):
            name: str
            age: int
            occupation: str

        logger = StepLogger("my_script")
        text = "John Smith is a 35-year-old software engineer..."

        person = extract(
            text=text,
            schema=Person,
            logger=logger
        )

        print(person.name)  # "John Smith"
        print(person.age)   # 35
        ```
    """
    # Determine schema type (single object or list)
    response_schema = list[schema] if return_list else schema

    # Build the extraction prompt
    full_prompt = f"{prompt}\n\n{text}"

    # Create request with json_mode enabled
    request = AIRequest(
        messages=[{"role": "user", "content": full_prompt}],
        model=model,
        provider=Provider.GOOGLE,
        max_tokens=10000,
        json_mode=True,
        response_schema=response_schema,
        step_name=step_name
    )

    # Call Gemini
    response = call_gemini(request, logger)

    # Parse JSON response
    try:
        data = json.loads(response.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response.content}")

    # Validate against Pydantic schema
    try:
        if return_list:
            # Validate list of objects
            if not isinstance(data, list):
                raise ValidationError(f"Expected list, got {type(data)}")
            return [schema.model_validate(item) for item in data]
        else:
            # Validate single object
            return schema.model_validate(data)
    except ValidationError as e:
        raise ValidationError(f"Failed to validate extracted data against schema: {e}\nData: {data}")
