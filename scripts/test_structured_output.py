"""Test structured output (JSON mode) for Anthropic and Gemini providers"""

import sys
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    call_anthropic,
    call_gemini,
    AIRequest,
    AnthropicModel,
    GeminiModel,
    Provider,
    TokenTracker,
    save_json
)

# Initialize
tracker = TokenTracker()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# Define test Pydantic models
class Person(BaseModel):
    """A person with basic information"""
    name: str = Field(description="Full name of the person")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")
    skills: List[str] = Field(default_factory=list, description="List of skills")


class Book(BaseModel):
    """A book with metadata"""
    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    year: int = Field(description="Publication year")
    genre: str = Field(description="Book genre")
    rating: float = Field(description="Rating out of 5.0", ge=0.0, le=5.0)


def test_person_extraction():
    """Test extracting person data with both providers"""
    print("🔍 Test 1: Person Extraction")
    print("=" * 60)

    test_text = """
    Dr. Sarah Johnson is a 42-year-old data scientist working at a leading tech company.
    She specializes in machine learning, natural language processing, and has extensive
    experience with Python, TensorFlow, and cloud computing platforms.
    """

    # Test with Gemini
    print("\n🤖 Testing with Gemini 2.0 Flash...")
    gemini_request = AIRequest(
        messages=[{
            "role": "user",
            "content": f"Extract the person's information from the text:\n\n{test_text}"
        }],
        model=GeminiModel.GEMINI_2_0_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=1024,
        json_mode=True,
        response_schema=Person,
        step_name="Gemini Person Extraction"
    )
    gemini_response = call_gemini(gemini_request, tracker)
    gemini_data = json.loads(gemini_response.content)
    gemini_person = Person.model_validate(gemini_data)

    print(f"✅ Gemini Result:")
    print(f"   Name: {gemini_person.name}")
    print(f"   Age: {gemini_person.age}")
    print(f"   Occupation: {gemini_person.occupation}")
    print(f"   Skills: {', '.join(gemini_person.skills)}")

    # Test with Anthropic
    print("\n🤖 Testing with Claude Sonnet 4...")
    anthropic_request = AIRequest(
        messages=[{
            "role": "user",
            "content": f"Extract the person's information from the text:\n\n{test_text}"
        }],
        model=AnthropicModel.CLAUDE_SONNET_4,
        provider=Provider.ANTHROPIC,
        max_tokens=1024,
        json_mode=True,
        response_schema=Person,
        step_name="Anthropic Person Extraction"
    )
    anthropic_response = call_anthropic(anthropic_request, tracker)
    anthropic_data = json.loads(anthropic_response.content)
    anthropic_person = Person.model_validate(anthropic_data)

    print(f"✅ Anthropic Result:")
    print(f"   Name: {anthropic_person.name}")
    print(f"   Age: {anthropic_person.age}")
    print(f"   Occupation: {anthropic_person.occupation}")
    print(f"   Skills: {', '.join(anthropic_person.skills)}")

    return {
        "gemini": gemini_person.model_dump(),
        "anthropic": anthropic_person.model_dump()
    }


def test_book_extraction():
    """Test extracting book data with both providers"""
    print("\n\n🔍 Test 2: Book Extraction")
    print("=" * 60)

    test_text = """
    "The Midnight Library" by Matt Haig was published in 2020. This philosophical fiction
    novel explores themes of regret, choice, and infinite possibilities. It has received
    widespread acclaim with an average rating of 4.2 out of 5 stars from readers worldwide.
    """

    # Test with Gemini
    print("\n🤖 Testing with Gemini 2.0 Flash...")
    gemini_request = AIRequest(
        messages=[{
            "role": "user",
            "content": f"Extract the book's metadata from the text:\n\n{test_text}"
        }],
        model=GeminiModel.GEMINI_2_0_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=1024,
        json_mode=True,
        response_schema=Book,
        step_name="Gemini Book Extraction"
    )
    gemini_response = call_gemini(gemini_request, tracker)
    gemini_data = json.loads(gemini_response.content)
    gemini_book = Book.model_validate(gemini_data)

    print(f"✅ Gemini Result:")
    print(f"   Title: {gemini_book.title}")
    print(f"   Author: {gemini_book.author}")
    print(f"   Year: {gemini_book.year}")
    print(f"   Genre: {gemini_book.genre}")
    print(f"   Rating: {gemini_book.rating}/5.0")

    # Test with Anthropic
    print("\n🤖 Testing with Claude Sonnet 4...")
    anthropic_request = AIRequest(
        messages=[{
            "role": "user",
            "content": f"Extract the book's metadata from the text:\n\n{test_text}"
        }],
        model=AnthropicModel.CLAUDE_SONNET_4,
        provider=Provider.ANTHROPIC,
        max_tokens=1024,
        json_mode=True,
        response_schema=Book,
        step_name="Anthropic Book Extraction"
    )
    anthropic_response = call_anthropic(anthropic_request, tracker)
    anthropic_data = json.loads(anthropic_response.content)
    anthropic_book = Book.model_validate(anthropic_data)

    print(f"✅ Anthropic Result:")
    print(f"   Title: {anthropic_book.title}")
    print(f"   Author: {anthropic_book.author}")
    print(f"   Year: {anthropic_book.year}")
    print(f"   Genre: {anthropic_book.genre}")
    print(f"   Rating: {anthropic_book.rating}/5.0")

    return {
        "gemini": gemini_book.model_dump(),
        "anthropic": anthropic_book.model_dump()
    }


def test_list_extraction():
    """Test extracting multiple items with both providers"""
    print("\n\n🔍 Test 3: List Extraction (Multiple Books)")
    print("=" * 60)

    test_text = """
    Here are three notable science fiction books:

    1. "Dune" by Frank Herbert, published in 1965, is an epic science fiction masterpiece
       rated 4.3/5.

    2. "The Three-Body Problem" by Liu Cixin came out in 2008. This hard science fiction
       novel has a rating of 4.0/5.

    3. "Project Hail Mary" by Andy Weir was released in 2021. This science fiction thriller
       has been rated 4.5/5 by readers.
    """

    # Test with Gemini
    print("\n🤖 Testing with Gemini 2.0 Flash...")
    gemini_request = AIRequest(
        messages=[{
            "role": "user",
            "content": f"Extract all books mentioned in the text as a list:\n\n{test_text}"
        }],
        model=GeminiModel.GEMINI_2_0_FLASH,
        provider=Provider.GOOGLE,
        max_tokens=2048,
        json_mode=True,
        response_schema=list[Book],  # Use lowercase list for Gemini
        step_name="Gemini List Extraction"
    )
    gemini_response = call_gemini(gemini_request, tracker)
    gemini_data = json.loads(gemini_response.content)
    gemini_books = [Book.model_validate(item) for item in gemini_data]

    print(f"✅ Gemini Result: {len(gemini_books)} books extracted")
    for i, book in enumerate(gemini_books, 1):
        print(f"   {i}. {book.title} by {book.author} ({book.year}) - {book.rating}/5")

    # Test with Anthropic
    print("\n🤖 Testing with Claude Sonnet 4...")
    anthropic_request = AIRequest(
        messages=[{
            "role": "user",
            "content": f"Extract all books mentioned in the text as a list:\n\n{test_text}"
        }],
        model=AnthropicModel.CLAUDE_SONNET_4,
        provider=Provider.ANTHROPIC,
        max_tokens=2048,
        json_mode=True,
        response_schema=List[Book],
        step_name="Anthropic List Extraction"
    )
    anthropic_response = call_anthropic(anthropic_request, tracker)
    anthropic_data = json.loads(anthropic_response.content)
    anthropic_books = [Book.model_validate(item) for item in anthropic_data]

    print(f"✅ Anthropic Result: {len(anthropic_books)} books extracted")
    for i, book in enumerate(anthropic_books, 1):
        print(f"   {i}. {book.title} by {book.author} ({book.year}) - {book.rating}/5")

    return {
        "gemini": [book.model_dump() for book in gemini_books],
        "anthropic": [book.model_dump() for book in anthropic_books]
    }


def main():
    print("🚀 Starting Structured Output Tests")
    print("Testing JSON mode for Anthropic (XML) and Gemini (native JSON)")
    print("=" * 60)

    results = {}

    try:
        # Run tests
        results["test_1_person"] = test_person_extraction()
        results["test_2_book"] = test_book_extraction()
        results["test_3_list"] = test_list_extraction()

        # Summary
        print("\n\n📊 Test Summary")
        print("=" * 60)
        print("✅ All tests passed!")
        print(f"✅ Both providers successfully extracted structured data")
        print(f"✅ Pydantic validation successful for all extractions")

        # Save results to cache
        output_file = f"structured_output_test_{timestamp}.json"
        save_json(
            results,
            output_file,
            output_dir="cache",
            description="Structured Output Test Results"
        )

        # Save token summary
        print("\n💰 Token Usage Summary")
        print("=" * 60)
        tracker.save_summary("structured_output_test", output_dir="outputs")

        total_usage = tracker.get_total_usage()
        print(f"\n📊 Total tokens used: {total_usage.total_tokens}")
        print(f"   Input: {total_usage.input_tokens}")
        print(f"   Output: {total_usage.output_tokens}")

        print("\n✅ Tests complete!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
