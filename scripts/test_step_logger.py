#!/usr/bin/env python3
"""Test script for StepLogger with both LLM providers and structured extraction"""

from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from utils import call_anthropic, AIRequest, AnthropicModel, extract
from utils.step_logger import StepLogger

load_dotenv()

# Pydantic schema for structured extraction
class Product(BaseModel):
    """Product information"""
    name: str
    category: str
    price: float
    features: List[str] = Field(default_factory=list)
    rating: float


def main():
    print("\n" + "="*60)
    print("🧪 STEP LOGGER TEST")
    print("="*60 + "\n")

    logger = StepLogger("test_step_logger")

    # Step 1: Generate product descriptions using Anthropic
    logger.step("Generate Product Descriptions", inputs={"count": 3})

    request = AIRequest(
        messages=[{
            "role": "user",
            "content": """Generate 3 fictional product descriptions for an online store.
            Include: product name, category, price, 3-5 key features, and rating (1-5 stars).

            Make them realistic and varied (e.g., electronics, home goods, clothing)."""
        }],
        model=AnthropicModel.CLAUDE_3_HAIKU,
        max_tokens=500,
        step_name="Generate Products (Anthropic)"
    )

    response = call_anthropic(request, logger)
    product_text = response.content

    print(f"\n📝 Generated text ({len(product_text)} chars)")
    print(f"Preview: {product_text[:200]}...\n")

    logger.output({
        "text": product_text,
        "text_length": len(product_text)
    })

    # Step 2: Extract structured data using Gemini
    logger.step("Extract Structured Data", inputs={"schema": "Product"})

    try:
        products = extract(
            text=product_text,
            schema=Product,
            prompt="Extract all product information from this text into a structured format:",
            logger=logger,
            return_list=True,
            step_name="Extract Products (Gemini)"
        )

        print(f"✅ Extracted {len(products)} products:\n")
        for i, product in enumerate(products, 1):
            logger.update({
                "extracted": i,
                "product_name": product.name
            })
            print(f"  {i}. {product.name}")
            print(f"     Category: {product.category}")
            print(f"     Price: ${product.price:.2f}")
            print(f"     Rating: {product.rating}⭐")
            print(f"     Features: {', '.join(product.features)}\n")

        logger.output({
            "products": [p.model_dump() for p in products],
            "products_extracted": len(products),
            "total_value": sum(p.price for p in products),
            "avg_rating": sum(p.rating for p in products) / len(products)
        })

    except Exception as e:
        print(f"⚠️ Extraction failed: {e}")
        logger.fail(e)

    # Step 3: Simple processing with progress updates
    logger.step("Process Products", inputs={"operation": "validate"})
    for i in range(5):
        import time
        time.sleep(0.2)  # Simulate work
        logger.update({
            "progress": f"{i+1}/5",
            "percentage": (i+1) * 20
        })
    logger.output({"validation_complete": True})

    # Finalize and show summary
    print("\n" + "="*60)
    log_path = logger.finalize()
    print("="*60)
    print(f"\n✅ Test complete! Log saved to:")
    print(f"   {log_path}\n")

if __name__ == "__main__":
    main()
