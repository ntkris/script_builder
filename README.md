# Script Builder

Lightweight prototyping environment for AI-powered scripts. Built for rapid experimentation with a focus on:
- **Minimal setup**: Just `uv` for package management
- **Reusable utilities**: Common LLM and I/O operations abstracted
- **Automatic step logging**: StepLogger tracks steps, token usage, and progress
- **Multi-provider support**: Universal interface for Anthropic, Gemini, and more

## Quick Start

1. **Setup environment:**
   ```bash
   # Create .env file with API keys
   echo "ANTHROPIC_API_KEY=your_key_here" > .env
   echo "GEMINI_API_KEY=your_key_here" >> .env
   ```

2. **Install project:**
   ```bash
   uv sync  # Installs dependencies and makes utils/ importable
   ```

3. **Run scripts:**
   ```bash
   uv run scripts/your_script.py
   ```

## Directory Structure

```
script_builder/
├── scripts/               # Your scripts go here
├── utils/                 # Reusable utilities
│   ├── ai/               # LLM provider interface (multi-provider)
│   │   ├── base.py       # Base models and enums
│   │   ├── anthropic_provider.py  # call_anthropic()
│   │   ├── gemini_provider.py     # call_gemini()
│   │   └── README.md     # Detailed AI usage docs
│   ├── tools/            # Common tools (search, extraction)
│   ├── io.py             # save_json, load_json helpers
│   └── step_logger.py    # StepLogger for step tracking + token usage
├── inputs/                # Input files (videos, images, data)
├── outputs/               # Final results only
├── cache/                 # Interim processing steps (JSON logs, step logs)
├── .env                   # API keys (gitignored)
└── pyproject.toml         # Project config (makes utils/ importable)
```

## Core Principles

1. **Scripts in `scripts/`**: All executable scripts go in the scripts directory
2. **Imports at the top**: Always put imports at the top of files, never inline
3. **Use StepLogger**: Track steps, progress, and token usage automatically
4. **Interim steps → `cache/`**: Save processing logs, step logs, intermediate results
5. **Final results → `outputs/`**: Only final artifacts (videos, reports, etc.)
6. **Reusable code → `utils/`**: Extract common patterns, avoid duplication

## Usage Examples

### Basic Script Structure

**IMPORTANT: Always put imports at the top. Never use inline imports.**

```python
#!/usr/bin/env python3
"""Script description"""

from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from utils import call_anthropic, AIRequest, AnthropicModel, save_json
from utils.step_logger import StepLogger

def main():
    # Initialize step logger
    logger = StepLogger("my_script")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Process data
    logger.step("Analyze Data", inputs={"source": "data.csv"})

    request = AIRequest(
        messages=[{"role": "user", "content": "Analyze this data..."}],
        model=AnthropicModel.CLAUDE_SONNET_4,
        max_tokens=1000,
        step_name="Data Analysis"
    )
    response = call_anthropic(request, logger)  # Automatic token tracking

    logger.output({"analysis": response.content})

    # Step 2: Save results
    logger.step("Save Results")
    save_json(
        {"analysis": response.content, "timestamp": timestamp},
        f"analysis_{timestamp}.json",
        output_dir="outputs",
        description="Final Analysis"
    )
    logger.output({"saved": True})

    # Finalize - prints summary and saves step log to cache/
    logger.finalize()

if __name__ == "__main__":
    main()
```

### Step Logging

```python
from utils.step_logger import StepLogger

logger = StepLogger("my_script")

# Start a step
logger.step("Process Data", inputs={"count": 100})

# Log progress during loops
for i in range(100):
    # ... do work ...
    logger.update({"progress": f"{i+1}/100"})

# Mark step complete with outputs
logger.output({"processed": 100, "errors": 0})

# At the end of script
logger.finalize()  # Prints summary, saves to cache/
```

### AI/LLM Calls

```python
from utils import call_anthropic, call_gemini, AIRequest
from utils import AnthropicModel, GeminiModel, Provider
from utils.step_logger import StepLogger

logger = StepLogger("my_script")

# Anthropic Claude
request = AIRequest(
    messages=[{"role": "user", "content": "Hello!"}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    max_tokens=100,
    step_name="Greeting"
)
response = call_anthropic(request, logger)  # Token tracking automatic
print(response.content)

# Google Gemini
request = AIRequest(
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    model=GeminiModel.GEMINI_2_5_FLASH,
    provider=Provider.GOOGLE,
    max_tokens=500,
    system="You are a physics professor. Be concise.",
    temperature=0.7,
    step_name="Physics Explanation"
)
response = call_gemini(request, logger)

# API keys automatically loaded from env vars
```

### Structured Extraction

```python
from utils import extract
from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    name: str
    price: float
    category: str

# Extract structured data using Gemini
products = extract(
    text="Apple iPhone costs $999 in electronics...",
    schema=Product,
    logger=logger,
    return_list=True,
    step_name="Extract Products"
)

for product in products:
    print(f"{product.name}: ${product.price}")
```

### Saving/Loading Data

```python
from utils import save_json, load_json

# Save to cache (interim steps)
save_json(
    {"step": 1, "result": "..."},
    "step1.json",
    output_dir="cache",
    description="Step 1"
)

# Save to outputs (final results)
save_json(
    {"final": "result"},
    "final.json",
    output_dir="outputs",
    description="Final Result"
)

# Load
data = load_json("cache/step1.json")
```


## Available Models

### Anthropic
- `AnthropicModel.CLAUDE_OPUS_4` - Best quality
- `AnthropicModel.CLAUDE_SONNET_4` - Balanced (recommended)
- `AnthropicModel.CLAUDE_SONNET_3_5` - Previous generation
- `AnthropicModel.CLAUDE_3_HAIKU` - Fast, cheap

### Google Gemini
- `GeminiModel.GEMINI_2_5_PRO` - Latest, best quality
- `GeminiModel.GEMINI_2_5_FLASH` - Latest, fast (recommended)
- `GeminiModel.GEMINI_2_5_FLASH_LITE` - Latest, very fast
- `GeminiModel.GEMINI_2_0_FLASH` - Previous generation

### OpenAI (coming soon)
- `OpenAIModel.GPT_4O`
- `OpenAIModel.GPT_4O_MINI`

## Key Patterns

- **Always import at top**: Never use inline imports
- **Use StepLogger**: Track all steps and LLM calls
- **Pydantic models**: For all structured data
- **Error handling**: Use `logger.fail(exception)` for failures
- **Progress updates**: Use `logger.update()` in loops
- **Cache vs Outputs**: Interim data → cache/, final results → outputs/

## Advanced Features

See `utils/ai/README.md` for:
- Tool/function calling
- Structured output (JSON mode)
- Multi-turn conversations
- Vision API usage
