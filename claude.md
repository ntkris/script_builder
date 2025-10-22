# Claude Context: Script Builder Repository

This document provides context for Claude when writing scripts in this repository.

## Repository Purpose

This is a **lightweight prototyping environment** for AI-powered scripts. The user wants to:
- Rapidly test ideas and iterate
- Keep things simple and minimal
- Avoid boilerplate and repetitive code
- Automatically track LLM usage and costs
- Log incremental steps for debugging

**Philosophy:** Lightweight, simple, reusable, well-logged.

## Directory Structure

```
script_builder/
├── scripts/               # All executable scripts (user's code)
├── utils/                 # Reusable utilities (shared across scripts)
│   ├── ai/               # LLM provider interface
│   │   ├── base.py       # Pydantic models, enums, base interfaces
│   │   ├── anthropic_provider.py  # call_anthropic() function
│   │   ├── gemini_provider.py     # call_gemini() function
│   │   └── README.md     # Detailed examples
│   ├── tools/            # Common tools (search, extraction)
│   ├── io.py             # save_json(), load_json()
│   └── step_logger.py    # StepLogger class for step tracking + token usage
├── inputs/                # Input files (videos, images, text, data)
├── outputs/               # ONLY final results (videos, reports, etc.)
├── cache/                 # Interim processing steps (JSON logs with timestamps)
├── .env                   # API keys (ANTHROPIC_API_KEY, GEMINI_API_KEY)
└── pyproject.toml         # Project configuration (makes utils/ importable)
```

## Key Patterns

### 1. File Organization
- **Scripts** go in `scripts/` directory
- **Step logs** (with token tracking) → automatically saved to `cache/` by StepLogger
- **Final outputs** (videos, final reports, completed deliverables) → `outputs/`
- Always include timestamps in filenames: `analysis_20250622_143022.json`

**Important**:
- ✅ **StepLogger is your single source of truth** - log rich data in `inputs` and `outputs`
- ✅ Log **actual data and decisions**, not just summary stats (for debugging)
- ✅ Save ONLY final, completed deliverables to `outputs/`
- ❌ Don't create separate cache files unless data is too large for StepLogger or needs reuse
- ❌ Never save step logs to `outputs/` (StepLogger auto-saves to `cache/`)

### 2. Utility Scripts

**Clear cache:**
```bash
uv run scripts/clear_cache.py
```

This clears all files from the `cache/` directory including step logs and resume state. Use this when you want to start completely fresh or clear old resume state files.

### 3. Utils Available

#### Step Logging with Token Tracking
```python
from utils import call_anthropic, AIRequest, AnthropicModel
from utils.step_logger import StepLogger

logger = StepLogger("script_name")

# Start a step - log actual inputs, not just counts
logger.step("Evaluate Candidates", inputs={
    "candidates": [{"id": 1, "name": "..."}, {"id": 2, "name": "..."}],
    "criteria": {"min_score": 0.8}
})

request = AIRequest(
    messages=[{"role": "user", "content": "..."}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    max_tokens=1000,
    step_name="Evaluate Candidate 1"
)
response = call_anthropic(request, logger)

# Log actual data and decisions, not just summary stats
logger.output({
    "evaluations": [
        {
            "candidate_id": 1,
            "score": 0.85,
            "decision": "accepted",
            "reason": "Strong technical background",
            "extraction": {...}  # Full extraction object
        },
        {
            "candidate_id": 2,
            "score": 0.65,
            "decision": "rejected",
            "reason": "Insufficient experience",
            "extraction": {...}
        }
    ],
    "summary": {"accepted": 1, "rejected": 1}  # Stats are OK as supplement
})

# At the end of script
logger.finalize()  # Prints summary and saves to cache/
```

**Key principles:**
- **Log rich data**: Include actual decisions, reasons, and extractions in `outputs`
- **Log real inputs**: Not just counts - include actual data structures for debugging
- **Single source of truth**: Everything you need to debug should be in the StepLogger JSON
- `StepLogger` handles both step tracking AND token tracking automatically
- API keys are handled internally (from `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` env vars)
- Don't import or initialize provider clients directly
- Always pass `logger` to LLM calls for automatic token tracking
- Use descriptive `step_name` for logging clarity
- **Use Pydantic models** for all structured data (not TypedDict or dataclasses)
- All step logs save to `cache/` directory automatically with timestamps

#### Resumable Scripts (for long-running operations)

For scripts that process many items (e.g., web scraping, batch processing), use resumable mode to recover from crashes:

```python
from utils.step_logger import StepLogger

# Enable resumable mode
logger = StepLogger("script_name", resumable=True)

# Step 1: Generate or load data
logger.step("Load Data")
if logger.should_run_step():
    data = expensive_operation()
    logger.output({"data": data})
else:
    data = logger.get_cached_output("data")
    print("✅ Loaded from cache")

# Step 2: Process items with resume support
logger.step("Process Items")
if logger.should_run_step():
    results = []
else:
    # Resume from previous run
    results = logger.get_cached_output("results") or []

for item in items:
    # Skip already processed items
    if logger.is_item_completed(item.id):
        print(f"⏭️  Skipping {item.id}")
        continue

    result = process_item(item)
    results.append(result)

    # Mark complete and save immediately
    logger.mark_item_complete(item.id)

logger.output({"results": results})

# Step 3: Save final output
logger.step("Save Results")
if logger.should_run_step():
    save_results(results)
    logger.output({"saved": True})

logger.finalize()  # Clears resume state on success
```

**Resumable CLI patterns:**
```python
parser.add_argument("--fresh", action="store_true",
                   help="Start fresh, ignore resume state")

# In main():
logger = StepLogger("script_name", resumable=True)

if args.fresh and logger.resume_state_file.exists():
    logger.resume_state_file.unlink()
    print("🗑️  Cleared resume state, starting fresh")
```

**How it works:**
- Creates `cache/resume_state_{script_name}.json` tracking completed steps and items
- `should_run_step()` returns False if step already completed
- `get_cached_output(key)` loads data from previous run
- `is_item_completed(item_id)` checks if item was processed
- `mark_item_complete(item_id)` saves state immediately (crash-safe)
- Resume state auto-deleted on successful `finalize()`
- Use `--fresh` flag to force clean run
- Use `uv run scripts/clear_cache.py` to manually clear all cache including resume state

**When to use resumable mode:**
- Processing 50+ items that take time (API calls, web scraping)
- Operations that might timeout or crash
- Expensive operations you don't want to repeat

**When NOT to use resumable mode:**
- Simple scripts with < 10 items
- Fast operations (< 1 minute total)
- Scripts that need fresh data every run

#### JSON I/O (Optional)
```python
from utils import save_json, load_json

# Prefer logging to StepLogger instead of separate cache files
# Only save separate files if:
# - Data is too large for StepLogger (e.g., full page text, images)
# - Data needs to be reused by another script/process
# - You're creating a final deliverable

# Save to outputs (final results only)
save_json(final, "result_20250622.json", output_dir="outputs", description="Final Report")

# Load previously saved data
data = load_json("cache/large_dataset.json")
```

#### Structured Extraction (Gemini)
```python
from utils import extract
from pydantic import BaseModel
from typing import List

class Product(BaseModel):
    name: str
    price: float
    features: List[str]

# Extract structured data from text
products = extract(
    text=text,
    schema=Product,
    logger=logger,
    return_list=True,
    step_name="Extract Products"
)
# Automatically uses Gemini Flash with native JSON mode
```

### 3. Common Script Structure

**IMPORTANT: Always put imports at the top of the file. Never use inline imports.**

```python
#!/usr/bin/env python3
"""Script description"""

from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

from utils import call_anthropic, AIRequest, AnthropicModel, save_json
from utils.step_logger import StepLogger

# Define Pydantic models for structured data
class Result(BaseModel):
    field: str
    score: float

def main():
    # Initialize step logger
    logger = StepLogger("script_name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Load inputs
    logger.step("Load Inputs", inputs={"source": "..."})
    # ... do work ...
    logger.output({"loaded": 10})

    # Step 2: Process with LLM
    logger.step("Process Data", inputs={"count": 10})
    request = AIRequest(
        messages=[{"role": "user", "content": "..."}],
        model=AnthropicModel.CLAUDE_SONNET_4,
        max_tokens=1000,
        step_name="LLM Analysis"
    )
    response = call_anthropic(request, logger)
    logger.output({"result": response.content})

    # Step 3: Save final output
    logger.step("Save Results")
    save_json({"final": "result"}, f"result_{timestamp}.json",
              output_dir="outputs", description="Final Result")
    logger.output({"saved": True})

    # Finalize - prints summary and saves step log to cache/
    logger.finalize()

if __name__ == "__main__":
    main()
```

**Key Changes:**
- ❌ No more `sys.path.insert()` - project is configured with `pyproject.toml`
- ✅ All imports at the top (never inline)
- ✅ Use `StepLogger` instead of `TokenTracker`
- ✅ Call `logger.step()`, `logger.output()`, `logger.finalize()`
- ✅ Step logs automatically save to `cache/` with token tracking included

### 4. Use Pydantic for Structured Data and Typing

**IMPORTANT: Always use Pydantic models for all structured data and type definitions.**

All data models should use Pydantic instead of TypedDict, dataclasses, or plain dicts:
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class VideoFrame(BaseModel):
    timestamp: str
    description: str
    scene_type: str
    score: float = 0.0
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[dict] = None
```

**Why Pydantic:**
- Runtime validation and type checking
- Automatic serialization/deserialization with `.model_dump()` and `.model_validate()`
- Consistent with the AI interface (AIRequest, AIResponse are Pydantic models)
- Better error messages and data validation
- Field-level validation and constraints with `Field()`

**Use Pydantic instead of:**
- ❌ TypedDict
- ❌ dataclasses
- ❌ Plain dictionaries with manual type hints
- ❌ NamedTuples

### 5. Logging and Visibility

**StepLogger automatically provides structured logging with emojis:**
- 🔹 Step start
- 📊 Token usage
- ✅ Step success
- ❌ Step failure

**Additional print statements for user feedback:**
```python
print("🔍 Analyzing video...")
print(f"🤖 Calling Claude with {len(frames)} frames...")
print(f"💾 Saved to cache: {output_path}")
print("✅ Complete!")
```

**Common emojis:**
- 🔍 Analysis/inspection
- 🤖 LLM calls
- 💾 Saving files
- 📊 Stats/metrics
- ✅ Success
- ❌ Errors
- 🔹 Step markers

## Available Models (Enums)

### Anthropic
```python
AnthropicModel.CLAUDE_OPUS_4       # Best quality, slow, expensive
AnthropicModel.CLAUDE_SONNET_4     # Balanced (default choice)
AnthropicModel.CLAUDE_SONNET_3_5   # Previous generation
AnthropicModel.CLAUDE_3_HAIKU      # Fast, cheap, simple tasks
```

### Google Gemini
```python
GeminiModel.GEMINI_2_5_PRO          # Latest, best quality
GeminiModel.GEMINI_2_5_FLASH        # Latest, fast, balanced (default for extract)
GeminiModel.GEMINI_2_5_FLASH_LITE   # Latest, very fast
GeminiModel.GEMINI_2_0_FLASH        # Previous generation
```

### OpenAI (coming soon)
```python
OpenAIModel.GPT_4O
OpenAIModel.GPT_4O_MINI
```

## AI Interface Design

The AI interface is designed to be **provider-agnostic** and support future needs:

### AIRequest (Pydantic model)
```python
AIRequest(
    messages=[{"role": "user", "content": "..."}],
    model=str,                          # Use enum or custom string
    provider=Provider.ANTHROPIC,        # Default
    max_tokens=1024,
    temperature=None,                   # Optional
    system=None,                        # System prompt
    tools=None,                         # List[ToolDefinition] for function calling
    tool_choice=None,                   # "auto", "required", or tool name
    step_name="LLM Call"                # For tracking
)
```

### AIResponse (Pydantic model)
```python
response.content          # Text response
response.model           # Model used
response.provider        # Provider enum
response.input_tokens    # Token counts
response.output_tokens
response.total_tokens
response.tool_calls      # List[ToolCall] if tools were used
response.finish_reason   # "end_turn", "tool_use", etc.
response.raw_response    # Original response object
```

### Tool Calling Support
```python
from utils import ToolDefinition

tools = [
    ToolDefinition(
        name="search",
        description="Search for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    )
]

request = AIRequest(
    messages=[{"role": "user", "content": "Search for cats"}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    tools=tools,
    tool_choice="auto"
)
response = call_anthropic(request, tracker)

if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool: {tool_call.name}, Args: {tool_call.arguments}")
```

## What to Move to Utils

Only move code to `utils/` if:
1. **Used by multiple scripts** - Don't premature optimize
2. **Generic and reusable** - Not task-specific
3. **Well-tested** - Works reliably

Keep task-specific logic in scripts.

## Common Patterns from Existing Scripts

### Video Processing
- Extract frames with ffmpeg
- Encode to base64 for vision APIs
- Use multi-part messages for vision: `[{"type": "text", ...}, {"type": "image", ...}]`
- Save frame analysis to cache
- Use UUIDs for tracking multiple videos

### Multi-step Processing with StepLogger
```python
logger = StepLogger("multi_step_script")

# Step 1 - Log actual file list, not just count
logger.step("Analyze Inputs", inputs={
    "files": ["doc1.txt", "doc2.txt", "doc3.txt"]
})
# ... analysis logic ...
logger.output({
    "analyses": [
        {"file": "doc1.txt", "category": "technical", "score": 0.8},
        {"file": "doc2.txt", "category": "business", "score": 0.6},
        {"file": "doc3.txt", "category": "technical", "score": 0.9}
    ]
})

# Step 2 - Log actual items and results, use updates for progress
logger.step("Filter and Process", inputs={
    "items": [...],  # Actual items
    "threshold": 0.7
})
results = []
for i, item in enumerate(items):
    processed = process_item(item)
    results.append(processed)
    logger.update({"progress": f"{i+1}/{len(items)}", "current": item.id})

logger.output({
    "processed": results,  # Actual results with decisions
    "stats": {"total": len(results), "accepted": 2, "rejected": 1}
})

# Step 3
logger.step("Generate Report")
report = generate_report(results)
logger.output({
    "report": report,
    "saved_to": "outputs/report_20250622.pdf"
})

# Finalize
logger.finalize()
```

### Error Handling
```python
def log_error_and_exit(message: str, exit_code: int = 1):
    print(f"❌ {message}")
    sys.exit(exit_code)

# Use early
if not path.exists():
    log_error_and_exit(f"File not found: {path}")
```

## When User Asks You to Write a Script

1. **Understand the task** - What inputs, what outputs, what processing?
2. **Design the flow** - Break into steps, identify LLM calls needed
3. **Assess if resumable needed** - Many items (50+)? Long-running? Might crash?
4. **Use existing utils** - Don't reinvent, use call_anthropic, extract, save_json, etc.
5. **Structure properly**:
   - Script in `scripts/` directory
   - **All imports at the top** (never inline)
   - Load dotenv before importing utils
   - Initialize `StepLogger("script_name", resumable=True)` if needed
   - Add `--fresh` flag if using resumable mode
   - Use `logger.step()`, `logger.output()`, `logger.update()`
   - Use `logger.mark_item_complete()` for each processed item in resumable scripts
   - Call `logger.finalize()` at end
   - Save interim data to cache, final results to outputs
6. **Use Pydantic models** for all structured data
7. **Add user-facing print statements** for visibility
8. **Handle errors** with `logger.fail(exception)`

## Example: If User Says "Write a script that analyzes text files"

```python
#!/usr/bin/env python3
"""Analyze text files using Claude"""

from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

from utils import call_anthropic, AIRequest, AnthropicModel, save_json, extract
from utils.step_logger import StepLogger

class TextAnalysis(BaseModel):
    summary: str
    key_points: List[str]
    sentiment: str

def main():
    logger = StepLogger("text_analyzer")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Step 1: Find files
    logger.step("Find Text Files", inputs={"directory": "inputs"})
    input_dir = Path("inputs")
    text_files = list(input_dir.glob("*.txt"))

    if not text_files:
        print("❌ No .txt files found in inputs/")
        return

    print(f"🔍 Found {len(text_files)} text files")
    # Log actual file list, not just count
    logger.output({"files": [str(f) for f in text_files]})

    # Step 2: Analyze each file
    # Log actual filenames in inputs
    logger.step("Analyze Files", inputs={
        "files": [f.name for f in text_files]
    })
    results = []

    for i, file_path in enumerate(text_files):
        print(f"🔍 Analyzing: {file_path.name}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Use structured extraction
        analysis = extract(
            text=content,
            schema=TextAnalysis,
            prompt="Analyze this text and extract summary, key points, and sentiment:",
            logger=logger,
            step_name=f"Analyze {file_path.name}"
        )

        results.append({
            "file": file_path.name,
            "analysis": analysis.model_dump()
        })

        logger.update({"analyzed": i + 1, "file": file_path.name})

    # Log actual analyses with decisions, not just count
    logger.output({
        "analyses": results,  # Full results for debugging
        "stats": {"total": len(results)}
    })

    # Step 3: Save final output (deliverable)
    logger.step("Save Results")
    final_report = {
        "timestamp": timestamp,
        "total_files": len(text_files),
        "analyses": results
    }
    save_json(final_report, f"report_{timestamp}.json",
              output_dir="outputs", description="Analysis Report")

    logger.output({"report_path": f"outputs/report_{timestamp}.json"})

    # Finalize (saves step log with token tracking to cache/)
    logger.finalize()
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
```

## Important Don'ts

❌ Don't use `sys.path.insert()` - project is configured with pyproject.toml
❌ Don't put imports inline or at the bottom - **always at the top**
❌ Don't initialize provider clients directly (anthropic.Anthropic(), genai.Client())
❌ Don't save interim steps to `outputs/` (only final results)
❌ Don't log summary stats without actual data - include decisions, reasons, extractions
❌ Don't create separate cache files when data can fit in StepLogger outputs
❌ Don't forget to initialize StepLogger and call `.finalize()`
❌ Don't use generic step names ("Step 1") - be descriptive
❌ Don't hard-code API keys
❌ Don't skip type hints
❌ Don't create utils prematurely - keep in script until needed by multiple scripts
❌ Don't forget to call `load_dotenv()` before importing utils
❌ Don't use TypedDict, dataclasses, or plain dicts - always use Pydantic BaseModel
❌ Don't forget `--fresh` flag when using resumable mode
❌ Don't forget to call `mark_item_complete()` after processing each item in resumable scripts

## Key Takeaway

When the user asks for a script, write clean, well-structured code that:
1. **All imports at the top** - never inline
2. Uses `StepLogger` as single source of truth - log rich data, not just stats
3. Logs **actual decisions and data** in inputs/outputs for debugging
4. Uses the utils properly (call_anthropic, call_gemini, extract)
5. Uses descriptive step names and passes logger to all LLM calls
6. **Uses resumable mode** for scripts processing 50+ items or long-running operations
7. Includes `--fresh` flag and proper `should_run_step()` / `mark_item_complete()` patterns for resumable scripts
8. Saves final deliverables to outputs (not interim data)
9. Uses Pydantic models for all structured data
10. Has good visibility (print statements with emojis)
11. Calls `logger.finalize()` at the end

**Remember**:
- StepLogger should contain everything you need to understand what happened and why
- For long-running scripts, resumable mode prevents re-work after crashes
- Use `uv run scripts/clear_cache.py` to clear all cached data including resume state

Keep it simple, keep it clean, keep it reusable.
