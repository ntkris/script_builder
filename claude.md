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
│   │   └── README.md     # Detailed examples
│   ├── io.py             # save_json(), load_json()
│   └── token_tracking.py # TokenTracker class
├── inputs/                # Input files (videos, images, text, data)
├── outputs/               # ONLY final results (videos, reports, etc.)
├── cache/                 # Interim processing steps (JSON logs with timestamps)
└── .env                   # API keys (ANTHROPIC_API_KEY)
```

## Key Patterns

### 1. File Organization
- **Scripts** go in `scripts/` directory
- **Interim steps** (analysis, processing logs) → `cache/` as JSON
- **Final outputs** (videos, final reports) → `outputs/`
- Always include timestamps in filenames: `analysis_20250622_143022.json`

### 2. Utils Available

#### AI/LLM Calls
```python
from utils import call_anthropic, AIRequest, AnthropicModel, TokenTracker

tracker = TokenTracker()

request = AIRequest(
    messages=[{"role": "user", "content": "..."}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    max_tokens=1000,
    step_name="Descriptive Step Name"  # Important for tracking
)
response = call_anthropic(request, tracker)
# Automatically tracks tokens and prints usage
```

**Key points:**
- API keys are handled internally (from `ANTHROPIC_API_KEY` env var)
- Don't import or initialize `anthropic` client directly
- Always pass `tracker` for automatic token tracking
- Use descriptive `step_name` for logging clarity

#### JSON I/O
```python
from utils import save_json, load_json

# Save to cache (interim steps)
save_json(data, "step1_20250622.json", output_dir="cache", description="Step 1")

# Save to outputs (final results)
save_json(final, "result_20250622.json", output_dir="outputs", description="Final")

# Load
data = load_json("cache/step1.json")
```

#### Token Tracking
```python
tracker = TokenTracker()
# ... make calls with call_anthropic(request, tracker) ...

# Get totals
total = tracker.get_total_usage()
print(f"Total: {total.total_tokens} tokens")

# Save summary to cache
tracker.save_summary("script_name", output_dir="cache")
```

### 3. Common Script Structure

```python
import sys
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_anthropic, AIRequest, AnthropicModel, TokenTracker, save_json

# Initialize
tracker = TokenTracker()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define Pydantic models for structured data
class Result(BaseModel):
    field: str
    score: float

def main():
    # 1. Load inputs
    # 2. Process with LLM (track tokens)
    # 3. Save interim steps to cache/
    # 4. Save final output to outputs/
    # 5. Save token summary
    tracker.save_summary("script_name", output_dir="cache")

if __name__ == "__main__":
    main()
```

### 4. Use Pydantic for Structured Data

All data models should use Pydantic:
```python
from pydantic import BaseModel, Field
from typing import List, Optional

class VideoFrame(BaseModel):
    timestamp: str
    description: str
    scene_type: str
    score: float = 0.0
```

This is consistent with the AI interface (AIRequest, AIResponse are Pydantic models).

### 5. Logging and Visibility

Use emojis for visual clarity in print statements:
- 🔍 Analysis/inspection
- 🤖 LLM calls
- 💾 Saving files
- 📊 Stats/metrics
- ✅ Success
- ❌ Errors

```python
print("🔍 Analyzing video...")
print(f"🤖 Calling Claude with {len(frames)} frames...")
print(f"💾 Saved to cache: {output_path}")
print("✅ Complete!")
```

## Available Models (Enums)

### Anthropic
```python
AnthropicModel.CLAUDE_OPUS_4       # Best quality, slow, expensive
AnthropicModel.CLAUDE_SONNET_4     # Balanced (default choice)
AnthropicModel.CLAUDE_SONNET_3_5   # Previous generation
AnthropicModel.CLAUDE_3_HAIKU      # Fast, cheap, simple tasks
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

### Multi-step Processing
1. Analyze inputs (save to cache)
2. Select/filter (save to cache)
3. Generate plan (save to cache)
4. Execute plan
5. Save final output
6. Save token summary

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
3. **Use existing utils** - Don't reinvent, use call_anthropic, save_json, etc.
4. **Structure properly**:
   - Script in `scripts/` directory
   - Import utils properly
   - Initialize TokenTracker
   - Save interim steps to cache
   - Save final output to outputs
   - Save token summary
5. **Use Pydantic models** for structured data
6. **Add logging** with emojis for clarity
7. **Handle errors** gracefully

## Example: If User Says "Write a script that analyzes text files"

```python
import sys
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_anthropic, AIRequest, AnthropicModel, TokenTracker, save_json

tracker = TokenTracker()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

class TextAnalysis(BaseModel):
    summary: str
    key_points: List[str]
    sentiment: str

def analyze_text_file(file_path: Path) -> TextAnalysis:
    """Analyze a text file using Claude"""
    print(f"🔍 Analyzing: {file_path.name}")

    # Read file
    with open(file_path, 'r') as f:
        content = f.read()

    # Call Claude
    request = AIRequest(
        messages=[{"role": "user", "content": f"Analyze this text:\n\n{content}"}],
        model=AnthropicModel.CLAUDE_SONNET_4,
        max_tokens=1000,
        step_name=f"Text Analysis - {file_path.name}"
    )
    response = call_anthropic(request, tracker)

    # Parse response (simplified - would need XML parsing in real script)
    # ... parse response.content into TextAnalysis ...

    return analysis

def main():
    input_dir = Path("inputs")
    text_files = list(input_dir.glob("*.txt"))

    if not text_files:
        print("❌ No .txt files found in inputs/")
        sys.exit(1)

    print(f"🔍 Found {len(text_files)} text files")

    results = []
    for file_path in text_files:
        analysis = analyze_text_file(file_path)
        results.append({
            "file": file_path.name,
            "analysis": analysis.model_dump()
        })

    # Save interim results to cache
    save_json(results, f"analyses_{timestamp}.json", output_dir="cache", description="Text Analyses")

    # Save summary to outputs
    summary = {"total_files": len(text_files), "timestamp": timestamp}
    save_json(summary, f"summary_{timestamp}.json", output_dir="outputs", description="Summary")

    # Save token usage
    tracker.save_summary("text_analyzer", output_dir="cache")

    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
```

## Important Don'ts

❌ Don't initialize `anthropic.Anthropic()` client directly
❌ Don't save interim steps to `outputs/` (only final results)
❌ Don't forget to track tokens
❌ Don't use generic step names ("Step 1") - be descriptive
❌ Don't import anthropic unless absolutely needed for advanced features
❌ Don't hard-code API keys
❌ Don't skip type hints
❌ Don't create utils prematurely - keep in script until needed by multiple scripts

## Key Takeaway

When the user asks for a script, write clean, well-structured code that:
1. Uses the utils properly
2. Logs incremental steps to cache
3. Tracks tokens automatically
4. Saves final results to outputs
5. Uses Pydantic models
6. Has good visibility (emojis, print statements)
7. Follows the patterns established in existing scripts (like video_generator.py)

Keep it simple, keep it clean, keep it reusable.
