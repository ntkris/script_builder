# Script Builder

Lightweight prototyping environment for AI-powered scripts. Built for rapid experimentation with a focus on:
- **Minimal setup**: Just `uv` for package management
- **Reusable utilities**: Common LLM and I/O operations abstracted
- **Automatic logging**: All LLM calls tracked, interim steps saved to cache
- **Multi-provider support**: Universal interface for Anthropic, OpenAI, and more

## Quick Start

1. **Setup environment:**
   ```bash
   # Create .env file
   echo "ANTHROPIC_API_KEY=your_key_here" > .env
   ```

2. **Install dependencies:**
   ```bash
   uv add package_name
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
│   │   ├── anthropic_provider.py
│   │   └── README.md     # Detailed AI usage docs
│   ├── io.py             # save_json, load_json helpers
│   └── token_tracking.py # TokenTracker for usage monitoring
├── inputs/                # Input files (videos, images, data)
├── outputs/               # Final results only
├── cache/                 # Interim processing steps (JSON logs)
├── .env                   # API keys (gitignored)
└── pyproject.toml         # Dependencies managed by uv
```

## Core Principles

1. **Scripts in `scripts/`**: All executable scripts go in the scripts directory
2. **Interim steps → `cache/`**: Save processing logs, intermediate results as JSON
3. **Final results → `outputs/`**: Only final artifacts (videos, reports, etc.)
4. **Reusable code → `utils/`**: Extract common patterns, avoid duplication
5. **Automatic tracking**: All LLM calls automatically logged with token usage

## Usage Examples

### Basic Script Structure

```python
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import call_anthropic, AIRequest, AnthropicModel, TokenTracker, save_json

# Initialize token tracker
tracker = TokenTracker()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Make LLM call with automatic token tracking
request = AIRequest(
    messages=[{"role": "user", "content": "Analyze this data..."}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    max_tokens=1000,
    step_name="Data Analysis"
)
response = call_anthropic(request, tracker)

# Save interim step to cache
save_json(
    {"analysis": response.content, "timestamp": timestamp},
    f"analysis_{timestamp}.json",
    output_dir="cache",
    description="Analysis Step"
)

# Save token usage summary
tracker.save_summary("my_script", output_dir="cache")
```

### AI/LLM Calls

```python
from utils import call_anthropic, AIRequest, AnthropicModel, TokenTracker

tracker = TokenTracker()

# Simple request
request = AIRequest(
    messages=[{"role": "user", "content": "Hello!"}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    max_tokens=100,
    step_name="Greeting"
)
response = call_anthropic(request, tracker)
print(response.content)

# With system prompt and temperature
request = AIRequest(
    messages=[{"role": "user", "content": "Explain quantum physics"}],
    model=AnthropicModel.CLAUDE_SONNET_4,
    max_tokens=500,
    system="You are a physics professor. Be concise.",
    temperature=0.7,
    step_name="Physics Explanation"
)
response = call_anthropic(request, tracker)

# API key automatically loaded from ANTHROPIC_API_KEY env var
# Override with: call_anthropic(request, tracker, api_key="custom_key")
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

### Token Tracking

```python
from utils import TokenTracker

tracker = TokenTracker()

# Make multiple LLM calls...
# (tracking happens automatically if you pass tracker to call_anthropic)

# Get total usage
total = tracker.get_total_usage()
print(f"Total tokens: {total.total_tokens}")

# Save detailed summary to cache
tracker.save_summary("my_script", output_dir="cache")
```

## Available Models

### Anthropic
- `AnthropicModel.CLAUDE_OPUS_4`
- `AnthropicModel.CLAUDE_SONNET_4`
- `AnthropicModel.CLAUDE_SONNET_3_5`
- `AnthropicModel.CLAUDE_3_OPUS`
- `AnthropicModel.CLAUDE_3_SONNET`
- `AnthropicModel.CLAUDE_3_HAIKU`

### OpenAI (coming soon)
- `OpenAIModel.GPT_4O`
- `OpenAIModel.GPT_4O_MINI`

## Advanced Features

See `utils/ai/README.md` for:
- Tool/function calling
- Streaming responses
- Multi-turn conversations
- Custom providers
