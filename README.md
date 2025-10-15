# Script Builder

Collection of AI-powered scripts using uv for package management.

## Directory Structure

```
script_builder/
├── *.py                   # Your scripts
├── .env                   # Environment variables
├── inputs/                # Input files
├── outputs/               # Generated outputs
├── cache/                 # Cache directory
└── README.md              # This file
```

## How to Run Scripts

```bash
uv run script_name.py
```

## How to Install New Packages

```bash
uv add package_name
```

## How to Save New Scripts

1. Create your script file: `my_script.py`
2. Add dependencies: `uv add dependency_name`
3. Run: `uv run my_script.py`

## Environment Setup

Create `.env` file for API keys:
```env
ANTHROPIC_API_KEY=your_key_here
```
