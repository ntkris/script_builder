"""Input/Output utilities for saving and loading data"""
import json
from pathlib import Path
from typing import Optional


def save_json(data: dict, filename: str, output_dir: str = "outputs", description: str = "data") -> Optional[str]:
    """
    Save data as JSON to specified directory.

    Args:
        data: Dictionary to save as JSON
        filename: Name of the file (without path)
        output_dir: Directory to save to (default: "outputs")
        description: Description for logging (default: "data")

    Returns:
        Path to saved file, or None if failed
    """
    try:
        Path(output_dir).mkdir(exist_ok=True)
        output_path = f"{output_dir}/{filename}"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"💾 {description}: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ Failed to save {description}: {e}")
        return None


def load_json(file_path: str) -> Optional[dict]:
    """
    Load JSON data from file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data as dict, or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {file_path}: {e}")
        return None
