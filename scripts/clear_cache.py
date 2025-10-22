#!/usr/bin/env python3
"""Clear all cached files from the cache directory."""

from pathlib import Path
import shutil


def main():
    cache = Path("cache")

    if not cache.exists():
        print("📁 Cache directory does not exist")
        return

    files = list(cache.glob("*"))

    if not files:
        print("✨ Cache is already empty")
        return

    print(f"🗑️  Deleting {len(files)} files from cache/...")

    deleted = 0
    for file in files:
        try:
            if file.is_file():
                file.unlink()
                deleted += 1
            elif file.is_dir():
                shutil.rmtree(file)
                deleted += 1
        except Exception as e:
            print(f"⚠️  Failed to delete {file.name}: {e}")

    print(f"✅ Deleted {deleted} items from cache/")


if __name__ == "__main__":
    main()
