#!/usr/bin/env python3
"""
AI Video Editor Agent

Uses Claude Agent SDK + FFmpeg to create intelligent video edits.

Features:
- Auto-detects videos from inputs/videos/
- AI analyzes content using vision
- Intelligently selects best moments
- Creates final edit with transitions
- Auto-saves to outputs/videos/

Usage:
    # Simplest - uses default prompt
    uv run scripts/video_editor_agent.py

    # Custom prompt
    uv run scripts/video_editor_agent.py --prompt "Create a 60-second action reel"
"""

import asyncio
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from utils.step_logger import StepLogger
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, create_sdk_mcp_server

# Import video tools
from utils.tools.video_tools import (
    analyze_video_content,
    get_video_info,
    trim_video,
    concatenate_videos,
    resize_video
)


load_dotenv()

# Pydantic models
class EditingTask(BaseModel):
    """Track the editing task"""
    input_videos: List[str]
    prompt: str
    output_path: str
    timestamp: str


class EditingSummary(BaseModel):
    """Summary of the editing session"""
    total_videos_analyzed: int
    clips_selected: int
    final_duration: float
    output_path: str
    success: bool
    error: Optional[str] = None


async def run_video_editing_agent(
    videos: List[Path],
    editing_prompt: str,
    output_path: Path,
    logger: StepLogger
) -> bool:
    """Run the video editing agent"""

    # Step 1: Setup tools
    logger.step("Initialize Video Editing Tools")

    print("🔧 Creating MCP server with video editing tools...")
    server = create_sdk_mcp_server(
        name="video-editing-tools",
        version="1.0.0",
        tools=[
            analyze_video_content,
            get_video_info,
            trim_video,
            concatenate_videos,
            resize_video
        ]
    )

    # Configure agent
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",  # Use Sonnet for vision + complex reasoning
        mcp_servers={"video": server},
        allowed_tools=[
            "Bash",  # Allow bash for ffmpeg if needed
            "Read",  # Allow reading files (including extracted frames)
            "Write", # Allow writing temp files
            "mcp__video__analyze_video_content",
            "mcp__video__get_video_info",
            "mcp__video__trim_video",
            "mcp__video__concatenate_videos",
            "mcp__video__resize_video"
        ],
        max_turns=50,  # Complex edits may need multiple steps
        max_buffer_size=20 * 1024 * 1024  # 20MB for video metadata + frames
    )

    logger.output({
        "tools_registered": 5,
        "model": "claude-sonnet-4-5",
        "max_turns": 50
    })

    # Step 2: Run agent
    logger.step("Execute Intelligent Video Editing", inputs={
        "videos": [str(v) for v in videos],
        "prompt": editing_prompt
    })

    # Create cache directory for intermediate files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = Path("cache") / f"video_editing_{timestamp}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    prompt = f"""You are a professional video editor with AI-powered content analysis and FFmpeg tools.

USER REQUEST: {editing_prompt}

INPUT VIDEOS:
{chr(10).join(f"- {v}" for v in videos)}

OUTPUT PATH: {output_path}

INTERMEDIATE FILES: Save to {cache_dir}/ directory

WORKFLOW:

1. **ANALYZE CONTENT** (Use analyze_video_content for each video):
   - Analyze BOTH video AND audio using Gemini 2.5 Flash
   - AUDIO IS CRITICAL: Listen for speech, introductions, narration, explanations
   - Identify interesting moments, scene changes, highlights
   - Score segments based on visual interest, motion, composition, AND audio content
   - Understand what's happening AND what's being said in each video

   **IMPORTANT:** Portrait/selfie shots with speech are likely INTROS and should come FIRST!
   Gemini will mark these with scene_type="intro" and has_speech=true.

2. **DECIDE WHICH CLIPS TO USE**:
   - Based on your analysis, select the BEST segments from each video
   - **PRIORITIZE INTRO SEGMENTS FIRST**: If any video has scene_type="intro" or has_speech=true at the start, place it FIRST in the final edit
   - Consider the user's prompt (e.g., "action moments" vs "calm scenes")
   - Aim for variety and visual interest
   - Plan total duration based on prompt (default: 30 seconds)
   - Logical flow: intro → main content → conclusion

3. **PLAN THE EDIT**:
   - **ORDER MATTERS**: Intro segments FIRST, then main content
   - Which segments to trim (start/end times)
   - Target format (vertical 1080x1920 for reels, or 1920x1080 for horizontal)
   - Save trimmed clips to cache directory
   - Plan concatenation order to create logical flow

4. **EXECUTE THE EDIT** using tools:
   - get_video_info: Check video metadata (duration, resolution)
   - trim_video: Extract selected segments to {cache_dir}/clip_1.mp4, clip_2.mp4, etc.
   - resize_video: Standardize all clips to target format (e.g., 1080x1920)
   - concatenate_videos: Join clips into final output at {output_path}

5. **IMPORTANT TECHNICAL NOTES**:
   - For concatenation to work, ALL clips must have the same resolution
   - Resize clips BEFORE concatenating
   - Use mode="crop" for resize to avoid black bars
   - Save trimmed clips as: {cache_dir}/clip_001.mp4, {cache_dir}/clip_002.mp4, etc.
   - Save resized clips as: {cache_dir}/resized_001.mp4, {cache_dir}/resized_002.mp4, etc.
   - Then concatenate all resized clips

6. **ERROR HANDLING**:
   - If analyze_video_content fails, use get_video_info and pick clips evenly
   - If tools fail, try Bash tool for direct ffmpeg
   - Simplify if needed - a working basic edit is better than a failed complex one

7. **REPORT PROGRESS**:
   - "Analyzing video 1 of 4..."
   - "Found interesting segment at 0:15-0:23 (high motion, vibrant colors)"
   - "Selected 4 segments totaling 28 seconds"
   - "Trimming clips..."
   - "Resizing to 1080x1920..."
   - "Creating final reel..."

**EXAMPLE WORKFLOW**:
```
1. analyze_video_content(video_1) → finds segment 5.0-12.0 (score 8.5)
2. analyze_video_content(video_2) → finds segment 15.0-22.0 (score 7.8)
3. trim_video(video_1, cache/clip_001.mp4, start=5.0, duration=7.0)
4. trim_video(video_2, cache/clip_002.mp4, start=15.0, duration=7.0)
5. resize_video(cache/clip_001.mp4, cache/resized_001.mp4, 1080, 1920, "crop")
6. resize_video(cache/clip_002.mp4, cache/resized_002.mp4, 1080, 1920, "crop")
7. concatenate_videos([cache/resized_001.mp4, cache/resized_002.mp4], {output_path})
```

**REMEMBER**: You have VISION capabilities. When analyze_video_content extracts frames,
you can READ and SEE them to make informed decisions about which clips are most interesting.

Begin the intelligent editing workflow now. Be systematic and report progress clearly."""

    messages = []
    token_usage = None
    total_cost = None
    result_message = None

    print("\n" + "="*60)
    print("🤖 Agent working...")
    print("="*60 + "\n")

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)

        async for message in client.receive_response():
            messages.append(message)
            # Print messages for user visibility
            print(f"  {message}")

            # Extract token usage from ResultMessage
            if hasattr(message, 'usage') and message.usage:
                token_usage = message.usage
            if hasattr(message, 'total_cost_usd') and message.total_cost_usd:
                total_cost = message.total_cost_usd
            if hasattr(message, 'subtype') and message.subtype == 'success':
                result_message = message

    logger.output({
        "messages_received": len(messages),
        "all_messages": [str(m) for m in messages],
        "output_exists": Path(output_path).exists(),
        "token_usage": token_usage,
        "total_cost_usd": total_cost,
        "result_summary": str(result_message) if result_message else None
    })

    # Return success status, token usage, and cost
    return (Path(output_path).exists(), token_usage, total_cost)


async def main():
    """Main execution"""
    import argparse

    # Default prompt - intelligent and engaging
    DEFAULT_PROMPT = "Create an engaging 30-second vertical reel. Analyze the videos, pick the most visually interesting moments with good motion and composition, and create a dynamic edit with smooth transitions. No music needed."

    parser = argparse.ArgumentParser(
        description="🎬 AI Video Editor - Intelligent editing with Claude + FFmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest - uses default prompt
  uv run scripts/video_editor_agent.py

  # Custom prompt
  uv run scripts/video_editor_agent.py --prompt "Create a 60-second action reel"

  # Calm montage
  uv run scripts/video_editor_agent.py --prompt "Create a calm 45-second montage"
        """
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Natural language editing instructions"
    )

    args = parser.parse_args()

    # Auto-detect input videos from inputs/videos/
    video_dir = Path("inputs/videos")
    if not video_dir.exists():
        print(f"❌ Directory not found: {video_dir}")
        print("💡 Create it and add your videos:")
        print(f"   mkdir -p {video_dir}")
        print(f"   # Then add .mp4, .mov, .avi, .mkv, or .webm files")
        return

    # Find all video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(video_dir.glob(f"*{ext}"))
        video_paths.extend(video_dir.glob(f"*{ext.upper()}"))

    if not video_paths:
        print(f"❌ No videos found in {video_dir}")
        print(f"💡 Supported formats: {', '.join(video_extensions)}")
        return

    # Sort by name for consistent ordering
    video_paths = sorted(video_paths)

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"reel_{timestamp}.mp4"

    # Initialize logger
    logger = StepLogger("video_editor_agent")

    print("\n" + "="*60)
    print("🎬 AI Video Editor - Intelligent Editing with Claude + FFmpeg")
    print("="*60)
    print(f"\n📁 Input directory: {video_dir}")
    print(f"📹 Found {len(video_paths)} videos:")
    for i, v in enumerate(video_paths, 1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"   {i}. {v.name} ({size_mb:.1f} MB)")

    print(f"\n📝 Editing prompt:")
    print(f"   {args.prompt}")
    print(f"\n📁 Output: {output_path}")
    print("="*60)
    print("\n🤖 Agent will:")
    print("  1. Analyze video content using vision AI")
    print("  2. Identify most interesting moments")
    print("  3. Select and trim best clips")
    print("  4. Resize to target format")
    print("  5. Concatenate into final edit")
    print("="*60 + "\n")

    try:
        # Run agent (returns success, token_usage, cost)
        result = await run_video_editing_agent(
            video_paths,
            args.prompt,
            output_path,
            logger
        )

        # Unpack result
        if isinstance(result, tuple):
            success, token_usage, total_cost = result
        else:
            success = result
            token_usage = None
            total_cost = None

        # Finalize
        logger.finalize()

        print("\n" + "="*60)
        if success:
            print("✅ Video editing complete!")
            print("="*60)
            print(f"\n📁 Output saved to: {output_path}")

            # Get file size
            if output_path.exists():
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"📊 File size: {size_mb:.1f} MB")

            # Print token usage summary
            if token_usage and total_cost is not None:
                print(f"\n💰 Token Usage & Cost:")
                print(f"   Input tokens: {token_usage.get('input_tokens', 0):,}")
                print(f"   Cache creation: {token_usage.get('cache_creation_input_tokens', 0):,}")
                print(f"   Cache read: {token_usage.get('cache_read_input_tokens', 0):,}")
                print(f"   Output tokens: {token_usage.get('output_tokens', 0):,}")
                print(f"   Total cost: ${total_cost:.4f}")
            else:
                print(f"\n💰 Token usage: See step logs in cache/")

            print(f"\n💡 You can view it with:")
            print(f"   open {output_path}")
        else:
            print("❌ Video editing failed")
            print("="*60)
            print("\n💡 Check the logs above for error details")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        logger.finalize()
    except Exception as e:
        logger.fail(e)
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
