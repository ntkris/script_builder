#!/usr/bin/env python3
"""
Video Editing Tools for Claude Agent SDK

Provides MCP tools for intelligent video editing:
- analyze_video_content: Vision-powered content analysis
- get_video_info: Extract video metadata
- trim_video: Cut video segments
- concatenate_videos: Join multiple videos
- resize_video: Change resolution and aspect ratio
"""

import subprocess
import tempfile
import base64
import json
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from claude_agent_sdk import tool

# Import existing AI utilities for video analysis
from utils import call_gemini, AIRequest, GeminiModel, Provider
import os
from google import genai
from google.genai import types


# Pydantic Models
class VideoSegment(BaseModel):
    """A scored segment from video analysis"""
    start_time: float
    end_time: float
    score: float = Field(ge=0.0, le=10.0, description="Visual interest score (0-10)")
    reason: str
    scene_type: str  # "action", "scenic", "people", "static", etc.


class VideoContentAnalysis(BaseModel):
    """Complete analysis of video content"""
    interesting_segments: List[VideoSegment] = Field(default_factory=list)
    scene_changes: List[float] = Field(default_factory=list)
    overall_quality: float = Field(ge=0.0, le=10.0)
    summary: str
    audio_summary: Optional[str] = None  # Summary of audio content
    has_speech: bool = False  # Whether video contains speech/narration


class VideoInfo(BaseModel):
    """Video metadata"""
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    audio_codec: Optional[str] = None
    file_size_mb: float


class EditResult(BaseModel):
    """Result of a video editing operation"""
    success: bool
    output_path: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


# Helper Functions
def extract_frames_from_video(
    video_path: str, output_dir: str, interval_seconds: float = 2.0, max_frames: int = 30
) -> List[Path]:
    """Extract frames from video at specified intervals using FFmpeg"""
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(exist_ok=True, parents=True)

    # Get video duration
    duration_cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception("Failed to get video duration")

    duration = float(result.stdout.strip())
    total_possible_frames = int(duration / interval_seconds)
    frames_to_extract = min(max_frames, total_possible_frames)

    if frames_to_extract == 0:
        frames_to_extract = 1
        interval_seconds = duration / 2

    frame_paths = []
    for i in range(frames_to_extract):
        timestamp = i * interval_seconds
        if timestamp >= duration:
            break

        frame_path = frames_dir / f"frame_{i:03d}_{timestamp:.1f}s.jpg"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(frame_path),
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            frame_paths.append(frame_path)

    return frame_paths


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def run_ffmpeg_command(command: List[str], timeout: int = 300) -> EditResult:
    """Execute ffmpeg command with error handling"""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return EditResult(success=True)
        else:
            return EditResult(success=False, error=result.stderr)

    except subprocess.TimeoutExpired:
        return EditResult(success=False, error=f"FFmpeg timed out after {timeout} seconds")
    except Exception as e:
        return EditResult(success=False, error=str(e))


# MCP Tools
@tool(
    name="analyze_video_content",
    description="Analyze video content using Gemini 2.5 Flash with BOTH video and audio analysis. Identifies interesting moments, scene changes, visual highlights, AND understands spoken content. Returns scored segments for intelligent clip selection.",
    input_schema={
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the video file to analyze"
            }
        },
        "required": ["video_path"]
    }
)
async def analyze_video_content(args):
    """
    Analyze video content using Gemini 2.5 Flash with video + audio.

    This is the KEY tool for intelligent clip selection.
    Uploads video to Gemini, analyzes BOTH visual and audio content, and returns scored segments.

    AUDIO ANALYSIS: Understands speech, introductions, narration, and background sounds.
    This is crucial for identifying intros, explanations, and context.
    """
    video_path = args["video_path"]

    try:
        # Get API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": "GEMINI_API_KEY not found in environment",
                        "success": False
                    })
                }]
            }

        # Initialize Gemini client
        client = genai.Client(api_key=api_key)

        # Upload video file to Gemini
        print(f"  📤 Uploading {Path(video_path).name} to Gemini...")
        video_file = client.files.upload(path=video_path)

        # Wait for processing to complete
        print(f"  ⏳ Processing video...")
        import time
        while video_file.state == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state != "ACTIVE":
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": f"Video processing failed: {video_file.state}",
                        "success": False
                    })
                }]
            }

        print(f"  ✅ Video processed, analyzing content...")

        # Build analysis prompt that includes BOTH video and audio
        analysis_prompt = """Analyze this video comprehensively using BOTH visual and audio information.

**AUDIO ANALYSIS IS CRITICAL:**
- Listen carefully for speech, narration, introductions, or explanations
- If someone is speaking/introducing, note this in the scene_type and reason
- Identify background sounds (cooking sounds, pouring, etc.)
- Mark segments with speech as high priority for intros or explanations

**VISUAL ANALYSIS:**
- Assess visual interest (motion, colors, composition) - score 0-10
- Identify scene types (intro, action, scenic, cooking, static, etc.)
- Note scene changes and transitions
- Evaluate quality and aesthetic appeal

**IMPORTANT CONTEXT CLUES:**
- Portrait/selfie shots with speech are likely INTRODUCTIONS → should come FIRST
- Cooking sounds (sizzling, boiling) indicate action moments
- Pouring sounds complement visual pour shots
- Static shots with speech may be explanations

Return JSON matching this structure (must be valid JSON):
{
  "interesting_segments": [
    {
      "start_time": 0.0,
      "end_time": 8.0,
      "score": 9.5,
      "reason": "Person introducing the video with speech - INTRO segment",
      "scene_type": "intro"
    },
    {
      "start_time": 15.0,
      "end_time": 22.0,
      "score": 8.0,
      "reason": "Active cooking with bubbling sounds and motion",
      "scene_type": "action"
    }
  ],
  "scene_changes": [0.0, 8.0, 15.0, 30.0],
  "overall_quality": 7.5,
  "summary": "Video starts with person introducing content, then shows cooking process",
  "audio_summary": "Speech introduction at start, then cooking sounds (sizzling, boiling)",
  "has_speech": true
}

**CRITICAL:** If there is ANY speech/narration, set has_speech=true and note it in scene_type and reason!"""

        # Generate content with video
        response = client.models.generate_content(
            model=GeminiModel.GEMINI_2_5_FLASH,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=analysis_prompt),
                        types.Part(file_data=types.FileData(file_uri=video_file.uri))
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )

        # Parse response
        try:
            analysis_text = response.text
            analysis_data = json.loads(analysis_text)

            # Validate with Pydantic
            analysis = VideoContentAnalysis.model_validate(analysis_data)

            # Clean up - delete uploaded file
            client.files.delete(name=video_file.name)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(analysis.model_dump(), indent=2)
                }]
            }
        except Exception as e:
            # Clean up on error
            try:
                client.files.delete(name=video_file.name)
            except:
                pass

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "error": f"Failed to parse analysis: {str(e)}",
                        "raw_response": response.text if hasattr(response, 'text') else str(response)
                    })
                }]
            }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "error": str(e),
                    "success": False
                })
            }]
        }


@tool(
    name="get_video_info",
    description="Get metadata about a video file including duration, resolution, codec, fps, and file size",
    input_schema={
        "type": "object",
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the video file"
            }
        },
        "required": ["video_path"]
    }
)
async def get_video_info(args):
    """Get video metadata using ffprobe"""
    video_path = args["video_path"]

    try:
        # Use ffprobe to get video info
        ffprobe_cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"error": "Failed to get video info", "success": False})
                }]
            }

        probe_data = json.loads(result.stdout)

        # Extract video stream info
        video_stream = next(
            (s for s in probe_data["streams"] if s["codec_type"] == "video"),
            None
        )

        # Extract audio stream info
        audio_stream = next(
            (s for s in probe_data["streams"] if s["codec_type"] == "audio"),
            None
        )

        if not video_stream:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({"error": "No video stream found", "success": False})
                }]
            }

        # Calculate FPS
        fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0

        # Get file size
        file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)

        info = VideoInfo(
            duration=float(probe_data["format"]["duration"]),
            width=int(video_stream["width"]),
            height=int(video_stream["height"]),
            fps=fps,
            codec=video_stream["codec_name"],
            audio_codec=audio_stream["codec_name"] if audio_stream else None,
            file_size_mb=file_size_mb
        )

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(info.model_dump(), indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"error": str(e), "success": False})
            }]
        }


@tool(
    name="trim_video",
    description="Extract a segment from a video by specifying start time and duration",
    input_schema={
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Path to the input video file"
            },
            "output_path": {
                "type": "string",
                "description": "Path for the output trimmed video"
            },
            "start_time": {
                "type": "number",
                "description": "Start time in seconds"
            },
            "duration": {
                "type": "number",
                "description": "Duration to extract in seconds"
            }
        },
        "required": ["input_path", "output_path", "start_time", "duration"]
    }
)
async def trim_video(args):
    """Trim video segment using ffmpeg"""
    input_path = args["input_path"]
    output_path = args["output_path"]
    start_time = args["start_time"]
    duration = args["duration"]

    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Build ffmpeg command
        # -ss before -i for faster seeking
        # -c copy for fast processing (no re-encoding)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-ss", str(start_time),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",  # Copy streams without re-encoding
            str(output_path)
        ]

        result = run_ffmpeg_command(ffmpeg_cmd)

        if result.success:
            result.output_path = output_path
            result.duration = duration

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result.model_dump(), indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(EditResult(success=False, error=str(e)).model_dump())
            }]
        }


@tool(
    name="concatenate_videos",
    description="Join multiple videos into one. Videos should have the same resolution and codec for best results.",
    input_schema={
        "type": "object",
        "properties": {
            "input_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of video file paths to concatenate (in order)"
            },
            "output_path": {
                "type": "string",
                "description": "Path for the output concatenated video"
            },
            "transition": {
                "type": "string",
                "enum": ["none", "fade"],
                "description": "Transition type between clips (default: none)",
                "default": "none"
            }
        },
        "required": ["input_paths", "output_path"]
    }
)
async def concatenate_videos(args):
    """Concatenate multiple videos using ffmpeg"""
    input_paths = args["input_paths"]
    output_path = args["output_path"]
    transition = args.get("transition", "none")

    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if len(input_paths) < 2:
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": "Need at least 2 videos to concatenate"
                    })
                }]
            }

        if transition == "none":
            # Simple concatenation using concat demuxer (fast)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for path in input_paths:
                    # Escape single quotes and write
                    escaped_path = str(path).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                str(output_path)
            ]

            result = run_ffmpeg_command(ffmpeg_cmd)

            # Clean up concat file
            Path(concat_file).unlink(missing_ok=True)

        else:
            # Concatenation with fade transition (requires re-encoding)
            # This is more complex - for MVP, we'll use simple concat
            # and note that transitions need re-encoding
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "success": False,
                        "error": "Fade transitions not yet implemented in MVP. Use transition='none' for now."
                    })
                }]
            }

        if result.success:
            result.output_path = output_path

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result.model_dump(), indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(EditResult(success=False, error=str(e)).model_dump())
            }]
        }


@tool(
    name="resize_video",
    description="Change video resolution and aspect ratio. Common presets: vertical reel (1080x1920), horizontal (1920x1080), square (1080x1080)",
    input_schema={
        "type": "object",
        "properties": {
            "input_path": {
                "type": "string",
                "description": "Path to the input video file"
            },
            "output_path": {
                "type": "string",
                "description": "Path for the output resized video"
            },
            "width": {
                "type": "integer",
                "description": "Target width in pixels"
            },
            "height": {
                "type": "integer",
                "description": "Target height in pixels"
            },
            "mode": {
                "type": "string",
                "enum": ["crop", "pad", "stretch"],
                "description": "Resize mode: crop (fill frame, crop edges), pad (fit with black bars), stretch (distort to fit)",
                "default": "crop"
            }
        },
        "required": ["input_path", "output_path", "width", "height"]
    }
)
async def resize_video(args):
    """Resize video using ffmpeg"""
    input_path = args["input_path"]
    output_path = args["output_path"]
    width = args["width"]
    height = args["height"]
    mode = args.get("mode", "crop")

    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Build filter based on mode
        if mode == "crop":
            # Scale to fill frame, then crop to exact size
            vf_filter = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
        elif mode == "pad":
            # Scale to fit within frame, then pad with black bars
            vf_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        else:  # stretch
            # Force exact dimensions (may distort)
            vf_filter = f"scale={width}:{height}"

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", vf_filter,
            "-c:v", "libx264",  # Re-encode with h264
            "-preset", "fast",
            "-c:a", "copy",  # Copy audio without re-encoding
            str(output_path)
        ]

        result = run_ffmpeg_command(ffmpeg_cmd, timeout=600)  # Longer timeout for encoding

        if result.success:
            result.output_path = output_path

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result.model_dump(), indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps(EditResult(success=False, error=str(e)).model_dump())
            }]
        }
