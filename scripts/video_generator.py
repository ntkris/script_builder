import os
import sys
import subprocess
import base64
import tempfile
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from dotenv import load_dotenv
import anthropic
from pydantic import BaseModel, Field
import xmltodict

# Add parent directory to path for utils import
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import save_json, load_json, TokenTracker

load_dotenv()

# AI Video Editor Configuration
TARGET_DURATION = 10.0  # seconds - target length of final video
MIN_CLIP_LENGTH = 2.0   # seconds - minimum clip duration
MAX_CLIP_LENGTH = 8.0   # seconds - maximum clip duration

# Pydantic Models
class KeyElement(BaseModel):
    element: str

class VideoFrame(BaseModel):
    timestamp: str
    description: str
    scene_type: str
    notable_changes: Optional[str] = ""
    key_elements: List[str] = Field(default_factory=list)
    video_id: str = ""  # Store UUID instead of full path

class VideoAnalysis(BaseModel):
    overall_summary: str
    frames: List[VideoFrame] = Field(default_factory=list)

class VideoAnalysisResult(BaseModel):
    video_path: str
    frames_analyzed: int
    frame_interval: float
    analysis: Optional[VideoAnalysis] = None
    error: Optional[str] = None
    success: bool

class ProcessingStep(BaseModel):
    step_number: int
    description: str
    ffmpeg_command: str
    input_file: str
    output_file: str
    timeframe: str
    reasoning: str

class QualitySettings(BaseModel):
    video_codec: str = ""
    audio_codec: str = ""
    resolution: str = ""
    bitrate: str = ""

class ExecutionPlan(BaseModel):
    task_summary: str
    complexity: str
    estimated_duration: str
    final_output: str
    steps: List[ProcessingStep] = Field(default_factory=list)
    potential_issues: List[str] = Field(default_factory=list)
    quality_settings: QualitySettings = Field(default_factory=QualitySettings)

class ExecutionPlanResult(BaseModel):
    success: bool
    plan: Optional[ExecutionPlan] = None
    video_path: str = ""
    user_prompt: str = ""
    error: Optional[str] = None

class SelectedClip(BaseModel):
    start_time: float
    end_time: float
    duration: float
    reason: str
    engagement_score: float
    video_id: str = ""  # Store UUID instead of full path
    source_frames: List[str] = Field(default_factory=list)

class ClipSelection(BaseModel):
    selected_clips: List[SelectedClip] = Field(default_factory=list)
    total_duration: float
    target_duration: float

# Global token tracking
token_tracker = TokenTracker()

def log_error_and_exit(message: str, exit_code: int = 1):
    """Simple error logging that exits immediately"""
    print(f"❌ {message}")
    sys.exit(exit_code)




def simple_parse_xml(xml_string: str, root_element: str) -> dict:
    """Simplified XML parsing using xmltodict - fails fast on errors"""
    # Clean the XML string
    xml_string = xml_string.strip()
    if not xml_string.startswith('<'):
        # Find the first < character
        start_idx = xml_string.find('<')
        if start_idx != -1:
            xml_string = xml_string[start_idx:]
    
    # Parse XML to dict using xmltodict
    parsed_dict = xmltodict.parse(xml_string)
    
    # Extract the root element content
    if root_element not in parsed_dict:
        raise ValueError(f"Root element '{root_element}' not found in XML")
        
    return parsed_dict[root_element]

def extract_frames_from_video(video_path: str, output_dir: str, interval_seconds: float = 5, max_frames: int = 10) -> List[Path]:
    """Extract frames from video at specified intervals using FFmpeg"""
    frames_dir = Path(output_dir) / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Get video duration
    duration_cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
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
    
    print(f"Extracting {frames_to_extract} frames at {interval_seconds}s intervals...")
    
    frame_paths = []
    for i in range(frames_to_extract):
        timestamp = i * interval_seconds
        if timestamp >= duration:
            break
            
        frame_path = frames_dir / f"frame_{i:03d}_{timestamp:.1f}s.jpg"
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-ss', str(timestamp), '-i', str(video_path),
            '-vframes', '1', '-q:v', '2', str(frame_path)
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            frame_paths.append(frame_path)
        else:
            print(f"  ⚠️ Failed frame at {timestamp:.1f}s")
    
    return frame_paths

def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 for Claude API"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_video_frames(video_path: str, frame_interval: float = 5, max_frames: int = 10, video_id: str = "") -> VideoAnalysisResult:
    """Analyze video by extracting frames and using Claude's vision capabilities"""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    print(f"\n🔍 Analyzing: {Path(video_path).name}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        frame_paths = extract_frames_from_video(video_path, temp_dir, frame_interval, max_frames)
        
        if not frame_paths:
            return VideoAnalysisResult(
                video_path=str(video_path),
                frames_analyzed=0,
                frame_interval=frame_interval,
                error="Failed to extract frames",
                success=False
            )
        
        analysis_prompt = """Analyze this video by examining the extracted frames. Return your analysis as XML:

<video_analysis>
  <overall_summary>Brief description of the video content</overall_summary>
  <frames>
    <frame>
      <timestamp>X.Xs</timestamp>
      <description>What's happening in this frame</description>
      <key_elements>
        <element>object or person</element>
      </key_elements>
      <scene_type>scene description</scene_type>
      <notable_changes>Changes from previous frame</notable_changes>
    </frame>
  </frames>
</video_analysis>"""
        
        # Prepare content with images
        content = [{"type": "text", "text": analysis_prompt}]
        
        for i, frame_path in enumerate(frame_paths):
            base64_image = encode_image_to_base64(frame_path)
            timestamp = frame_path.stem.split('_')[-1]
            content.extend([
                {"type": "text", "text": f"\nFrame {i+1} (at {timestamp}):"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ])
        
        print("🤖 Analyzing with Claude...")
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}]
        )
        
        token_tracker.track("Video Analysis", message)
        
        # Print raw LLM output
        raw_response = message.content[0].text.strip()
        print(f"\n🤖 RAW VIDEO ANALYSIS RESPONSE:")
        print("="*80)
        print(raw_response)
        print("="*80)
        
        # Parse response
        xml_data = simple_parse_xml(raw_response, "video_analysis")
        
        if not xml_data:
            return VideoAnalysisResult(
                video_path=str(video_path),
                frames_analyzed=len(frame_paths),
                frame_interval=frame_interval,
                error="Failed to parse analysis response",
                success=False
            )
        
        # Build analysis from parsed data
        frames = []
        frames_data = xml_data.get('frames', {})
        frame_list = frames_data.get('frame', [])
        
        # Normalize single frame to list
        if isinstance(frame_list, dict):
            frame_list = [frame_list]
        
        for frame_data in frame_list:
            # Skip non-dict items
            if not isinstance(frame_data, dict):
                continue
                
            # Handle key elements
            key_elements = []
            elements_data = frame_data.get('key_elements', {})
            element_list = elements_data.get('element', [])
            
            # Normalize single element to list
            if isinstance(element_list, str):
                key_elements = [element_list]
            else:
                key_elements = element_list
            
            frames.append(VideoFrame(
                timestamp=frame_data.get('timestamp', ''),
                description=frame_data.get('description', ''),
                scene_type=frame_data.get('scene_type', ''),
                notable_changes=frame_data.get('notable_changes', ''),
                key_elements=key_elements,
                video_id=video_id
            ))
        
        analysis = VideoAnalysis(
            overall_summary=xml_data.get('overall_summary', ''),
            frames=frames
        )
        
        result = VideoAnalysisResult(
            video_path=str(video_path),
            frames_analyzed=len(frame_paths),
            frame_interval=frame_interval,
            analysis=analysis,
            success=True
        )
        
        print("✅ Analysis completed!")
        return result

def select_clips_from_analysis(analysis_result: VideoAnalysisResult, target_duration: float, video_id_map: Dict[str, str], min_clip_length: float = 2.0, max_clip_length: float = 8.0) -> ClipSelection:
    """Select optimal clips from video analysis using Claude Sonnet 4"""
    if not analysis_result.success or not analysis_result.analysis:
        return ClipSelection(
            total_duration=0.0,
            target_duration=target_duration
        )
    
    print(f"\n🎯 Using Claude to select clips for {target_duration}s target duration...")
    
    frames = analysis_result.analysis.frames
    if not frames:
        return ClipSelection(
            total_duration=0.0,
            target_duration=target_duration
        )
    
    # Prepare analysis data for Claude
    frames_summary = []
    for frame in frames:
        frames_summary.append({
            "timestamp": frame.timestamp,
            "description": frame.description,
            "scene_type": frame.scene_type,
            "notable_changes": frame.notable_changes,
            "key_elements": frame.key_elements,
            "video_id": frame.video_id
        })
    
    # Create video ID reference for Claude
    video_id_reference = {vid: Path(path).name for vid, path in video_id_map.items()}
    
    clip_selection_prompt = f"""You are an expert video editor. Analyze this video frame data and select the best clips for a {target_duration}-second highlight reel.

VIDEO ANALYSIS:
Summary: {analysis_result.analysis.overall_summary}

VIDEO ID REFERENCE:
{json.dumps(video_id_reference, indent=2)}

FRAME DATA:
{json.dumps(frames_summary, indent=2)}

CONSTRAINTS:
- Target duration: {target_duration} seconds
- Min clip length: {min_clip_length} seconds  
- Max clip length: {max_clip_length} seconds
- Clips cannot overlap
- Select clips that capture the most engaging/dynamic moments

Return your selection as XML:
<clip_selection>
  <clip>
    <start_time>X.X</start_time>
    <end_time>X.X</end_time>
    <duration>X.X</duration>
    <reason>Why this clip was selected</reason>
    <engagement_score>X.X</engagement_score>
    <video_id>the video_id from the frames</video_id>
    <source_frames>timestamps of key frames</source_frames>
  </clip>
</clip_selection>

Look at the video_id field in each frame to determine which video the clip comes from.
Focus on action, movement, technique, and dynamic moments. Avoid static positioning or setup time."""

    # Call Claude
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    print("🤖 Asking Claude to select optimal clips...")
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": clip_selection_prompt}]
    )
    
    token_tracker.track("Clip Selection", message)
    
    # Print raw LLM output
    raw_response = message.content[0].text.strip()
    print(f"\n🤖 RAW CLIP SELECTION RESPONSE:")
    print("="*80)
    print(raw_response)
    print("="*80)
    
    # Parse Claude's response
    xml_data = simple_parse_xml(raw_response, "clip_selection")
    
    # Extract clips from Claude's response
    selected_clips = []
    clip_data = xml_data.get('clip', [])
    
    for clip_info in clip_data:
        if not isinstance(clip_info, dict):
            continue
            
        try:
            start_time = float(clip_info.get('start_time', 0))
            end_time = float(clip_info.get('end_time', 0))
            duration = float(clip_info.get('duration', 0))
            
            # Validate clip bounds
            if end_time <= start_time or duration <= 0:
                continue
                
            source_frames_text = clip_info.get('source_frames', '')
            source_frames = [source_frames_text] if source_frames_text else []
            
            selected_clips.append(SelectedClip(
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                reason=clip_info.get('reason', 'Claude-selected clip'),
                engagement_score=float(clip_info.get('engagement_score', 5.0)),
                video_id=clip_info.get('video_id', ''),
                source_frames=source_frames
            ))
            
            print(f"  ✓ Claude selected: {start_time:.1f}s-{end_time:.1f}s ({duration:.1f}s)")
            
        except (ValueError, TypeError) as e:
            print(f"  ⚠️ Skipped malformed clip: {e}")
            continue
    
    total_duration = sum(clip.duration for clip in selected_clips)
    
    result = ClipSelection(
        selected_clips=selected_clips,
        total_duration=total_duration,
        target_duration=target_duration
    )
    
    print(f"🎬 Claude selected {len(selected_clips)} clips totaling {total_duration:.1f}s (target: {target_duration}s)")
    
    return result

def generate_execution_plan(video_path: str, user_prompt: str, analysis_data: VideoAnalysisResult, clip_selection: ClipSelection, video_id_map: Dict[str, str]) -> ExecutionPlanResult:
    """Generate simple FFmpeg command from clips"""
    clips = clip_selection.selected_clips
    if not clips:
        return ExecutionPlanResult(success=False, error="No clips selected")
    
    # Build FFmpeg command
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/final_combined_{timestamp}.mp4"
    
    inputs = []
    filters = []
    
    for i, clip in enumerate(clips):
        video_file = video_id_map.get(clip.video_id, "inputs/tennis_short.mov")
        inputs.append(f'-i "{video_file}"')
        filters.append(f'[{i}:v]trim=start={clip.start_time}:duration={clip.duration},setpts=PTS-STARTPTS[v{i}]')
        filters.append(f'[{i}:a]atrim=start={clip.start_time}:duration={clip.duration},asetpts=PTS-STARTPTS[a{i}]')
    
    # Concatenate all clips - inputs must be in [v0][a0][v1][a1]... format
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    filters.append(f'{concat_inputs}concat=n={len(clips)}:v=1:a=1[outv][outa]')
    
    ffmpeg_cmd = f'ffmpeg -y {" ".join(inputs)} -filter_complex "{";".join(filters)}" -map "[outv]" -map "[outa]" -c:v libx264 -c:a aac "{output_file}"'
    
    return ExecutionPlanResult(
        success=True,
        plan=ExecutionPlan(
            task_summary=f"Extract {len(clips)} clips",
            steps=[ProcessingStep(step_number=1, description="Extract clips", ffmpeg_command=ffmpeg_cmd, input_file="", output_file=output_file, timeframe="", reasoning="")],
            complexity="simple",
            estimated_duration="fast",
            final_output=output_file
        )
    )

def display_results(analysis_result: Optional[VideoAnalysisResult] = None, plan_result: Optional[ExecutionPlanResult] = None, clip_selection: Optional[ClipSelection] = None) -> None:
    """Unified display function for analysis and execution plan results"""
    
    if analysis_result and analysis_result.success and analysis_result.analysis:
        analysis = analysis_result.analysis
        print(f"\n📊 ANALYSIS RESULTS")
        print("="*50)
        print(f"🎬 {analysis.overall_summary}")
        
        if analysis.frames:
            print(f"\n🖼️ Frame Analysis ({len(analysis.frames)} frames):")
            for i, frame in enumerate(analysis.frames[:3], 1):  # Show first 3 frames
                print(f"  {frame.timestamp}: {frame.description}")
        print("="*50)
    
    if clip_selection and clip_selection.selected_clips:
        print(f"\n🎯 SELECTED CLIPS")
        print("="*50)
        print(f"🎬 Target Duration: {clip_selection.target_duration}s")
        print(f"📏 Actual Duration: {clip_selection.total_duration:.1f}s")
        
        print(f"\n🎞️ Selected Clips ({len(clip_selection.selected_clips)}):")
        for i, clip in enumerate(clip_selection.selected_clips, 1):
            print(f"  {i}. {clip.start_time:.1f}s-{clip.end_time:.1f}s ({clip.duration:.1f}s)")
            print(f"     Score: {clip.engagement_score:.1f} | {clip.reason}")
        print("="*50)
    
    if plan_result and plan_result.success and plan_result.plan:
        plan = plan_result.plan
        print(f"\n🎯 EXECUTION PLAN")
        print("="*50)
        print(f"📋 {plan.task_summary}")
        print(f"⚡ Complexity: {plan.complexity.upper()}")
        print(f"⏱️ Duration: {plan.estimated_duration}")
        
        print(f"\n📝 Steps ({len(plan.steps)}):")
        for step in plan.steps:
            print(f"  {step.step_number}. {step.description}")
            print(f"     → {step.ffmpeg_command}")
        
        print(f"\n🎬 Result: {plan.final_output}")
        
        if plan.potential_issues:
            print(f"\n⚠️ Issues: {', '.join(plan.potential_issues)}")
        print("="*50)

def execute_processing_plan(plan_result: ExecutionPlanResult) -> bool:
    """Execute the processing plan without user confirmations"""
    if not plan_result.success or not plan_result.plan:
        log_error_and_exit("Cannot execute invalid plan")
    
    steps = plan_result.plan.steps
    if not steps:
        log_error_and_exit("No processing steps found")
    
    print(f"\n🚀 Executing {len(steps)} step(s)...")
    
    completed_outputs = []
    
    for i, step in enumerate(steps, 1):
        print(f"\n📋 Step {i}/{len(steps)}: {step.description}")
        
        if not step.ffmpeg_command:
            print(f"❌ No command for step {i}")
            continue
        
        print(f"🔄 Running...")
        
        result = subprocess.run(
            step.ffmpeg_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=1200
        )
        
        if result.returncode == 0:
            print(f"✅ Step {i} completed")
            if step.output_file and Path(step.output_file).exists():
                completed_outputs.append(step.output_file)
        else:
            print(f"❌ Step {i} failed: {result.stderr}")
            return False
    
    print(f"\n🎉 Processing completed! Generated {len(completed_outputs)} file(s):")
    for output in completed_outputs:
        print(f"  📁 {output}")
    
    return True

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        log_error_and_exit("ANTHROPIC_API_KEY not found. Create .env file with your API key")
    
    # Configuration - now supports multiple input files
    INPUT_FILES = [
        "inputs/recipe_1.mov",
        "inputs/recipe_2.mov",
    ]
    FRAME_INTERVAL = 10.0
    MAX_FRAMES = 10
    USER_PROMPT = "Create a 10 second highlight reel of the most interesting moments from all videos"
    
    # Check input files exist
    valid_files = []
    for input_file in INPUT_FILES:
        if Path(input_file).exists():
            valid_files.append(input_file)
            print(f"✅ Found: {input_file}")
        else:
            print(f"⚠️ Not found: {input_file}")
    
    if not valid_files:
        log_error_and_exit("No valid input files found")
    
    INPUT_FILES = valid_files
    print(f"\n🎬 Processing {len(INPUT_FILES)} video(s)")
    
    # Create video ID map
    video_id_map: Dict[str, str] = {}  # video_id -> video_path
    path_to_id_map: Dict[str, str] = {}  # video_path -> video_id
    
    for video_path in INPUT_FILES:
        video_id = str(uuid.uuid4())[:8]  # Short UUID
        video_id_map[video_id] = video_path
        path_to_id_map[video_path] = video_id
        print(f"🆔 {Path(video_path).name} -> {video_id}")
    
    # Create outputs directory
    Path("outputs").mkdir(exist_ok=True)
    
    # Step 1: Analyze all videos
    all_analyses = []
    for i, input_file in enumerate(INPUT_FILES, 1):
        print(f"\n📊 Analyzing video {i}/{len(INPUT_FILES)}: {Path(input_file).name}")
        analysis_result = analyze_video_frames(input_file, FRAME_INTERVAL, MAX_FRAMES, path_to_id_map[input_file])
        
        if not analysis_result.success:
            print(f"❌ Analysis failed for {input_file}: {analysis_result.error}")
            continue
            
        all_analyses.append(analysis_result)
    
    if not all_analyses:
        log_error_and_exit("No videos could be analyzed")
    
    print(f"✅ Successfully analyzed {len(all_analyses)}/{len(INPUT_FILES)} videos")
    
    # Step 2: Combine all analysis results into one for clip selection
    combined_frames = []
    for analysis in all_analyses:
        if analysis.analysis and analysis.analysis.frames:
            combined_frames.extend(analysis.analysis.frames)
    
    # Create combined analysis result for clip selection
    combined_summary = " | ".join([a.analysis.overall_summary for a in all_analyses if a.analysis])
    
    combined_analysis = VideoAnalysisResult(
        video_path="combined_videos",
        frames_analyzed=sum(a.frames_analyzed for a in all_analyses),
        frame_interval=FRAME_INTERVAL,
        analysis=VideoAnalysis(
            overall_summary=f"Combined analysis of {len(all_analyses)} videos: {combined_summary}",
            frames=combined_frames
        ),
        success=True
    )
    
    # Step 3: Select clips from combined analysis  
    clip_selection = select_clips_from_analysis(combined_analysis, TARGET_DURATION, video_id_map, MIN_CLIP_LENGTH, MAX_CLIP_LENGTH)
    
    if not clip_selection.selected_clips:
        log_error_and_exit("No clips could be selected from the combined analysis")
    
    # Step 4: Generate execution plan for multi-video stitching
    plan_result = generate_execution_plan("combined_videos", USER_PROMPT, combined_analysis, clip_selection, video_id_map)
    
    if not plan_result.success:
        log_error_and_exit(f"Plan generation failed: {plan_result.error}")
    
    # Step 5: Display results
    display_results(combined_analysis, plan_result, clip_selection)
    
    # Step 6: Save combined results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual analyses
    for analysis in all_analyses:
        video_name = Path(analysis.video_path).stem
        analysis_data = {
            "generated_at": datetime.now().isoformat(),
            "video_analysis": analysis.model_dump()
        }
        save_json(analysis_data, f"analysis_{video_name}_{timestamp}.json", output_dir="cache", description=f"Analysis ({video_name})")
    
    # Save combined analysis
    combined_data = {
        "generated_at": datetime.now().isoformat(),
        "combined_analysis": combined_analysis.model_dump(),
        "source_videos": [a.video_path for a in all_analyses]
    }
    save_json(combined_data, f"combined_analysis_{timestamp}.json", output_dir="cache", description="Combined Analysis")
    
    # Save clip selection
    clip_data = {
        "generated_at": datetime.now().isoformat(),
        "clip_selection": clip_selection.model_dump()
    }
    save_json(clip_data, f"clips_combined_{timestamp}.json", output_dir="cache", description="Clip Selection")
    
    # Save execution plan
    plan_data = {
        "generated_at": datetime.now().isoformat(),
        "execution_plan": plan_result.model_dump()
    }
    save_json(plan_data, f"plan_combined_{timestamp}.json", output_dir="cache", description="Plan")
    
    # Step 7: Execute plan automatically
    success = execute_processing_plan(plan_result)
    
    # Save token consumption
    token_tracker.save_summary("combined_videos", output_dir="cache", user_prompt=USER_PROMPT)
    
    if not success:
        log_error_and_exit("Processing failed")
    
    print("\n🎉 Multi-video session completed!")

if __name__ == "__main__":
    main()
