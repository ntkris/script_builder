#!/usr/bin/env python3
"""
Floorplan Object Detection Benchmark - Multi-Model Comparison

Tests whether vision models can detect objects that human annotators
marked in the FloorPlanCAD dataset. Focuses on detection rate (presence/absence)
rather than count accuracy, acknowledging that ground truth is incomplete.

Supports: Gemini Flash 2.5, Gemini Pro 2.5, Claude Sonnet 4.5, Claude Haiku 4.5, GPT-5, GPT-5-mini
"""

import argparse
import random
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from utils import (
    call_gemini, call_anthropic, call_openai,
    AIRequest, GeminiModel, AnthropicModel, OpenAIModel,
    ReasoningEffort, Verbosity, save_json
)
from utils.step_logger import StepLogger
load_dotenv()

# Pydantic models for structured data
class ObjectCount(BaseModel):
    """Count of a specific object category"""
    category: str
    count: int
    confidence: Optional[str] = Field(default="medium", description="high, medium, or low")


class FloorplanAnalysis(BaseModel):
    """Vision model's structured analysis of a floorplan"""
    object_counts: List[ObjectCount] = Field(description="List of object counts by category")
    total_objects_detected: int = Field(description="Sum of all objects found")
    analysis_notes: Optional[str] = Field(default=None, description="Any observations or uncertainties")


class GroundTruthAnnotation(BaseModel):
    """Parsed YOLO annotation"""
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float


class SampleResult(BaseModel):
    """Result for a single floorplan sample"""
    image_id: str
    image_path: str
    ground_truth_counts: Dict[str, int]
    model_counts: Dict[str, int]
    total_gt: int
    total_model: int
    categories_detected: int
    categories_in_gt: int
    error_message: Optional[str] = None


class CategoryMetrics(BaseModel):
    """Detection metrics for a specific object category"""
    category: str
    samples_with_gt: int = 0
    samples_detected: int = 0
    detection_rate: float = 0.0
    total_gt_count: int = 0
    total_model_count: int = 0


class ModelResult(BaseModel):
    """Results for a single model"""
    model_name: str
    model_id: str
    samples_processed: int
    samples_failed: int
    overall_detection_rate: float
    category_metrics: List[CategoryMetrics]
    sample_results: List[SampleResult]


# YOLO class mapping for FloorPlanCAD dataset (28 categories)
YOLO_CLASSES = {
    0: "door_single",
    1: "door_double",
    2: "door_sliding",
    3: "window",
    4: "window_bay",
    5: "window_blind",
    6: "stairs",
    7: "elevator",
    8: "escalator",
    9: "sofa",
    10: "bed",
    11: "table",
    12: "chair",
    13: "sink",
    14: "toilet",
    15: "bath",
    16: "shower",
    17: "stove",
    18: "refrigerator",
    19: "washer",
    20: "dryer",
    21: "dishwasher",
    22: "wardrobe",
    23: "cabinet",
    24: "tv_stand",
    25: "desk",
    26: "bookshelf",
    27: "plant"
}

# Model configurations: (display_name, provider, model_enum)
ALL_MODELS = [
    ("gemini-2.5-flash", "gemini", GeminiModel.GEMINI_2_5_FLASH),
    ("gemini-2.5-pro", "gemini", GeminiModel.GEMINI_2_5_PRO),
    ("sonnet-4.5", "anthropic", AnthropicModel.CLAUDE_SONNET_4_5),
    ("haiku-4.5", "anthropic", AnthropicModel.CLAUDE_HAIKU_4_5),
    ("gpt-5", "openai", OpenAIModel.GPT_5),
    ("gpt-5-mini", "openai", OpenAIModel.GPT_5_MINI)
]


def parse_yolo_annotation(annotation_path: Path, class_mapping: Dict[int, str]) -> List[GroundTruthAnnotation]:
    """Parse YOLO format annotation file"""
    annotations = []

    if not annotation_path.exists():
        return annotations

    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_name = class_mapping.get(class_id, f"unknown_{class_id}")

                annotations.append(GroundTruthAnnotation(
                    class_id=class_id,
                    class_name=class_name,
                    x_center=float(parts[1]),
                    y_center=float(parts[2]),
                    width=float(parts[3]),
                    height=float(parts[4])
                ))

    return annotations


def count_objects_by_category(annotations: List[GroundTruthAnnotation]) -> Dict[str, int]:
    """Count objects by category from annotations"""
    counts = {}
    for ann in annotations:
        counts[ann.class_name] = counts.get(ann.class_name, 0) + 1
    return counts


def encode_image_to_base64(image_path: Path) -> str:
    """Encode image to base64 for vision models"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def analyze_floorplan_with_vision(
    image_path: Path,
    logger: StepLogger,
    step_name: str,
    provider: str,
    model_enum: str
) -> FloorplanAnalysis:
    """Use vision model with structured extraction to count objects in floorplan"""

    # Encode image
    image_b64 = encode_image_to_base64(image_path)

    # Create prompt with all categories
    categories_list = "\n".join([f"- {name}" for name in sorted(set(YOLO_CLASSES.values()))])

    prompt = f"""You are analyzing an ARCHITECTURAL CAD DRAWING (technical floorplan), not a photograph.

IMPORTANT: This is a 2D architectural plan view with standardized symbols and line work. Objects are represented as:
- Line drawings and geometric shapes
- Standardized architectural symbols
- Abstract representations (not realistic)
- Often colored lines (walls, fixtures, etc.)

LOOK FOR THESE CAD SYMBOLS AND REPRESENTATIONS:

DOORS:
- door_single: Arc/curved line showing door swing, single leaf
- door_double: Two arc lines side by side (French doors)
- door_sliding: Parallel lines with directional arrow

WINDOWS:
- window: Double parallel lines within walls, may have cross hatching
- window_bay: Angled projection from wall
- window_blind: Window symbol with horizontal lines

CIRCULATION:
- stairs: Parallel diagonal lines with arrow, may be hatched
- elevator: Square box labeled "E", "ELEV", or elevator icon
- escalator: Diagonal parallel lines with mechanical symbol

FURNITURE (often shown as simple shapes):
- sofa: Rectangular outline, may show cushion divisions
- bed: Rectangle with pillow end indicated
- table: Circle/rectangle outline
- chair: Small rectangle or simplified chair shape
- desk: Rectangle against wall
- bookshelf: Rectangle with line divisions
- tv_stand: Low rectangle
- wardrobe: Large rectangle, may show doors
- cabinet: Rectangle outline, often in kitchen

FIXTURES (plumbing shown with specific symbols):
- toilet: Oval or rounded rectangle, may be labeled "WC"
- sink: Small oval/rectangle at counter
- bath: Large rectangle/oval for bathtub
- shower: Square with drain symbol

APPLIANCES (kitchen symbols):
- stove/range: Square with 4 circles (burners)
- refrigerator: Rectangle, often larger
- washer: Circle or square, may show drum
- dryer: Similar to washer
- dishwasher: Rectangle in kitchen counter

DECORATIVE:
- plant: Circle with organic shape or leaf symbol

Count objects in these categories:
{categories_list}

RULES:
1. Only count clearly visible symbols/objects
2. If uncertain or symbol is unclear, SKIP that category
3. Use exact category names (with underscores)
4. If you see 0 of a category, omit it from object_counts array
5. Provide confidence: "high" if clear, "medium" if uncertain, "low" if guessing

OUTPUT FORMAT - Respond ONLY with valid JSON matching this structure:
{{
  "object_counts": [
    {{"category": "door_single", "count": 2, "confidence": "high"}},
    {{"category": "window", "count": 4, "confidence": "medium"}}
  ],
  "total_objects_detected": 6,
  "analysis_notes": "Any observations"
}}"""

    # Prepare multipart message (text + image)
    message_content = [
        {"type": "text", "text": prompt},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": image_b64
            }
        }
    ]

    # Route to appropriate provider
    if provider == "gemini":
        request = AIRequest(
            messages=[{"role": "user", "content": message_content}],
            model=model_enum,
            max_tokens=8000,
            temperature=0.1,
            json_mode=True,
            response_schema=FloorplanAnalysis,
            step_name=step_name
        )
        response = call_gemini(request, logger)

    elif provider == "anthropic":
        request = AIRequest(
            messages=[{"role": "user", "content": message_content}],
            model=model_enum,
            max_tokens=4096,
            temperature=0.1,
            json_mode=True,
            response_schema=FloorplanAnalysis,
            step_name=step_name
        )
        response = call_anthropic(request, logger)

    elif provider == "openai":
        request = AIRequest(
            messages=[{"role": "user", "content": message_content}],
            model=model_enum,
            max_tokens=4096,
            temperature=0.1,
            json_mode=True,
            response_schema=FloorplanAnalysis,
            reasoning_effort=ReasoningEffort.MINIMAL,  # Fast, lightweight classification
            verbosity=Verbosity.LOW,  # Concise JSON output
            step_name=step_name
        )
        response = call_openai(request, logger)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Debug: Check if we got content
    if not response.content or response.content.strip() == "":
        print(f"   ⚠️  Empty response from {provider}")
        print(f"      Finish reason: {response.finish_reason}")
        print(f"      Tokens: {response.input_tokens}→{response.output_tokens}")
        return FloorplanAnalysis(
            object_counts=[],
            total_objects_detected=0,
            analysis_notes=f"Empty response, finish_reason: {response.finish_reason}"
        )

    # Parse and validate JSON response
    try:
        # Clean up response (might have markdown code blocks)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)

        # Validate against schema
        analysis = FloorplanAnalysis.model_validate(data)
        return analysis

    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON parsing error: {e}")
        print(f"      First 200 chars: {response.content[:200]}")
        return FloorplanAnalysis(
            object_counts=[],
            total_objects_detected=0,
            analysis_notes=f"JSON parse error: {str(e)}"
        )
    except Exception as e:
        print(f"   ⚠️  Validation error: {e}")
        return FloorplanAnalysis(
            object_counts=[],
            total_objects_detected=0,
            analysis_notes=f"Validation error: {str(e)}"
        )


def calculate_category_metrics(results: List[SampleResult]) -> List[CategoryMetrics]:
    """Calculate detection metrics per category - focus on presence/absence"""

    # Collect all categories that appear in ground truth
    gt_categories = set()
    for result in results:
        gt_categories.update(result.ground_truth_counts.keys())

    metrics_list = []

    for category in sorted(gt_categories):
        samples_with_gt = 0
        samples_detected = 0
        total_gt = 0
        total_model = 0

        for result in results:
            gt_count = result.ground_truth_counts.get(category, 0)
            model_count = result.model_counts.get(category, 0)

            # Count samples where GT has this category
            if gt_count > 0:
                samples_with_gt += 1
                total_gt += gt_count

                # Did model detect it (any count > 0)?
                if model_count > 0:
                    samples_detected += 1
                    total_model += model_count

        # Calculate detection rate
        detection_rate = samples_detected / samples_with_gt if samples_with_gt > 0 else 0.0

        metrics_list.append(CategoryMetrics(
            category=category,
            samples_with_gt=samples_with_gt,
            samples_detected=samples_detected,
            detection_rate=detection_rate,
            total_gt_count=total_gt,
            total_model_count=total_model
        ))

    # Sort by detection rate (worst first for identifying weaknesses)
    metrics_list.sort(key=lambda x: (x.detection_rate, -x.samples_with_gt))

    return metrics_list


def calculate_overall_detection_rate(results: List[SampleResult]) -> float:
    """Calculate overall detection rate across all samples"""
    total_gt_instances = 0
    total_detected = 0

    for result in results:
        for category in result.ground_truth_counts:
            total_gt_instances += 1
            if result.model_counts.get(category, 0) > 0:
                total_detected += 1

    return total_detected / total_gt_instances if total_gt_instances > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model benchmark for floorplan object detection"
    )
    parser.add_argument("--dataset-path", type=str, default="inputs/floorplans_dataset",
                       help="Path to FloorPlanCAD dataset (default: inputs/floorplans_dataset)")
    parser.add_argument("--num-samples", type=int, default=2,
                       help="Number of random samples (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--all", action="store_true",
                       help="Run all 6 models (gemini-2.5-flash, gemini-2.5-pro, sonnet-4.5, haiku-4.5, gpt-5, gpt-5-mini)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                       choices=["gemini-2.5-flash", "gemini-2.5-pro", "sonnet-4.5", "haiku-4.5", "gpt-5", "gpt-5-mini"],
                       help="Single model to test (default: gemini-2.5-flash)")

    args = parser.parse_args()

    # Initialize
    logger = StepLogger("floorplan_benchmark")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("🔍 Floorplan Object Detection Benchmark")
    print("📊 Testing {args.num_samples} random samples")
    if args.all:
        print("🤖 Running ALL 6 models")
    else:
        print(f"🤖 Running single model: {args.model}")
    print("🎯 Goal: Measure detection rate for ground truth objects\n")

    # Step 1: Find and select samples WITH ANNOTATIONS
    logger.step("Find Dataset Samples", inputs={
        "dataset_path": args.dataset_path,
        "num_samples": args.num_samples,
        "seed": args.seed
    })

    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        print("\nTo get the dataset:")
        print("1. Download from: https://www.kaggle.com/datasets/samirshabani/architecture")
        print("2. Extract to inputs/floorplans_dataset/")
        return

    # Find images and labels
    image_dir = dataset_path / "images"
    label_dir = dataset_path / "labels"

    if not image_dir.exists():
        image_dir = dataset_path / "train" / "images"
        label_dir = dataset_path / "train" / "labels"

    if not image_dir.exists():
        image_dir = dataset_path
        label_dir = dataset_path

    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return

    print(f"📁 Found {len(image_files)} total images")

    # Randomly select and filter for annotations
    random.seed(args.seed)
    candidate_samples = random.sample(image_files, min(args.num_samples * 3, len(image_files)))

    samples_with_annotations = []
    for img_path in candidate_samples:
        annotation_path = label_dir / f"{img_path.stem}.txt"
        if annotation_path.exists():
            annotations = parse_yolo_annotation(annotation_path, YOLO_CLASSES)
            if len(annotations) > 0:  # Only use samples with actual annotations
                samples_with_annotations.append({
                    "image_path": img_path,
                    "annotation_path": annotation_path,
                    "annotation_count": len(annotations)
                })
                if len(samples_with_annotations) >= args.num_samples:
                    break

    if len(samples_with_annotations) == 0:
        print("❌ No samples with annotations found")
        return

    print("✅ Selected {len(samples_with_annotations)} samples with annotations")
    print("   (Filtered out samples with empty annotation files)\n")

    logger.output({
        "total_images": len(image_files),
        "selected_samples": len(samples_with_annotations),
        "samples_with_annotations": len(samples_with_annotations)
    })

    # Determine which models to run
    if args.all:
        models_to_run = ALL_MODELS
    else:
        # Find the matching model config
        model_config = next((m for m in ALL_MODELS if m[0] == args.model), None)
        if not model_config:
            print(f"❌ Unknown model: {args.model}")
            return
        models_to_run = [model_config]

    # Step 2: Run benchmark for each model
    all_model_results = {}

    for model_name, provider, model_enum in models_to_run:
        print(f"\n{'='*70}")
        print(f"🤖 Testing Model: {model_name}")
        print(f"{'='*70}\n")

        logger.step(f"Benchmark {model_name}", inputs={
            "model": model_name,
            "provider": provider,
            "samples": len(samples_with_annotations)
        })

        sample_results = []
        failed_count = 0

        for i, sample in enumerate(samples_with_annotations):
            sample_id = sample["image_path"].stem
            print(f"🔍 [{i+1}/{len(samples_with_annotations)}] {sample_id}")

            try:
                # Parse ground truth
                annotations = parse_yolo_annotation(sample["annotation_path"], YOLO_CLASSES)
                gt_counts = count_objects_by_category(annotations)

                gt_categories = list(gt_counts.keys())
                print(f"   📋 GT: {', '.join(gt_categories)}")

                # Analyze with model
                analysis = analyze_floorplan_with_vision(
                    sample["image_path"],
                    logger,
                    step_name=f"{model_name}:{sample_id}",
                    provider=provider,
                    model_enum=model_enum
                )

                model_counts = {
                    oc.category: oc.count
                    for oc in analysis.object_counts
                    if oc.count > 0
                }

                # Check detection
                detected = [cat for cat in gt_counts if model_counts.get(cat, 0) > 0]
                missed = [cat for cat in gt_counts if model_counts.get(cat, 0) == 0]

                detection_rate = len(detected) / len(gt_counts) if gt_counts else 0.0
                status = "✅" if detection_rate == 1.0 else "⚠️" if detection_rate > 0 else "🚨"
                print(f"   {status} Detected: {len(detected)}/{len(gt_counts)} ({detection_rate:.0%})")
                if missed:
                    print(f"      Missed: {', '.join(missed)}")
                print()

                # Store result
                sample_results.append(SampleResult(
                    image_id=sample_id,
                    image_path=str(sample["image_path"]),
                    ground_truth_counts=gt_counts,
                    model_counts=model_counts,
                    total_gt=sum(gt_counts.values()),
                    total_model=sum(model_counts.values()),
                    categories_detected=len(model_counts),
                    categories_in_gt=len(gt_counts)
                ))

            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed_count += 1
                # Store failed result
                sample_results.append(SampleResult(
                    image_id=sample_id,
                    image_path=str(sample["image_path"]),
                    ground_truth_counts={},
                    model_counts={},
                    total_gt=0,
                    total_model=0,
                    categories_detected=0,
                    categories_in_gt=0,
                    error_message=str(e)
                ))
                print()
                continue

        # Calculate metrics for this model
        successful_results = [r for r in sample_results if r.error_message is None]

        if len(successful_results) > 0:
            category_metrics = calculate_category_metrics(successful_results)
            overall_rate = calculate_overall_detection_rate(successful_results)
        else:
            category_metrics = []
            overall_rate = 0.0

        model_result = ModelResult(
            model_name=model_name,
            model_id=str(model_enum),
            samples_processed=len(successful_results),
            samples_failed=failed_count,
            overall_detection_rate=overall_rate,
            category_metrics=category_metrics,
            sample_results=sample_results
        )

        all_model_results[model_name] = model_result

        logger.output({
            "model": model_name,
            "processed": len(successful_results),
            "failed": failed_count,
            "overall_detection_rate": overall_rate
        })

        print(f"✅ {model_name}: {overall_rate:.1%} detection rate ({len(successful_results)} samples)\n")

    # Step 3: Generate final benchmark report
    logger.step("Generate Benchmark Report")

    print("\n{'='*70}")
    print("📊 FINAL BENCHMARK RESULTS")
    print("{'='*70}\n")

    # Create comprehensive report
    report = {
        "metadata": {
            "timestamp": timestamp,
            "dataset_path": str(dataset_path),
            "num_samples": len(samples_with_annotations),
            "seed": args.seed,
            "models_tested": [m[0] for m in models_to_run],
            "test_type": "detection_rate",
            "description": "Multi-model benchmark comparing detection rates for ground truth objects"
        },
        "model_results": {
            model_name: result.model_dump()
            for model_name, result in all_model_results.items()
        },
        "summary": {
            "overall_rankings": sorted(
                [(name, result.overall_detection_rate) for name, result in all_model_results.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }
    }

    # Save to outputs (ALWAYS, even if something failed)
    report_filename = f"floorplan_benchmark_{timestamp}.json"
    try:
        save_json(report, report_filename, output_dir="outputs",
                 description="Multi-Model Floorplan Detection Benchmark")
        print(f"💾 Benchmark report saved: outputs/{report_filename}\n")
    except Exception as e:
        print(f"⚠️  Failed to save report: {e}")
        # Try to save to current directory as fallback
        try:
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"💾 Backup saved to: {report_filename}\n")
        except Exception as e:
            print("❌ Could not save report anywhere\n")

    # Print summary
    print("📊 Detection Rate by Model:")
    for model_name, rate in report["summary"]["overall_rankings"]:
        result = all_model_results[model_name]
        print(f"   {model_name:20s} | {rate:>5.1%} ({result.samples_processed}/{result.samples_processed + result.samples_failed} samples)")

    if len(all_model_results) > 1:
        best_model = report["summary"]["overall_rankings"][0][0]
        best_rate = report["summary"]["overall_rankings"][0][1]
        print(f"\n🏆 Best Overall: {best_model} ({best_rate:.1%})")

    # Print category analysis
    if len(all_model_results) > 0:
        print("\n📋 Category Performance Analysis:")

        # Collect all categories
        all_categories = set()
        for result in all_model_results.values():
            for metric in result.category_metrics:
                all_categories.add(metric.category)

        # Show top/bottom categories
        for model_name, result in all_model_results.items():
            if len(result.category_metrics) == 0:
                continue

            print(f"\n   {model_name}:")
            excellent = [m for m in result.category_metrics if m.detection_rate >= 0.75]
            poor = [m for m in result.category_metrics if m.detection_rate < 0.25]

            if excellent:
                print(f"      ✅ Strong: {', '.join(m.category for m in excellent[:5])}")
            if poor:
                print(f"      🚨 Weak: {', '.join(m.category for m in poor[:5])}")

    logger.output({"report_path": f"outputs/{report_filename}"})
    logger.finalize()

    print(f"\n{'='*70}")
    print("✅ Benchmark complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
