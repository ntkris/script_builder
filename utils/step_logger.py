"""Step logging with integrated token tracking for script execution"""

import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StepUpdate(BaseModel):
    """Single update within a step"""
    timestamp: str
    data: Dict[str, Any]


class TokenUsage(BaseModel):
    """Token usage counters"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class StepError(BaseModel):
    """Error details for a failed step"""
    type: str
    message: str
    traceback: str


class StepRecord(BaseModel):
    """Complete record of a single step"""
    step_number: int
    name: str
    status: str  # "in_progress", "success", "failed"
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    updates: List[StepUpdate] = Field(default_factory=list)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    error: Optional[StepError] = None


class StepLogSummary(BaseModel):
    """Summary of all steps in a script execution"""
    script_name: str
    timestamp: str
    steps: List[StepRecord] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)


class StepLogger:
    """
    Logger for tracking script execution steps with automatic token tracking.

    Replaces the standalone TokenTracker - all LLM calls pass the logger.

    Example:
        logger = StepLogger("tender_finder")

        logger.step("Generate Queries", inputs={"max": 6})
        queries = generate_queries(...)
        logger.output({"count": len(queries)})

        logger.step("Search", inputs={"queries": queries})
        for i, query in enumerate(queries):
            result = search(query)
            logger.update({"completed": i + 1})
        logger.output({"total": len(results)})

    Resumable Example:
        logger = StepLogger("tender_finder", resumable=True)

        logger.step("Load tenders")
        if logger.should_run_step():
            tenders = load_tenders()
            logger.output({"tenders": tenders})
        else:
            tenders = logger.get_cached_output("tenders")

        logger.step("Process tenders")
        if logger.should_run_step():
            for tender in tenders:
                if logger.is_item_completed(tender.url):
                    print(f"⏭️  Skipping {tender.url}")
                    continue
                result = process_tender(tender)
                logger.mark_item_complete(tender.url, result)
    """

    def __init__(self, script_name: str, output_dir: str = "cache", resumable: bool = False):
        self.script_name = script_name
        self.output_dir = output_dir
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.steps: List[StepRecord] = []
        self.current_step: Optional[StepRecord] = None
        self.step_counter = 0
        self.resumable = resumable

        # Token tracking for current step
        self._current_step_start_tokens = TokenUsage()

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # File path for incremental saves
        self.log_file = Path(output_dir) / f"step_log_{script_name}_{self.session_timestamp}.json"

        # Resume state tracking
        self.resume_state_file = Path(output_dir) / f"resume_state_{script_name}.json"
        self.completed_steps: List[str] = []
        self.step_outputs: Dict[str, Any] = {}
        self.completed_items: Dict[str, List[str]] = {}
        self.is_resuming = False

        # Load resume state if it exists
        if self.resumable and self.resume_state_file.exists():
            self._load_resume_state()
            print(f"🔄 Resuming from previous run: {len(self.completed_steps)} steps completed")

    def step(self, name: str, inputs: Optional[Dict[str, Any]] = None) -> None:
        """
        Start a new step (auto-completes the previous step if exists).

        Args:
            name: Descriptive name for this step
            inputs: Input data/parameters for this step
        """
        # Complete previous step if exists
        if self.current_step and self.current_step.status == "in_progress":
            self._complete_current_step(status="success")

        # Create new step
        self.step_counter += 1
        self.current_step = StepRecord(
            step_number=self.step_counter,
            name=name,
            status="in_progress",
            start_time=datetime.now().isoformat(),
            inputs=inputs or {}
        )

        # Reset token tracking for this step
        self._current_step_start_tokens = TokenUsage()

        # Check if step was already completed (for resumable)
        if self.resumable and name in self.completed_steps:
            print(f"\n⏭️  Step {self.step_counter}: {name} (skipped - already completed)")
        else:
            print(f"\n🔹 Step {self.step_counter}: {name}")

        # Save immediately
        self._save()

    def update(self, data: Dict[str, Any]) -> None:
        """
        Add an incremental update to the current step (e.g., loop progress).
        Saves immediately to persist progress.

        Args:
            data: Progress data to log
        """
        if not self.current_step:
            print("⚠️ No active step - call step() first")
            return

        update = StepUpdate(
            timestamp=datetime.now().isoformat(),
            data=data
        )
        self.current_step.updates.append(update)

        # Save immediately to capture incremental progress
        self._save()

    def output(self, data: Dict[str, Any]) -> None:
        """
        Set the output for the current step and mark it as successful.
        Auto-saves and completes the step.

        Args:
            data: Output data from this step
        """
        if not self.current_step:
            print("⚠️ No active step - call step() first")
            return

        self.current_step.outputs = data

        # Save to resume state if resumable
        if self.resumable:
            step_name = self.current_step.name
            self.step_outputs[step_name] = data
            self.completed_steps.append(step_name)
            self._save_resume_state()

        self._complete_current_step(status="success")

    def fail(self, error: Exception) -> None:
        """
        Mark the current step as failed with error details.

        Args:
            error: Exception that caused the failure
        """
        if not self.current_step:
            print("⚠️ No active step to mark as failed")
            return

        self.current_step.error = StepError(
            type=type(error).__name__,
            message=str(error),
            traceback=traceback.format_exc()
        )

        self._complete_current_step(status="failed")
        print(f"❌ Step failed: {error}")

    def track(self, step_name: str, response: Any) -> TokenUsage:
        """
        Track token usage from an AIResponse.
        Compatible with existing TokenTracker interface for LLM providers.

        Args:
            step_name: Name of the LLM call (can be different from step name)
            response: AIResponse object with token counts

        Returns:
            TokenUsage object
        """
        if not self.current_step:
            print("⚠️ No active step - tokens not tracked")
            return TokenUsage()

        # Extract tokens from AIResponse
        usage = TokenUsage(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens
        )

        # Add to current step's token count
        self.current_step.tokens.input_tokens += usage.input_tokens
        self.current_step.tokens.output_tokens += usage.output_tokens
        self.current_step.tokens.total_tokens += usage.total_tokens

        print(f"📊 {step_name}: {usage.input_tokens}→{usage.output_tokens} tokens")

        # Save after tracking tokens
        self._save()

        return usage

    def _complete_current_step(self, status: str) -> None:
        """Complete the current step with the given status"""
        if not self.current_step:
            return

        self.current_step.status = status
        self.current_step.end_time = datetime.now().isoformat()

        # Calculate duration
        start = datetime.fromisoformat(self.current_step.start_time)
        end = datetime.fromisoformat(self.current_step.end_time)
        self.current_step.duration_seconds = (end - start).total_seconds()

        # Add to steps list
        self.steps.append(self.current_step)

        # Print completion
        status_emoji = "✅" if status == "success" else "❌"
        print(f"{status_emoji} Step {self.current_step.step_number} complete ({self.current_step.duration_seconds:.1f}s)")

        # Clear current step
        self.current_step = None

        # Save
        self._save()

    def _save(self) -> None:
        """Save current state to JSON file"""
        # Build steps list (include in-progress step if exists)
        steps_to_save = self.steps.copy()
        if self.current_step:
            steps_to_save.append(self.current_step)

        # Calculate summary
        summary = self._calculate_summary(steps_to_save)

        log = StepLogSummary(
            script_name=self.script_name,
            timestamp=self.session_timestamp,
            steps=steps_to_save,
            summary=summary
        )

        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log.model_dump(), f, indent=2, ensure_ascii=False)

    def _calculate_summary(self, steps: List[StepRecord]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        total_steps = len(steps)
        successful = sum(1 for s in steps if s.status == "success")
        failed = sum(1 for s in steps if s.status == "failed")
        in_progress = sum(1 for s in steps if s.status == "in_progress")

        total_duration = sum(
            s.duration_seconds for s in steps
            if s.duration_seconds is not None
        )

        total_tokens = TokenUsage()
        for step in steps:
            total_tokens.input_tokens += step.tokens.input_tokens
            total_tokens.output_tokens += step.tokens.output_tokens
            total_tokens.total_tokens += step.tokens.total_tokens

        return {
            "total_steps": total_steps,
            "successful_steps": successful,
            "failed_steps": failed,
            "in_progress_steps": in_progress,
            "total_duration_seconds": round(total_duration, 2),
            "total_tokens": total_tokens.model_dump()
        }

    def finalize(self) -> str:
        """
        Finalize the log (complete any in-progress step and save).
        Call this at the end of your script.

        Returns:
            Path to the saved log file
        """
        # Auto-complete current step if exists
        if self.current_step and self.current_step.status == "in_progress":
            self._complete_current_step(status="success")

        # Print final summary
        summary = self._calculate_summary(self.steps)
        print(f"\n📊 STEP SUMMARY")
        print(f"  Total steps: {summary['total_steps']}")
        print(f"  Successful: {summary['successful_steps']}")
        print(f"  Failed: {summary['failed_steps']}")
        print(f"  Duration: {summary['total_duration_seconds']}s")

        tokens = summary['total_tokens']
        print(f"\n💰 TOTAL TOKENS")
        print(f"  Input: {tokens['input_tokens']}")
        print(f"  Output: {tokens['output_tokens']}")
        print(f"  Total: {tokens['total_tokens']}")

        print(f"\n💾 Step log saved: {self.log_file}")

        # Clear resume state on successful completion
        if self.resumable and self.resume_state_file.exists():
            self.resume_state_file.unlink()
            print(f"🗑️  Resume state cleared")

        return str(self.log_file)

    # Resumable functionality

    def should_run_step(self) -> bool:
        """
        Check if the current step should be executed.
        Returns False if the step was already completed in a previous run.

        Returns:
            True if step should run, False if already completed
        """
        if not self.resumable or not self.current_step:
            return True

        return self.current_step.name not in self.completed_steps

    def get_cached_output(self, key: str) -> Any:
        """
        Get cached output from a previously completed step.

        Args:
            key: Key to retrieve from step outputs

        Returns:
            The cached value, or None if not found
        """
        if not self.resumable:
            print("⚠️ get_cached_output() only works with resumable=True")
            return None

        if not self.current_step:
            print("⚠️ No active step")
            return None

        step_name = self.current_step.name
        if step_name not in self.step_outputs:
            print(f"⚠️ No cached output for step: {step_name}")
            return None

        step_output = self.step_outputs[step_name]
        if key not in step_output:
            print(f"⚠️ Key '{key}' not found in cached output for step: {step_name}")
            return None

        return step_output[key]

    def is_item_completed(self, item_id: str) -> bool:
        """
        Check if an item has been completed in the current step.

        Args:
            item_id: Unique identifier for the item

        Returns:
            True if item was already completed
        """
        if not self.resumable or not self.current_step:
            return False

        step_name = self.current_step.name
        if step_name not in self.completed_items:
            return False

        return item_id in self.completed_items[step_name]

    def mark_item_complete(self, item_id: str, result: Optional[Any] = None) -> None:
        """
        Mark an item as completed in the current step and save resume state.

        Args:
            item_id: Unique identifier for the item
            result: Optional result data to store
        """
        if not self.resumable:
            print("⚠️ mark_item_complete() only works with resumable=True")
            return

        if not self.current_step:
            print("⚠️ No active step - call step() first")
            return

        step_name = self.current_step.name

        # Initialize list if needed
        if step_name not in self.completed_items:
            self.completed_items[step_name] = []

        # Add item if not already there
        if item_id not in self.completed_items[step_name]:
            self.completed_items[step_name].append(item_id)

        # Save resume state immediately
        self._save_resume_state()

    def _load_resume_state(self) -> None:
        """Load resume state from file"""
        try:
            with open(self.resume_state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)

            self.completed_steps = state.get("completed_steps", [])
            self.step_outputs = state.get("step_outputs", {})
            self.completed_items = state.get("completed_items", {})
            self.is_resuming = True

        except Exception as e:
            print(f"⚠️ Failed to load resume state: {e}")
            # Reset to clean state
            self.completed_steps = []
            self.step_outputs = {}
            self.completed_items = {}

    def _save_resume_state(self) -> None:
        """Save resume state to file"""
        state = {
            "script_name": self.script_name,
            "timestamp": datetime.now().isoformat(),
            "completed_steps": self.completed_steps,
            "step_outputs": self.step_outputs,
            "completed_items": self.completed_items
        }

        try:
            with open(self.resume_state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save resume state: {e}")
