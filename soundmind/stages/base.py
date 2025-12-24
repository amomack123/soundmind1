"""
SoundMind v1 Stage Base Utilities.

Responsibilities:
- StageFailure exception for pipeline control flow
- Error object builder per contract
- Stage status.json writer

Invariants:
- Stages always write status.json before raising StageFailure
- Timestamps use PST offset from utils.now_iso()
"""

import json
from pathlib import Path

from soundmind.utils import now_iso


class StageFailure(Exception):
    """
    Raised when a stage fails.
    
    Stage must write its status.json before raising this exception.
    The orchestrator catches this to stop pipeline execution.
    """
    
    def __init__(self, stage: str, errors: list[dict]):
        self.stage = stage
        self.errors = errors
        super().__init__(f"Stage '{stage}' failed")


def build_error(
    code: str,
    message: str,
    stage: str,
    detail: dict | None = None,
    traceback: str | None = None,
) -> dict:
    """
    Build structured error object per contract.
    
    Args:
        code: Error code (e.g., "INGEST_INPUT_MISSING")
        message: Human-readable error message
        stage: Stage name where error occurred
        detail: Optional additional details
        traceback: Optional traceback string
    
    Returns:
        Structured error dictionary.
    """
    error: dict = {
        "code": code,
        "message": message,
        "stage": stage,
    }
    if detail is not None:
        error["detail"] = detail
    if traceback is not None:
        error["traceback"] = traceback
    return error


def write_stage_status(
    stage_dir: Path,
    job_id: str,
    stage: str,
    success: bool,
    started_at: str,
    artifacts: list | None = None,
    errors: list | None = None,
) -> None:
    """
    Write stage status.json to stage directory.
    
    Args:
        stage_dir: Path to stage directory
        job_id: Job identifier
        stage: Stage name
        success: Whether stage succeeded
        started_at: ISO-8601 timestamp when stage started
        artifacts: List of artifact paths (default: [])
        errors: List of error objects (default: [])
    
    Note:
        Uses None defaults to avoid mutable default argument issues.
    """
    # Fixed mutable defaults
    artifacts = [] if artifacts is None else artifacts
    errors = [] if errors is None else errors
    
    status = {
        "job_id": job_id,
        "stage": stage,
        "version": "v1",
        "started_at": started_at,
        "completed_at": now_iso(),
        "success": success,
        "artifacts": artifacts,
        "errors": errors,
    }
    
    status_path = stage_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
