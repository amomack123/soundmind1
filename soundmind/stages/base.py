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


# =============================================================================
# Artifact Helpers (Commit 4)
# =============================================================================


def ensure_artifact_path(stage_dir: Path, rel_path: str) -> Path:
    """
    Ensure parent directories exist for an artifact path.
    
    Args:
        stage_dir: Path to stage directory (e.g., jobs/<job_id>/separation)
        rel_path: Relative path within stage (e.g., "stems/speech.wav")
    
    Returns:
        Absolute path to the artifact file.
    
    Note:
        Creates parent directories if they don't exist.
        Does NOT create the file itself.
    """
    artifact_path = stage_dir / rel_path
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    return artifact_path


def write_artifact(
    stage_dir: Path,
    rel_path: str,
    content: bytes | dict | str,
    binary: bool = False,
) -> str:
    """
    Write content to an artifact file.
    
    Args:
        stage_dir: Path to stage directory
        rel_path: Relative path within stage
        content: Content to write
        binary: If True, write content as bytes.
                If False, JSON-serialize dict or write str directly.
    
    Returns:
        Relative path from job root (e.g., "separation/stems/speech.wav")
    
    Note:
        Creates parent directories if needed.
    """
    artifact_path = ensure_artifact_path(stage_dir, rel_path)
    
    if binary:
        artifact_path.write_bytes(content)
    else:
        if isinstance(content, dict):
            artifact_path.write_text(
                json.dumps(content, indent=2, sort_keys=True) + "\n"
            )
        else:
            artifact_path.write_text(content)
    
    # Return relative path from job root: stage_name/rel_path
    return f"{stage_dir.name}/{rel_path}"


def build_artifact_ref(
    path: str,
    artifact_type: str,
    role: str,
    description: str,
) -> dict:
    """
    Build a standardized artifact reference.
    
    Args:
        path: Relative path from job root (e.g., "separation/stems/speech.wav")
        artifact_type: MIME type (e.g., "audio/wav", "application/json")
        role: Artifact role (e.g., "speech", "residual", "diarization")
        description: Human-readable description
    
    Returns:
        Frozen ArtifactRef dict with exactly four fields:
        {path, type, role, description}
    
    Note:
        This shape is FROZEN. No additional fields in Commit 4.
    """
    return {
        "path": path,
        "type": artifact_type,
        "role": role,
        "description": description,
    }
