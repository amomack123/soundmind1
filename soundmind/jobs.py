"""
SoundMind v1 Jobs - Job ID resolution and workspace creation.

Responsibilities:
- Job ID resolution (generate or use explicit)
- Workspace creation (jobs/<job_id>/)
- Full workspace structure creation
- Collision detection

Forbidden:
- No schema logic
- No pipeline execution
"""

import uuid
from pathlib import Path


# Fixed stage names — DO NOT modify order or names
STAGE_NAMES = ["ingest", "separation", "sqi", "diarization", "events", "rollup"]


class WorkspaceExistsError(Exception):
    """Raised when attempting to create a workspace that already exists."""
    pass


def resolve_job_id(explicit_id: str | None) -> str:
    """
    Resolve the job ID.
    
    Args:
        explicit_id: If provided, use this ID. Otherwise generate UUID.
    
    Returns:
        The resolved job ID.
    """
    if explicit_id is not None:
        return explicit_id
    return str(uuid.uuid4())


def create_workspace(jobs_root: Path, job_id: str) -> Path:
    """
    Create the job workspace directory (basic, for backwards compatibility).
    
    Args:
        jobs_root: Root directory for all job workspaces.
        job_id: The job identifier.
    
    Returns:
        Path to the created job directory.
    
    Raises:
        WorkspaceExistsError: If the workspace already exists.
    """
    job_dir = jobs_root / job_id
    
    if job_dir.exists():
        raise WorkspaceExistsError(f"Job workspace already exists: {job_dir}")
    
    # Create jobs_root if it doesn't exist
    jobs_root.mkdir(parents=True, exist_ok=True)
    
    # Create job directory
    job_dir.mkdir(parents=False, exist_ok=False)
    
    return job_dir


def create_full_workspace(jobs_root: Path, job_id: str) -> dict[str, Path]:
    """
    Create complete job workspace with all directories.
    
    Creates:
        jobs/<job_id>/
          meta/
          input/
          ingest/
          separation/
          sqi/
          diarization/
          events/
          rollup/
    
    Args:
        jobs_root: Root directory for all job workspaces.
        job_id: The job identifier.
    
    Returns:
        Dict with keys: job_dir, meta_dir, input_dir, and each stage name.
    
    Raises:
        WorkspaceExistsError: If workspace already exists.
    """
    job_dir = jobs_root / job_id
    
    if job_dir.exists():
        raise WorkspaceExistsError(f"Job workspace already exists: {job_dir}")
    
    # Create jobs_root if it doesn't exist
    jobs_root.mkdir(parents=True, exist_ok=True)
    
    # Create job directory
    job_dir.mkdir(parents=False, exist_ok=False)
    
    # Create subdirectories
    meta_dir = job_dir / "meta"
    input_dir = job_dir / "input"
    meta_dir.mkdir()
    input_dir.mkdir()
    
    # Create stage directories
    stage_dirs = {}
    for stage_name in STAGE_NAMES:
        stage_dir = job_dir / stage_name
        stage_dir.mkdir()
        stage_dirs[stage_name] = stage_dir
    
    # Return all paths
    return {
        "job_dir": job_dir,
        "meta_dir": meta_dir,
        "input_dir": input_dir,
        **stage_dirs,
    }
