"""
SoundMind v1 Jobs - Job ID resolution and workspace creation.

Responsibilities:
- Job ID resolution (generate or use explicit)
- Workspace creation (jobs/<job_id>/)
- Collision detection

Forbidden:
- No schema logic
- No knowledge of pipeline stages
"""

import uuid
from pathlib import Path


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
    Create the job workspace directory.
    
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
