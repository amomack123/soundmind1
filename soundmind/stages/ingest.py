"""
Stage A: Ingest & Normalize

Responsibilities:
    - Accept input audio file (WAV or other supported format)
    - Validate file integrity
    - Compute SHA-256 hash for reproducibility
    - Normalize to canonical format (sample rate, bit depth, channels)
    - Output: original.wav (canonical) + metadata

Invariants:
    - Same input file = same SHA-256 hash
    - Canonical output format is deterministic

Commit 3: Validates input exists, writes status.json (no DSP).
"""

from soundmind.context import JobContext
from soundmind.stages.base import build_error, write_stage_status, StageFailure
from soundmind.utils import now_iso


def run(ctx: JobContext) -> JobContext:
    """
    Stage A: Verify input exists.
    
    Checks that input/original.wav exists in the workspace.
    Writes ingest/status.json.
    Raises StageFailure if input is missing.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["ingest"]
    
    if not ctx.input_wav_path.exists():
        error = build_error(
            code="INGEST_INPUT_MISSING",
            message="original.wav not found in input directory",
            stage="ingest",
            detail={"expected_path": str(ctx.input_wav_path)},
        )
        write_stage_status(stage_dir, ctx.job_id, "ingest", False, started_at, errors=[error])
        raise StageFailure("ingest", [error])
    
    write_stage_status(stage_dir, ctx.job_id, "ingest", True, started_at)
    return ctx
