"""
Stage D: Diarization

Responsibilities:
    - Identify distinct speakers in the speech track
    - Output speaker segments with timestamps (seconds, float)
    - Generate per-speaker audio files

Output schema: schemas/diarization.schema.json
    - sample_rate: integer
    - speakers: array of {speaker_id, segments: [{start_s, end_s}]}

Invariants:
    - Operates on speech track only (from Stage B)
    - All timestamps in seconds (float)
    - No semantic inference (no emotion, intent, identity)
    - Same input + same version = identical output

Commit 4: Creates stub diarization.json, populates artifacts[].
"""

from soundmind.context import JobContext
from soundmind.stages.base import (
    build_artifact_ref,
    write_artifact,
    write_stage_status,
)
from soundmind.utils import now_iso


# Deterministic stub content (frozen)
STUB_DIARIZATION = {"sample_rate": 16000, "speakers": []}


def run(ctx: JobContext) -> JobContext:
    """
    Stage D: Create stub diarization output.
    
    Writes diarization.json with empty speakers list.
    Real diarization will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["diarization"]
    
    # Write stub JSON
    artifact_path = write_artifact(stage_dir, "diarization.json", STUB_DIARIZATION)
    
    # Build artifact ref
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="diarization",
            description="Stub diarization output",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "diarization", True, started_at, artifacts=artifacts
    )
    return ctx
