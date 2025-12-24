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

Commit 3: No-op stub, writes status.json (no ML).
"""

from soundmind.context import JobContext
from soundmind.stages.base import write_stage_status
from soundmind.utils import now_iso


def run(ctx: JobContext) -> JobContext:
    """Stage D: No-op diarization stub."""
    started_at = now_iso()
    write_stage_status(ctx.stage_dirs["diarization"], ctx.job_id, "diarization", True, started_at)
    return ctx
