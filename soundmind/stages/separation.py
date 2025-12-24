"""
Stage B: Separation

Responsibilities:
    - Separate input audio into speech and residual (non-speech) tracks
    - Output: speech.wav, residual.wav

Structural segmentation:
    - speech / silence / non_speech

Invariants:
    - Same input + same version = identical separation
    - No semantic inference

Commit 3: No-op stub, writes status.json (no DSP).
"""

from soundmind.context import JobContext
from soundmind.stages.base import write_stage_status
from soundmind.utils import now_iso


def run(ctx: JobContext) -> JobContext:
    """Stage B: No-op separation stub."""
    started_at = now_iso()
    write_stage_status(ctx.stage_dirs["separation"], ctx.job_id, "separation", True, started_at)
    return ctx
