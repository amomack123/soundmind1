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

Commit 4: Creates stub stems (copies of input), populates artifacts[].
"""

import shutil

from soundmind.context import JobContext
from soundmind.stages.base import (
    build_artifact_ref,
    ensure_artifact_path,
    write_stage_status,
)
from soundmind.utils import now_iso


def run(ctx: JobContext) -> JobContext:
    """
    Stage B: Create stub separation stems.
    
    Copies input audio to stems/speech.wav and stems/residual.wav.
    These are stubs — real separation will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["separation"]
    
    # Create stub stems by copying input
    speech_path = ensure_artifact_path(stage_dir, "stems/speech.wav")
    residual_path = ensure_artifact_path(stage_dir, "stems/residual.wav")
    
    shutil.copy2(ctx.input_wav_path, speech_path)
    shutil.copy2(ctx.input_wav_path, residual_path)
    
    # Build artifact refs
    artifacts = [
        build_artifact_ref(
            path="separation/stems/speech.wav",
            artifact_type="audio/wav",
            role="speech",
            description="Stub speech stem (copied from input)",
        ),
        build_artifact_ref(
            path="separation/stems/residual.wav",
            artifact_type="audio/wav",
            role="residual",
            description="Stub residual stem (copied from input)",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "separation", True, started_at, artifacts=artifacts
    )
    return ctx
