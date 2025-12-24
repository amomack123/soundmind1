"""
Stage C: Signal Quality Indicators

Responsibilities:
    - Compute signal quality metrics for audio tracks
    - Metrics (all numeric):
        - snr_proxy_db: Signal-to-noise ratio proxy (dB)
        - clipping_pct: Percentage of clipped samples
        - dropout_pct: Percentage of dropout/silence artifacts
        - reverb_proxy_rt60_ms: Reverberation time proxy (ms)
        - loudness_lufs: Integrated loudness (LUFS)
        - rms_dbfs: RMS level (dBFS)

Invariants:
    - Metrics are objective, non-semantic
    - Same input + same version = identical metrics

Commit 4: Creates stub sqi.json, populates artifacts[].
"""

from soundmind.context import JobContext
from soundmind.stages.base import (
    build_artifact_ref,
    write_artifact,
    write_stage_status,
)
from soundmind.utils import now_iso


# Deterministic stub content (frozen)
STUB_SQI = {"metrics": {}}


def run(ctx: JobContext) -> JobContext:
    """
    Stage C: Create stub SQI output.
    
    Writes sqi.json with empty metrics.
    Real metrics computation will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["sqi"]
    
    # Write stub JSON
    artifact_path = write_artifact(stage_dir, "sqi.json", STUB_SQI)
    
    # Build artifact ref
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="sqi",
            description="Stub signal quality indicators",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "sqi", True, started_at, artifacts=artifacts
    )
    return ctx
