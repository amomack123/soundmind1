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

Commit 3: No-op stub, writes status.json (no DSP).
"""

from soundmind.context import JobContext
from soundmind.stages.base import write_stage_status
from soundmind.utils import now_iso


def run(ctx: JobContext) -> JobContext:
    """Stage C: No-op SQI stub."""
    started_at = now_iso()
    write_stage_status(ctx.stage_dirs["sqi"], ctx.job_id, "sqi", True, started_at)
    return ctx
