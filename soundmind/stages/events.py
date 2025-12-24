"""
Stage E: Acoustic Event Candidates

Responsibilities:
    - Detect non-semantic acoustic events in non_speech segments
    - Output event candidates with timestamps and confidence

Event types (FROZEN — trigger-only, no fine-grained taxonomy):
    - impulsive_sound
    - tonal_alarm_like
    - vehicle_like

Output schema: schemas/events.schema.json
    - events: array of {type, start_s, end_s, confidence}

Invariants:
    - Trigger-only detection (recall-biased)
    - Operates ONLY inside non_speech segments
    - No semantic inference (no gunshot, siren, shouting labels)
    - All timestamps in seconds (float)
    - Confidence in range [0.0, 1.0]
    - Same input + same version = identical output

Commit 3: No-op stub, writes status.json (no ML).
"""

from soundmind.context import JobContext
from soundmind.stages.base import write_stage_status
from soundmind.utils import now_iso


def run(ctx: JobContext) -> JobContext:
    """Stage E: No-op events stub."""
    started_at = now_iso()
    write_stage_status(ctx.stage_dirs["events"], ctx.job_id, "events", True, started_at)
    return ctx
