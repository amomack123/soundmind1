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

Commit 4: Creates stub events.json, populates artifacts[].
"""

from soundmind.context import JobContext
from soundmind.stages.base import (
    build_artifact_ref,
    write_artifact,
    write_stage_status,
)
from soundmind.utils import now_iso


# Deterministic stub content (frozen)
STUB_EVENTS = {"events": []}


def run(ctx: JobContext) -> JobContext:
    """
    Stage E: Create stub events output.
    
    Writes events.json with empty events array.
    Real event detection will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["events"]
    
    # Write stub JSON
    artifact_path = write_artifact(stage_dir, "events.json", STUB_EVENTS)
    
    # Build artifact ref
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="events",
            description="Stub acoustic event candidates",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "events", True, started_at, artifacts=artifacts
    )
    return ctx
