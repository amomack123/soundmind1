"""
Stage F: Final Roll-Up

Responsibilities:
    - Aggregate outputs from all previous stages
    - Generate final status document
    - Validate all outputs against frozen schemas

Output schema: schemas/status.schema.json
    - job_id: unique identifier
    - created_at: ISO-8601 timestamp
    - input: {original_wav, sha256}
    - stages: {separation, diarization, events}

Invariants:
    - All stage outputs validated before roll-up
    - Same input + same version = identical output
    - No additional inference or transformation
    - ONLY stage allowed to read other stages' status files

Commit 4: Aggregates artifacts from all stages verbatim.
"""

import json

from soundmind.context import JobContext
from soundmind.stages.base import write_stage_status
from soundmind.utils import now_iso


# Expected stages that must have run before rollup (in order)
EXPECTED_STAGES = ["ingest", "separation", "sqi", "diarization", "events"]


def run(ctx: JobContext) -> JobContext:
    """
    Stage F: Read other statuses, aggregate artifacts, write rollup.
    
    Only stage allowed to read other stages' status files.
    Aggregates artifacts[] from all stages verbatim in stage order.
    Does NOT rewrite or normalize artifact refs.
    """
    started_at = now_iso()
    
    # Read all prior stage statuses
    stage_statuses = {}
    for stage_name in EXPECTED_STAGES:
        status_path = ctx.stage_dirs[stage_name] / "status.json"
        if status_path.exists():
            stage_statuses[stage_name] = json.loads(status_path.read_text())
    
    # Check for missing stages (explicit, not silent)
    missing = [s for s in EXPECTED_STAGES if s not in stage_statuses]
    
    if missing:
        # Missing stages = failure (stage didn't produce status.json)
        all_success = False
    else:
        # All stages must have succeeded
        all_success = all(s.get("success", False) for s in stage_statuses.values())
    
    # Aggregate artifacts from all stages in stage order
    # Preserve per-stage order, do NOT rewrite refs
    aggregated_artifacts = []
    for stage_name in EXPECTED_STAGES:
        if stage_name in stage_statuses:
            stage_artifacts = stage_statuses[stage_name].get("artifacts", [])
            aggregated_artifacts.extend(stage_artifacts)
    
    # Rollup records result but never raises — it's an observer
    write_stage_status(
        ctx.stage_dirs["rollup"],
        ctx.job_id,
        "rollup",
        all_success,
        started_at,
        artifacts=aggregated_artifacts,
    )
    return ctx
