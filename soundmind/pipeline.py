"""
SoundMind v1 Pipeline Orchestrator

PIPELINE STAGES (FIXED ORDER — DO NOT MODIFY):

    A. Ingest & Normalize      → soundmind.stages.ingest
    B. Separation              → soundmind.stages.separation
    C. Signal Quality (SQI)    → soundmind.stages.sqi
    D. Diarization             → soundmind.stages.diarization
    E. Acoustic Events         → soundmind.stages.events
    F. Final Roll-Up           → soundmind.stages.rollup

INVARIANTS:
    - Stages execute in order A → F
    - Stages never call each other (only orchestrator sequences)
    - Each stage exposes exactly one entrypoint: run(ctx) -> ctx
    - Pipeline stops on first stage failure
    - Same input + same version = identical output

SCHEMAS (FROZEN):
    - schemas/status.schema.json
    - schemas/diarization.schema.json
    - schemas/events.schema.json
"""

import importlib
import json

from soundmind.context import JobContext
from soundmind.stages.base import StageFailure
from soundmind.utils import now_iso, serialize_json


# Stage registry: (name, module_path)
# Uses dynamic imports for future extensibility (plugins, cloud workers)
# FROZEN — DO NOT MODIFY ORDER OR NAMES
STAGE_ORDER = [
    ("ingest", "soundmind.stages.ingest"),
    ("separation", "soundmind.stages.separation"),
    ("sqi", "soundmind.stages.sqi"),
    ("diarization", "soundmind.stages.diarization"),
    ("events", "soundmind.stages.events"),
    ("rollup", "soundmind.stages.rollup"),
]


def run_pipeline(ctx: JobContext) -> bool:
    """
    Execute all stages A → F in order.
    
    Args:
        ctx: JobContext with all paths configured.
    
    Returns:
        True if all stages succeeded, False otherwise.
    
    Note:
        Overwrites the initialized job-level status.json from Commit 2.5.
        Commit 4: Includes aggregated artifacts from rollup.
    """
    started_at = now_iso()
    failed_stage = None
    pipeline_errors: list[dict] = []
    
    for stage_name, module_path in STAGE_ORDER:
        try:
            module = importlib.import_module(module_path)
            ctx = module.run(ctx)
        except StageFailure as e:
            failed_stage = e.stage
            pipeline_errors = e.errors
            break
    
    completed_at = now_iso()
    success = failed_stage is None
    
    # Collect artifacts from rollup (already aggregated in stage order)
    rollup_status_path = ctx.stage_dirs["rollup"] / "status.json"
    if rollup_status_path.exists():
        rollup_status = json.loads(rollup_status_path.read_text())
        aggregated_artifacts = rollup_status.get("artifacts", [])
    else:
        aggregated_artifacts = []
    
    # Build job-level status.json (overwrites "initialized" state)
    # Artifacts list order = stage order, preserving per-stage artifact order
    job_status = {
        "job_id": ctx.job_id,
        "version": "v1",
        "started_at": started_at,
        "completed_at": completed_at,
        "success": success,
        "failed_stage": failed_stage,
        "stages": {
            name: f"{name}/status.json"
            for name, _ in STAGE_ORDER
            if (ctx.stage_dirs[name] / "status.json").exists()
        },
        "artifacts": aggregated_artifacts,
        "errors": pipeline_errors,
    }
    
    (ctx.job_dir / "status.json").write_text(serialize_json(job_status))
    
    return success
