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

Commit 5: Implements Stage class with contract.
    - Requires: all 6 artifact roles from prior stages
    - Produces: {} (aggregation only, no new artifacts)
"""

import json

from soundmind.context import JobContext
from soundmind.contracts import Stage, StageContract, StageContext
from soundmind.stages.base import (
    ArtifactRef,
    write_stage_status,
    write_stage_status_v2,
)
from soundmind.utils import now_iso


# =============================================================================
# Stage Contract (LOCKED)
# =============================================================================

CONTRACT = StageContract(
    name="rollup",
    requires=frozenset({
        "audio/original",
        "audio/speech",
        "audio/residual",
        "metadata/sqi",
        "metadata/diarization",
        "metadata/events",
    }),
    produces=frozenset(),  # Rollup only aggregates; creates no new artifacts
    version="1.0.0",
)


# Expected stages that must have run before rollup (in order)
EXPECTED_STAGES = ["ingest", "separation", "sqi", "diarization", "events"]


# =============================================================================
# RollupStage Class (Commit 5)
# =============================================================================


class RollupStage(Stage):
    """
    Stage F: Aggregate outputs from all previous stages.
    
    Only stage allowed to read other stages' status files.
    Aggregates artifacts from all stages verbatim in stage order.
    Does NOT rewrite or normalize artifact refs.
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute rollup stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            Empty list (rollup produces no new artifacts).
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "rollup"
        
        # All input artifacts from prior stages
        input_artifacts = list(ctx.artifacts)
        
        # Read all prior stage statuses for compatibility with existing code
        stage_statuses = {}
        for stage_name in EXPECTED_STAGES:
            status_path = ctx.workspace / stage_name / "status.json"
            if status_path.exists():
                stage_statuses[stage_name] = json.loads(status_path.read_text())
        
        # Check for missing stages
        missing = [s for s in EXPECTED_STAGES if s not in stage_statuses]
        
        if missing:
            all_success = False
        else:
            all_success = all(s.get("success", False) for s in stage_statuses.values())
        
        # Write enhanced status
        write_stage_status_v2(
            stage_dir=stage_dir,
            stage_name=self.contract.name,
            stage_version=self.contract.version,
            start_time=start_time,
            input_artifacts=input_artifacts,
            output_artifacts=[],  # Rollup produces nothing
            success=all_success,
        )
        
        return []


# =============================================================================
# Backward-Compatible Adapter (TEMPORARY — remove in Commit 6/7)
# =============================================================================


def run(ctx: JobContext) -> JobContext:
    """
    Stage F: Read other statuses, aggregate artifacts, write rollup.
    
    TEMPORARY ADAPTER: Maintains compatibility with current pipeline.
    This adapter exists only to avoid breaking current pipeline wiring.
    
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
