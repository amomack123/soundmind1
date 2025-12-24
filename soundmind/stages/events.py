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

Commit 5: Implements Stage class with contract.
    - Requires: audio/residual
    - Produces: metadata/events
"""

from soundmind.context import JobContext
from soundmind.contracts import Stage, StageContract, StageContext
from soundmind.stages.base import (
    ArtifactRef,
    build_artifact_ref,
    write_artifact,
    write_stage_status,
    write_stage_status_v2,
)
from soundmind.utils import now_iso


# =============================================================================
# Stage Contract (LOCKED)
# =============================================================================

CONTRACT = StageContract(
    name="events",
    requires=frozenset({"audio/residual"}),
    produces=frozenset({"metadata/events"}),
    version="1.0.0",
)


# Deterministic stub content (frozen)
STUB_EVENTS = {"events": []}


# =============================================================================
# EventsStage Class (Commit 5)
# =============================================================================


class EventsStage(Stage):
    """
    Stage E: Detect acoustic event candidates.
    
    Creates stub events.json with empty events array.
    Real event detection will replace in future commits.
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute events stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List containing single metadata/events artifact reference.
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "events"
        
        # Get input artifact for status
        input_artifacts = [a for a in ctx.artifacts if a.role == "audio/residual"]
        
        # Write stub JSON
        artifact_path = write_artifact(stage_dir, "events.json", STUB_EVENTS)
        
        # Build artifact ref with new role format
        artifact = build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/events",
            description="Stub acoustic event candidates",
        )
        
        # Write enhanced status
        write_stage_status_v2(
            stage_dir=stage_dir,
            stage_name=self.contract.name,
            stage_version=self.contract.version,
            start_time=start_time,
            input_artifacts=list(input_artifacts),
            output_artifacts=[artifact],
        )
        
        return [artifact]


# =============================================================================
# Backward-Compatible Adapter (TEMPORARY — remove in Commit 6/7)
# =============================================================================


def run(ctx: JobContext) -> JobContext:
    """
    Stage E: Create stub events output.
    
    TEMPORARY ADAPTER: Delegates to EventsStage.run() internally.
    This adapter exists only to avoid breaking current pipeline wiring.
    
    Writes events.json with empty events array.
    Real event detection will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["events"]
    
    # Write stub JSON
    artifact_path = write_artifact(stage_dir, "events.json", STUB_EVENTS)
    
    # Build artifact ref with new role format (Commit 5)
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/events",
            description="Stub acoustic event candidates",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "events", True, started_at, artifacts=artifacts
    )
    return ctx
