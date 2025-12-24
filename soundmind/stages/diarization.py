"""
Stage D: Diarization

Responsibilities:
    - Identify distinct speakers in the speech track
    - Output speaker segments with timestamps (seconds, float)
    - Generate per-speaker audio files

Output schema: schemas/diarization.schema.json
    - sample_rate: integer
    - speakers: array of {speaker_id, segments: [{start_s, end_s}]}

Invariants:
    - Operates on speech track only (from Stage B)
    - All timestamps in seconds (float)
    - No semantic inference (no emotion, intent, identity)
    - Same input + same version = identical output

Commit 5: Implements Stage class with contract.
    - Requires: audio/speech
    - Produces: metadata/diarization
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
    name="diarization",
    requires=frozenset({"audio/speech"}),
    produces=frozenset({"metadata/diarization"}),
    version="1.0.0",
)


# Deterministic stub content (frozen)
STUB_DIARIZATION = {"sample_rate": 16000, "speakers": []}


# =============================================================================
# DiarizationStage Class (Commit 5)
# =============================================================================


class DiarizationStage(Stage):
    """
    Stage D: Identify distinct speakers in speech track.
    
    Creates stub diarization.json with empty speakers list.
    Real diarization will replace in future commits.
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute diarization stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List containing single metadata/diarization artifact reference.
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "diarization"
        
        # Get input artifact for status
        input_artifacts = [a for a in ctx.artifacts if a.role == "audio/speech"]
        
        # Write stub JSON
        artifact_path = write_artifact(stage_dir, "diarization.json", STUB_DIARIZATION)
        
        # Build artifact ref with new role format
        artifact = build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/diarization",
            description="Stub diarization output",
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
    Stage D: Create stub diarization output.
    
    TEMPORARY ADAPTER: Delegates to DiarizationStage.run() internally.
    This adapter exists only to avoid breaking current pipeline wiring.
    
    Writes diarization.json with empty speakers list.
    Real diarization will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["diarization"]
    
    # Write stub JSON
    artifact_path = write_artifact(stage_dir, "diarization.json", STUB_DIARIZATION)
    
    # Build artifact ref with new role format (Commit 5)
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/diarization",
            description="Stub diarization output",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "diarization", True, started_at, artifacts=artifacts
    )
    return ctx
