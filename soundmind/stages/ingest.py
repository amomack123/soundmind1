"""
Stage A: Ingest & Normalize

Responsibilities:
    - Accept input audio file (WAV or other supported format)
    - Validate file integrity
    - Compute SHA-256 hash for reproducibility
    - Normalize to canonical format (sample rate, bit depth, channels)
    - Output: original.wav (canonical) + metadata

Invariants:
    - Same input file = same SHA-256 hash
    - Canonical output format is deterministic

Commit 5: Implements Stage class with contract.
    - Produces audio/original artifact
    - Does NOT copy audio; references the canonical input path
"""

from soundmind.context import JobContext
from soundmind.contracts import Stage, StageContract, StageContext
from soundmind.stages.base import (
    ArtifactRef,
    build_artifact_ref,
    build_error,
    write_stage_status,
    write_stage_status_v2,
    StageFailure,
)
from soundmind.utils import now_iso


# =============================================================================
# Stage Contract (LOCKED)
# =============================================================================

CONTRACT = StageContract(
    name="ingest",
    requires=frozenset(),  # No requirements — first stage
    produces=frozenset({"audio/original"}),
    version="1.0.0",
)


# =============================================================================
# IngestStage Class (Commit 5)
# =============================================================================


class IngestStage(Stage):
    """
    Stage A: Validate input exists and produce audio/original artifact.
    
    The Ingest stage produces an ArtifactRef whose path points to the
    canonical input audio file; no file duplication occurs.
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute ingest stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List containing single audio/original artifact reference.
        
        Raises:
            StageFailure: If input audio file does not exist.
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "ingest"
        
        # Validate input exists
        if not ctx.input_audio.exists():
            error = build_error(
                code="INGEST_INPUT_MISSING",
                message="original.wav not found in input directory",
                stage="ingest",
                detail={"expected_path": str(ctx.input_audio)},
            )
            # Write status before raising
            write_stage_status_v2(
                stage_dir=stage_dir,
                stage_name=self.contract.name,
                stage_version=self.contract.version,
                start_time=start_time,
                input_artifacts=[],
                output_artifacts=[],
                success=False,
                errors=[error],
            )
            raise StageFailure("ingest", [error])
        
        # Produce audio/original artifact
        # Note: References canonical input path, no duplication
        artifact = build_artifact_ref(
            path="input/original.wav",
            artifact_type="audio/wav",
            role="audio/original",
            description="Canonical input audio file",
        )
        
        # Write enhanced status
        write_stage_status_v2(
            stage_dir=stage_dir,
            stage_name=self.contract.name,
            stage_version=self.contract.version,
            start_time=start_time,
            input_artifacts=[],  # No inputs for first stage
            output_artifacts=[artifact],
        )
        
        return [artifact]


# =============================================================================
# Backward-Compatible Adapter (TEMPORARY — remove in Commit 6/7)
# =============================================================================


def run(ctx: JobContext) -> JobContext:
    """
    Stage A: Verify input exists.
    
    TEMPORARY ADAPTER: Delegates to IngestStage.run() internally.
    This adapter exists only to avoid breaking current pipeline wiring.
    Will be removed when pipeline switches to Stage-based execution.
    
    Checks that input/original.wav exists in the workspace.
    Writes ingest/status.json.
    Raises StageFailure if input is missing.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["ingest"]
    
    if not ctx.input_wav_path.exists():
        error = build_error(
            code="INGEST_INPUT_MISSING",
            message="original.wav not found in input directory",
            stage="ingest",
            detail={"expected_path": str(ctx.input_wav_path)},
        )
        write_stage_status(stage_dir, ctx.job_id, "ingest", False, started_at, errors=[error])
        raise StageFailure("ingest", [error])
    
    # Produce audio/original artifact (Commit 5)
    artifact = build_artifact_ref(
        path="input/original.wav",
        artifact_type="audio/wav",
        role="audio/original",
        description="Canonical input audio file",
    )
    
    write_stage_status(stage_dir, ctx.job_id, "ingest", True, started_at, artifacts=[artifact])
    return ctx
