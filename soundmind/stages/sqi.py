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

Commit 5: Implements Stage class with contract.
    - Requires: audio/speech
    - Produces: metadata/sqi
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
    name="sqi",
    requires=frozenset({"audio/speech"}),
    produces=frozenset({"metadata/sqi"}),
    version="1.0.0",
)


# Deterministic stub content (frozen)
STUB_SQI = {"metrics": {}}


# =============================================================================
# SqiStage Class (Commit 5)
# =============================================================================


class SqiStage(Stage):
    """
    Stage C: Compute signal quality indicators.
    
    Creates stub sqi.json with empty metrics.
    Real metrics computation will replace in future commits.
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute SQI stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List containing single metadata/sqi artifact reference.
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "sqi"
        
        # Get input artifact for status
        input_artifacts = [a for a in ctx.artifacts if a.role == "audio/speech"]
        
        # Write stub JSON
        artifact_path = write_artifact(stage_dir, "sqi.json", STUB_SQI)
        
        # Build artifact ref with new role format
        artifact = build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/sqi",
            description="Stub signal quality indicators",
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
    Stage C: Create stub SQI output.
    
    TEMPORARY ADAPTER: Delegates to SqiStage.run() internally.
    This adapter exists only to avoid breaking current pipeline wiring.
    
    Writes sqi.json with empty metrics.
    Real metrics computation will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["sqi"]
    
    # Write stub JSON
    artifact_path = write_artifact(stage_dir, "sqi.json", STUB_SQI)
    
    # Build artifact ref with new role format (Commit 5)
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/sqi",
            description="Stub signal quality indicators",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "sqi", True, started_at, artifacts=artifacts
    )
    return ctx
