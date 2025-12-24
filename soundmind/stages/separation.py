"""
Stage B: Separation

Responsibilities:
    - Separate input audio into speech and residual (non-speech) tracks
    - Output: speech.wav, residual.wav

Structural segmentation:
    - speech / silence / non_speech

Invariants:
    - Same input + same version = identical separation
    - No semantic inference

Commit 5: Implements Stage class with contract.
    - Requires: audio/original
    - Produces: audio/speech, audio/residual
"""

import shutil

from soundmind.context import JobContext
from soundmind.contracts import Stage, StageContract, StageContext
from soundmind.stages.base import (
    ArtifactRef,
    build_artifact_ref,
    ensure_artifact_path,
    write_stage_status,
    write_stage_status_v2,
)
from soundmind.utils import now_iso


# =============================================================================
# Stage Contract (LOCKED)
# =============================================================================

CONTRACT = StageContract(
    name="separation",
    requires=frozenset({"audio/original"}),
    produces=frozenset({"audio/speech", "audio/residual"}),
    version="1.0.0",
)


# =============================================================================
# SeparationStage Class (Commit 5)
# =============================================================================


class SeparationStage(Stage):
    """
    Stage B: Separate audio into speech and residual stems.
    
    Creates stub stems by copying input audio.
    Real separation will replace in future commits.
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute separation stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List of audio/speech and audio/residual artifact references.
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "separation"
        
        # Get input artifact for status
        input_artifacts = [a for a in ctx.artifacts if a.role == "audio/original"]
        
        # Create stub stems by copying input
        speech_path = ensure_artifact_path(stage_dir, "stems/speech.wav")
        residual_path = ensure_artifact_path(stage_dir, "stems/residual.wav")
        
        shutil.copy2(ctx.input_audio, speech_path)
        shutil.copy2(ctx.input_audio, residual_path)
        
        # Build artifact refs with new role format
        artifacts = [
            build_artifact_ref(
                path="separation/stems/speech.wav",
                artifact_type="audio/wav",
                role="audio/speech",
                description="Stub speech stem (copied from input)",
            ),
            build_artifact_ref(
                path="separation/stems/residual.wav",
                artifact_type="audio/wav",
                role="audio/residual",
                description="Stub residual stem (copied from input)",
            ),
        ]
        
        # Write enhanced status
        write_stage_status_v2(
            stage_dir=stage_dir,
            stage_name=self.contract.name,
            stage_version=self.contract.version,
            start_time=start_time,
            input_artifacts=list(input_artifacts),
            output_artifacts=artifacts,
        )
        
        return artifacts


# =============================================================================
# Backward-Compatible Adapter (TEMPORARY — remove in Commit 6/7)
# =============================================================================


def run(ctx: JobContext) -> JobContext:
    """
    Stage B: Create stub separation stems.
    
    TEMPORARY ADAPTER: Delegates to SeparationStage.run() internally.
    This adapter exists only to avoid breaking current pipeline wiring.
    
    Copies input audio to stems/speech.wav and stems/residual.wav.
    These are stubs — real separation will replace in future commits.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["separation"]
    
    # Create stub stems by copying input
    speech_path = ensure_artifact_path(stage_dir, "stems/speech.wav")
    residual_path = ensure_artifact_path(stage_dir, "stems/residual.wav")
    
    shutil.copy2(ctx.input_wav_path, speech_path)
    shutil.copy2(ctx.input_wav_path, residual_path)
    
    # Build artifact refs with new role format (Commit 5)
    artifacts = [
        build_artifact_ref(
            path="separation/stems/speech.wav",
            artifact_type="audio/wav",
            role="audio/speech",
            description="Stub speech stem (copied from input)",
        ),
        build_artifact_ref(
            path="separation/stems/residual.wav",
            artifact_type="audio/wav",
            role="audio/residual",
            description="Stub residual stem (copied from input)",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "separation", True, started_at, artifacts=artifacts
    )
    return ctx
