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

Commit 6: Real energy-based RMS masking.
    - speech.wav = input * mask
    - residual.wav = input - speech
    - Same length, sample rate, deterministic output
"""

import numpy as np

from soundmind import audio
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
# SeparationStage Class (Commit 6 — Real Implementation)
# =============================================================================


class SeparationStage(Stage):
    """
    Stage B: Separate audio into speech and residual stems.
    
    Commit 6: Energy-based RMS masking.
        - speech = input * mask
        - residual = input - speech
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
        
        # Load and normalize input audio
        samples, sr = audio.read_wav(ctx.input_audio)
        samples = audio.normalize_audio(samples, sr)
        
        # Build speech mask using energy-based RMS thresholding
        mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
        
        # Create stems
        speech = samples * mask
        residual = samples - speech  # = samples * (1 - mask)
        
        # Prepare output paths
        speech_path = ensure_artifact_path(stage_dir, "stems/speech.wav")
        residual_path = ensure_artifact_path(stage_dir, "stems/residual.wav")
        
        # Write with hard clipping (already handled in write_wav)
        audio.write_wav(speech_path, speech)
        audio.write_wav(residual_path, residual)
        
        # Build artifact refs
        artifacts = [
            build_artifact_ref(
                path="separation/stems/speech.wav",
                artifact_type="audio/wav",
                role="audio/speech",
                description="Speech stem (energy-based RMS mask)",
            ),
            build_artifact_ref(
                path="separation/stems/residual.wav",
                artifact_type="audio/wav",
                role="audio/residual",
                description="Residual stem (input - speech)",
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
    Stage B: Separate audio into speech and residual stems.
    
    TEMPORARY ADAPTER: Maintains backward compatibility with pipeline.
    
    Commit 6: Real energy-based separation.
        - speech = input * mask
        - residual = input - speech
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["separation"]
    
    # Load and normalize input audio
    samples, sr = audio.read_wav(ctx.input_wav_path)
    samples = audio.normalize_audio(samples, sr)
    
    # Build speech mask using energy-based RMS thresholding
    mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
    
    # Create stems
    speech = samples * mask
    residual = samples - speech
    
    # Prepare output paths
    speech_path = ensure_artifact_path(stage_dir, "stems/speech.wav")
    residual_path = ensure_artifact_path(stage_dir, "stems/residual.wav")
    
    # Write WAVs
    audio.write_wav(speech_path, speech)
    audio.write_wav(residual_path, residual)
    
    # Build artifact refs
    artifacts = [
        build_artifact_ref(
            path="separation/stems/speech.wav",
            artifact_type="audio/wav",
            role="audio/speech",
            description="Speech stem (energy-based RMS mask)",
        ),
        build_artifact_ref(
            path="separation/stems/residual.wav",
            artifact_type="audio/wav",
            role="audio/residual",
            description="Residual stem (input - speech)",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "separation", True, started_at, artifacts=artifacts
    )
    return ctx
