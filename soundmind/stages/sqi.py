"""
Stage C: Signal Quality Indicators

Responsibilities:
    - Compute signal quality metrics for audio tracks
    - Metrics (Commit 6 locked set):
        - duration_sec: float
        - sample_rate_hz: int (always 16000)
        - num_samples: int
        - rms: float
        - peak_abs: float
        - zero_crossing_rate: float
        - speech_ratio: float (in [0, 1])

Invariants:
    - Metrics are objective, non-semantic
    - Same input + same version = identical metrics
    - No additional fields beyond locked set

Commit 5: Implements Stage class with contract.
    - Requires: audio/speech
    - Produces: metadata/sqi

Commit 6: Real metrics computation.
    - Computes all 7 locked metrics deterministically
"""

import numpy as np

from soundmind import audio
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


# =============================================================================
# SqiStage Class (Commit 6 — Real Implementation)
# =============================================================================


class SqiStage(Stage):
    """
    Stage C: Compute signal quality indicators.
    
    Commit 6: Real metrics computation with locked set.
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
        
        # Find speech artifact and load it
        speech_path = ctx.workspace / "separation" / "stems" / "speech.wav"
        samples, sr = audio.read_wav(speech_path)
        samples = audio.normalize_audio(samples, sr)
        
        # Compute speech ratio (recomputed locally, same method as Stage B)
        mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
        speech_ratio = float(np.mean(mask))
        
        # Compute metrics (locked set, deterministic order via sort_keys)
        sqi_data = {
            "metrics": {
                "duration_sec": float(len(samples) / audio.CANONICAL_SAMPLE_RATE),
                "num_samples": int(len(samples)),
                "peak_abs": audio.compute_peak_abs(samples),
                "rms": audio.compute_rms(samples),
                "sample_rate_hz": audio.CANONICAL_SAMPLE_RATE,
                "speech_ratio": speech_ratio,
                "zero_crossing_rate": audio.compute_zero_crossing_rate(samples),
            }
        }
        
        # Write JSON (sort_keys=True is handled by write_artifact)
        artifact_path = write_artifact(stage_dir, "sqi.json", sqi_data)
        
        # Build artifact ref
        artifact = build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/sqi",
            description="Signal quality indicators (7 metrics)",
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
    Stage C: Compute signal quality indicators.
    
    TEMPORARY ADAPTER: Maintains backward compatibility with pipeline.
    
    Commit 6: Real metrics computation with locked set.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["sqi"]
    
    # Load speech stem
    speech_path = ctx.stage_dirs["separation"] / "stems" / "speech.wav"
    samples, sr = audio.read_wav(speech_path)
    samples = audio.normalize_audio(samples, sr)
    
    # Compute speech ratio (recomputed locally)
    mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
    speech_ratio = float(np.mean(mask))
    
    # Compute metrics (locked set)
    sqi_data = {
        "metrics": {
            "duration_sec": float(len(samples) / audio.CANONICAL_SAMPLE_RATE),
            "num_samples": int(len(samples)),
            "peak_abs": audio.compute_peak_abs(samples),
            "rms": audio.compute_rms(samples),
            "sample_rate_hz": audio.CANONICAL_SAMPLE_RATE,
            "speech_ratio": speech_ratio,
            "zero_crossing_rate": audio.compute_zero_crossing_rate(samples),
        }
    }
    
    # Write JSON
    artifact_path = write_artifact(stage_dir, "sqi.json", sqi_data)
    
    # Build artifact ref
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/sqi",
            description="Signal quality indicators (7 metrics)",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "sqi", True, started_at, artifacts=artifacts
    )
    return ctx
