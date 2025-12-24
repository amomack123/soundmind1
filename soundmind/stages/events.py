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

Commit 6: Impulse detection in non-speech regions.
    - Only "impulsive_sound" type (no spectral analysis for tonal/vehicle)
    - confidence: 1.0 (deterministic detection = fixed value per schema)
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
    name="events",
    requires=frozenset({"audio/residual"}),
    produces=frozenset({"metadata/events"}),
    version="1.0.0",
)


# =============================================================================
# EventsStage Class (Commit 6 — Real Implementation)
# =============================================================================


class EventsStage(Stage):
    """
    Stage E: Detect acoustic event candidates.
    
    Commit 6: Impulse detection in non-speech regions.
        - Only detects impulsive_sound (no spectral for tonal/vehicle)
        - confidence: 1.0 (schema-required, deterministic)
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
        
        # Load residual stem (non-speech audio)
        residual_path = ctx.workspace / "separation" / "stems" / "residual.wav"
        samples, sr = audio.read_wav(residual_path)
        samples = audio.normalize_audio(samples, sr)
        
        # Build non-speech mask (inverse of speech mask)
        speech_mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
        non_speech_mask = 1.0 - speech_mask
        
        # Detect impulses in non-speech regions
        impulses = audio.detect_impulses(samples, non_speech_mask)
        
        # Build events output with schema-compliant structure
        events_data = {
            "events": [{
                "type": "impulsive_sound",
                "start_s": e["start"],
                "end_s": e["end"],
                "confidence": 1.0,  # Schema-required, deterministic
            } for e in impulses]
        }
        
        # Write JSON
        artifact_path = write_artifact(stage_dir, "events.json", events_data)
        
        # Build artifact ref
        artifact = build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/events",
            description="Impulse events detected in non-speech regions",
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
    Stage E: Detect acoustic event candidates.
    
    TEMPORARY ADAPTER: Maintains backward compatibility with pipeline.
    
    Commit 6: Impulse detection in non-speech regions.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["events"]
    
    # Load residual stem
    residual_path = ctx.stage_dirs["separation"] / "stems" / "residual.wav"
    samples, sr = audio.read_wav(residual_path)
    samples = audio.normalize_audio(samples, sr)
    
    # Build non-speech mask
    speech_mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
    non_speech_mask = 1.0 - speech_mask
    
    # Detect impulses
    impulses = audio.detect_impulses(samples, non_speech_mask)
    
    # Build events output
    events_data = {
        "events": [{
            "type": "impulsive_sound",
            "start_s": e["start"],
            "end_s": e["end"],
            "confidence": 1.0,
        } for e in impulses]
    }
    
    # Write JSON
    artifact_path = write_artifact(stage_dir, "events.json", events_data)
    
    # Build artifact ref
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/events",
            description="Impulse events detected in non-speech regions",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "events", True, started_at, artifacts=artifacts
    )
    return ctx
