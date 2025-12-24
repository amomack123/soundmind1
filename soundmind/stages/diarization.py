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

Commit 6: Energy-based segmentation with single pseudo-speaker.
    - Speaker label: "SPEAKER_00" (fixed, deterministic)
    - Merge gaps < 0.3s, drop segments < 0.2s
    - No multi-speaker clustering or ML
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
    name="diarization",
    requires=frozenset({"audio/speech"}),
    produces=frozenset({"metadata/diarization"}),
    version="1.0.0",
)


# =============================================================================
# Diarization Constants (FROZEN)
# =============================================================================

MERGE_GAP_THRESHOLD = 0.3  # seconds
MIN_SEGMENT_DURATION = 0.2  # seconds
SPEAKER_ID = "SPEAKER_00"  # Fixed pseudo-speaker


# =============================================================================
# DiarizationStage Class (Commit 6 — Real Implementation)
# =============================================================================


class DiarizationStage(Stage):
    """
    Stage D: Identify distinct speakers in speech track.
    
    Commit 6: Energy-based segmentation with single pseudo-speaker.
        - Fixed merge/drop rules
        - No multi-speaker clustering
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
        
        # Load speech stem
        speech_path = ctx.workspace / "separation" / "stems" / "speech.wav"
        samples, sr = audio.read_wav(speech_path)
        samples = audio.normalize_audio(samples, sr)
        
        # Build speech mask (same method as Stage B)
        mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
        
        # Find contiguous speech regions
        regions = audio.find_contiguous_regions(mask)
        
        # Convert to time-based segments
        segments = audio.regions_to_time_segments(regions, audio.CANONICAL_SAMPLE_RATE)
        
        # Merge gaps < 0.3s
        segments = audio.merge_segments(segments, MERGE_GAP_THRESHOLD)
        
        # Drop segments < 0.2s
        segments = audio.drop_short_segments(segments, MIN_SEGMENT_DURATION)
        
        # Build diarization output
        diarization = {
            "sample_rate": audio.CANONICAL_SAMPLE_RATE,
            "speakers": [{
                "speaker_id": SPEAKER_ID,
                "segments": [{"start_s": s, "end_s": e} for s, e in segments]
            }] if segments else []
        }
        
        # Write JSON
        artifact_path = write_artifact(stage_dir, "diarization.json", diarization)
        
        # Build artifact ref
        artifact = build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/diarization",
            description="Energy-based speaker segmentation (single pseudo-speaker)",
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
    Stage D: Identify distinct speakers in speech track.
    
    TEMPORARY ADAPTER: Maintains backward compatibility with pipeline.
    
    Commit 6: Energy-based segmentation with single pseudo-speaker.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["diarization"]
    
    # Load speech stem
    speech_path = ctx.stage_dirs["separation"] / "stems" / "speech.wav"
    samples, sr = audio.read_wav(speech_path)
    samples = audio.normalize_audio(samples, sr)
    
    # Build speech mask
    mask = audio.build_speech_mask(samples, sr=audio.CANONICAL_SAMPLE_RATE)
    
    # Find contiguous speech regions
    regions = audio.find_contiguous_regions(mask)
    
    # Convert to time-based segments
    segments = audio.regions_to_time_segments(regions, audio.CANONICAL_SAMPLE_RATE)
    
    # Merge gaps < 0.3s
    segments = audio.merge_segments(segments, MERGE_GAP_THRESHOLD)
    
    # Drop segments < 0.2s
    segments = audio.drop_short_segments(segments, MIN_SEGMENT_DURATION)
    
    # Build diarization output
    diarization = {
        "sample_rate": audio.CANONICAL_SAMPLE_RATE,
        "speakers": [{
            "speaker_id": SPEAKER_ID,
            "segments": [{"start_s": s, "end_s": e} for s, e in segments]
        }] if segments else []
    }
    
    # Write JSON
    artifact_path = write_artifact(stage_dir, "diarization.json", diarization)
    
    # Build artifact ref
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/diarization",
            description="Energy-based speaker segmentation (single pseudo-speaker)",
        ),
    ]
    
    write_stage_status(
        stage_dir, ctx.job_id, "diarization", True, started_at, artifacts=artifacts
    )
    return ctx
