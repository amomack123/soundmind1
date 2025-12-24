"""
Stage D: Diarization

Responsibilities:
    - Identify distinct speakers in the speech track
    - Output speaker segments with timestamps (seconds, float)
    - Generate per-speaker audio files (Commit 8)

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

Commit 7: Deterministic diarization semantics.
    - Speaker label: "SPEAKER_00" (fixed, deterministic)
    - Consumes speech.wav directly (no mask rebuild)
    - Exact zero comparison: non-zero = speech, zero = silence
    - One segment per contiguous speech run >= 0.20s
    - NO merging, NO smoothing, NO heuristics
    - Times rounded to 6 decimals at write-time

Commit 8: Per-speaker audio materialization.
    - Produces per_speaker/SPEAKER_00.wav
    - Pure projection: consumes semantics, does not modify them
    - Floor-based integer sample indexing
    - No amplitude normalization or dtype conversion
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
    produces=frozenset({"metadata/diarization", "audio/diarized_speaker"}),
    version="1.0.0",
)


# =============================================================================
# Diarization Constants (FROZEN)
# =============================================================================

MIN_SEGMENT_DURATION = 0.2  # seconds
SPEAKER_ID = "SPEAKER_00"  # Fixed pseudo-speaker


# =============================================================================
# Local Segmentation Logic (Commit 7 — No audio.py helpers)
# =============================================================================


def find_speech_regions(samples: "np.ndarray", sr: int) -> list[tuple[float, float]]:
    """
    Find contiguous non-zero regions in speech.wav.
    
    Commit 7 rule: A sample is speech iff its value is non-zero.
    No thresholds, no energy inference.
    
    Args:
        samples: Speech samples (already masked by separation stage)
        sr: Sample rate
    
    Returns:
        List of (start_sec, end_sec) tuples
    """
    regions = []
    in_region = False
    start = 0
    
    for i in range(len(samples)):
        val = samples[i]
        if val != 0.0 and not in_region:
            in_region = True
            start = i
        elif val == 0.0 and in_region:
            in_region = False
            regions.append((start / sr, i / sr))
    
    # Handle region extending to end
    if in_region:
        regions.append((start / sr, len(samples) / sr))
    
    return regions


def drop_short_segments(
    segments: list[tuple[float, float]],
    min_duration: float,
) -> list[tuple[float, float]]:
    """
    Drop segments shorter than minimum duration.
    
    Args:
        segments: List of (start_sec, end_sec) tuples
        min_duration: Minimum duration to keep (seconds)
    
    Returns:
        Filtered segment list
    """
    return [(s, e) for s, e in segments if (e - s) >= min_duration]


# =============================================================================
# DiarizationStage Class (Commit 7 — Deterministic Semantics)
# =============================================================================


class DiarizationStage(Stage):
    """
    Stage D: Identify distinct speakers in speech track.
    
    Commit 7: Deterministic diarization semantics.
        - Consumes speech.wav directly
        - Exact zero comparison for speech detection
        - No merging, no smoothing
    """
    
    contract = CONTRACT
    
    def run(self, ctx: StageContext) -> list[ArtifactRef]:
        """
        Execute diarization stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List of artifact references (metadata/diarization + audio/diarized_speaker if segments exist).
        """
        start_time = now_iso()
        stage_dir = ctx.workspace / "diarization"
        
        # Get input artifact for status
        input_artifacts = [a for a in ctx.artifacts if a.role == "audio/speech"]
        
        # Load speech.wav (already masked by separation stage)
        speech_path = ctx.workspace / "separation" / "stems" / "speech.wav"
        samples, sr = audio.read_wav(speech_path)
        
        # Find contiguous non-zero regions (exact zero comparison)
        segments = find_speech_regions(samples, sr)
        
        # Drop segments < 0.2s (NO MERGING)
        segments = drop_short_segments(segments, MIN_SEGMENT_DURATION)
        
        # Build diarization output with 6-decimal rounding at write-time
        diarization = {
            "sample_rate": audio.CANONICAL_SAMPLE_RATE,
            "speakers": [{
                "speaker_id": SPEAKER_ID,
                "segments": [
                    {"start_s": round(s, 6), "end_s": round(e, 6)}
                    for s, e in segments
                ]
            }] if segments else []
        }
        
        # Write JSON
        artifact_path = write_artifact(stage_dir, "diarization.json", diarization)
        
        # Build artifact ref for JSON
        artifacts = [
            build_artifact_ref(
                path=artifact_path,
                artifact_type="application/json",
                role="metadata/diarization",
                description="Energy-based speaker segmentation (single pseudo-speaker)",
            )
        ]
        
        # Commit 8: Generate per-speaker WAV (only if segments exist)
        if segments:
            per_speaker_dir = stage_dir / "per_speaker"
            per_speaker_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract and concatenate segments (floor-based integer indexing)
            speaker_samples = audio.extract_and_concatenate(samples, segments, sr)
            
            # Write speaker WAV (deterministic, bit-exact)
            speaker_wav_path = per_speaker_dir / f"{SPEAKER_ID}.wav"
            audio.write_wav(speaker_wav_path, speaker_samples, sr)
            
            # Build artifact ref for speaker WAV
            artifacts.append(
                build_artifact_ref(
                    path=f"diarization/per_speaker/{SPEAKER_ID}.wav",
                    artifact_type="audio/wav",
                    role="audio/diarized_speaker",
                    description=f"Concatenated speech for {SPEAKER_ID}",
                )
            )
        
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
    Stage D: Identify distinct speakers in speech track.
    
    TEMPORARY ADAPTER: Maintains backward compatibility with pipeline.
    
    Commit 7: Deterministic diarization semantics.
    Commit 8: Per-speaker audio materialization.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["diarization"]
    
    # Load speech.wav (already masked by separation stage)
    speech_path = ctx.stage_dirs["separation"] / "stems" / "speech.wav"
    samples, sr = audio.read_wav(speech_path)
    
    # Find contiguous non-zero regions (exact zero comparison)
    segments = find_speech_regions(samples, sr)
    
    # Drop segments < 0.2s (NO MERGING)
    segments = drop_short_segments(segments, MIN_SEGMENT_DURATION)
    
    # Build diarization output with 6-decimal rounding at write-time
    diarization = {
        "sample_rate": audio.CANONICAL_SAMPLE_RATE,
        "speakers": [{
            "speaker_id": SPEAKER_ID,
            "segments": [
                {"start_s": round(s, 6), "end_s": round(e, 6)}
                for s, e in segments
            ]
        }] if segments else []
    }
    
    # Write JSON
    artifact_path = write_artifact(stage_dir, "diarization.json", diarization)
    
    # Build artifact refs
    artifacts = [
        build_artifact_ref(
            path=artifact_path,
            artifact_type="application/json",
            role="metadata/diarization",
            description="Energy-based speaker segmentation (single pseudo-speaker)",
        ),
    ]
    
    # Commit 8: Generate per-speaker WAV (only if segments exist)
    if segments:
        per_speaker_dir = stage_dir / "per_speaker"
        per_speaker_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and concatenate segments (floor-based integer indexing)
        speaker_samples = audio.extract_and_concatenate(samples, segments, sr)
        
        # Write speaker WAV (deterministic, bit-exact)
        speaker_wav_path = per_speaker_dir / f"{SPEAKER_ID}.wav"
        audio.write_wav(speaker_wav_path, speaker_samples, sr)
        
        # Build artifact ref for speaker WAV
        artifacts.append(
            build_artifact_ref(
                path=f"diarization/per_speaker/{SPEAKER_ID}.wav",
                artifact_type="audio/wav",
                role="audio/diarized_speaker",
                description=f"Concatenated speech for {SPEAKER_ID}",
            )
        )
    
    write_stage_status(
        stage_dir, ctx.job_id, "diarization", True, started_at, artifacts=artifacts
    )
    return ctx
