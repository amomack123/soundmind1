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
    - Non-speech derived from speech.wav raw PCM-16 exact-zero (Commit 9)
    - No semantic inference (no gunshot, siren, shouting labels)
    - All timestamps in seconds (float, 6 decimal places in serialization)
    - Confidence in range [0.0, 1.0]
    - Same input + same version = identical output
    - Events sorted by (start_s, end_s, type)

Commit 5: Implements Stage class with contract.
    - Requires: audio/residual
    - Produces: metadata/events

Commit 6: Impulse detection in non-speech regions.
    - Only "impulsive_sound" type (no spectral analysis for tonal/vehicle)
    - confidence: 1.0 (deterministic detection = fixed value per schema)

Commit 9: Deterministic non-speech event materialization.
    - Non-speech mask from speech.wav raw int16 == 0 (immune to float quirks)
    - Stable sort by (start_s, end_s, type)
    - 6-decimal-place timestamp formatting via float(f"{v:.6f}")
    - Locked constants for all thresholds
    - Impulse candidates overlapping speech are discarded (not clipped)
    - Shared logic factored into _compute_events / _write_events_json
"""

import json

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
    name="events",
    requires=frozenset({"audio/residual"}),
    produces=frozenset({"metadata/events"}),
    version="1.0.0",
)


# =============================================================================
# Locked Constants (Commit 9)
# =============================================================================

IMPULSE_PEAK_THRESHOLD = 0.01     # Minimum absolute peak value for impulse detection
IMPULSE_PEAK_RATIO = 0.6          # Frame peak must exceed global_peak * ratio
IMPULSE_MAX_DURATION_SEC = 0.1    # Maximum impulse duration (seconds)
EVENT_DECIMAL_PLACES = 6          # Fixed decimal precision for timestamps


# =============================================================================
# Non-Speech Mask from speech.wav (Commit 9)
# =============================================================================


def _build_non_speech_mask_from_int16(speech_int16):
    """Build sample-level non-speech mask from raw PCM-16 speech samples.

    Uses integer comparison (== 0) which is immune to floating-point
    representation quirks. Per Commit 7 semantics:
        - Non-zero sample = speech
        - Exact zero sample = non-speech

    Args:
        speech_int16: Raw int16 samples from speech.wav

    Returns:
        Sample-level mask (1.0 = non-speech, 0.0 = speech), float32
    """
    return (speech_int16 == 0).astype(np.float32)


# =============================================================================
# Impulse Detection with Non-Speech Enforcement (Commit 9)
# =============================================================================


def _detect_impulses_non_speech_only(
    residual_samples,
    non_speech_mask,
    sr=audio.CANONICAL_SAMPLE_RATE,
    frame_ms=audio.FRAME_MS,
    hop_ms=audio.HOP_MS,
):
    """Detect impulse events strictly within non-speech regions.

    Uses locked constants for all thresholds. Any candidate that
    overlaps speech (even partially) is discarded entirely.

    Args:
        residual_samples: Residual audio samples (1D, float32)
        non_speech_mask: Sample-level mask (1.0 = non-speech), float32
        sr: Sample rate
        frame_ms: Frame length for analysis
        hop_ms: Hop length

    Returns:
        List of event dicts with type, start_s, end_s, confidence.
        Sorted by (start_s, end_s, type).
    """
    events = []

    frame_samples = int(sr * frame_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)

    # Global peak for threshold
    global_peak = audio.compute_peak_abs(residual_samples)
    threshold = max(global_peak * IMPULSE_PEAK_RATIO, IMPULSE_PEAK_THRESHOLD)

    max_duration_samples = int(IMPULSE_MAX_DURATION_SEC * sr)

    n_frames = max(1, (len(residual_samples) - frame_samples) // hop_samples + 1)

    i = 0
    while i < n_frames:
        start_sample = i * hop_samples
        end_sample = min(start_sample + frame_samples, len(residual_samples))

        # Check if frame is in non-speech region (mean > 0.5)
        frame_mask_mean = np.mean(non_speech_mask[start_sample:end_sample])

        if frame_mask_mean < 0.5:
            i += 1
            continue

        # Check frame peak in residual
        frame = residual_samples[start_sample:end_sample]
        frame_peak = np.max(np.abs(frame))

        if frame_peak > threshold:
            # Potential impulse - find extent
            impulse_start = start_sample
            impulse_end = end_sample

            # Extend forward while peak remains high (within max duration)
            j = i + 1
            while j < n_frames:
                next_start = j * hop_samples
                next_end = min(next_start + frame_samples, len(residual_samples))

                if next_end - impulse_start > max_duration_samples:
                    break

                if np.mean(non_speech_mask[next_start:next_end]) < 0.5:
                    break

                next_frame = residual_samples[next_start:next_end]
                if np.max(np.abs(next_frame)) < threshold:
                    break

                impulse_end = next_end
                j += 1

            # Verify duration constraint
            if (impulse_end - impulse_start) / sr <= IMPULSE_MAX_DURATION_SEC:
                # Commit 9: Verify ENTIRE candidate is in non-speech.
                # If any sample overlaps speech, discard the candidate.
                candidate_mask = non_speech_mask[impulse_start:impulse_end]
                if np.all(candidate_mask > 0.5):
                    # Floor-based time conversion, then deterministic rounding
                    start_s = float(f"{audio.samples_to_seconds(impulse_start, sr):.{EVENT_DECIMAL_PLACES}f}")
                    end_s = float(f"{audio.samples_to_seconds(impulse_end, sr):.{EVENT_DECIMAL_PLACES}f}")

                    events.append({
                        "type": "impulsive_sound",
                        "start_s": start_s,
                        "end_s": end_s,
                        "confidence": 1.0,
                    })

            # Skip past this impulse
            i = j
        else:
            i += 1

    # Stable sort by (start_s, end_s, type)
    events.sort(key=lambda e: (e["start_s"], e["end_s"], e["type"]))

    return events


# =============================================================================
# Deterministic JSON Serialization (Commit 9)
# =============================================================================


def _serialize_events_json(events_data):
    """Serialize events data to deterministic JSON string.

    Guarantees:
        - start_s, end_s rendered with exactly 6 decimal places
        - Sorted keys in all objects
        - Consistent indentation (2 spaces)
        - Trailing newline

    Note:
        Uses manual formatting instead of json.dumps to guarantee
        6-decimal trailing zeros in the text output (which json.dumps
        would strip). Values are pre-rounded via float(f"{v:.6f}")
        in _detect_impulses_non_speech_only.
    """
    lines = ["{"]
    lines.append('  "events": [')

    events = events_data["events"]
    for idx, event in enumerate(events):
        comma = "," if idx < len(events) - 1 else ""
        lines.append("    {")
        lines.append(f'      "confidence": {event["confidence"]},')
        lines.append(f'      "end_s": {event["end_s"]:.{EVENT_DECIMAL_PLACES}f},')
        lines.append(f'      "start_s": {event["start_s"]:.{EVENT_DECIMAL_PLACES}f},')
        lines.append(f'      "type": "{event["type"]}"')
        lines.append("    }" + comma)

    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


# =============================================================================
# Shared Logic (Commit 9 — single implementation, two callers)
# =============================================================================


def _compute_events(separation_dir):
    """Compute events data from separation outputs.

    Reads speech.wav (int16 for exact-zero mask) and residual.wav (float32)
    from the given separation directory.

    Args:
        separation_dir: Path to separation stage directory
            (contains stems/speech.wav and stems/residual.wav)

    Returns:
        dict with "events" key containing list of event dicts.
    """
    # Read speech.wav as raw int16 for exact-zero comparison (Commit 9)
    speech_path = separation_dir / "stems" / "speech.wav"
    speech_int16, sr = audio.read_wav_int16(speech_path)

    # Build non-speech mask from raw integer samples
    non_speech_mask = _build_non_speech_mask_from_int16(speech_int16)

    # Load residual stem as float32 for impulse detection
    residual_path = separation_dir / "stems" / "residual.wav"
    residual_samples, _ = audio.read_wav(residual_path)

    # Detect impulses in non-speech regions of residual
    events = _detect_impulses_non_speech_only(
        residual_samples, non_speech_mask, sr=sr,
    )

    return {"events": events}


def _write_events_json(stage_dir, events_data):
    """Write events.json with deterministic formatting.

    Args:
        stage_dir: Path to events stage directory
        events_data: dict with "events" key

    Returns:
        Artifact relative path string (e.g. "events/events.json")
    """
    artifact_path = ensure_artifact_path(stage_dir, "events.json")
    artifact_path.write_text(_serialize_events_json(events_data))
    return f"{stage_dir.name}/events.json"


# =============================================================================
# EventsStage Class (Commit 9 — Deterministic Non-Speech Events)
# =============================================================================


class EventsStage(Stage):
    """
    Stage E: Detect acoustic event candidates.

    Commit 9: Deterministic non-speech event materialization.
        - Non-speech mask from speech.wav int16 exact-zero
        - Only detects impulsive_sound (no spectral for tonal/vehicle)
        - confidence: 1.0 (schema-required, deterministic)
        - Stable sort by (start_s, end_s, type)
        - 6-decimal-place timestamp formatting
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

        # Compute events via shared logic
        separation_dir = ctx.workspace / "separation"
        events_data = _compute_events(separation_dir)

        # Write deterministic JSON
        artifact_path = _write_events_json(stage_dir, events_data)

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

    Commit 9: Deterministic non-speech event materialization.
    """
    started_at = now_iso()
    stage_dir = ctx.stage_dirs["events"]

    # Compute events via shared logic
    separation_dir = ctx.stage_dirs["separation"]
    events_data = _compute_events(separation_dir)

    # Write deterministic JSON
    artifact_path = _write_events_json(stage_dir, events_data)

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
