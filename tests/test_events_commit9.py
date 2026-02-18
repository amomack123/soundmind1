"""
SoundMind v1 Events Tests — Commit 9

Test-first tests for deterministic non-speech event materialization.

Rules:
- Tests must fail before implementation
- No mocking
- No randomness
- No tolerance-based asserts

Commit 9 Boundaries (LOCKED):
- Events emitted ONLY in non-speech regions (speech.wav exact-zero semantics)
- Deterministic: same input → byte-identical events.json
- Stable sort by (start_s, end_s, type)
- Timestamps formatted to exactly 6 decimal places
- No changes to Commit 1–8 artifacts

NOTE ON TEST AUDIO (DC Offset):
    Test helpers use constant DC-offset values (e.g., 0.3) instead of sine waves.
    This is intentional: Commit 7 diarization uses exact zero comparison
    (non-zero = speech), so sine waves create fragmented regions at each
    zero-crossing. Using DC-offset ensures contiguous speech regions.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from soundmind import audio
from tests.conftest import run_cli

# Import schema validation tools
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from validate_schema import load_schema, validate_document


# =============================================================================
# Helpers
# =============================================================================


def _run_pipeline(tmp_path, input_wav, job_id):
    """Run pipeline and return job directory, asserting success."""
    jobs_root = tmp_path / "jobs"
    result = run_cli(
        "run",
        "--input", str(input_wav),
        "--jobs-root", str(jobs_root),
        "--job-id", job_id,
    )
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    return jobs_root / job_id


def _create_wav(path, samples, sr=audio.CANONICAL_SAMPLE_RATE):
    """Write samples to WAV."""
    audio.write_wav(path, samples.astype(np.float32), sr)


# =============================================================================
# TestEventsDeterminism — Byte-identical and stable ordering
# =============================================================================


class TestEventsDeterminism:
    """Test that events output is perfectly deterministic."""

    def test_events_json_byte_identical(self, tmp_path):
        """Running pipeline twice produces byte-identical events.json."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Non-speech impulse at 0.1s (well within silent region)
        impulse_pos = int(0.1 * sr)
        samples[impulse_pos:impulse_pos + 3] = 0.8

        # Speech region at 0.5s to 1.5s (DC offset)
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        samples[speech_start:speech_end] = 0.3

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        jobs_root = tmp_path / "jobs"

        run1 = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", "run1",
        )
        assert run1.returncode == 0

        run2 = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", "run2",
        )
        assert run2.returncode == 0

        events1 = (jobs_root / "run1" / "events" / "events.json").read_bytes()
        events2 = (jobs_root / "run2" / "events" / "events.json").read_bytes()

        assert events1 == events2, "events.json not byte-identical across runs"

    def test_events_sorted_stably(self, tmp_path):
        """Events are sorted by (start_s, end_s, type) deterministically."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 3.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Two impulses in non-speech region, at different times
        # Impulse 1 at 0.05s
        pos1 = int(0.05 * sr)
        samples[pos1:pos1 + 3] = 0.9

        # Impulse 2 at 0.15s
        pos2 = int(0.15 * sr)
        samples[pos2:pos2 + 3] = 0.85

        # Speech region at 1.0s to 2.0s (keeps impulses in non-speech)
        speech_start = int(1.0 * sr)
        speech_end = int(2.0 * sr)
        samples[speech_start:speech_end] = 0.3

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "sort-test")

        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())
        events = events_data["events"]

        # Verify sorted by (start_s, end_s, type)
        sort_keys = [(e["start_s"], e["end_s"], e["type"]) for e in events]
        assert sort_keys == sorted(sort_keys), (
            f"Events not sorted by (start_s, end_s, type): {sort_keys}"
        )


# =============================================================================
# TestEventsSchema — Schema compliance and precision
# =============================================================================


class TestEventsSchema:
    """Test schema validation and formatting precision."""

    def test_events_schema_validation(self, tmp_path):
        """events.json validates against schemas/events.schema.json."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Impulse in non-speech region
        impulse_pos = int(0.1 * sr)
        samples[impulse_pos:impulse_pos + 3] = 0.8

        # Speech region
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        samples[speech_start:speech_end] = 0.3

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "schema-test")

        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())

        schema = load_schema("events")
        errors = validate_document(events_data, schema)
        assert errors == [], f"Schema validation failed: {errors}"

    def test_event_times_six_decimals(self, tmp_path):
        """start_s and end_s are formatted to exactly 6 decimal places in JSON."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Impulse in non-speech region
        impulse_pos = int(0.1 * sr)
        samples[impulse_pos:impulse_pos + 3] = 0.8

        # Speech region
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        samples[speech_start:speech_end] = 0.3

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "decimal-test")

        events_path = job_dir / "events" / "events.json"
        raw_text = events_path.read_text()
        events_data = json.loads(raw_text)

        if not events_data["events"]:
            pytest.skip("No events detected to check formatting")

        # Check raw JSON text for decimal formatting
        # Pattern: "start_s": <number with exactly 6 decimal places>
        # Match number tokens after "start_s" and "end_s" keys
        for key in ("start_s", "end_s"):
            pattern = rf'"{key}":\s*(\d+\.\d+)'
            matches = re.findall(pattern, raw_text)
            assert len(matches) > 0, f"No {key} values found in JSON text"
            for match in matches:
                # Check exactly 6 decimal places
                integer_part, decimal_part = match.split(".")
                assert len(decimal_part) == 6, (
                    f"{key} value {match} has {len(decimal_part)} decimal places, expected 6"
                )


# =============================================================================
# TestEventsNonSpeech — Non-speech-only constraints
# =============================================================================


class TestEventsNonSpeech:
    """Test that events only appear in non-speech regions."""

    def test_no_event_overlaps_speech(self, tmp_path):
        """No emitted event overlaps any speech region."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Speech region: 0.3s to 1.0s (DC offset)
        speech_start_sec = 0.3
        speech_end_sec = 1.0
        speech_start = int(speech_start_sec * sr)
        speech_end = int(speech_end_sec * sr)
        samples[speech_start:speech_end] = 0.3

        # Impulse AT 0.35s — inside speech region
        # This must NOT produce an event
        impulse_pos = int(0.35 * sr)
        samples[impulse_pos:impulse_pos + 3] = 0.9

        # Also add impulse at 0.05s — outside speech, should be detected
        safe_impulse_pos = int(0.05 * sr)
        samples[safe_impulse_pos:safe_impulse_pos + 3] = 0.8

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "overlap-test")

        # Load speech.wav to get actual speech mask
        speech_wav = job_dir / "separation" / "stems" / "speech.wav"
        speech_samples, _ = audio.read_wav(speech_wav)

        # Find speech regions (non-zero samples)
        speech_regions = []
        in_speech = False
        start = 0
        for i in range(len(speech_samples)):
            if speech_samples[i] != 0.0 and not in_speech:
                in_speech = True
                start = i
            elif speech_samples[i] == 0.0 and in_speech:
                in_speech = False
                speech_regions.append((start / sr, i / sr))
        if in_speech:
            speech_regions.append((start / sr, len(speech_samples) / sr))

        # Load events
        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())

        # Verify no event overlaps any speech region
        for event in events_data["events"]:
            for s_start, s_end in speech_regions:
                # Overlap if event_start < speech_end AND event_end > speech_start
                assert not (event["start_s"] < s_end and event["end_s"] > s_start), (
                    f"Event [{event['start_s']}, {event['end_s']}] overlaps "
                    f"speech [{s_start}, {s_end}]"
                )

    def test_events_clipped_to_non_speech(self, tmp_path):
        """Impulse straddling speech/non-speech boundary is discarded."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Speech region: 0.5s to 1.5s
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        samples[speech_start:speech_end] = 0.3

        # Impulse straddling the boundary at 0.5s
        # Starts at 0.48s (non-speech), extends into 0.52s (speech)
        boundary_impulse_start = int(0.48 * sr)
        boundary_impulse_end = int(0.52 * sr)
        samples[boundary_impulse_start:boundary_impulse_end] = 0.9

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "boundary-test")

        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())

        # The boundary-straddling impulse must be discarded entirely
        for event in events_data["events"]:
            # No event should overlap the speech region
            assert event["end_s"] <= 0.5 or event["start_s"] >= 1.5, (
                f"Boundary-straddling event not discarded: [{event['start_s']}, {event['end_s']}]"
            )


# =============================================================================
# TestEventsBaseline — Basic detection behavior
# =============================================================================


class TestEventsBaseline:
    """Test baseline event detection behavior."""

    def test_silence_produces_empty_events(self, tmp_path):
        """All-zero speech.wav + all-zero residual → empty events array."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 1.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "silence-test")

        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())

        assert events_data["events"] == [], (
            f"Expected empty events for silence, got {len(events_data['events'])} events"
        )

    def test_impulse_detected_in_non_speech(self, tmp_path):
        """Single-sample spike in non-speech region emits impulsive_sound."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Speech region: 1.0s to 2.0s (DC offset)
        speech_start = int(1.0 * sr)
        speech_end = int(2.0 * sr)
        samples[speech_start:speech_end] = 0.3

        # Strong impulse at 0.1s (well within non-speech, before speech starts)
        impulse_pos = int(0.1 * sr)
        samples[impulse_pos:impulse_pos + 3] = 0.8

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "impulse-test")

        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())

        assert len(events_data["events"]) >= 1, "Expected at least one impulse event"

        # Verify event properties
        event = events_data["events"][0]
        assert event["type"] == "impulsive_sound"
        assert event["confidence"] == 1.0
        assert event["start_s"] >= 0.0
        assert event["end_s"] > event["start_s"]

    def test_subthreshold_noise_ignored(self, tmp_path):
        """Low amplitude noise below threshold produces no events."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Speech region to create a non-trivial signal
        speech_start = int(1.0 * sr)
        speech_end = int(2.0 * sr)
        samples[speech_start:speech_end] = 0.3

        # Very low amplitude noise in non-speech region (below min_peak threshold)
        # Use a value well below 0.01 (the IMPULSE_PEAK_THRESHOLD)
        noise_start = int(0.1 * sr)
        noise_end = int(0.2 * sr)
        samples[noise_start:noise_end] = 0.005

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "noise-test")

        events_path = job_dir / "events" / "events.json"
        events_data = json.loads(events_path.read_text())

        assert events_data["events"] == [], (
            f"Expected no events for subthreshold noise, got {len(events_data['events'])}"
        )


# =============================================================================
# TestEventsGuardrail — No semantic drift in prior artifacts
# =============================================================================


class TestEventsGuardrail:
    """Test that Commit 1-8 artifacts are unchanged by events stage changes."""

    def test_commit8_artifacts_unchanged(self, tmp_path):
        """Prior-stage artifacts are byte-identical across two pipeline runs.

        Only events/events.json may differ (and should become deterministic).
        """
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Speech region (DC offset)
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        samples[speech_start:speech_end] = 0.3

        # Impulse in non-speech
        impulse_pos = int(0.1 * sr)
        samples[impulse_pos:impulse_pos + 3] = 0.8

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        jobs_root = tmp_path / "jobs"

        run1 = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", "guard1",
        )
        assert run1.returncode == 0

        run2 = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", "guard2",
        )
        assert run2.returncode == 0

        # These Commit 1-8 artifacts MUST be byte-identical
        frozen_artifacts = [
            "separation/stems/speech.wav",
            "separation/stems/residual.wav",
            "diarization/diarization.json",
            "sqi/sqi.json",
        ]

        for rel_path in frozen_artifacts:
            path1 = jobs_root / "guard1" / rel_path
            path2 = jobs_root / "guard2" / rel_path

            assert path1.exists(), f"Missing artifact: {path1}"
            assert path2.exists(), f"Missing artifact: {path2}"

            bytes1 = path1.read_bytes()
            bytes2 = path2.read_bytes()

            assert bytes1 == bytes2, f"Commit 1-8 artifact {rel_path} not byte-identical"

        # Check per-speaker WAV if it exists
        speaker_wav1 = jobs_root / "guard1" / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        speaker_wav2 = jobs_root / "guard2" / "diarization" / "per_speaker" / "SPEAKER_00.wav"

        if speaker_wav1.exists() and speaker_wav2.exists():
            assert speaker_wav1.read_bytes() == speaker_wav2.read_bytes(), (
                "SPEAKER_00.wav not byte-identical"
            )


# =============================================================================
# TestEventsIOPath — IO path integrity for non-speech mask
# =============================================================================


class TestEventsIOPath:
    """Test that the IO path preserves exact zeros for non-speech detection."""

    def test_speech_wav_zeros_survive_int16_roundtrip(self, tmp_path):
        """Masked regions in speech.wav are exactly 0 when read as int16.

        This guards against float normalization quirks — the events stage
        uses int16 comparison (== 0) for the non-speech mask, so PCM-16
        zeros must survive the write → read roundtrip exactly.
        """
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * 2.0)
        samples = np.zeros(num_samples, dtype=np.float32)

        # Speech region: 0.5s to 1.5s (DC offset)
        speech_start = int(0.5 * sr)
        speech_end = int(1.5 * sr)
        samples[speech_start:speech_end] = 0.3

        input_wav = tmp_path / "input.wav"
        _create_wav(input_wav, samples)

        job_dir = _run_pipeline(tmp_path, input_wav, "io-test")

        # Read speech.wav as int16 (the way events stage reads it)
        speech_path = job_dir / "separation" / "stems" / "speech.wav"
        speech_int16, _ = audio.read_wav_int16(speech_path)

        # Non-speech regions (before 0.5s and after 1.5s) must be exactly 0
        non_speech_before = speech_int16[0:speech_start]
        non_speech_after = speech_int16[speech_end:]

        assert np.all(non_speech_before == 0), (
            f"Non-speech region before speech has {np.count_nonzero(non_speech_before)} "
            f"non-zero int16 samples"
        )
        assert np.all(non_speech_after == 0), (
            f"Non-speech region after speech has {np.count_nonzero(non_speech_after)} "
            f"non-zero int16 samples"
        )

        # Speech region must be non-zero
        speech_region = speech_int16[speech_start:speech_end]
        assert np.all(speech_region != 0), (
            f"Speech region has {np.count_nonzero(speech_region == 0)} "
            f"zero int16 samples (expected all non-zero)"
        )

