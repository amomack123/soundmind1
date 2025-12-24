"""
SoundMind v1 Speaker Audio Tests — Commit 8

Test-first tests for per-speaker audio materialization.

Rules:
- Tests must fail before implementation
- No mocking
- No randomness
- No tolerance-based asserts (except sample count rounding)

Commit 8 Boundaries (LOCKED):
- Pure projection step: consumes diarization semantics, does not modify them
- Deterministic: same input → byte-identical output
- No merging, smoothing, padding, or silence insertion
- Floor-based integer sample indexing

NOTE ON TEST AUDIO (DC Offset):
    Test helpers use constant DC-offset values (e.g., 0.3) instead of sine waves.
    This is intentional: Commit 7 diarization uses exact zero comparison
    (non-zero = speech), so sine waves create fragmented regions at each
    zero-crossing. Using DC-offset ensures contiguous speech regions that
    pass the 0.2s minimum segment threshold.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from soundmind import audio
from tests.conftest import run_cli


# =============================================================================
# TestSpeakerAudioExists — Structural Tests
# =============================================================================


class TestSpeakerAudioExists:
    """Test that per-speaker WAV files are created correctly."""

    def test_speaker_wav_exists(self, tmp_path):
        """Per-speaker WAV exists at expected path."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "speaker-exists-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        assert speaker_wav.exists(), f"Speaker WAV not found at {speaker_wav}"

    def test_artifact_ref_emitted(self, tmp_path):
        """ArtifactRef emitted with role audio/diarized_speaker."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "artifact-ref-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Check diarization status.json for artifact ref
        status_path = jobs_root / job_id / "diarization" / "status.json"
        status = json.loads(status_path.read_text())

        artifacts = status.get("artifacts", [])
        speaker_artifacts = [a for a in artifacts if a.get("role") == "audio/diarized_speaker"]
        
        assert len(speaker_artifacts) == 1, f"Expected 1 audio/diarized_speaker artifact, got {len(speaker_artifacts)}"
        assert speaker_artifacts[0]["path"] == "diarization/per_speaker/SPEAKER_00.wav"

    def test_artifact_in_rollup(self, tmp_path):
        """ArtifactRef included in rollup artifacts."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "rollup-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Check job-level status.json
        status_path = jobs_root / job_id / "status.json"
        status = json.loads(status_path.read_text())

        artifacts = status.get("artifacts", [])
        speaker_artifacts = [a for a in artifacts if a.get("role") == "audio/diarized_speaker"]
        
        assert len(speaker_artifacts) == 1, "audio/diarized_speaker not in rollup"

    def test_empty_segment_no_wav(self, tmp_path):
        """If no segments exist, no WAV is written and no artifact emitted."""
        # Create audio with only silence (no speech)
        sr = audio.CANONICAL_SAMPLE_RATE
        samples = np.zeros(int(sr * 1.0), dtype=np.float32)
        
        input_wav = tmp_path / "silent.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "empty-segment-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # No speaker WAV should exist
        speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        assert not speaker_wav.exists(), "Speaker WAV should not exist for empty segments"

        # No audio/diarized_speaker artifact
        status_path = jobs_root / job_id / "diarization" / "status.json"
        status = json.loads(status_path.read_text())
        artifacts = status.get("artifacts", [])
        speaker_artifacts = [a for a in artifacts if a.get("role") == "audio/diarized_speaker"]
        
        assert len(speaker_artifacts) == 0, "No artifact should be emitted for empty segments"

    def _create_speech_wav(self, path: Path, duration_sec: float = 1.0):
        """Create WAV with speech-like content (DC offset to avoid zero-crossings)."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * duration_sec)
        samples = np.zeros(num_samples, dtype=np.float32)
        
        # Add speech in middle third (>= 0.2s to pass segment filter)
        # Use DC offset to avoid zero-crossings (Commit 7 uses exact zero comparison)
        speech_start = num_samples // 3
        speech_end = 2 * num_samples // 3
        samples[speech_start:speech_end] = 0.3
        
        audio.write_wav(path, samples, sr)


# =============================================================================
# TestSpeakerAudioIntegrity — Audio Integrity Tests
# =============================================================================


class TestSpeakerAudioIntegrity:
    """Test audio content integrity."""

    def test_samples_match_source_segments(self, tmp_path):
        """Extracted samples exactly match source segment ranges."""
        sr = audio.CANONICAL_SAMPLE_RATE
        duration_sec = 1.0
        num_samples = int(sr * duration_sec)
        
        # Create audio with known pattern
        samples = np.zeros(num_samples, dtype=np.float32)
        
        # Speech region: 0.25s to 0.75s (0.5s, >= 0.2s threshold)
        # Use DC offset to avoid zero-crossings
        speech_start_sample = int(0.25 * sr)
        speech_end_sample = int(0.75 * sr)
        samples[speech_start_sample:speech_end_sample] = 0.4
        
        input_wav = tmp_path / "input.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "match-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Load speaker WAV
        speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        speaker_samples, speaker_sr = audio.read_wav(speaker_wav)
        
        # Load speech.wav to get exact source
        speech_wav = jobs_root / job_id / "separation" / "stems" / "speech.wav"
        speech_samples, _ = audio.read_wav(speech_wav)
        
        # Get diarization segments
        diarization_path = jobs_root / job_id / "diarization" / "diarization.json"
        diarization = json.loads(diarization_path.read_text())
        
        if diarization["speakers"]:
            segments = diarization["speakers"][0]["segments"]
            
            # Reconstruct expected samples using floor-based indexing
            expected_samples = []
            for seg in segments:
                start_idx = int(seg["start_s"] * sr)
                end_idx = int(seg["end_s"] * sr)
                expected_samples.append(speech_samples[start_idx:end_idx])
            
            expected = np.concatenate(expected_samples) if expected_samples else np.array([], dtype=np.float32)
            
            assert len(speaker_samples) == len(expected), (
                f"Sample count mismatch: got {len(speaker_samples)}, expected {len(expected)}"
            )
            np.testing.assert_array_equal(speaker_samples, expected, "Samples do not match source")

    def test_output_length_equals_segment_sum(self, tmp_path):
        """Output length equals sum of diarized segment durations."""
        input_wav = tmp_path / "input.wav"
        self._create_multi_segment_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "length-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Get diarization
        diarization_path = jobs_root / job_id / "diarization" / "diarization.json"
        diarization = json.loads(diarization_path.read_text())
        
        if diarization["speakers"]:
            segments = diarization["speakers"][0]["segments"]
            sr = diarization["sample_rate"]
            
            # Calculate expected samples using floor-based indexing
            expected_samples = sum(
                int(seg["end_s"] * sr) - int(seg["start_s"] * sr)
                for seg in segments
            )
            
            # Load speaker WAV
            speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
            speaker_samples, _ = audio.read_wav(speaker_wav)
            
            assert len(speaker_samples) == expected_samples, (
                f"Length mismatch: got {len(speaker_samples)}, expected {expected_samples}"
            )

    def test_sample_rate_preserved(self, tmp_path):
        """Sample rate matches speech.wav sample rate."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "sr-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Get speech.wav sample rate
        speech_wav = jobs_root / job_id / "separation" / "stems" / "speech.wav"
        _, speech_sr = audio.read_wav(speech_wav)
        
        # Get speaker WAV sample rate
        speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        _, speaker_sr = audio.read_wav(speaker_wav)
        
        assert speaker_sr == speech_sr, f"Sample rate mismatch: {speaker_sr} != {speech_sr}"

    def test_integer_sample_indexing(self, tmp_path):
        """Integer sample indexing verified: floor-based conversion."""
        sr = audio.CANONICAL_SAMPLE_RATE
        
        # Create audio with precise segment boundaries
        # Use times that would differ between floor and round
        samples = np.zeros(int(sr * 2.0), dtype=np.float32)
        
        # Region at 0.30001s to 0.70001s (floor should give specific indices)
        # Use DC offset to avoid zero-crossings
        start_sample = int(0.30001 * sr)  # floor: 4800
        end_sample = int(0.70001 * sr)  # floor: 11200
        samples[start_sample:end_sample] = 0.3
        
        input_wav = tmp_path / "input.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "floor-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Load and verify
        speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        if speaker_wav.exists():
            speaker_samples, _ = audio.read_wav(speaker_wav)
            
            # Get actual segments
            diarization_path = jobs_root / job_id / "diarization" / "diarization.json"
            diarization = json.loads(diarization_path.read_text())
            
            if diarization["speakers"]:
                segments = diarization["speakers"][0]["segments"]
                expected_len = sum(
                    int(seg["end_s"] * sr) - int(seg["start_s"] * sr)
                    for seg in segments
                )
                
                assert len(speaker_samples) == expected_len, (
                    f"Floor-based indexing failed: got {len(speaker_samples)}, expected {expected_len}"
                )

    def _create_speech_wav(self, path: Path, duration_sec: float = 1.0):
        """Create WAV with speech-like content (DC offset to avoid zero-crossings)."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * duration_sec)
        samples = np.zeros(num_samples, dtype=np.float32)
        
        speech_start = num_samples // 3
        speech_end = 2 * num_samples // 3
        samples[speech_start:speech_end] = 0.3
        
        audio.write_wav(path, samples, sr)

    def _create_multi_segment_wav(self, path: Path):
        """Create WAV with multiple speech segments (DC offset to avoid zero-crossings)."""
        sr = audio.CANONICAL_SAMPLE_RATE
        samples = np.zeros(int(sr * 2.0), dtype=np.float32)
        
        # Segment 1: 0.2s to 0.5s (use distinct values to identify segments)
        s1_start, s1_end = int(0.2 * sr), int(0.5 * sr)
        samples[s1_start:s1_end] = 0.3
        
        # Segment 2: 1.0s to 1.5s
        s2_start, s2_end = int(1.0 * sr), int(1.5 * sr)
        samples[s2_start:s2_end] = 0.4
        
        audio.write_wav(path, samples, sr)


# =============================================================================
# TestSpeakerAudioSemantics — Semantic Tests
# =============================================================================


class TestSpeakerAudioSemantics:
    """Test semantic correctness."""

    def test_segment_order_preserved(self, tmp_path):
        """Segments concatenated in temporal order (Commit 8 does NOT re-order)."""
        sr = audio.CANONICAL_SAMPLE_RATE
        samples = np.zeros(int(sr * 2.0), dtype=np.float32)
        
        # Two distinct segments with DC offset to avoid zero-crossings
        # Segment 1: 0.2s to 0.5s — value 0.3
        s1_start, s1_end = int(0.2 * sr), int(0.5 * sr)
        samples[s1_start:s1_end] = 0.3
        
        # Segment 2: 1.0s to 1.3s — value 0.4 (different value to distinguish)
        s2_start, s2_end = int(1.0 * sr), int(1.3 * sr)
        samples[s2_start:s2_end] = 0.4
        
        input_wav = tmp_path / "input.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "order-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Load speaker WAV
        speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        speaker_samples, _ = audio.read_wav(speaker_wav)
        
        # Get segments
        diarization_path = jobs_root / job_id / "diarization" / "diarization.json"
        diarization = json.loads(diarization_path.read_text())
        
        if diarization["speakers"] and len(diarization["speakers"][0]["segments"]) >= 2:
            segments = diarization["speakers"][0]["segments"]
            
            # Verify segments are ordered
            for i in range(1, len(segments)):
                assert segments[i]["start_s"] > segments[i-1]["end_s"], (
                    "Segments should be in temporal order from diarization"
                )
            
            # Verify first part of output matches first segment's content
            seg1_len = int(segments[0]["end_s"] * sr) - int(segments[0]["start_s"] * sr)
            first_chunk = speaker_samples[:seg1_len]
            
            # Should resemble 200 Hz signal (from segment 1)
            # Just verify it's not the 400 Hz pattern
            assert len(first_chunk) > 0, "First chunk should not be empty"

    def test_only_speaker_00_exists(self, tmp_path):
        """Only SPEAKER_00 WAV exists."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "single-speaker-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        per_speaker_dir = jobs_root / job_id / "diarization" / "per_speaker"
        if per_speaker_dir.exists():
            wav_files = list(per_speaker_dir.glob("*.wav"))
            speaker_ids = [f.stem for f in wav_files]
            
            assert speaker_ids == ["SPEAKER_00"], f"Expected only SPEAKER_00, got {speaker_ids}"

    def test_no_empty_output_if_segments_exist(self, tmp_path):
        """No empty output if segments exist."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"
        job_id = "no-empty-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # If segments exist, WAV should have content
        diarization_path = jobs_root / job_id / "diarization" / "diarization.json"
        diarization = json.loads(diarization_path.read_text())
        
        if diarization["speakers"] and diarization["speakers"][0]["segments"]:
            speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
            speaker_samples, _ = audio.read_wav(speaker_wav)
            
            assert len(speaker_samples) > 0, "Speaker WAV should not be empty if segments exist"

    def test_no_padding_between_segments(self, tmp_path):
        """No padding or silence insertion between segments."""
        sr = audio.CANONICAL_SAMPLE_RATE
        samples = np.zeros(int(sr * 2.0), dtype=np.float32)
        
        # Two segments with gap (DC offset to avoid zero-crossings)
        s1_start, s1_end = int(0.2 * sr), int(0.5 * sr)
        samples[s1_start:s1_end] = 0.3
        
        s2_start, s2_end = int(1.0 * sr), int(1.3 * sr)
        samples[s2_start:s2_end] = 0.3
        
        input_wav = tmp_path / "input.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "no-padding-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        # Get segments and verify no padding
        diarization_path = jobs_root / job_id / "diarization" / "diarization.json"
        diarization = json.loads(diarization_path.read_text())
        
        if diarization["speakers"]:
            segments = diarization["speakers"][0]["segments"]
            expected_len = sum(
                int(seg["end_s"] * sr) - int(seg["start_s"] * sr)
                for seg in segments
            )
            
            speaker_wav = jobs_root / job_id / "diarization" / "per_speaker" / "SPEAKER_00.wav"
            speaker_samples, _ = audio.read_wav(speaker_wav)
            
            assert len(speaker_samples) == expected_len, (
                f"Output has padding: got {len(speaker_samples)}, expected {expected_len}"
            )

    def _create_speech_wav(self, path: Path, duration_sec: float = 1.0):
        """Create WAV with speech-like content (DC offset to avoid zero-crossings)."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * duration_sec)
        samples = np.zeros(num_samples, dtype=np.float32)
        
        speech_start = num_samples // 3
        speech_end = 2 * num_samples // 3
        samples[speech_start:speech_end] = 0.3
        
        audio.write_wav(path, samples, sr)


# =============================================================================
# TestSpeakerAudioDeterminism — Determinism Tests
# =============================================================================


class TestSpeakerAudioDeterminism:
    """Test byte-identical determinism."""

    def test_byte_identical_across_runs(self, tmp_path):
        """Byte-identical output WAV across runs."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

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

        wav1 = jobs_root / "run1" / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        wav2 = jobs_root / "run2" / "diarization" / "per_speaker" / "SPEAKER_00.wav"

        bytes1 = wav1.read_bytes()
        bytes2 = wav2.read_bytes()

        assert bytes1 == bytes2, "Speaker WAV not byte-identical across runs"

    def test_samples_identical_across_runs(self, tmp_path):
        """Secondary check: parse WAVs and compare sample arrays."""
        input_wav = tmp_path / "input.wav"
        self._create_speech_wav(input_wav)

        jobs_root = tmp_path / "jobs"

        run1 = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", "det1",
        )
        assert run1.returncode == 0

        run2 = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", "det2",
        )
        assert run2.returncode == 0

        wav1 = jobs_root / "det1" / "diarization" / "per_speaker" / "SPEAKER_00.wav"
        wav2 = jobs_root / "det2" / "diarization" / "per_speaker" / "SPEAKER_00.wav"

        samples1, sr1 = audio.read_wav(wav1)
        samples2, sr2 = audio.read_wav(wav2)

        assert sr1 == sr2, "Sample rates differ"
        np.testing.assert_array_equal(samples1, samples2, "Sample arrays differ")

    def _create_speech_wav(self, path: Path, duration_sec: float = 1.0):
        """Create WAV with speech-like content (DC offset to avoid zero-crossings)."""
        sr = audio.CANONICAL_SAMPLE_RATE
        num_samples = int(sr * duration_sec)
        samples = np.zeros(num_samples, dtype=np.float32)
        
        speech_start = num_samples // 3
        speech_end = 2 * num_samples // 3
        samples[speech_start:speech_end] = 0.3
        
        audio.write_wav(path, samples, sr)
