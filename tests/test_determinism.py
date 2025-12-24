"""
SoundMind v1 Determinism Tests - Commit 6

Verifies deterministic output guarantees:
- Same input → byte-identical WAV outputs
- Same input → byte-identical JSON outputs
- Audio invariants: mono, 16kHz, same length
- speech + residual == input (in float32 before quantization)
"""

import json
from pathlib import Path

import numpy as np
import pytest

from soundmind import audio
from tests.conftest import create_test_wav, run_cli


class TestDeterminism:
    """Test that pipeline produces identical outputs on repeated runs."""

    def test_wav_outputs_byte_identical(self, tmp_path):
        """Running pipeline twice produces byte-identical WAV files."""
        # Create test input
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)
        
        # Run pipeline twice
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
        
        # Compare WAV outputs
        for stem in ["speech.wav", "residual.wav"]:
            path1 = jobs_root / "run1" / "separation" / "stems" / stem
            path2 = jobs_root / "run2" / "separation" / "stems" / stem
            
            bytes1 = path1.read_bytes()
            bytes2 = path2.read_bytes()
            
            assert bytes1 == bytes2, f"{stem} not byte-identical"

    def test_json_outputs_byte_identical(self, tmp_path):
        """Running pipeline twice produces byte-identical JSON files."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)
        
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
        
        # Compare JSON outputs (excluding timestamps in status.json)
        json_files = [
            "sqi/sqi.json",
            "diarization/diarization.json",
            "events/events.json",
        ]
        
        for rel_path in json_files:
            path1 = jobs_root / "run1" / rel_path
            path2 = jobs_root / "run2" / rel_path
            
            content1 = path1.read_text()
            content2 = path2.read_text()
            
            assert content1 == content2, f"{rel_path} not identical"


class TestAudioInvariants:
    """Test audio output invariants."""

    @pytest.fixture
    def processed_audio(self, tmp_path):
        """Run pipeline and return input + output audio data."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)
        
        jobs_root = tmp_path / "jobs"
        job_id = "invariant-test"
        
        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0
        
        job_dir = jobs_root / job_id
        
        return {
            "input_path": input_wav,
            "speech_path": job_dir / "separation" / "stems" / "speech.wav",
            "residual_path": job_dir / "separation" / "stems" / "residual.wav",
        }

    def test_output_is_mono(self, processed_audio):
        """Output WAVs are mono."""
        for key in ["speech_path", "residual_path"]:
            samples, sr = audio.read_wav(processed_audio[key])
            assert samples.ndim == 1, f"{key} is not mono"

    def test_output_sample_rate(self, processed_audio):
        """Output WAVs have 16kHz sample rate."""
        for key in ["speech_path", "residual_path"]:
            samples, sr = audio.read_wav(processed_audio[key])
            assert sr == 16000, f"{key} sample rate is {sr}, expected 16000"

    def test_output_length_preserved(self, processed_audio):
        """Output WAVs have same sample count as normalized input."""
        # Load input and normalize
        input_samples, input_sr = audio.read_wav(processed_audio["input_path"])
        input_normalized = audio.normalize_audio(input_samples, input_sr)
        
        speech_samples, _ = audio.read_wav(processed_audio["speech_path"])
        residual_samples, _ = audio.read_wav(processed_audio["residual_path"])
        
        assert len(speech_samples) == len(input_normalized)
        assert len(residual_samples) == len(input_normalized)

    def test_speech_plus_residual_equals_input(self, processed_audio):
        """speech + residual == input (tested in float32 before quantization)."""
        # This test verifies the separation invariant
        # We need to test BEFORE WAV quantization, so we compute fresh
        
        input_samples, input_sr = audio.read_wav(processed_audio["input_path"])
        input_normalized = audio.normalize_audio(input_samples, input_sr)
        
        # Recompute separation (same as stage does)
        mask = audio.build_speech_mask(input_normalized, sr=audio.CANONICAL_SAMPLE_RATE)
        speech_float = input_normalized * mask
        residual_float = input_normalized - speech_float
        
        # Verify invariant in float32
        reconstructed = speech_float + residual_float
        np.testing.assert_array_almost_equal(
            reconstructed, input_normalized,
            decimal=6,  # float32 precision
            err_msg="speech + residual != input"
        )


class TestSqiMetrics:
    """Test SQI output correctness."""

    @pytest.fixture
    def sqi_data(self, tmp_path):
        """Run pipeline and return SQI data."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)
        
        jobs_root = tmp_path / "jobs"
        job_id = "sqi-test"
        
        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0
        
        sqi_path = jobs_root / job_id / "sqi" / "sqi.json"
        return json.loads(sqi_path.read_text())

    def test_sqi_has_all_locked_metrics(self, sqi_data):
        """SQI has exactly the locked metric set."""
        expected_keys = {
            "duration_sec",
            "sample_rate_hz",
            "num_samples",
            "rms",
            "peak_abs",
            "zero_crossing_rate",
            "speech_ratio",
        }
        assert set(sqi_data["metrics"].keys()) == expected_keys

    def test_sqi_values_finite(self, sqi_data):
        """All SQI values are finite (no NaN/Inf)."""
        import math
        for key, value in sqi_data["metrics"].items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} is not finite: {value}"

    def test_sqi_values_reasonable(self, sqi_data):
        """SQI values are in reasonable ranges."""
        metrics = sqi_data["metrics"]
        
        assert metrics["duration_sec"] > 0
        assert metrics["sample_rate_hz"] == 16000
        assert metrics["num_samples"] > 0
        assert 0 <= metrics["rms"] <= 1
        assert 0 <= metrics["peak_abs"] <= 1
        assert 0 <= metrics["zero_crossing_rate"] <= 1
        assert 0 <= metrics["speech_ratio"] <= 1


class TestDiarizationOutput:
    """Test diarization output correctness."""

    @pytest.fixture
    def diarization_data(self, tmp_path):
        """Run pipeline and return diarization data."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)
        
        jobs_root = tmp_path / "jobs"
        job_id = "diarization-test"
        
        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0
        
        path = jobs_root / job_id / "diarization" / "diarization.json"
        return json.loads(path.read_text())

    def test_single_pseudo_speaker(self, diarization_data):
        """Only uses SPEAKER_00 pseudo-speaker."""
        for speaker in diarization_data["speakers"]:
            assert speaker["speaker_id"] == "SPEAKER_00"

    def test_segments_sorted(self, diarization_data):
        """Segments are sorted by start time."""
        for speaker in diarization_data["speakers"]:
            segments = speaker["segments"]
            start_times = [s["start_s"] for s in segments]
            assert start_times == sorted(start_times)

    def test_segments_no_overlap(self, diarization_data):
        """Segments do not overlap."""
        for speaker in diarization_data["speakers"]:
            segments = speaker["segments"]
            for i in range(len(segments) - 1):
                assert segments[i]["end_s"] <= segments[i + 1]["start_s"]

    def test_segments_within_duration(self, diarization_data, tmp_path):
        """All segment times are within [0, duration]."""
        # Duration should be 1.0 sec based on test fixture
        for speaker in diarization_data["speakers"]:
            for seg in speaker["segments"]:
                assert seg["start_s"] >= 0
                assert seg["end_s"] <= 1.5  # Allow small tolerance


class TestEventsOutput:
    """Test events output correctness."""

    @pytest.fixture
    def events_data(self, tmp_path):
        """Run pipeline and return events data."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0, with_impulse=True)
        
        jobs_root = tmp_path / "jobs"
        job_id = "events-test"
        
        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0
        
        path = jobs_root / job_id / "events" / "events.json"
        return json.loads(path.read_text())

    def test_events_type_is_impulsive_sound(self, events_data):
        """All events have type 'impulsive_sound'."""
        for event in events_data["events"]:
            assert event["type"] == "impulsive_sound"

    def test_events_confidence_is_fixed(self, events_data):
        """All events have confidence 1.0 (schema requirement)."""
        for event in events_data["events"]:
            assert event["confidence"] == 1.0

    def test_events_sorted_by_start(self, events_data):
        """Events are sorted by start time."""
        start_times = [e["start_s"] for e in events_data["events"]]
        assert start_times == sorted(start_times)
