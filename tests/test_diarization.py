"""
SoundMind v1 Diarization Tests — Commit 7

Test-first tests for deterministic diarization semantics.

Rules:
- Tests must fail before implementation
- No mocking
- No randomness
- No tolerance-based asserts
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from soundmind import audio
from tests.conftest import create_test_wav, run_cli

# Import schema validation
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from validate_schema import load_schema, validate_document


class TestDiarizationDeterminism:
    """Test byte-identical diarization outputs."""

    def test_byte_identical_diarization_json(self, tmp_path):
        """Running pipeline twice produces byte-identical diarization.json."""
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

        path1 = jobs_root / "run1" / "diarization" / "diarization.json"
        path2 = jobs_root / "run2" / "diarization" / "diarization.json"

        bytes1 = path1.read_bytes()
        bytes2 = path2.read_bytes()

        assert bytes1 == bytes2, "diarization.json not byte-identical"


class TestDiarizationSchema:
    """Test schema compliance."""

    def test_schema_validation(self, tmp_path):
        """Diarization output validates against schema."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)

        jobs_root = tmp_path / "jobs"
        job_id = "schema-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        path = jobs_root / job_id / "diarization" / "diarization.json"
        doc = json.loads(path.read_text())

        schema = load_schema("diarization")
        errors = validate_document(doc, schema)

        assert errors == [], f"Schema validation failed: {errors}"


class TestDiarizationSemantics:
    """Test Commit 7 semantic rules."""

    def test_single_speaker_only(self, tmp_path):
        """Output contains exactly one speaker: SPEAKER_00."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)

        jobs_root = tmp_path / "jobs"
        job_id = "speaker-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        path = jobs_root / job_id / "diarization" / "diarization.json"
        doc = json.loads(path.read_text())

        # Must have exactly one speaker entry (or empty if no speech)
        if doc["speakers"]:
            assert len(doc["speakers"]) == 1
            assert doc["speakers"][0]["speaker_id"] == "SPEAKER_00"

    def test_segment_times_six_decimals(self, tmp_path):
        """Segment times are rounded to 6 decimal places."""
        input_wav = tmp_path / "input.wav"
        create_test_wav(input_wav, duration_sec=1.0)

        jobs_root = tmp_path / "jobs"
        job_id = "decimal-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        path = jobs_root / job_id / "diarization" / "diarization.json"
        doc = json.loads(path.read_text())

        for speaker in doc["speakers"]:
            for seg in speaker["segments"]:
                start_str = str(seg["start_s"])
                end_str = str(seg["end_s"])

                # Check decimal places (at most 6 after decimal point)
                if "." in start_str:
                    decimals = len(start_str.split(".")[1])
                    assert decimals <= 6, f"start_s has {decimals} decimals: {start_str}"
                if "." in end_str:
                    decimals = len(end_str.split(".")[1])
                    assert decimals <= 6, f"end_s has {decimals} decimals: {end_str}"


class TestDiarizationNoMerging:
    """Test that Commit 7 does NOT merge segments."""

    def test_no_merging(self, tmp_path):
        """Segments with gaps are NOT merged (Commit 7 rule)."""
        # Create audio with two distinct speech regions separated by silence
        sr = audio.CANONICAL_SAMPLE_RATE
        duration_sec = 1.0
        num_samples = int(sr * duration_sec)

        samples = np.zeros(num_samples, dtype=np.float32)

        # Region 1: 0.1s to 0.4s (0.3s duration, >= 0.2s threshold)
        r1_start = int(0.1 * sr)
        r1_end = int(0.4 * sr)
        t1 = np.arange(r1_end - r1_start) / sr
        samples[r1_start:r1_end] = (0.3 * np.sin(2 * np.pi * 200 * t1)).astype(np.float32)

        # Gap: 0.4s to 0.5s (0.1s gap - would be merged in Commit 6, NOT in Commit 7)

        # Region 2: 0.5s to 0.8s (0.3s duration, >= 0.2s threshold)
        r2_start = int(0.5 * sr)
        r2_end = int(0.8 * sr)
        t2 = np.arange(r2_end - r2_start) / sr
        samples[r2_start:r2_end] = (0.3 * np.sin(2 * np.pi * 200 * t2)).astype(np.float32)

        input_wav = tmp_path / "input.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "no-merge-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        path = jobs_root / job_id / "diarization" / "diarization.json"
        doc = json.loads(path.read_text())

        # With no merging, we expect 2 separate segments
        if doc["speakers"]:
            segments = doc["speakers"][0]["segments"]
            # Commit 7: NO merging, so should have 2 segments
            assert len(segments) >= 2, (
                f"Expected at least 2 segments (no merging), got {len(segments)}: {segments}"
            )


class TestDiarizationShortRuns:
    """Test that short runs are dropped."""

    def test_short_runs_dropped(self, tmp_path):
        """Runs shorter than 0.20s produce no segments."""
        sr = audio.CANONICAL_SAMPLE_RATE
        duration_sec = 1.0
        num_samples = int(sr * duration_sec)

        samples = np.zeros(num_samples, dtype=np.float32)

        # Short region: 0.1s to 0.19s (0.09s duration, < 0.2s threshold)
        short_start = int(0.1 * sr)
        short_end = int(0.19 * sr)
        t = np.arange(short_end - short_start) / sr
        samples[short_start:short_end] = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        input_wav = tmp_path / "input.wav"
        audio.write_wav(input_wav, samples, sr)

        jobs_root = tmp_path / "jobs"
        job_id = "short-run-test"

        result = run_cli(
            "run",
            "--input", str(input_wav),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0

        path = jobs_root / job_id / "diarization" / "diarization.json"
        doc = json.loads(path.read_text())

        # Short run should be dropped, so no segments or empty speaker list
        if doc["speakers"]:
            segments = doc["speakers"][0]["segments"]
            for seg in segments:
                duration = seg["end_s"] - seg["start_s"]
                assert duration >= 0.20, f"Found segment shorter than 0.20s: {seg}"
