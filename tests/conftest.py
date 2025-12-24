"""
SoundMind v1 Test Configuration

Provides fixtures for creating valid WAV files for testing.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from soundmind import audio


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run soundmind CLI as subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "soundmind", *args],
        capture_output=True,
        text=True,
    )


def create_test_wav(path: Path, duration_sec: float = 1.0, with_impulse: bool = False) -> None:
    """
    Create a valid WAV file for testing.
    
    Args:
        path: Output path for WAV file
        duration_sec: Duration in seconds
        with_impulse: If True, add impulse events to the audio
    """
    sr = audio.CANONICAL_SAMPLE_RATE
    num_samples = int(sr * duration_sec)
    
    # Create a simple test signal: speech-like region + silence
    samples = np.zeros(num_samples, dtype=np.float32)
    
    # Add "speech-like" energy in the middle third (simple sine + noise)
    speech_start = num_samples // 3
    speech_end = 2 * num_samples // 3
    t = np.arange(speech_end - speech_start) / sr
    
    # Deterministic "speech": sum of a few sine waves
    speech = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 600 * t)
    ).astype(np.float32)
    
    samples[speech_start:speech_end] = speech
    
    if with_impulse:
        # Add a short impulse in the silent region
        impulse_pos = num_samples // 6  # In the first silent third
        samples[impulse_pos:impulse_pos + 10] = 0.8
    
    # Write as valid WAV
    audio.write_wav(path, samples, sr)


@pytest.fixture
def test_wav_path(tmp_path) -> Path:
    """Create a simple test WAV file and return its path."""
    wav_path = tmp_path / "test_input.wav"
    create_test_wav(wav_path, duration_sec=1.0)
    return wav_path


@pytest.fixture
def test_wav_with_impulse(tmp_path) -> Path:
    """Create a test WAV file with impulse and return its path."""
    wav_path = tmp_path / "test_with_impulse.wav"
    create_test_wav(wav_path, duration_sec=1.0, with_impulse=True)
    return wav_path


@pytest.fixture
def pipeline_result(tmp_path, test_wav_path):
    """
    Run pipeline with valid WAV and return paths.
    
    This fixture creates a valid test WAV and runs the full pipeline.
    Use for integration tests that need pipeline output.
    """
    job_id = "test-job"
    jobs_root = tmp_path / "jobs"
    
    result = run_cli(
        "run",
        "--input", str(test_wav_path),
        "--jobs-root", str(jobs_root),
        "--job-id", job_id,
    )
    
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    
    return {
        "job_dir": jobs_root / job_id,
        "input_path": test_wav_path,
        "result": result,
    }
