"""
SoundMind v1 Artifact Tests - Commit 4

Verifies artifact plumbing:
- Artifact files exist after pipeline run
- artifacts[] populated in stage status.json
- Rollup aggregates artifacts in stage order
- Job-level status.json contains artifacts
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run soundmind CLI as subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "soundmind", *args],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def pipeline_result(tmp_path):
    """Run pipeline and return paths."""
    # Create a fake input file
    input_file = tmp_path / "test_input.wav"
    input_file.write_bytes(b"fake wav content for testing")
    
    job_id = "artifact-test-job"
    jobs_root = tmp_path / "jobs"
    
    result = run_cli(
        "run",
        "--input", str(input_file),
        "--jobs-root", str(jobs_root),
        "--job-id", job_id,
    )
    
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    
    return {
        "job_dir": jobs_root / job_id,
        "result": result,
    }


class TestArtifactFilesExist:
    """Verify artifact files are created."""

    def test_separation_stems_exist(self, pipeline_result):
        """Separation creates speech.wav and residual.wav stubs."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "separation" / "stems" / "speech.wav").exists()
        assert (job_dir / "separation" / "stems" / "residual.wav").exists()

    def test_sqi_json_exists(self, pipeline_result):
        """SQI creates sqi.json stub."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "sqi" / "sqi.json").exists()

    def test_diarization_json_exists(self, pipeline_result):
        """Diarization creates diarization.json stub."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "diarization" / "diarization.json").exists()

    def test_events_json_exists(self, pipeline_result):
        """Events creates events.json stub."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "events" / "events.json").exists()


class TestStubContent:
    """Verify stub content is deterministic."""

    def test_sqi_stub_content(self, pipeline_result):
        """SQI stub has expected content."""
        job_dir = pipeline_result["job_dir"]
        sqi = json.loads((job_dir / "sqi" / "sqi.json").read_text())
        assert sqi == {"metrics": {}}

    def test_diarization_stub_content(self, pipeline_result):
        """Diarization stub has expected content."""
        job_dir = pipeline_result["job_dir"]
        diarization = json.loads((job_dir / "diarization" / "diarization.json").read_text())
        assert diarization == {"sample_rate": 16000, "speakers": []}

    def test_events_stub_content(self, pipeline_result):
        """Events stub has expected content."""
        job_dir = pipeline_result["job_dir"]
        events = json.loads((job_dir / "events" / "events.json").read_text())
        assert events == {"events": []}


class TestArtifactRefs:
    """Verify artifacts[] in stage status.json."""

    def test_ingest_produces_audio_original(self, pipeline_result):
        """Ingest produces audio/original artifact (Commit 5)."""
        job_dir = pipeline_result["job_dir"]
        status = json.loads((job_dir / "ingest" / "status.json").read_text())
        assert len(status["artifacts"]) == 1
        assert status["artifacts"][0]["role"] == "audio/original"
        assert status["artifacts"][0]["path"] == "input/original.wav"

    def test_separation_has_two_artifacts(self, pipeline_result):
        """Separation produces speech and residual artifact refs."""
        job_dir = pipeline_result["job_dir"]
        status = json.loads((job_dir / "separation" / "status.json").read_text())
        assert len(status["artifacts"]) == 2
        
        paths = [a["path"] for a in status["artifacts"]]
        assert "separation/stems/speech.wav" in paths
        assert "separation/stems/residual.wav" in paths

    def test_artifact_ref_shape(self, pipeline_result):
        """Artifact refs have frozen shape: path, type, role, description."""
        job_dir = pipeline_result["job_dir"]
        status = json.loads((job_dir / "separation" / "status.json").read_text())
        
        for artifact in status["artifacts"]:
            assert set(artifact.keys()) == {"path", "type", "role", "description"}


class TestRollupAggregation:
    """Verify rollup aggregates all artifacts."""

    def test_rollup_has_all_artifacts(self, pipeline_result):
        """Rollup aggregates artifacts from all stages."""
        job_dir = pipeline_result["job_dir"]
        status = json.loads((job_dir / "rollup" / "status.json").read_text())
        
        # Should have 6 artifacts: 1 from ingest, 2 from separation, 1 from sqi, 1 from diarization, 1 from events
        assert len(status["artifacts"]) == 6

    def test_rollup_preserves_stage_order(self, pipeline_result):
        """Rollup artifacts are in stage order."""
        job_dir = pipeline_result["job_dir"]
        status = json.loads((job_dir / "rollup" / "status.json").read_text())
        
        paths = [a["path"] for a in status["artifacts"]]
        
        # Order should be: ingest, separation, sqi, diarization, events (Commit 5)
        assert paths[0].startswith("input/")
        assert paths[1].startswith("separation/")
        assert paths[2].startswith("separation/")
        assert paths[3].startswith("sqi/")
        assert paths[4].startswith("diarization/")
        assert paths[5].startswith("events/")


class TestJobLevelArtifacts:
    """Verify job-level status.json contains artifacts."""

    def test_job_status_has_artifacts(self, pipeline_result):
        """Job-level status.json includes aggregated artifacts."""
        job_dir = pipeline_result["job_dir"]
        status = json.loads((job_dir / "status.json").read_text())
        
        assert "artifacts" in status
        assert len(status["artifacts"]) == 6  # Commit 5: includes ingest's audio/original

    def test_job_artifacts_match_rollup(self, pipeline_result):
        """Job-level artifacts match rollup artifacts."""
        job_dir = pipeline_result["job_dir"]
        
        job_status = json.loads((job_dir / "status.json").read_text())
        rollup_status = json.loads((job_dir / "rollup" / "status.json").read_text())
        
        assert job_status["artifacts"] == rollup_status["artifacts"]
