"""
SoundMind v1 Artifact Tests - Commit 4/5/6

Verifies artifact plumbing:
- Artifact files exist after pipeline run
- artifacts[] populated in stage status.json
- Rollup aggregates artifacts in stage order
- Job-level status.json contains artifacts

Commit 6: Updated to verify real content (not stubs).
"""

import json
from pathlib import Path

import pytest

from tests.conftest import create_test_wav


class TestArtifactFilesExist:
    """Verify artifact files are created."""

    def test_separation_stems_exist(self, pipeline_result):
        """Separation creates speech.wav and residual.wav."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "separation" / "stems" / "speech.wav").exists()
        assert (job_dir / "separation" / "stems" / "residual.wav").exists()

    def test_sqi_json_exists(self, pipeline_result):
        """SQI creates sqi.json."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "sqi" / "sqi.json").exists()

    def test_diarization_json_exists(self, pipeline_result):
        """Diarization creates diarization.json."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "diarization" / "diarization.json").exists()

    def test_events_json_exists(self, pipeline_result):
        """Events creates events.json."""
        job_dir = pipeline_result["job_dir"]
        assert (job_dir / "events" / "events.json").exists()


class TestRealContent:
    """Verify real content (Commit 6 replaces stubs)."""

    def test_sqi_has_metrics(self, pipeline_result):
        """SQI has metrics with locked keys."""
        job_dir = pipeline_result["job_dir"]
        sqi = json.loads((job_dir / "sqi" / "sqi.json").read_text())
        
        assert "metrics" in sqi
        metrics = sqi["metrics"]
        
        # Locked metric set
        expected_keys = {
            "duration_sec",
            "sample_rate_hz",
            "num_samples",
            "rms",
            "peak_abs",
            "zero_crossing_rate",
            "speech_ratio",
        }
        assert set(metrics.keys()) == expected_keys
        
        # All values are finite
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            if isinstance(value, float):
                import math
                assert math.isfinite(value), f"{key} is not finite"

    def test_diarization_structure(self, pipeline_result):
        """Diarization has proper structure."""
        job_dir = pipeline_result["job_dir"]
        diarization = json.loads((job_dir / "diarization" / "diarization.json").read_text())
        
        assert diarization["sample_rate"] == 16000
        assert "speakers" in diarization
        
        # If there are speakers, verify structure
        for speaker in diarization["speakers"]:
            assert speaker["speaker_id"] == "SPEAKER_00"
            assert "segments" in speaker
            for seg in speaker["segments"]:
                assert "start_s" in seg
                assert "end_s" in seg

    def test_events_structure(self, pipeline_result):
        """Events has proper structure."""
        job_dir = pipeline_result["job_dir"]
        events = json.loads((job_dir / "events" / "events.json").read_text())
        
        assert "events" in events
        
        # Each event has required fields
        for event in events["events"]:
            assert event["type"] == "impulsive_sound"
            assert "start_s" in event
            assert "end_s" in event
            assert event["confidence"] == 1.0


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
