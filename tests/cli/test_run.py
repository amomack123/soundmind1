"""
SoundMind v1 CLI Tests

Black-box subprocess tests only. No imports from soundmind.cli.

Updated for Commit 2.5+ CLI contract:
- --input is required
- Pipeline runs after workspace creation
- status.json reflects pipeline completion
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run soundmind CLI as subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "soundmind", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


@pytest.fixture
def input_file(tmp_path):
    """Create a valid test WAV file for testing."""
    from tests.conftest import create_test_wav
    input_path = tmp_path / "test_input.wav"
    create_test_wav(input_path, duration_sec=0.5)  # Short for faster tests
    return input_path


class TestHelpText:
    """Verify frozen help text."""

    def test_main_help(self):
        result = run_cli("--help")
        assert result.returncode == 0
        assert "SoundMind v1 command-line interface." in result.stdout
        assert "{run}" in result.stdout
        assert "Initialize a new SoundMind job workspace." in result.stdout

    def test_run_help(self):
        result = run_cli("run", "--help")
        assert result.returncode == 0
        assert "Initialize a new SoundMind job workspace." in result.stdout
        assert "--input PATH" in result.stdout
        assert "--job-id JOB_ID" in result.stdout
        assert "--jobs-root PATH" in result.stdout
        assert "--dry-run" in result.stdout


class TestRunCommand:
    """Test soundmind run command."""

    def test_run_creates_job_directory(self, tmp_path, input_file):
        """soundmind run creates jobs/<job_id>/"""
        jobs_root = tmp_path / "jobs"
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
        )
        assert result.returncode == 0

        # Should have created exactly one job directory
        job_dirs = list(jobs_root.iterdir())
        assert len(job_dirs) == 1
        assert job_dirs[0].is_dir()

    def test_run_respects_job_id(self, tmp_path, input_file):
        """--job-id is respected."""
        job_id = "my-test-job-123"
        jobs_root = tmp_path / "jobs"
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0
        assert (jobs_root / job_id).is_dir()

    def test_run_collision_fails(self, tmp_path, input_file):
        """Collision fails cleanly."""
        job_id = "duplicate-job"
        jobs_root = tmp_path / "jobs"
        
        # First run succeeds
        result1 = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result1.returncode == 0
        
        # Second run with same job_id fails
        result2 = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result2.returncode != 0
        assert "exists" in result2.stderr.lower()

    def test_run_requires_input(self, tmp_path):
        """--input is required."""
        result = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", "test-job",
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_run_fails_on_missing_input(self, tmp_path):
        """Fails if input file doesn't exist."""
        result = run_cli(
            "run",
            "--input", "/nonexistent/file.wav",
            "--jobs-root", str(tmp_path),
            "--job-id", "test-job",
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestStatusJson:
    """Test status.json creation after pipeline run."""

    def test_status_json_exists(self, tmp_path, input_file):
        """status.json exists in job directory."""
        job_id = "status-test-job"
        jobs_root = tmp_path / "jobs"
        run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        
        status_path = jobs_root / job_id / "status.json"
        assert status_path.exists()

    def test_status_json_has_job_id(self, tmp_path, input_file):
        """status.json contains correct job_id."""
        job_id = "shape-test-job"
        jobs_root = tmp_path / "jobs"
        run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        
        status_path = jobs_root / job_id / "status.json"
        status = json.loads(status_path.read_text())
        
        assert status["job_id"] == job_id
        assert status["version"] == "v1"

    def test_status_json_has_success(self, tmp_path, input_file):
        """status.json indicates pipeline success."""
        job_id = "success-test-job"
        jobs_root = tmp_path / "jobs"
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0
        
        status_path = jobs_root / job_id / "status.json"
        status = json.loads(status_path.read_text())
        
        assert status["success"] is True
        assert status["failed_stage"] is None

    def test_status_json_has_stages(self, tmp_path, input_file):
        """status.json contains stage status references."""
        job_id = "stages-test-job"
        jobs_root = tmp_path / "jobs"
        run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        
        status_path = jobs_root / job_id / "status.json"
        status = json.loads(status_path.read_text())
        
        # Pipeline runs all stages
        expected_stages = ["ingest", "separation", "sqi", "diarization", "events", "rollup"]
        for stage in expected_stages:
            assert stage in status["stages"]


class TestDryRun:
    """Test --dry-run flag."""

    def test_dry_run_no_filesystem_writes(self, tmp_path, input_file):
        """--dry-run creates no files or directories."""
        jobs_root = tmp_path / "jobs"
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", "dry-run-job",
            "--dry-run",
        )
        assert result.returncode == 0
        
        # jobs_root should not exist
        assert not jobs_root.exists()

    def test_dry_run_prints_job_id(self, tmp_path, input_file):
        """--dry-run prints resolved job ID."""
        job_id = "dry-run-print-job"
        jobs_root = tmp_path / "jobs"
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
            "--dry-run",
        )
        assert result.returncode == 0
        assert job_id in result.stdout

    def test_dry_run_prints_input_path(self, tmp_path, input_file):
        """--dry-run prints input file path."""
        jobs_root = tmp_path / "jobs"
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", "dry-run-paths-job",
            "--dry-run",
        )
        assert result.returncode == 0
        assert str(input_file) in result.stdout or input_file.name in result.stdout


class TestNegativeGuarantees:
    """Test negative guarantees - what must NOT happen."""

    def test_no_stage_modules_imported_at_parse_time(self):
        """No stage modules imported during CLI module load."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; import soundmind.cli; "
                "stage_modules = [k for k in sys.modules if 'soundmind.stages' in k]; "
                "print(','.join(stage_modules) if stage_modules else 'NONE')"
            ],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "NONE", f"Stage modules imported: {result.stdout}"

    def test_no_pipeline_imported_at_parse_time(self):
        """soundmind.pipeline not imported during CLI module load."""
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; import soundmind.cli; "
                "print('IMPORTED' if 'soundmind.pipeline' in sys.modules else 'NONE')"
            ],
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "NONE", "soundmind.pipeline was imported"


class TestWorkspaceStructure:
    """Test workspace directory structure."""

    def test_job_directory_has_stage_dirs(self, tmp_path, input_file):
        """Job directory contains stage subdirectories."""
        job_id = "structure-test-job"
        jobs_root = tmp_path / "jobs"
        run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        
        job_dir = jobs_root / job_id
        expected_dirs = ["meta", "input", "ingest", "separation", "sqi", "diarization", "events", "rollup"]
        for dir_name in expected_dirs:
            assert (job_dir / dir_name).is_dir(), f"Missing directory: {dir_name}"

    def test_input_file_is_copied(self, tmp_path, input_file):
        """Input file is copied to input/original.wav."""
        job_id = "copy-test-job"
        jobs_root = tmp_path / "jobs"
        run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        
        original_wav = jobs_root / job_id / "input" / "original.wav"
        assert original_wav.exists()
        assert original_wav.read_bytes() == input_file.read_bytes()
