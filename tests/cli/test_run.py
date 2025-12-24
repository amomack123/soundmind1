"""
SoundMind v1 CLI Tests - Commit 2

Black-box subprocess tests only. No imports from soundmind.cli.
"""

import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

# Import schema validation from tools (allowed for test verification)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
from validate_schema import load_schema, validate_document


def run_cli(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run soundmind CLI as subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "soundmind", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


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
        assert "--job-id JOB_ID" in result.stdout
        assert "--jobs-root PATH" in result.stdout
        assert "--dry-run" in result.stdout
        assert "No audio processing, machine learning, or pipeline" in result.stdout


class TestRunCommand:
    """Test soundmind run command."""

    def test_run_creates_job_directory(self, tmp_path):
        """soundmind run creates jobs/<job_id>/"""
        result = run_cli("run", "--jobs-root", str(tmp_path), cwd=str(tmp_path))
        assert result.returncode == 0

        # Should have created exactly one directory
        job_dirs = list(tmp_path.iterdir())
        assert len(job_dirs) == 1
        assert job_dirs[0].is_dir()

    def test_run_respects_job_id(self, tmp_path):
        """--job-id is respected."""
        job_id = "my-test-job-123"
        result = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0
        assert (tmp_path / job_id).is_dir()

    def test_run_collision_fails(self, tmp_path):
        """Collision fails cleanly."""
        job_id = "duplicate-job"
        
        # First run succeeds
        result1 = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        assert result1.returncode == 0
        
        # Second run with same job_id fails
        result2 = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        assert result2.returncode != 0
        assert "exists" in result2.stderr.lower() or "collision" in result2.stderr.lower()


class TestStatusJson:
    """Test status.json creation and validation."""

    def test_status_json_exists(self, tmp_path):
        """status.json exists in job directory."""
        job_id = "status-test-job"
        run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        
        status_path = tmp_path / job_id / "status.json"
        assert status_path.exists()

    def test_status_json_validates_against_schema(self, tmp_path):
        """status.json validates against frozen schema."""
        job_id = "validation-test-job"
        run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        
        status_path = tmp_path / job_id / "status.json"
        with open(status_path) as f:
            status = json.load(f)
        
        schema = load_schema("status")
        errors = validate_document(status, schema)
        assert errors == [], f"Schema validation errors: {errors}"

    def test_status_json_has_exact_shape(self, tmp_path):
        """status.json has exact required shape."""
        job_id = "shape-test-job"
        run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        
        status_path = tmp_path / job_id / "status.json"
        with open(status_path) as f:
            status = json.load(f)
        
        # Verify exact structure
        assert status["job_id"] == job_id
        assert "created_at" in status
        assert status["input"] == {"original_wav": "", "sha256": ""}
        assert status["stages"] == {
            "separation": None,
            "diarization": None,
            "events": None,
        }

    def test_all_stage_values_are_null(self, tmp_path):
        """All stage values are null."""
        job_id = "null-stages-job"
        run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        
        status_path = tmp_path / job_id / "status.json"
        with open(status_path) as f:
            status = json.load(f)
        
        for stage_name, stage_value in status["stages"].items():
            assert stage_value is None, f"Stage {stage_name} should be null"


class TestDryRun:
    """Test --dry-run flag."""

    def test_dry_run_no_filesystem_writes(self, tmp_path):
        """--dry-run creates no files or directories."""
        result = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", "dry-run-job",
            "--dry-run",
        )
        assert result.returncode == 0
        
        # No directories should be created
        assert list(tmp_path.iterdir()) == []

    def test_dry_run_prints_job_id(self, tmp_path):
        """--dry-run prints resolved job ID."""
        job_id = "dry-run-print-job"
        result = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
            "--dry-run",
        )
        assert result.returncode == 0
        assert job_id in result.stdout

    def test_dry_run_prints_planned_paths(self, tmp_path):
        """--dry-run prints planned paths."""
        job_id = "dry-run-paths-job"
        result = run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
            "--dry-run",
        )
        assert result.returncode == 0
        assert "status.json" in result.stdout


class TestNegativeGuarantees:
    """Test negative guarantees - what must NOT happen."""

    def test_no_stage_modules_imported(self):
        """No stage modules imported during CLI execution."""
        # Run a command that would trigger any imports
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

    def test_no_pipeline_imported(self):
        """soundmind.pipeline not imported during CLI execution."""
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

    def test_directory_contains_only_status_json(self, tmp_path):
        """Job directory contains only status.json."""
        job_id = "only-status-job"
        run_cli(
            "run",
            "--jobs-root", str(tmp_path),
            "--job-id", job_id,
        )
        
        job_dir = tmp_path / job_id
        files = list(job_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == "status.json"
