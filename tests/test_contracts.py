"""
SoundMind v1 Contract Tests - Commit 5

Tests for stage contracts, centralized validation, and deterministic metadata.

Coverage:
- Missing required artifacts → stage does not run (ValidationError)
- Incorrect role usage → validation error
- Stage status.json schema correctness
- Context immutability (frozen dataclass)
- Deterministic ordering of artifacts & metadata
- Contract enforcement (produced roles match contract)
"""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from soundmind.contracts import (
    PIPELINE_VERSION,
    Stage,
    StageContract,
    StageContext,
    StageValidator,
    ValidationError,
)
from soundmind.stages.base import ArtifactRef, build_artifact_ref


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_artifacts() -> list[ArtifactRef]:
    """Create sample artifacts for testing."""
    return [
        build_artifact_ref(
            path="input/original.wav",
            artifact_type="audio/wav",
            role="audio/original",
            description="Test original audio",
        ),
        build_artifact_ref(
            path="separation/stems/speech.wav",
            artifact_type="audio/wav",
            role="audio/speech",
            description="Test speech stem",
        ),
        build_artifact_ref(
            path="separation/stems/residual.wav",
            artifact_type="audio/wav",
            role="audio/residual",
            description="Test residual stem",
        ),
        build_artifact_ref(
            path="sqi/sqi.json",
            artifact_type="application/json",
            role="metadata/sqi",
            description="Test SQI",
        ),
    ]


@pytest.fixture
def validator() -> StageValidator:
    """Create a validator instance."""
    return StageValidator()


# =============================================================================
# Test: Validation - Missing Required Artifacts
# =============================================================================


class TestValidationMissingArtifacts:
    """Test that missing required artifacts cause validation failure."""

    def test_missing_required_artifact_raises_validation_error(self, validator):
        """Stage with missing required input raises ValidationError."""
        contract = StageContract(
            name="test_stage",
            requires=frozenset({"audio/speech", "audio/residual"}),
            produces=frozenset({"metadata/test"}),
            version="1.0.0",
        )
        
        # Only provide audio/speech, missing audio/residual
        available = [
            build_artifact_ref(
                path="test.wav",
                artifact_type="audio/wav",
                role="audio/speech",
                description="Test",
            )
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(contract, available)
        
        error = exc_info.value
        assert error.stage == "test_stage"
        assert "audio/residual" in error.missing_roles
        assert "audio/speech" in error.available_roles

    def test_validation_passes_when_all_roles_present(self, validator, sample_artifacts):
        """Validation passes when all required roles are available."""
        contract = StageContract(
            name="test_stage",
            requires=frozenset({"audio/original", "audio/speech"}),
            produces=frozenset({"metadata/test"}),
            version="1.0.0",
        )
        
        # Should not raise
        validator.validate(contract, sample_artifacts)

    def test_empty_requires_always_passes(self, validator):
        """Stage with no requirements always passes validation."""
        contract = StageContract(
            name="ingest",
            requires=frozenset(),
            produces=frozenset({"audio/original"}),
            version="1.0.0",
        )
        
        # Should not raise even with no artifacts
        validator.validate(contract, [])


# =============================================================================
# Test: Validation - Role/Type Compatibility
# =============================================================================


class TestValidationRoleTypeCompatibility:
    """Test role/type compatibility validation."""

    def test_audio_role_with_wrong_type_raises_error(self, validator):
        """audio/* roles must have audio type."""
        contract = StageContract(
            name="test_stage",
            requires=frozenset({"audio/speech"}),
            produces=frozenset(),
            version="1.0.0",
        )
        
        # Wrong type for audio role
        available = [
            build_artifact_ref(
                path="test.json",
                artifact_type="application/json",  # Wrong! Should be audio/*
                role="audio/speech",
                description="Test",
            )
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(contract, available)
        
        error = exc_info.value
        assert len(error.type_errors) > 0
        assert "audio/speech" in error.type_errors[0]

    def test_metadata_role_with_wrong_type_raises_error(self, validator):
        """metadata/* roles must have application/json type."""
        contract = StageContract(
            name="test_stage",
            requires=frozenset({"metadata/sqi"}),
            produces=frozenset(),
            version="1.0.0",
        )
        
        # Wrong type for metadata role
        available = [
            build_artifact_ref(
                path="test.wav",
                artifact_type="audio/wav",  # Wrong! Should be application/json
                role="metadata/sqi",
                description="Test",
            )
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(contract, available)
        
        error = exc_info.value
        assert len(error.type_errors) > 0
        assert "metadata/sqi" in error.type_errors[0]

    def test_correct_types_pass_validation(self, validator, sample_artifacts):
        """Correct type/role combinations pass validation."""
        contract = StageContract(
            name="test_stage",
            requires=frozenset({"audio/original", "metadata/sqi"}),
            produces=frozenset(),
            version="1.0.0",
        )
        
        # Should not raise - all types are correct
        validator.validate(contract, sample_artifacts)


# =============================================================================
# Test: Context Immutability
# =============================================================================


class TestContextImmutability:
    """Test that StageContext is immutable."""

    def test_stage_context_is_frozen(self, tmp_path):
        """StageContext cannot be mutated."""
        ctx = StageContext(
            job_id="test-job",
            input_audio=tmp_path / "test.wav",
            workspace=tmp_path,
            artifacts=(),
            pipeline_version="1.0.0",
        )
        
        with pytest.raises(FrozenInstanceError):
            ctx.job_id = "changed"

    def test_artifacts_is_tuple_not_list(self, tmp_path):
        """Context artifacts is a tuple (immutable), not list."""
        artifacts = (
            build_artifact_ref("a.wav", "audio/wav", "audio/original", "Test"),
        )
        
        ctx = StageContext(
            job_id="test-job",
            input_audio=tmp_path / "test.wav",
            workspace=tmp_path,
            artifacts=artifacts,
            pipeline_version="1.0.0",
        )
        
        assert isinstance(ctx.artifacts, tuple)

    def test_stage_contract_is_frozen(self):
        """StageContract cannot be mutated."""
        contract = StageContract(
            name="test",
            requires=frozenset({"audio/original"}),
            produces=frozenset({"audio/speech"}),
            version="1.0.0",
        )
        
        with pytest.raises(FrozenInstanceError):
            contract.name = "changed"


# =============================================================================
# Test: ArtifactRef Immutability
# =============================================================================


class TestArtifactRefImmutability:
    """Test that ArtifactRef is immutable."""

    def test_artifact_ref_is_frozen(self):
        """ArtifactRef cannot be mutated."""
        artifact = build_artifact_ref(
            path="test.wav",
            artifact_type="audio/wav",
            role="audio/original",
            description="Test",
        )
        
        with pytest.raises(FrozenInstanceError):
            artifact.path = "changed.wav"

    def test_artifact_ref_to_dict(self):
        """ArtifactRef.to_dict() returns correct dictionary."""
        artifact = build_artifact_ref(
            path="test.wav",
            artifact_type="audio/wav",
            role="audio/original",
            description="Test description",
        )
        
        d = artifact.to_dict()
        assert d == {
            "path": "test.wav",
            "type": "audio/wav",
            "role": "audio/original",
            "description": "Test description",
        }


# =============================================================================
# Test: Stage Status JSON Schema (Commit 5 Format)
# =============================================================================


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run soundmind CLI as subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "soundmind", *args],
        capture_output=True,
        text=True,
    )


class TestStageStatusSchema:
    """Test enhanced stage status.json schema correctness."""

    @pytest.fixture
    def pipeline_result(self, tmp_path):
        """Run pipeline and return job directory."""
        from tests.conftest import create_test_wav
        input_file = tmp_path / "test_input.wav"
        create_test_wav(input_file, duration_sec=0.5)
        
        job_id = "status-schema-test"
        jobs_root = tmp_path / "jobs"
        
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        return jobs_root / job_id

    def test_status_json_has_artifacts_key(self, pipeline_result):
        """Each stage status.json has artifacts key."""
        for stage in ["ingest", "separation", "sqi", "diarization", "events", "rollup"]:
            status_path = pipeline_result / stage / "status.json"
            assert status_path.exists()
            status = json.loads(status_path.read_text())
            assert "artifacts" in status

    def test_artifact_roles_use_prefix_format(self, pipeline_result):
        """All artifact roles use audio/* or metadata/* prefix."""
        for stage in ["ingest", "separation", "sqi", "diarization", "events"]:
            status = json.loads((pipeline_result / stage / "status.json").read_text())
            for artifact in status.get("artifacts", []):
                role = artifact["role"]
                assert role.startswith("audio/") or role.startswith("metadata/"), \
                    f"Role '{role}' in {stage} doesn't have prefix"


# =============================================================================
# Test: Deterministic Ordering
# =============================================================================


class TestDeterministicOrdering:
    """Test that artifacts and metadata have deterministic ordering."""

    @pytest.fixture
    def pipeline_result(self, tmp_path):
        """Run pipeline and return job directory."""
        from tests.conftest import create_test_wav
        input_file = tmp_path / "test_input.wav"
        create_test_wav(input_file, duration_sec=0.5)
        
        job_id = "ordering-test"
        jobs_root = tmp_path / "jobs"
        
        result = run_cli(
            "run",
            "--input", str(input_file),
            "--jobs-root", str(jobs_root),
            "--job-id", job_id,
        )
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
        return jobs_root / job_id

    def test_rollup_artifacts_in_stage_order(self, pipeline_result):
        """Rollup artifacts follow stage execution order."""
        status = json.loads((pipeline_result / "rollup" / "status.json").read_text())
        paths = [a["path"] for a in status["artifacts"]]
        
        # Verify order: ingest → separation → sqi → diarization → events
        assert paths[0].startswith("input/")  # ingest
        assert paths[1].startswith("separation/")
        assert paths[2].startswith("separation/")
        assert paths[3].startswith("sqi/")
        assert paths[4].startswith("diarization/")
        assert paths[5].startswith("events/")

    def test_artifact_dict_keys_sorted(self, pipeline_result):
        """JSON output has sorted keys for determinism."""
        status_text = (pipeline_result / "ingest" / "status.json").read_text()
        status = json.loads(status_text)
        
        # Check that when re-serialized with sort_keys, it matches
        reserialized = json.dumps(status, indent=2, sort_keys=True) + "\n"
        assert status_text == reserialized


# =============================================================================
# Test: Contract Definitions
# =============================================================================


class TestContractDefinitions:
    """Test that stage contracts are defined correctly."""

    def test_ingest_contract(self):
        """Ingest contract: requires={}, produces={audio/original}."""
        from soundmind.stages.ingest import CONTRACT
        
        assert CONTRACT.name == "ingest"
        assert CONTRACT.requires == frozenset()
        assert CONTRACT.produces == frozenset({"audio/original"})

    def test_separation_contract(self):
        """Separation contract: requires={audio/original}, produces={audio/speech, audio/residual}."""
        from soundmind.stages.separation import CONTRACT
        
        assert CONTRACT.name == "separation"
        assert CONTRACT.requires == frozenset({"audio/original"})
        assert CONTRACT.produces == frozenset({"audio/speech", "audio/residual"})

    def test_sqi_contract(self):
        """SQI contract: requires={audio/speech}, produces={metadata/sqi}."""
        from soundmind.stages.sqi import CONTRACT
        
        assert CONTRACT.name == "sqi"
        assert CONTRACT.requires == frozenset({"audio/speech"})
        assert CONTRACT.produces == frozenset({"metadata/sqi"})

    def test_diarization_contract(self):
        """Diarization contract: requires={audio/speech}, produces={metadata/diarization}."""
        from soundmind.stages.diarization import CONTRACT
        
        assert CONTRACT.name == "diarization"
        assert CONTRACT.requires == frozenset({"audio/speech"})
        assert CONTRACT.produces == frozenset({"metadata/diarization"})

    def test_events_contract(self):
        """Events contract: requires={audio/residual}, produces={metadata/events}."""
        from soundmind.stages.events import CONTRACT
        
        assert CONTRACT.name == "events"
        assert CONTRACT.requires == frozenset({"audio/residual"})
        assert CONTRACT.produces == frozenset({"metadata/events"})

    def test_rollup_contract(self):
        """Rollup contract: requires all 6 artifacts, produces={}."""
        from soundmind.stages.rollup import CONTRACT
        
        assert CONTRACT.name == "rollup"
        assert CONTRACT.requires == frozenset({
            "audio/original",
            "audio/speech",
            "audio/residual",
            "metadata/sqi",
            "metadata/diarization",
            "metadata/events",
        })
        assert CONTRACT.produces == frozenset()


# =============================================================================
# Test: ValidationError Structure
# =============================================================================


class TestValidationErrorStructure:
    """Test ValidationError provides actionable information."""

    def test_validation_error_has_all_fields(self, validator):
        """ValidationError includes missing, available, and type errors."""
        contract = StageContract(
            name="test_stage",
            requires=frozenset({"audio/missing"}),
            produces=frozenset(),
            version="1.0.0",
        )
        
        available = [
            build_artifact_ref("a.wav", "audio/wav", "audio/present", "Test")
        ]
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(contract, available)
        
        error = exc_info.value
        assert error.stage == "test_stage"
        assert error.missing_roles == {"audio/missing"}
        assert error.available_roles == {"audio/present"}
        assert isinstance(error.type_errors, list)

    def test_validation_error_message_is_readable(self, validator):
        """ValidationError message is human-readable."""
        contract = StageContract(
            name="sqi",
            requires=frozenset({"audio/speech"}),
            produces=frozenset(),
            version="1.0.0",
        )
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(contract, [])
        
        message = str(exc_info.value)
        assert "sqi" in message
        assert "audio/speech" in message
