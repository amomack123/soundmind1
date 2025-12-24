"""
SoundMind v1 Stage Contracts — Commit 5

Formal stage contracts, centralized validation, and deterministic metadata.

This module provides:
- StageContract: Frozen, declarative contract for a stage
- StageContext: Immutable execution context snapshot
- Stage: Abstract base class for all stages
- StageValidator: Centralized input validation
- ValidationError: Structured validation failure

INVARIANTS:
- Contracts are frozen and immutable
- Validation happens before stage execution
- Stages do NOT validate their own inputs
- Stages do NOT mutate context
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soundmind.stages.base import ArtifactRef


# =============================================================================
# StageContract — Frozen, Declarative
# =============================================================================


@dataclass(frozen=True)
class StageContract:
    """
    Frozen contract declaring what a stage requires and produces.
    
    Attributes:
        name: Stage identifier (e.g., "ingest", "separation")
        requires: Set of artifact roles required to run (e.g., {"audio/original"})
        produces: Set of artifact roles this stage creates (e.g., {"audio/speech"})
        version: Semantic version for reproducibility (e.g., "1.0.0")
    
    Rules:
        - No behavior, no mutation
        - One contract per stage
        - Hashable for use in sets/dicts
    """
    name: str
    requires: frozenset[str]
    produces: frozenset[str]
    version: str


# =============================================================================
# StageContext — Immutable Execution Snapshot
# =============================================================================


@dataclass(frozen=True)
class StageContext:
    """
    Immutable snapshot of execution context passed to stages.
    
    Attributes:
        job_id: Unique job identifier
        input_audio: Path to canonical input audio file
        workspace: Path to job workspace directory
        artifacts: Tuple of all artifacts produced by prior stages
        pipeline_version: Version of the pipeline for reproducibility
    
    Rules:
        - Constructed by pipeline, not stages
        - Read-only (frozen dataclass)
        - Single source of truth for stage execution
    """
    job_id: str
    input_audio: Path
    workspace: Path
    artifacts: tuple["ArtifactRef", ...]  # Immutable tuple of prior artifacts
    pipeline_version: str


# =============================================================================
# Stage — Abstract Base Class
# =============================================================================


class Stage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Subclasses must:
        - Define a `contract` class attribute of type StageContract
        - Implement `run(ctx)` returning newly created artifacts
    
    Rules:
        - Stages do NOT validate inputs (validator does that)
        - Stages do NOT mutate context
        - Stages return only newly created artifacts
    """
    
    contract: StageContract
    
    @abstractmethod
    def run(self, ctx: StageContext) -> list:
        """
        Execute the stage.
        
        Args:
            ctx: Immutable execution context
        
        Returns:
            List of ArtifactRef objects for newly created artifacts.
            Must match exactly the roles declared in contract.produces.
        """
        ...


# =============================================================================
# ValidationError — Structured Validation Failure
# =============================================================================


class ValidationError(Exception):
    """
    Raised when stage input validation fails.
    
    Provides structured, actionable error information including
    what was missing and what was available.
    
    Attributes:
        stage: Name of the stage that failed validation
        missing_roles: Roles required but not available
        available_roles: Roles that were available
        type_errors: List of role/type compatibility errors
    """
    
    def __init__(
        self,
        stage: str,
        missing_roles: set[str],
        available_roles: set[str],
        type_errors: list[str],
    ):
        self.stage = stage
        self.missing_roles = missing_roles
        self.available_roles = available_roles
        self.type_errors = type_errors
        
        # Build human-readable message
        parts = [f"Validation failed for stage '{stage}'"]
        if missing_roles:
            parts.append(f"Missing roles: {sorted(missing_roles)}")
            parts.append(f"Available roles: {sorted(available_roles)}")
        if type_errors:
            parts.append(f"Type errors: {type_errors}")
        
        super().__init__("; ".join(parts))


# =============================================================================
# StageValidator — Centralized Input Validation
# =============================================================================


class StageValidator:
    """
    Validates that stage inputs satisfy contract requirements.
    
    Validation checks:
        1. All required artifact roles exist in available artifacts
        2. Role/type compatibility (MIME-strict):
           - audio/* → artifact_type must start with "audio/"
           - metadata/* → artifact_type must be "application/json"
    
    Rules:
        - Validation happens BEFORE stage.run()
        - No side effects
        - Fail fast with structured ValidationError
    """
    
    def validate(
        self,
        contract: StageContract,
        available_artifacts: list,
    ) -> None:
        """
        Validate that available artifacts satisfy contract requirements.
        
        Args:
            contract: Stage contract declaring required roles
            available_artifacts: List of ArtifactRef from prior stages
        
        Raises:
            ValidationError: If validation fails
        
        Note:
            available_artifacts is list[ArtifactRef] but typed as list
            to avoid circular import issues.
        """
        # Extract available roles
        available_roles: set[str] = {a.role for a in available_artifacts}
        
        # Check 1: All required roles exist
        missing_roles = contract.requires - available_roles
        
        # Check 2: Role/type compatibility
        type_errors: list[str] = []
        for artifact in available_artifacts:
            role = artifact.role
            artifact_type = artifact.type
            
            if role.startswith("audio/"):
                # audio/* roles must have MIME type starting with audio/
                if not artifact_type.startswith("audio/"):
                    type_errors.append(
                        f"Role '{role}' has type '{artifact_type}', expected type starting with 'audio/'"
                    )
            elif role.startswith("metadata/"):
                # metadata/* roles must be JSON
                if artifact_type != "application/json":
                    type_errors.append(
                        f"Role '{role}' has type '{artifact_type}', expected 'application/json'"
                    )
        
        # Fail if any validation errors
        if missing_roles or type_errors:
            raise ValidationError(
                stage=contract.name,
                missing_roles=missing_roles,
                available_roles=available_roles,
                type_errors=type_errors,
            )


# =============================================================================
# Pipeline Version
# =============================================================================

PIPELINE_VERSION = "1.0.0"
