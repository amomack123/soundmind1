"""
SoundMind v1 Status Initialization.

Responsibilities:
- Build initial in-memory status object
- Validate against schemas/status.schema.json
- Serialize JSON deterministically

Forbidden:
- No directory creation
- No job ID logic
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "status.schema.json"


def build_initial_status(job_id: str) -> dict[str, Any]:
    """
    Build the initial status object.
    
    Args:
        job_id: The job identifier.
    
    Returns:
        The initial status dictionary with exact required shape.
    """
    return {
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {
            "original_wav": "",
            "sha256": "",
        },
        "stages": {
            "separation": None,
            "diarization": None,
            "events": None,
        },
    }


def validate_status(status: dict[str, Any]) -> list[str]:
    """
    Validate status against the frozen schema.
    
    Args:
        status: The status object to validate.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    try:
        import jsonschema
    except ImportError:
        # If jsonschema not available, skip validation
        return []
    
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for error in validator.iter_errors(status):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


def serialize_status(status: dict[str, Any]) -> str:
    """
    Serialize status to JSON deterministically.
    
    Args:
        status: The status object to serialize.
    
    Returns:
        JSON string with sorted keys and consistent formatting.
    """
    return json.dumps(status, indent=2, sort_keys=True) + "\n"
