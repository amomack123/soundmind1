#!/usr/bin/env python3
"""
SoundMind v1 Schema Validation Tool

Standalone utility for validating JSON documents against SoundMind v1 schemas.
Located outside the runtime package (tools/) per project requirements.

Usage:
    python tools/validate_schema.py <schema_name> <json_file>

Where schema_name is one of: status, diarization, events

Example:
    python tools/validate_schema.py status output/job_status.json
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema
except ImportError:
    sys.exit("Error: jsonschema package required. Install with: pip install jsonschema")


SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

SCHEMA_FILES = {
    "status": "status.schema.json",
    "diarization": "diarization.schema.json",
    "events": "events.schema.json",
}


def load_schema(schema_name: str) -> dict:
    """Load a schema by name."""
    if schema_name not in SCHEMA_FILES:
        raise ValueError(f"Unknown schema: {schema_name}. Valid: {list(SCHEMA_FILES.keys())}")
    
    schema_path = SCHEMA_DIR / SCHEMA_FILES[schema_name]
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_document(document: dict, schema: dict) -> list[str]:
    """
    Validate a document against a schema.
    
    Returns:
        List of error messages (empty if valid).
    """
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for error in validator.iter_errors(document):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"{path}: {error.message}")
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate JSON against SoundMind v1 schemas"
    )
    parser.add_argument(
        "schema",
        choices=list(SCHEMA_FILES.keys()),
        help="Schema to validate against",
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to JSON file to validate",
    )
    
    args = parser.parse_args()
    
    # Load schema
    try:
        schema = load_schema(args.schema)
    except FileNotFoundError:
        sys.exit(f"Error: Schema file not found: {SCHEMA_DIR / SCHEMA_FILES[args.schema]}")
    
    # Load document
    try:
        with open(args.json_file, "r") as f:
            document = json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: File not found: {args.json_file}")
    except json.JSONDecodeError as e:
        sys.exit(f"Error: Invalid JSON: {e}")
    
    # Validate
    errors = validate_document(document, schema)
    
    if errors:
        print(f"INVALID: {len(errors)} error(s) found:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print(f"VALID: Document conforms to {args.schema} schema.")
        sys.exit(0)


if __name__ == "__main__":
    main()
