"""
SoundMind v1 Utilities - Shared helper functions.

Responsibilities:
- Time formatting with explicit PST offset
- JSON serialization helpers

Invariants:
- PST offset is fixed at -08:00 (no DST handling)
- All timestamps use ISO-8601 format
"""

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Mapping


# Explicit PST timezone (no DST, fixed -08:00)
PST = timezone(timedelta(hours=-8))


def now_iso() -> str:
    """
    Return current time as ISO-8601 with explicit PST offset.
    
    Returns:
        ISO-8601 formatted string, e.g., "2025-12-23T17:02:10-08:00"
    """
    return datetime.now(PST).isoformat()


def serialize_json(data: Mapping[str, Any]) -> str:
    """
    Serialize dictionary to JSON deterministically.
    
    Args:
        data: Dictionary to serialize.
    
    Returns:
        JSON string with sorted keys, 2-space indent, trailing newline.
    """
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
