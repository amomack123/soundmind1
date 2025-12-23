"""
SoundMind v1 Schema Validation Tests

Tests that frozen schemas are valid and correctly validate documents.
"""

import json
from pathlib import Path

import pytest

# Import validation functions from tools
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from validate_schema import load_schema, validate_document, SCHEMA_FILES


SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class TestSchemaLoading:
    """Test that all schemas load correctly."""

    def test_status_schema_loads(self):
        schema = load_schema("status")
        assert schema["title"] == "SoundMind v1 Status"
        assert "job_id" in schema["required"]

    def test_diarization_schema_loads(self):
        schema = load_schema("diarization")
        assert schema["title"] == "SoundMind v1 Diarization"
        assert "speakers" in schema["required"]

    def test_events_schema_loads(self):
        schema = load_schema("events")
        assert schema["title"] == "SoundMind v1 Acoustic Event Candidates"
        assert "events" in schema["required"]

    def test_invalid_schema_name_raises(self):
        with pytest.raises(ValueError, match="Unknown schema"):
            load_schema("invalid_schema")


class TestStatusSchemaValidation:
    """Test status.schema.json validation."""

    @pytest.fixture
    def schema(self):
        return load_schema("status")

    def test_valid_minimal_status(self, schema):
        doc = {
            "job_id": "test-123",
            "created_at": "2024-01-01T00:00:00Z",
            "input": {
                "original_wav": "/path/to/audio.wav",
                "sha256": "abc123def456"
            },
            "stages": {
                "separation": None,
                "diarization": None,
                "events": None
            }
        }
        errors = validate_document(doc, schema)
        assert errors == []

    def test_valid_complete_status(self, schema):
        doc = {
            "job_id": "test-456",
            "created_at": "2024-01-01T12:00:00Z",
            "input": {
                "original_wav": "/data/original.wav",
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            },
            "stages": {
                "separation": {
                    "speech_wav": "/output/speech.wav",
                    "residual_wav": "/output/residual.wav",
                    "metrics": {
                        "snr_proxy_db": 25.5,
                        "clipping_pct": 0.01,
                        "dropout_pct": 0.0,
                        "reverb_proxy_rt60_ms": 300.0,
                        "loudness_lufs": -23.0,
                        "rms_dbfs": -18.5
                    }
                },
                "diarization": {
                    "diarization_json": "/output/diarization.json",
                    "rttm": "/output/diarization.rttm",
                    "per_speaker": {
                        "speaker_0": "/output/speaker_0.wav",
                        "speaker_1": "/output/speaker_1.wav"
                    },
                    "metrics": None
                },
                "events": {
                    "events_json": "/output/events.json"
                }
            }
        }
        errors = validate_document(doc, schema)
        assert errors == []

    def test_missing_job_id_fails(self, schema):
        doc = {
            "created_at": "2024-01-01T00:00:00Z",
            "input": {"original_wav": "/path.wav", "sha256": "abc"},
            "stages": {"separation": None, "diarization": None, "events": None}
        }
        errors = validate_document(doc, schema)
        assert len(errors) == 1
        assert "job_id" in errors[0]

    def test_extra_property_fails(self, schema):
        doc = {
            "job_id": "test",
            "created_at": "2024-01-01T00:00:00Z",
            "input": {"original_wav": "/path.wav", "sha256": "abc"},
            "stages": {"separation": None, "diarization": None, "events": None},
            "extra_field": "not_allowed"
        }
        errors = validate_document(doc, schema)
        assert len(errors) == 1
        assert "extra_field" in errors[0]


class TestDiarizationSchemaValidation:
    """Test diarization.schema.json validation."""

    @pytest.fixture
    def schema(self):
        return load_schema("diarization")

    def test_valid_diarization(self, schema):
        doc = {
            "sample_rate": 16000,
            "speakers": [
                {
                    "speaker_id": "speaker_0",
                    "segments": [
                        {"start_s": 0.0, "end_s": 5.5},
                        {"start_s": 10.2, "end_s": 15.8}
                    ]
                },
                {
                    "speaker_id": "speaker_1",
                    "segments": [
                        {"start_s": 5.5, "end_s": 10.2}
                    ]
                }
            ]
        }
        errors = validate_document(doc, schema)
        assert errors == []

    def test_valid_with_notes(self, schema):
        doc = {
            "sample_rate": 44100,
            "speakers": [],
            "notes": "No speakers detected in audio"
        }
        errors = validate_document(doc, schema)
        assert errors == []

    def test_timestamps_must_be_numbers(self, schema):
        doc = {
            "sample_rate": 16000,
            "speakers": [{
                "speaker_id": "speaker_0",
                "segments": [{"start_s": "0.0", "end_s": 5.0}]  # string not allowed
            }]
        }
        errors = validate_document(doc, schema)
        assert len(errors) == 1

    def test_sample_rate_must_be_integer(self, schema):
        doc = {
            "sample_rate": 16000.5,  # float not allowed
            "speakers": []
        }
        errors = validate_document(doc, schema)
        assert len(errors) == 1


class TestEventsSchemaValidation:
    """Test events.schema.json validation."""

    @pytest.fixture
    def schema(self):
        return load_schema("events")

    def test_valid_events(self, schema):
        doc = {
            "events": [
                {
                    "type": "impulsive_sound",
                    "start_s": 12.5,
                    "end_s": 12.8,
                    "confidence": 0.85
                },
                {
                    "type": "tonal_alarm_like",
                    "start_s": 45.0,
                    "end_s": 48.5,
                    "confidence": 0.92
                },
                {
                    "type": "vehicle_like",
                    "start_s": 100.0,
                    "end_s": 105.0,
                    "confidence": 0.6
                }
            ]
        }
        errors = validate_document(doc, schema)
        assert errors == []

    def test_empty_events_valid(self, schema):
        doc = {"events": []}
        errors = validate_document(doc, schema)
        assert errors == []

    def test_invalid_event_type_fails(self, schema):
        doc = {
            "events": [{
                "type": "gunshot",  # not in enum
                "start_s": 0.0,
                "end_s": 0.5,
                "confidence": 0.9
            }]
        }
        errors = validate_document(doc, schema)
        assert len(errors) == 1
        assert "gunshot" in errors[0] or "enum" in errors[0].lower()

    def test_confidence_out_of_range_fails(self, schema):
        doc = {
            "events": [{
                "type": "impulsive_sound",
                "start_s": 0.0,
                "end_s": 0.5,
                "confidence": 1.5  # > 1.0
            }]
        }
        errors = validate_document(doc, schema)
        assert len(errors) == 1

    def test_missing_required_field_fails(self, schema):
        doc = {
            "events": [{
                "type": "impulsive_sound",
                "start_s": 0.0,
                # missing end_s and confidence
            }]
        }
        errors = validate_document(doc, schema)
        assert len(errors) >= 1


class TestSchemaInvariants:
    """Test that schemas enforce SoundMind v1 invariants."""

    def test_no_semantic_event_types_allowed(self):
        """Verify event types are trigger-only, non-semantic."""
        schema = load_schema("events")
        allowed_types = schema["properties"]["events"]["items"]["properties"]["type"]["enum"]
        
        # These semantic labels must NOT be allowed
        forbidden = ["gunshot", "siren", "shouting", "scream", "explosion", "glass_break"]
        for label in forbidden:
            assert label not in allowed_types, f"Semantic label '{label}' should not be in schema"
        
        # Only these three trigger types allowed
        assert set(allowed_types) == {"impulsive_sound", "tonal_alarm_like", "vehicle_like"}

    def test_timestamps_are_floats(self):
        """Verify timestamp fields accept float values."""
        diarization = load_schema("diarization")
        segment_props = diarization["properties"]["speakers"]["items"]["properties"]["segments"]["items"]["properties"]
        
        assert segment_props["start_s"]["type"] == "number"
        assert segment_props["end_s"]["type"] == "number"

        events = load_schema("events")
        event_props = events["properties"]["events"]["items"]["properties"]
        
        assert event_props["start_s"]["type"] == "number"
        assert event_props["end_s"]["type"] == "number"
