"""
SoundMind v1 Pipeline Orchestrator (STUB)

This module documents the fixed stage order and invariants
for the SoundMind v1 pipeline. It intentionally contains
no executable orchestration logic in Commit 1.

PIPELINE STAGES (FIXED ORDER — DO NOT MODIFY):

    A. Ingest & Normalize      → soundmind.stages.ingest
    B. Separation              → soundmind.stages.separation
    C. Signal Quality (SQI)    → soundmind.stages.sqi
    D. Diarization             → soundmind.stages.diarization
    E. Acoustic Events         → soundmind.stages.events
    F. Final Roll-Up           → soundmind.stages.rollup

INVARIANTS:
    - Stages execute in order A → F
    - Event candidates (E) only run inside non_speech segments
    - Same input + same version = identical output
    - All timestamps are seconds (float)

SCHEMAS (FROZEN):
    - schemas/status.schema.json
    - schemas/diarization.schema.json
    - schemas/events.schema.json
"""

# No implementation in Commit 1.
# This file exists to document the pipeline contract.
