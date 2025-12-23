"""
SoundMind v1 — Forensic Audio Processing Pipeline

CONTRACTS ARE FROZEN. Do not modify schemas or stage order.

Pipeline Stages (fixed order):
    A. Ingest & Normalize
    B. Separation (speech vs residual)
    C. Signal Quality Indicators (SQI)
    D. Diarization (speech only)
    E. Acoustic Event Candidates (non-semantic, triggers only)
    F. Final Roll-Up

Invariants:
    - All timestamps are seconds (float)
    - No semantic inference (intent, emotion, deception)
    - No fine-grained acoustic taxonomy
    - Stage E is trigger-only and recall-biased
    - Same input + same version = identical output
"""

__version__ = "1.0.0.dev0"
