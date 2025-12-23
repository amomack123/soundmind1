"""
SoundMind v1 Pipeline Stages

Fixed order — DO NOT add, remove, reorder, or rename:
    A. ingest       — Ingest & Normalize
    B. separation   — Speech vs Residual Separation
    C. sqi          — Signal Quality Indicators
    D. diarization  — Speaker Diarization (speech only)
    E. events       — Acoustic Event Candidates (trigger-only)
    F. rollup       — Final Roll-Up
"""
