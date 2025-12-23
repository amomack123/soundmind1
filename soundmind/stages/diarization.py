"""
Stage D: Diarization (STUB)

Responsibilities:
    - Identify distinct speakers in the speech track
    - Output speaker segments with timestamps (seconds, float)
    - Generate per-speaker audio files

Output schema: schemas/diarization.schema.json
    - sample_rate: integer
    - speakers: array of {speaker_id, segments: [{start_s, end_s}]}

Invariants:
    - Operates on speech track only (from Stage B)
    - All timestamps in seconds (float)
    - No semantic inference (no emotion, intent, identity)
    - Same input + same version = identical output

No implementation in Commit 1.
"""
