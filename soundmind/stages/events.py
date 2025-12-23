"""
Stage E: Acoustic Event Candidates (STUB)

Responsibilities:
    - Detect non-semantic acoustic events in non_speech segments
    - Output event candidates with timestamps and confidence

Event types (FROZEN — trigger-only, no fine-grained taxonomy):
    - impulsive_sound
    - tonal_alarm_like
    - vehicle_like

Output schema: schemas/events.schema.json
    - events: array of {type, start_s, end_s, confidence}

Invariants:
    - Trigger-only detection (recall-biased)
    - Operates ONLY inside non_speech segments
    - No semantic inference (no gunshot, siren, shouting labels)
    - All timestamps in seconds (float)
    - Confidence in range [0.0, 1.0]
    - Same input + same version = identical output

No implementation in Commit 1.
"""
