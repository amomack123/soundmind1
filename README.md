# SoundMind v1

Forensic audio processing pipeline.

> **⚠️ CONTRACTS ARE FROZEN**  
> Schemas, stages, and invariants in this repository are locked for v1.  
> Do not modify without versioning to v2.

## Pipeline Stages (Fixed Order)

| Stage | Module | Purpose |
|-------|--------|---------|
| A | `soundmind.stages.ingest` | Ingest & Normalize |
| B | `soundmind.stages.separation` | Speech vs Residual Separation |
| C | `soundmind.stages.sqi` | Signal Quality Indicators |
| D | `soundmind.stages.diarization` | Speaker Diarization |
| E | `soundmind.stages.events` | Acoustic Event Candidates |
| F | `soundmind.stages.rollup` | Final Roll-Up |

## Frozen Schemas

- `schemas/status.schema.json` — Job status and outputs
- `schemas/diarization.schema.json` — Speaker segments
- `schemas/events.schema.json` — Acoustic event candidates

## Invariants

- All timestamps are seconds (float)
- No semantic inference (intent, emotion, deception)
- No fine-grained acoustic taxonomy (no gunshot, siren, shouting)
- Stage E is trigger-only and recall-biased
- Event candidates only run inside non_speech segments
- Same input + same version = identical output

### Events (Commit 9)

- `events/events.json` is byte-identical across repeated runs for identical input
- Events are non-speech-only: derived from `speech.wav` exact-zero semantics (non-zero sample = speech, exact zero = non-speech)
- Timestamps (`start_s`, `end_s`) formatted to exactly 6 decimal places
- Events sorted deterministically by `(start_s, end_s, type)`
- Impulse candidates overlapping any speech region are discarded entirely

## Schema Validation

```bash
python tools/validate_schema.py status output/job_status.json
python tools/validate_schema.py diarization output/diarization.json
python tools/validate_schema.py events output/events.json
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Status

**Commit 9**: Deterministic non-speech event materialization with locked tests.

