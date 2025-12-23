"""
Stage C: Signal Quality Indicators (STUB)

Responsibilities:
    - Compute signal quality metrics for audio tracks
    - Metrics (all numeric):
        - snr_proxy_db: Signal-to-noise ratio proxy (dB)
        - clipping_pct: Percentage of clipped samples
        - dropout_pct: Percentage of dropout/silence artifacts
        - reverb_proxy_rt60_ms: Reverberation time proxy (ms)
        - loudness_lufs: Integrated loudness (LUFS)
        - rms_dbfs: RMS level (dBFS)

Invariants:
    - Metrics are objective, non-semantic
    - Same input + same version = identical metrics

No implementation in Commit 1.
"""
