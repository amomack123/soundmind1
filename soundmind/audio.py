"""
SoundMind v1 Audio Utilities — Commit 6

Deterministic, CPU-only audio processing primitives.

Library Stack:
    - soundfile: WAV I/O (libsndfile-backed)
    - numpy: Array operations
    - scipy.signal.resample_poly: Deterministic resampling

INVARIANTS:
    - All operations are deterministic
    - No randomness, seeds, or time-based logic
    - No multithreading affecting order
    - Same input → identical output
    - CPU-only execution

ALLOWED PRIMITIVES (per Commit 6 rules):
    - WAV read/write (PCM 16-bit, mono, 16kHz)
    - Mono downmix (mean)
    - Resampling to 16kHz (scipy.signal.resample_poly)
    - Framing (fixed window/hop)
    - RMS, peak absolute value, zero-crossing rate
    - Boolean masking, elementwise multiply/subtract
    - Hard clipping to [-1, 1]
    - Contiguous region detection
    - Fixed merge/drop thresholds

FORBIDDEN:
    - FFT / STFT / MFCC
    - Spectral features
    - ML inference
    - Dithering or noise injection
"""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# =============================================================================
# Constants (FROZEN)
# =============================================================================

CANONICAL_SAMPLE_RATE = 16000
FRAME_MS = 30
HOP_MS = 10
EPS = 1e-10  # Fixed epsilon for thresholding


# =============================================================================
# WAV I/O
# =============================================================================


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """
    Read WAV file and return samples with sample rate.
    
    Args:
        path: Path to WAV file
    
    Returns:
        Tuple of (samples as float32 in [-1, 1], sample_rate)
    
    Raises:
        RuntimeError: If file cannot be read
    """
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    return samples, sr


def read_wav_int16(path: Path) -> tuple[np.ndarray, int]:
    """
    Read WAV file and return raw PCM-16 integer samples.
    
    Args:
        path: Path to WAV file (must be PCM-16 subtype)
    
    Returns:
        Tuple of (samples as int16, sample_rate)
    
    Note:
        Commit 9: Used for exact-zero speech/non-speech detection.
        Integer comparison (== 0) is immune to floating-point
        representation quirks that could arise from normalization.
    """
    samples, sr = sf.read(path, dtype="int16", always_2d=False)
    return samples, sr



def write_wav(path: Path, samples: np.ndarray, sample_rate: int = CANONICAL_SAMPLE_RATE) -> None:
    """
    Write samples to WAV file as PCM 16-bit.
    
    Args:
        path: Output path
        samples: Audio samples (float32, [-1, 1])
        sample_rate: Sample rate (default: 16000)
    
    Note:
        - Hard clips to [-1, 1] before writing
        - Writes PCM 16-bit (subtype='PCM_16')
        - Deterministic output (no dithering)
    """
    # Hard clip to prevent overflow
    clipped = np.clip(samples, -1.0, 1.0)
    sf.write(path, clipped, sample_rate, subtype="PCM_16")


# =============================================================================
# Canonicalization
# =============================================================================


def normalize_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Normalize audio to canonical format: mono, 16kHz, float32 [-1, 1].
    
    Args:
        samples: Input samples (may be stereo, any sample rate)
        sample_rate: Input sample rate
    
    Returns:
        Normalized samples (mono, 16kHz, float32)
    
    Note:
        - Stereo → mono by arithmetic mean
        - Resampling uses scipy.signal.resample_poly (deterministic)
        - No amplitude normalization / AGC
    """
    # Mono downmix if multi-channel
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    
    # Ensure float32
    samples = samples.astype(np.float32)
    
    # Resample to canonical rate if needed
    if sample_rate != CANONICAL_SAMPLE_RATE:
        samples = _resample_deterministic(samples, sample_rate, CANONICAL_SAMPLE_RATE)
    
    return samples


def _resample_deterministic(samples: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """
    Resample using scipy.signal.resample_poly with fixed integer factors.
    
    Args:
        samples: Input samples
        sr_from: Source sample rate
        sr_to: Target sample rate
    
    Returns:
        Resampled samples
    
    Note:
        Uses integer up/down factors for determinism.
    """
    # Compute integer resampling factors
    g = gcd(sr_from, sr_to)
    up = sr_to // g
    down = sr_from // g
    
    # resample_poly is deterministic with fixed up/down
    return resample_poly(samples, up, down).astype(np.float32)


# =============================================================================
# Frame-based DSP
# =============================================================================


def compute_frame_rms(
    samples: np.ndarray,
    frame_ms: int = FRAME_MS,
    hop_ms: int = HOP_MS,
    sr: int = CANONICAL_SAMPLE_RATE,
) -> np.ndarray:
    """
    Compute RMS for each frame.
    
    Args:
        samples: Input samples (1D)
        frame_ms: Frame length in milliseconds
        hop_ms: Hop length in milliseconds
        sr: Sample rate
    
    Returns:
        Array of RMS values, one per frame
    """
    frame_samples = int(sr * frame_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    
    n_frames = max(1, (len(samples) - frame_samples) // hop_samples + 1)
    rms = np.zeros(n_frames, dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop_samples
        end = start + frame_samples
        frame = samples[start:end]
        rms[i] = np.sqrt(np.mean(frame ** 2) + EPS)
    
    return rms


def build_speech_mask(
    samples: np.ndarray,
    sr: int = CANONICAL_SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
    hop_ms: int = HOP_MS,
) -> np.ndarray:
    """
    Build speech mask using energy-based RMS thresholding.
    
    Args:
        samples: Input samples (1D, normalized)
        sr: Sample rate
        frame_ms: Frame length in milliseconds
        hop_ms: Hop length in milliseconds
    
    Returns:
        Sample-level mask (1.0 = speech, 0.0 = non-speech)
    
    Note:
        Threshold = max(global_rms * 0.5, EPS)
        Frame mask is converted to sample mask using nearest-frame rule.
    """
    # Compute frame RMS
    frame_rms = compute_frame_rms(samples, frame_ms, hop_ms, sr)
    
    # Compute global RMS
    global_rms = np.sqrt(np.mean(samples ** 2) + EPS)
    
    # Deterministic threshold
    threshold = max(global_rms * 0.5, EPS)
    
    # Frame-level mask
    frame_mask = (frame_rms > threshold).astype(np.float32)
    
    # Convert to sample mask
    return frame_mask_to_samples(frame_mask, len(samples), hop_ms, sr)


def frame_mask_to_samples(
    frame_mask: np.ndarray,
    num_samples: int,
    hop_ms: int = HOP_MS,
    sr: int = CANONICAL_SAMPLE_RATE,
) -> np.ndarray:
    """
    Convert frame-level mask to sample-level mask using nearest-frame rule.
    
    Args:
        frame_mask: Frame-level mask (1D, values 0 or 1)
        num_samples: Number of output samples
        hop_ms: Hop length in milliseconds
        sr: Sample rate
    
    Returns:
        Sample-level mask (1.0 or 0.0 per sample)
    
    Mapping rule (FROZEN):
        For sample i, use frame index = i // hop_samples
        Clamp to valid frame range.
    """
    hop_samples = int(sr * hop_ms / 1000)
    sample_mask = np.zeros(num_samples, dtype=np.float32)
    
    for i in range(num_samples):
        frame_idx = min(i // hop_samples, len(frame_mask) - 1)
        sample_mask[i] = frame_mask[frame_idx]
    
    return sample_mask


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_rms(samples: np.ndarray) -> float:
    """Compute RMS of entire signal."""
    return float(np.sqrt(np.mean(samples ** 2)))


def compute_peak_abs(samples: np.ndarray) -> float:
    """Compute peak absolute value of signal."""
    return float(np.max(np.abs(samples)))


def compute_zero_crossing_rate(samples: np.ndarray) -> float:
    """
    Compute zero-crossing rate (per-sample, averaged across signal).
    
    Returns:
        ZCR as fraction (0.0 to 1.0)
    """
    if len(samples) < 2:
        return 0.0
    
    # Count sign changes
    signs = np.sign(samples)
    crossings = np.sum(signs[1:] != signs[:-1])
    
    return float(crossings / (len(samples) - 1))


# =============================================================================
# Segmentation
# =============================================================================


def find_contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Find contiguous regions of 1s in a binary mask.
    
    Args:
        mask: Binary mask (values 0 or 1)
    
    Returns:
        List of (start_sample, end_sample) tuples (end is exclusive)
    """
    regions = []
    in_region = False
    start = 0
    
    for i, val in enumerate(mask):
        if val > 0.5 and not in_region:
            # Start of region
            in_region = True
            start = i
        elif val <= 0.5 and in_region:
            # End of region
            in_region = False
            regions.append((start, i))
    
    # Handle region extending to end
    if in_region:
        regions.append((start, len(mask)))
    
    return regions


def merge_segments(
    segments: list[tuple[float, float]],
    gap_threshold: float,
) -> list[tuple[float, float]]:
    """
    Merge segments with gaps smaller than threshold.
    
    Args:
        segments: List of (start_sec, end_sec) tuples
        gap_threshold: Maximum gap to merge (seconds)
    
    Returns:
        Merged segment list
    """
    if not segments:
        return []
    
    # Sort by start time
    sorted_segs = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segs[0]]
    
    for start, end in sorted_segs[1:]:
        prev_start, prev_end = merged[-1]
        
        if start - prev_end <= gap_threshold:
            # Merge with previous
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    
    return merged


def drop_short_segments(
    segments: list[tuple[float, float]],
    min_duration: float,
) -> list[tuple[float, float]]:
    """
    Drop segments shorter than minimum duration.
    
    Args:
        segments: List of (start_sec, end_sec) tuples
        min_duration: Minimum duration to keep (seconds)
    
    Returns:
        Filtered segment list
    """
    return [(s, e) for s, e in segments if (e - s) >= min_duration]


def samples_to_seconds(sample_idx: int, sr: int = CANONICAL_SAMPLE_RATE) -> float:
    """Convert sample index to seconds."""
    return float(sample_idx / sr)


def regions_to_time_segments(
    regions: list[tuple[int, int]],
    sr: int = CANONICAL_SAMPLE_RATE,
) -> list[tuple[float, float]]:
    """
    Convert sample-based regions to time-based segments.
    
    Args:
        regions: List of (start_sample, end_sample) tuples
        sr: Sample rate
    
    Returns:
        List of (start_sec, end_sec) tuples
    """
    return [
        (samples_to_seconds(s, sr), samples_to_seconds(e, sr))
        for s, e in regions
    ]


# =============================================================================
# Speaker Audio Extraction (Commit 8)
# =============================================================================


def extract_and_concatenate(
    samples: np.ndarray,
    segments: list[tuple[float, float]],
    sr: int,
) -> np.ndarray:
    """
    Extract and concatenate sample regions based on time segments.
    
    Args:
        samples: Source audio samples
        segments: List of (start_sec, end_sec) tuples, ordered by start time
        sr: Sample rate (must match source; no default to enforce explicit passing)
    
    Returns:
        Concatenated samples from all segments
    
    Note:
        - Time-to-sample: start_idx = int(start_sec * sr), end_idx = int(end_sec * sr) (floor)
        - Segments must be in temporal order (Commit 8 does NOT re-order)
        - Returns empty array if no segments
        - No amplitude normalization or dtype conversion
    """
    if not segments:
        return np.array([], dtype=np.float32)
    
    chunks = []
    for start_sec, end_sec in segments:
        # Floor-based integer indexing (locked per Commit 8)
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        
        # Clamp to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(samples), end_idx)
        
        if end_idx > start_idx:
            chunks.append(samples[start_idx:end_idx])
    
    if not chunks:
        return np.array([], dtype=np.float32)
    
    return np.concatenate(chunks)


# =============================================================================
# Impulse Detection (Events Stage)
# =============================================================================


def detect_impulses(
    samples: np.ndarray,
    non_speech_mask: np.ndarray,
    sr: int = CANONICAL_SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
    hop_ms: int = HOP_MS,
    peak_ratio: float = 0.6,
    max_duration: float = 0.1,
    min_peak: float = 0.01,
) -> list[dict]:
    """
    Detect impulse events in non-speech regions.
    
    Args:
        samples: Input samples (1D)
        non_speech_mask: Sample-level mask (1.0 = non-speech)
        sr: Sample rate
        frame_ms: Frame length for analysis
        hop_ms: Hop length
        peak_ratio: Frame peak must exceed global_peak * ratio
        max_duration: Maximum impulse duration (seconds)
        min_peak: Minimum absolute peak value
    
    Returns:
        List of event dicts with start, end (seconds)
    
    Note:
        - Only detects in non-speech regions
        - Uses short-time peak detector
    """
    events = []
    
    frame_samples = int(sr * frame_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    
    # Global peak for threshold
    global_peak = compute_peak_abs(samples)
    threshold = max(global_peak * peak_ratio, min_peak)
    
    max_duration_samples = int(max_duration * sr)
    
    n_frames = max(1, (len(samples) - frame_samples) // hop_samples + 1)
    
    i = 0
    while i < n_frames:
        start_sample = i * hop_samples
        end_sample = min(start_sample + frame_samples, len(samples))
        
        # Check if in non-speech region
        frame_mask_mean = np.mean(non_speech_mask[start_sample:end_sample])
        
        if frame_mask_mean < 0.5:
            # Not in non-speech, skip
            i += 1
            continue
        
        # Check frame peak
        frame = samples[start_sample:end_sample]
        frame_peak = np.max(np.abs(frame))
        
        if frame_peak > threshold:
            # Potential impulse - find extent
            impulse_start = start_sample
            impulse_end = end_sample
            
            # Extend forward while peak remains high (but within max duration)
            j = i + 1
            while j < n_frames:
                next_start = j * hop_samples
                next_end = min(next_start + frame_samples, len(samples))
                
                if next_end - impulse_start > max_duration_samples:
                    break
                
                if np.mean(non_speech_mask[next_start:next_end]) < 0.5:
                    break
                
                next_frame = samples[next_start:next_end]
                if np.max(np.abs(next_frame)) < threshold:
                    break
                
                impulse_end = next_end
                j += 1
            
            # Verify duration constraint
            if (impulse_end - impulse_start) / sr <= max_duration:
                events.append({
                    "start": samples_to_seconds(impulse_start, sr),
                    "end": samples_to_seconds(impulse_end, sr),
                })
            
            # Skip past this impulse
            i = j
        else:
            i += 1
    
    return events
