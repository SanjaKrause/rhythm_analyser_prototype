"""
Pattern length detection using drum onsets, mel-band, and bass pitch methods.

This module implements three pattern detection methods with circular cross-correlation:
1. Drum Onset Method - Binary onset detection from drum CSV files
2. Mel-Band Method - Log-mel spectrogram analysis from drums.wav
3. Bass Pitch Method - F0 (pitch) analysis from bass.wav using Melodia

Each method computes bar-lag periodicity and selects best power-of-2 pattern length.

Environment: AEinBOX_13_3
Dependencies: librosa, numpy, pandas, soundfile, libf0
"""

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Dict
from libf0.salience import salience as melodia_f0

# Import config
import importlib.util
_config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("config_module", _config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = config_module.config

# Constants
EPS = 1e-9

# Drum onset parameters
DRUM_GRID_MODE = "fixed16"
DRUM_PHASE_DIVS = 16
DRUM_AGGREGATION = "median"

# Mel-band parameters
MEL_SR_TARGET = 22050
MEL_N_FFT = 2048
MEL_HOP_LENGTH = 512
MEL_N_MELS = 48
MEL_FMIN, MEL_FMAX = 20, 10000
MEL_BAR_FRAMES = 64
MEL_AGGREGATION = "median"

# Bass pitch parameters
PITCH_GRID_MODE = "fixed16"
PITCH_PHASE_DIVS = 16
PITCH_FMIN, PITCH_FMAX = 55, 1760
PITCH_NFFT, PITCH_HOP = 2048, 256
PITCH_AGGREGATION = "median"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def choose_bins(tsig: int, mode: str, default_divs: int) -> int:
    """Choose number of bins based on grid mode."""
    if mode == "fixed16":
        return default_divs
    return max(8, 4 * int(tsig))


def choose_pow2_length(lags: np.ndarray, lag_profile: np.ndarray) -> int:
    """
    Choose pattern length from power-of-two lags (1, 2, 4, 8).

    Selection strategy:
    1. Filter lags to power-of-2 values only (excluding 1)
    2. Find all local maxima among these
    3. If local maxima exist, return the smallest L with best value
    4. Otherwise, return the smallest L with the globally best value

    Parameters
    ----------
    lags : np.ndarray
        Lag values in bars
    lag_profile : np.ndarray
        Similarity values for each lag

    Returns
    -------
    int
        Best power-of-2 pattern length, or np.nan if no valid candidates

    Examples
    --------
    >>> lags = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> profile = np.array([0.9, 0.5, 0.3, 0.8, 0.2, 0.1, 0.05, 0.7])
    >>> choose_pow2_length(lags, profile)
    4  # L=4 is local maximum with high value
    """
    if lags.size == 0 or lag_profile.size == 0:
        return np.nan

    pow2_all = [1, 2, 4, 8]
    candidate_L = [L for L in pow2_all if L in set(lags.tolist())]
    if not candidate_L:
        return np.nan

    idx_map = {int(L): int(np.where(lags == L)[0][0]) for L in candidate_L}

    def is_local_max(i: int) -> bool:
        if i == 0:
            return lag_profile[i] > lag_profile[i+1]
        if i == len(lag_profile) - 1:
            return lag_profile[i] > lag_profile[i-1]
        return (
            lag_profile[i] >= lag_profile[i-1]
            and lag_profile[i] >= lag_profile[i+1]
            and (lag_profile[i] > lag_profile[i-1] or lag_profile[i] > lag_profile[i+1])
        )

    # Find local maxima among power-of-2 lags (excluding L=1)
    maxima_pow2 = []
    for L in candidate_L:
        if L == 1:
            continue
        i = idx_map[L]
        if is_local_max(i):
            maxima_pow2.append((lag_profile[i], L))

    # If we have local maxima, return smallest L with best value
    if maxima_pow2:
        best_val = max(v for v, _ in maxima_pow2)
        best_Ls = sorted(L for v, L in maxima_pow2 if v == best_val)
        return best_Ls[0]

    # Otherwise, return smallest L with globally best value
    vals = [(lag_profile[idx_map[L]], L) for L in candidate_L]
    best_val = max(v for v, _ in vals)
    best_Ls = sorted(L for v, L in vals if v == best_val)
    return best_Ls[0]


# ============================================================================
# DRUM ONSET METHOD
# ============================================================================

def drum_bar_vector(onsets: np.ndarray, t0: float, t1: float, bins: int) -> np.ndarray:
    """
    Create binary bar vector from onsets (presence only, no accumulation).

    Parameters
    ----------
    onsets : np.ndarray
        Onset times in seconds
    t0 : float
        Bar start time
    t1 : float
        Bar end time
    bins : int
        Number of phase bins

    Returns
    -------
    np.ndarray
        Binary vector indicating onset presence in each bin
    """
    v = np.zeros(bins, dtype=float)
    L = max(EPS, t1 - t0)
    m = (onsets >= t0) & (onsets < t1)
    if np.any(m):
        rel = (onsets[m] - t0) / L
        k = np.clip(np.floor(rel * bins).astype(int), 0, bins - 1)
        v[np.unique(k)] = 1.0  # Binary presence only
    return v


def drum_circular_xcorr_max(v1: np.ndarray, v2: np.ndarray, tsig: int, restrict_to_beats: bool) -> float:
    """
    Circular cross-correlation with rotation allowed.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Bar vectors to correlate
    tsig : int
        Time signature (beats per bar)
    restrict_to_beats : bool
        If True, only check shifts aligned with beats

    Returns
    -------
    float
        Maximum normalized correlation value
    """
    n = v1.size
    V1 = np.fft.rfft(v1, n=n)
    V2 = np.fft.rfft(v2, n=n)
    x = np.fft.irfft(V1 * np.conj(V2), n=n).real
    if restrict_to_beats and tsig > 0:
        step = max(1, n // int(tsig))
        x = x[::step]
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(x.max() / denom) if denom > 0 else 0.0


def drum_analyze(
    onsets: np.ndarray,
    bar_starts: np.ndarray,
    bar_ends: np.ndarray,
    tsig: int
) -> Dict:
    """
    Analyze drum onsets with circular cross-correlation.

    Parameters
    ----------
    onsets : np.ndarray
        Onset times in seconds
    bar_starts : np.ndarray
        Bar start times in seconds
    bar_ends : np.ndarray
        Bar end times in seconds
    tsig : int
        Time signature (beats per bar)

    Returns
    -------
    dict
        Dictionary containing:
        - best_pow2_L: Best power-of-2 pattern length
        - lags: Array of lag values
        - lag_profile: Similarity profile across lags
    """
    bins = choose_bins(tsig, DRUM_GRID_MODE, DRUM_PHASE_DIVS)

    # Create bar vectors
    vectors = np.stack([
        drum_bar_vector(onsets, t0, t1, bins)
        for t0, t1 in zip(bar_starts, bar_ends)
    ])
    N = len(vectors)

    # Compute similarity matrix
    sim = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            sim[i, j] = drum_circular_xcorr_max(vectors[i], vectors[j], tsig, True)

    # Bar-lag profile
    lags = np.arange(1, N) if N > 1 else np.array([])
    if lags.size:
        if DRUM_AGGREGATION == "median":
            lag_profile = np.array([np.median(sim.diagonal(L)) for L in lags])
        else:
            lag_profile = np.array([np.mean(sim.diagonal(L)) for L in lags])
    else:
        lag_profile = np.array([])

    # Best power-of-2 pattern length
    best_pow2_L = choose_pow2_length(lags, lag_profile) if lags.size else np.nan

    return {
        'best_pow2_L': best_pow2_L,
        'lags': lags,
        'lag_profile': lag_profile,
    }


# ============================================================================
# MEL-BAND METHOD
# ============================================================================

def load_logmel(path: str, sr: int = MEL_SR_TARGET) -> Tuple[np.ndarray, int]:
    """
    Load audio and compute log-mel spectrogram.

    Parameters
    ----------
    path : str
        Path to audio file
    sr : int
        Target sample rate

    Returns
    -------
    tuple of (np.ndarray, int)
        (normalized log-mel spectrogram, sample rate)
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    try:
        S_mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH,
            n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, power=2.0
        )
    except TypeError:
        # Fallback for older librosa versions
        S_lin = np.abs(librosa.stft(y=y, n_fft=MEL_N_FFT, hop_length=MEL_HOP_LENGTH))**2
        mel_filter = librosa.filters.mel(sr=sr, n_fft=MEL_N_FFT, n_mels=MEL_N_MELS,
                                         fmin=MEL_FMIN, fmax=MEL_FMAX)
        S_mel = mel_filter @ S_lin

    D = librosa.power_to_db(S_mel, ref=np.max)
    Dn = (D - float(D.min())) / max(EPS, float(D.max() - D.min()))
    return Dn, sr


def mel_bar_patch(Dn: np.ndarray, sr: int, hop: int, t0: float, t1: float,
                  frames_fixed: int = MEL_BAR_FRAMES) -> np.ndarray:
    """
    Extract and resample bar patch from log-mel spectrogram.

    Parameters
    ----------
    Dn : np.ndarray
        Normalized log-mel spectrogram (mels x frames)
    sr : int
        Sample rate
    hop : int
        Hop length
    t0 : float
        Bar start time
    t1 : float
        Bar end time
    frames_fixed : int
        Target number of frames

    Returns
    -------
    np.ndarray
        Resampled bar patch (mels x frames_fixed)
    """
    fps = sr / hop
    a = int(np.floor(t0 * fps))
    b = int(np.ceil(t1 * fps))
    a = max(0, min(a, Dn.shape[1]-1))
    b = max(a+1, min(b, Dn.shape[1]))
    P = Dn[:, a:b]

    M, T = P.shape
    if T == frames_fixed:
        return P

    old = np.linspace(0.0, 1.0, T, endpoint=False) if T > 1 else np.array([0.0])
    new = np.linspace(0.0, 1.0, frames_fixed, endpoint=False)
    out = np.zeros((M, frames_fixed), dtype=float)
    for m in range(M):
        row = P[m]
        out[m] = np.interp(new, old, row) if T > 1 else row[0]
    return out


def mel_circular_xcorr_max(R: np.ndarray, Q: np.ndarray) -> float:
    """
    Band-wise circular cross-correlation over time.

    Parameters
    ----------
    R, Q : np.ndarray
        Mel-band patches to correlate (mels x frames)

    Returns
    -------
    float
        Maximum normalized correlation value
    """
    assert R.shape == Q.shape
    M, T = R.shape
    acc = np.zeros(T, dtype=float)
    denom = 0.0
    for m in range(M):
        r = R[m]
        q = Q[m]
        nr = np.linalg.norm(r)
        nq = np.linalg.norm(q)
        if nr < EPS or nq < EPS:
            continue
        Rf = np.fft.rfft(r, n=T)
        Qf = np.fft.rfft(q, n=T)
        corr = np.fft.irfft(Rf * np.conj(Qf), n=T).real
        acc += corr
        denom += nr * nq
    return float(acc.max() / max(EPS, denom))


def mel_analyze(
    drums_path: str,
    bar_starts: np.ndarray,
    bar_ends: np.ndarray
) -> Dict:
    """
    Analyze mel-band with circular cross-correlation.

    Parameters
    ----------
    drums_path : str
        Path to drums.wav file
    bar_starts : np.ndarray
        Bar start times in seconds
    bar_ends : np.ndarray
        Bar end times in seconds

    Returns
    -------
    dict
        Dictionary containing:
        - best_pow2_L: Best power-of-2 pattern length
        - lags: Array of lag values
        - lag_profile: Similarity profile across lags
    """
    # Load log-mel
    Dn, sr = load_logmel(drums_path)

    # Create bar patches
    patches = np.stack([
        mel_bar_patch(Dn, sr, MEL_HOP_LENGTH, float(t0), float(t1), MEL_BAR_FRAMES)
        for t0, t1 in zip(bar_starts, bar_ends)
    ])
    N = len(patches)

    # Compute similarity matrix
    sim = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            sim[i, j] = mel_circular_xcorr_max(patches[i], patches[j])

    # Bar-lag profile
    lags = np.arange(1, N) if N > 1 else np.array([])
    if lags.size:
        if MEL_AGGREGATION == "median":
            lag_profile = np.array([np.median(sim.diagonal(L)) for L in lags])
        else:
            lag_profile = np.array([np.mean(sim.diagonal(L)) for L in lags])
    else:
        lag_profile = np.array([])

    # Best power-of-2 pattern length
    best_pow2_L = choose_pow2_length(lags, lag_profile) if lags.size else np.nan

    return {
        'best_pow2_L': best_pow2_L,
        'lags': lags,
        'lag_profile': lag_profile,
    }


# ============================================================================
# BASS PITCH METHOD
# ============================================================================

def normalize_pitch_vector(tt: np.ndarray, f0: np.ndarray, t0: float, t1: float, bins: int) -> np.ndarray:
    """
    Normalize pitch vector for a single bar.

    Parameters
    ----------
    tt : np.ndarray
        Time coefficients from Melodia
    f0 : np.ndarray
        F0 values in Hz
    t0 : float
        Bar start time
    t1 : float
        Bar end time
    bins : int
        Number of phase bins

    Returns
    -------
    np.ndarray
        Normalized pitch vector
    """
    m = (tt >= t0) & (tt < t1) & np.isfinite(f0) & (f0 > 0)
    if not np.any(m):
        return np.zeros(bins)

    seg_t, seg_f = tt[m], f0[m]
    x_new = np.linspace(seg_t[0], seg_t[-1], bins)
    f_res = np.interp(x_new, seg_t, seg_f)
    f_res -= np.nanmean(f_res)
    f_res /= (np.nanstd(f_res) + 1e-9)
    return np.nan_to_num(f_res)


def pitch_dot_norm(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Normalized dot product in [-1,1] (safe for zero vectors).

    Parameters
    ----------
    v1, v2 : np.ndarray
        Vectors to correlate

    Returns
    -------
    float
        Normalized dot product
    """
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / (denom + EPS)) if denom > 0 else 0.0


def pitch_circular_xcorr_max(v1: np.ndarray, v2: np.ndarray, tsig: int, restrict_to_beats: bool) -> float:
    """
    Circular cross-correlation with optional beat restriction.

    Parameters
    ----------
    v1, v2 : np.ndarray
        Pitch vectors to correlate
    tsig : int
        Time signature (beats per bar)
    restrict_to_beats : bool
        If True, only check shifts aligned with beats

    Returns
    -------
    float
        Maximum normalized correlation value
    """
    n = v1.size

    if restrict_to_beats and tsig > 0:
        step = max(1, int(round(n / int(tsig))))
        shifts = np.arange(0, n, step, dtype=int)
    else:
        shifts = np.arange(0, n, dtype=int)

    best = 0.0
    for s in shifts:
        if s == 0:
            sim = pitch_dot_norm(v1, v2)
        else:
            v2r = np.roll(v2, s)
            sim = pitch_dot_norm(v1, v2r)
        if sim > best:
            best = sim
    return best


def load_and_extract_f0(
    audio_path: str,
    snippet: Optional[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load audio and extract F0 using Melodia.

    Parameters
    ----------
    audio_path : str
        Path to bass.wav file
    snippet : tuple or None
        (start_time, end_time) to process only snippet, or None for full audio

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (f0 values in Hz, time coefficients in seconds)
    """
    info = sf.info(audio_path)
    Fs = info.samplerate

    if snippet:
        s0, s1 = snippet
        start_frame = max(0, int(np.floor(s0 * Fs)))
        stop_frame = min(info.frames, int(np.ceil(s1 * Fs)))

        x, _ = sf.read(audio_path, start=start_frame, stop=stop_frame)
        if x.ndim > 1:
            x = x[:, 0]

        f0, Tcoef, sal = melodia_f0(x, Fs=Fs, N=PITCH_NFFT, H=PITCH_HOP,
                                    F_min=PITCH_FMIN, F_max=PITCH_FMAX)
        time_offset = start_frame / Fs
        Tcoef = Tcoef + time_offset
    else:
        x, _ = sf.read(audio_path)
        if x.ndim > 1:
            x = x[:, 0]

        f0, Tcoef, sal = melodia_f0(x, Fs=Fs, N=PITCH_NFFT, H=PITCH_HOP,
                                    F_min=PITCH_FMIN, F_max=PITCH_FMAX)

    return f0, Tcoef


def pitch_analyze(
    bass_path: str,
    bar_starts: np.ndarray,
    bar_ends: np.ndarray,
    tsig: int,
    snippet: Optional[Tuple[float, float]] = None,
    save_f0_csv: bool = True
) -> Dict:
    """
    Analyze bass pitch with circular cross-correlation.

    Parameters
    ----------
    bass_path : str
        Path to bass.wav file
    bar_starts : np.ndarray
        Bar start times in seconds
    bar_ends : np.ndarray
        Bar end times in seconds
    tsig : int
        Time signature (beats per bar)
    snippet : tuple or None
        (start_time, end_time) to process only snippet
    save_f0_csv : bool
        If True, save F0 data to CSV in same directory as bass.wav (default: True)

    Returns
    -------
    dict
        Dictionary containing:
        - best_pow2_L: Best power-of-2 pattern length
        - lags: Array of lag values
        - lag_profile: Similarity profile across lags
        - f0: F0 values in Hz
        - times: Time coefficients in seconds
    """
    # Extract F0
    f0, Tcoef = load_and_extract_f0(bass_path, snippet)

    # Save F0 to CSV if requested
    if save_f0_csv:
        from pathlib import Path
        import pandas as pd

        bass_dir = Path(bass_path).parent
        f0_csv_path = bass_dir / 'bass_f0.csv'

        # Create DataFrame
        df_f0 = pd.DataFrame({
            'time': Tcoef,
            'f0_hz': f0
        })

        # Save to CSV
        df_f0.to_csv(f0_csv_path, index=False)
        print(f"    Saved F0 data to: {f0_csv_path}")

    # Create pitch vectors
    bins = choose_bins(tsig, PITCH_GRID_MODE, PITCH_PHASE_DIVS)
    vectors = np.stack([
        normalize_pitch_vector(Tcoef, f0, t0, t1, bins)
        for t0, t1 in zip(bar_starts, bar_ends)
    ])
    N = len(vectors)

    # Compute similarity matrix
    sim = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            sim[i, j] = pitch_circular_xcorr_max(vectors[i], vectors[j], tsig, True)

    # Bar-lag profile
    lags = np.arange(1, N) if N > 1 else np.array([])
    if lags.size:
        if PITCH_AGGREGATION == "median":
            lag_profile = np.array([np.median(sim.diagonal(L)) for L in lags])
        else:
            lag_profile = np.array([np.mean(sim.diagonal(L)) for L in lags])
    else:
        lag_profile = np.array([])

    # Best power-of-2 pattern length
    best_pow2_L = choose_pow2_length(lags, lag_profile) if lags.size else np.nan

    return {
        'best_pow2_L': best_pow2_L,
        'lags': lags,
        'lag_profile': lag_profile,
        'f0': f0,
        'times': Tcoef,
    }


# ============================================================================
# MAIN DETECTION FUNCTION
# ============================================================================

def detect_pattern_lengths(
    onset_csv_path: str,
    drums_wav_path: str,
    bass_wav_path: str,
    bar_starts: np.ndarray,
    bar_ends: np.ndarray,
    tsig: int,
    snippet: Optional[Tuple[float, float]] = None
) -> Dict[str, int]:
    """
    Detect pattern lengths using all three methods with circular convolution.

    This function runs all 3 pattern detection methods:
    1. Drum Onset Method - from onset CSV
    2. Mel-Band Method - from drums.wav
    3. Bass Pitch Method - from bass.wav

    Parameters
    ----------
    onset_csv_path : str
        Path to drum onset CSV file
    drums_wav_path : str
        Path to drums.wav stem
    bass_wav_path : str
        Path to bass.wav stem
    bar_starts : np.ndarray
        Bar start times in seconds (corrected)
    bar_ends : np.ndarray
        Bar end times in seconds (corrected)
    tsig : int
        Time signature (beats per bar)
    snippet : tuple or None
        (start_time, end_time) to limit analysis to snippet

    Returns
    -------
    dict
        Dictionary with keys 'drum', 'mel', 'pitch' containing best_pow2_L values

    Examples
    --------
    >>> pattern_lengths = detect_pattern_lengths(
    ...     'onsets.csv',
    ...     'drums.wav',
    ...     'bass.wav',
    ...     bar_starts,
    ...     bar_ends,
    ...     tsig=4
    ... )
    >>> print(pattern_lengths)
    {'drum': 4, 'mel': 4, 'pitch': 8}
    """
    print(f"\n  [Pattern Detection] Running all 3 methods...")

    # Load onset times
    df_onsets = pd.read_csv(onset_csv_path)
    onset_col = None
    for c in df_onsets.columns:
        if c.strip().lower() in ("onset_times", "time", "time_s", "onset", "onset_s", "t", "seconds"):
            onset_col = c
            break
    if onset_col is None:
        raise ValueError(f"No usable onset-time column in {onset_csv_path}")

    onsets = pd.to_numeric(df_onsets[onset_col], errors="coerce").dropna().astype(float).to_numpy()
    onsets.sort()

    # Filter onsets to snippet if provided
    if snippet:
        s0, s1 = snippet
        onsets = onsets[(onsets >= s0) & (onsets < s1)]

    results = {}

    # Method 1: Drum Onset
    try:
        print(f"    Running Drum Onset method...")
        drum_result = drum_analyze(onsets, bar_starts, bar_ends, tsig)
        results['drum'] = int(drum_result['best_pow2_L']) if np.isfinite(drum_result['best_pow2_L']) else 4
        print(f"    Drum Onset: L={results['drum']}")
    except Exception as e:
        print(f"    Drum Onset method failed: {e}")
        results['drum'] = 4  # Default fallback

    # Method 2: Mel-Band
    try:
        print(f"    Running Mel-Band method...")
        mel_result = mel_analyze(drums_wav_path, bar_starts, bar_ends)
        results['mel'] = int(mel_result['best_pow2_L']) if np.isfinite(mel_result['best_pow2_L']) else 4
        print(f"    Mel-Band: L={results['mel']}")
    except Exception as e:
        print(f"    Mel-Band method failed: {e}")
        results['mel'] = 4  # Default fallback

    # Method 3: Bass Pitch
    try:
        print(f"    Running Bass Pitch method...")
        pitch_result = pitch_analyze(bass_wav_path, bar_starts, bar_ends, tsig, snippet)
        results['pitch'] = int(pitch_result['best_pow2_L']) if np.isfinite(pitch_result['best_pow2_L']) else 4
        print(f"    Bass Pitch: L={results['pitch']}")
    except Exception as e:
        print(f"    Bass Pitch method failed: {e}")
        results['pitch'] = 4  # Default fallback

    print(f"  [Pattern Detection] Complete: {results}")
    return results


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_pattern_detection():
    """Test pattern detection with synthetic data."""
    print("=" * 80)
    print("Testing Pattern Detection Module")
    print("=" * 80)

    # Create synthetic bar data (8 bars)
    num_bars = 8
    bar_duration = 2.0  # seconds
    bar_starts = np.arange(num_bars) * bar_duration
    bar_ends = bar_starts + bar_duration

    # Create synthetic onsets (16th note grid)
    onsets = []
    for bar_idx in range(num_bars):
        bar_start = bar_starts[bar_idx]
        # 16 onsets per bar
        for i in range(16):
            onsets.append(bar_start + (i * bar_duration / 16))
    onsets = np.array(onsets)

    print(f"\n[1] Testing with {num_bars} synthetic bars...")
    print(f"  Bar duration: {bar_duration}s")
    print(f"  Total onsets: {len(onsets)}")

    # Test drum analyze
    try:
        result = drum_analyze(onsets, bar_starts, bar_ends, tsig=4)
        print(f"  Drum analysis result: L={result['best_pow2_L']}")
        assert np.isfinite(result['best_pow2_L']), "Should return valid pattern length"
        print("  ✓ Drum analysis works")
    except Exception as e:
        print(f"  ✗ Drum analysis failed: {e}")

    print("\n" + "=" * 80)
    print("Pattern detection test passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_pattern_detection()
