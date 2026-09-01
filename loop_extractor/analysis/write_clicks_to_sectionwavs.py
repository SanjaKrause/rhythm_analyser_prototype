#!/usr/bin/env python3
"""
Write rhythm pattern click tracks onto section WAV files.

For each section WAV in 9.1_sections/{stem}/, creates a 2nd WAV file with
click sounds overlaid at rhythm pattern positions, using grid times from
6.2_filtered_patterns and pattern values from 6.6_anchored_rhythm_histograms.

Click volume reflects pattern_value:
  - 1.0 → full volume (0 dB)
  - 0.5 → reduced volume (-6 dB)

Input:
  - 9.1_sections/{stem}/*.wav  (plain section audio)
  - 6.2_filtered_patterns/{stem}/*_anchored.csv  (grid times + metadata)
  - 6.6_anchored_rhythm_histograms/{stem}/*_filtered_anchored_rhythm_patterns.csv

Output:
  - 9.1_sections/{stem}/*_section_clicks.wav  (section audio + click track)

Usage:
    python write_clicks_to_sectionwavs.py <track_dir> [--stems drums,bass,vocals]

Example:
    python write_clicks_to_sectionwavs.py "/Volumes/PortableSSD/06_Testing/output 100 madmom/93_positions - Ariana Grande"
"""

import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import audio utilities from sibling package
_parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_parent_dir))

from utils.audio_export import (
    generate_click,
    mix_audio_with_clicks,
    CLICK_FREQUENCY,
    CLICK_DURATION,
)

# Stems that make up the full mix (matches config.STEMS). Summed to reconstruct
# the all-stems audio for the additional "_section_clicks_fullmix.wav" variant.
FULLMIX_STEM_NAMES = ["drums", "vocals", "bass", "piano", "other"]


def build_fullmix_section(
    stems_dir: Path,
    start_time: float,
    end_time: float,
    target_len: int,
    fade_duration: float = 0.05,
    sr: int = 44100,
) -> Optional[np.ndarray]:
    """
    Reconstruct the full-mix (all stems) audio for one section by summing the
    per-stem WAVs in 1_stems/ over [start_time, end_time].

    Returns a mono float32 array of length `target_len` (to align with the
    stem section audio / click track), or None if no stem WAVs are found.
    """
    duration = end_time - start_time
    mix = np.zeros(target_len, dtype=np.float32)
    n_found = 0
    for name in FULLMIX_STEM_NAMES:
        stem_path = stems_dir / f"{name}.wav"
        if not stem_path.exists():
            continue
        y, _ = librosa.load(str(stem_path), sr=sr, mono=True,
                            offset=start_time, duration=duration)
        n_found += 1
        L = min(len(y), target_len)
        mix[:L] += y[:L]

    if n_found == 0:
        return None

    # Match the fade of the stem section (extract_sections uses 50ms in/out)
    fade_samples = int(fade_duration * sr)
    if 0 < fade_samples < target_len:
        mix[:fade_samples] *= np.linspace(0, 1, fade_samples)
        mix[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return mix


def read_metadata(csv_path: Path) -> Dict[str, str]:
    """Read metadata from comment header lines (# key=value) in CSV."""
    metadata = {}
    with open(csv_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                line = line[1:].strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()
            else:
                break
    return metadata


def parse_section_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse a section WAV filename into its components.

    Example: SecNo1_L1_verse_0.0538_section.wav
    Returns: {'sec_no': '1', 'pattern_length': '1', 'section_label': 'verse',
              'ratio': '0.0538', 'base': 'SecNo1_L1_verse_0.0538'}
    """
    stem = Path(filename).stem  # SecNo1_L1_verse_0.0538_section
    if not stem.endswith('_section'):
        return None

    base = stem.replace('_section', '')  # SecNo1_L1_verse_0.0538
    parts = base.split('_')
    if len(parts) < 4:
        return None

    # SecNo1, L1, verse, 0.0538
    sec_no_str = parts[0]  # SecNo1
    pattern_length_str = parts[1]  # L1

    if not sec_no_str.startswith('SecNo') or not pattern_length_str.startswith('L'):
        return None

    sec_no = sec_no_str.replace('SecNo', '')
    pattern_length = pattern_length_str.replace('L', '')
    section_label = parts[2]
    ratio = parts[3]

    return {
        'sec_no': sec_no,
        'pattern_length': pattern_length,
        'section_label': section_label,
        'ratio': ratio,
        'base': base,
    }


def build_section_id(sec_no: str, section_label: str, pattern_length: str) -> str:
    """
    Build section_id as used in rhythm patterns CSV.

    Example: SecNo1_verse_L1
    """
    return f"SecNo{sec_no}_{section_label}_L{pattern_length}"


def create_rhythm_click_track(
    grid_times: np.ndarray,
    pattern_values: np.ndarray,
    median_tick_phases: np.ndarray,
    positions: np.ndarray,
    pattern_length: int,
    section_start: float,
    total_duration: float,
    sr: int = 44100,
) -> np.ndarray:
    """
    Create a click track for a section based on rhythm pattern data and grid times.

    For each active position in the rhythm pattern (pattern_value > 0), places
    clicks at every repetition of that position across the section, using actual
    grid times from the anchored CSV and applying median_tick_phase micro-timing.

    Parameters
    ----------
    grid_times : np.ndarray
        All grid times from the anchored CSV, shape (n_bars, 16).
        Indexed as grid_times[bar_index, tick_16th].
    pattern_values : np.ndarray
        Pattern value per position (1.0, 0.5, or 0.0). Length = pattern_length * 16.
    median_tick_phases : np.ndarray
        Median tick phase per position (micro-timing offset in phase units).
        Length = pattern_length * 16.
    positions : np.ndarray
        1-based position indices. Length = pattern_length * 16.
    pattern_length : int
        Pattern length in bars (1, 2, or 4).
    section_start : float
        Absolute start time of the section (used_section_start).
    total_duration : float
        Duration of the section audio in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    tuple of (np.ndarray, list)
        Click track waveform (mono, float32) and list of
        (relative_time, pattern_value) for each placed click.
    """
    total_samples = int(total_duration * sr)
    click_track = np.zeros(total_samples, dtype=np.float32)

    # Pre-generate click templates at different volumes
    click_full = generate_click(sr, CLICK_FREQUENCY, CLICK_DURATION, amplitude=1.0)
    click_half = generate_click(sr, 2000, CLICK_DURATION, amplitude=1.0)  # small click: lower pitch (2kHz), same volume

    n_bars = grid_times.shape[0]
    n_repetitions = n_bars // pattern_length

    click_events = []  # (relative_time, pattern_value)

    for pos_idx, (position, pv, phase) in enumerate(
        zip(positions, pattern_values, median_tick_phases)
    ):
        if pv == 0.0 or np.isnan(pv):
            continue

        # Position is 1-based: bar_in_pattern = (position-1) // 16, tick = (position-1) % 16
        bar_in_pattern = (int(position) - 1) // 16
        tick_16th = (int(position) - 1) % 16

        # Select click based on pattern_value
        click = click_full if pv >= 1.0 else click_half

        # Place click at every repetition of this pattern position
        for rep in range(n_repetitions):
            bar_index = rep * pattern_length + bar_in_pattern

            if bar_index >= n_bars:
                break

            grid_time = grid_times[bar_index, tick_16th]
            if np.isnan(grid_time):
                continue

            # Apply micro-timing: median_tick_phase is in phase units (0-1 within a beat)
            # Convert to seconds using step duration (time between adjacent 16th notes)
            if tick_16th < 15:
                next_grid = grid_times[bar_index, tick_16th + 1]
            elif bar_index + 1 < n_bars:
                next_grid = grid_times[bar_index + 1, 0]
            else:
                next_grid = np.nan

            if not np.isnan(phase) and not np.isnan(next_grid):
                step_duration = next_grid - grid_time
                click_time = grid_time + phase * step_duration
            else:
                click_time = grid_time

            # Convert to sample position relative to section start
            relative_time = click_time - section_start
            sample_pos = int(relative_time * sr)

            if sample_pos >= total_samples:
                continue

            # Clamp to start if phase pushes click slightly before section
            if sample_pos < 0:
                sample_pos = 0
                relative_time = 0.0

            # Add click to track
            end_pos = min(sample_pos + len(click), total_samples)
            click_len = end_pos - sample_pos
            if click_len > 0:
                click_track[sample_pos:end_pos] += click[:click_len]
                click_events.append((relative_time, pv))

    return click_track, click_events


def plot_waveform_with_clicks(
    audio: np.ndarray,
    click_events: List[Tuple[float, float]],
    sr: int,
    output_path: Path,
    title: str = '',
    grid_times: Optional[np.ndarray] = None,
    section_start: float = 0.0,
    bar_numbers: Optional[List[int]] = None,
    pattern_length: Optional[int] = None,
):
    """
    Plot section waveform with click positions and bar/beat/16th grid.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (mono).
    click_events : list of (float, float)
        List of (relative_time, pattern_value) for each click.
    sr : int
        Sample rate.
    output_path : Path
        Output PNG path.
    title : str
        Plot title.
    grid_times : np.ndarray, optional
        Grid times array (n_bars, 16) with absolute times.
    section_start : float
        Absolute start time of the section.
    bar_numbers : list of int, optional
        Original bar numbers for labeling.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _plot_waveform_on_ax(plt.figure(figsize=(14, 3)).add_subplot(111),
                         audio, click_events, sr, title,
                         grid_times, section_start, bar_numbers, pattern_length)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def _plot_waveform_on_ax(
    ax,
    audio: np.ndarray,
    click_events: List[Tuple[float, float]],
    sr: int,
    title: str = '',
    grid_times: Optional[np.ndarray] = None,
    section_start: float = 0.0,
    bar_numbers: Optional[List[int]] = None,
    pattern_length: Optional[int] = None,
):
    """Render waveform + clicks + grid onto a matplotlib Axes."""
    from matplotlib.lines import Line2D

    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))
    ymin, ymax = -1.0, 1.0
    amp_max = np.abs(audio).max()
    if amp_max > 0:
        ymin, ymax = -amp_max * 1.05, amp_max * 1.05

    # Waveform
    ax.plot(time_axis, audio, color='#4a90d9', linewidth=0.3, alpha=0.7, zorder=2)

    # Click lines
    for t, pv in sorted(click_events):
        color = '#e74c3c' if pv >= 1.0 else '#f39c12'
        alpha = 0.9 if pv >= 1.0 else 0.6
        ax.axvline(t, color=color, alpha=alpha, linewidth=0.5, zorder=3)

    ax.set_xlim(0, duration)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Amplitude')
    if title:
        ax.set_title(title, fontsize=10)

    # Bottom x-axis: bar/beat/16th ticks
    if grid_times is not None:
        n_bars_grid = grid_times.shape[0]

        # bold vertical separators between loops (every pattern_length bars)
        if pattern_length:
            for _b in range(pattern_length, n_bars_grid, pattern_length):
                _rt = grid_times[_b, 0] - section_start
                if not np.isnan(_rt):
                    ax.axvline(_rt, color='#111111', lw=2.5, zorder=5)

        tick_positions = []
        tick_labels = []
        major_positions = []  # bars
        minor_positions = []  # beats

        for bar_idx in range(n_bars_grid):
            bar_label = str(bar_numbers[bar_idx]) if bar_numbers else str(bar_idx)
            for tick in range(16):
                gt = grid_times[bar_idx, tick]
                if np.isnan(gt):
                    continue
                rel_t = gt - section_start

                if tick == 0:
                    tick_positions.append(rel_t)
                    tick_labels.append(f'{bar_label}.1')
                    major_positions.append(rel_t)
                elif tick % 4 == 0:
                    beat_num = tick // 4 + 1
                    tick_positions.append(rel_t)
                    tick_labels.append(f'.{beat_num}')
                    minor_positions.append(rel_t)
                else:
                    tick_positions.append(rel_t)
                    tick_labels.append('')

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=5)
        ax.set_xlabel('Bar / Beat / 16th', fontsize=8)

        # Style: bar ticks longer, beat ticks medium, 16th ticks short
        for tick_obj in ax.xaxis.get_major_ticks():
            pos = tick_obj.get_loc()
            if pos in major_positions:
                tick_obj.tick1line.set_markersize(8)
                tick_obj.tick1line.set_markeredgewidth(1.0)
                tick_obj.label1.set_fontweight('bold')
                tick_obj.label1.set_fontsize(7)
            elif pos in minor_positions:
                tick_obj.tick1line.set_markersize(5)
                tick_obj.tick1line.set_markeredgewidth(0.7)
                tick_obj.label1.set_fontsize(6)
            else:
                tick_obj.tick1line.set_markersize(2)
                tick_obj.tick1line.set_markeredgewidth(0.3)

        # Top x-axis: absolute time in seconds
        ax_top = ax.twiny()
        ax_top.set_xlim(section_start, section_start + duration)
        ax_top.set_xlabel('Time (s)', fontsize=8)
        ax_top.tick_params(labelsize=6)
    else:
        ax.set_xlabel('Time (s)')

    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linewidth=1.5, label='x = 1.0'),
        Line2D([0], [0], color='#f39c12', linewidth=1.5, label='. = 0.5'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)


def process_section_wav(
    section_wav: Path,
    anchored_csv: Path,
    rhythm_df: pd.DataFrame,
    output_path: Path,
    sr: int = 44100,
    click_volume_db: float = 0.0,
    verbose: bool = True,
    stems_dir: Optional[Path] = None,
    add_fullmix: bool = False,
    fullmix_click_gain: float = 3.0,
) -> Optional[Path]:
    """
    Create a click-track version of a section WAV.

    If ``add_fullmix`` is set and ``stems_dir`` points at a 1_stems folder, an
    additional ``*_section_clicks_fullmix.wav`` is written where the SAME click
    track is mixed onto the full-mix (all stems summed) section audio.

    Parameters
    ----------
    section_wav : Path
        Path to the plain section WAV file.
    anchored_csv : Path
        Path to the 6.2 anchored CSV (has grid_time data + metadata).
    rhythm_df : pd.DataFrame
        Rows from the rhythm patterns CSV matching this section.
    output_path : Path
        Output WAV path for the click version.
    sr : int
        Sample rate.
    click_volume_db : float
        Click volume in dB relative to full scale.
    verbose : bool
        Print progress.

    Returns
    -------
    Optional[Path]
        Path to created file, or None on failure.
    """
    # Read metadata from anchored CSV
    metadata = read_metadata(anchored_csv)
    used_section_start = float(metadata['used_section_start'])
    used_section_end = float(metadata['used_section_end'])
    pattern_length = int(metadata['pattern_length'])

    # Read anchored CSV grid data
    df_grid = pd.read_csv(anchored_csv, comment='#')

    # Build grid_times array: (n_bars, 16)
    bar_numbers = sorted(df_grid['bar_number'].unique())
    n_bars = len(bar_numbers)
    grid_times = np.full((n_bars, 16), np.nan)

    for _, row in df_grid.iterrows():
        bar_idx = bar_numbers.index(row['bar_number'])
        tick = int(row['tick_16th'])
        if pd.notna(row.get('grid_time')):
            grid_times[bar_idx, tick] = row['grid_time']

    # Get rhythm pattern data for this section
    if rhythm_df.empty:
        if verbose:
            print(f"    Skipping {section_wav.name}: no rhythm pattern data")
        return None

    positions = rhythm_df['position'].values
    pattern_values = rhythm_df['pattern_value'].values
    median_tick_phases = rhythm_df['median_tick_phase'].values

    # Replace NaN phases with 0
    median_tick_phases = np.where(np.isnan(median_tick_phases), 0.0, median_tick_phases)

    # Load section audio (mono for mixing)
    audio, file_sr = librosa.load(str(section_wav), sr=sr, mono=True)
    total_duration = len(audio) / sr

    # Create click track
    click_track, click_events = create_rhythm_click_track(
        grid_times=grid_times,
        pattern_values=pattern_values,
        median_tick_phases=median_tick_phases,
        positions=positions,
        pattern_length=pattern_length,
        section_start=used_section_start,
        total_duration=total_duration,
        sr=sr,
    )

    # Mix
    mixed = mix_audio_with_clicks(audio, click_track, click_volume_db=click_volume_db)

    # Save WAV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mixed, sr)

    # ADDITIONAL variant: same drum click track over the FULL-MIX (all stems) section
    if add_fullmix and stems_dir is not None:
        fullmix_audio = build_fullmix_section(
            stems_dir, used_section_start, used_section_end, len(audio), sr=sr
        )
        if fullmix_audio is not None:
            # Clicks a bit louder over the full mix (they must cut through all stems).
            # fullmix_click_gain is an AMPLITUDE factor (3.0 = +200%) -> dB.
            fm_click_db = click_volume_db + 20.0 * np.log10(fullmix_click_gain)
            mixed_fm = mix_audio_with_clicks(
                fullmix_audio, click_track, click_volume_db=fm_click_db
            )
            fm_path = output_path.with_name(output_path.stem + '_fullmix.wav')
            sf.write(str(fm_path), mixed_fm, sr)
            if verbose:
                print(f"      + {fm_path.name}  (clicks over full mix)")
        elif verbose:
            print(f"      (no stems in {stems_dir}, skipped fullmix variant)")

    # Build rhythm pattern string: x=1.0, .=0.5, o=0.0
    # Grouped in beats of 4, separated by spaces; | separates bars
    pattern_chars = []
    for i, pv in enumerate(pattern_values):
        if i > 0 and i % 16 == 0:
            pattern_chars.append(' | ')
        elif i > 0 and i % 4 == 0:
            pattern_chars.append(' - ')
        if np.isnan(pv) or pv == 0.0:
            pattern_chars.append('o')
        elif pv >= 1.0:
            pattern_chars.append('x')
        else:
            pattern_chars.append('.')
    pattern_str = ''.join(pattern_chars)

    # Save individual PNG
    plot_path = output_path.with_suffix('.png')
    section_label = metadata.get('section_label', '')
    title = f"{output_path.stem}  ({section_label}, L={pattern_length}, {n_bars} bars, {len(click_events)} clicks)\nPattern: {pattern_str}"
    plot_waveform_with_clicks(audio, click_events, sr, plot_path, title=title,
                              grid_times=grid_times, section_start=used_section_start,
                              bar_numbers=bar_numbers, pattern_length=pattern_length)

    if verbose:
        n_active = int(np.sum(pattern_values > 0))
        print(f"    ✓ {output_path.name}  ({n_active} active positions, {len(click_events)} clicks, {n_bars} bars)")

    # Return path + plot data for combined PDF
    plot_info = {
        'audio': audio,
        'click_events': click_events,
        'sr': sr,
        'title': title,
        'grid_times': grid_times,
        'section_start': used_section_start,
        'bar_numbers': bar_numbers,
        'pattern_length': pattern_length,
    }
    return output_path, plot_info


def save_combined_pdf(
    plot_infos: List[Dict],
    output_path: Path,
):
    """Save all section waveform+click plots as a single multi-page PDF."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        for info in plot_infos:
            fig, ax = plt.subplots(figsize=(14, 3))
            _plot_waveform_on_ax(
                ax,
                audio=info['audio'],
                click_events=info['click_events'],
                sr=info['sr'],
                title=info['title'],
                grid_times=info.get('grid_times'),
                section_start=info.get('section_start', 0.0),
                bar_numbers=info.get('bar_numbers'),
                pattern_length=info.get('pattern_length'),
            )
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def write_clicks_to_section_wavs(
    track_dir: Path,
    stems: Optional[List[str]] = None,
    sr: int = 44100,
    click_volume_db: float = 0.0,
    verbose: bool = True,
    add_fullmix: bool = True,
    fullmix_click_gain: float = 3.0,
) -> List[Path]:
    """
    Write rhythm pattern click tracks onto all section WAVs for a track.

    Parameters
    ----------
    track_dir : Path
        Track output directory (e.g. .../93_positions - Ariana Grande/).
    stems : list of str, optional
        Which stems to process. If None, processes all stems found in 9.1_sections.
    sr : int
        Sample rate.
    click_volume_db : float
        Click volume in dB.
    verbose : bool
        Print progress.

    Returns
    -------
    List[Path]
        List of created click WAV files.
    """
    track_dir = Path(track_dir)
    sections_dir = track_dir / '9.1_sections'
    filtered_dir = track_dir / '6.2_filtered_patterns'
    rhythm_dir = track_dir / '6.6_anchored_rhythm_histograms'
    stems_dir_1 = track_dir / '1_stems'          # for the full-mix click variant

    if not sections_dir.exists():
        print(f"Error: 9.1_sections not found in {track_dir}")
        return []

    # Detect track_id from rhythm patterns CSV name
    track_id = track_dir.name

    # Discover stems
    if stems is None:
        stems = [d.name for d in sorted(sections_dir.iterdir()) if d.is_dir()]

    if verbose:
        print(f"\n[write_clicks_to_sectionwavs]")
        print(f"  Track: {track_id}")
        print(f"  Stems: {', '.join(stems)}")

    created_files = []

    for stem in stems:
        stem_sections_dir = sections_dir / stem
        stem_filtered_dir = filtered_dir / stem
        stem_rhythm_dir = rhythm_dir / stem

        if not stem_sections_dir.exists():
            if verbose:
                print(f"\n  Skipping stem '{stem}': no section WAVs")
            continue

        # Load rhythm patterns CSV for this stem
        rhythm_csv = stem_rhythm_dir / f'{track_id}_filtered_anchored_rhythm_patterns.csv'
        if not rhythm_csv.exists():
            if verbose:
                print(f"\n  Skipping stem '{stem}': no rhythm patterns CSV")
            continue

        df_rhythm = pd.read_csv(rhythm_csv)

        if verbose:
            print(f"\n  Stem: {stem}")
            print(f"    Rhythm CSV: {rhythm_csv.name}")

        # Process each section WAV
        section_wavs = sorted(
            f for f in stem_sections_dir.glob('*_section.wav')
            if not f.name.startswith('._')
        )
        stem_plot_infos = []

        for section_wav in section_wavs:
            parsed = parse_section_filename(section_wav.name)
            if parsed is None:
                if verbose:
                    print(f"    Skipping {section_wav.name}: could not parse filename")
                continue

            # Build section_id to match rhythm CSV rows
            section_id = build_section_id(
                parsed['sec_no'], parsed['section_label'], parsed['pattern_length']
            )

            # Filter rhythm data for this section
            rhythm_rows = df_rhythm[df_rhythm['section_id'] == section_id]

            # Find matching anchored CSV
            anchored_csv = stem_filtered_dir / f"{parsed['base']}_anchored.csv"
            if not anchored_csv.exists():
                if verbose:
                    print(f"    Skipping {section_wav.name}: anchored CSV not found")
                continue

            # Output path: replace _section.wav with _section_clicks.wav
            output_name = section_wav.stem.replace('_section', '_section_clicks') + '.wav'
            output_path = stem_sections_dir / output_name

            result = process_section_wav(
                section_wav=section_wav,
                anchored_csv=anchored_csv,
                rhythm_df=rhythm_rows,
                output_path=output_path,
                sr=sr,
                click_volume_db=click_volume_db,
                verbose=verbose,
                stems_dir=stems_dir_1,
                add_fullmix=add_fullmix,
                fullmix_click_gain=fullmix_click_gain,
            )

            if result is not None:
                wav_path, plot_info = result
                created_files.append(wav_path)
                stem_plot_infos.append(plot_info)

        # Save combined PDF for this stem
        if stem_plot_infos:
            pdf_path = stem_sections_dir / f'{track_id}_section_clicks.pdf'
            save_combined_pdf(stem_plot_infos, pdf_path)
            if verbose:
                print(f"    ✓ {pdf_path.name}  ({len(stem_plot_infos)} pages)")

    if verbose:
        print(f"\n  Created {len(created_files)} click WAV files")

    return created_files


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Write rhythm pattern click tracks onto section WAVs'
    )
    parser.add_argument(
        'track_dir',
        type=str,
        help='Track output directory (containing 9.1_sections, 6.2_filtered_patterns, 6.6_anchored_rhythm_histograms)'
    )
    parser.add_argument(
        '--stems',
        type=str,
        default=None,
        help='Comma-separated list of stems to process (default: all)'
    )
    parser.add_argument(
        '--click-volume-db',
        type=float,
        default=0.0,
        help='Click volume in dB (default: 0.0)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=44100,
        help='Sample rate (default: 44100)'
    )
    parser.add_argument(
        '--no-fullmix',
        action='store_true',
        help='Do NOT also write the *_section_clicks_fullmix.wav variant (clicks over the all-stems mix)'
    )
    parser.add_argument(
        '--fullmix-click-gain',
        type=float,
        default=3.0,
        help='Amplitude factor for clicks in the fullmix variant (default 3.0 = +200%% louder)'
    )

    args = parser.parse_args()

    stems_list = args.stems.split(',') if args.stems else None

    results = write_clicks_to_section_wavs(
        track_dir=Path(args.track_dir),
        stems=stems_list,
        sr=args.sr,
        click_volume_db=args.click_volume_db,
        verbose=True,
        add_fullmix=not args.no_fullmix,
        fullmix_click_gain=args.fullmix_click_gain,
    )

    print(f"\nDone. Created {len(results)} click WAV files.")
