#!/usr/bin/env python3
"""
Step 13: Spotify Audio Features Analysis

This module fetches audio features from the Spotify API for a given track.
Requires Spotify API credentials (client_id and client_secret).

Audio Features retrieved:
- danceability: How suitable for dancing (0.0-1.0)
- energy: Perceptual intensity/activity (0.0-1.0)
- valence: Musical positiveness/happiness (0.0-1.0)
- tempo: Estimated tempo in BPM
- loudness: Overall loudness in dB
- speechiness: Presence of spoken words (0.0-1.0)
- acousticness: Confidence track is acoustic (0.0-1.0)
- instrumentalness: Predicts if track has no vocals (0.0-1.0)
- liveness: Presence of audience (0.0-1.0)
- key: Pitch class (0=C, 1=C#, ..., 11=B)
- mode: Modality (0=minor, 1=major)
- time_signature: Estimated time signature (3-7)

Output folder: 13_spotify/

Reference:
    Spotify Web API - Audio Features
    https://developer.spotify.com/documentation/web-api/reference/get-audio-features
"""

import sys
import json
import argparse
import base64
import csv
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import config
import importlib.util
_config_path = Path(__file__).parent / "config.py"
if _config_path.exists():
    spec = importlib.util.spec_from_file_location("config_module", _config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.config
else:
    config = None


# Spotify API endpoints
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
SPOTIFY_AUDIO_FEATURES_URL = "https://api.spotify.com/v1/audio-features"


# Default paths for local groove-data
GROOVE_DATA_DIR = Path(__file__).parent.parent / "groove-data"
SPOTIFY_IDS_CSV = GROOVE_DATA_DIR / "spotify" / "spotify_ids.csv"
SPOTIFY_AUDIOANALYSIS_DIR = GROOVE_DATA_DIR / "spotify_audioanalysis"


def lookup_spotify_id(song_id: int, csv_path: Path = SPOTIFY_IDS_CSV) -> Optional[str]:
    """
    Look up Spotify track ID from song_id using the CSV mapping file.

    Parameters
    ----------
    song_id : int
        The song ID to look up (numeric part of track_id like "17" from "17_Panini")
    csv_path : Path
        Path to the spotify_ids.csv file

    Returns
    -------
    str or None
        Spotify track ID if found, None otherwise
    """
    if not csv_path.exists():
        return None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                try:
                    if int(row[0]) == int(song_id):
                        return row[1]  # spotify_id
                except (ValueError, IndexError):
                    continue
    return None


def load_sections_from_audioanalysis(
    spotify_id: str,
    audioanalysis_dir: Path = SPOTIFY_AUDIOANALYSIS_DIR
) -> Optional[List[Dict[str, Any]]]:
    """
    Load sections data from Spotify audio analysis JSON file.

    Parameters
    ----------
    spotify_id : str
        Spotify track ID
    audioanalysis_dir : Path
        Directory containing audio analysis JSON files

    Returns
    -------
    list or None
        List of section dictionaries, or None if file not found
    """
    json_path = audioanalysis_dir / f"{spotify_id}.json"
    if not json_path.exists():
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get("sections", [])


def plot_sections_timeline(
    sections: List[Dict[str, Any]],
    snippet_start: float,
    snippet_duration: float,
    output_path: Path,
    track_name: str = "",
    song_id: Optional[int] = None,
    verbose: bool = True
) -> Optional[Path]:
    """
    Create a horizontal bar plot showing song sections within the snippet timerange.

    Parameters
    ----------
    sections : list
        List of section dictionaries with 'start', 'duration', 'loudness', 'tempo', etc.
    snippet_start : float
        Start time of the snippet in seconds
    snippet_duration : float
        Duration of the snippet in seconds
    output_path : Path
        Path to save the plot
    track_name : str
        Track name for the title
    song_id : int, optional
        Song ID to include in the title
    verbose : bool
        Print progress messages

    Returns
    -------
    Path or None
        Path to saved plot, or None if no sections overlap with snippet
    """
    snippet_end = snippet_start + snippet_duration

    # Filter sections that overlap with snippet timerange
    overlapping_sections = []
    for section in sections:
        sec_start = section["start"]
        sec_end = sec_start + section["duration"]

        # Check if section overlaps with snippet
        if sec_start < snippet_end and sec_end > snippet_start:
            overlapping_sections.append(section)

    if not overlapping_sections:
        if verbose:
            print("  No sections overlap with snippet timerange")
        return None

    # Create color map based on section index (cycling through colors)
    colors = plt.cm.Set3(np.linspace(0, 1, 12))  # 12 distinct colors

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot each section as a horizontal bar
    y_pos = 0.5
    bar_height = 0.6

    for i, section in enumerate(overlapping_sections):
        sec_start = section["start"]
        sec_end = sec_start + section["duration"]

        # Clip section to snippet bounds for display
        display_start = max(sec_start, snippet_start)
        display_end = min(sec_end, snippet_end)
        display_width = display_end - display_start

        color = colors[i % len(colors)]

        # Draw the section bar
        rect = mpatches.FancyBboxPatch(
            (display_start, y_pos - bar_height/2),
            display_width, bar_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)

        # Add section info as label (if bar is wide enough)
        if display_width > snippet_duration * 0.08:
            label_x = display_start + display_width / 2
            # Section info: tempo and key
            tempo = section.get("tempo", 0)
            key_num = section.get("key", -1)
            mode = section.get("mode", 0)

            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_str = key_names[key_num] if 0 <= key_num < 12 else "?"
            mode_str = "maj" if mode == 1 else "min"

            label = f"Sec {i+1}\n{tempo:.0f} BPM\n{key_str} {mode_str}"
            ax.text(label_x, y_pos, label, ha='center', va='center',
                   fontsize=8, fontweight='bold')

    # Draw snippet boundaries
    ax.axvline(x=snippet_start, color='green', linestyle='--', linewidth=2, label='Snippet start')
    ax.axvline(x=snippet_end, color='red', linestyle='--', linewidth=2, label='Snippet end')

    # Add time axis markers for full song context
    song_duration = sections[-1]["start"] + sections[-1]["duration"] if sections else snippet_end

    # Set axis limits with some padding
    x_min = max(0, snippet_start - snippet_duration * 0.1)
    x_max = min(song_duration, snippet_end + snippet_duration * 0.1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)

    # Labels and formatting
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_yticks([])

    # Build title with optional song_id
    if song_id is not None and track_name:
        title = f'Song Sections Timeline - {song_id}: {track_name}'
    elif song_id is not None:
        title = f'Song Sections Timeline - {song_id}'
    elif track_name:
        title = f'Song Sections Timeline - {track_name}'
    else:
        title = 'Song Sections Timeline'
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add legend
    ax.legend(loc='upper right', fontsize=9)

    # Add grid for time reference
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {output_path.name}")

    return output_path


def get_spotify_token(client_id: str, client_secret: str) -> Optional[str]:
    """
    Get Spotify API access token using client credentials flow.

    Parameters
    ----------
    client_id : str
        Spotify API client ID
    client_secret : str
        Spotify API client secret

    Returns
    -------
    str or None
        Access token if successful, None otherwise
    """
    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()

    request = urllib.request.Request(SPOTIFY_AUTH_URL, data=data, headers=headers)

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode())
            return result.get("access_token")
    except urllib.error.URLError as e:
        print(f"  ERROR getting Spotify token: {e}")
        return None


def search_track(token: str, track_name: str, artist: Optional[str] = None) -> Optional[str]:
    """
    Search for a track on Spotify and return its ID.

    Parameters
    ----------
    token : str
        Spotify API access token
    track_name : str
        Name of the track to search for
    artist : str, optional
        Artist name to narrow search

    Returns
    -------
    str or None
        Spotify track ID if found, None otherwise
    """
    query = track_name
    if artist:
        query = f"track:{track_name} artist:{artist}"

    params = urllib.parse.urlencode({
        "q": query,
        "type": "track",
        "limit": 1
    })

    url = f"{SPOTIFY_SEARCH_URL}?{params}"
    headers = {"Authorization": f"Bearer {token}"}

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request) as response:
            result = json.loads(response.read().decode())
            tracks = result.get("tracks", {}).get("items", [])
            if tracks:
                return tracks[0]["id"]
            return None
    except urllib.error.URLError as e:
        print(f"  ERROR searching track: {e}")
        return None


def get_audio_features(token: str, track_id: str) -> Optional[Dict[str, Any]]:
    """
    Get audio features for a Spotify track.

    Parameters
    ----------
    token : str
        Spotify API access token
    track_id : str
        Spotify track ID

    Returns
    -------
    dict or None
        Audio features dictionary if successful, None otherwise
    """
    url = f"{SPOTIFY_AUDIO_FEATURES_URL}/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as e:
        print(f"  ERROR getting audio features: {e}")
        return None


def run_spotify_analysis(
    track_name: str,
    output_dir: str,
    track_id: str,
    artist: Optional[str] = None,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    spotify_track_id: Optional[str] = None,
    snippet_start: Optional[float] = None,
    snippet_duration: Optional[float] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Spotify audio features analysis and section timeline plotting.

    Parameters
    ----------
    track_name : str
        Name of the track (used for searching if spotify_track_id not provided)
    output_dir : str
        Output directory for the track
    track_id : str
        Internal track identifier for output files (e.g., "17_Panini - Lil Nas X")
    artist : str, optional
        Artist name to help with search
    client_id : str, optional
        Spotify API client ID (can also use env var SPOTIFY_CLIENT_ID)
    client_secret : str, optional
        Spotify API client secret (can also use env var SPOTIFY_CLIENT_SECRET)
    spotify_track_id : str, optional
        Direct Spotify track ID (skips search if provided)
    snippet_start : float, optional
        Start time of the snippet in seconds (for section timeline plot)
    snippet_duration : float, optional
        Duration of the snippet in seconds (for section timeline plot)
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results dictionary with audio features and output paths
    """
    import os

    # Get credentials from params or environment
    if client_id is None:
        client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if client_secret is None:
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

    # Create output directory
    output_path = Path(output_dir) / "13_spotify"
    output_path.mkdir(parents=True, exist_ok=True)

    output_json = output_path / f"{track_id}_spotify_features.json"

    results = {
        "track_id": track_id,
        "track_name": track_name,
        "artist": artist,
        "spotify_track_id": None,
        "audio_features": {},
        "errors": []
    }

    if verbose:
        print(f"\n[Step 13: Spotify Analysis] Fetching audio features...")
        print(f"  Track: {track_name}")
        if artist:
            print(f"  Artist: {artist}")

    # Check credentials
    if not client_id or not client_secret:
        error_msg = "Spotify credentials not provided. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables."
        results["errors"].append(error_msg)
        if verbose:
            print(f"  ERROR: {error_msg}")

        # Save results even with error
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        results["output_json"] = str(output_json)
        return results

    # Get access token
    if verbose:
        print(f"  Authenticating with Spotify API...")

    token = get_spotify_token(client_id, client_secret)
    if not token:
        error_msg = "Failed to get Spotify access token"
        results["errors"].append(error_msg)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        results["output_json"] = str(output_json)
        return results

    # Get track ID (search or use provided)
    if spotify_track_id:
        sp_track_id = spotify_track_id
        if verbose:
            print(f"  Using provided Spotify track ID: {sp_track_id}")
    else:
        if verbose:
            print(f"  Searching for track...")
        sp_track_id = search_track(token, track_name, artist)

        if not sp_track_id:
            error_msg = f"Track not found on Spotify: {track_name}"
            results["errors"].append(error_msg)
            if verbose:
                print(f"  ERROR: {error_msg}")
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2)
            results["output_json"] = str(output_json)
            return results

        if verbose:
            print(f"  Found Spotify track ID: {sp_track_id}")

    results["spotify_track_id"] = sp_track_id

    # Get audio features
    if verbose:
        print(f"  Fetching audio features...")

    features = get_audio_features(token, sp_track_id)

    if not features:
        error_msg = "Failed to get audio features"
        results["errors"].append(error_msg)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        results["output_json"] = str(output_json)
        return results

    # Extract relevant features
    relevant_features = [
        "danceability", "energy", "valence", "tempo", "loudness",
        "speechiness", "acousticness", "instrumentalness", "liveness",
        "key", "mode", "time_signature", "duration_ms"
    ]

    results["audio_features"] = {
        key: features.get(key)
        for key in relevant_features
        if key in features
    }

    # Add Spotify URLs
    results["spotify_uri"] = features.get("uri")
    results["spotify_url"] = f"https://open.spotify.com/track/{sp_track_id}"

    # Save results
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    results["output_json"] = str(output_json)

    if verbose:
        print(f"  Saved: {output_json.name}")
        print(f"  Audio features retrieved:")
        for key, value in results["audio_features"].items():
            print(f"    {key}: {value}")
        print(f"  ✓ Spotify analysis completed")

    return results


def run_spotify_sections_analysis(
    output_dir: str,
    track_id: str,
    snippet_start: float,
    snippet_duration: float,
    track_name: str = "",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run Spotify section timeline analysis using local groove-data files.

    This function looks up the Spotify ID from local CSV and loads
    sections from local audio analysis JSON files.

    Parameters
    ----------
    output_dir : str
        Output directory for the track
    track_id : str
        Internal track identifier (e.g., "17_Panini - Lil Nas X")
    snippet_start : float
        Start time of the snippet in seconds
    snippet_duration : float
        Duration of the snippet in seconds
    track_name : str
        Track name for plot title
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results dictionary with sections data and output paths
    """
    # Create output directory
    output_path = Path(output_dir) / "13_spotify"
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "track_id": track_id,
        "snippet_start": snippet_start,
        "snippet_duration": snippet_duration,
        "sections": [],
        "errors": []
    }

    if verbose:
        print(f"\n[Step 13: Spotify Sections Analysis]")
        print(f"  Track ID: {track_id}")
        print(f"  Snippet: {snippet_start:.2f}s - {snippet_start + snippet_duration:.2f}s")

    # Extract numeric song_id from track_id (e.g., "17" from "17_Panini - Lil Nas X")
    try:
        song_id = int(track_id.split("_")[0])
    except (ValueError, IndexError):
        error_msg = f"Could not extract song_id from track_id: {track_id}"
        results["errors"].append(error_msg)
        if verbose:
            print(f"  ERROR: {error_msg}")
        return results

    # Look up Spotify ID from local CSV
    if verbose:
        print(f"  Looking up Spotify ID for song_id={song_id}...")

    spotify_id = lookup_spotify_id(song_id)

    if not spotify_id:
        error_msg = f"Spotify ID not found for song_id={song_id}"
        results["errors"].append(error_msg)
        if verbose:
            print(f"  WARNING: {error_msg}")
        return results

    results["spotify_id"] = spotify_id
    if verbose:
        print(f"  Found Spotify ID: {spotify_id}")

    # Load sections from local audio analysis JSON
    if verbose:
        print(f"  Loading sections from audio analysis...")

    sections = load_sections_from_audioanalysis(spotify_id)

    if not sections:
        error_msg = f"Sections not found for spotify_id={spotify_id}"
        results["errors"].append(error_msg)
        if verbose:
            print(f"  WARNING: {error_msg}")
        return results

    results["sections"] = sections
    if verbose:
        print(f"  Found {len(sections)} sections")

    # Plot section timeline
    output_plot = output_path / f"{track_id}_sections_timeline.png"

    plot_path = plot_sections_timeline(
        sections=sections,
        snippet_start=snippet_start,
        snippet_duration=snippet_duration,
        output_path=output_plot,
        track_name=track_name,
        song_id=song_id,
        verbose=verbose
    )

    if plot_path:
        results["output_plot"] = str(plot_path)

    # Save sections JSON
    output_json = output_path / f"{track_id}_sections.json"
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    results["output_json"] = str(output_json)

    # Save section changes CSV with timings relative to snippet start
    # Only include sections that overlap with the snippet
    snippet_end = snippet_start + snippet_duration
    output_csv = output_path / f"{track_id}_section_changes.csv"

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "section_num", "start_absolute_s", "start_relative_s",
            "duration_s", "tempo", "key", "mode", "loudness"
        ])

        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for i, section in enumerate(sections):
            sec_start = section["start"]
            sec_end = sec_start + section["duration"]

            # Only include sections that START within the snippet
            # (i.e., section changes that happen inside the snippet)
            if not (sec_start >= snippet_start and sec_start < snippet_end):
                continue

            # Calculate relative timing (relative to snippet start)
            relative_start = sec_start - snippet_start

            # Get key name
            key_num = section.get("key", -1)
            key_str = key_names[key_num] if 0 <= key_num < 12 else "?"
            mode_str = "major" if section.get("mode", 0) == 1 else "minor"

            writer.writerow([
                i + 1,
                round(sec_start, 3),
                round(relative_start, 3),
                round(section["duration"], 3),
                round(section.get("tempo", 0), 1),
                key_str,
                mode_str,
                round(section.get("loudness", 0), 2)
            ])

    results["output_csv"] = str(output_csv)

    # Compute onsets per pattern per section using flexStart data
    # Look for the 2-bar flexStart CSV in the 5_grid folder
    grid_dir = Path(output_dir) / "5_grid"
    flexstart_csv = grid_dir / f"{track_id}_comprehensive_phases_2bar_flexStart.csv"

    if flexstart_csv.exists():
        if verbose:
            print(f"  Computing onsets per pattern per section...")
        onsets_csv = compute_onsets_per_pattern_per_section(
            flexstart_csv=flexstart_csv,
            sections=sections,
            snippet_start=snippet_start,
            snippet_duration=snippet_duration,
            output_path=output_path,
            track_id=track_id,
            pattern_length=2,
            verbose=verbose
        )
        if onsets_csv:
            results["output_onsets_csv"] = str(onsets_csv)

        # Also create the onsets per pattern bar plot
        onsets_plot = plot_onsets_per_pattern(
            flexstart_csv=flexstart_csv,
            sections=sections,
            snippet_start=snippet_start,
            snippet_duration=snippet_duration,
            output_path=output_path,
            track_id=track_id,
            pattern_length=2,
            verbose=verbose
        )
        if onsets_plot:
            results["output_onsets_plot"] = str(onsets_plot)
    else:
        if verbose:
            print(f"  Skipping onsets per pattern (flexStart CSV not found)")

    if verbose:
        print(f"  Saved: {output_json.name}")
        print(f"  Saved: {output_csv.name}")
        print(f"  ✓ Spotify sections analysis completed")

    return results


def compute_onsets_per_pattern_per_section(
    flexstart_csv: Path,
    sections: List[Dict[str, Any]],
    snippet_start: float,
    snippet_duration: float,
    output_path: Path,
    track_id: str,
    pattern_length: int = 2,
    verbose: bool = True
) -> Optional[Path]:
    """
    Compute average onsets per pattern for each section.

    Reads the flexStart CSV to identify patterns (groups of L bars) and counts
    onsets within each pattern. Only includes full patterns that fall completely
    within each section.

    Parameters
    ----------
    flexstart_csv : Path
        Path to the comprehensive_phases_Lbar_flexStart.csv file
    sections : list
        List of section dictionaries with 'start' and 'duration' keys
    snippet_start : float
        Start time of the snippet in seconds (absolute)
    snippet_duration : float
        Duration of the snippet in seconds
    output_path : Path
        Output directory for the CSV
    track_id : str
        Track identifier for filename
    pattern_length : int
        Number of bars per pattern (L value, default 2)
    verbose : bool
        Print progress messages

    Returns
    -------
    Path or None
        Path to output CSV, or None if failed
    """
    if not flexstart_csv.exists():
        if verbose:
            print(f"  WARNING: flexStart CSV not found: {flexstart_csv}")
        return None

    snippet_end = snippet_start + snippet_duration

    # Read flexStart CSV, skip comment lines
    bars_data = []  # List of (bar_number, onset_time or None)
    with open(flexstart_csv, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('bar_number'):
                continue  # header
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    bar_num = int(parts[0])
                    onset_time = float(parts[3]) if parts[3] else None
                    bars_data.append((bar_num, onset_time))
                except (ValueError, IndexError):
                    continue

    if not bars_data:
        if verbose:
            print("  WARNING: No data found in flexStart CSV")
        return None

    # Group by bar number and collect onset times
    bar_onsets = {}  # bar_num -> list of onset times
    bar_times = {}   # bar_num -> (first_onset_time, last_onset_time) for timing
    for bar_num, onset_time in bars_data:
        if bar_num not in bar_onsets:
            bar_onsets[bar_num] = []
            bar_times[bar_num] = [None, None]
        if onset_time is not None:
            bar_onsets[bar_num].append(onset_time)
            if bar_times[bar_num][0] is None or onset_time < bar_times[bar_num][0]:
                bar_times[bar_num][0] = onset_time
            if bar_times[bar_num][1] is None or onset_time > bar_times[bar_num][1]:
                bar_times[bar_num][1] = onset_time

    # Get unique bar numbers sorted
    unique_bars = sorted(bar_onsets.keys())

    # Group bars into patterns (L bars each)
    patterns = []  # List of (pattern_idx, [bar_nums], onset_count, start_time, end_time)
    for i in range(0, len(unique_bars), pattern_length):
        pattern_bars = unique_bars[i:i + pattern_length]
        if len(pattern_bars) < pattern_length:
            continue  # Skip incomplete patterns

        # Count onsets in this pattern
        onset_count = sum(len(bar_onsets[b]) for b in pattern_bars)

        # Get pattern time range from onset times
        all_times = []
        for b in pattern_bars:
            all_times.extend(bar_onsets[b])

        if all_times:
            start_time = min(all_times)
            end_time = max(all_times)
        else:
            # No onsets, estimate from grid times if available
            start_time = None
            end_time = None

        patterns.append({
            'pattern_idx': i // pattern_length,
            'bars': pattern_bars,
            'onset_count': onset_count,
            'start_time': start_time,
            'end_time': end_time
        })

    if verbose:
        print(f"  Found {len(patterns)} complete {pattern_length}-bar patterns")

    # For each section, find full patterns within it
    output_csv = output_path / f"{track_id}_onsets_per_pattern_per_section.csv"

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'section_num', 'section_start_relative_s', 'num_patterns',
            'bars_included', 'total_onsets', 'avg_onsets_per_pattern'
        ])

        for i, section in enumerate(sections):
            sec_start = section['start']
            sec_end = sec_start + section['duration']

            # Only process sections that overlap with snippet
            if not (sec_start < snippet_end and sec_end > snippet_start):
                continue

            # Find patterns completely within this section
            section_patterns = []
            section_bars = []
            for p in patterns:
                if p['start_time'] is None or p['end_time'] is None:
                    continue
                # Check if pattern is completely within section
                if p['start_time'] >= sec_start and p['end_time'] <= sec_end:
                    section_patterns.append(p)
                    section_bars.extend(p['bars'])

            num_patterns = len(section_patterns)
            total_onsets = sum(p['onset_count'] for p in section_patterns)
            avg_onsets = total_onsets / num_patterns if num_patterns > 0 else 0

            # Format bars included as range or list
            if section_bars:
                bars_str = f"{min(section_bars)}-{max(section_bars)}"
            else:
                bars_str = ""

            relative_start = sec_start - snippet_start

            writer.writerow([
                i + 1,
                round(relative_start, 3),
                num_patterns,
                bars_str,
                total_onsets,
                round(avg_onsets, 2)
            ])

            if verbose:
                print(f"    Section {i+1}: {num_patterns} patterns, {total_onsets} onsets, avg={avg_onsets:.2f}")

    if verbose:
        print(f"  Saved: {output_csv.name}")

    return output_csv


def parse_filtered_csv_metadata(filtered_csv: Path) -> Dict[str, Any]:
    """
    Parse metadata from the header comments of a filtered flexStart CSV.

    Parameters
    ----------
    filtered_csv : Path
        Path to the _flexStart_filtered.csv file

    Returns
    -------
    dict
        Dictionary with parsed metadata (lower_bound, upper_bound, median, etc.)
    """
    metadata = {}

    if not filtered_csv.exists():
        return metadata

    with open(filtered_csv, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                break
            # Parse comment line: # key=value
            line = line.strip()[2:]  # Remove '# '
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                # Try to convert to numeric
                try:
                    if '.' in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value

    return metadata


def plot_onsets_per_pattern(
    flexstart_csv: Path,
    sections: List[Dict[str, Any]],
    snippet_start: float,
    snippet_duration: float,
    output_path: Path,
    track_id: str,
    pattern_length: int = 2,
    verbose: bool = True
) -> Optional[Path]:
    """
    Create a bar plot showing number of onsets per pattern.

    Primary x-axis shows pattern number, secondary x-axis shows relative time.
    Section boundaries are indicated with vertical lines.
    Horizontal lines show Tukey outlier thresholds and median (if Tukey method was used).

    Parameters
    ----------
    flexstart_csv : Path
        Path to the comprehensive_phases_Lbar_flexStart.csv file
    sections : list
        List of section dictionaries with 'start' and 'duration' keys
    snippet_start : float
        Start time of the snippet in seconds (absolute)
    snippet_duration : float
        Duration of the snippet in seconds
    output_path : Path
        Output directory for the plot
    track_id : str
        Track identifier for filename
    pattern_length : int
        Number of bars per pattern (L value, default 2)
    verbose : bool
        Print progress messages

    Returns
    -------
    Path or None
        Path to output PNG, or None if failed
    """
    if not flexstart_csv.exists():
        if verbose:
            print(f"  WARNING: flexStart CSV not found: {flexstart_csv}")
        return None

    snippet_end = snippet_start + snippet_duration

    # Try to find the filtered CSV to get Tukey threshold values
    filtered_csv = Path(str(flexstart_csv).replace('_flexStart.csv', '_flexStart_filtered.csv'))
    filter_metadata = parse_filtered_csv_metadata(filtered_csv)

    # Read flexStart CSV, skip comment lines
    bars_data = []
    with open(flexstart_csv, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            if line.startswith('bar_number'):
                continue
            parts = line.strip().split(',')
            if len(parts) >= 4:
                try:
                    bar_num = int(parts[0])
                    onset_time = float(parts[3]) if parts[3] else None
                    bars_data.append((bar_num, onset_time))
                except (ValueError, IndexError):
                    continue

    if not bars_data:
        if verbose:
            print("  WARNING: No data found in flexStart CSV")
        return None

    # Group by bar number and collect onset times
    bar_onsets = {}
    for bar_num, onset_time in bars_data:
        if bar_num not in bar_onsets:
            bar_onsets[bar_num] = []
        if onset_time is not None:
            bar_onsets[bar_num].append(onset_time)

    # Get unique bar numbers sorted
    unique_bars = sorted(bar_onsets.keys())

    # Group bars into patterns
    patterns = []
    for i in range(0, len(unique_bars), pattern_length):
        pattern_bars = unique_bars[i:i + pattern_length]
        if len(pattern_bars) < pattern_length:
            continue

        onset_count = sum(len(bar_onsets[b]) for b in pattern_bars)
        all_times = []
        for b in pattern_bars:
            all_times.extend(bar_onsets[b])

        if all_times:
            start_time = min(all_times)
            center_time = np.mean(all_times)
        else:
            start_time = None
            center_time = None

        patterns.append({
            'pattern_num': (i // pattern_length) + 1,
            'bars': pattern_bars,
            'onset_count': onset_count,
            'start_time': start_time,
            'center_time': center_time,
            'relative_time': (start_time - snippet_start) if start_time else None
        })

    if not patterns:
        if verbose:
            print("  WARNING: No complete patterns found")
        return None

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Data for plotting
    pattern_nums = [p['pattern_num'] for p in patterns]
    onset_counts = [p['onset_count'] for p in patterns]
    relative_times = [p['relative_time'] for p in patterns if p['relative_time'] is not None]

    # Bar plot
    bars = ax1.bar(pattern_nums, onset_counts, color='steelblue', edgecolor='black', alpha=0.8)

    # Add onset count labels on bars
    for bar, count in zip(bars, onset_counts):
        height = bar.get_height()
        ax1.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # Primary x-axis (pattern number)
    ax1.set_xlabel('Pattern Number', fontsize=11)
    ax1.set_ylabel('Number of Onsets', fontsize=11)
    ax1.set_xticks(pattern_nums)

    # Secondary x-axis (relative time)
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    # Create time labels for each pattern
    time_ticks = []
    time_labels = []
    for p in patterns:
        if p['relative_time'] is not None:
            time_ticks.append(p['pattern_num'])
            time_labels.append(f"{p['relative_time']:.1f}s")

    ax2.set_xticks(time_ticks)
    ax2.set_xticklabels(time_labels, fontsize=8)
    ax2.set_xlabel('Relative Time (from snippet start)', fontsize=10)

    # Add section boundaries as vertical lines
    section_colors = plt.cm.Set1(np.linspace(0, 1, 9))
    legend_handles = []

    for idx, section in enumerate(sections):
        sec_start = section['start']
        sec_end = sec_start + section['duration']

        # Check if section overlaps with snippet
        if not (sec_start < snippet_end and sec_end > snippet_start):
            continue

        sec_relative_start = sec_start - snippet_start

        # Find which pattern this section start falls into
        pattern_pos = None
        for p in patterns:
            if p['relative_time'] is not None:
                if p['relative_time'] <= sec_relative_start:
                    # Interpolate position between patterns
                    pattern_pos = p['pattern_num']
                    # Find next pattern for interpolation
                    for p2 in patterns:
                        if p2['pattern_num'] == p['pattern_num'] + 1 and p2['relative_time'] is not None:
                            # Linear interpolation
                            t_range = p2['relative_time'] - p['relative_time']
                            if t_range > 0:
                                frac = (sec_relative_start - p['relative_time']) / t_range
                                pattern_pos = p['pattern_num'] + frac
                            break

        # If section starts before all patterns, place line at start
        if pattern_pos is None and patterns and sec_relative_start < patterns[0].get('relative_time', float('inf')):
            pattern_pos = 0.5  # Place before first pattern

        # Only draw section line if it starts within the snippet and we found a position
        if sec_start >= snippet_start and sec_start < snippet_end and pattern_pos is not None:
            color = section_colors[idx % len(section_colors)]
            line = ax1.axvline(x=pattern_pos, color=color, linestyle='--',
                              linewidth=2, alpha=0.8)
            legend_handles.append(
                mpatches.Patch(color=color, label=f'Section {idx+1} ({sec_relative_start:.1f}s)')
            )

    # Add horizontal lines for Tukey thresholds (if available)
    filtering_method = filter_metadata.get('filtering_method', '')
    if 'Tukey' in str(filtering_method):
        lower_bound = filter_metadata.get('lower_bound')
        upper_bound = filter_metadata.get('upper_bound')
        median_val = filter_metadata.get('median')

        if lower_bound is not None:
            ax1.axhline(y=lower_bound, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
            legend_handles.append(
                mpatches.Patch(color='red', label=f'Lower bound ({lower_bound:.1f})')
            )
        if upper_bound is not None:
            ax1.axhline(y=upper_bound, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
            legend_handles.append(
                mpatches.Patch(color='red', label=f'Upper bound ({upper_bound:.1f})')
            )
        if median_val is not None:
            ax1.axhline(y=median_val, color='green', linestyle='-', linewidth=1.5, alpha=0.8)
            legend_handles.append(
                mpatches.Patch(color='green', label=f'Median ({median_val:.1f})')
            )

    # Add legend for sections and thresholds if any
    if legend_handles:
        ax1.legend(handles=legend_handles, loc='upper right', fontsize=9)

    # Title - add q1/q3 info if Tukey method was used, or note if running mean
    title = f'Onsets per {pattern_length}-Bar Pattern\n{track_id}'
    filtering_method = str(filter_metadata.get('filtering_method', ''))
    if 'Tukey' in filtering_method:
        q1 = filter_metadata.get('q1')
        q3 = filter_metadata.get('q3')
        if q1 is not None and q3 is not None:
            title += f'\nQ1={q1:.1f}, Q3={q3:.1f}'
    elif 'running mean' in filtering_method:
        no_of_rep_th = filter_metadata.get('no_of_repetitions_TH', 2)
        title += f'\nTukey not used, repetitions ≤ {no_of_rep_th}'
    ax1.set_title(title, fontsize=12, fontweight='bold')

    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_png = output_path / f"{track_id}_onsets_per_pattern.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {output_png.name}")

    return output_png


def main():
    """Command-line interface for Spotify analysis."""
    parser = argparse.ArgumentParser(
        description="Step 13: Fetch Spotify audio features for a track",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python spotify_analysis.py "Panini" -o output/ --track-id 17 --artist "Lil Nas X"
  python spotify_analysis.py "Billie Jean" -o output/ --track-id 42 --artist "Michael Jackson"

Environment variables:
  SPOTIFY_CLIENT_ID     - Spotify API client ID
  SPOTIFY_CLIENT_SECRET - Spotify API client secret
        """
    )

    parser.add_argument("track_name", help="Name of the track to search")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--track-id", required=True,
                        help="Internal track identifier")
    parser.add_argument("--artist", help="Artist name (helps with search accuracy)")
    parser.add_argument("--spotify-id", help="Direct Spotify track ID (skips search)")
    parser.add_argument("--client-id", help="Spotify API client ID")
    parser.add_argument("--client-secret", help="Spotify API client secret")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")

    args = parser.parse_args()

    results = run_spotify_analysis(
        track_name=args.track_name,
        output_dir=args.output_dir,
        track_id=args.track_id,
        artist=args.artist,
        client_id=args.client_id,
        client_secret=args.client_secret,
        spotify_track_id=args.spotify_id,
        verbose=not args.quiet
    )

    if not args.quiet:
        print(f"\nResults saved to: {results.get('output_json', 'N/A')}")

    return 0 if not results["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
