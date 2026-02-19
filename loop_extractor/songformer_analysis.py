"""
Step 2.5: SongFormer Music Structure Analysis

Runs SongFormer model to detect music structure (sections, boundaries).
State-of-the-art accuracy: ~70% boundary detection at 0.5s tolerance (vs ~60% Spotify).

Should run after beat detection (Step 2) and before grid calculations (Step 5),
as section boundaries can inform pattern selection and analysis.

Output format (MSA TXT):
    start_time_1 label_1
    start_time_2 label_2
    ...
    end_time end

Labels: intro, verse, chorus, bridge, inst, outro, silence, pre-chorus

Usage:
    python songformer_analysis.py <audio_file> -o <output_dir> [--track-id ID]
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add SongFormer paths
SONGFORMER_ROOT = Path(__file__).parent.parent / "songformer"
SONGFORMER_SRC = SONGFORMER_ROOT / "src" / "SongFormer"


def setup_songformer_paths():
    """Add SongFormer to Python path"""
    sys.path.insert(0, str(SONGFORMER_SRC))
    sys.path.insert(0, str(SONGFORMER_ROOT / "src" / "third_party"))
    # MuQ package is inside MuQ/src/
    sys.path.insert(0, str(SONGFORMER_ROOT / "src" / "third_party" / "MuQ" / "src"))
    # musicfm package
    sys.path.insert(0, str(SONGFORMER_ROOT / "src" / "third_party" / "musicfm"))
    # Change to SongFormer directory for relative imports
    os.chdir(str(SONGFORMER_SRC))


def run_songformer(
    audio_path: Path,
    output_dir: Path,
    track_id: str = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run SongFormer music structure analysis on an audio file.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file (MP3, WAV, etc.)
    output_dir : Path
        Output directory for results
    track_id : str, optional
        Track identifier for output filenames
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Results including sections, boundaries, and output file paths
    """
    setup_songformer_paths()

    # Import after path setup
    import math
    import importlib
    import numpy as np
    import scipy
    scipy.inf = np.inf  # monkey patch for msaf

    import torch
    import librosa
    from omegaconf import OmegaConf
    from ema_pytorch import EMA
    from muq import MuQ
    from musicfm.model.musicfm_25hz import MusicFM25Hz
    from postprocessing.functional import postprocess_functional_structure
    from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID, ID_TO_LABEL

    # Constants from SongFormer
    MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")
    AFTER_DOWNSAMPLING_FRAME_RATES = 8.333
    DATASET_LABEL = "SongForm-HX-8Class"
    DATASET_IDS = [5]
    TIME_DUR = 420
    INPUT_SAMPLING_RATE = 24000
    NUM_CLASSES = 128

    results = {
        "input_file": str(audio_path),
        "sections": [],
        "boundaries": [],
        "errors": []
    }

    if track_id is None:
        track_id = audio_path.stem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"SongFormer: Analyzing {audio_path.name}")

    # Set device - MPS doesn't support ComplexFloat needed for FFT/spectrograms,
    # so we must use CPU on Apple Silicon for this model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"  Using device: {device}")

    try:
        # Load models
        if verbose:
            print("  Loading MuQ model...")
        muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
        muq_model = muq_model.to(device).eval()

        if verbose:
            print("  Loading MusicFM model...")
        musicfm_model = MusicFM25Hz(
            is_flash=False,
            stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
            model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
        )
        musicfm_model = musicfm_model.to(device).eval()

        if verbose:
            print("  Loading SongFormer model...")
        module = importlib.import_module("models.SongFormer")
        Model = getattr(module, "Model")
        hp = OmegaConf.load(os.path.join("configs", "SongFormer.yaml"))
        msa_model = Model(hp)

        # Load checkpoint
        checkpoint_path = os.path.join("ckpts", "SongFormer.safetensors")
        from safetensors.torch import load_file
        ckpt = {"model_ema": load_file(checkpoint_path, device=str(device))}

        model_ema = EMA(msa_model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        msa_model.load_state_dict(model_ema.ema_model.state_dict())
        msa_model.to(device).eval()

        # Load audio
        if verbose:
            print(f"  Loading audio...")
        wav, sr = librosa.load(str(audio_path), sr=INPUT_SAMPLING_RATE)
        audio = torch.tensor(wav).to(device)
        audio_duration = len(wav) / INPUT_SAMPLING_RATE

        if verbose:
            print(f"  Audio duration: {audio_duration:.1f}s")

        # Prepare output arrays
        win_size = 420
        hop_size = 420
        total_len = ((audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR * TIME_DUR) + TIME_DUR
        total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

        logits = {
            "function_logits": np.zeros([total_frames, NUM_CLASSES]),
            "boundary_logits": np.zeros([total_frames]),
        }
        logits_num = {
            "function_logits": np.zeros([total_frames, NUM_CLASSES]),
            "boundary_logits": np.zeros([total_frames]),
        }

        # Prepare label masks
        dataset_id2label_mask = {}
        for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
            dataset_id2label_mask[key] = np.ones(NUM_CLASSES, dtype=bool)
            dataset_id2label_mask[key][allowed_ids] = False

        if verbose:
            print("  Running inference...")

        i = 0
        with torch.no_grad():
            while True:
                start_idx = i * INPUT_SAMPLING_RATE
                end_idx = min((i + win_size) * INPUT_SAMPLING_RATE, audio.shape[-1])
                if start_idx >= audio.shape[-1]:
                    break
                if end_idx - start_idx <= 1024:
                    break

                audio_seg = audio[start_idx:end_idx]

                # MuQ embedding (420s context)
                muq_output = muq_model(audio_seg.unsqueeze(0), output_hidden_states=True)
                muq_embd_420s = muq_output["hidden_states"][10]
                del muq_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # MusicFM embedding (420s context)
                _, musicfm_hidden_states = musicfm_model.get_predictions(audio_seg.unsqueeze(0))
                musicfm_embd_420s = musicfm_hidden_states[10]
                del musicfm_hidden_states
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 30s embeddings
                wraped_muq_embd_30s = []
                wraped_musicfm_embd_30s = []

                for idx_30s in range(i, i + hop_size, 30):
                    start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                    end_idx_30s = min(
                        (idx_30s + 30) * INPUT_SAMPLING_RATE,
                        audio.shape[-1],
                        (i + hop_size) * INPUT_SAMPLING_RATE,
                    )
                    if start_idx_30s >= audio.shape[-1]:
                        break
                    if end_idx_30s - start_idx_30s <= 1024:
                        continue

                    wraped_muq_embd_30s.append(
                        muq_model(
                            audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                            output_hidden_states=True,
                        )["hidden_states"][10]
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    wraped_musicfm_embd_30s.append(
                        musicfm_model.get_predictions(
                            audio[start_idx_30s:end_idx_30s].unsqueeze(0)
                        )[1][10]
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                wraped_musicfm_embd_30s = torch.concatenate(wraped_musicfm_embd_30s, dim=1)

                all_embds = [
                    wraped_musicfm_embd_30s,
                    wraped_muq_embd_30s,
                    musicfm_embd_420s,
                    muq_embd_420s,
                ]

                # Align embedding lengths
                if len(all_embds) > 1:
                    embd_lens = [x.shape[1] for x in all_embds]
                    min_embd_len = min(embd_lens)
                    for idx in range(len(all_embds)):
                        all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                embd = torch.concatenate(all_embds, axis=-1)

                dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)
                msa_info, chunk_logits = msa_model.infer(
                    input_embeddings=embd,
                    dataset_ids=dataset_ids,
                    label_id_masks=torch.Tensor(
                        dataset_id2label_mask[DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]]
                    ).to(device, dtype=bool).unsqueeze(0).unsqueeze(0),
                    with_logits=True,
                )

                start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                end_frame = start_frame + min(
                    math.ceil(hop_size * AFTER_DOWNSAMPLING_FRAME_RATES),
                    chunk_logits["boundary_logits"][0].shape[0],
                )

                logits["function_logits"][start_frame:end_frame, :] += (
                    chunk_logits["function_logits"][0].detach().cpu().numpy()
                )
                logits["boundary_logits"][start_frame:end_frame] = (
                    chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                )
                logits_num["function_logits"][start_frame:end_frame, :] += 1
                logits_num["boundary_logits"][start_frame:end_frame] += 1

                i += hop_size

        # Normalize logits
        logits["function_logits"] /= np.maximum(logits_num["function_logits"], 1)
        logits["boundary_logits"] /= np.maximum(logits_num["boundary_logits"], 1)

        # Post-process to get MSA output
        if verbose:
            print("  Post-processing...")

        # Create config object for postprocessing
        class PostprocessConfig:
            frame_rates = AFTER_DOWNSAMPLING_FRAME_RATES
            local_maxima_filter_size = 17  # Default from SongFormer config

        postprocess_logits = {
            "function_logits": torch.Tensor(logits["function_logits"]).unsqueeze(0),
            "boundary_logits": torch.Tensor(logits["boundary_logits"]).unsqueeze(0),
        }

        msa_output = postprocess_functional_structure(
            logits=postprocess_logits,
            config=PostprocessConfig(),
        )

        # Rule-based post-processing
        msa_output = rule_post_processing(msa_output)

        # Filter out sections beyond actual audio duration
        # (SongFormer pads to TIME_DUR=420s, causing spurious detections in padded silence)
        msa_output = [(t, label) for t, label in msa_output if t < audio_duration]
        # Add end marker at actual audio duration
        if msa_output:
            msa_output.append((audio_duration, "end"))

        # Convert to sections format
        sections = []
        for i in range(len(msa_output) - 1):
            start_time, label = msa_output[i]
            end_time = msa_output[i + 1][0]
            sections.append({
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "label": label
            })

        results["sections"] = sections
        results["boundaries"] = [round(t, 3) for t, _ in msa_output[:-1]]

        # Save outputs
        # 1. MSA TXT format
        msa_txt_path = output_dir / "SF_sections.txt"
        with open(msa_txt_path, 'w') as f:
            for time, label in msa_output:
                f.write(f"{time:.2f} {label}\n")
        results["output_msa_txt"] = str(msa_txt_path)

        # 2. JSON format
        json_path = output_dir / "SF_sections.json"
        with open(json_path, 'w') as f:
            json.dump({
                "track_id": track_id,
                "audio_file": str(audio_path),
                "duration": audio_duration,
                "sections": sections,
                "boundaries": results["boundaries"],
                "model": "SongFormer",
                "model_version": "HX-8Class"
            }, f, indent=2)
        results["output_json"] = str(json_path)

        # 3. CSV format (compatible with Spotify section_changes.csv)
        csv_path = output_dir / "SF_all_sections.csv"
        with open(csv_path, 'w') as f:
            f.write("section_num,start_s,duration_s,label\n")
            for i, section in enumerate(sections):
                f.write(f"{i},{section['start']},{section['duration']},{section['label']}\n")
        results["output_csv"] = str(csv_path)

        if verbose:
            print(f"  Found {len(sections)} sections:")
            for s in sections:
                print(f"    {s['start']:6.2f}s - {s['end']:6.2f}s: {s['label']}")
            print(f"  Saved: {msa_txt_path.name}")
            print(f"  Saved: {json_path.name}")
            print(f"  Saved: {csv_path.name}")

    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


def plot_songformer_snippet_sections(
    sections: List[Dict[str, Any]],
    snippet_start: float,
    snippet_duration: float,
    output_path: Path,
    track_name: str = "",
    song_id: Optional[int] = None,
    verbose: bool = True
) -> Optional[Path]:
    """
    Create a horizontal bar plot showing SongFormer sections within the snippet timerange.

    Similar to plot_sections_timeline but for SongFormer output which uses
    section labels (intro, verse, chorus, etc.) instead of Spotify's tempo/key info.

    Parameters
    ----------
    sections : list
        List of section dictionaries with 'start', 'duration', 'label' keys
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
            print("  No SongFormer sections overlap with snippet timerange")
        return None

    # Color map for section labels
    label_colors = {
        'intro': '#98FB98',      # pale green
        'verse': '#87CEEB',      # sky blue
        'chorus': '#FFB6C1',     # light pink
        'bridge': '#DDA0DD',     # plum
        'inst': '#F0E68C',       # khaki
        'outro': '#D3D3D3',      # light gray
        'silence': '#FFFFFF',    # white
        'pre-chorus': '#B0E0E6', # powder blue
    }
    default_color = '#E6E6FA'    # lavender for unknown labels

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot each section as a horizontal bar
    y_pos = 0.5
    bar_height = 0.6

    for i, section in enumerate(overlapping_sections):
        sec_start = section["start"]
        sec_end = sec_start + section["duration"]
        label = section.get("label", "unknown")

        # Clip section to snippet bounds for display
        display_start = max(sec_start, snippet_start)
        display_end = min(sec_end, snippet_end)
        display_width = display_end - display_start

        color = label_colors.get(label, default_color)

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

        # Add section label (if bar is wide enough)
        if display_width > snippet_duration * 0.05:
            label_x = display_start + display_width / 2
            ax.text(label_x, y_pos, label.upper(), ha='center', va='center',
                   fontsize=9, fontweight='bold')

    # Set axis limits to snippet bounds only
    ax.set_xlim(snippet_start, snippet_end)
    ax.set_ylim(0, 1)

    # Labels and formatting - primary x-axis shows absolute time
    ax.set_xlabel('Absolute Time (seconds)', fontsize=11)
    ax.set_yticks([])

    # Add secondary x-axis for relative time
    ax2 = ax.twiny()
    ax2.set_xlim(0, snippet_duration)
    ax2.set_xlabel('Relative Time (seconds)', fontsize=11)

    # Build title with optional song_id
    if song_id is not None and track_name:
        title = f'SongFormer Snippet Sections - {song_id}: {track_name}'
    elif song_id is not None:
        title = f'SongFormer Snippet Sections - {song_id}'
    elif track_name:
        title = f'SongFormer Snippet Sections - {track_name}'
    else:
        title = 'SongFormer Snippet Sections'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=25)

    # Add legend for section types (only those present in data)
    present_labels = set(s.get("label", "unknown") for s in overlapping_sections)
    legend_patches = [mpatches.Patch(color=label_colors.get(l, default_color), label=l.capitalize())
                      for l in present_labels if l in label_colors]
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)

    # Add grid for time reference
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {output_path.name}")

    return output_path


def plot_songformer_song_sections(
    sections: List[Dict[str, Any]],
    song_duration: float,
    snippet_start: float,
    snippet_duration: float,
    output_path: Path,
    track_name: str = "",
    song_id: Optional[int] = None,
    downbeats_file: Optional[Path] = None,
    verbose: bool = True
) -> Optional[Path]:
    """
    Create a horizontal bar plot showing ALL SongFormer sections for the full song,
    with snippet boundaries marked.

    Parameters
    ----------
    sections : list
        List of section dictionaries with 'start', 'duration', 'label' keys
    song_duration : float
        Total duration of the song in seconds
    snippet_start : float
        Start time of the snippet in seconds (for marking boundaries)
    snippet_duration : float
        Duration of the snippet in seconds (for marking boundaries)
    output_path : Path
        Path to save the plot
    track_name : str
        Track name for the title
    song_id : int, optional
        Song ID to include in the title
    downbeats_file : Path, optional
        Path to corrected downbeats file for adding bar markers
    verbose : bool
        Print progress messages

    Returns
    -------
    Path or None
        Path to saved plot, or None if no sections
    """
    if not sections:
        if verbose:
            print("  No SongFormer sections to plot")
        return None

    snippet_end = snippet_start + snippet_duration

    # Color map for section labels
    label_colors = {
        'intro': '#98FB98',      # pale green
        'verse': '#87CEEB',      # sky blue
        'chorus': '#FFB6C1',     # light pink
        'bridge': '#DDA0DD',     # plum
        'inst': '#F0E68C',       # khaki
        'outro': '#D3D3D3',      # light gray
        'silence': '#FFFFFF',    # white
        'pre-chorus': '#B0E0E6', # powder blue
    }
    default_color = '#E6E6FA'    # lavender for unknown labels

    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot each section as a horizontal bar
    y_pos = 0.5
    bar_height = 0.6

    for i, section in enumerate(sections):
        sec_start = section["start"]
        sec_duration = section["duration"]
        label = section.get("label", "unknown")

        color = label_colors.get(label, default_color)

        # Draw the section bar
        rect = mpatches.FancyBboxPatch(
            (sec_start, y_pos - bar_height/2),
            sec_duration, bar_height,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect)

        # Add section label (if bar is wide enough)
        if sec_duration > song_duration * 0.03:
            label_x = sec_start + sec_duration / 2
            ax.text(label_x, y_pos, label.upper(), ha='center', va='center',
                   fontsize=8, fontweight='bold')

    # Add snippet boundary lines
    ax.axvline(x=snippet_start, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(x=snippet_end, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # Add shaded region for snippet
    ax.axvspan(snippet_start, snippet_end, alpha=0.15, color='red')

    # Add downbeat markers if provided
    if downbeats_file and Path(downbeats_file).exists():
        import pandas as pd
        try:
            # Read downbeats file (tab-separated, skip comment lines)
            df = pd.read_csv(downbeats_file, sep='\t', comment='#')
            downbeat_times = df['corrected_downbeat_time(s)'].values
            bar_numbers = df['corrected_bar_num'].values

            # Draw tick marks below the section bars
            tick_y = y_pos - bar_height/2 - 0.05
            for bar_num, db_time in zip(bar_numbers, downbeat_times):
                if 0 <= db_time <= song_duration:
                    # Draw tick mark
                    ax.plot([db_time, db_time], [tick_y, tick_y - 0.08],
                            color='black', linewidth=0.8, alpha=0.7)
                    # Add bar number label (show every 4th bar to avoid clutter)
                    if bar_num % 4 == 1:
                        ax.text(db_time, tick_y - 0.12, str(int(bar_num)),
                                ha='center', va='top', fontsize=6, alpha=0.8)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not read downbeats file: {e}")

    # Set axis limits to full song
    ax.set_xlim(0, song_duration)
    ax.set_ylim(-0.3, 1)

    # Labels and formatting
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_yticks([])

    # Build title with optional song_id
    if song_id is not None and track_name:
        title = f'SongFormer Song Sections - {song_id}: {track_name}'
    elif song_id is not None:
        title = f'SongFormer Song Sections - {song_id}'
    elif track_name:
        title = f'SongFormer Song Sections - {track_name}'
    else:
        title = 'SongFormer Song Sections'
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add legend for section types (only those present in data) + snippet marker + bar markers
    present_labels = set(s.get("label", "unknown") for s in sections)
    legend_patches = [mpatches.Patch(color=label_colors.get(l, default_color), label=l.capitalize())
                      for l in present_labels if l in label_colors]
    # Add snippet boundary to legend
    legend_patches.append(mpatches.Patch(color='red', alpha=0.3, label=f'Snippet ({snippet_start:.1f}s - {snippet_end:.1f}s)'))
    # Add bar markers to legend if downbeats were added
    if downbeats_file and Path(downbeats_file).exists():
        from matplotlib.lines import Line2D
        legend_patches.append(Line2D([0], [0], color='black', linewidth=1, label='Bar downbeats'))
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)

    # Add grid for time reference
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"  Saved: {output_path.name}")

    return output_path


def create_songformer_plots(
    songformer_json_path: Path,
    output_dir: Path,
    track_id: str,
    snippet_start: float,
    snippet_duration: float,
    track_name: str = "",
    song_id: Optional[int] = None,
    downbeats_file: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Create SongFormer visualization plots and section change CSV.

    This function reads the SongFormer JSON output and creates:
    1. Snippet sections plot (zoomed to snippet timerange)
    2. Full song sections plot (with snippet boundaries marked)
    3. Section changes CSV (sections that start within snippet)

    Parameters
    ----------
    songformer_json_path : Path
        Path to the SongFormer JSON output file
    output_dir : Path
        Output directory for plots and CSV
    track_id : str
        Track identifier for output filenames
    snippet_start : float
        Start time of the snippet in seconds
    snippet_duration : float
        Duration of the snippet in seconds
    track_name : str, optional
        Track name for plot titles
    song_id : int, optional
        Song ID to include in plot titles
    downbeats_file : Path, optional
        Path to corrected downbeats file for adding bar markers
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Dictionary with paths to created files
    """
    results = {}

    if not songformer_json_path.exists():
        if verbose:
            print(f"  Skipping SongFormer plots (no data at {songformer_json_path})")
        return results

    if verbose:
        print(f"  Creating SongFormer sections plots...")

    with open(songformer_json_path, 'r') as f:
        sf_data = json.load(f)
        sf_sections = sf_data.get("sections", [])
        song_duration = sf_data.get("duration", 0)

    if not sf_sections:
        if verbose:
            print("  No SongFormer sections found in JSON")
        return results

    output_dir = Path(output_dir)

    # 1. Snippet sections plot (zoomed to snippet timerange)
    sf_snippet_plot = output_dir / "SF_snippet_sections.png"
    sf_snippet_path = plot_songformer_snippet_sections(
        sections=sf_sections,
        snippet_start=snippet_start,
        snippet_duration=snippet_duration,
        output_path=sf_snippet_plot,
        track_name=track_name,
        song_id=song_id,
        verbose=verbose
    )
    if sf_snippet_path:
        results["output_songformer_snippet_plot"] = str(sf_snippet_path)

    # 2. Full song sections plot (with snippet boundaries marked)
    sf_song_plot = output_dir / "SF_song_sections.png"
    sf_song_path = plot_songformer_song_sections(
        sections=sf_sections,
        song_duration=song_duration,
        snippet_start=snippet_start,
        snippet_duration=snippet_duration,
        output_path=sf_song_plot,
        track_name=track_name,
        song_id=song_id,
        downbeats_file=downbeats_file,
        verbose=verbose
    )
    if sf_song_path:
        results["output_songformer_song_plot"] = str(sf_song_path)

    # 3. Create snippet-filtered section changes CSV (sections that START within snippet)
    snippet_end = snippet_start + snippet_duration
    sf_changes_csv = output_dir / "SF_section_changes.csv"
    with open(sf_changes_csv, 'w') as f:
        f.write("section_num,start_absolute_s,start_relative_s,duration_s,label\n")
        for i, section in enumerate(sf_sections):
            sec_start = section["start"]
            # Only include sections that START within the snippet (not before it)
            if sec_start >= snippet_start and sec_start < snippet_end:
                relative_start = sec_start - snippet_start
                f.write(f"{i},{sec_start:.3f},{relative_start:.3f},{section['duration']:.3f},{section['label']}\n")
    results["output_songformer_changes_csv"] = str(sf_changes_csv)
    if verbose:
        print(f"  Saved: {sf_changes_csv.name}")

    # 4. Create overlapping sections CSV (sections that OVERLAP with snippet, with ratios)
    # -------------------------------------------------------------------------
    # RATIO DEFINITIONS (note: different denominators!)
    # -------------------------------------------------------------------------
    # ratio_to_snippet:      section_duration / snippet_duration
    #                        "How big is this section relative to the snippet?"
    #                        Can be > 1 if section is longer than snippet
    #
    # ratio_in_snippet:      duration_inside_snippet / snippet_duration
    #                        "What fraction of the SNIPPET does this section cover?"
    #                        Always <= 1, answers "how much of snippet is this section"
    #
    # ratio_outside_snippet: duration_outside_snippet / section_duration
    #                        "What fraction of the SECTION lies outside the snippet?"
    #                        0 = section fully inside, > 0 = part of section is cut off
    #                        Different denominator! Measures section completeness.
    # -------------------------------------------------------------------------
    sf_overlapping_csv = output_dir / "SF_overlapping_sections.csv"
    with open(sf_overlapping_csv, 'w') as f:
        f.write("section_num,start_absolute_s,start_relative_s,duration_s,label,ratio_to_snippet,ratio_in_snippet,ratio_outside_snippet\n")
        for i, section in enumerate(sf_sections):
            sec_start = section["start"]
            sec_end = sec_start + section["duration"]
            # Check if section overlaps with snippet
            if sec_start < snippet_end and sec_end > snippet_start:
                relative_start = sec_start - snippet_start

                # ratio_to_snippet: section_duration / snippet_duration
                ratio_to_snippet = section["duration"] / snippet_duration

                # Clip section to snippet bounds
                clipped_start = max(sec_start, snippet_start)
                clipped_end = min(sec_end, snippet_end)
                duration_in_snippet = clipped_end - clipped_start

                # ratio_in_snippet: duration_inside / snippet_duration
                ratio_in_snippet = duration_in_snippet / snippet_duration

                # ratio_outside_snippet: duration_outside / section_duration (different denom!)
                duration_outside_snippet = section["duration"] - duration_in_snippet
                ratio_outside_snippet = duration_outside_snippet / section["duration"]

                f.write(f"{i},{sec_start:.3f},{relative_start:.3f},{section['duration']:.3f},{section['label']},{ratio_to_snippet:.4f},{ratio_in_snippet:.4f},{ratio_outside_snippet:.4f}\n")
    results["output_songformer_overlapping_csv"] = str(sf_overlapping_csv)
    if verbose:
        print(f"  Saved: {sf_overlapping_csv.name}")

    # 5. Create snippet timings CSV (simple 1-column file with snippet start/end on separate rows)
    sf_timings_csv = output_dir / "SF_snippet_timings.csv"
    with open(sf_timings_csv, 'w') as f:
        f.write("time\n")
        f.write(f"{snippet_start:.3f}\n")
        f.write(f"{snippet_start + snippet_duration:.3f}\n")
    results["output_songformer_timings_csv"] = str(sf_timings_csv)
    if verbose:
        print(f"  Saved: {sf_timings_csv.name}")

    return results


def rule_post_processing(msa_list: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """Apply rule-based post-processing to clean up short segments"""
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    # Remove very short first segments
    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    # Remove very short last segments
    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    # Merge consecutive same labels at start
    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    # Merge consecutive same labels at end
    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def main():
    """Command-line interface for SongFormer analysis."""
    parser = argparse.ArgumentParser(
        description="Step 2.5: SongFormer Music Structure Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python songformer_analysis.py song.mp3 -o output/
  python songformer_analysis.py audio.wav -o output/ --track-id "17_Panini"
"""
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (MP3, WAV, etc.)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--track-id",
        type=str,
        default=None,
        help="Track identifier for output filenames"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    results = run_songformer(
        audio_path=audio_path,
        output_dir=Path(args.output),
        track_id=args.track_id,
        verbose=not args.quiet
    )

    if results["errors"]:
        print(f"Errors: {results['errors']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
