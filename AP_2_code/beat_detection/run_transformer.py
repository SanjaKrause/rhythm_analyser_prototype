"""
Beat-Transformer standalone script for beat/downbeat detection.

This script runs in the new_beatnet_env environment and is called via subprocess
from the main pipeline running in AEinBOX_13_3.

Environment: new_beatnet_env
Dependencies: torch, madmom, numpy
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch

# Add Beat-Transformer code to path
BEAT_TRANSFORMER_CODE = Path("/Users/alexk/mastab/loop_extractor_python/Beat-Transformer/code")
sys.path.insert(0, str(BEAT_TRANSFORMER_CODE))

from DilatedTransformer import Demixed_DilatedTransformerModel
from madmom.features.beats import DBNBeatTrackingProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor


def load_model(checkpoint_path: str, device: str = 'cpu') -> Demixed_DilatedTransformerModel:
    """
    Load Beat-Transformer model from checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint (.pt file)
    device : str
        Device to load model on ('cpu' or 'cuda')

    Returns
    -------
    Demixed_DilatedTransformerModel
        Loaded model in evaluation mode
    """
    model = Demixed_DilatedTransformerModel(
        attn_len=5,
        instr=5,
        ntoken=2,
        dmodel=256,
        nhead=8,
        d_hid=1024,
        nlayers=9,
        norm_first=True
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            # Checkpoint has wrapper: {'state_dict': <weights>}
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            # Alternative wrapper format
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
    else:
        # Checkpoint is already state dict
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    return model


def predict_activations(model: Demixed_DilatedTransformerModel, spec5: np.ndarray, device: str = 'cpu') -> tuple:
    """
    Run Beat-Transformer forward pass to get beat/downbeat activations.

    Parameters
    ----------
    model : Demixed_DilatedTransformerModel
        Loaded Beat-Transformer model
    spec5 : np.ndarray
        5-stem mel-spectrogram, shape (5, time_frames, n_mels)
    device : str
        Device to run on

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        (beat_activations, downbeat_activations), each shape (time_frames,)
    """
    # Convert to tensor and add batch dimension
    inp = torch.from_numpy(spec5).unsqueeze(0).float().to(device)

    # Forward pass
    with torch.no_grad():
        activation, _ = model(inp)

    # Remove batch dimension and convert to numpy
    activation = activation[0].cpu().numpy()  # shape: (time_frames, 2)

    # Apply sigmoid to convert logits to probabilities
    beat_act = 1.0 / (1.0 + np.exp(-activation[:, 0]))
    downbeat_act = 1.0 / (1.0 + np.exp(-activation[:, 1]))

    return beat_act, downbeat_act


def track_beats(
    beat_act: np.ndarray,
    fps: float = 43.066,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    transition_lambda: int = 100,
    observation_lambda: int = 6,
    threshold: float = 0.2
) -> np.ndarray:
    """
    Convert beat activations to beat timestamps using madmom DBN.

    Parameters
    ----------
    beat_act : np.ndarray
        Beat activation function
    fps : float
        Frames per second
    min_bpm : float
        Minimum BPM
    max_bpm : float
        Maximum BPM
    transition_lambda : int
        DBN transition lambda
    observation_lambda : int
        DBN observation lambda
    threshold : float
        Detection threshold

    Returns
    -------
    np.ndarray
        Beat timestamps in seconds
    """
    tracker = DBNBeatTrackingProcessor(
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=fps,
        transition_lambda=transition_lambda,
        observation_lambda=observation_lambda,
        threshold=threshold
    )

    beat_times = tracker(beat_act)
    return beat_times


def track_downbeats(
    beat_act: np.ndarray,
    downbeat_act: np.ndarray,
    fps: float = 43.066,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    transition_lambda: int = 100,
    observation_lambda: int = 6,
    threshold: float = 0.2,
    beats_per_bar: list = [3, 4]
) -> np.ndarray:
    """
    Convert beat/downbeat activations to downbeat timestamps using madmom DBN.

    Parameters
    ----------
    beat_act : np.ndarray
        Beat activation function
    downbeat_act : np.ndarray
        Downbeat activation function
    fps : float
        Frames per second
    min_bpm : float
        Minimum BPM
    max_bpm : float
        Maximum BPM
    transition_lambda : int
        DBN transition lambda
    observation_lambda : int
        DBN observation lambda
    threshold : float
        Detection threshold
    beats_per_bar : list
        Supported time signatures (e.g., [3, 4] for 3/4 and 4/4)

    Returns
    -------
    np.ndarray
        Downbeat events, shape (n_downbeats, 2) with columns [time, beat_position]
    """
    tracker = DBNDownBeatTrackingProcessor(
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=fps,
        beats_per_bar=beats_per_bar,
        transition_lambda=transition_lambda,
        observation_lambda=observation_lambda,
        threshold=threshold
    )

    # Create combined activation (beat-only and downbeat)
    combined_act = np.column_stack([
        np.maximum(beat_act - downbeat_act, 0),
        downbeat_act
    ])

    downbeat_events = tracker(combined_act)
    return downbeat_events


def assign_beats_to_bars(beat_times: np.ndarray, downbeat_times: np.ndarray) -> list:
    """
    Assign each beat to a bar and calculate beat positions.

    Parameters
    ----------
    beat_times : np.ndarray
        Array of beat timestamps
    downbeat_times : np.ndarray
        Array of downbeat timestamps

    Returns
    -------
    list of dict
        List of beat dictionaries with keys:
        - beat_time: timestamp of beat
        - downbeat_time: timestamp of bar's downbeat
        - beat_pos: position within bar (1, 2, 3, ...)
        - global_beat: sequential beat number
        - bar_num: bar number (0 for pre-first-downbeat, 1+ for regular bars)
    """
    beats_data = []

    for global_beat_idx, beat_time in enumerate(beat_times):
        # Find which bar this beat belongs to
        bar_idx = np.searchsorted(downbeat_times, beat_time, side='right') - 1

        if bar_idx < 0:
            # Beat before first downbeat
            bar_num = 0
            downbeat_time = 0.0
            beat_pos = 0
        else:
            bar_num = bar_idx + 1
            downbeat_time = downbeat_times[bar_idx]

            # Count beats in this bar up to current beat
            beats_in_bar = beat_times[(beat_times >= downbeat_time) & (beat_times <= beat_time)]
            beat_pos = len(beats_in_bar)

        beats_data.append({
            'beat_time': beat_time,
            'downbeat_time': downbeat_time,
            'beat_pos': beat_pos,
            'global_beat': global_beat_idx,
            'bar_num': bar_num
        })

    return beats_data


def save_beats_output(beats_data: list, output_path: str):
    """
    Save beat data to tab-separated text file.

    Parameters
    ----------
    beats_data : list of dict
        Beat data from assign_beats_to_bars()
    output_path : str
        Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header
        f.write("beat_time(s)\tdownbeat_time(s)\tbeat_pos\tglobal_beat\tbar_num\n")

        # Write data
        for beat in beats_data:
            f.write(f"{beat['beat_time']:.6f}\t")
            f.write(f"{beat['downbeat_time']:.6f}\t")
            f.write(f"{beat['beat_pos']}\t")
            f.write(f"{beat['global_beat']}\t")
            f.write(f"{beat['bar_num']}\n")

    print(f"Saved beat output to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Beat-Transformer beat/downbeat detection')
    parser.add_argument('--input', required=True, help='Input NPZ file with 5-stem mel-spectrograms')
    parser.add_argument('--output', required=True, help='Output text file for beat data')
    parser.add_argument('--checkpoint', required=True, help='Path to Beat-Transformer checkpoint')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--fps', type=float, default=43.066, help='Frames per second')
    parser.add_argument('--min-bpm', type=float, default=55.0, help='Minimum BPM')
    parser.add_argument('--max-bpm', type=float, default=215.0, help='Maximum BPM')

    args = parser.parse_args()

    print("=" * 80)
    print("Beat-Transformer Beat/Downbeat Detection")
    print("=" * 80)

    # Load input NPZ
    print(f"\nLoading input: {args.input}")
    data = np.load(args.input)
    spec5 = data['spec']
    print(f"Input shape: {spec5.shape}")

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)
    print("Model loaded successfully")

    # Predict activations
    print("\nRunning Beat-Transformer forward pass...")
    beat_act, downbeat_act = predict_activations(model, spec5, args.device)
    print(f"Activation shapes: beat={beat_act.shape}, downbeat={downbeat_act.shape}")

    # Track beats
    print("\nTracking beats with madmom DBN...")
    beat_times = track_beats(
        beat_act,
        fps=args.fps,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm
    )
    print(f"Detected {len(beat_times)} beats")

    # Track downbeats
    print("Tracking downbeats with madmom DBN...")
    downbeat_events = track_downbeats(
        beat_act,
        downbeat_act,
        fps=args.fps,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm
    )
    # Extract only downbeats (where beat_position == 1)
    downbeat_times = downbeat_events[downbeat_events[:, 1] == 1, 0]
    print(f"Detected {len(downbeat_times)} downbeats")

    # Assign beats to bars
    print("\nAssigning beats to bars...")
    beats_data = assign_beats_to_bars(beat_times, downbeat_times)

    # Save output
    print(f"\nSaving output...")
    save_beats_output(beats_data, args.output)

    print("\n" + "=" * 80)
    print("Completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
