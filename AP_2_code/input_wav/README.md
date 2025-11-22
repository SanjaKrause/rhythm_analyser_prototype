# Input Audio Files

Place your audio files (`.wav`, `.mp3`, `.flac`, etc.) in this directory for processing.

## Usage

```bash
# Process a single track
python ../main.py --audio input_wav/track.wav --track-id track_123 --output-dir ../output_analysis/
```

## Supported Formats

The pipeline supports any audio format that librosa can read:
- WAV (recommended for best quality)
- MP3
- FLAC
- OGG
- M4A
- etc.

## File Naming

- Use descriptive filenames
- The `--track-id` parameter will be used for the output directory name
- Example: `26_Down - Marian Hill.wav` → track-id: `26`
