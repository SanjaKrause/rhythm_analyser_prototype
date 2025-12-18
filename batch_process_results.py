#!/usr/bin/env python3
"""
Batch Processing Results Aggregator

This script processes multiple track outputs and creates comprehensive PDF reports
combining tempo plots, raster plots, and analysis summaries.

Usage:
    python batch_process_results.py --input OUTPUT_EXAMPLES --output batch_processing_results
    python batch_process_results.py --input /path/to/output --output /path/to/results

Requirements:
    pip install PyPDF2 Pillow reportlab pandas matplotlib
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
import sys

try:
    from PyPDF2 import PdfMerger
    from PIL import Image
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install PyPDF2 Pillow reportlab pandas matplotlib")
    sys.exit(1)


def find_track_directories(input_dir: Path) -> List[Path]:
    """
    Find all track directories containing pipeline_results.json.

    Parameters
    ----------
    input_dir : Path
        Root directory to search

    Returns
    -------
    List[Path]
        List of track directories
    """
    track_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir():
            results_file = item / 'pipeline_results.json'
            if results_file.exists():
                track_dirs.append(item)

    return sorted(track_dirs)


def load_track_data(track_dir: Path) -> Optional[Dict]:
    """
    Load pipeline results JSON for a track.

    Parameters
    ----------
    track_dir : Path
        Track directory

    Returns
    -------
    Dict or None
        Track data dictionary, or None if loading failed
    """
    results_file = track_dir / 'pipeline_results.json'
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Failed to load {results_file}: {e}")
        return None


def create_cover_page(output_pdf: Path, tracks: List[Dict], title: str = "Loop Extractor Batch Analysis"):
    """
    Create a cover page with batch summary.

    Parameters
    ----------
    output_pdf : Path
        Output PDF path
    tracks : List[Dict]
        List of track data dictionaries
    title : str
        Report title
    """
    c = canvas.Canvas(str(output_pdf), pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 24)
    c.drawCentredString(width / 2, height - 1.5 * inch, title)

    # Summary stats
    c.setFont("Helvetica", 12)
    y = height - 2.5 * inch

    c.drawString(inch, y, f"Total Tracks: {len(tracks)}")
    y -= 0.3 * inch

    successful = sum(1 for t in tracks if not t.get('errors', []))
    c.drawString(inch, y, f"Successful: {successful}")
    y -= 0.3 * inch

    failed = len(tracks) - successful
    c.drawString(inch, y, f"Failed: {failed}")
    y -= 0.5 * inch

    # Track list
    c.setFont("Helvetica-Bold", 14)
    c.drawString(inch, y, "Tracks:")
    y -= 0.3 * inch

    c.setFont("Helvetica", 10)
    for i, track in enumerate(tracks, 1):
        if y < inch:
            c.showPage()
            y = height - inch

        track_id = track.get('track_id', 'Unknown')
        status = "✓" if not track.get('errors', []) else "✗"
        c.drawString(inch, y, f"{i}. {status} {track_id}")
        y -= 0.25 * inch

    c.save()
    print(f"  ✓ Created cover page: {output_pdf}")


def png_to_pdf(png_path: Path, pdf_path: Path):
    """
    Convert PNG to PDF page.

    Parameters
    ----------
    png_path : Path
        Input PNG path
    pdf_path : Path
        Output PDF path
    """
    img = Image.open(png_path)

    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Create PDF
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    # Calculate scaling to fit page
    img_width, img_height = img.size
    scale = min(width / img_width, height / img_height) * 0.9

    scaled_width = img_width * scale
    scaled_height = img_height * scale

    # Center image
    x = (width - scaled_width) / 2
    y = (height - scaled_height) / 2

    # Draw image
    c.drawImage(ImageReader(img), x, y, scaled_width, scaled_height)
    c.save()


def create_track_summary_page(pdf_path: Path, track_data: Dict):
    """
    Create a summary page for a track with metadata.

    Parameters
    ----------
    pdf_path : Path
        Output PDF path
    track_data : Dict
        Track data dictionary
    """
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Track title
    track_id = track_data.get('track_id', 'Unknown')
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - inch, track_id)

    y = height - 1.8 * inch
    c.setFont("Helvetica", 11)

    # Correction stats
    if 'correction_stats' in track_data:
        stats = track_data['correction_stats']
        c.setFont("Helvetica-Bold", 13)
        c.drawString(inch, y, "Downbeat Correction:")
        y -= 0.3 * inch

        c.setFont("Helvetica", 11)
        c.drawString(inch + 0.3 * inch, y, f"Time Signature: {stats.get('time_signature', 'N/A')}")
        y -= 0.25 * inch
        c.drawString(inch + 0.3 * inch, y, f"Bars: {stats.get('raw_bars', 'N/A')} → {stats.get('corrected_bars', 'N/A')}")
        y -= 0.25 * inch
        c.drawString(inch + 0.3 * inch, y, f"Dominant Pattern: {stats.get('dominant_pattern', 'N/A')}")
        y -= 0.25 * inch
        c.drawString(inch + 0.3 * inch, y, f"Usable Bars: {stats.get('usable_count', 'N/A')}")
        y -= 0.25 * inch
        c.drawString(inch + 0.3 * inch, y, f"Avg Tempo: {stats.get('base_avg_corrected', 0):.2f} BPM")
        y -= 0.4 * inch

    # Pattern lengths
    if 'pattern_lengths' in track_data:
        patterns = track_data['pattern_lengths']
        c.setFont("Helvetica-Bold", 13)
        c.drawString(inch, y, "Pattern Detection:")
        y -= 0.3 * inch

        c.setFont("Helvetica", 11)
        c.drawString(inch + 0.3 * inch, y, f"Drum Method: L = {patterns.get('drum', 'N/A')} bars")
        y -= 0.25 * inch
        c.drawString(inch + 0.3 * inch, y, f"Mel Method: L = {patterns.get('mel', 'N/A')} bars")
        y -= 0.25 * inch
        c.drawString(inch + 0.3 * inch, y, f"Pitch Method: L = {patterns.get('pitch', 'N/A')} bars")
        y -= 0.4 * inch

    # Onset count
    if 'num_onsets' in track_data:
        c.setFont("Helvetica-Bold", 13)
        c.drawString(inch, y, "Onset Detection:")
        y -= 0.3 * inch

        c.setFont("Helvetica", 11)
        c.drawString(inch + 0.3 * inch, y, f"Total Onsets: {track_data['num_onsets']}")
        y -= 0.4 * inch

    # Steps completed
    steps = track_data.get('steps_completed', [])
    c.setFont("Helvetica-Bold", 13)
    c.drawString(inch, y, f"Pipeline Steps: {len(steps)} completed")
    y -= 0.3 * inch

    # Errors
    errors = track_data.get('errors', [])
    if errors:
        y -= 0.3 * inch
        c.setFont("Helvetica-Bold", 13)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawString(inch, y, f"Errors: {len(errors)}")
        y -= 0.3 * inch

        c.setFont("Helvetica", 10)
        for error in errors[:5]:  # Show max 5 errors
            if y < inch:
                break
            c.drawString(inch + 0.3 * inch, y, f"• {error[:80]}")
            y -= 0.2 * inch

    c.save()


def merge_pdfs(pdf_list: List[Path], output_pdf: Path):
    """
    Merge multiple PDFs into one.

    Parameters
    ----------
    pdf_list : List[Path]
        List of PDF paths to merge
    output_pdf : Path
        Output merged PDF path
    """
    merger = PdfMerger()

    for pdf_path in pdf_list:
        if pdf_path.exists():
            try:
                merger.append(str(pdf_path))
            except Exception as e:
                print(f"  ⚠️  Failed to add {pdf_path.name}: {e}")

    merger.write(str(output_pdf))
    merger.close()


def create_rms_comparison_plot(tracks: List[Dict], output_pdf: Path):
    """
    Create a comparison plot of RMS values across tracks.

    Parameters
    ----------
    tracks : List[Dict]
        List of track data
    output_pdf : Path
        Output PDF path
    """
    # Extract RMS data
    track_ids = []
    rms_uncorrected = []
    rms_per_snippet = []
    rms_drum = []

    for track in tracks:
        if 'rms_values' in track:
            rms = track['rms_values']
            track_ids.append(track.get('track_id', 'Unknown'))
            rms_uncorrected.append(rms.get('uncorrected_ms', 0))
            rms_per_snippet.append(rms.get('per_snippet_ms', 0))
            rms_drum.append(rms.get('drum_ms', 0))

    if not track_ids:
        print("  ⚠️  No RMS data found, skipping comparison plot")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(track_ids))
    width = 0.25

    ax.bar([i - width for i in x], rms_uncorrected, width, label='Uncorrected', alpha=0.8)
    ax.bar(x, rms_per_snippet, width, label='Per-Snippet', alpha=0.8)
    ax.bar([i + width for i in x], rms_drum, width, label='Drum Method', alpha=0.8)

    ax.set_xlabel('Track')
    ax.set_ylabel('RMS (ms)')
    ax.set_title('RMS Comparison Across Tracks')
    ax.set_xticks(x)
    ax.set_xticklabels(track_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_pdf, dpi=150)
    plt.close()

    print(f"  ✓ Created RMS comparison plot: {output_pdf}")


def batch_process(input_dir: Path, output_dir: Path, verbose: bool = True):
    """
    Process all tracks and create comprehensive PDF reports.

    Parameters
    ----------
    input_dir : Path
        Input directory containing track folders
    output_dir : Path
        Output directory for batch results
    verbose : bool
        Print progress messages
    """
    if verbose:
        print("=" * 80)
        print("Batch Processing Results Aggregator")
        print("=" * 80)
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all track directories
    track_dirs = find_track_directories(input_dir)

    if not track_dirs:
        print("ERROR: No track directories found with pipeline_results.json")
        return

    if verbose:
        print(f"Found {len(track_dirs)} tracks")
        print()

    # Load all track data
    tracks_data = []
    for track_dir in track_dirs:
        if verbose:
            print(f"Loading: {track_dir.name}")

        data = load_track_data(track_dir)
        if data:
            data['_track_dir'] = track_dir
            tracks_data.append(data)

    if verbose:
        print()
        print(f"Loaded {len(tracks_data)} tracks successfully")
        print()

    # Create temporary directory for intermediate PDFs
    temp_dir = output_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)

    # Create cover page
    cover_pdf = temp_dir / '00_cover.pdf'
    create_cover_page(cover_pdf, tracks_data)

    # Process each track
    all_pdfs = [cover_pdf]

    for i, track_data in enumerate(tracks_data, 1):
        track_id = track_data.get('track_id', f'track_{i}')
        track_dir = track_data['_track_dir']

        if verbose:
            print(f"[{i}/{len(tracks_data)}] Processing: {track_id}")

        track_pdfs = []

        # 1. Create summary page
        summary_pdf = temp_dir / f'{i:02d}_{track_id}_summary.pdf'
        create_track_summary_page(summary_pdf, track_data)
        track_pdfs.append(summary_pdf)

        # 2. Add tempo plots PDF if exists
        tempo_pdf = track_dir / f'{track_id}_tempo_plots.pdf'
        if tempo_pdf.exists():
            track_pdfs.append(tempo_pdf)
            if verbose:
                print(f"  ✓ Found tempo plots")

        # 3. Convert raster plots PNG to PDF
        raster_png = track_dir / f'{track_id}_raster_comparison.png'
        if raster_png.exists():
            raster_pdf = temp_dir / f'{i:02d}_{track_id}_raster.pdf'
            png_to_pdf(raster_png, raster_pdf)
            track_pdfs.append(raster_pdf)
            if verbose:
                print(f"  ✓ Converted raster plots")

        all_pdfs.extend(track_pdfs)

    # Create RMS comparison plot
    if verbose:
        print()
        print("Creating comparison plots...")

    rms_plot_pdf = temp_dir / 'zz_rms_comparison.pdf'
    create_rms_comparison_plot(tracks_data, rms_plot_pdf)
    if rms_plot_pdf.exists():
        all_pdfs.append(rms_plot_pdf)

    # Merge all PDFs
    if verbose:
        print()
        print("Merging all PDFs...")

    final_pdf = output_dir / 'batch_analysis_report.pdf'
    merge_pdfs(all_pdfs, final_pdf)

    if verbose:
        print(f"  ✓ Created final report: {final_pdf}")
        print()

    # Create summary CSV
    summary_csv = output_dir / 'batch_summary.csv'
    create_summary_csv(tracks_data, summary_csv)

    if verbose:
        print(f"  ✓ Created summary CSV: {summary_csv}")
        print()

    # Cleanup temp directory
    import shutil
    shutil.rmtree(temp_dir)

    if verbose:
        print("=" * 80)
        print("Batch processing complete!")
        print(f"  Output: {output_dir}")
        print(f"  Report: {final_pdf}")
        print(f"  Summary: {summary_csv}")
        print("=" * 80)


def create_summary_csv(tracks_data: List[Dict], output_csv: Path):
    """
    Create a CSV summary of all tracks.

    Parameters
    ----------
    tracks_data : List[Dict]
        List of track data
    output_csv : Path
        Output CSV path
    """
    rows = []

    for track in tracks_data:
        row = {
            'track_id': track.get('track_id', 'Unknown'),
            'success': len(track.get('errors', [])) == 0,
            'steps_completed': len(track.get('steps_completed', [])),
        }

        # Correction stats
        if 'correction_stats' in track:
            stats = track['correction_stats']
            row['time_signature'] = stats.get('time_signature')
            row['raw_bars'] = stats.get('raw_bars')
            row['corrected_bars'] = stats.get('corrected_bars')
            row['dominant_pattern'] = stats.get('dominant_pattern')
            row['avg_tempo_bpm'] = stats.get('base_avg_corrected')

        # Pattern lengths
        if 'pattern_lengths' in track:
            patterns = track['pattern_lengths']
            row['pattern_drum'] = patterns.get('drum')
            row['pattern_mel'] = patterns.get('mel')
            row['pattern_pitch'] = patterns.get('pitch')

        # RMS values
        if 'rms_values' in track:
            rms = track['rms_values']
            row['rms_uncorrected_ms'] = rms.get('uncorrected_ms')
            row['rms_per_snippet_ms'] = rms.get('per_snippet_ms')
            row['rms_drum_ms'] = rms.get('drum_ms')

        # Onset count
        row['num_onsets'] = track.get('num_onsets')

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Batch process Loop Extractor results and create comprehensive PDF reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python batch_process_results.py --input OUTPUT_EXAMPLES --output batch_processing_results
  python batch_process_results.py --input /path/to/output --output /path/to/results --quiet
"""
    )

    parser.add_argument('--input', required=True, help='Input directory containing track folders')
    parser.add_argument('--output', required=True, help='Output directory for batch results')
    parser.add_argument('--quiet', action='store_true', help='Minimize output')

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"ERROR: Input path is not a directory: {input_dir}")
        sys.exit(1)

    try:
        batch_process(input_dir, output_dir, verbose=not args.quiet)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
