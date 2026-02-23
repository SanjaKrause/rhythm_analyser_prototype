#!/usr/bin/env python3
"""
Simple script to merge all plot PDFs from batch processing into combined PDFs.

Merges:
- All tempo plots into all_tempo_plots.pdf
- All raster comparison plots into all_raster_comparison.pdf
- All raster standard plots into all_raster_standard.pdf
- All new grid corrections plots into all_new_grid_corrections.pdf
- All microtiming plots into all_microtiming_plots.pdf
- All rhythm histograms into all_rhythm_histograms.pdf
- All rhythm histograms with style into all_rhythm_histograms_with_style.pdf
- All rhythm histograms with medians and IQR into all_rhythm_histograms_with_medians_and_iqr.pdf
- All groove pulse histograms into all_groove_pulse_histograms.pdf
- All groove pulse click tracks into all_groove_pulse_click_tracks.pdf
- All rhythm patterns into all_rhythm_patterns.pdf
- All new 4-method raster plots into all_raster_4method.pdf
- All Spotify sections timelines into all_spotify_sections.pdf
- All SongFormer sections timelines into all_songformer_sections.pdf
- All onsets per 2-bar pattern plots into all_onsets_per_pattern_2bar.pdf
- All onsets per 4-bar pattern plots into all_onsets_per_pattern_4bar.pdf
- All onsets per bar (2-bar grid) plots into all_onsets_per_bar_2bar.pdf
- All onsets per bar (4-bar grid) plots into all_onsets_per_bar_4bar.pdf
- All section anchoring raster plots into all_section_anchoring_raster.pdf
- All filtered section anchoring raster plots into all_section_anchoring_raster_filtered.pdf
- All anchored rhythm histograms into all_anchored_rhythm_histograms.pdf
- All anchored groove pulse histograms into all_anchored_groove_pulse_histograms.pdf
- All anchored rhythm patterns into all_anchored_rhythm_patterns.pdf
- All anchored beat histograms into all_anchored_beat_histograms.pdf
- All anchored beat histograms (all onsets) into all_anchored_beat_histograms_all_onsets.pdf

Usage:
    python merge_plots.py /path/to/batch/output
"""

import sys
import shutil
from pathlib import Path
from PyPDF2 import PdfMerger
from PIL import Image
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


def png_to_pdf(png_path: Path, pdf_path: Path):
    """Convert PNG to PDF with exact image dimensions."""
    img = Image.open(png_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Use image dimensions as page size (convert pixels to points at 72 DPI)
    img_width, img_height = img.size
    # Assuming 150 DPI (as used in raster plot savefig), convert to points (72 points per inch)
    width_points = (img_width / 150.0) * 72
    height_points = (img_height / 150.0) * 72

    c = pdf_canvas.Canvas(str(pdf_path), pagesize=(width_points, height_points))
    c.drawImage(ImageReader(img), 0, 0, width_points, height_points)
    c.save()


def merge_plots(output_dir: Path):
    """
    Merge all plot PDFs from track folders into combined PDFs.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    """
    # Create batch_analysis folder
    batch_dir = output_dir / 'batch_analysis'
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Get all track directories (exclude batch_analysis and any other special folders)
    track_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis']])

    if not track_dirs:
        print('No track directories found!')
        return

    # 1. Merge tempo plots
    print('Looking for tempo plots...')
    tempo_pdfs = []
    for track_dir in track_dirs:
        tempo_pdf = track_dir / '3.5_tempo_plots' / f'{track_dir.name}_tempo_plots.pdf'
        if tempo_pdf.exists():
            tempo_pdfs.append(tempo_pdf)
            print(f'  Found tempo: {track_dir.name}')

    if tempo_pdfs:
        print(f'\nMerging {len(tempo_pdfs)} tempo PDFs...')
        output_pdf = batch_dir / 'all_tempo_plots.pdf'
        merger = PdfMerger()
        for pdf in tempo_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # Create single temp directory for all PNG conversions
    temp_dir = batch_dir / '_temp'
    temp_dir.mkdir(exist_ok=True)

    # 2. Merge raster comparison plots (PNG files in 5_grid folder)
    print('\nLooking for raster comparison plots...')
    raster_comparison_pngs = []
    for track_dir in track_dirs:
        raster_png = track_dir / '5_grid' / f'{track_dir.name}_raster_comparison.png'
        if raster_png.exists():
            raster_comparison_pngs.append(raster_png)
            print(f'  Found raster comparison: {track_dir.name}')

    if raster_comparison_pngs:
        print(f'\nConverting and merging {len(raster_comparison_pngs)} raster comparison PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(raster_comparison_pngs):
            temp_pdf = temp_dir / f'raster_comp_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_raster_comparison.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 3. Merge raster standard plots (PNG files in 5_grid folder)
    print('\nLooking for raster standard plots...')
    raster_standard_pngs = []
    for track_dir in track_dirs:
        raster_png = track_dir / '5_grid' / f'{track_dir.name}_raster_standard.png'
        if raster_png.exists():
            raster_standard_pngs.append(raster_png)
            print(f'  Found raster standard: {track_dir.name}')

    if raster_standard_pngs:
        print(f'\nConverting and merging {len(raster_standard_pngs)} raster standard PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(raster_standard_pngs):
            temp_pdf = temp_dir / f'raster_std_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_raster_standard.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 4. Merge new grid corrections plots (PNG files in 5_grid folder)
    print('\nLooking for new grid corrections plots...')
    new_grid_corrections_pngs = []
    for track_dir in track_dirs:
        new_grid_png = track_dir / '5_grid' / f'{track_dir.name}_new_grid_corrections.png'
        if new_grid_png.exists():
            new_grid_corrections_pngs.append(new_grid_png)
            print(f'  Found new grid corrections: {track_dir.name}')

    if new_grid_corrections_pngs:
        print(f'\nConverting and merging {len(new_grid_corrections_pngs)} new grid corrections PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(new_grid_corrections_pngs):
            temp_pdf = temp_dir / f'new_grid_corr_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_new_grid_corrections.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 5. Merge microtiming plots (PDF files in 5_grid folder)
    print('\nLooking for microtiming plots...')
    microtiming_pdfs = []
    for track_dir in track_dirs:
        microtiming_pdf = track_dir / '5_grid' / f'{track_dir.name}_microtiming_plots.pdf'
        if microtiming_pdf.exists():
            microtiming_pdfs.append(microtiming_pdf)
            print(f'  Found microtiming: {track_dir.name}')

    if microtiming_pdfs:
        print(f'\nMerging {len(microtiming_pdfs)} microtiming PDFs...')
        output_pdf = batch_dir / 'all_microtiming_plots.pdf'
        merger = PdfMerger()
        for pdf in microtiming_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 6. Merge rhythm histograms (PDF files in 5.5_rhythm folder)
    print('\nLooking for rhythm histograms...')
    rhythm_pdfs = []
    for track_dir in track_dirs:
        rhythm_pdf = track_dir / '5.5_rhythm' / f'{track_dir.name}_rhythm_histograms.pdf'
        if rhythm_pdf.exists():
            rhythm_pdfs.append(rhythm_pdf)
            print(f'  Found rhythm histogram: {track_dir.name}')

    if rhythm_pdfs:
        print(f'\nMerging {len(rhythm_pdfs)} rhythm histogram PDFs...')
        output_pdf = batch_dir / 'all_rhythm_histograms.pdf'
        merger = PdfMerger()
        for pdf in rhythm_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 7. Merge rhythm histograms with style (PDF files in 5.5_rhythm folder)
    print('\nLooking for rhythm histograms with style...')
    rhythm_style_pdfs = []
    for track_dir in track_dirs:
        rhythm_style_pdf = track_dir / '5.5_rhythm' / f'{track_dir.name}_rhythm_histograms_with_style.pdf'
        if rhythm_style_pdf.exists():
            rhythm_style_pdfs.append(rhythm_style_pdf)
            print(f'  Found rhythm histogram with style: {track_dir.name}')

    if rhythm_style_pdfs:
        print(f'\nMerging {len(rhythm_style_pdfs)} rhythm histogram with style PDFs...')
        output_pdf = batch_dir / 'all_rhythm_histograms_with_style.pdf'
        merger = PdfMerger()
        for pdf in rhythm_style_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 8. Merge rhythm histograms with medians and IQR (PDF files in 5.5_rhythm folder)
    print('\nLooking for rhythm histograms with medians and IQR...')
    rhythm_medians_iqr_pdfs = []
    for track_dir in track_dirs:
        rhythm_medians_iqr_pdf = track_dir / '5.5_rhythm' / f'{track_dir.name}_rhythm_histograms_with_medians_and_iqr.pdf'
        if rhythm_medians_iqr_pdf.exists():
            rhythm_medians_iqr_pdfs.append(rhythm_medians_iqr_pdf)
            print(f'  Found rhythm histogram with medians and IQR: {track_dir.name}')

    if rhythm_medians_iqr_pdfs:
        print(f'\nMerging {len(rhythm_medians_iqr_pdfs)} rhythm histogram with medians and IQR PDFs...')
        output_pdf = batch_dir / 'all_rhythm_histograms_with_medians_and_iqr.pdf'
        merger = PdfMerger()
        for pdf in rhythm_medians_iqr_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 9. Merge groove pulse histograms (PDF files in 5.5_rhythm folder)
    print('\nLooking for groove pulse histograms...')
    groove_pulse_pdfs = []
    for track_dir in track_dirs:
        groove_pulse_pdf = track_dir / '5.5_rhythm' / f'{track_dir.name}_groove_pulse_histograms.pdf'
        if groove_pulse_pdf.exists():
            groove_pulse_pdfs.append(groove_pulse_pdf)
            print(f'  Found groove pulse histogram: {track_dir.name}')

    if groove_pulse_pdfs:
        print(f'\nMerging {len(groove_pulse_pdfs)} groove pulse histogram PDFs...')
        output_pdf = batch_dir / 'all_groove_pulse_histograms.pdf'
        merger = PdfMerger()
        for pdf in groove_pulse_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 10. Merge groove pulse click track plots (PNG files in 7_audio_examples or 5_grid folder)
    print('\nLooking for groove pulse click track plots...')
    groove_click_pngs = []
    for track_dir in track_dirs:
        # Check 7_audio_examples first (newer location), then 5_grid (legacy)
        groove_click_png = track_dir / '7_audio_examples' / 'groove_pulse_click_times.png'
        if groove_click_png.exists():
            groove_click_pngs.append(groove_click_png)
            print(f'  Found groove pulse click track: {track_dir.name}')
        else:
            groove_click_png = track_dir / '5_grid' / 'groove_pulse_click_times.png'
            if groove_click_png.exists():
                groove_click_pngs.append(groove_click_png)
                print(f'  Found groove pulse click track (legacy): {track_dir.name}')

    if groove_click_pngs:
        print(f'\nConverting and merging {len(groove_click_pngs)} groove pulse click track PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(groove_click_pngs):
            temp_pdf = temp_dir / f'groove_click_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_groove_pulse_click_tracks.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 11. Merge rhythm pattern plots (PNG files in 5.5_rhythm folder)
    print('\nLooking for rhythm pattern plots...')
    rhythm_pattern_pngs = []
    for track_dir in track_dirs:
        rhythm_pattern_png = track_dir / '5.5_rhythm' / f'{track_dir.name}_rhythm_patterns.png'
        if rhythm_pattern_png.exists():
            rhythm_pattern_pngs.append(rhythm_pattern_png)
            print(f'  Found rhythm pattern: {track_dir.name}')

    if rhythm_pattern_pngs:
        print(f'\nConverting and merging {len(rhythm_pattern_pngs)} rhythm pattern PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(rhythm_pattern_pngs):
            temp_pdf = temp_dir / f'rhythm_pattern_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_rhythm_patterns.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 12. Merge beat histogram plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for beat histogram plots...')
    beat_histogram_pdfs = []
    for track_dir in track_dirs:
        beat_histogram_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_beat_histograms.pdf'
        if beat_histogram_pdf.exists():
            beat_histogram_pdfs.append(beat_histogram_pdf)
            print(f'  Found beat histogram: {track_dir.name}')

    if beat_histogram_pdfs:
        print(f'\nMerging {len(beat_histogram_pdfs)} beat histogram PDFs...')
        output_pdf = batch_dir / 'all_beat_histograms.pdf'
        merger = PdfMerger()
        for pdf in beat_histogram_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 13. Merge beat histogram all onsets plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for beat histogram all onsets plots...')
    beat_histogram_all_onsets_pdfs = []
    for track_dir in track_dirs:
        beat_histogram_all_onsets_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_beat_histograms_all_onsets.pdf'
        if beat_histogram_all_onsets_pdf.exists():
            beat_histogram_all_onsets_pdfs.append(beat_histogram_all_onsets_pdf)
            print(f'  Found beat histogram all onsets: {track_dir.name}')

    if beat_histogram_all_onsets_pdfs:
        print(f'\nMerging {len(beat_histogram_all_onsets_pdfs)} beat histogram all onsets PDFs...')
        output_pdf = batch_dir / 'all_beat_histograms_all_onsets.pdf'
        merger = PdfMerger()
        for pdf in beat_histogram_all_onsets_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 14. Merge new 4-method raster plots (PNG files in 5_grid folder)
    print('\nLooking for new 4-method raster plots...')
    raster_4method_pngs = []
    for track_dir in track_dirs:
        raster_png = track_dir / '5_grid' / f'{track_dir.name}_raster.png'
        if raster_png.exists():
            raster_4method_pngs.append(raster_png)
            print(f'  Found 4-method raster: {track_dir.name}')

    if raster_4method_pngs:
        print(f'\nConverting and merging {len(raster_4method_pngs)} 4-method raster PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(raster_4method_pngs):
            temp_pdf = temp_dir / f'raster_4method_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_raster_4method.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 15. Merge simple IOI histogram plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for simple IOI histogram plots...')
    simple_ioi_pdfs = []
    for track_dir in track_dirs:
        simple_ioi_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_simple_ioi_histogram.pdf'
        if simple_ioi_pdf.exists():
            simple_ioi_pdfs.append(simple_ioi_pdf)
            print(f'  Found simple IOI histogram: {track_dir.name}')

    if simple_ioi_pdfs:
        print(f'\nMerging {len(simple_ioi_pdfs)} simple IOI histogram PDFs...')
        output_pdf = batch_dir / 'all_simple_ioi_histograms.pdf'
        merger = PdfMerger()
        for pdf in simple_ioi_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 15b. Merge simple IOI all crosses plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for simple IOI all crosses plots...')
    simple_ioi_crosses_pdfs = []
    for track_dir in track_dirs:
        simple_ioi_crosses_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_simple_ioi_all_crosses.pdf'
        if simple_ioi_crosses_pdf.exists():
            simple_ioi_crosses_pdfs.append(simple_ioi_crosses_pdf)
            print(f'  Found simple IOI all crosses: {track_dir.name}')

    if simple_ioi_crosses_pdfs:
        print(f'\nMerging {len(simple_ioi_crosses_pdfs)} simple IOI all crosses PDFs...')
        output_pdf = batch_dir / 'all_simple_ioi_all_crosses.pdf'
        merger = PdfMerger()
        for pdf in simple_ioi_crosses_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 15c. Merge snippet IOI all crosses plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for snippet IOI all crosses plots...')
    snippet_ioi_crosses_pdfs = []
    for track_dir in track_dirs:
        snippet_ioi_crosses_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_snippet_ioi_all_crosses.pdf'
        if snippet_ioi_crosses_pdf.exists():
            snippet_ioi_crosses_pdfs.append(snippet_ioi_crosses_pdf)
            print(f'  Found snippet IOI all crosses: {track_dir.name}')

    if snippet_ioi_crosses_pdfs:
        print(f'\nMerging {len(snippet_ioi_crosses_pdfs)} snippet IOI all crosses PDFs...')
        output_pdf = batch_dir / 'all_snippet_ioi_all_crosses.pdf'
        merger = PdfMerger()
        for pdf in snippet_ioi_crosses_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 15d. Merge snippet IOI histogram plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for snippet IOI histogram plots...')
    snippet_ioi_histogram_pdfs = []
    for track_dir in track_dirs:
        snippet_ioi_histogram_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_snippet_ioi_histogram.pdf'
        if snippet_ioi_histogram_pdf.exists():
            snippet_ioi_histogram_pdfs.append(snippet_ioi_histogram_pdf)
            print(f'  Found snippet IOI histogram: {track_dir.name}')

    if snippet_ioi_histogram_pdfs:
        print(f'\nMerging {len(snippet_ioi_histogram_pdfs)} snippet IOI histogram PDFs...')
        output_pdf = batch_dir / 'all_snippet_ioi_histograms.pdf'
        merger = PdfMerger()
        for pdf in snippet_ioi_histogram_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 16. Merge simple beat histogram plots (PDF files in 5.7_beat_histograms folder)
    print('\nLooking for simple beat histogram plots...')
    simple_beat_pdfs = []
    for track_dir in track_dirs:
        simple_beat_pdf = track_dir / '5.7_beat_histograms' / f'{track_dir.name}_simple_beat_histograms.pdf'
        if simple_beat_pdf.exists():
            simple_beat_pdfs.append(simple_beat_pdf)
            print(f'  Found simple beat histogram: {track_dir.name}')

    if simple_beat_pdfs:
        print(f'\nMerging {len(simple_beat_pdfs)} simple beat histogram PDFs...')
        output_pdf = batch_dir / 'all_simple_beat_histograms.pdf'
        merger = PdfMerger()
        for pdf in simple_beat_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 17. Merge full song IOI histogram plots (PDF files in 5.8_full_histograms folder)
    print('\nLooking for full song IOI histogram plots...')
    full_ioi_pdfs = []
    for track_dir in track_dirs:
        full_ioi_pdf = track_dir / '5.8_full_histograms' / f'{track_dir.name}_full_song_ioi_histogram.pdf'
        if full_ioi_pdf.exists():
            full_ioi_pdfs.append(full_ioi_pdf)
            print(f'  Found full song IOI histogram: {track_dir.name}')

    if full_ioi_pdfs:
        print(f'\nMerging {len(full_ioi_pdfs)} full song IOI histogram PDFs...')
        output_pdf = batch_dir / 'all_full_song_ioi_histograms.pdf'
        merger = PdfMerger()
        for pdf in full_ioi_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 18. Merge Spotify sections timeline plots (PNG files in 13_spotify folder)
    print('\nLooking for Spotify sections timeline plots...')
    spotify_sections_pngs = []
    for track_dir in track_dirs:
        spotify_sections_png = track_dir / '13_spotify' / f'{track_dir.name}_sections_timeline.png'
        if spotify_sections_png.exists():
            spotify_sections_pngs.append(spotify_sections_png)
            print(f'  Found Spotify sections timeline: {track_dir.name}')

    if spotify_sections_pngs:
        print(f'\nConverting and merging {len(spotify_sections_pngs)} Spotify sections timeline PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(spotify_sections_pngs):
            temp_pdf = temp_dir / f'spotify_sections_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_spotify_sections.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 19. Merge onsets per pattern plots (PNG files in 13_spotify folder)
    # Process both 2-bar and 4-bar patterns
    for pattern_len in [2, 4]:
        print(f'\nLooking for onsets per {pattern_len}-bar pattern plots...')
        onsets_per_pattern_pngs = []
        for track_dir in track_dirs:
            onsets_png = track_dir / '13_spotify' / f'{track_dir.name}_onsets_per_pattern_{pattern_len}bar.png'
            if onsets_png.exists():
                onsets_per_pattern_pngs.append(onsets_png)
                print(f'  Found onsets per {pattern_len}-bar pattern: {track_dir.name}')

        if onsets_per_pattern_pngs:
            print(f'\nConverting and merging {len(onsets_per_pattern_pngs)} onsets per {pattern_len}-bar pattern PNGs...')

            merger = PdfMerger()
            for i, png in enumerate(onsets_per_pattern_pngs):
                temp_pdf = temp_dir / f'onsets_per_pattern_{pattern_len}bar_{i}.pdf'
                png_to_pdf(png, temp_pdf)
                merger.append(str(temp_pdf))

            output_pdf = batch_dir / f'all_onsets_per_pattern_{pattern_len}bar.pdf'
            merger.write(str(output_pdf))
            merger.close()

            print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 20. Merge onsets per bar plots (PNG files in 13_spotify folder)
    # Process both 2-bar and 4-bar grids
    for pattern_len in [2, 4]:
        print(f'\nLooking for onsets per bar ({pattern_len}-bar grid) plots...')
        onsets_per_bar_pngs = []
        for track_dir in track_dirs:
            onsets_png = track_dir / '13_spotify' / f'{track_dir.name}_onsets_per_bar_{pattern_len}bar.png'
            if onsets_png.exists():
                onsets_per_bar_pngs.append(onsets_png)
                print(f'  Found onsets per bar ({pattern_len}-bar grid): {track_dir.name}')

        if onsets_per_bar_pngs:
            print(f'\nConverting and merging {len(onsets_per_bar_pngs)} onsets per bar ({pattern_len}-bar grid) PNGs...')

            merger = PdfMerger()
            for i, png in enumerate(onsets_per_bar_pngs):
                temp_pdf = temp_dir / f'onsets_per_bar_{pattern_len}bar_{i}.pdf'
                png_to_pdf(png, temp_pdf)
                merger.append(str(temp_pdf))

            output_pdf = batch_dir / f'all_onsets_per_bar_{pattern_len}bar.pdf'
            merger.write(str(output_pdf))
            merger.close()

            print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 21. Merge SongFormer snippet sections plots (PNG files in 2.5_songformer_sections folder)
    print('\nLooking for SongFormer snippet sections plots...')
    sf_sections_pngs = []
    for track_dir in track_dirs:
        sf_png = track_dir / '2.5_songformer_sections' / 'SF_snippet_sections.png'
        if sf_png.exists():
            sf_sections_pngs.append(sf_png)
            print(f'  Found SongFormer sections: {track_dir.name}')

    if sf_sections_pngs:
        print(f'\nConverting and merging {len(sf_sections_pngs)} SongFormer sections PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(sf_sections_pngs):
            temp_pdf = temp_dir / f'sf_sections_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_songformer_sections.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 22. Merge SongFormer song sections plots (PNG files in 2.5_songformer_sections folder)
    print('\nLooking for SongFormer song sections plots...')
    sf_song_pngs = []
    for track_dir in track_dirs:
        sf_png = track_dir / '2.5_songformer_sections' / 'SF_song_sections.png'
        if sf_png.exists():
            sf_song_pngs.append(sf_png)
            print(f'  Found SongFormer song sections: {track_dir.name}')

    if sf_song_pngs:
        print(f'\nConverting and merging {len(sf_song_pngs)} SongFormer song sections PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(sf_song_pngs):
            temp_pdf = temp_dir / f'sf_song_sections_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_songformer_song_sections.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 23. Merge section anchoring raster plots (PNG files in 6.1_anchoring folder)
    print('\nLooking for section anchoring raster plots...')
    section_anchoring_pngs = []
    for track_dir in track_dirs:
        anchoring_png = track_dir / '6.1_anchoring' / f'{track_dir.name}_section_anchoring_raster.png'
        if anchoring_png.exists():
            section_anchoring_pngs.append(anchoring_png)
            print(f'  Found section anchoring raster: {track_dir.name}')

    if section_anchoring_pngs:
        print(f'\nConverting and merging {len(section_anchoring_pngs)} section anchoring raster PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(section_anchoring_pngs):
            temp_pdf = temp_dir / f'section_anchoring_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_section_anchoring_raster.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 24. Merge filtered section anchoring raster plots (PNG files in 6.2_filtered_patterns folder)
    print('\nLooking for filtered section anchoring raster plots...')
    filtered_anchoring_pngs = []
    for track_dir in track_dirs:
        filtered_png = track_dir / '6.2_filtered_patterns' / f'{track_dir.name}_section_anchoring_raster.png'
        if filtered_png.exists():
            filtered_anchoring_pngs.append(filtered_png)
            print(f'  Found filtered anchoring raster: {track_dir.name}')

    if filtered_anchoring_pngs:
        print(f'\nConverting and merging {len(filtered_anchoring_pngs)} filtered anchoring raster PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(filtered_anchoring_pngs):
            temp_pdf = temp_dir / f'filtered_anchoring_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_section_anchoring_raster_filtered.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 25. Merge unfiltered onset histograms (from 6.1_anchoring folder)
    print('\nLooking for unfiltered onset histograms...')
    unfiltered_pattern_pngs = []
    unfiltered_bar_pngs = []
    for track_dir in track_dirs:
        pattern_png = track_dir / '6.1_anchoring' / f'{track_dir.name}_onsets_per_pattern.png'
        bar_png = track_dir / '6.1_anchoring' / f'{track_dir.name}_onsets_per_bar.png'
        if pattern_png.exists():
            unfiltered_pattern_pngs.append(pattern_png)
            print(f'  Found unfiltered onsets_per_pattern: {track_dir.name}')
        if bar_png.exists():
            unfiltered_bar_pngs.append(bar_png)

    if unfiltered_pattern_pngs:
        print(f'\nConverting and merging {len(unfiltered_pattern_pngs)} unfiltered onsets_per_pattern PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(unfiltered_pattern_pngs):
            temp_pdf = temp_dir / f'unfiltered_pattern_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_onsets_per_pattern.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    if unfiltered_bar_pngs:
        print(f'\nConverting and merging {len(unfiltered_bar_pngs)} unfiltered onsets_per_bar PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(unfiltered_bar_pngs):
            temp_pdf = temp_dir / f'unfiltered_bar_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_onsets_per_bar.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 26. Merge filtered onset histograms (from 6.2_filtered_patterns folder)
    print('\nLooking for filtered onset histograms...')
    filtered_pattern_pngs = []
    filtered_bar_pngs = []
    for track_dir in track_dirs:
        pattern_png = track_dir / '6.2_filtered_patterns' / f'{track_dir.name}_onsets_per_pattern.png'
        bar_png = track_dir / '6.2_filtered_patterns' / f'{track_dir.name}_onsets_per_bar.png'
        if pattern_png.exists():
            filtered_pattern_pngs.append(pattern_png)
            print(f'  Found filtered onsets_per_pattern: {track_dir.name}')
        if bar_png.exists():
            filtered_bar_pngs.append(bar_png)

    if filtered_pattern_pngs:
        print(f'\nConverting and merging {len(filtered_pattern_pngs)} filtered onsets_per_pattern PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(filtered_pattern_pngs):
            temp_pdf = temp_dir / f'filtered_pattern_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_filtered_onsets_per_pattern.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    if filtered_bar_pngs:
        print(f'\nConverting and merging {len(filtered_bar_pngs)} filtered onsets_per_bar PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(filtered_bar_pngs):
            temp_pdf = temp_dir / f'filtered_bar_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_filtered_onsets_per_bar.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 27. Merge anchored rhythm histograms (PNG files in 6.6_anchored_rhythm_histograms folder)
    print('\nLooking for anchored rhythm histograms...')
    anchored_rhythm_pngs = []
    for track_dir in track_dirs:
        # Look for both filtered and non-filtered versions
        for suffix in ['_filtered_anchored_rhythm_histograms.png', '_anchored_rhythm_histograms.png']:
            anchored_png = track_dir / '6.6_anchored_rhythm_histograms' / f'{track_dir.name}{suffix}'
            if anchored_png.exists():
                anchored_rhythm_pngs.append(anchored_png)
                print(f'  Found anchored rhythm histogram: {track_dir.name}')
                break

    if anchored_rhythm_pngs:
        print(f'\nConverting and merging {len(anchored_rhythm_pngs)} anchored rhythm histogram PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(anchored_rhythm_pngs):
            temp_pdf = temp_dir / f'anchored_rhythm_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_anchored_rhythm_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 28. Merge anchored groove pulse histograms (PNG files in 6.6_anchored_rhythm_histograms folder)
    print('\nLooking for anchored groove pulse histograms...')
    anchored_groove_pngs = []
    for track_dir in track_dirs:
        for suffix in ['_filtered_anchored_groove_pulse_histograms.png', '_anchored_groove_pulse_histograms.png']:
            anchored_png = track_dir / '6.6_anchored_rhythm_histograms' / f'{track_dir.name}{suffix}'
            if anchored_png.exists():
                anchored_groove_pngs.append(anchored_png)
                print(f'  Found anchored groove pulse histogram: {track_dir.name}')
                break

    if anchored_groove_pngs:
        print(f'\nConverting and merging {len(anchored_groove_pngs)} anchored groove pulse histogram PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(anchored_groove_pngs):
            temp_pdf = temp_dir / f'anchored_groove_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_anchored_groove_pulse_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 29. Merge anchored rhythm patterns (PNG files in 6.6_anchored_rhythm_histograms folder)
    print('\nLooking for anchored rhythm patterns...')
    anchored_pattern_pngs = []
    for track_dir in track_dirs:
        for suffix in ['_filtered_anchored_rhythm_patterns.png', '_anchored_rhythm_patterns.png']:
            anchored_png = track_dir / '6.6_anchored_rhythm_histograms' / f'{track_dir.name}{suffix}'
            if anchored_png.exists():
                anchored_pattern_pngs.append(anchored_png)
                print(f'  Found anchored rhythm pattern: {track_dir.name}')
                break

    if anchored_pattern_pngs:
        print(f'\nConverting and merging {len(anchored_pattern_pngs)} anchored rhythm pattern PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(anchored_pattern_pngs):
            temp_pdf = temp_dir / f'anchored_pattern_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_anchored_rhythm_patterns.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 30. Merge anchored beat histograms (PNG files in 6.7_anchored_beat_histograms folder)
    print('\nLooking for anchored beat histograms...')
    anchored_beat_pngs = []
    for track_dir in track_dirs:
        anchored_png = track_dir / '6.7_anchored_beat_histograms' / f'{track_dir.name}_anchored_beat_histograms.png'
        if anchored_png.exists():
            anchored_beat_pngs.append(anchored_png)
            print(f'  Found anchored beat histogram: {track_dir.name}')

    if anchored_beat_pngs:
        print(f'\nConverting and merging {len(anchored_beat_pngs)} anchored beat histogram PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(anchored_beat_pngs):
            temp_pdf = temp_dir / f'anchored_beat_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_anchored_beat_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 31. Merge anchored beat histograms all onsets (PNG files in 6.7_anchored_beat_histograms folder)
    print('\nLooking for anchored beat histograms (all onsets)...')
    anchored_beat_all_pngs = []
    for track_dir in track_dirs:
        anchored_png = track_dir / '6.7_anchored_beat_histograms' / f'{track_dir.name}_anchored_beat_histograms_all_onsets.png'
        if anchored_png.exists():
            anchored_beat_all_pngs.append(anchored_png)
            print(f'  Found anchored beat histogram (all onsets): {track_dir.name}')

    if anchored_beat_all_pngs:
        print(f'\nConverting and merging {len(anchored_beat_all_pngs)} anchored beat histogram (all onsets) PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(anchored_beat_all_pngs):
            temp_pdf = temp_dir / f'anchored_beat_all_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = batch_dir / 'all_anchored_beat_histograms_all_onsets.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # Clean up temporary files at the end
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f'Warning: Could not fully clean up temp directory: {e}')

    print('\n✓ All plots merged successfully!')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python merge_plots.py /path/to/batch/output')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    merge_plots(output_dir)
