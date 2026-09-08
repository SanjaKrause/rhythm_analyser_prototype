#!/usr/bin/env python3
"""
Simple script to merge all plot PDFs from batch processing into combined PDFs.

Merges:
- All tempo plots into all_tempo_plots.pdf
- All full-song (no-snippet, shared-scale) tempo plots into all_tempo_plots_fullsong.pdf
- All raster comparison plots into all_raster_comparison.pdf
- All raster standard plots into all_raster_standard.pdf
- All new grid corrections plots into all_new_grid_corrections.pdf
- All microtiming plots into all_microtiming_plots.pdf
- All new 4-method raster plots into all_raster_4method.pdf
- All Spotify sections timelines into all_spotify_sections.pdf
- All SongFormer sections timelines into all_songformer_sections.pdf
- All SongFormer song sections (no snippet) into all_songformer_song_sections_no_snippet.pdf
- All onsets per 1-bar pattern plots into all_onsets_per_pattern_1bar.pdf
- All onsets per 2-bar pattern plots into all_onsets_per_pattern_2bar.pdf
- All onsets per 4-bar pattern plots into all_onsets_per_pattern_4bar.pdf
- All onsets per bar (1-bar grid) plots into all_onsets_per_bar_1bar.pdf
- All onsets per bar (2-bar grid) plots into all_onsets_per_bar_2bar.pdf
- All onsets per bar (4-bar grid) plots into all_onsets_per_bar_4bar.pdf
- All section anchoring raster plots into all_section_anchoring_raster.pdf
- All filtered section anchoring raster plots into all_section_anchoring_raster_filtered.pdf
- All anchored rhythm histograms into all_anchored_rhythm_histograms.pdf
- All anchored groove pulse histograms into all_anchored_groove_pulse_histograms.pdf
- All anchored rhythm patterns into all_anchored_rhythm_patterns.pdf
- All anchored beat histograms into all_anchored_beat_histograms.pdf
- All anchored beat histograms (all onsets) into all_anchored_beat_histograms_all_onsets.pdf
- All anchored microtiming plots into all_anchored_microtiming_plots.pdf (per stem)

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


def merge_plots(output_dir: Path, stem: str = 'drums'):
    """
    Merge all plot PDFs from track folders into combined PDFs.

    Parameters
    ----------
    output_dir : Path
        The batch output directory containing individual track folders
    stem : str
        Stem to collect plots for (default: 'drums')
    """
    # Create batch_analysis folder (stem-specific for stem-related plots)
    batch_dir = output_dir / 'batch_analysis'
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Stem-specific output folder for anchored plots
    stem_batch_dir = batch_dir / stem
    stem_batch_dir.mkdir(parents=True, exist_ok=True)

    # Get all track directories (exclude batch_analysis and any other special folders)
    track_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'feature_sets']])

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

    # 1b. Merge full-song (no-snippet, shared-scale) tempo plots
    print('Looking for full-song tempo plots...')
    tempo_fullsong_pdfs = []
    for track_dir in track_dirs:
        tempo_pdf = track_dir / '3.5_tempo_plots' / f'{track_dir.name}_tempo_plots_fullsong.pdf'
        if tempo_pdf.exists():
            tempo_fullsong_pdfs.append(tempo_pdf)
            print(f'  Found full-song tempo: {track_dir.name}')

    if tempo_fullsong_pdfs:
        print(f'\nMerging {len(tempo_fullsong_pdfs)} full-song tempo PDFs...')
        output_pdf = batch_dir / 'all_tempo_plots_fullsong.pdf'
        merger = PdfMerger()
        for pdf in tempo_fullsong_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # Create single temp directory for all PNG conversions
    # (remove stale leftovers from a previous aborted run first)
    temp_dir = batch_dir / '_temp'
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(exist_ok=True)

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

    # 22b. Merge SongFormer song sections plots WITHOUT snippet marking (PNG files in 2.5_songformer_sections folder)
    print('\nLooking for SongFormer song sections plots (no snippet)...')
    sf_song_no_snippet_pngs = []
    for track_dir in track_dirs:
        sf_png = track_dir / '2.5_songformer_sections' / 'SF_song_sections_no_snippet.png'
        if sf_png.exists():
            sf_song_no_snippet_pngs.append(sf_png)
            print(f'  Found SongFormer song sections (no snippet): {track_dir.name}')

    if sf_song_no_snippet_pngs:
        print(f'\nConverting and merging {len(sf_song_no_snippet_pngs)} SongFormer song sections (no snippet) PNGs...')

        merger = PdfMerger()
        for i, png in enumerate(sf_song_no_snippet_pngs):
            temp_pdf = temp_dir / f'sf_song_sections_no_snippet_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))

        output_pdf = batch_dir / 'all_songformer_song_sections_no_snippet.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 23. Merge section anchoring raster plots (PNG files in 6.1_anchoring/{stem} folder)
    print(f'\nLooking for section anchoring raster plots ({stem})...')
    section_anchoring_pngs = []
    for track_dir in track_dirs:
        anchoring_png = track_dir / '6.1_anchoring' / stem / f'{track_dir.name}_section_anchoring_raster.png'
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

        output_pdf = stem_batch_dir / 'all_section_anchoring_raster.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 24. Merge filtered section anchoring raster plots (PNG files in 6.2_filtered_patterns/{stem} folder)
    print(f'\nLooking for filtered section anchoring raster plots ({stem})...')
    filtered_anchoring_pngs = []
    for track_dir in track_dirs:
        filtered_png = track_dir / '6.2_filtered_patterns' / stem / f'{track_dir.name}_section_anchoring_raster.png'
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

        output_pdf = stem_batch_dir / 'all_section_anchoring_raster_filtered.pdf'
        merger.write(str(output_pdf))
        merger.close()

        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 25. Merge unfiltered onset histograms (from 6.1_anchoring/{stem} folder)
    print(f'\nLooking for unfiltered onset histograms ({stem})...')
    unfiltered_pattern_pngs = []
    unfiltered_bar_pngs = []
    for track_dir in track_dirs:
        pattern_png = track_dir / '6.1_anchoring' / stem / f'{track_dir.name}_onsets_per_pattern.png'
        bar_png = track_dir / '6.1_anchoring' / stem / f'{track_dir.name}_onsets_per_bar.png'
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
        output_pdf = stem_batch_dir / 'all_onsets_per_pattern.pdf'
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
        output_pdf = stem_batch_dir / 'all_onsets_per_bar.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 26. Merge filtered onset histograms (from 6.2_filtered_patterns/{stem} folder)
    print(f'\nLooking for filtered onset histograms ({stem})...')
    filtered_pattern_pngs = []
    filtered_bar_pngs = []
    for track_dir in track_dirs:
        pattern_png = track_dir / '6.2_filtered_patterns' / stem / f'{track_dir.name}_onsets_per_pattern.png'
        bar_png = track_dir / '6.2_filtered_patterns' / stem / f'{track_dir.name}_onsets_per_bar.png'
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
        output_pdf = stem_batch_dir / 'all_filtered_onsets_per_pattern.pdf'
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
        output_pdf = stem_batch_dir / 'all_filtered_onsets_per_bar.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 26b. Merge anchored microtiming plots (PDF files in 6.2_filtered_patterns/{stem} folder)
    # One PDF per track per stem: {track_id}_anchored_microtiming_plots.pdf
    print(f'\nLooking for anchored microtiming plots ({stem})...')
    anchored_microtiming_pdfs = []
    for track_dir in track_dirs:
        stem_dir = track_dir / '6.2_filtered_patterns' / stem
        if stem_dir.exists():
            # Look for the single combined file per track
            pdf = stem_dir / f'{track_dir.name}_anchored_microtiming_plots.pdf'
            if pdf.exists():
                anchored_microtiming_pdfs.append(pdf)
                print(f'  Found anchored microtiming: {track_dir.name}')

    if anchored_microtiming_pdfs:
        print(f'\nMerging {len(anchored_microtiming_pdfs)} anchored microtiming PDFs...')
        output_pdf = stem_batch_dir / 'all_anchored_microtiming_plots.pdf'
        merger = PdfMerger()
        for pdf in anchored_microtiming_pdfs:
            merger.append(str(pdf))
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 27. Merge anchored rhythm histograms (PNG files in 6.6_anchored_rhythm_histograms/{stem} folder)
    print(f'\nLooking for anchored rhythm histograms ({stem})...')
    anchored_rhythm_pngs = []
    for track_dir in track_dirs:
        # Look for both filtered and non-filtered versions
        for suffix in ['_filtered_anchored_rhythm_histograms.png', '_anchored_rhythm_histograms.png']:
            anchored_png = track_dir / '6.6_anchored_rhythm_histograms' / stem / f'{track_dir.name}{suffix}'
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
        output_pdf = stem_batch_dir / 'all_anchored_rhythm_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 28. Merge anchored groove pulse histograms (PNG files in 6.6_anchored_rhythm_histograms/{stem} folder)
    print(f'\nLooking for anchored groove pulse histograms ({stem})...')
    anchored_groove_pngs = []
    for track_dir in track_dirs:
        for suffix in ['_filtered_anchored_groove_pulse_histograms.png', '_anchored_groove_pulse_histograms.png']:
            anchored_png = track_dir / '6.6_anchored_rhythm_histograms' / stem / f'{track_dir.name}{suffix}'
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
        output_pdf = stem_batch_dir / 'all_anchored_groove_pulse_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 29. Merge anchored rhythm patterns (PNG files in 6.6_anchored_rhythm_histograms/{stem} folder)
    print(f'\nLooking for anchored rhythm patterns ({stem})...')
    anchored_pattern_pngs = []
    for track_dir in track_dirs:
        for suffix in ['_filtered_anchored_rhythm_patterns.png', '_anchored_rhythm_patterns.png']:
            anchored_png = track_dir / '6.6_anchored_rhythm_histograms' / stem / f'{track_dir.name}{suffix}'
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
        output_pdf = stem_batch_dir / 'all_anchored_rhythm_patterns.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 30. Merge anchored beat histograms (PNG files in 6.7_anchored_beat_histograms/{stem} folder)
    print(f'\nLooking for anchored beat histograms ({stem})...')
    anchored_beat_pngs = []
    for track_dir in track_dirs:
        anchored_png = track_dir / '6.7_anchored_beat_histograms' / stem / f'{track_dir.name}_anchored_beat_histograms.png'
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
        output_pdf = stem_batch_dir / 'all_anchored_beat_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 31. Merge anchored beat histograms all onsets (PNG files in 6.7_anchored_beat_histograms/{stem} folder)
    print(f'\nLooking for anchored beat histograms (all onsets) ({stem})...')
    anchored_beat_all_pngs = []
    for track_dir in track_dirs:
        anchored_png = track_dir / '6.7_anchored_beat_histograms' / stem / f'{track_dir.name}_anchored_beat_histograms_all_onsets.png'
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
        output_pdf = stem_batch_dir / 'all_anchored_beat_histograms_all_onsets.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 32. Merge groove pulse beat histograms (PNG files in 6.7_anchored_beat_histograms/{stem} folder)
    print(f'\nLooking for groove pulse beat histograms ({stem})...')
    groove_pulse_pngs = []
    for track_dir in track_dirs:
        groove_png = track_dir / '6.7_anchored_beat_histograms' / stem / f'{track_dir.name}_groove_pulse_beat_histograms.png'
        if groove_png.exists():
            groove_pulse_pngs.append(groove_png)
            print(f'  Found groove pulse beat histogram: {track_dir.name}')

    if groove_pulse_pngs:
        print(f'\nConverting and merging {len(groove_pulse_pngs)} groove pulse beat histogram PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(groove_pulse_pngs):
            temp_pdf = temp_dir / f'groove_pulse_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = stem_batch_dir / 'all_groove_pulse_beat_histograms.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 33. Merge groove pulse beat histograms all onsets (PNG files in 6.7_anchored_beat_histograms/{stem} folder)
    print(f'\nLooking for groove pulse beat histograms (all onsets) ({stem})...')
    groove_pulse_all_pngs = []
    for track_dir in track_dirs:
        groove_png = track_dir / '6.7_anchored_beat_histograms' / stem / f'{track_dir.name}_groove_pulse_beat_histograms_all_onsets.png'
        if groove_png.exists():
            groove_pulse_all_pngs.append(groove_png)
            print(f'  Found groove pulse beat histogram (all onsets): {track_dir.name}')

    if groove_pulse_all_pngs:
        print(f'\nConverting and merging {len(groove_pulse_all_pngs)} groove pulse beat histogram (all onsets) PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(groove_pulse_all_pngs):
            temp_pdf = temp_dir / f'groove_pulse_all_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = stem_batch_dir / 'all_groove_pulse_beat_histograms_all_onsets.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # 34. Merge anchored beat patterns (PNG files in 6.7_anchored_beat_histograms/{stem} folder)
    print(f'\nLooking for anchored beat patterns ({stem})...')
    beat_pattern_pngs = []
    for track_dir in track_dirs:
        pattern_png = track_dir / '6.7_anchored_beat_histograms' / stem / f'{track_dir.name}_anchored_beat_patterns.png'
        if pattern_png.exists():
            beat_pattern_pngs.append(pattern_png)
            print(f'  Found anchored beat pattern: {track_dir.name}')

    if beat_pattern_pngs:
        print(f'\nConverting and merging {len(beat_pattern_pngs)} anchored beat pattern PNGs...')
        merger = PdfMerger()
        for i, png in enumerate(beat_pattern_pngs):
            temp_pdf = temp_dir / f'beat_pattern_{i}.pdf'
            png_to_pdf(png, temp_pdf)
            merger.append(str(temp_pdf))
        output_pdf = stem_batch_dir / 'all_anchored_beat_patterns.pdf'
        merger.write(str(output_pdf))
        merger.close()
        print(f'✓ Created: {output_pdf.name} ({output_pdf.stat().st_size / 1024:.1f} KB)')

    # Clean up temporary files at the end
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f'Warning: Could not fully clean up temp directory: {e}')

    print(f'\n✓ All plots merged successfully for stem: {stem}!')


def detect_available_stems(track_dirs):
    """Detect which stems have data by checking the first few tracks."""
    all_stems = ['vocals', 'drums', 'bass', 'piano', 'other', 'fullmix']
    found_stems = set()

    for track_dir in track_dirs[:5]:
        rhythm_hist_dir = track_dir / '6.6_anchored_rhythm_histograms'
        if rhythm_hist_dir.exists():
            for stem in all_stems:
                stem_dir = rhythm_hist_dir / stem
                if stem_dir.exists() and any(stem_dir.glob('*.png')):
                    found_stems.add(stem)

    if not found_stems:
        return ['drums']

    return sorted(found_stems, key=lambda s: all_stems.index(s))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python merge_plots.py /path/to/batch/output [stem]')
        print('If stem is not specified, all available stems will be processed.')
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f'Error: Directory does not exist: {output_dir}')
        sys.exit(1)

    # Find track directories
    track_dirs = sorted([
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name not in ['batch_analysis', '_batch_analysis', 'feature_sets']
    ])

    # Determine which stems to process
    if len(sys.argv) >= 3:
        stems = [sys.argv[2]]
    else:
        stems = detect_available_stems(track_dirs)
        print(f"Detected stems: {stems}")

    # Merge plots for each stem
    for stem in stems:
        merge_plots(output_dir, stem=stem)
