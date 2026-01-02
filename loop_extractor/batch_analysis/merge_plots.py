#!/usr/bin/env python3
"""
Simple script to merge all plot PDFs from batch processing into combined PDFs.

Merges:
- All tempo plots into all_tempo_plots.pdf
- All raster comparison plots into all_raster_comparison.pdf
- All raster standard plots into all_raster_standard.pdf
- All microtiming plots into all_microtiming_plots.pdf
- All rhythm histograms into all_rhythm_histograms.pdf

Usage:
    python merge_plots.py /path/to/batch/output
"""

import sys
import shutil
from pathlib import Path
from PyPDF2 import PdfMerger
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader


def png_to_pdf(png_path: Path, pdf_path: Path):
    """Convert PNG to PDF."""
    img = Image.open(png_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    img_width, img_height = img.size
    scale = min(width / img_width, height / img_height) * 0.9

    scaled_width = img_width * scale
    scaled_height = img_height * scale
    x = (width - scaled_width) / 2
    y = (height - scaled_height) / 2

    c.drawImage(ImageReader(img), x, y, scaled_width, scaled_height)
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

    # 4. Merge microtiming plots (PDF files in 5_grid folder)
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

    # 5. Merge rhythm histograms (PDF files in 5.5_rhythm folder)
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

    # Clean up temporary files at the end
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

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
