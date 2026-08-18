#!/usr/bin/env python3
"""
Rhythm Pattern Extractor - GUI Application

A graphical interface for the rhythm analysis and loop extraction pipeline.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess
import threading
import sys
import os
import time
import math

from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Colour palette (light / white theme)
# ---------------------------------------------------------------------------
BG          = '#FFFFFF'   # window / panel background (white)
PANEL       = '#F4F5F7'   # subtle panel / entry background
INK         = '#1F2430'   # primary text
MUTED       = '#6B7280'   # secondary text / hints
ACCENT      = '#C50E1F'   # TU Berlin red (title / brand accent)
BTN         = '#2563EB'   # primary buttons (blue)
BTN_ACT     = '#1D4ED8'   # primary button hover
RUN_BG      = '#16A34A'   # run button (green)
RUN_ACT     = '#15803D'   # run button hover
GREEN       = '#15803D'   # positive / value text on light bg
SELECT      = '#DCE3EA'   # radio / check indicator fill
TROUGH      = '#D1D5DB'   # slider trough
BORDER      = '#D1D5DB'   # thin separators / entry borders
CONSOLE_BG  = '#111827'   # system monitor (dark console)
CONSOLE_FG  = '#22C55E'   # system monitor text (terminal green)

LOGO_DIR = Path(__file__).parent / 'logos'


class LoopExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rhythm Pattern Extractor")
        self.root.geometry("800x880")
        self.root.configure(bg=BG)

        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Track running process
        self.running_process = None

        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.last_input_dir = None
        self.last_output_dir = None
        self.apply_to_folder = tk.BooleanVar(value=True)  # Default: apply to all files

        # Progress tracking
        self.songs_processed = 0
        self.songs_total = 0
        self.processing_times = []  # List of processing times in seconds
        self.current_song_start_time = None  # Track when current song started

        # Output mode
        self.output_mode = tk.StringVar(value="detailed")  # Default: detailed analysis + plots

        # Export format
        self.export_format = tk.StringVar(value="wav")  # "wav" or "mp3" - Default: wav

        # Time selection mode
        self.use_snippet_times = tk.BooleanVar(value=True)  # Default: use snippet times
        self.manual_start_time = tk.DoubleVar(value=50.0)
        self.manual_end_time = tk.DoubleVar(value=80.0)

        # Onset calculation mode
        self.onset_mode = tk.StringVar(value="librosa")  # Default: librosa

        # Anchoring mode
        self.anchoring_mode = tk.StringVar(value="double")  # Default: double

        # Reuse existing files (skip steps 1-4.5, 5.5)
        self.reuse_existing = tk.BooleanVar(value=False)  # Default: run full pipeline

        # Onset stems: drums only or all stems
        self.onset_all_stems = tk.BooleanVar(value=False)  # Default: drums only

        # Fullmix calculation
        self.calculate_fullmix = tk.BooleanVar(value=False)  # Default: don't calculate fullmix
        self.fullmix_dir_path = tk.StringVar(value="/Volumes/PortableSSD/mastabfiles/renamed")  # Default path

        # Keep references to logo images so they are not garbage-collected
        self._logo_refs = []

        self.setup_ui()

    # ------------------------------------------------------------------ helpers
    def _load_logo(self, filename, height):
        """Load a logo PNG, resize to `height` px (keeping aspect) and flatten
        onto white so transparency renders cleanly on the light background."""
        img = Image.open(LOGO_DIR / filename).convert('RGBA')
        w, h = img.size
        new_w = max(1, int(round(w * height / h)))
        img = img.resize((new_w, height), Image.LANCZOS)
        white = Image.new('RGBA', img.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(white, img).convert('RGB')
        photo = ImageTk.PhotoImage(flat)
        self._logo_refs.append(photo)  # prevent GC
        return photo

    def _flat_button(self, parent, text, command, bg, bg_active,
                     fg='white', font=('Arial', 12, 'bold'), padx=16, pady=9):
        """A flat, solid-colour button built from a Label so the colour is
        honoured on macOS (native tk.Button ignores bg there)."""
        btn = tk.Label(parent, text=text, font=font, bg=bg, fg=fg,
                       cursor='hand2', padx=padx, pady=pady)
        btn._bg, btn._bg_active = bg, bg_active
        btn.bind('<Enter>', lambda e: btn.config(bg=btn._bg_active))
        btn.bind('<Leave>', lambda e: btn.config(bg=btn._bg))
        btn.bind('<Button-1>', lambda e: command())
        return btn

    def _radio(self, parent, text, variable, value, font=('Arial', 10), fg=INK):
        return tk.Radiobutton(
            parent, text=text, variable=variable, value=value, font=font,
            fg=fg, bg=BG, selectcolor=SELECT, activebackground=BG,
            activeforeground=fg, highlightthickness=0, anchor='w'
        )

    def _check(self, parent, text, variable, font=('Arial', 10), fg=INK):
        return tk.Checkbutton(
            parent, text=text, variable=variable, font=font,
            fg=fg, bg=BG, selectcolor=SELECT, activebackground=BG,
            activeforeground=fg, highlightthickness=0, anchor='w'
        )

    # ------------------------------------------------------------------ layout
    def setup_ui(self):
        # ---- Header with logos + title ------------------------------------
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill=tk.X, padx=24, pady=(16, 6))

        akt_logo = self._load_logo('akt.png', 58)
        tk.Label(header, image=akt_logo, bg=BG).pack(side=tk.LEFT)

        tu_logo = self._load_logo('tuBerlin.png', 52)
        tk.Label(header, image=tu_logo, bg=BG).pack(side=tk.RIGHT)

        tk.Label(
            header,
            text="Rhythm Pattern Extractor",
            font=('Arial', 26, 'bold'),
            fg=ACCENT,
            bg=BG
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # thin separator under the header
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill=tk.X, padx=24)

        # Main container
        main_frame = tk.Frame(self.root, bg=BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left column - Input/Output
        left_frame = tk.Frame(main_frame, bg=BG, width=450)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 20))
        left_frame.pack_propagate(False)

        # Input section
        input_label = tk.Label(
            left_frame,
            text="/input_wav_example",
            font=('Arial', 12),
            fg=INK,
            bg=BG
        )
        input_label.pack(anchor='w')

        input_display = tk.Entry(
            left_frame,
            textvariable=self.input_path,
            font=('Arial', 12),
            state='readonly',
            readonlybackground=PANEL,
            fg=INK,
            relief=tk.SOLID,
            bd=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            width=40
        )
        input_display.pack(pady=(5, 10), fill=tk.X)

        load_button = self._flat_button(
            left_frame, "Load file / folder", self.load_input, BTN, BTN_ACT
        )
        load_button.pack(pady=5)

        folder_check = self._check(
            left_frame, "apply to all files in folder", self.apply_to_folder
        )
        folder_check.pack(anchor='w', pady=10)

        # Output section
        output_label = tk.Label(
            left_frame,
            text="/output_folder",
            font=('Arial', 12),
            fg=INK,
            bg=BG
        )
        output_label.pack(anchor='w', pady=(20, 0))

        output_display = tk.Entry(
            left_frame,
            textvariable=self.output_path,
            font=('Arial', 12),
            state='readonly',
            readonlybackground=PANEL,
            fg=INK,
            relief=tk.SOLID,
            bd=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            width=40
        )
        output_display.pack(pady=(5, 10), fill=tk.X)

        output_button = self._flat_button(
            left_frame, "Choose output path", self.choose_output, BTN, BTN_ACT
        )
        output_button.pack(pady=5)

        # Run button
        self.run_button = self._flat_button(
            left_frame, "RUN ANALYSIS", self.run_analysis, RUN_BG, RUN_ACT,
            font=('Arial', 14, 'bold'), padx=40, pady=12
        )
        self.run_button.pack(pady=20)
        self.run_button.lift()  # Ensure button is on top layer

        # Horizontal container for TIME RANGE and ONSET CALCULATION
        time_onset_container = tk.Frame(left_frame, bg=BG)
        time_onset_container.pack(fill=tk.X, pady=(10, 10))

        # Time selection section (left side)
        time_frame = tk.Frame(time_onset_container, bg=BG)
        time_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        time_label = tk.Label(
            time_frame,
            text="TIME RANGE:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        time_label.pack(anchor='w', pady=(0, 10))

        # Radio button: Use snippet times
        snippet_radio = self._radio(
            time_frame, "Use snippet times (30s)", self.use_snippet_times, True
        )
        snippet_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: Manual time range
        manual_radio = self._radio(
            time_frame, "Manual time range:", self.use_snippet_times, False
        )
        manual_radio.pack(anchor='w', pady=(0, 8))

        # Sliders container (centered)
        sliders_container = tk.Frame(time_frame, bg=BG)
        sliders_container.pack(pady=(5, 0))

        # Start time slider
        start_slider_frame = tk.Frame(sliders_container, bg=BG)
        start_slider_frame.pack(pady=(0, 8))

        start_label = tk.Label(
            start_slider_frame,
            text="Start:",
            font=('Arial', 9),
            fg=INK,
            bg=BG,
            width=6,
            anchor='w'
        )
        start_label.pack(side=tk.LEFT)

        start_slider = tk.Scale(
            start_slider_frame,
            from_=0,
            to=300,
            orient=tk.HORIZONTAL,
            variable=self.manual_start_time,
            bg=BG,
            fg=INK,
            highlightbackground=BG,
            troughcolor=TROUGH,
            activebackground=BTN,
            showvalue=False,
            length=200
        )
        start_slider.pack(side=tk.LEFT, padx=5)

        self.start_value_label = tk.Label(
            start_slider_frame,
            text=f"{int(self.manual_start_time.get())}s",
            font=('Arial', 9, 'bold'),
            fg=GREEN,
            bg=BG,
            width=5,
            anchor='w'
        )
        self.start_value_label.pack(side=tk.LEFT)

        # End time slider
        end_slider_frame = tk.Frame(sliders_container, bg=BG)
        end_slider_frame.pack(pady=(0, 0))

        end_label = tk.Label(
            end_slider_frame,
            text="End:",
            font=('Arial', 9),
            fg=INK,
            bg=BG,
            width=6,
            anchor='w'
        )
        end_label.pack(side=tk.LEFT)

        end_slider = tk.Scale(
            end_slider_frame,
            from_=0,
            to=300,
            orient=tk.HORIZONTAL,
            variable=self.manual_end_time,
            bg=BG,
            fg=INK,
            highlightbackground=BG,
            troughcolor=TROUGH,
            activebackground=BTN,
            showvalue=False,
            length=200
        )
        end_slider.pack(side=tk.LEFT, padx=5)

        self.end_value_label = tk.Label(
            end_slider_frame,
            text=f"{int(self.manual_end_time.get())}s",
            font=('Arial', 9, 'bold'),
            fg=GREEN,
            bg=BG,
            width=5,
            anchor='w'
        )
        self.end_value_label.pack(side=tk.LEFT)

        # Update labels when sliders move
        self.manual_start_time.trace_add('write', lambda *args: self.start_value_label.config(text=f"{int(self.manual_start_time.get())}s"))
        self.manual_end_time.trace_add('write', lambda *args: self.end_value_label.config(text=f"{int(self.manual_end_time.get())}s"))

        # ONSET STEMS section (below TIME RANGE)
        stems_label = tk.Label(
            time_frame,
            text="ONSET STEMS:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        stems_label.pack(anchor='w', pady=(20, 10))

        # Radio button: Drums only (default)
        drums_only_radio = self._radio(
            time_frame, "Drums only", self.onset_all_stems, False
        )
        drums_only_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: All 5 stems
        all_stems_radio = self._radio(
            time_frame, "All 5 stems", self.onset_all_stems, True
        )
        all_stems_radio.pack(anchor='w', pady=(0, 8))

        # FULLMIX CALCULATION section
        fullmix_label = tk.Label(
            time_frame,
            text="ALSO CALCULATE ON FULL WAV:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        fullmix_label.pack(anchor='w', pady=(20, 10))

        # Checkbox: Calculate fullmix
        fullmix_check = self._check(time_frame, "Yes", self.calculate_fullmix)
        fullmix_check.pack(anchor='w', pady=(0, 8))

        # Fullmix directory path label
        fullmix_path_label = tk.Label(
            time_frame,
            text="Original WAV folder (required if reuse enabled):",
            font=('Arial', 9),
            fg=MUTED,
            bg=BG
        )
        fullmix_path_label.pack(anchor='w', pady=(5, 3))

        # Fullmix path entry and browse button container
        fullmix_path_frame = tk.Frame(time_frame, bg=BG)
        fullmix_path_frame.pack(fill=tk.X, pady=(0, 5))

        fullmix_path_entry = tk.Entry(
            fullmix_path_frame,
            textvariable=self.fullmix_dir_path,
            font=('Arial', 9),
            bg=PANEL,
            fg=INK,
            insertbackground=INK,
            relief=tk.SOLID,
            bd=1,
            highlightthickness=1,
            highlightbackground=BORDER,
            width=30
        )
        fullmix_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        fullmix_browse_btn = self._flat_button(
            fullmix_path_frame, "Browse", self.browse_fullmix_dir, BTN, BTN_ACT,
            font=('Arial', 9, 'bold'), padx=10, pady=4
        )
        fullmix_browse_btn.pack(side=tk.LEFT)

        # Hint label
        fullmix_hint_label = tk.Label(
            time_frame,
            text="Hint: Required when 'Use existing stems/beats' is checked",
            font=('Arial', 8, 'italic'),
            fg=GREEN,
            bg=BG
        )
        fullmix_hint_label.pack(anchor='w', pady=(2, 0))

        # Onset calculation section (right side, next to TIME RANGE)
        onset_frame = tk.Frame(time_onset_container, bg=BG)
        onset_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        onset_label = tk.Label(
            onset_frame,
            text="ONSET CALCULATION:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        onset_label.pack(anchor='w', pady=(0, 10))

        # Radio button: Librosa (default)
        librosa_radio = self._radio(
            onset_frame, "Librosa onset detection", self.onset_mode, "librosa"
        )
        librosa_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: DrumTranscriber
        drumtranscriber_radio = self._radio(
            onset_frame, "DrumTranscriber CNN", self.onset_mode, "drumtranscriber"
        )
        drumtranscriber_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: Madmom CNN
        madmom_radio = self._radio(
            onset_frame, "Madmom CNN", self.onset_mode, "madmom"
        )
        madmom_radio.pack(anchor='w', pady=(0, 8))

        # ANCHORING MODE section
        anchoring_label = tk.Label(
            onset_frame,
            text="ANCHORING MODE:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        anchoring_label.pack(anchor='w', pady=(15, 10))

        # Radio button: Double anchoring (default)
        double_radio = self._radio(
            onset_frame, "Double (start + end)", self.anchoring_mode, "double"
        )
        double_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: Single anchoring
        single_radio = self._radio(
            onset_frame, "Single (start only)", self.anchoring_mode, "single"
        )
        single_radio.pack(anchor='w', pady=(0, 8))

        # REUSE EXISTING FILES section
        reuse_label = tk.Label(
            onset_frame,
            text="REUSE FILES:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        reuse_label.pack(anchor='w', pady=(15, 10))

        # Checkbox: Reuse stems + SongFormer
        reuse_check = self._check(
            onset_frame, "Use existing stems/beats", self.reuse_existing
        )
        reuse_check.pack(anchor='w', pady=(0, 8))

        # Right column - Plots and Status
        right_frame = tk.Frame(main_frame, bg=BG)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        outputs_label = tk.Label(
            right_frame,
            text="OUTPUT MODE:",
            font=('Arial', 14, 'bold'),
            fg=INK,
            bg=BG
        )
        outputs_label.pack(anchor='w', pady=(0, 10))

        # Radio button: Detailed Analysis + Plots
        detailed_radio = self._radio(
            right_frame, "Detailed Analysis + Plots", self.output_mode, "detailed",
            font=('Arial', 12, 'bold'), fg=ACCENT
        )
        detailed_radio.pack(anchor='w', pady=(0, 5))

        # Detailed mode steps
        detailed_steps = tk.Label(
            right_frame,
            text="  • Stems\n  • Beat Detection\n  • Downbeat Correction\n  • Onset Detection\n  • Pattern Detection (all methods)\n  • Grid Analysis\n  • RMS Analysis\n  • Tempo Plots\n  • Raster Plots\n  • Audio Examples\n  • MIDI Export (all methods)\n  • Loop Export (all methods)",
            font=('Arial', 9),
            fg=MUTED,
            bg=BG,
            justify=tk.LEFT
        )
        detailed_steps.pack(anchor='w', pady=(0, 15))

        # Radio button: DAW Ready Loops
        daw_radio = self._radio(
            right_frame, "DAW Ready Loops", self.output_mode, "daw_ready",
            font=('Arial', 12, 'bold'), fg=ACCENT
        )
        daw_radio.pack(anchor='w', pady=(0, 5))

        # DAW Ready mode steps
        daw_steps = tk.Label(
            right_frame,
            text="  • Stems\n  • Loops (drum pattern method only)\n  • MIDI (drum pattern method only)",
            font=('Arial', 9),
            fg=MUTED,
            bg=BG,
            justify=tk.LEFT
        )
        daw_steps.pack(anchor='w', pady=(0, 15))

        # Export format section
        format_label = tk.Label(
            right_frame,
            text="EXPORT FORMAT:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        format_label.pack(anchor='w', pady=(15, 5))

        format_frame = tk.Frame(right_frame, bg=BG)
        format_frame.pack(anchor='w')

        # WAV radio button
        wav_radio = self._radio(format_frame, "WAV", self.export_format, "wav")
        wav_radio.pack(side=tk.LEFT, padx=(0, 15))

        # MP3 radio button
        mp3_radio = self._radio(format_frame, "MP3", self.export_format, "mp3")
        mp3_radio.pack(side=tk.LEFT)

        # Circular progress indicator section
        progress_indicator_frame = tk.Frame(right_frame, bg=BG)
        progress_indicator_frame.pack(fill=tk.X, pady=(20, 10))

        # Container for pie chart and labels
        pie_container = tk.Frame(progress_indicator_frame, bg=BG)
        pie_container.pack()

        # Circular progress canvas (pie chart)
        self.pie_size = 80
        self.progress_canvas = tk.Canvas(
            pie_container,
            width=self.pie_size,
            height=self.pie_size,
            bg=BG,
            highlightthickness=0
        )
        self.progress_canvas.pack(side=tk.LEFT, padx=(0, 15))

        # Draw initial empty pie
        self._draw_pie_chart(0)

        # Labels container (right of pie chart)
        labels_container = tk.Frame(pie_container, bg=BG)
        labels_container.pack(side=tk.LEFT, fill=tk.Y)

        # Songs processed counter
        self.songs_counter_label = tk.Label(
            labels_container,
            text="Songs Analysed: 0/0",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        self.songs_counter_label.pack(anchor='w', pady=(5, 5))

        # Estimated time remaining
        self.time_remaining_label = tk.Label(
            labels_container,
            text="Est. Time Left: --:--",
            font=('Arial', 10),
            fg=GREEN,
            bg=BG
        )
        self.time_remaining_label.pack(anchor='w', pady=(0, 5))

        # Progress bar
        progress_frame = tk.Frame(right_frame, bg=BG)
        progress_frame.pack(fill=tk.X, pady=(20, 10))

        # Main progress bar (animated)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Status/log area (system monitor)
        status_frame = tk.Frame(right_frame, bg=BG)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        status_label = tk.Label(
            status_frame,
            text="System Monitor:",
            font=('Arial', 11, 'bold'),
            fg=INK,
            bg=BG
        )
        status_label.pack(anchor='w')

        self.status_text = tk.Text(
            status_frame,
            height=20,
            width=60,
            bg=CONSOLE_BG,
            fg=CONSOLE_FG,
            insertbackground=CONSOLE_FG,
            relief=tk.FLAT,
            font=('Courier', 10),
            state=tk.DISABLED
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, pady=5)

        scrollbar = tk.Scrollbar(self.status_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.status_text.yview)

    def load_input(self):
        """Load input file or folder"""
        if self.apply_to_folder.get():
            path = filedialog.askdirectory(
                title="Select input folder",
                initialdir=self.last_input_dir
            )
        else:
            path = filedialog.askopenfilename(
                title="Select audio file",
                initialdir=self.last_input_dir,
                filetypes=[
                    ("Audio files", "*.wav *.mp3 *.flac"),
                    ("All files", "*.*")
                ]
            )

        if path:
            self.input_path.set(path)
            # Remember the directory for next time
            self.last_input_dir = str(Path(path).parent if Path(path).is_file() else path)
            self.log_status(f"Input selected: {path}")

    def choose_output(self):
        """Choose output directory"""
        path = filedialog.askdirectory(
            title="Select output folder",
            initialdir=self.last_output_dir
        )
        if path:
            self.output_path.set(path)
            # Remember the directory for next time
            self.last_output_dir = path
            self.log_status(f"Output path: {path}")

    def browse_fullmix_dir(self):
        """Browse for fullmix original WAV folder"""
        current_path = self.fullmix_dir_path.get()
        initial_dir = current_path if Path(current_path).exists() else None

        path = filedialog.askdirectory(
            title="Select original WAV folder (for fullmix)",
            initialdir=initial_dir
        )
        if path:
            self.fullmix_dir_path.set(path)
            self.log_status(f"Fullmix WAV folder: {path}")

    def log_status(self, message):
        """Add message to status log"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def _draw_pie_chart(self, progress_fraction):
        """
        Draw a circular pie chart showing progress.

        Parameters
        ----------
        progress_fraction : float
            Progress from 0.0 to 1.0
        """
        self.progress_canvas.delete("all")

        # Dimensions
        padding = 5
        x0, y0 = padding, padding
        x1, y1 = self.pie_size - padding, self.pie_size - padding

        # Draw background circle (light)
        self.progress_canvas.create_oval(
            x0, y0, x1, y1,
            fill=PANEL,
            outline=BORDER,
            width=2
        )

        # Draw progress arc (blue fill) if there's any progress
        if progress_fraction > 0:
            # Arc starts at top (90 degrees) and goes clockwise (negative extent)
            extent = -360 * progress_fraction
            self.progress_canvas.create_arc(
                x0, y0, x1, y1,
                start=90,
                extent=extent,
                fill=BTN,
                outline=BTN
            )

        # Draw center text showing percentage
        center_x = self.pie_size / 2
        center_y = self.pie_size / 2
        percentage = int(progress_fraction * 100)
        self.progress_canvas.create_text(
            center_x, center_y,
            text=f"{percentage}%",
            fill=BTN if progress_fraction > 0 else MUTED,
            font=('Arial', 10, 'bold')
        )

    def _update_progress_display(self):
        """Update the progress counter, pie chart, and time estimate."""
        # Update counter label
        self.songs_counter_label.config(
            text=f"Songs Analysed: {self.songs_processed}/{self.songs_total}"
        )

        # Update pie chart
        if self.songs_total > 0:
            progress_fraction = self.songs_processed / self.songs_total
        else:
            progress_fraction = 0
        self._draw_pie_chart(progress_fraction)

        # Update time estimate
        if self.songs_processed > 0 and self.songs_processed < self.songs_total:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            songs_remaining = self.songs_total - self.songs_processed
            est_seconds = avg_time * songs_remaining

            # Format time
            if est_seconds >= 3600:
                hours = int(est_seconds // 3600)
                minutes = int((est_seconds % 3600) // 60)
                time_str = f"{hours}h {minutes}m"
            elif est_seconds >= 60:
                minutes = int(est_seconds // 60)
                seconds = int(est_seconds % 60)
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{int(est_seconds)}s"

            self.time_remaining_label.config(text=f"Est. Time Left: {time_str}")
        elif self.songs_processed >= self.songs_total and self.songs_total > 0:
            self.time_remaining_label.config(text="Est. Time Left: Done!")
        else:
            self.time_remaining_label.config(text="Est. Time Left: --:--")

    def _reset_progress(self):
        """Reset progress tracking for a new batch."""
        self.songs_processed = 0
        self.songs_total = 0
        self.processing_times = []
        self.current_song_start_time = None
        self._update_progress_display()

    def _count_audio_files(self, input_path):
        """Count audio files to process."""
        input_path = Path(input_path)
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}

        if input_path.is_file():
            return 1
        elif input_path.is_dir():
            count = 0
            for ext in audio_extensions:
                count += len(list(input_path.glob(f'*{ext}')))
            return count
        return 0

    def run_analysis(self):
        """Run the analysis pipeline"""
        print("DEBUG: run_analysis called!")  # Debug print

        if not self.input_path.get():
            print("DEBUG: No input path")  # Debug print
            messagebox.showerror("Error", "Please select an input file or folder")
            return

        if not self.output_path.get():
            print("DEBUG: No output path")  # Debug print
            messagebox.showerror("Error", "Please select an output folder")
            return

        print("DEBUG: Starting analysis")  # Debug print
        self.log_status("\n" + "="*50)
        self.log_status("Starting analysis...")
        self.log_status("="*50)

        # Start progress bar
        self.progress_bar.start(10)

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._run_pipeline)
        thread.daemon = True
        thread.start()

    def _run_pipeline(self):
        """Execute the pipeline (runs in separate thread)"""
        try:
            # Reset and initialize progress tracking
            self._reset_progress()
            self.songs_total = self._count_audio_files(self.input_path.get())
            self.root.after(0, self._update_progress_display)

            # Build command
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "loop_extractor" / "main.py")
            ]

            # Determine if input is a file or directory
            input_path = Path(self.input_path.get())

            if input_path.is_dir():
                # Process all files in directory
                cmd.extend(["--audio-dir", self.input_path.get()])
                cmd.append("--analyse-all")
                track_id = "batch"
            else:
                # Process single file
                cmd.extend(["--audio", self.input_path.get()])
                track_id = input_path.stem

            cmd.extend(["--output-dir", self.output_path.get()])
            cmd.extend(["--track-id", track_id])

            # Add mode-specific flags
            if self.output_mode.get() == "daw_ready":
                cmd.append("--daw-ready")  # Only export drum method, stems, loops, midi

            # Add export format
            cmd.extend(["--export-format", self.export_format.get()])

            # Add time selection parameters
            if not self.use_snippet_times.get():
                # Manual time mode
                start_time = self.manual_start_time.get()
                end_time = self.manual_end_time.get()
                duration = end_time - start_time

                cmd.extend(["--manual-start", str(start_time)])
                cmd.extend(["--manual-duration", str(duration)])

            # Add onset mode
            cmd.extend(["--onset-mode", self.onset_mode.get()])

            # Add anchoring mode
            cmd.extend(["--anchoring-mode", self.anchoring_mode.get()])

            # Add reuse existing files flag
            if self.reuse_existing.get():
                cmd.append("--reuse-existing")

            # Add all stems flag
            if self.onset_all_stems.get():
                cmd.append("--all-stems")

            # Add fullmix calculation flag and directory
            if self.calculate_fullmix.get():
                cmd.append("--fullmix")
                cmd.extend(["--fullmix-dir", self.fullmix_dir_path.get()])

            # Add onset threshold for drumtranscriber (default 0.5)
            cmd.extend(["--onset-threshold-drumtranscriber", "0.5"])

            # Add loop start offset (default 0.0ms - use grid time exactly)
            cmd.extend(["--loop-start-offset-ms", "0.0"])

            self.log_status(f"\nCommand: {' '.join(cmd)}\n")
            self.log_status(f"Mode: {self.output_mode.get()}")
            self.log_status(f"Export format: {self.export_format.get().upper()}")

            # Log time settings
            if self.use_snippet_times.get():
                self.log_status(f"Time: Automatic snippet detection (30s)")
            else:
                self.log_status(f"Time: Manual range {start_time}s - {end_time}s (duration: {duration}s)")

            # Log onset mode
            onset_method = "Librosa" if self.onset_mode.get() == "librosa" else "DrumTranscriber CNN"
            self.log_status(f"Onset detection: {onset_method}")
            self.log_status("")

            # Run pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Store process reference for cleanup
            self.running_process = process

            # Stream output to status window and track progress
            for line in process.stdout:
                self.log_status(line.rstrip())

                # Detect when a new song starts processing and extract actual total count
                # Format: "Processing [1/25]: track_name"
                if "Processing [" in line and "]: " in line:
                    try:
                        # Extract the total from "Processing [X/Y]:"
                        import re
                        match = re.search(r'Processing \[(\d+)/(\d+)\]:', line)
                        if match:
                            current_num = int(match.group(1))
                            total_num = int(match.group(2))

                            # Update total if we haven't set it yet or if it's different
                            # (reuse mode might have different count than file count)
                            if self.songs_total != total_num:
                                self.songs_total = total_num
                                self.root.after(0, self._update_progress_display)
                    except:
                        pass

                    self.current_song_start_time = time.time()

                # Detect when a song completes (successfully or with errors/failure)
                # Only count completion if we have a start time (i.e., we saw "Processing [X/Y]:" first)
                # This prevents counting intermediate "completed" messages from pipeline steps
                if self.current_song_start_time is not None:
                    if ("completed successfully" in line or
                        "completed with" in line and "errors" in line or
                        "failed:" in line):
                        # Record processing time
                        elapsed = time.time() - self.current_song_start_time
                        self.processing_times.append(elapsed)
                        self.current_song_start_time = None

                        # Increment processed count
                        self.songs_processed += 1
                        # Update display on main thread
                        self.root.after(0, self._update_progress_display)

            process.wait()

            # Clear process reference
            self.running_process = None

            # Stop progress bar
            self.progress_bar.stop()

            if process.returncode == 0:
                self.log_status("\n" + "="*50)
                self.log_status("✓ Analysis completed successfully!")
                self.log_status("="*50)

                messagebox.showinfo("Success", "Analysis completed successfully!")
            else:
                self.log_status("\n" + "="*50)
                self.log_status(f"✗ Analysis failed with code {process.returncode}")
                self.log_status("="*50)
                messagebox.showerror("Error", f"Analysis failed with code {process.returncode}")

        except Exception as e:
            # Stop progress bar on error
            self.progress_bar.stop()
            self.running_process = None
            self.log_status(f"\n✗ Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def on_closing(self):
        """Handle window closing - cleanup processes"""
        # Kill any running subprocess
        if self.running_process and self.running_process.poll() is None:
            self.running_process.terminate()
            try:
                self.running_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.running_process.kill()

        # Destroy the window and exit
        self.root.destroy()


def main():
    root = tk.Tk()
    app = LoopExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
