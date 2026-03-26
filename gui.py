#!/usr/bin/env python3
"""
Loop Extractor 2000 - GUI Application

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

class LoopExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LOOP EXTRACTOR 2000")
        self.root.geometry("750x800")
        self.root.configure(bg='#000080')  # Dark blue background

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

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#000080')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title = tk.Label(
            main_frame,
            text="LOOP EXTRACTOR 2000",
            font=('Arial', 24, 'bold'),
            fg='#0000FF',
            bg='#000080'
        )
        title.pack(pady=(0, 20))

        # Left column - Input/Output
        left_frame = tk.Frame(main_frame, bg='#000080', width=450)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 20))
        left_frame.pack_propagate(False)

        # Input section
        input_label = tk.Label(
            left_frame,
            text="/input_wav_example",
            font=('Arial', 12),
            fg='white',
            bg='#000080'
        )
        input_label.pack(anchor='w')

        input_display = tk.Entry(
            left_frame,
            textvariable=self.input_path,
            font=('Arial', 12),
            state='readonly',
            width=40
        )
        input_display.pack(pady=(5, 10), fill=tk.X)

        load_button = tk.Button(
            left_frame,
            text="Load file/folder BUTTON",
            font=('Arial', 12, 'bold'),
            bg='#0000FF',
            fg='black',
            activebackground='#0000CC',
            activeforeground='black',
            command=self.load_input,
            relief=tk.RAISED,
            bd=3
        )
        load_button.pack(pady=5)

        folder_check = tk.Checkbutton(
            left_frame,
            text="apply to all files in folder",
            variable=self.apply_to_folder,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        folder_check.pack(anchor='w', pady=10)

        # Output section
        output_label = tk.Label(
            left_frame,
            text="/output_folder",
            font=('Arial', 12),
            fg='white',
            bg='#000080'
        )
        output_label.pack(anchor='w', pady=(20, 0))

        output_display = tk.Entry(
            left_frame,
            textvariable=self.output_path,
            font=('Arial', 12),
            state='readonly',
            width=40
        )
        output_display.pack(pady=(5, 10), fill=tk.X)

        output_button = tk.Button(
            left_frame,
            text="Choose output path",
            font=('Arial', 12, 'bold'),
            bg='#0000FF',
            fg='black',
            activebackground='#0000CC',
            activeforeground='black',
            command=self.choose_output,
            relief=tk.RAISED,
            bd=3
        )
        output_button.pack(pady=5)

        # Run button
        self.run_button = tk.Button(
            left_frame,
            text="RUN ANALYSIS",
            font=('Arial', 14, 'bold'),
            bg='#00FF00',
            fg='black',
            activebackground='#00CC00',
            activeforeground='black',
            command=self.run_analysis,
            relief=tk.RAISED,
            bd=4,
            width=20,
            cursor='hand2',
            state=tk.NORMAL
        )
        self.run_button.pack(pady=20)
        self.run_button.lift()  # Ensure button is on top layer

        # Debug: bind additional click event
        self.run_button.bind('<Button-1>', lambda e: print("DEBUG: Button clicked!"))

        # Horizontal container for TIME RANGE and ONSET CALCULATION
        time_onset_container = tk.Frame(left_frame, bg='#000080')
        time_onset_container.pack(fill=tk.X, pady=(10, 10))

        # Time selection section (left side)
        time_frame = tk.Frame(time_onset_container, bg='#000080')
        time_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        time_label = tk.Label(
            time_frame,
            text="TIME RANGE:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        time_label.pack(anchor='w', pady=(0, 10))

        # Radio button: Use snippet times
        snippet_radio = tk.Radiobutton(
            time_frame,
            text="Use snippet times (30s)",
            variable=self.use_snippet_times,
            value=True,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        snippet_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: Manual time range
        manual_radio = tk.Radiobutton(
            time_frame,
            text="Manual time range:",
            variable=self.use_snippet_times,
            value=False,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        manual_radio.pack(anchor='w', pady=(0, 8))

        # Sliders container (centered)
        sliders_container = tk.Frame(time_frame, bg='#000080')
        sliders_container.pack(pady=(5, 0))

        # Start time slider
        start_slider_frame = tk.Frame(sliders_container, bg='#000080')
        start_slider_frame.pack(pady=(0, 8))

        start_label = tk.Label(
            start_slider_frame,
            text="Start:",
            font=('Arial', 9),
            fg='white',
            bg='#000080',
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
            bg='#000080',
            fg='white',
            highlightbackground='#000080',
            troughcolor='#0000FF',
            activebackground='#0000CC',
            showvalue=False,
            length=200
        )
        start_slider.pack(side=tk.LEFT, padx=5)

        self.start_value_label = tk.Label(
            start_slider_frame,
            text=f"{int(self.manual_start_time.get())}s",
            font=('Arial', 9),
            fg='#00FF00',
            bg='#000080',
            width=5,
            anchor='w'
        )
        self.start_value_label.pack(side=tk.LEFT)

        # End time slider
        end_slider_frame = tk.Frame(sliders_container, bg='#000080')
        end_slider_frame.pack(pady=(0, 0))

        end_label = tk.Label(
            end_slider_frame,
            text="End:",
            font=('Arial', 9),
            fg='white',
            bg='#000080',
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
            bg='#000080',
            fg='white',
            highlightbackground='#000080',
            troughcolor='#0000FF',
            activebackground='#0000CC',
            showvalue=False,
            length=200
        )
        end_slider.pack(side=tk.LEFT, padx=5)

        self.end_value_label = tk.Label(
            end_slider_frame,
            text=f"{int(self.manual_end_time.get())}s",
            font=('Arial', 9),
            fg='#00FF00',
            bg='#000080',
            width=5,
            anchor='w'
        )
        self.end_value_label.pack(side=tk.LEFT)

        # Update labels when sliders move
        self.manual_start_time.trace_add('write', lambda *args: self.start_value_label.config(text=f"{int(self.manual_start_time.get())}s"))
        self.manual_end_time.trace_add('write', lambda *args: self.end_value_label.config(text=f"{int(self.manual_end_time.get())}s"))

        # Onset calculation section (right side, next to TIME RANGE)
        onset_frame = tk.Frame(time_onset_container, bg='#000080')
        onset_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        onset_label = tk.Label(
            onset_frame,
            text="ONSET CALCULATION:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        onset_label.pack(anchor='w', pady=(0, 10))

        # Radio button: Librosa (default)
        librosa_radio = tk.Radiobutton(
            onset_frame,
            text="Librosa onset detection",
            variable=self.onset_mode,
            value="librosa",
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        librosa_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: DrumTranscriber
        drumtranscriber_radio = tk.Radiobutton(
            onset_frame,
            text="DrumTranscriber CNN",
            variable=self.onset_mode,
            value="drumtranscriber",
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        drumtranscriber_radio.pack(anchor='w', pady=(0, 8))

        # ANCHORING MODE section
        anchoring_label = tk.Label(
            onset_frame,
            text="ANCHORING MODE:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        anchoring_label.pack(anchor='w', pady=(15, 10))

        # Radio button: Double anchoring (default)
        double_radio = tk.Radiobutton(
            onset_frame,
            text="Double (start + end)",
            variable=self.anchoring_mode,
            value="double",
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        double_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: Single anchoring
        single_radio = tk.Radiobutton(
            onset_frame,
            text="Single (start only)",
            variable=self.anchoring_mode,
            value="single",
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        single_radio.pack(anchor='w', pady=(0, 8))

        # REUSE EXISTING FILES section
        reuse_label = tk.Label(
            onset_frame,
            text="REUSE FILES:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        reuse_label.pack(anchor='w', pady=(15, 10))

        # Checkbox: Reuse stems + SongFormer
        reuse_check = tk.Checkbutton(
            onset_frame,
            text="Use existing stems/beats",
            variable=self.reuse_existing,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        reuse_check.pack(anchor='w', pady=(0, 8))

        # ONSET STEMS section
        stems_label = tk.Label(
            onset_frame,
            text="ONSET STEMS:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        stems_label.pack(anchor='w', pady=(15, 10))

        # Radio button: Drums only (default)
        drums_only_radio = tk.Radiobutton(
            onset_frame,
            text="Drums only",
            variable=self.onset_all_stems,
            value=False,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        drums_only_radio.pack(anchor='w', pady=(0, 8))

        # Radio button: All 5 stems
        all_stems_radio = tk.Radiobutton(
            onset_frame,
            text="All 5 stems",
            variable=self.onset_all_stems,
            value=True,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        all_stems_radio.pack(anchor='w', pady=(0, 8))

        # Right column - Plots and Status
        right_frame = tk.Frame(main_frame, bg='#000080')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        outputs_label = tk.Label(
            right_frame,
            text="OUTPUT MODE:",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#000080'
        )
        outputs_label.pack(anchor='w', pady=(0, 10))

        # Radio button: Detailed Analysis + Plots
        detailed_radio = tk.Radiobutton(
            right_frame,
            text="Detailed Analysis + Plots",
            variable=self.output_mode,
            value="detailed",
            font=('Arial', 12, 'bold'),
            fg='#00FF00',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='#00FF00'
        )
        detailed_radio.pack(anchor='w', pady=(0, 5))

        # Detailed mode steps
        detailed_steps = tk.Label(
            right_frame,
            text="  • Stems\n  • Beat Detection\n  • Downbeat Correction\n  • Onset Detection\n  • Pattern Detection (all methods)\n  • Grid Analysis\n  • RMS Analysis\n  • Tempo Plots\n  • Raster Plots\n  • Audio Examples\n  • MIDI Export (all methods)\n  • Loop Export (all methods)",
            font=('Arial', 9),
            fg='white',
            bg='#000080',
            justify=tk.LEFT
        )
        detailed_steps.pack(anchor='w', pady=(0, 15))

        # Radio button: DAW Ready Loops
        daw_radio = tk.Radiobutton(
            right_frame,
            text="DAW Ready Loops",
            variable=self.output_mode,
            value="daw_ready",
            font=('Arial', 12, 'bold'),
            fg='#00FF00',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='#00FF00'
        )
        daw_radio.pack(anchor='w', pady=(0, 5))

        # DAW Ready mode steps
        daw_steps = tk.Label(
            right_frame,
            text="  • Stems\n  • Loops (drum pattern method only)\n  • MIDI (drum pattern method only)",
            font=('Arial', 9),
            fg='white',
            bg='#000080',
            justify=tk.LEFT
        )
        daw_steps.pack(anchor='w', pady=(0, 15))

        # Export format section
        format_label = tk.Label(
            right_frame,
            text="EXPORT FORMAT:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        format_label.pack(anchor='w', pady=(15, 5))

        format_frame = tk.Frame(right_frame, bg='#000080')
        format_frame.pack(anchor='w')

        # WAV radio button
        wav_radio = tk.Radiobutton(
            format_frame,
            text="WAV",
            variable=self.export_format,
            value="wav",
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        wav_radio.pack(side=tk.LEFT, padx=(0, 15))

        # MP3 radio button
        mp3_radio = tk.Radiobutton(
            format_frame,
            text="MP3",
            variable=self.export_format,
            value="mp3",
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080',
            activeforeground='white'
        )
        mp3_radio.pack(side=tk.LEFT)

        # Circular progress indicator section
        progress_indicator_frame = tk.Frame(right_frame, bg='#000080')
        progress_indicator_frame.pack(fill=tk.X, pady=(20, 10))

        # Container for pie chart and labels
        pie_container = tk.Frame(progress_indicator_frame, bg='#000080')
        pie_container.pack()

        # Circular progress canvas (pie chart)
        self.pie_size = 80
        self.progress_canvas = tk.Canvas(
            pie_container,
            width=self.pie_size,
            height=self.pie_size,
            bg='#000080',
            highlightthickness=0
        )
        self.progress_canvas.pack(side=tk.LEFT, padx=(0, 15))

        # Draw initial empty pie (black circle)
        self._draw_pie_chart(0)

        # Labels container (right of pie chart)
        labels_container = tk.Frame(pie_container, bg='#000080')
        labels_container.pack(side=tk.LEFT, fill=tk.Y)

        # Songs processed counter
        self.songs_counter_label = tk.Label(
            labels_container,
            text="Songs Analysed: 0/0",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        self.songs_counter_label.pack(anchor='w', pady=(5, 5))

        # Estimated time remaining
        self.time_remaining_label = tk.Label(
            labels_container,
            text="Est. Time Left: --:--",
            font=('Arial', 10),
            fg='#00FF00',
            bg='#000080'
        )
        self.time_remaining_label.pack(anchor='w', pady=(0, 5))

        # Progress bar
        progress_frame = tk.Frame(right_frame, bg='#000080')
        progress_frame.pack(fill=tk.X, pady=(20, 10))

        # Main progress bar (animated)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='indeterminate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Status/log area (system monitor)
        status_frame = tk.Frame(right_frame, bg='#000080')
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        status_label = tk.Label(
            status_frame,
            text="System Monitor:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        status_label.pack(anchor='w')

        self.status_text = tk.Text(
            status_frame,
            height=20,
            width=60,
            bg='#000040',
            fg='#00FF00',
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

        # Draw background circle (black/dark)
        self.progress_canvas.create_oval(
            x0, y0, x1, y1,
            fill='#000000',
            outline='#404040',
            width=2
        )

        # Draw progress arc (white fill) if there's any progress
        if progress_fraction > 0:
            # Arc starts at top (90 degrees) and goes clockwise (negative extent)
            extent = -360 * progress_fraction
            self.progress_canvas.create_arc(
                x0, y0, x1, y1,
                start=90,
                extent=extent,
                fill='white',
                outline='white'
            )

        # Draw center text showing percentage
        center_x = self.pie_size / 2
        center_y = self.pie_size / 2
        percentage = int(progress_fraction * 100)
        self.progress_canvas.create_text(
            center_x, center_y,
            text=f"{percentage}%",
            fill='#00FF00' if progress_fraction > 0 else '#808080',
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
