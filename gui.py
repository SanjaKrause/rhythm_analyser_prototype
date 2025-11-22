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

class LoopExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LOOP EXTRACTOR 2000")
        self.root.geometry("1200x700")
        self.root.configure(bg='#000080')  # Dark blue background

        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.apply_to_folder = tk.BooleanVar(value=False)

        # Pattern detection modes
        self.drum_events = tk.BooleanVar(value=True)
        self.drum_melbands = tk.BooleanVar(value=True)
        self.bass_pitch = tk.BooleanVar(value=True)

        # Output options
        self.output_tempo_plots = tk.BooleanVar(value=True)
        self.output_raster_plots = tk.BooleanVar(value=True)
        self.output_midi_loops = tk.BooleanVar(value=True)

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
        left_frame = tk.Frame(main_frame, bg='#000080')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))

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
            fg='white',
            activebackground='#0000CC',
            command=self.load_input,
            relief=tk.RAISED,
            bd=3
        )
        load_button.pack(pady=5)

        folder_check = tk.Checkbutton(
            left_frame,
            text="(X) apply to all files in folder",
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
            fg='white',
            activebackground='#0000CC',
            command=self.choose_output,
            relief=tk.RAISED,
            bd=3
        )
        output_button.pack(pady=5)

        # Pattern detection mode
        pattern_label = tk.Label(
            left_frame,
            text="PATTERN DETECTION MODE:",
            font=('Arial', 12, 'bold'),
            fg='white',
            bg='#000080'
        )
        pattern_label.pack(anchor='w', pady=(30, 10))

        pattern_frame = tk.Frame(left_frame, bg='#000080')
        pattern_frame.pack(anchor='w')

        drum_events_check = tk.Checkbutton(
            pattern_frame,
            text="()DRUM EVENTS",
            variable=self.drum_events,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080'
        )
        drum_events_check.pack(side=tk.LEFT, padx=(0, 20))

        melbands_check = tk.Checkbutton(
            pattern_frame,
            text="()DRUM MELBANDS",
            variable=self.drum_melbands,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080'
        )
        melbands_check.pack(side=tk.LEFT, padx=(0, 20))

        bass_check = tk.Checkbutton(
            pattern_frame,
            text="()BASS PITCH",
            variable=self.bass_pitch,
            font=('Arial', 10),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080'
        )
        bass_check.pack(side=tk.LEFT)

        # Raster plot note
        raster_note = tk.Label(
            left_frame,
            text="• HERE 1 RASTER PLOT FOR\n  CHOSEN METHOD",
            font=('Arial', 11),
            fg='black',
            bg='#000080',
            justify=tk.LEFT
        )
        raster_note.pack(anchor='w', pady=(40, 20))

        # Run button
        run_button = tk.Button(
            left_frame,
            text="RUN ANALYSIS",
            font=('Arial', 14, 'bold'),
            bg='#00FF00',
            fg='black',
            activebackground='#00CC00',
            command=self.run_analysis,
            relief=tk.RAISED,
            bd=4,
            width=20
        )
        run_button.pack(pady=20)

        # Right column - Outputs
        right_frame = tk.Frame(main_frame, bg='#000080')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        outputs_label = tk.Label(
            right_frame,
            text="OUTPUTS:",
            font=('Arial', 14, 'bold'),
            fg='white',
            bg='#000080'
        )
        outputs_label.pack(anchor='w')

        tempo_check = tk.Checkbutton(
            right_frame,
            text="()1. TEMPO PLOTS",
            variable=self.output_tempo_plots,
            font=('Arial', 11),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080'
        )
        tempo_check.pack(anchor='w', pady=5)

        raster_check = tk.Checkbutton(
            right_frame,
            text="()2. RASTER PLOTS",
            variable=self.output_raster_plots,
            font=('Arial', 11),
            fg='white',
            bg='#000080',
            selectcolor='#000080',
            activebackground='#000080'
        )
        raster_check.pack(anchor='w', pady=5)

        midi_label = tk.Label(
            right_frame,
            text="()10 MIDI LOOP etc etc",
            font=('Arial', 11),
            fg='white',
            bg='#000080'
        )
        midi_label.pack(anchor='w', pady=(40, 5))

        # Status/log area
        status_frame = tk.Frame(right_frame, bg='#000080')
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))

        status_label = tk.Label(
            status_frame,
            text="Status:",
            font=('Arial', 11, 'bold'),
            fg='white',
            bg='#000080'
        )
        status_label.pack(anchor='w')

        self.status_text = tk.Text(
            status_frame,
            height=15,
            width=50,
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
            path = filedialog.askdirectory(title="Select input folder")
        else:
            path = filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[
                    ("Audio files", "*.wav *.mp3 *.flac"),
                    ("All files", "*.*")
                ]
            )

        if path:
            self.input_path.set(path)
            self.log_status(f"Input selected: {path}")

    def choose_output(self):
        """Choose output directory"""
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_path.set(path)
            self.log_status(f"Output path: {path}")

    def log_status(self, message):
        """Add message to status log"""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def run_analysis(self):
        """Run the analysis pipeline"""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input file or folder")
            return

        if not self.output_path.get():
            messagebox.showerror("Error", "Please select an output folder")
            return

        self.log_status("\n" + "="*50)
        self.log_status("Starting analysis...")
        self.log_status("="*50)

        # Run in separate thread to avoid blocking GUI
        thread = threading.Thread(target=self._run_pipeline)
        thread.daemon = True
        thread.start()

    def _run_pipeline(self):
        """Execute the pipeline (runs in separate thread)"""
        try:
            # Build command
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "AP_2_code" / "main.py")
            ]

            if self.apply_to_folder.get():
                cmd.extend(["--audio-dir", self.input_path.get()])
                cmd.append("--analyse-all")
            else:
                cmd.extend(["--audio", self.input_path.get()])

            cmd.extend(["--output-dir", self.output_path.get()])

            # Add track ID (use filename)
            input_path = Path(self.input_path.get())
            if input_path.is_file():
                track_id = input_path.stem
            else:
                track_id = "batch"
            cmd.extend(["--track-id", track_id])

            self.log_status(f"\nCommand: {' '.join(cmd)}\n")

            # Run pipeline
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to status window
            for line in process.stdout:
                self.log_status(line.rstrip())

            process.wait()

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
            self.log_status(f"\n✗ Error: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")


def main():
    root = tk.Tk()
    app = LoopExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
