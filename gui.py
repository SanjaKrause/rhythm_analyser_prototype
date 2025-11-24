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

class LoopExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LOOP EXTRACTOR 2000")
        self.root.geometry("750x575")
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
        self.apply_to_folder = tk.BooleanVar(value=False)

        # Output mode
        self.output_mode = tk.StringVar(value="daw_ready")  # "detailed" or "daw_ready"

        # Time selection mode
        self.use_snippet_times = tk.BooleanVar(value=False)  # Use manual time range by default
        self.manual_start_time = tk.DoubleVar(value=50.0)
        self.manual_end_time = tk.DoubleVar(value=80.0)

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

        # Time selection section
        time_frame = tk.Frame(left_frame, bg='#000080')
        time_frame.pack(fill=tk.X, pady=(10, 10))

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

            # Add time selection parameters
            if not self.use_snippet_times.get():
                # Manual time mode
                start_time = self.manual_start_time.get()
                end_time = self.manual_end_time.get()
                duration = end_time - start_time

                cmd.extend(["--manual-start", str(start_time)])
                cmd.extend(["--manual-duration", str(duration)])

            self.log_status(f"\nCommand: {' '.join(cmd)}\n")
            self.log_status(f"Mode: {self.output_mode.get()}")

            # Log time settings
            if self.use_snippet_times.get():
                self.log_status(f"Time: Automatic snippet detection (30s)")
            else:
                self.log_status(f"Time: Manual range {start_time}s - {end_time}s (duration: {duration}s)")
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

            # Stream output to status window
            for line in process.stdout:
                self.log_status(line.rstrip())

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
