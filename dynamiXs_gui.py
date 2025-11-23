#!/usr/bin/env python3
"""
DynamiXs GUI - A comprehensive interface for NMR relaxation data analysis

This GUI provides access to:
- T1/T2 fitting analysis
- CPMG relaxation dispersion
- Spectral density function analysis
- Data plotting and comparison tools
- Data formatting utilities

Built with CustomTkinter following LunaNMR UX Style Guide v0.9
for cross-platform compatibility (macOS, Windows, Linux)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk
import customtkinter as ctk
import os
import sys
import subprocess
import threading
from pathlib import Path
import pandas as pd

# Import visualization module
from visualization.results_viewer import ResultsViewer

# Import LunaNMR style components
from gui_components import (
    BG_COLOR, PANEL_BG_COLOR, FRAME_BG_COLOR,
    PRIMARY_TEXT, SECONDARY_TEXT,
    PRIMARY_BUTTON_BG, PRIMARY_BUTTON_HOVER, PRIMARY_BUTTON_TEXT,
    SECONDARY_BUTTON_BG, SECONDARY_BUTTON_HOVER, SECONDARY_BUTTON_TEXT, SECONDARY_BUTTON_BORDER,
    DESTRUCTIVE_BUTTON_BG, DESTRUCTIVE_BUTTON_HOVER, DESTRUCTIVE_BUTTON_TEXT,
    SUCCESS_GREEN, SUCCESS_GREEN_SOFT, SUCCESS_GREEN_HOVER, WARNING_ORANGE, ERROR_RED, INFO_BLUE,
    BORDER_COLOR, SEPARATOR_COLOR,
    BUTTON_CORNER_RADIUS, FRAME_CORNER_RADIUS,
    SPACING_XS, SPACING_SM, SPACING_MD, SPACING_LG, SPACING_XL,
    FONT_LARGE_HEADER, FONT_MEDIUM_HEADER, FONT_SECTION_LABEL, FONT_BODY, FONT_SMALL, FONT_MONO,
    CTkLabelFrame,
    create_primary_button, create_secondary_button, create_destructive_button,
    create_label, create_entry
)


class DynamiXsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DynamiXs - NMR Relaxation Analysis Suite")
        self.root.geometry("900x700")

        # Configure window background
        self.root.configure(fg_color=BG_COLOR)

        # Get the dynamiXs directory path
        self.dynamixs_path = Path(__file__).parent

        # Current working directory for file operations
        self.current_dir = os.getcwd()

        # Setup main interface
        self.setup_main_interface()

    def setup_main_interface(self):
        """Create the main interface with 5 main options"""

        # Main container with proper padding
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Title section
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        title_label = create_label(title_frame, text="DynamiXs - NMR Relaxation Analysis Suite",
                                   font=FONT_LARGE_HEADER)
        title_label.pack(anchor=tk.W)

        subtitle_label = create_label(title_frame, text="Choose your analysis type:",
                                      font=FONT_BODY)
        subtitle_label.configure(text_color=SECONDARY_TEXT)
        subtitle_label.pack(anchor=tk.W, pady=(SPACING_XS, 0))

        # Main options frame with rounded corners
        options_frame = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR,
                                    corner_radius=FRAME_CORNER_RADIUS)
        options_frame.pack(fill=tk.BOTH, expand=True, pady=(0, SPACING_MD))

        # Inner padding for options
        options_inner = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_inner.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Option 1: Model Free Analysis
        btn_integrated = create_primary_button(options_inner, text="Model Free Analysis",
                                              command=self.show_integrated_analysis, width=200)
        btn_integrated.configure(font=("SF Pro", 18, "bold"))
        btn_integrated.pack(pady=(0, SPACING_SM))

        # Dotted separator line
        separator_canvas = tk.Canvas(options_inner, height=2, bg=PANEL_BG_COLOR, highlightthickness=0)
        separator_canvas.pack(pady=SPACING_SM, fill=tk.X)
        # Draw dotted line across the canvas width (will match button width automatically)
        def draw_dotted_line(event=None):
            separator_canvas.delete("all")
            width = separator_canvas.winfo_width()
            # Draw dotted line with 5px dash, 3px gap
            for x in range(0, width, 8):
                separator_canvas.create_line(x, 1, min(x+5, width), 1, fill=SEPARATOR_COLOR, width=2)
        separator_canvas.bind("<Configure>", draw_dotted_line)
        separator_canvas.after(100, draw_dotted_line)

        # Option 2: T1/T2 Fitting
        btn_t1_t2 = create_primary_button(options_inner, text="T1/T2 Fitting Analysis",
                                         command=self.show_t1_t2_menu, width=200)
        btn_t1_t2.configure(font=("SF Pro", 18, "bold"))
        btn_t1_t2.pack(pady=(0, SPACING_SM))

        # Option 3: Spectral Density Analysis
        btn_spectral = create_primary_button(options_inner, text="Spectral Density Analysis",
                                            command=self.show_spectral_density_menu, width=200)
        btn_spectral.configure(font=("SF Pro", 18, "bold"))
        btn_spectral.pack(pady=(0, SPACING_SM))

        # Option 4: Plot Data
        btn_plot = create_secondary_button(options_inner, text="Plot Data",
                                          command=self.show_plot_menu, width=200)
        btn_plot.configure(font=("SF Pro", 18, "bold"))
        btn_plot.pack()

        # Working directory info
        dir_frame = ctk.CTkFrame(main_container, fg_color=FRAME_BG_COLOR,
                                corner_radius=BUTTON_CORNER_RADIUS,
                                border_width=1, border_color=BORDER_COLOR)
        dir_frame.pack(fill=tk.X)

        dir_inner = ctk.CTkFrame(dir_frame, fg_color="transparent")
        dir_inner.pack(fill=tk.X, padx=SPACING_MD, pady=SPACING_SM)

        dir_label = create_label(dir_inner, text=f"Working Directory: {self.current_dir}",
                                font=FONT_SMALL)
        dir_label.configure(text_color=SECONDARY_TEXT)
        dir_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        change_dir_btn = create_secondary_button(dir_inner, text="Change Directory",
                                                command=self.change_working_directory, width=140)
        change_dir_btn.pack(side=tk.RIGHT, padx=(SPACING_SM, 0))

    def change_working_directory(self):
        """Allow user to change working directory"""
        new_dir = filedialog.askdirectory(title="Select Working Directory",
                                         initialdir=self.current_dir)
        if new_dir:
            self.current_dir = new_dir
            os.chdir(new_dir)
            self.clear_window()
            self.setup_main_interface()  # Refresh interface

    def show_format_data(self):
        """Show format data options (currently empty)"""
        messagebox.showinfo("Format Data",
                           "Format Data functionality is not yet implemented.\n"
                           "This will contain data formatting utilities.")

    def show_t1_t2_menu(self):
        """Show T1/T2 fitting menu"""
        self.clear_window()
        T1T2FittingGUI(self.root, self)

    def show_spectral_density_menu(self):
        """Show spectral density analysis menu"""
        self.clear_window()
        SpectralDensityGUI(self.root, self)

    def show_plot_menu(self):
        """Show plotting menu"""
        self.clear_window()
        PlottingGUI(self.root, self)

    def show_cpmg_analysis(self):
        """Show CPMG analysis options (currently empty)"""
        messagebox.showinfo("CPMG Analysis",
                           "CPMG Analysis functionality is not yet implemented.\n"
                           "This will contain relaxation dispersion analysis.")

    def show_integrated_analysis(self):
        """Show integrated spectral density analysis workflow"""
        self.clear_window()
        IntegratedAnalysisGUI(self.root, self)

    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()

    def go_back_to_main(self):
        """Return to main menu"""
        self.clear_window()
        self.setup_main_interface()


class T1T2FittingGUI:
    def __init__(self, root, parent_gui):
        self.root = root
        self.parent_gui = parent_gui
        self.input_file = None
        self.output_prefix = ""
        self.experiment_type = "T1"

        self.setup_interface()

    def setup_interface(self):
        """Setup T1/T2 fitting interface"""

        # Main container with proper padding
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Header section
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_inner.pack(fill=tk.X)

        title_label = create_label(header_inner, text="T1/T2 Fitting Analysis",
                                   font=FONT_LARGE_HEADER)
        title_label.pack(side=tk.LEFT)

        back_btn = create_secondary_button(header_inner, text="← Back to Main",
                                          command=self.parent_gui.go_back_to_main,
                                          width=120)
        back_btn.pack(side=tk.RIGHT)

        # Content frame
        content_frame = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR,
                                    corner_radius=FRAME_CORNER_RADIUS)
        content_frame.pack(fill=tk.BOTH, expand=True)

        content_inner = ctk.CTkFrame(content_frame, fg_color="transparent")
        content_inner.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Experiment type selection
        exp_frame = ctk.CTkFrame(content_inner, fg_color="transparent")
        exp_frame.pack(fill=tk.X, pady=(0, SPACING_SM))

        exp_label = create_label(exp_frame, text="Select Experiment Type:", font=FONT_BODY)
        exp_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.exp_var = tk.StringVar(value="T1")

        radio_frame = ctk.CTkFrame(exp_frame, fg_color="transparent")
        radio_frame.pack(side=tk.LEFT)

        t1_radio = ctk.CTkRadioButton(radio_frame, text="T1", variable=self.exp_var,
                                     value="T1", command=self.update_experiment_type,
                                     font=FONT_BODY, text_color=PRIMARY_TEXT,
                                     fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        t1_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        t2_radio = ctk.CTkRadioButton(radio_frame, text="T2", variable=self.exp_var,
                                     value="T2", command=self.update_experiment_type,
                                     font=FONT_BODY, text_color=PRIMARY_TEXT,
                                     fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        t2_radio.pack(side=tk.LEFT)

        # File import section
        file_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        file_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        file_label = create_label(file_section, text="Import Data File:", font=FONT_BODY)
        file_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        file_row = ctk.CTkFrame(file_section, fg_color="transparent")
        file_row.pack(fill=tk.X)

        self.file_var = tk.StringVar(value="No file selected")
        file_display = create_label(file_row, text="", font=FONT_SMALL)
        file_display.configure(textvariable=self.file_var, text_color=SECONDARY_TEXT)
        file_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACING_SM))

        browse_btn = create_secondary_button(file_row, text="Browse",
                                            command=self.browse_input_file, width=80)
        browse_btn.pack(side=tk.RIGHT)

        # Results prefix
        prefix_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        prefix_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        prefix_label = create_label(prefix_section, text="Results Prefix:", font=FONT_BODY)
        prefix_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.prefix_var = tk.StringVar(value="T1_analysis")
        prefix_entry = create_entry(prefix_section, textvariable=self.prefix_var, width=400)
        prefix_entry.pack(fill=tk.X)

        # Processing mode
        mode_frame = ctk.CTkFrame(content_inner, fg_color="transparent")
        mode_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        mode_label = create_label(mode_frame, text="Processing Mode:", font=FONT_BODY)
        mode_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.mode_var = tk.StringVar(value="single")

        radio_mode_frame = ctk.CTkFrame(mode_frame, fg_color="transparent")
        radio_mode_frame.pack(side=tk.LEFT)

        single_radio = ctk.CTkRadioButton(radio_mode_frame, text="Single Core",
                                         variable=self.mode_var, value="single",
                                         font=FONT_BODY, text_color=PRIMARY_TEXT,
                                         fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        single_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        multi_radio = ctk.CTkRadioButton(radio_mode_frame, text="Multi Core",
                                        variable=self.mode_var, value="multi",
                                        font=FONT_BODY, text_color=PRIMARY_TEXT,
                                        fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        multi_radio.pack(side=tk.LEFT)

        # Advanced parameters
        advanced_frame = CTkLabelFrame(content_inner, text="Advanced Parameters", padding=SPACING_MD)
        advanced_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Initial Amplitude
        amp_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        amp_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        amp_label = create_label(amp_row, text="Initial Amplitude:", font=FONT_BODY)
        amp_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.initial_A_var = tk.StringVar(value="5")
        amp_entry = create_entry(amp_row, textvariable=self.initial_A_var, width=100)
        amp_entry.pack(side=tk.LEFT)

        # Initial Time Constant
        time_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        time_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        time_label = create_label(time_row, text="Initial Time Constant:", font=FONT_BODY)
        time_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.initial_t2_var = tk.StringVar(value="100")
        time_entry = create_entry(time_row, textvariable=self.initial_t2_var, width=100)
        time_entry.pack(side=tk.LEFT)

        # Bootstrap Iterations
        boot_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        boot_row.pack(fill=tk.X)

        boot_label = create_label(boot_row, text="Bootstrap Iterations:", font=FONT_BODY)
        boot_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.bootstrap_var = tk.StringVar(value="1000")
        boot_entry = create_entry(boot_row, textvariable=self.bootstrap_var, width=100)
        boot_entry.pack(side=tk.LEFT)

        # Start analysis button
        start_btn = create_primary_button(content_inner, text="Start Analysis",
                                         command=self.start_analysis, width=200)
        start_btn.pack(pady=(0, SPACING_MD))

        # Results display area
        results_frame = CTkLabelFrame(content_inner, text="Analysis Results", padding=SPACING_MD)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, SPACING_SM))

        self.results_text = ctk.CTkTextbox(results_frame.content, height=200,
                                          font=FONT_MONO, fg_color=FRAME_BG_COLOR,
                                          text_color=PRIMARY_TEXT, wrap="word")
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=(0, SPACING_SM))

        # Display results button
        display_btn = create_secondary_button(results_frame.content, text="Display Results",
                                             command=self.display_results, width=140)
        display_btn.pack()

    def update_experiment_type(self):
        """Update experiment type and default prefix"""
        self.experiment_type = self.exp_var.get()
        self.prefix_var.set(f"{self.experiment_type}_analysis")

    def browse_input_file(self):
        """Browse for input CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            self.input_file = filename
            self.file_var.set(os.path.basename(filename))

    def start_analysis(self):
        """Start the T1/T2 analysis"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first.")
            return

        # Clear results
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Starting analysis...\n")
        self.root.update()

        # Run analysis in separate thread
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()

    def _run_analysis(self):
        """Run the analysis (called in separate thread)"""
        try:
            # Import the fitting script functions
            script_path = self.parent_gui.dynamixs_path / "dynamiXs_T1_T2"
            sys.path.insert(0, str(script_path))

            if self.mode_var.get() == "single":
                from fit_Tx_NMRRE import run_analysis_with_params
            else:
                from fitmulti__Tx_NMRRE import run_analysis_with_params

            # Prepare parameters
            params = {
                'input_csv_file': self.input_file,
                'output_prefix': self.prefix_var.get(),
                'results_txt_file': f"{self.prefix_var.get()}_fit_results.txt",
                'experiment_type': self.experiment_type,
                'time_units': "ms",
                'signal_units': "Intensity",
                'initial_A': float(self.initial_A_var.get()),
                'initial_t2': float(self.initial_t2_var.get()),
                'n_bootstrap': int(self.bootstrap_var.get()),
                'n_plots_per_figure': 20
            }

            # Run analysis
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Running fitting analysis...\n"))

            results = run_analysis_with_params(params)

            # Update GUI with results
            self.root.after(0, lambda: self._display_analysis_results(results))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}\n"
            self.root.after(0, lambda: self.results_text.insert(tk.END, error_msg))

    def _display_analysis_results(self, results):
        """Display analysis results in the GUI"""
        self.results_text.insert(tk.END, "\nAnalysis completed successfully!\n")
        self.results_text.insert(tk.END, f"Number of residues fitted: {results.get('n_fitted', 'N/A')}\n")
        self.results_text.insert(tk.END, f"Results saved to: {results.get('results_file', 'N/A')}\n")
        if 'plots_prefix' in results:
            self.results_text.insert(tk.END, f"Plots saved with prefix: {results['plots_prefix']}\n")
        self.results_text.see(tk.END)

    def display_results(self):
        """Display saved results"""
        results_file = f"{self.prefix_var.get()}_fit_results.txt"
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    content = f.read()

                # Create new window for results
                results_window = ctk.CTkToplevel(self.root)
                results_window.title(f"Results: {results_file}")
                results_window.geometry("800x600")
                results_window.configure(fg_color=BG_COLOR)

                # Main container
                container = ctk.CTkFrame(results_window, fg_color=BG_COLOR, corner_radius=0)
                container.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

                text_widget = ctk.CTkTextbox(container, wrap="word", font=FONT_MONO,
                                           fg_color=FRAME_BG_COLOR, text_color=PRIMARY_TEXT)
                text_widget.pack(fill=tk.BOTH, expand=True)
                text_widget.insert("1.0", content)
                text_widget.configure(state='disabled')

            except Exception as e:
                messagebox.showerror("Error", f"Error reading results file: {str(e)}")
        else:
            messagebox.showwarning("Warning", f"Results file not found: {results_file}")


class SpectralDensityGUI:
    def __init__(self, root, parent_gui):
        self.root = root
        self.parent_gui = parent_gui
        self.input_file1 = None
        self.input_file2 = None
        self.setup_interface()

    def setup_interface(self):
        """Setup spectral density interface"""

        # Main container with proper padding
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Header section
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_inner.pack(fill=tk.X)

        title_label = create_label(header_inner, text="Spectral Density Analysis",
                                   font=FONT_LARGE_HEADER)
        title_label.pack(side=tk.LEFT)

        back_btn = create_secondary_button(header_inner, text="← Back to Main",
                                          command=self.parent_gui.go_back_to_main,
                                          width=120)
        back_btn.pack(side=tk.RIGHT)

        # Scrollable content frame
        content_frame = ctk.CTkScrollableFrame(main_container, fg_color=PANEL_BG_COLOR,
                                              corner_radius=FRAME_CORNER_RADIUS)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Content goes directly into scrollable frame (no need for extra wrapper)
        content_inner = content_frame

        # Analysis type selection
        analysis_frame = ctk.CTkFrame(content_inner, fg_color="transparent")
        analysis_frame.pack(fill=tk.X, pady=(0, SPACING_SM))

        analysis_label = create_label(analysis_frame, text="Analysis Type:", font=FONT_BODY)
        analysis_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.analysis_var = tk.StringVar(value="single_field")

        radio_analysis_frame = ctk.CTkFrame(analysis_frame, fg_color="transparent")
        radio_analysis_frame.pack(side=tk.LEFT)

        single_radio = ctk.CTkRadioButton(radio_analysis_frame, text="Single Field",
                                         variable=self.analysis_var, value="single_field",
                                         command=self.update_analysis_type,
                                         font=FONT_BODY, text_color=PRIMARY_TEXT,
                                         fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        single_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        dual_radio = ctk.CTkRadioButton(radio_analysis_frame, text="Dual Field",
                                       variable=self.analysis_var, value="dual_field",
                                       command=self.update_analysis_type,
                                       font=FONT_BODY, text_color=PRIMARY_TEXT,
                                       fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        dual_radio.pack(side=tk.LEFT)

        # Spectral density method selection
        method_frame = ctk.CTkFrame(content_inner, fg_color="transparent")
        method_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        method_label = create_label(method_frame, text="Spectral Density Method:", font=FONT_BODY)
        method_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.spectral_method_var = tk.StringVar(value="jwh")

        radio_method_frame = ctk.CTkFrame(method_frame, fg_color="transparent")
        radio_method_frame.pack(side=tk.LEFT)

        jwh_radio = ctk.CTkRadioButton(radio_method_frame, text="J(ωH)",
                                      variable=self.spectral_method_var, value="jwh",
                                      font=FONT_BODY, text_color=PRIMARY_TEXT,
                                      fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        jwh_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        j087_radio = ctk.CTkRadioButton(radio_method_frame, text="J(0.87ωH)",
                                       variable=self.spectral_method_var, value="j087wh",
                                       font=FONT_BODY, text_color=PRIMARY_TEXT,
                                       fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        j087_radio.pack(side=tk.LEFT)

        # Field parameters
        field_frame = CTkLabelFrame(content_inner, text="Field Parameters", padding=SPACING_MD)
        field_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Field 1
        field1_row = ctk.CTkFrame(field_frame.content, fg_color="transparent")
        field1_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        field1_label = create_label(field1_row, text="Field 1 Frequency (MHz):", font=FONT_BODY)
        field1_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.field1_var = tk.StringVar(value="600.0")
        field1_entry = create_entry(field1_row, textvariable=self.field1_var, width=100)
        field1_entry.pack(side=tk.LEFT)

        # Field 2
        field2_row = ctk.CTkFrame(field_frame.content, fg_color="transparent")
        field2_row.pack(fill=tk.X)

        field2_label = create_label(field2_row, text="Field 2 Frequency (MHz):", font=FONT_BODY)
        field2_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.field2_var = tk.StringVar(value="700.0")
        self.field2_entry = create_entry(field2_row, textvariable=self.field2_var, width=100)
        self.field2_entry.configure(state="disabled")
        self.field2_entry.pack(side=tk.LEFT)

        # File import section
        files_frame = CTkLabelFrame(content_inner, text="Data Import", padding=SPACING_MD)
        files_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # File 1
        file1_label = create_label(files_frame.content, text="Data File 1:", font=FONT_BODY)
        file1_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        file1_row = ctk.CTkFrame(files_frame.content, fg_color="transparent")
        file1_row.pack(fill=tk.X, pady=(0, SPACING_SM))

        self.file1_var = tk.StringVar(value="No file selected")
        file1_display = create_label(file1_row, text="", font=FONT_SMALL)
        file1_display.configure(textvariable=self.file1_var, text_color=SECONDARY_TEXT)
        file1_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACING_SM))

        browse1_btn = create_secondary_button(file1_row, text="Browse",
                                             command=self.browse_file1, width=80)
        browse1_btn.pack(side=tk.RIGHT)

        # File 2
        file2_label = create_label(files_frame.content, text="Data File 2:", font=FONT_BODY)
        file2_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        file2_row = ctk.CTkFrame(files_frame.content, fg_color="transparent")
        file2_row.pack(fill=tk.X)

        self.file2_var = tk.StringVar(value="Not required for single field")
        self.file2_display = create_label(file2_row, text="", font=FONT_SMALL)
        self.file2_display.configure(textvariable=self.file2_var, text_color=SECONDARY_TEXT)
        self.file2_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACING_SM))

        self.browse2_btn = create_secondary_button(file2_row, text="Browse",
                                                   command=self.browse_file2, width=80)
        self.browse2_btn.configure(state="disabled")
        self.browse2_btn.pack(side=tk.RIGHT)

        # Advanced parameters
        advanced_frame = CTkLabelFrame(content_inner, text="Advanced Parameters", padding=SPACING_MD)
        advanced_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # N-H Bond Length
        rnh_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        rnh_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        rnh_label = create_label(rnh_row, text="N-H Bond Length (Å):", font=FONT_BODY)
        rnh_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.rnh_var = tk.StringVar(value="1.015")
        rnh_entry = create_entry(rnh_row, textvariable=self.rnh_var, width=100)
        rnh_entry.pack(side=tk.LEFT)

        # 15N CSA
        csa_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        csa_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        csa_label = create_label(csa_row, text="15N CSA (ppm):", font=FONT_BODY)
        csa_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.csa_var = tk.StringVar(value="-160.0")
        csa_entry = create_entry(csa_row, textvariable=self.csa_var, width=100)
        csa_entry.pack(side=tk.LEFT)

        # Monte Carlo checkbox
        mc_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        mc_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        self.monte_carlo_var = tk.BooleanVar(value=False)
        mc_check = ctk.CTkCheckBox(mc_row, text="Use Monte Carlo Errors",
                                   variable=self.monte_carlo_var,
                                   font=FONT_BODY, text_color=PRIMARY_TEXT,
                                   fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        mc_check.pack(side=tk.LEFT)

        # Monte Carlo Samples
        mc_samples_row = ctk.CTkFrame(advanced_frame.content, fg_color="transparent")
        mc_samples_row.pack(fill=tk.X)

        mc_samples_label = create_label(mc_samples_row, text="Monte Carlo Samples:", font=FONT_BODY)
        mc_samples_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.mc_samples_var = tk.StringVar(value="50")
        mc_samples_entry = create_entry(mc_samples_row, textvariable=self.mc_samples_var, width=100)
        mc_samples_entry.pack(side=tk.LEFT)

        # Output prefix
        prefix_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        prefix_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        prefix_label = create_label(prefix_section, text="Output Prefix:", font=FONT_BODY)
        prefix_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.prefix_var = tk.StringVar(value="spectral_density_analysis")
        prefix_entry = create_entry(prefix_section, textvariable=self.prefix_var, width=400)
        prefix_entry.pack(fill=tk.X)

        # Start analysis button
        start_btn = create_primary_button(content_inner, text="Start Analysis",
                                         command=self.start_analysis, width=200)
        start_btn.pack(pady=(0, SPACING_MD))

        # Results display area
        results_frame = CTkLabelFrame(content_inner, text="Analysis Results", padding=SPACING_MD)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, SPACING_SM))

        self.results_text = ctk.CTkTextbox(results_frame.content, height=200,
                                          font=FONT_MONO, fg_color=FRAME_BG_COLOR,
                                          text_color=PRIMARY_TEXT, wrap="word")
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=(0, SPACING_SM))

        # Display results button
        display_btn = create_secondary_button(results_frame.content, text="Display Results",
                                             command=self.display_results, width=140)
        display_btn.pack()

    def update_analysis_type(self):
        """Update interface based on analysis type selection"""
        if self.analysis_var.get() == "dual_field":
            # Enable dual field options
            self.field2_entry.configure(state="normal")
            self.browse2_btn.configure(state="normal")
            self.file2_var.set("No file selected")
        else:
            # Disable dual field options
            self.field2_entry.configure(state="disabled")
            self.browse2_btn.configure(state="disabled")
            self.file2_var.set("Not required for single field")

    def browse_file1(self):
        """Browse for first data file"""
        filename = filedialog.askopenfilename(
            title="Select First Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            self.input_file1 = filename
            self.file1_var.set(os.path.basename(filename))

    def browse_file2(self):
        """Browse for second data file"""
        filename = filedialog.askopenfilename(
            title="Select Second Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            self.input_file2 = filename
            self.file2_var.set(os.path.basename(filename))

    def start_analysis(self):
        """Start the spectral density analysis"""
        if not self.input_file1:
            messagebox.showerror("Error", "Please select at least one input file.")
            return

        if self.analysis_var.get() == "dual_field" and not self.input_file2:
            messagebox.showerror("Error", "Please select both input files for dual field analysis.")
            return

        # Clear results
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Starting spectral density analysis...\n")
        self.root.update()

        # Run analysis in separate thread
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()

    def _run_analysis(self):
        """Run the analysis (called in separate thread)"""
        try:
            # Import the appropriate spectral density script based on analysis type and method
            script_path = self.parent_gui.dynamixs_path / "dynamiXs_density_functions"
            sys.path.insert(0, str(script_path))

            # Determine which script to use (prefer multicore versions)
            analysis_type = self.analysis_var.get()
            spectral_method = self.spectral_method_var.get()

            # Map to script names and class names
            if analysis_type == "dual_field":
                if spectral_method == "j087wh":
                    # Try multicore first, fallback to single-core
                    try:
                        from ZZ_multi_2fields_density_087 import DualFieldSpectralDensityAnalysis
                        script_name = "ZZ_multi_2fields_density_087 (multicore)"
                    except ImportError:
                        from ZZ_2fields_density_087 import DualFieldSpectralDensityAnalysis
                        script_name = "ZZ_2fields_density_087 (single-core)"
                else:  # jwh
                    try:
                        from ZZ_multi_2fields_density import DualFieldSpectralDensityAnalysis
                        script_name = "ZZ_multi_2fields_density (multicore)"
                    except ImportError:
                        from ZZ_2fields_density import DualFieldSpectralDensityAnalysis
                        script_name = "ZZ_2fields_density (single-core)"
            else:  # single_field
                if spectral_method == "j087wh":
                    try:
                        from ZZ_multi_density_087 import ReducedSpectralDensityAnalysis
                        script_name = "ZZ_multi_density_087 (multicore)"
                    except ImportError:
                        from ZZ_density import ReducedSpectralDensityAnalysis
                        script_name = "ZZ_density (single-core)"
                else:  # jwh
                    try:
                        from ZZ_multi_density import ReducedSpectralDensityAnalysis
                        script_name = "ZZ_multi_density (multicore)"
                    except ImportError:
                        from ZZ_density import ReducedSpectralDensityAnalysis
                        script_name = "ZZ_density (single-core)"

            self.root.after(0, lambda: self.results_text.insert(tk.END, f"Using: {script_name}\n"))
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Running spectral density analysis...\n"))

            # Convert parameters
            rNH_meters = float(self.rnh_var.get()) * 1e-10  # Å to meters
            csaN_units = float(self.csa_var.get()) * 1e-6   # ppm to proper units

            # Create analyzer instance and run analysis
            if analysis_type == "dual_field":
                # Dual-field analysis
                field1_freq = float(self.field1_var.get())
                field2_freq = float(self.field2_var.get())

                analyzer = DualFieldSpectralDensityAnalysis(
                    field1_freq=field1_freq,
                    field2_freq=field2_freq,
                    rNH=rNH_meters,
                    csaN=csaN_units
                )

                results_df = analyzer.analyze_dual_field_csv(
                    csv_file1=self.input_file1,
                    csv_file2=self.input_file2,
                    use_monte_carlo_errors=self.monte_carlo_var.get(),
                    n_monte_carlo=int(self.mc_samples_var.get()),
                    use_multiprocessing=True if 'multi' in script_name else False,
                    n_processes=None
                )

                # Save results
                output_prefix = self.prefix_var.get()
                basic_csv = f"{output_prefix}_spectral_density_basic.csv"
                detailed_csv = f"{output_prefix}_spectral_density_detailed.csv"
                plots_pdf = f"{output_prefix}_spectral_density_plots.pdf"

                results_df.to_csv(basic_csv, index=False)
                analyzer.save_dual_field_results(results_df, detailed_csv)
                analyzer.plot_dual_field_results(results_df, save_plots=True, plot_filename=plots_pdf)

                # Prepare results summary
                results = {
                    'success': True,
                    'analysis_type': 'dual_field',
                    'n_processed': len(results_df),
                    'results_file': basic_csv,
                    'field1_freq': field1_freq,
                    'field2_freq': field2_freq,
                    'plots_prefix': output_prefix
                }

                # Calculate means if data available
                if len(results_df) > 0:
                    results['mean_J0_f1'] = results_df['J0_f1'].mean() if 'J0_f1' in results_df.columns else 0
                    results['mean_J0_f2'] = results_df['J0_f2'].mean() if 'J0_f2' in results_df.columns else 0
                    results['mean_JwN_f1'] = results_df['JwN_f1'].mean() if 'JwN_f1' in results_df.columns else 0
                    results['mean_JwN_f2'] = results_df['JwN_f2'].mean() if 'JwN_f2' in results_df.columns else 0

            else:
                # Single-field analysis
                field_freq = float(self.field1_var.get())

                analyzer = ReducedSpectralDensityAnalysis(
                    field_freq=field_freq,
                    rNH=rNH_meters,
                    csaN=csaN_units
                )

                results_df = analyzer.analyze_csv(
                    csv_file=self.input_file1,
                    use_monte_carlo_errors=self.monte_carlo_var.get(),
                    n_monte_carlo=int(self.mc_samples_var.get()),
                    use_multiprocessing=True if 'multi' in script_name else False,
                    n_processes=None
                )

                # Save results
                output_prefix = self.prefix_var.get()
                basic_csv = f"{output_prefix}_spectral_density_basic.csv"
                detailed_csv = f"{output_prefix}_spectral_density_detailed.csv"
                plots_pdf = f"{output_prefix}_spectral_density_plots.pdf"

                results_df.to_csv(basic_csv, index=False)
                analyzer.save_results(results_df, detailed_csv)
                analyzer.plot_results(results_df, save_plots=True, plot_filename=plots_pdf)

                # Prepare results summary
                results = {
                    'success': True,
                    'analysis_type': 'single_field',
                    'n_processed': len(results_df),
                    'results_file': basic_csv,
                    'field_freq': field_freq,
                    'plots_prefix': output_prefix
                }

                # Calculate means if data available
                if len(results_df) > 0:
                    results['mean_J0'] = results_df['J0'].mean() if 'J0' in results_df.columns else 0
                    results['mean_JwN'] = results_df['JwN'].mean() if 'JwN' in results_df.columns else 0
                    results['mean_JwH'] = results_df['JwH'].mean() if 'JwH' in results_df.columns else 0

            # Update GUI with results
            self.root.after(0, lambda: self._display_analysis_results(results))

        except Exception as e:
            import traceback
            error_msg = f"Error during analysis: {str(e)}\n"
            error_msg += traceback.format_exc()
            self.root.after(0, lambda: self.results_text.insert(tk.END, error_msg))

    def _display_analysis_results(self, results):
        """Display analysis results in the GUI"""
        if results.get('success', False):
            self.results_text.insert(tk.END, "\nSpectral density analysis completed successfully!\n")
            self.results_text.insert(tk.END, f"Analysis type: {results.get('analysis_type', 'Unknown')}\n")
            self.results_text.insert(tk.END, f"Number of residues processed: {results.get('n_processed', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Results saved to: {results.get('results_file', 'N/A')}\n")

            if results['analysis_type'] == 'dual_field':
                self.results_text.insert(tk.END, f"Field 1: {results.get('field1_freq', 'N/A')} MHz\n")
                self.results_text.insert(tk.END, f"Field 2: {results.get('field2_freq', 'N/A')} MHz\n")
                self.results_text.insert(tk.END, f"Mean J(0) Field 1: {results.get('mean_J0_f1', 'N/A'):.3e} ns/rad²\n")
                self.results_text.insert(tk.END, f"Mean J(0) Field 2: {results.get('mean_J0_f2', 'N/A'):.3e} ns/rad²\n")
                self.results_text.insert(tk.END, f"Mean J(ωN) Field 1: {results.get('mean_JwN_f1', 'N/A'):.3e} ns/rad²\n")
                self.results_text.insert(tk.END, f"Mean J(ωN) Field 2: {results.get('mean_JwN_f2', 'N/A'):.3e} ns/rad²\n")
            else:
                self.results_text.insert(tk.END, f"Field: {results.get('field_freq', 'N/A')} MHz\n")
                self.results_text.insert(tk.END, f"Mean J(0): {results.get('mean_J0', 'N/A'):.3e} ns/rad²\n")
                self.results_text.insert(tk.END, f"Mean J(ωN): {results.get('mean_JwN', 'N/A'):.3e} ns/rad²\n")
                self.results_text.insert(tk.END, f"Mean J(ωH): {results.get('mean_JwH', 'N/A'):.3e} ns/rad²\n")

            if 'plots_prefix' in results:
                self.results_text.insert(tk.END, f"Plots saved with prefix: {results['plots_prefix']}\n")
        else:
            self.results_text.insert(tk.END, f"\nAnalysis failed: {results.get('error', 'Unknown error')}\n")

        self.results_text.see(tk.END)

    def display_results(self):
        """Display saved results"""
        results_file = f"{self.prefix_var.get()}_results.csv"
        if os.path.exists(results_file):
            try:
                # Load and display results
                results_df = pd.read_csv(results_file)

                # Create new window for results
                results_window = ctk.CTkToplevel(self.root)
                results_window.title(f"Spectral Density Results: {results_file}")
                results_window.geometry("1000x600")
                results_window.configure(fg_color=BG_COLOR)

                # Main container
                container = ctk.CTkFrame(results_window, fg_color=BG_COLOR, corner_radius=0)
                container.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

                # Create treeview for data display (using ttk since CustomTkinter doesn't have Treeview)
                tree_frame = ctk.CTkFrame(container, fg_color=FRAME_BG_COLOR,
                                        corner_radius=FRAME_CORNER_RADIUS)
                tree_frame.pack(fill=tk.BOTH, expand=True)

                tree = ttk.Treeview(tree_frame)
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=SPACING_SM, pady=SPACING_SM)

                # Configure columns
                columns = list(results_df.columns)
                tree['columns'] = columns
                tree['show'] = 'headings'

                # Set column headings and widths
                for col in columns:
                    tree.heading(col, text=col)
                    tree.column(col, width=100, anchor='center')

                # Insert data
                for _, row in results_df.iterrows():
                    values = [f"{val:.3f}" if isinstance(val, (int, float)) and not pd.isna(val) else str(val)
                             for val in row]
                    tree.insert('', 'end', values=values)

                # Add scrollbar
                scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=SPACING_SM)
                tree.configure(yscrollcommand=scrollbar.set)

            except Exception as e:
                messagebox.showerror("Error", f"Error reading results file: {str(e)}")
        else:
            messagebox.showwarning("Warning", f"Results file not found: {results_file}")


class PlottingGUI:
    def __init__(self, root, parent_gui):
        self.root = root
        self.parent_gui = parent_gui
        self.setup_interface()

    def setup_interface(self):
        """Setup plotting interface"""

        # Main container with proper padding
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Header section
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_inner.pack(fill=tk.X)

        title_label = create_label(header_inner, text="Plot Data",
                                   font=FONT_LARGE_HEADER)
        title_label.pack(side=tk.LEFT)

        back_btn = create_secondary_button(header_inner, text="← Back to Main",
                                          command=self.parent_gui.go_back_to_main,
                                          width=120)
        back_btn.pack(side=tk.RIGHT)

        # Content frame
        content_frame = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR,
                                    corner_radius=FRAME_CORNER_RADIUS)
        content_frame.pack(fill=tk.BOTH, expand=True)

        content_inner = ctk.CTkFrame(content_frame, fg_color="transparent")
        content_inner.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Spectral density method selection
        method_frame = ctk.CTkFrame(content_inner, fg_color="transparent")
        method_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        method_label = create_label(method_frame, text="Spectral Density Method:", font=FONT_BODY)
        method_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.spectral_method_var = tk.StringVar(value="jwh")

        radio_method_frame = ctk.CTkFrame(method_frame, fg_color="transparent")
        radio_method_frame.pack(side=tk.LEFT)

        jwh_radio = ctk.CTkRadioButton(radio_method_frame, text="J(ωH)",
                                      variable=self.spectral_method_var, value="jwh",
                                      font=FONT_BODY, text_color=PRIMARY_TEXT,
                                      fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        jwh_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        j087_radio = ctk.CTkRadioButton(radio_method_frame, text="J(0.87ωH)",
                                       variable=self.spectral_method_var, value="j087wh",
                                       font=FONT_BODY, text_color=PRIMARY_TEXT,
                                       fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        j087_radio.pack(side=tk.LEFT)

        # Plot options
        dual_btn = create_primary_button(content_inner, text="Plot Dual Field Data",
                                        command=self.show_dual_field_plot, width=300)
        dual_btn.pack(pady=(0, SPACING_SM))

        compare_btn = create_primary_button(content_inner, text="Compare Datasets",
                                           command=self.show_compare_datasets, width=300)
        compare_btn.pack()

    def show_dual_field_plot(self):
        """Show dual field plotting interface"""
        self.parent_gui.clear_window()
        DualFieldPlottingGUI(self.root, self.parent_gui, self.spectral_method_var.get())

    def show_compare_datasets(self):
        """Show dataset comparison interface"""
        self.parent_gui.clear_window()
        CompareDatasetGUI(self.root, self.parent_gui, self.spectral_method_var.get())


class DualFieldPlottingGUI:
    def __init__(self, root, parent_gui, spectral_method="jwh"):
        self.root = root
        self.parent_gui = parent_gui
        self.spectral_method = spectral_method
        self.input_file = None
        self.setup_interface()

    def setup_interface(self):
        """Setup dual field plotting interface"""

        # Main container with proper padding
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Header section
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_inner.pack(fill=tk.X)

        method_text = "J(0.87ωH)" if self.spectral_method == "j087wh" else "J(ωH)"
        title_text = f"Plot Dual Field Data ({method_text})"
        title_label = create_label(header_inner, text=title_text,
                                   font=FONT_LARGE_HEADER)
        title_label.pack(side=tk.LEFT)

        back_btn = create_secondary_button(header_inner, text="← Back to Plot Menu",
                                          command=self.go_back_to_plot_menu,
                                          width=140)
        back_btn.pack(side=tk.RIGHT)

        # Content frame
        content_frame = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR,
                                    corner_radius=FRAME_CORNER_RADIUS)
        content_frame.pack(fill=tk.BOTH, expand=True)

        content_inner = ctk.CTkFrame(content_frame, fg_color="transparent")
        content_inner.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # File import section
        file_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        file_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        file_label = create_label(file_section, text="Import Data File:", font=FONT_BODY)
        file_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        file_row = ctk.CTkFrame(file_section, fg_color="transparent")
        file_row.pack(fill=tk.X)

        self.file_var = tk.StringVar(value="No file selected")
        file_display = create_label(file_row, text="", font=FONT_SMALL)
        file_display.configure(textvariable=self.file_var, text_color=SECONDARY_TEXT)
        file_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACING_SM))

        browse_btn = create_secondary_button(file_row, text="Browse",
                                            command=self.browse_input_file, width=80)
        browse_btn.pack(side=tk.RIGHT)

        # Output prefix
        prefix_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        prefix_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        prefix_label = create_label(prefix_section, text="Output Prefix:", font=FONT_BODY)
        prefix_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.prefix_var = tk.StringVar(value="dual_field_plot")
        prefix_entry = create_entry(prefix_section, textvariable=self.prefix_var, width=400)
        prefix_entry.pack(fill=tk.X)

        # Plot parameters
        plot_frame = CTkLabelFrame(content_inner, text="Plot Parameters", padding=SPACING_MD)
        plot_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Dataset Label
        label_row = ctk.CTkFrame(plot_frame.content, fg_color="transparent")
        label_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        label_label = create_label(label_row, text="Dataset Label:", font=FONT_BODY)
        label_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.label_var = tk.StringVar(value="Dataset")
        label_entry = create_entry(label_row, textvariable=self.label_var, width=200)
        label_entry.pack(side=tk.LEFT)

        # Figure Size
        size_row = ctk.CTkFrame(plot_frame.content, fg_color="transparent")
        size_row.pack(fill=tk.X, pady=(0, SPACING_XS))

        size_label = create_label(size_row, text="Figure Size (width x height):", font=FONT_BODY)
        size_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.width_var = tk.StringVar(value="12")
        width_entry = create_entry(size_row, textvariable=self.width_var, width=60)
        width_entry.pack(side=tk.LEFT)

        x_label = create_label(size_row, text=" x ", font=FONT_BODY)
        x_label.pack(side=tk.LEFT, padx=SPACING_XS)

        self.height_var = tk.StringVar(value="8")
        height_entry = create_entry(size_row, textvariable=self.height_var, width=60)
        height_entry.pack(side=tk.LEFT)

        # Include Secondary Structure checkbox
        ss_row = ctk.CTkFrame(plot_frame.content, fg_color="transparent")
        ss_row.pack(fill=tk.X)

        self.ss_var = tk.BooleanVar(value=True)
        ss_check = ctk.CTkCheckBox(ss_row, text="Include Secondary Structure",
                                   variable=self.ss_var,
                                   font=FONT_BODY, text_color=PRIMARY_TEXT,
                                   fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        ss_check.pack(side=tk.LEFT)

        # Generate plot button
        plot_btn = create_primary_button(content_inner, text="Generate Plot",
                                        command=self.start_plotting, width=200)
        plot_btn.pack(pady=(0, SPACING_MD))

        # Results display area
        results_frame = CTkLabelFrame(content_inner, text="Plot Generation Results", padding=SPACING_MD)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = ctk.CTkTextbox(results_frame.content, height=150,
                                          font=FONT_MONO, fg_color=FRAME_BG_COLOR,
                                          text_color=PRIMARY_TEXT, wrap="word")
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def browse_input_file(self):
        """Browse for input CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            self.input_file = filename
            self.file_var.set(os.path.basename(filename))

    def start_plotting(self):
        """Start the dual field plotting"""
        if not self.input_file:
            messagebox.showerror("Error", "Please select an input file first.")
            return

        # Clear results
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Starting plot generation...\n")
        self.root.update()

        # Run plotting in separate thread
        thread = threading.Thread(target=self._run_plotting)
        thread.daemon = True
        thread.start()

    def _run_plotting(self):
        """Run the plotting (called in separate thread)"""
        try:
            # Import the appropriate plotting script based on spectral method
            script_path = self.parent_gui.dynamixs_path / "dynamiXs_plot"
            sys.path.insert(0, str(script_path))

            if self.spectral_method == "j087wh":
                from plot_dual_field_datasets_087 import plot_dual_field_data
            else:
                from plot_dual_field_datasets import plot_dual_field_data

            # Prepare parameters
            params = {
                'input_file': self.input_file,
                'output_prefix': self.prefix_var.get(),
                'dataset_label': self.label_var.get(),
                'figure_width': float(self.width_var.get()),
                'figure_height': float(self.height_var.get()),
                'include_ss': self.ss_var.get(),
                'spectral_method': self.spectral_method
            }

            # Run plotting
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Generating plots...\n"))

            # For now, display what would be plotted (until actual implementation is ready)
            self.root.after(0, lambda: self._display_plot_info())

        except Exception as e:
            error_msg = f"Error during plotting: {str(e)}\n"
            self.root.after(0, lambda: self.results_text.insert(tk.END, error_msg))

    def _display_plot_info(self):
        """Display plot information"""
        method_text = "J(0.87ωH)" if self.spectral_method == "j087wh" else "J(ωH)"
        self.results_text.insert(tk.END, "\nDual field plot would be generated with the following parameters:\n")
        self.results_text.insert(tk.END, f"Input file: {self.input_file}\n")
        self.results_text.insert(tk.END, f"Spectral method: {method_text}\n")
        self.results_text.insert(tk.END, f"Output prefix: {self.prefix_var.get()}\n")
        self.results_text.insert(tk.END, f"Dataset label: {self.label_var.get()}\n")
        self.results_text.insert(tk.END, f"Figure size: {self.width_var.get()} x {self.height_var.get()}\n")
        self.results_text.insert(tk.END, f"Include secondary structure: {self.ss_var.get()}\n")
        self.results_text.insert(tk.END, "\nNote: Full plotting functionality requires the plot_dual_field_datasets.py script to be properly integrated.\n")
        self.results_text.see(tk.END)

    def go_back_to_plot_menu(self):
        """Return to plotting menu"""
        self.parent_gui.clear_window()
        PlottingGUI(self.root, self.parent_gui)


class CompareDatasetGUI:
    def __init__(self, root, parent_gui, spectral_method="jwh"):
        self.root = root
        self.parent_gui = parent_gui
        self.spectral_method = spectral_method
        self.file1 = None
        self.file2 = None
        self.setup_interface()

    def setup_interface(self):
        """Setup dataset comparison interface"""

        # Main container with proper padding
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Header section
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        header_inner = ctk.CTkFrame(header_frame, fg_color="transparent")
        header_inner.pack(fill=tk.X)

        method_text = "J(0.87ωH)" if self.spectral_method == "j087wh" else "J(ωH)"
        title_text = f"Compare Datasets ({method_text})"
        title_label = create_label(header_inner, text=title_text,
                                   font=FONT_LARGE_HEADER)
        title_label.pack(side=tk.LEFT)

        back_btn = create_secondary_button(header_inner, text="← Back to Plot Menu",
                                          command=self.go_back_to_plot_menu,
                                          width=140)
        back_btn.pack(side=tk.RIGHT)

        # Content frame
        content_frame = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR,
                                    corner_radius=FRAME_CORNER_RADIUS)
        content_frame.pack(fill=tk.BOTH, expand=True)

        content_inner = ctk.CTkFrame(content_frame, fg_color="transparent")
        content_inner.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # File 1 selection
        file1_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        file1_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        file1_label = create_label(file1_section, text="Dataset 1 (Reference):", font=FONT_BODY)
        file1_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        file1_row = ctk.CTkFrame(file1_section, fg_color="transparent")
        file1_row.pack(fill=tk.X)

        self.file1_var = tk.StringVar(value="No file selected")
        file1_display = create_label(file1_row, text="", font=FONT_SMALL)
        file1_display.configure(textvariable=self.file1_var, text_color=SECONDARY_TEXT)
        file1_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACING_SM))

        browse1_btn = create_secondary_button(file1_row, text="Browse",
                                             command=self.browse_file1, width=80)
        browse1_btn.pack(side=tk.RIGHT)

        # File 2 selection
        file2_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        file2_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        file2_label = create_label(file2_section, text="Dataset 2 (Comparison):", font=FONT_BODY)
        file2_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        file2_row = ctk.CTkFrame(file2_section, fg_color="transparent")
        file2_row.pack(fill=tk.X)

        self.file2_var = tk.StringVar(value="No file selected")
        file2_display = create_label(file2_row, text="", font=FONT_SMALL)
        file2_display.configure(textvariable=self.file2_var, text_color=SECONDARY_TEXT)
        file2_display.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, SPACING_SM))

        browse2_btn = create_secondary_button(file2_row, text="Browse",
                                             command=self.browse_file2, width=80)
        browse2_btn.pack(side=tk.RIGHT)

        # Labels
        label1_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        label1_section.pack(fill=tk.X, pady=(0, SPACING_XS))

        label1_label = create_label(label1_section, text="Label 1:", font=FONT_BODY)
        label1_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.label1_var = tk.StringVar(value="Dataset 1")
        label1_entry = create_entry(label1_section, textvariable=self.label1_var, width=400)
        label1_entry.pack(fill=tk.X)

        label2_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        label2_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        label2_label = create_label(label2_section, text="Label 2:", font=FONT_BODY)
        label2_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.label2_var = tk.StringVar(value="Dataset 2")
        label2_entry = create_entry(label2_section, textvariable=self.label2_var, width=400)
        label2_entry.pack(fill=tk.X)

        # Columns to compare
        columns_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        columns_section.pack(fill=tk.X, pady=(0, SPACING_XS))

        columns_label = create_label(columns_section, text="Columns to Compare:", font=FONT_BODY)
        columns_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.columns_var = tk.StringVar(value="R1,R2,hetNOE")
        columns_entry = create_entry(columns_section, textvariable=self.columns_var, width=400)
        columns_entry.pack(fill=tk.X, pady=(0, SPACING_XS))

        hint_label = create_label(columns_section, text="(comma-separated, e.g., R1,R2,hetNOE,S2)",
                                 font=FONT_SMALL)
        hint_label.configure(text_color=SECONDARY_TEXT)
        hint_label.pack(anchor=tk.W)

        # Field selection
        field_frame = ctk.CTkFrame(content_inner, fg_color="transparent")
        field_frame.pack(fill=tk.X, pady=(0, SPACING_SM))

        field_label = create_label(field_frame, text="Field Selection:", font=FONT_BODY)
        field_label.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        self.field_var = tk.StringVar(value="f1")

        radio_field_frame = ctk.CTkFrame(field_frame, fg_color="transparent")
        radio_field_frame.pack(side=tk.LEFT)

        f1_radio = ctk.CTkRadioButton(radio_field_frame, text="f1",
                                     variable=self.field_var, value="f1",
                                     font=FONT_BODY, text_color=PRIMARY_TEXT,
                                     fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        f1_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        f2_radio = ctk.CTkRadioButton(radio_field_frame, text="f2",
                                     variable=self.field_var, value="f2",
                                     font=FONT_BODY, text_color=PRIMARY_TEXT,
                                     fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        f2_radio.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        both_radio = ctk.CTkRadioButton(radio_field_frame, text="both",
                                       variable=self.field_var, value="both",
                                       font=FONT_BODY, text_color=PRIMARY_TEXT,
                                       fg_color=PRIMARY_BUTTON_BG, hover_color=PRIMARY_BUTTON_HOVER)
        both_radio.pack(side=tk.LEFT)

        # Output file
        output_section = ctk.CTkFrame(content_inner, fg_color="transparent")
        output_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        output_label = create_label(output_section, text="Output File:", font=FONT_BODY)
        output_label.pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.output_var = tk.StringVar(value="comparison_plot.pdf")
        output_entry = create_entry(output_section, textvariable=self.output_var, width=400)
        output_entry.pack(fill=tk.X)

        # Start comparison button
        compare_btn = create_primary_button(content_inner, text="Start Comparison",
                                           command=self.start_comparison, width=200)
        compare_btn.pack(pady=(0, SPACING_MD))

        # Results display area
        results_frame = CTkLabelFrame(content_inner, text="Comparison Results", padding=SPACING_MD)
        results_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = ctk.CTkTextbox(results_frame.content, height=150,
                                          font=FONT_MONO, fg_color=FRAME_BG_COLOR,
                                          text_color=PRIMARY_TEXT, wrap="word")
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def browse_file1(self):
        """Browse for first dataset file"""
        filename = filedialog.askopenfilename(
            title="Select First Dataset (Reference)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            self.file1 = filename
            self.file1_var.set(os.path.basename(filename))

    def browse_file2(self):
        """Browse for second dataset file"""
        filename = filedialog.askopenfilename(
            title="Select Second Dataset (Comparison)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            self.file2 = filename
            self.file2_var.set(os.path.basename(filename))

    def start_comparison(self):
        """Start the dataset comparison"""
        if not self.file1 or not self.file2:
            messagebox.showerror("Error", "Please select both dataset files.")
            return

        # Clear results
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Starting comparison...\n")
        self.root.update()

        # Run comparison in separate thread
        thread = threading.Thread(target=self._run_comparison)
        thread.daemon = True
        thread.start()

    def _run_comparison(self):
        """Run the comparison (called in separate thread)"""
        try:
            # Import the appropriate comparison script based on spectral method
            script_path = self.parent_gui.dynamixs_path / "dynamiXs_plot"
            sys.path.insert(0, str(script_path))

            if self.spectral_method == "j087wh":
                from compare_datasets_087 import main as compare_main
            else:
                from compare_datasets import main as compare_main

            # Prepare command line arguments
            args = [
                self.file1,
                self.file2,
                "--column", self.columns_var.get(),
                "--label1", self.label1_var.get(),
                "--label2", self.label2_var.get(),
                "--field", self.field_var.get(),
                "--output", self.output_var.get()
            ]

            # Update GUI
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Running comparison analysis...\n"))

            # For now, show a message that this would run the comparison
            self.root.after(0, lambda: self._display_comparison_info(args))

        except Exception as e:
            error_msg = f"Error during comparison: {str(e)}\n"
            self.root.after(0, lambda: self.results_text.insert(tk.END, error_msg))

    def _display_comparison_info(self, args):
        """Display comparison information"""
        method_text = "J(0.87ωH)" if self.spectral_method == "j087wh" else "J(ωH)"
        self.results_text.insert(tk.END, "\nComparison would be run with the following parameters:\n")
        self.results_text.insert(tk.END, f"File 1: {self.file1}\n")
        self.results_text.insert(tk.END, f"File 2: {self.file2}\n")
        self.results_text.insert(tk.END, f"Spectral method: {method_text}\n")
        self.results_text.insert(tk.END, f"Columns: {self.columns_var.get()}\n")
        self.results_text.insert(tk.END, f"Field: {self.field_var.get()}\n")
        self.results_text.insert(tk.END, f"Output: {self.output_var.get()}\n")
        self.results_text.insert(tk.END, "\nNote: Full comparison functionality requires the compare_datasets.py script to be properly integrated.\n")
        self.results_text.see(tk.END)

    def go_back_to_plot_menu(self):
        """Return to plotting menu"""
        self.parent_gui.clear_window()
        PlottingGUI(self.root, self.parent_gui)


class IntegratedAnalysisGUI:
    """Integrated Spectral Density Analysis - Automated Workflow"""

    def __init__(self, root, parent_gui):
        self.root = root
        self.parent_gui = parent_gui

        # File paths
        self.field1_t1_file = None
        self.field1_t2_file = None
        self.field1_noe_sat_file = None
        self.field1_noe_unsat_file = None

        self.field2_t1_file = None
        self.field2_t2_file = None
        self.field2_noe_sat_file = None
        self.field2_noe_unsat_file = None

        self.setup_interface()

    def setup_interface(self):
        """Setup integrated analysis interface"""

        # Main container
        main_container = ctk.CTkFrame(self.root, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_XL, pady=SPACING_XL)

        # Header
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_LG))

        # Title with subtitle combined
        title_container = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_container.pack(side=tk.LEFT)

        title_label = create_label(title_container, text="Model Free Analysis",
                                   font=FONT_LARGE_HEADER)
        title_label.pack(anchor=tk.W)

        subtitle_label = create_label(title_container,
                                      text="Automated workflow from raw NMR data to model-free parameters",
                                      font=FONT_SMALL)
        subtitle_label.configure(text_color=SECONDARY_TEXT)
        subtitle_label.pack(anchor=tk.W)

        back_btn = create_secondary_button(header_frame, text="← Back to Main",
                                          command=self.parent_gui.go_back_to_main, width=140)
        back_btn.pack(side=tk.RIGHT)

        # Scrollable frame for content
        scroll_frame = ctk.CTkScrollableFrame(main_container, fg_color=PANEL_BG_COLOR,
                                              corner_radius=FRAME_CORNER_RADIUS)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # === FIELD 1 DATA (Required) ===
        field1_frame = CTkLabelFrame(scroll_frame, text="📊 Field 1 Data (Required)", padding=SPACING_MD)
        field1_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Add field strength to header (no label)
        self.field1_freq_var = tk.StringVar(value="600.0")
        freq1_entry = create_entry(field1_frame.header, textvariable=self.field1_freq_var, width=80)
        freq1_entry.pack(side=tk.LEFT, padx=(SPACING_MD, 0))
        create_label(field1_frame.header, text="MHz").pack(side=tk.LEFT, padx=(SPACING_XS, 0))

        # Auto-load from folder button
        auto_load_row = ctk.CTkFrame(field1_frame.content, fg_color="transparent")
        auto_load_row.pack(fill=tk.X, pady=(SPACING_SM, SPACING_MD))
        self.field1_auto_load_btn = create_primary_button(auto_load_row, text="📁 Auto-Load from Folder",
                            command=lambda: self.auto_load_folder("field1"),
                            width=200)
        self.field1_auto_load_btn.pack(side=tk.LEFT)
        help_label = create_label(auto_load_row, text="Automatically detect and load T1, T2, and hetNOE files",
                                 font=FONT_SMALL)
        help_label.configure(text_color=SECONDARY_TEXT)
        help_label.pack(side=tk.LEFT, padx=(SPACING_SM, 0))

        # Manual file loading (collapsible)
        self.field1_manual_collapsed = tk.BooleanVar(value=True)
        field1_manual_container = ctk.CTkFrame(field1_frame.content, fg_color="transparent")

        def toggle_field1_manual():
            self.field1_manual_collapsed.set(not self.field1_manual_collapsed.get())
            if self.field1_manual_collapsed.get():
                field1_manual_content.pack_forget()
                field1_manual_toggle_btn.configure(text="or load files manually ▶")
            else:
                field1_manual_content.pack(fill=tk.X, pady=(SPACING_SM, 0))
                field1_manual_toggle_btn.configure(text="or load files manually ▼")

        field1_manual_toggle_btn = create_secondary_button(field1_manual_container,
                                                           text="or load files manually ▶",
                                                           command=toggle_field1_manual,
                                                           width=200)
        field1_manual_toggle_btn.pack(anchor=tk.W)
        field1_manual_container.pack(fill=tk.X, pady=(0, SPACING_SM))

        # Manual file selection content (hidden by default)
        field1_manual_content = ctk.CTkFrame(field1_frame.content, fg_color="transparent")

        # Field 1 T1 file
        self._create_file_row(field1_manual_content, "T1 Data:", "field1_t1")

        # Field 1 T2 file
        self._create_file_row(field1_manual_content, "T2 Data:", "field1_t2")

        # Field 1 hetNOE saturated
        self._create_file_row(field1_manual_content, "hetNOE Saturated:", "field1_noe_sat")

        # Field 1 hetNOE unsaturated
        self._create_file_row(field1_manual_content, "hetNOE Unsaturated:", "field1_noe_unsat")

        # === FIELD 2 DATA (Optional) ===
        field2_frame = CTkLabelFrame(scroll_frame, text="📊 Field 2 Data (Optional - for dual-field)",
                                     padding=SPACING_MD)
        field2_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Add field strength to header (no label)
        self.field2_freq_var = tk.StringVar(value="700.0")
        self.freq2_entry = create_entry(field2_frame.header, textvariable=self.field2_freq_var, width=80)
        self.freq2_entry.pack(side=tk.LEFT, padx=(SPACING_MD, 0))
        self.freq2_entry.configure(state="disabled")
        create_label(field2_frame.header, text="MHz").pack(side=tk.LEFT, padx=(SPACING_XS, 0))

        # Enable dual-field button right after MHz label
        self.dual_field_var = tk.BooleanVar(value=False)
        self.dual_field_toggle_btn = create_secondary_button(field2_frame.header, text="Enable",
                                                             command=self.toggle_dual_field,
                                                             width=100)
        self.dual_field_toggle_btn.pack(side=tk.LEFT, padx=(SPACING_SM, 0))

        # Auto-load from folder button (Field 2)
        auto_load_row2 = ctk.CTkFrame(field2_frame.content, fg_color="transparent")
        auto_load_row2.pack(fill=tk.X, pady=(SPACING_SM, SPACING_MD))
        self.field2_auto_load_btn = create_primary_button(auto_load_row2, text="📁 Auto-Load from Folder",
                                                          command=lambda: self.auto_load_folder("field2"),
                                                          width=200, state="disabled")
        self.field2_auto_load_btn.pack(side=tk.LEFT)
        help_label2 = create_label(auto_load_row2, text="Automatically detect and load T1, T2, and hetNOE files",
                                   font=FONT_SMALL)
        help_label2.configure(text_color=SECONDARY_TEXT)
        help_label2.pack(side=tk.LEFT, padx=(SPACING_SM, 0))

        # Manual file loading (collapsible) - Field 2
        self.field2_manual_collapsed = tk.BooleanVar(value=True)
        field2_manual_container = ctk.CTkFrame(field2_frame.content, fg_color="transparent")

        def toggle_field2_manual():
            self.field2_manual_collapsed.set(not self.field2_manual_collapsed.get())
            if self.field2_manual_collapsed.get():
                field2_manual_content.pack_forget()
                field2_manual_toggle_btn.configure(text="or load files manually ▶")
            else:
                field2_manual_content.pack(fill=tk.X, pady=(SPACING_SM, 0))
                field2_manual_toggle_btn.configure(text="or load files manually ▼")

        self.field2_manual_toggle_btn = create_secondary_button(field2_manual_container,
                                                                text="or load files manually ▶",
                                                                command=toggle_field2_manual,
                                                                width=200, state="disabled")
        self.field2_manual_toggle_btn.pack(anchor=tk.W)
        field2_manual_container.pack(fill=tk.X, pady=(0, SPACING_SM))

        # Manual file selection content (hidden by default)
        field2_manual_content = ctk.CTkFrame(field2_frame.content, fg_color="transparent")

        # Field 2 files (disabled by default)
        self.field2_widgets = []
        self.field2_widgets.append(self._create_file_row(field2_manual_content, "T1 Data:", "field2_t1", enabled=False))
        self.field2_widgets.append(self._create_file_row(field2_manual_content, "T2 Data:", "field2_t2", enabled=False))
        self.field2_widgets.append(self._create_file_row(field2_manual_content, "hetNOE Saturated:", "field2_noe_sat", enabled=False))
        self.field2_widgets.append(self._create_file_row(field2_manual_content, "hetNOE Unsaturated:", "field2_noe_unsat", enabled=False))

        # === ANALYSIS PARAMETERS (Collapsible) ===
        params_container = ctk.CTkFrame(scroll_frame, fg_color=PANEL_BG_COLOR, corner_radius=FRAME_CORNER_RADIUS)
        params_container.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Collapsible header button
        self.params_collapsed = tk.BooleanVar(value=True)
        params_header = ctk.CTkFrame(params_container, fg_color="transparent")
        params_header.pack(fill=tk.X, padx=SPACING_MD, pady=SPACING_MD)

        # Content frame (collapsible) - define before toggle function
        params_content_frame = ctk.CTkFrame(params_container, fg_color="transparent")

        def toggle_params():
            self.params_collapsed.set(not self.params_collapsed.get())
            if self.params_collapsed.get():
                params_content_frame.pack_forget()
                params_toggle_btn.configure(text="⚙️ Analysis Parameters ▶")
            else:
                params_content_frame.pack(fill=tk.X, padx=SPACING_MD, pady=(0, SPACING_MD))
                params_toggle_btn.configure(text="⚙️ Analysis Parameters ▼")

        params_toggle_btn = create_secondary_button(params_header, text="⚙️ Analysis Parameters ▶",
                                                     command=toggle_params, width=300)
        params_toggle_btn.pack(anchor=tk.W)

        # Analysis method
        create_label(params_content_frame, text="Analysis Method:", font=FONT_SECTION_LABEL).pack(anchor=tk.W, pady=(0, SPACING_XS))

        self.method_var = tk.StringVar(value="dual_087")
        methods = [
            ("Single-field J(ωH)", "single_jwh"),
            ("Single-field J(0.87ωH)", "single_087"),
            ("Dual-field J(ωH) [requires Field 2]", "dual_jwh"),
            ("Dual-field J(0.87ωH) [requires Field 2] ⭐", "dual_087")
        ]

        self.method_radios = []
        for label, value in methods:
            radio = ctk.CTkRadioButton(params_content_frame, text=label, variable=self.method_var,
                                      value=value, fg_color=PRIMARY_BUTTON_BG,
                                      hover_color=PRIMARY_BUTTON_HOVER, font=FONT_BODY)
            radio.pack(anchor=tk.W, pady=(0, SPACING_XS))
            self.method_radios.append((radio, value))

        # Physical parameters
        phys_label = create_label(params_content_frame, text="Physical Parameters:", font=FONT_SECTION_LABEL)
        phys_label.pack(anchor=tk.W, pady=(SPACING_MD, SPACING_XS))

        phys_grid = ctk.CTkFrame(params_content_frame, fg_color="transparent")
        phys_grid.pack(fill=tk.X, pady=(0, SPACING_SM))

        create_label(phys_grid, text="N-H bond length:").grid(row=0, column=0, sticky=tk.W, pady=SPACING_XS)
        self.rnh_var = tk.StringVar(value="1.015")
        create_entry(phys_grid, textvariable=self.rnh_var, width=80).grid(row=0, column=1, padx=(SPACING_SM, SPACING_XS))
        create_label(phys_grid, text="Å").grid(row=0, column=2, sticky=tk.W)

        create_label(phys_grid, text="15N CSA:").grid(row=1, column=0, sticky=tk.W, pady=SPACING_XS)
        self.csa_var = tk.StringVar(value="-160.0")
        create_entry(phys_grid, textvariable=self.csa_var, width=80).grid(row=1, column=1, padx=(SPACING_SM, SPACING_XS))
        create_label(phys_grid, text="ppm").grid(row=1, column=2, sticky=tk.W)

        # Fitting parameters
        fit_label = create_label(params_content_frame, text="Fitting Parameters:", font=FONT_SECTION_LABEL)
        fit_label.pack(anchor=tk.W, pady=(SPACING_MD, SPACING_XS))

        fit_grid = ctk.CTkFrame(params_content_frame, fg_color="transparent")
        fit_grid.pack(fill=tk.X, pady=(0, SPACING_SM))

        create_label(fit_grid, text="T1 bootstrap:").grid(row=0, column=0, sticky=tk.W, pady=SPACING_XS)
        self.t1_bootstrap_var = tk.StringVar(value="1000")
        create_entry(fit_grid, textvariable=self.t1_bootstrap_var, width=80).grid(row=0, column=1, padx=(SPACING_SM, SPACING_XS))
        create_label(fit_grid, text="iterations").grid(row=0, column=2, sticky=tk.W)

        create_label(fit_grid, text="T2 bootstrap:").grid(row=1, column=0, sticky=tk.W, pady=SPACING_XS)
        self.t2_bootstrap_var = tk.StringVar(value="1000")
        create_entry(fit_grid, textvariable=self.t2_bootstrap_var, width=80).grid(row=1, column=1, padx=(SPACING_SM, SPACING_XS))
        create_label(fit_grid, text="iterations").grid(row=1, column=2, sticky=tk.W)

        create_label(fit_grid, text="Monte Carlo:").grid(row=2, column=0, sticky=tk.W, pady=SPACING_XS)
        self.mc_var = tk.StringVar(value="50")
        create_entry(fit_grid, textvariable=self.mc_var, width=80).grid(row=2, column=1, padx=(SPACING_SM, SPACING_XS))
        create_label(fit_grid, text="iterations").grid(row=2, column=2, sticky=tk.W)

        # === OUTPUT CONFIGURATION (Outside collapsible) ===
        output_frame = CTkLabelFrame(scroll_frame, text="", padding=SPACING_MD)
        output_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Output directory selection
        output_dir_row = ctk.CTkFrame(output_frame.content, fg_color="transparent")
        output_dir_row.pack(fill=tk.X, pady=(0, SPACING_SM))
        create_label(output_dir_row, text="Output Directory:").pack(side=tk.LEFT)

        self.output_dir = str(Path(__file__).parent / 'results')  # Default to results/ folder
        self.output_dir_var = tk.StringVar(value=os.path.basename(self.output_dir))

        output_dir_label = ctk.CTkLabel(output_dir_row, textvariable=self.output_dir_var,
                                        fg_color=FRAME_BG_COLOR, corner_radius=BUTTON_CORNER_RADIUS,
                                        font=FONT_SMALL, text_color=SECONDARY_TEXT, anchor="w")
        output_dir_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SPACING_SM, SPACING_SM))

        output_dir_btn = create_secondary_button(output_dir_row, text="Browse Folder",
                                                 command=self.browse_output_folder, width=120)
        output_dir_btn.pack(side=tk.RIGHT)

        # Output prefix (filename prefix)
        output_prefix_row = ctk.CTkFrame(output_frame.content, fg_color="transparent")
        output_prefix_row.pack(fill=tk.X, pady=(0, SPACING_SM))
        create_label(output_prefix_row, text="File Prefix:").pack(side=tk.LEFT)
        self.output_prefix_var = tk.StringVar(value="integrated_analysis")
        create_entry(output_prefix_row, textvariable=self.output_prefix_var, width=250).pack(side=tk.LEFT, padx=(SPACING_SM, 0), fill=tk.X, expand=True)

        # JSON data folder (for fit visualization data)
        json_folder_row = ctk.CTkFrame(output_frame.content, fg_color="transparent")
        json_folder_row.pack(fill=tk.X, pady=(0, SPACING_SM))
        create_label(json_folder_row, text="JSON Data Folder:").pack(side=tk.LEFT)

        # Default: {output_directory}/json
        self.json_folder_var = tk.StringVar(value="json")

        json_folder_label = ctk.CTkLabel(json_folder_row, textvariable=self.json_folder_var,
                                        fg_color=FRAME_BG_COLOR, corner_radius=BUTTON_CORNER_RADIUS,
                                        font=FONT_SMALL, text_color=SECONDARY_TEXT, anchor="w")
        json_folder_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SPACING_SM, SPACING_SM))

        json_folder_btn = create_secondary_button(json_folder_row, text="Browse Folder",
                                                  command=self.browse_json_folder, width=120)
        json_folder_btn.pack(side=tk.LEFT)

        # Action buttons
        button_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        button_frame.pack(fill=tk.X, pady=(SPACING_MD, 0))

        start_btn = create_primary_button(button_frame, text="Start Integrated Analysis",
                                          command=self.start_analysis, width=200)
        start_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        reset_btn = create_secondary_button(button_frame, text="Reset",
                                           command=self.reset_form, width=80)
        reset_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        view_results_btn = create_secondary_button(button_frame, text="View Results",
                                                   command=self.open_results_viewer, width=120)
        view_results_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

        view_fits_btn = create_secondary_button(button_frame, text="View Fits",
                                                command=self.open_fit_viewer, width=100)
        view_fits_btn.pack(side=tk.LEFT)

        # Progress log
        log_frame = CTkLabelFrame(scroll_frame, text="📋 Progress Log", padding=SPACING_MD)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(SPACING_MD, 0))

        self.progress_text = ctk.CTkTextbox(log_frame.content, height=200, width=600,
                                           font=FONT_MONO, fg_color=FRAME_BG_COLOR,
                                           text_color=PRIMARY_TEXT)
        self.progress_text.pack(fill=tk.BOTH, expand=True)

    def _create_file_row(self, parent, label_text, field_name, enabled=True):
        """Helper to create a file selection row"""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X, pady=(0, SPACING_SM))

        label = create_label(row, text=label_text)
        label.pack(side=tk.LEFT)
        label.configure(anchor="w", width=150)

        file_var = tk.StringVar(value="No file selected")
        setattr(self, f"{field_name}_var", file_var)

        file_label = ctk.CTkLabel(row, textvariable=file_var, fg_color=FRAME_BG_COLOR,
                                 corner_radius=BUTTON_CORNER_RADIUS, font=FONT_SMALL,
                                 text_color=SECONDARY_TEXT, anchor="w")
        file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SPACING_SM, SPACING_SM))

        browse_btn = create_secondary_button(row, text="Browse",
                                            command=lambda: self.browse_file(field_name), width=80)
        browse_btn.pack(side=tk.RIGHT)

        if not enabled:
            file_label.configure(state="disabled")
            browse_btn.configure(state="disabled")

        return (file_label, browse_btn)

    def browse_file(self, field_name):
        """Browse for a file"""
        filename = filedialog.askopenfilename(
            title=f"Select {field_name.replace('_', ' ').title()} File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.parent_gui.current_dir
        )
        if filename:
            setattr(self, f"{field_name}_file", filename)
            getattr(self, f"{field_name}_var").set(os.path.basename(filename))

    def auto_load_folder(self, field_prefix):
        """
        Auto-detect and load files from a folder based on keywords.

        Detection rules:
        - 'T1' in filename → T1 data
        - 'T2' in filename → T2 data
        - 'SAT' or 'saturated' in filename (case-insensitive) → hetNOE saturated
        - 'NOSAT' or 'unsaturated' in filename (case-insensitive) → hetNOE unsaturated
        """
        folder_path = filedialog.askdirectory(
            title=f"Select Folder for {field_prefix.upper()} Data",
            initialdir=self.parent_gui.current_dir
        )

        if not folder_path:
            return

        # Get all CSV files in the folder
        import glob
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not csv_files:
            messagebox.showwarning("No Files", "No CSV files found in the selected folder.")
            return

        # Detection results
        detected = {
            't1': None,
            't2': None,
            'noe_sat': None,
            'noe_unsat': None
        }

        # Detect files based on keywords (case-insensitive)
        for filepath in csv_files:
            filename = os.path.basename(filepath).lower()

            # T1 detection
            if 't1' in filename and detected['t1'] is None:
                detected['t1'] = filepath

            # T2 detection
            elif 't2' in filename and detected['t2'] is None:
                detected['t2'] = filepath

            # hetNOE unsaturated detection (check this FIRST before saturated)
            elif 'nosat' in filename or 'unsaturated' in filename:
                if detected['noe_unsat'] is None:
                    detected['noe_unsat'] = filepath

            # hetNOE saturated detection
            elif 'sat' in filename or 'saturated' in filename:
                if detected['noe_sat'] is None:
                    detected['noe_sat'] = filepath

        # Apply detected files
        loaded_count = 0
        missing = []

        if detected['t1']:
            setattr(self, f"{field_prefix}_t1_file", detected['t1'])
            getattr(self, f"{field_prefix}_t1_var").set(os.path.basename(detected['t1']))
            loaded_count += 1
        else:
            missing.append("T1")

        if detected['t2']:
            setattr(self, f"{field_prefix}_t2_file", detected['t2'])
            getattr(self, f"{field_prefix}_t2_var").set(os.path.basename(detected['t2']))
            loaded_count += 1
        else:
            missing.append("T2")

        if detected['noe_sat']:
            setattr(self, f"{field_prefix}_noe_sat_file", detected['noe_sat'])
            getattr(self, f"{field_prefix}_noe_sat_var").set(os.path.basename(detected['noe_sat']))
            loaded_count += 1
        else:
            missing.append("hetNOE Saturated")

        if detected['noe_unsat']:
            setattr(self, f"{field_prefix}_noe_unsat_file", detected['noe_unsat'])
            getattr(self, f"{field_prefix}_noe_unsat_var").set(os.path.basename(detected['noe_unsat']))
            loaded_count += 1
        else:
            missing.append("hetNOE Unsaturated")

        # Show results and update button color
        if loaded_count == 4:
            # Change button to green when all files are loaded
            if field_prefix == "field1":
                self.field1_auto_load_btn.configure(
                    fg_color=SUCCESS_GREEN_SOFT,
                    hover_color=SUCCESS_GREEN_HOVER
                )
            elif field_prefix == "field2":
                self.field2_auto_load_btn.configure(
                    fg_color=SUCCESS_GREEN_SOFT,
                    hover_color=SUCCESS_GREEN_HOVER
                )
            messagebox.showinfo("Success",
                              f"✓ All 4 files loaded successfully from:\n{folder_path}")
        elif loaded_count > 0:
            # Keep button blue if partial load
            missing_str = ", ".join(missing)
            messagebox.showwarning("Partial Load",
                                  f"✓ Loaded {loaded_count}/4 files\n\n"
                                  f"Missing: {missing_str}\n\n"
                                  f"Please select these files manually using the Browse buttons.")
        else:
            messagebox.showerror("No Files Detected",
                               "Could not auto-detect any files.\n\n"
                               "Files must contain keywords:\n"
                               "• 'T1' for T1 data\n"
                               "• 'T2' for T2 data\n"
                               "• 'SAT' or 'saturated' for hetNOE saturated\n"
                               "• 'NOSAT' or 'unsaturated' for hetNOE unsaturated")

    def browse_output_folder(self):
        """Browse for output directory"""
        folder_path = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir
        )
        if folder_path:
            self.output_dir = folder_path
            # Update label to show just the folder name (or full path if preferred)
            # Show full path in label for clarity
            self.output_dir_var.set(folder_path)

    def browse_json_folder(self):
        """Browse for JSON data folder"""
        # Get current json folder path (absolute)
        current_json = self.json_folder_var.get()
        if current_json == "json":
            initial_dir = os.path.join(self.output_dir, "json")
        else:
            initial_dir = current_json

        folder_path = filedialog.askdirectory(
            title="Select JSON Data Folder",
            initialdir=initial_dir if os.path.exists(initial_dir) else self.output_dir
        )
        if folder_path:
            # Show full path
            self.json_folder_var.set(folder_path)

    def toggle_dual_field(self):
        """Enable/disable Field 2 inputs"""
        # Toggle the state
        self.dual_field_var.set(not self.dual_field_var.get())
        enabled = self.dual_field_var.get()

        # Update button text
        self.dual_field_toggle_btn.configure(text="Disable" if enabled else "Enable")

        # Enable/disable frequency entry
        self.freq2_entry.configure(state="normal" if enabled else "disabled")

        # Enable/disable auto-load button
        self.field2_auto_load_btn.configure(state="normal" if enabled else "disabled")

        # Enable/disable manual load button
        self.field2_manual_toggle_btn.configure(state="normal" if enabled else "disabled")

        # Enable/disable file widgets
        for label, button in self.field2_widgets:
            label.configure(state="normal" if enabled else "disabled")
            button.configure(state="normal" if enabled else "disabled")

        # Update method radio buttons (disable dual-field if not enabled)
        for radio, value in self.method_radios:
            if value.startswith("dual_"):
                radio.configure(state="normal" if enabled else "disabled")

        # Reset to single-field method if dual-field is disabled
        if not enabled and self.method_var.get().startswith("dual_"):
            self.method_var.set("single_087")

    def start_analysis(self):
        """Start the integrated analysis"""
        # Validate inputs
        if not all([self.field1_t1_file, self.field1_t2_file,
                   self.field1_noe_sat_file, self.field1_noe_unsat_file]):
            messagebox.showerror("Error", "Please select all Field 1 data files.")
            return

        if self.dual_field_var.get():
            if not all([self.field2_t1_file, self.field2_t2_file,
                       self.field2_noe_sat_file, self.field2_noe_unsat_file]):
                messagebox.showerror("Error", "Please select all Field 2 data files for dual-field analysis.")
                return

        # Clear progress log
        self.progress_text.delete("1.0", tk.END)
        self.progress_text.insert(tk.END, "Starting integrated analysis...\n")

        # Run analysis in background thread
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()

    def _run_analysis(self):
        """Run the analysis (in background thread)"""
        try:
            # Import integrated analysis module
            from dynamiXs_integrated import IntegratedAnalysisParameters, IntegratedAnalysisPipeline

            # Create parameters
            params = IntegratedAnalysisParameters()

            # Field 1
            params.field1_freq_mhz = float(self.field1_freq_var.get())
            params.field1_t1_file = self.field1_t1_file
            params.field1_t2_file = self.field1_t2_file
            params.field1_noe_sat_file = self.field1_noe_sat_file
            params.field1_noe_unsat_file = self.field1_noe_unsat_file

            # Field 2
            params.enable_dual_field = self.dual_field_var.get()
            if params.enable_dual_field:
                params.field2_freq_mhz = float(self.field2_freq_var.get())
                params.field2_t1_file = self.field2_t1_file
                params.field2_t2_file = self.field2_t2_file
                params.field2_noe_sat_file = self.field2_noe_sat_file
                params.field2_noe_unsat_file = self.field2_noe_unsat_file

            # Analysis parameters
            params.analysis_method = self.method_var.get()
            params.rNH_angstrom = float(self.rnh_var.get())
            params.csaN_ppm = float(self.csa_var.get())
            params.t1_bootstrap_iterations = int(self.t1_bootstrap_var.get())
            params.t2_bootstrap_iterations = int(self.t2_bootstrap_var.get())
            params.monte_carlo_iterations = int(self.mc_var.get())

            # Output configuration
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Combine output directory + prefix for full path
            params.output_prefix = str(output_dir / self.output_prefix_var.get())

            # JSON folder configuration
            json_folder = self.json_folder_var.get()
            if json_folder == "json":
                # Relative path - combine with output directory
                params.json_folder = str(output_dir / "json")
            else:
                # Absolute path
                params.json_folder = json_folder

            # Log output location
            self._log_progress(f"Output directory: {self.output_dir}")
            self._log_progress(f"File prefix: {self.output_prefix_var.get()}")
            self._log_progress(f"JSON folder: {params.json_folder}")

            # Create pipeline with progress callback
            pipeline = IntegratedAnalysisPipeline(params, progress_callback=self._log_progress)

            # Run analysis
            results = pipeline.run_complete_analysis()

            # Display results
            self.root.after(0, lambda: self._display_completion(results))

        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}\n"
            import traceback
            error_msg += traceback.format_exc()
            self.root.after(0, lambda: self.progress_text.insert(tk.END, error_msg))

    def _log_progress(self, message):
        """Log progress message to GUI"""
        self.root.after(0, lambda: self.progress_text.insert(tk.END, message + "\n"))
        self.root.after(0, lambda: self.progress_text.see(tk.END))

    def _display_completion(self, results):
        """Display completion message and open results viewer"""
        self.progress_text.insert(tk.END, "\n" + "="*60 + "\n")
        self.progress_text.insert(tk.END, "ANALYSIS COMPLETE!\n")
        self.progress_text.insert(tk.END, "="*60 + "\n")
        self.progress_text.see(tk.END)

        # Open results viewer automatically
        try:
            # Find the results CSV file
            output_prefix = self.output_prefix_var.get()
            output_dir = Path(self.output_dir)

            # Look for results file (prioritize detailed CSV)
            results_file = None
            possible_files = [
                output_dir / f"{output_prefix}_spectral_density_detailed.csv",
                output_dir / f"{output_prefix}_spectral_density_basic.csv",
                output_dir / f"{output_prefix}_results.csv",
                output_dir / f"{output_prefix}_field1_results.csv",
                output_dir / f"{output_prefix}_field2_results.csv",
                output_dir / f"{output_prefix}_model_free.csv",
            ]

            for file_path in possible_files:
                if file_path.exists():
                    results_file = str(file_path)
                    break

            # If not found, try to find any CSV in output directory (avoid input files)
            if results_file is None:
                csv_files = [f for f in output_dir.glob(f"{output_prefix}*.csv")
                            if "input" not in f.name and "field1" not in f.name and "field2" not in f.name]
                if csv_files:
                    results_file = str(csv_files[0])

            if results_file:
                self.progress_text.insert(tk.END, f"\nOpening results viewer for: {os.path.basename(results_file)}\n")
                self.progress_text.see(tk.END)

                # Open the results viewer
                viewer = ResultsViewer(
                    parent=self.root,
                    results_file=results_file,
                    field1_freq=float(self.field1_freq_var.get()),
                    field2_freq=float(self.field2_freq_var.get()) if self.dual_field_var.get() else None,
                    is_dual_field=self.dual_field_var.get()
                )
            else:
                self.progress_text.insert(tk.END, "\nWarning: Could not find results CSV file to display.\n")
                self.progress_text.see(tk.END)

        except Exception as e:
            self.progress_text.insert(tk.END, f"\nError opening results viewer: {str(e)}\n")
            self.progress_text.see(tk.END)

        # Also open Fit Viewer
        try:
            # Determine JSON folder path
            json_folder = self.json_folder_var.get()
            if json_folder == "json":
                json_folder_path = os.path.join(self.output_dir, "json")
            else:
                json_folder_path = json_folder

            if os.path.exists(json_folder_path):
                self.progress_text.insert(tk.END, f"\nOpening T1/T2 fit viewer...\n")
                self.progress_text.see(tk.END)

                from visualization.fit_viewer import FitViewer

                fit_viewer = FitViewer(
                    parent=self.root,
                    json_folder=json_folder_path
                )
        except Exception as e:
            self.progress_text.insert(tk.END, f"\nError opening fit viewer: {str(e)}\n")
            self.progress_text.see(tk.END)

    def open_results_viewer(self):
        """Open the Results Viewer (always available)"""
        try:
            # Try to find the most recent results file
            output_dir = Path(self.output_dir_var.get())
            output_prefix = self.output_prefix_var.get()
            results_file = None

            if output_dir.exists():
                # Look for results files in priority order
                possible_files = [
                    output_dir / f"{output_prefix}_spectral_density_detailed.csv",
                    output_dir / f"{output_prefix}_spectral_density_basic.csv",
                    output_dir / f"{output_prefix}_field1_input.csv",
                ]

                for file_path in possible_files:
                    if file_path.exists():
                        results_file = file_path
                        break

            # Open viewer with or without results file
            viewer = ResultsViewer(
                parent=self.root,
                results_file=str(results_file) if results_file else None,
                field1_freq=float(self.field1_freq_var.get()),
                field2_freq=float(self.field2_freq_var.get()) if self.dual_field_var.get() else None,
                is_dual_field=self.dual_field_var.get()
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error opening results viewer:\n{str(e)}")

    def open_fit_viewer(self):
        """Open the T1/T2 Fit Viewer (always available)"""
        try:
            # Determine JSON folder path
            json_folder = self.json_folder_var.get()
            if json_folder == "json":
                # Relative path - combine with output directory
                json_folder_path = os.path.join(self.output_dir, "json")
            else:
                # Absolute path
                json_folder_path = json_folder

            # Import FitViewer
            from visualization.fit_viewer import FitViewer

            # Open viewer
            viewer = FitViewer(
                parent=self.root,
                json_folder=json_folder_path if os.path.exists(json_folder_path) else None
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error opening fit viewer:\n{str(e)}")

    def reset_form(self):
        """Reset all form fields"""
        # Reset file selections
        for field in ['field1_t1', 'field1_t2', 'field1_noe_sat', 'field1_noe_unsat',
                     'field2_t1', 'field2_t2', 'field2_noe_sat', 'field2_noe_unsat']:
            setattr(self, f"{field}_file", None)
            getattr(self, f"{field}_var").set("No file selected")

        # Reset parameters to defaults
        self.field1_freq_var.set("600.0")
        self.field2_freq_var.set("700.0")
        self.dual_field_var.set(False)
        self.method_var.set("dual_087")
        self.rnh_var.set("1.015")
        self.csa_var.set("-160.0")
        self.t1_bootstrap_var.set("1000")
        self.t2_bootstrap_var.set("1000")
        self.mc_var.set("50")
        self.output_prefix_var.set("integrated_analysis")

        # Clear progress log
        self.progress_text.delete("1.0", tk.END)

        # Update dual-field state
        self.toggle_dual_field()


def main():
    """Main function to start the GUI"""
    root = ctk.CTk()  # Use CTk instead of Tk for CustomTkinter
    app = DynamiXsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
