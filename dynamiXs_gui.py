#!/usr/bin/env python3
"""
DynamiXs GUI - A comprehensive interface for NMR relaxation data analysis

This GUI provides access to:
- T1/T2 fitting analysis
- CPMG relaxation dispersion 
- Spectral density function analysis
- Data plotting and comparison tools
- Data formatting utilities

Built with Tkinter for cross-platform compatibility (macOS, Windows, Linux)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import subprocess
import threading
from pathlib import Path
import pandas as pd

class DynamiXsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DynamiXs - NMR Relaxation Analysis Suite")
        self.root.geometry("800x600")
        
        # Get the dynamiXs directory path
        self.dynamixs_path = Path(__file__).parent
        
        # Current working directory for file operations
        self.current_dir = os.getcwd()
        
        # Setup main interface
        self.setup_main_interface()
        
    def setup_main_interface(self):
        """Create the main interface with 5 main options"""
        
        # Title
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="DynamiXs - NMR Relaxation Analysis Suite", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0)
        
        subtitle_label = ttk.Label(title_frame, text="Choose your analysis type:", 
                                  font=("Arial", 12))
        subtitle_label.grid(row=1, column=0, pady=(5, 0))
        
        # Main options frame
        options_frame = ttk.Frame(self.root, padding="20")
        options_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        options_frame.columnconfigure(0, weight=1)
        
        # Option 1: Format Data (empty for now)
        btn_format = ttk.Button(options_frame, text="1. Format Data", 
                               command=self.show_format_data,
                               width=30)
        btn_format.grid(row=0, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Option 2: T1/T2 Fitting
        btn_t1_t2 = ttk.Button(options_frame, text="2. T1/T2 Fitting Analysis", 
                              command=self.show_t1_t2_menu,
                              width=30)
        btn_t1_t2.grid(row=1, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Option 3: Spectral Density Analysis
        btn_spectral = ttk.Button(options_frame, text="3. Spectral Density Analysis", 
                                 command=self.show_spectral_density_menu,
                                 width=30)
        btn_spectral.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Option 4: Plot Data
        btn_plot = ttk.Button(options_frame, text="4. Plot Data", 
                             command=self.show_plot_menu,
                             width=30)
        btn_plot.grid(row=3, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Option 5: CPMG Analysis (empty for now)
        btn_cpmg = ttk.Button(options_frame, text="5. CPMG Analysis", 
                             command=self.show_cpmg_analysis,
                             width=30)
        btn_cpmg.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Working directory info
        dir_frame = ttk.Frame(self.root, padding="10")
        dir_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        dir_label = ttk.Label(dir_frame, text=f"Working Directory: {self.current_dir}", 
                             font=("Arial", 9))
        dir_label.grid(row=0, column=0, sticky=tk.W)
        
        change_dir_btn = ttk.Button(dir_frame, text="Change Directory", 
                                   command=self.change_working_directory)
        change_dir_btn.grid(row=0, column=1, padx=(10, 0))
        
    def change_working_directory(self):
        """Allow user to change working directory"""
        new_dir = filedialog.askdirectory(title="Select Working Directory", 
                                         initialdir=self.current_dir)
        if new_dir:
            self.current_dir = new_dir
            os.chdir(new_dir)
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
        
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(header_frame, text="T1/T2 Fitting Analysis", 
                 font=("Arial", 16, "bold")).grid(row=0, column=0)
        
        ttk.Button(header_frame, text="← Back to Main", 
                  command=self.parent_gui.go_back_to_main).grid(row=0, column=1, padx=(20, 0))
        
        # Main content
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Experiment type selection
        ttk.Label(main_frame, text="Select Experiment Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        exp_frame = ttk.Frame(main_frame)
        exp_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.exp_var = tk.StringVar(value="T1")
        ttk.Radiobutton(exp_frame, text="T1", variable=self.exp_var, 
                       value="T1", command=self.update_experiment_type).grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(exp_frame, text="T2", variable=self.exp_var, 
                       value="T2", command=self.update_experiment_type).grid(row=0, column=1)
        
        # File import
        ttk.Label(main_frame, text="Import Data File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_var, 
                 relief="sunken", padding="5").grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_input_file).grid(row=0, column=1, padx=(10, 0))
        
        # Results name
        ttk.Label(main_frame, text="Results Prefix:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.prefix_var = tk.StringVar(value="T1_analysis")
        ttk.Entry(main_frame, textvariable=self.prefix_var, 
                 width=30).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Processing options
        ttk.Label(main_frame, text="Processing Mode:").grid(row=3, column=0, sticky=tk.W, pady=5)
        
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.mode_var = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Single Core", variable=self.mode_var, 
                       value="single").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(mode_frame, text="Multi Core", variable=self.mode_var, 
                       value="multi").grid(row=0, column=1)
        
        # Advanced parameters
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
        advanced_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        advanced_frame.columnconfigure(1, weight=1)
        
        ttk.Label(advanced_frame, text="Initial Amplitude:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.initial_A_var = tk.StringVar(value="5")
        ttk.Entry(advanced_frame, textvariable=self.initial_A_var, width=15).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(advanced_frame, text="Initial Time Constant:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.initial_t2_var = tk.StringVar(value="100")
        ttk.Entry(advanced_frame, textvariable=self.initial_t2_var, width=15).grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(advanced_frame, text="Bootstrap Iterations:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.bootstrap_var = tk.StringVar(value="1000")
        ttk.Entry(advanced_frame, textvariable=self.bootstrap_var, width=15).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # Start analysis button
        ttk.Button(main_frame, text="Start Analysis", 
                  command=self.start_analysis,
                  style="Accent.TButton").grid(row=5, column=0, columnspan=2, pady=20)
        
        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)
        
        # Display results button
        ttk.Button(results_frame, text="Display Results", 
                  command=self.display_results).grid(row=1, column=0, pady=(10, 0))
    
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
        self.results_text.delete(1.0, tk.END)
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
                results_window = tk.Toplevel(self.root)
                results_window.title(f"Results: {results_file}")
                results_window.geometry("800x600")
                
                text_widget = scrolledtext.ScrolledText(results_window, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                text_widget.insert(tk.END, content)
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
        
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(header_frame, text="Spectral Density Analysis", 
                 font=("Arial", 16, "bold")).grid(row=0, column=0)
        
        ttk.Button(header_frame, text="← Back to Main", 
                  command=self.parent_gui.go_back_to_main).grid(row=0, column=1, padx=(20, 0))
        
        # Main content
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Analysis type selection
        ttk.Label(main_frame, text="Analysis Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        analysis_frame = ttk.Frame(main_frame)
        analysis_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.analysis_var = tk.StringVar(value="single_field")
        ttk.Radiobutton(analysis_frame, text="Single Field", variable=self.analysis_var, 
                       value="single_field", command=self.update_analysis_type).grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(analysis_frame, text="Dual Field", variable=self.analysis_var, 
                       value="dual_field", command=self.update_analysis_type).grid(row=0, column=1)
        
        # Spectral density calculation method selection
        ttk.Label(main_frame, text="Spectral Density Method:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        method_frame = ttk.Frame(main_frame)
        method_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.spectral_method_var = tk.StringVar(value="jwh")
        ttk.Radiobutton(method_frame, text="J(ωH)", variable=self.spectral_method_var, 
                       value="jwh").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(method_frame, text="J(0.87ωH)", variable=self.spectral_method_var, 
                       value="j087wh").grid(row=0, column=1)
        
        # Field parameters
        field_frame = ttk.LabelFrame(main_frame, text="Field Parameters", padding="10")
        field_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        field_frame.columnconfigure(1, weight=1)
        
        ttk.Label(field_frame, text="Field 1 Frequency (MHz):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.field1_var = tk.StringVar(value="600.0")
        ttk.Entry(field_frame, textvariable=self.field1_var, width=15).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(field_frame, text="Field 2 Frequency (MHz):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.field2_var = tk.StringVar(value="700.0")
        self.field2_entry = ttk.Entry(field_frame, textvariable=self.field2_var, width=15, state="disabled")
        self.field2_entry.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # File import section
        files_frame = ttk.LabelFrame(main_frame, text="Data Import", padding="10")
        files_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        files_frame.columnconfigure(1, weight=1)
        
        # File 1
        ttk.Label(files_frame, text="Data File 1:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file1_frame = ttk.Frame(files_frame)
        file1_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        file1_frame.columnconfigure(0, weight=1)
        
        self.file1_var = tk.StringVar(value="No file selected")
        ttk.Label(file1_frame, textvariable=self.file1_var, 
                 relief="sunken", padding="5").grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(file1_frame, text="Browse", 
                  command=self.browse_file1).grid(row=0, column=1, padx=(10, 0))
        
        # File 2
        ttk.Label(files_frame, text="Data File 2:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file2_frame = ttk.Frame(files_frame)
        file2_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        file2_frame.columnconfigure(0, weight=1)
        
        self.file2_var = tk.StringVar(value="Not required for single field")
        self.file2_label = ttk.Label(file2_frame, textvariable=self.file2_var, 
                                    relief="sunken", padding="5", state="disabled")
        self.file2_label.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.file2_button = ttk.Button(file2_frame, text="Browse", 
                                      command=self.browse_file2, state="disabled")
        self.file2_button.grid(row=0, column=1, padx=(10, 0))
        
        # Advanced parameters
        advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
        advanced_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        advanced_frame.columnconfigure(1, weight=1)
        
        ttk.Label(advanced_frame, text="N-H Bond Length (Å):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.rnh_var = tk.StringVar(value="1.015")
        ttk.Entry(advanced_frame, textvariable=self.rnh_var, width=15).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(advanced_frame, text="15N CSA (ppm):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.csa_var = tk.StringVar(value="-160.0")
        ttk.Entry(advanced_frame, textvariable=self.csa_var, width=15).grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(advanced_frame, text="Use Monte Carlo Errors:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.monte_carlo_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(advanced_frame, variable=self.monte_carlo_var).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(advanced_frame, text="Monte Carlo Samples:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.mc_samples_var = tk.StringVar(value="50")
        ttk.Entry(advanced_frame, textvariable=self.mc_samples_var, width=15).grid(row=3, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # Results name
        ttk.Label(main_frame, text="Output Prefix:").grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.prefix_var = tk.StringVar(value="spectral_density_analysis")
        ttk.Entry(main_frame, textvariable=self.prefix_var, 
                 width=30).grid(row=5, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Start analysis button
        ttk.Button(main_frame, text="Start Analysis", 
                  command=self.start_analysis,
                  style="Accent.TButton").grid(row=6, column=0, columnspan=2, pady=20)
        
        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Analysis Results", padding="10")
        results_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(7, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)
        
        # Display results button
        ttk.Button(results_frame, text="Display Results", 
                  command=self.display_results).grid(row=1, column=0, pady=(10, 0))
    
    def update_analysis_type(self):
        """Update interface based on analysis type selection"""
        if self.analysis_var.get() == "dual_field":
            # Enable dual field options
            self.field2_entry.configure(state="normal")
            self.file2_label.configure(state="normal")
            self.file2_button.configure(state="normal")
            self.file2_var.set("No file selected")
        else:
            # Disable dual field options
            self.field2_entry.configure(state="disabled")
            self.file2_label.configure(state="disabled")
            self.file2_button.configure(state="disabled")
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
        self.results_text.delete(1.0, tk.END)
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
            
            # Determine which script to use
            analysis_type = self.analysis_var.get()
            spectral_method = self.spectral_method_var.get()
            
            if analysis_type == "dual_field":
                if spectral_method == "j087wh":
                    from ZZ_2fields_density_087 import run_spectral_density_analysis_with_params
                else:
                    from ZZ_2fields_density_claude_rex_mcmc_error import run_spectral_density_analysis_with_params
            else:  # single_field
                if spectral_method == "j087wh":
                    from ZZ_density_087 import run_spectral_density_analysis_with_params
                else:
                    from ZZ_density import run_spectral_density_analysis_with_params
            
            # Prepare parameters
            params = {
                'input_file1': self.input_file1,
                'input_file2': self.input_file2,
                'field1_freq': float(self.field1_var.get()),
                'field2_freq': float(self.field2_var.get()) if self.analysis_var.get() == "dual_field" else float(self.field1_var.get()),
                'analysis_type': self.analysis_var.get(),
                'spectral_method': self.spectral_method_var.get(),
                'output_prefix': self.prefix_var.get(),
                'use_monte_carlo': self.monte_carlo_var.get(),
                'n_monte_carlo': int(self.mc_samples_var.get()),
                'rNH': float(self.rnh_var.get()) * 1e-10,  # Convert Å to meters
                'csaN': float(self.csa_var.get()) * 1e-6   # Convert ppm to frequency units
            }
            
            # Run analysis
            self.root.after(0, lambda: self.results_text.insert(tk.END, "Running spectral density analysis...\n"))
            
            results = run_spectral_density_analysis_with_params(params)
            
            # Update GUI with results
            self.root.after(0, lambda: self._display_analysis_results(results))
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}\n"
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
                results_window = tk.Toplevel(self.root)
                results_window.title(f"Spectral Density Results: {results_file}")
                results_window.geometry("1000x600")
                
                # Create treeview for data display
                tree_frame = ttk.Frame(results_window)
                tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                tree = ttk.Treeview(tree_frame)
                tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
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
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
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
        
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(header_frame, text="Plot Data", 
                 font=("Arial", 16, "bold")).grid(row=0, column=0)
        
        ttk.Button(header_frame, text="← Back to Main", 
                  command=self.parent_gui.go_back_to_main).grid(row=0, column=1, padx=(20, 0))
        
        # Main content
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)
        
        # Spectral density method selection
        ttk.Label(main_frame, text="Spectral Density Method:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        method_frame = ttk.Frame(main_frame)
        method_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.spectral_method_var = tk.StringVar(value="jwh")
        ttk.Radiobutton(method_frame, text="J(ωH)", variable=self.spectral_method_var, 
                       value="jwh").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(method_frame, text="J(0.87ωH)", variable=self.spectral_method_var, 
                       value="j087wh").grid(row=0, column=1)
        
        # Plot options
        ttk.Button(main_frame, text="Plot Dual Field Data", 
                  command=self.show_dual_field_plot,
                  width=30).grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(main_frame, text="Compare Datasets", 
                  command=self.show_compare_datasets,
                  width=30).grid(row=2, column=0, columnspan=2, pady=10)
    
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
        
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        method_text = "J(0.87ωH)" if self.spectral_method == "j087wh" else "J(ωH)"
        title_text = f"Plot Dual Field Data ({method_text})"
        ttk.Label(header_frame, text=title_text, 
                 font=("Arial", 16, "bold")).grid(row=0, column=0)
        
        ttk.Button(header_frame, text="← Back to Plot Menu", 
                  command=lambda: self.go_back_to_plot_menu()).grid(row=0, column=1, padx=(20, 0))
        
        # Main content
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # File import
        ttk.Label(main_frame, text="Import Data File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_var, 
                 relief="sunken", padding="5").grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_input_file).grid(row=0, column=1, padx=(10, 0))
        
        # Results name
        ttk.Label(main_frame, text="Output Prefix:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.prefix_var = tk.StringVar(value="dual_field_plot")
        ttk.Entry(main_frame, textvariable=self.prefix_var, 
                 width=30).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Plot parameters
        plot_frame = ttk.LabelFrame(main_frame, text="Plot Parameters", padding="10")
        plot_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        plot_frame.columnconfigure(1, weight=1)
        
        ttk.Label(plot_frame, text="Dataset Label:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.label_var = tk.StringVar(value="Dataset")
        ttk.Entry(plot_frame, textvariable=self.label_var, width=20).grid(row=0, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        ttk.Label(plot_frame, text="Figure Size (width x height):").grid(row=1, column=0, sticky=tk.W, pady=2)
        size_frame = ttk.Frame(plot_frame)
        size_frame.grid(row=1, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        self.width_var = tk.StringVar(value="12")
        self.height_var = tk.StringVar(value="8")
        ttk.Entry(size_frame, textvariable=self.width_var, width=5).grid(row=0, column=0)
        ttk.Label(size_frame, text=" x ").grid(row=0, column=1)
        ttk.Entry(size_frame, textvariable=self.height_var, width=5).grid(row=0, column=2)
        
        ttk.Label(plot_frame, text="Include Secondary Structure:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.ss_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(plot_frame, variable=self.ss_var).grid(row=2, column=1, sticky=tk.W, pady=2, padx=(10, 0))
        
        # Start plotting button
        ttk.Button(main_frame, text="Generate Plot", 
                  command=self.start_plotting,
                  style="Accent.TButton").grid(row=3, column=0, columnspan=2, pady=20)
        
        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Plot Generation Results", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)
    
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
        self.results_text.delete(1.0, tk.END)
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
        
        # Header
        header_frame = ttk.Frame(self.root, padding="10")
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        method_text = "J(0.87ωH)" if self.spectral_method == "j087wh" else "J(ωH)"
        title_text = f"Compare Datasets ({method_text})"
        ttk.Label(header_frame, text=title_text, 
                 font=("Arial", 16, "bold")).grid(row=0, column=0)
        
        ttk.Button(header_frame, text="← Back to Plot Menu", 
                  command=lambda: self.go_back_to_plot_menu()).grid(row=0, column=1, padx=(20, 0))
        
        # Main content
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # File 1 selection
        ttk.Label(main_frame, text="Dataset 1 (Reference):").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        file1_frame = ttk.Frame(main_frame)
        file1_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        file1_frame.columnconfigure(0, weight=1)
        
        self.file1_var = tk.StringVar(value="No file selected")
        ttk.Label(file1_frame, textvariable=self.file1_var, 
                 relief="sunken", padding="5").grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(file1_frame, text="Browse", 
                  command=self.browse_file1).grid(row=0, column=1, padx=(10, 0))
        
        # File 2 selection
        ttk.Label(main_frame, text="Dataset 2 (Comparison):").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file2_frame = ttk.Frame(main_frame)
        file2_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        file2_frame.columnconfigure(0, weight=1)
        
        self.file2_var = tk.StringVar(value="No file selected")
        ttk.Label(file2_frame, textvariable=self.file2_var, 
                 relief="sunken", padding="5").grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(file2_frame, text="Browse", 
                  command=self.browse_file2).grid(row=0, column=1, padx=(10, 0))
        
        # Labels
        ttk.Label(main_frame, text="Label 1:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.label1_var = tk.StringVar(value="Dataset 1")
        ttk.Entry(main_frame, textvariable=self.label1_var, 
                 width=20).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(main_frame, text="Label 2:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.label2_var = tk.StringVar(value="Dataset 2")
        ttk.Entry(main_frame, textvariable=self.label2_var, 
                 width=20).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Column selection
        ttk.Label(main_frame, text="Columns to Compare:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.columns_var = tk.StringVar(value="R1,R2,hetNOE")
        ttk.Entry(main_frame, textvariable=self.columns_var, 
                 width=40).grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        ttk.Label(main_frame, text="(comma-separated, e.g., R1,R2,hetNOE,S2)", 
                 font=("Arial", 9)).grid(row=5, column=1, sticky=tk.W, padx=(10, 0))
        
        # Field selection
        ttk.Label(main_frame, text="Field Selection:").grid(row=6, column=0, sticky=tk.W, pady=5)
        
        field_frame = ttk.Frame(main_frame)
        field_frame.grid(row=6, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        self.field_var = tk.StringVar(value="f1")
        ttk.Radiobutton(field_frame, text="f1", variable=self.field_var, 
                       value="f1").grid(row=0, column=0, padx=(0, 10))
        ttk.Radiobutton(field_frame, text="f2", variable=self.field_var, 
                       value="f2").grid(row=0, column=1, padx=(0, 10))
        ttk.Radiobutton(field_frame, text="both", variable=self.field_var, 
                       value="both").grid(row=0, column=2)
        
        # Output file
        ttk.Label(main_frame, text="Output File:").grid(row=7, column=0, sticky=tk.W, pady=5)
        self.output_var = tk.StringVar(value="comparison_plot.pdf")
        ttk.Entry(main_frame, textvariable=self.output_var, 
                 width=30).grid(row=7, column=1, sticky=(tk.W, tk.E), pady=5, padx=(10, 0))
        
        # Start comparison button
        ttk.Button(main_frame, text="Start Comparison", 
                  command=self.start_comparison,
                  style="Accent.TButton").grid(row=8, column=0, columnspan=2, pady=20)
        
        # Results display area
        results_frame = ttk.LabelFrame(main_frame, text="Comparison Results", padding="10")
        results_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(9, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)
    
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
        self.results_text.delete(1.0, tk.END)
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


def main():
    """Main function to start the GUI"""
    root = tk.Tk()
    app = DynamiXsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()