"""
Results Viewer Module - Interactive visualization of model-free analysis results

Displays fitted parameters (R1, R2, hetNOE, J values, S², τe, Rex) vs residue sequence
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import os
from pathlib import Path

# Import GUI components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gui_components import (
    BG_COLOR, PANEL_BG_COLOR, FRAME_BG_COLOR,
    PRIMARY_TEXT, SECONDARY_TEXT,
    PRIMARY_BUTTON_BG, PRIMARY_BUTTON_HOVER,
    SECONDARY_BUTTON_BG, SECONDARY_BUTTON_HOVER,
    SUCCESS_GREEN, WARNING_ORANGE, ERROR_RED,
    SPACING_XS, SPACING_SM, SPACING_MD,
    FONT_SECTION_LABEL, FONT_BODY, FONT_SMALL,
    create_primary_button, create_secondary_button, create_label
)


class ResultsViewer(tk.Toplevel):
    """
    Interactive viewer for model-free analysis results

    Features:
    - Parameter selection (R1, R2, hetNOE, J values, S², τe, Rex)
    - Field selection (Field 1, Field 2, Overlay)
    - Error bar display
    - Export functionality
    """

    def __init__(self, parent, results_file=None, field1_freq=600.0, field2_freq=None, is_dual_field=False):
        """
        Initialize the Results Viewer

        Parameters:
        -----------
        parent : tk widget
            Parent window
        results_file : str, optional
            Path to the CSV results file (can be None for blank state)
        field1_freq : float
            Field 1 frequency in MHz (default: 600.0)
        field2_freq : float, optional
            Field 2 frequency in MHz (for dual-field analysis)
        is_dual_field : bool
            Whether dual-field analysis was performed
        """
        super().__init__(parent)

        self.title("Model-Free Results Viewer")
        self.geometry("1400x900")
        self.configure(bg=BG_COLOR)

        # Store parameters
        self.results_file = results_file
        self.field1_freq = field1_freq
        self.field2_freq = field2_freq
        self.is_dual_field = is_dual_field

        # Load data
        self.df = None
        self.current_parameter = None
        self.current_field = "field1"
        self.show_errors = tk.BooleanVar(value=True)
        self.field_section = None  # Will hold field selection controls
        self.field_var = tk.StringVar(value="field1")  # Initialize early

        # Load results if file provided
        if results_file:
            self._load_results()
            # Detect if loaded data has dual-field columns
            self._detect_dual_field()

        # Build UI
        self._create_ui()

        # Initial blank state - no plot until user selects parameter
        self._show_blank_state()

    def _load_results(self):
        """Load results from CSV file"""
        try:
            if os.path.exists(self.results_file):
                self.df = pd.read_csv(self.results_file)
                print(f"Loaded results: {len(self.df)} residues")
                print(f"Columns: {list(self.df.columns)}")
            else:
                messagebox.showwarning("File Not Found",
                                      f"Results file not found:\n{self.results_file}\n\nYou can load a CSV file using the 'Load CSV File' button.")
                self.df = None
        except Exception as e:
            messagebox.showerror("Load Error",
                               f"Error loading results:\n{str(e)}")
            self.df = None

    def _load_csv_file(self):
        """Open file dialog to load a CSV file"""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Results CSV File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            initialdir=os.path.dirname(self.results_file) if self.results_file else os.getcwd()
        )

        if not file_path:
            return  # User cancelled

        # Try to load the selected file
        try:
            self.df = pd.read_csv(file_path)
            self.results_file = file_path
            print(f"Loaded results: {len(self.df)} residues")
            print(f"Columns: {list(self.df.columns)}")

            # Detect if dual-field data is present
            self._detect_dual_field()

            # Update field controls visibility
            self._update_field_controls()

            # Update parameter buttons availability
            self._update_parameter_buttons()

            # Show blank state (waiting for parameter selection)
            self._show_blank_state()

            messagebox.showinfo("File Loaded",
                              f"Successfully loaded {len(self.df)} residues from:\n{os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Load Error",
                               f"Error loading CSV file:\n{str(e)}")

    def _detect_dual_field(self):
        """Detect if loaded CSV contains dual-field data"""
        if self.df is None:
            return

        # Check if any column has _f2 or _field2 suffix
        has_field2 = any('_f2' in col or '_field2' in col or '_2' in col
                        for col in self.df.columns)

        if has_field2:
            self.is_dual_field = True
            # Set default field2 frequency if not already set
            if self.field2_freq is None:
                self.field2_freq = 700.0  # Default
        else:
            self.is_dual_field = False

    def _update_field_controls(self):
        """Show or hide field controls based on dual-field detection"""
        if not hasattr(self, 'control_inner'):
            return  # UI not built yet

        # Remove existing field section if it exists
        if self.field_section is not None:
            self.field_section.destroy()
            self.field_section = None

        # Create field section if dual-field data detected
        if self.is_dual_field:
            self.field_section = ctk.CTkFrame(self.control_inner, fg_color="transparent")
            self.field_section.pack(fill=tk.X, pady=(SPACING_SM, 0), before=self.options_section)

            create_label(self.field_section, text="Field:",
                        font=FONT_SECTION_LABEL).pack(side=tk.LEFT, padx=(0, SPACING_MD))

            field1_btn = ctk.CTkRadioButton(self.field_section,
                                           text=f"Field 1 ({self.field1_freq} MHz)",
                                           variable=self.field_var, value="field1",
                                           command=self._on_field_change,
                                           fg_color=PRIMARY_BUTTON_BG,
                                           hover_color=PRIMARY_BUTTON_HOVER,
                                           font=FONT_BODY)
            field1_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

            field2_btn = ctk.CTkRadioButton(self.field_section,
                                           text=f"Field 2 ({self.field2_freq} MHz)",
                                           variable=self.field_var, value="field2",
                                           command=self._on_field_change,
                                           fg_color=PRIMARY_BUTTON_BG,
                                           hover_color=PRIMARY_BUTTON_HOVER,
                                           font=FONT_BODY)
            field2_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

            overlay_btn = ctk.CTkRadioButton(self.field_section,
                                            text="Overlay Both",
                                            variable=self.field_var, value="overlay",
                                            command=self._on_field_change,
                                            fg_color=PRIMARY_BUTTON_BG,
                                            hover_color=PRIMARY_BUTTON_HOVER,
                                            font=FONT_BODY)
            overlay_btn.pack(side=tk.LEFT)

    def _update_parameter_buttons(self):
        """Update parameter button availability based on loaded data"""
        if self.df is None:
            return

        # Re-check which parameters are available
        for param, btn in self.parameter_buttons.items():
            if self._find_column_with_field(param):
                btn.configure(state="normal")
            else:
                btn.configure(state="disabled")

    def _create_ui(self):
        """Create the user interface"""
        # Main container
        main_container = ctk.CTkFrame(self, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Header
        header_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        header_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        title_label = create_label(header_frame, text="Model-Free Results Viewer",
                                   font=("SF Pro", 20, "bold"))
        title_label.pack(side=tk.LEFT)

        # Add Load CSV File button
        load_csv_btn = create_primary_button(header_frame, text="Load CSV File",
                                             command=self._load_csv_file, width=120)
        load_csv_btn.pack(side=tk.RIGHT, padx=(SPACING_SM, 0))

        close_btn = create_secondary_button(header_frame, text="✕ Close",
                                           command=self.destroy, width=80)
        close_btn.pack(side=tk.RIGHT)

        # Control panel
        self._create_control_panel(main_container)

        # Plot area
        self._create_plot_area(main_container)

    def _create_control_panel(self, parent):
        """Create the control panel with parameter/field selectors"""
        control_frame = ctk.CTkFrame(parent, fg_color=PANEL_BG_COLOR, corner_radius=8)
        control_frame.pack(fill=tk.X, pady=(0, SPACING_MD))

        # Inner container for padding
        self.control_inner = ctk.CTkFrame(control_frame, fg_color="transparent")
        self.control_inner.pack(fill=tk.X, padx=SPACING_MD, pady=SPACING_MD)

        # Parameter selection section
        param_section = ctk.CTkFrame(self.control_inner, fg_color="transparent")
        param_section.pack(fill=tk.X, pady=(0, SPACING_SM))

        create_label(param_section, text="Display Parameter:",
                    font=FONT_SECTION_LABEL).pack(side=tk.LEFT, padx=(0, SPACING_MD))

        # Add "Show All" button
        show_all_btn = create_primary_button(param_section, text="Show All",
                                            command=self._show_all_parameters,
                                            width=100)
        show_all_btn.pack(side=tk.LEFT, padx=(0, SPACING_MD))
        self.show_all_btn = show_all_btn

        # Determine available parameters from DataFrame columns
        self.parameter_buttons = {}
        self.selected_params = []  # Track multiple selections

        # Define parameter groups with multi-select capability
        self.param_groups = {
            "Relaxation": ["R1", "R2", "hetNOE"],
            "Spectral Density": ["J(0)", "J(wN)", "J(0.87wH)"],
            "Model-Free": ["S2", "te", "Rex"]
        }

        for group_name, params in self.param_groups.items():
            # Create group container
            group_frame = ctk.CTkFrame(param_section, fg_color="transparent")
            group_frame.pack(side=tk.LEFT, padx=(0, SPACING_MD))

            create_label(group_frame, text=f"{group_name}:",
                        font=FONT_SMALL).pack(anchor=tk.W)

            button_row = ctk.CTkFrame(group_frame, fg_color="transparent")
            button_row.pack(fill=tk.X)

            for param in params:
                # Check if parameter exists in dataframe
                if self._find_column_with_field(param):
                    btn = create_secondary_button(button_row, text=param,
                                                 command=lambda p=param: self._toggle_parameter(p),
                                                 width=100)
                    btn.pack(side=tk.LEFT, padx=(0, SPACING_XS))
                    self.parameter_buttons[param] = btn

        # Field selection section (only if dual-field)
        # This will be created/updated by _update_field_controls()
        if self.is_dual_field:
            self.field_section = ctk.CTkFrame(self.control_inner, fg_color="transparent")
            self.field_section.pack(fill=tk.X, pady=(SPACING_SM, 0))

            create_label(self.field_section, text="Field:",
                        font=FONT_SECTION_LABEL).pack(side=tk.LEFT, padx=(0, SPACING_MD))

            field1_btn = ctk.CTkRadioButton(self.field_section,
                                           text=f"Field 1 ({self.field1_freq} MHz)",
                                           variable=self.field_var, value="field1",
                                           command=self._on_field_change,
                                           fg_color=PRIMARY_BUTTON_BG,
                                           hover_color=PRIMARY_BUTTON_HOVER,
                                           font=FONT_BODY)
            field1_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

            field2_btn = ctk.CTkRadioButton(self.field_section,
                                           text=f"Field 2 ({self.field2_freq} MHz)",
                                           variable=self.field_var, value="field2",
                                           command=self._on_field_change,
                                           fg_color=PRIMARY_BUTTON_BG,
                                           hover_color=PRIMARY_BUTTON_HOVER,
                                           font=FONT_BODY)
            field2_btn.pack(side=tk.LEFT, padx=(0, SPACING_SM))

            overlay_btn = ctk.CTkRadioButton(self.field_section,
                                            text="Overlay Both",
                                            variable=self.field_var, value="overlay",
                                            command=self._on_field_change,
                                            fg_color=PRIMARY_BUTTON_BG,
                                            hover_color=PRIMARY_BUTTON_HOVER,
                                            font=FONT_BODY)
            overlay_btn.pack(side=tk.LEFT)

        # Options section
        self.options_section = ctk.CTkFrame(self.control_inner, fg_color="transparent")
        self.options_section.pack(fill=tk.X, pady=(SPACING_SM, 0))

        error_check = ctk.CTkCheckBox(self.options_section, text="Show error bars",
                                     variable=self.show_errors,
                                     command=self._update_plot,
                                     fg_color=PRIMARY_BUTTON_BG,
                                     hover_color=PRIMARY_BUTTON_HOVER,
                                     font=FONT_BODY)
        error_check.pack(side=tk.LEFT, padx=(0, SPACING_MD))

    def _create_plot_area(self, parent):
        """Create the matplotlib plot area"""
        plot_frame = ctk.CTkFrame(parent, fg_color=PANEL_BG_COLOR, corner_radius=8)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.figure = Figure(figsize=(14, 7), facecolor=PANEL_BG_COLOR)
        self.ax = self.figure.add_subplot(111)

        # Style the plot
        self.ax.set_facecolor(FRAME_BG_COLOR)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Add toolbar
        toolbar_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        toolbar_frame.pack(fill=tk.X, padx=SPACING_MD, pady=(0, SPACING_SM))

        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    def _find_column_with_field(self, param_name, field=None):
        """
        Find column name in dataframe that matches parameter and field

        Parameters:
        -----------
        param_name : str
            Parameter name (e.g., 'R1', 'S2')
        field : str, optional
            Field specification ('field1', 'field2', or None for any)

        Returns:
        --------
        str or None : Column name if found
        """
        if self.df is None:
            return None

        # Define variations for parameter names
        # Based on actual CSV output: R1_f1, R2_f1, hetNOE_f1, J0_f1, JwN_f1, JwH_087_f1, S2, tc, te, Rex_f1
        variations = {
            "R1": ["R1"],
            "R2": ["R2"],
            "hetNOE": ["hetNOE", "NOE", "noe"],
            "J(0)": ["J0", "J_0", "J(0)"],
            "J(wN)": ["JwN", "J_wN", "J(wN)", "JN"],
            "J(wH)": ["JwH", "J_wH", "J(wH)", "JH"],
            "J(0.87wH)": ["JwH_087", "J0.87wH", "J_0.87wH", "J(0.87wH)", "JwH0.87"],
            "S2": ["S2", "S²", "order_parameter"],
            "te": ["te", "tau_e", "taue"],
            "Rex": ["Rex", "R_ex"],
            "tc": ["tc", "tau_c", "tauc"]
        }

        # Get all possible base names
        base_names = [param_name]
        if param_name in variations:
            base_names.extend(variations[param_name])

        # Field suffixes to try (prioritize _f1/_f2 as that's what the output uses)
        if field == "field1":
            field_suffixes = ["_f1", "_field1", "_1", ""]
        elif field == "field2":
            field_suffixes = ["_f2", "_field2", "_2", ""]
        else:
            field_suffixes = ["", "_f1", "_f2", "_field1", "_field2", "_1", "_2"]

        # Try all combinations
        for base in base_names:
            for suffix in field_suffixes:
                col_name = f"{base}{suffix}"
                if col_name in self.df.columns:
                    return col_name

        return None

    def _find_column(self, param_name):
        """
        Find column name based on current field selection
        """
        # Check if this is a global parameter (no field suffix)
        # S2, tc, te are global; Rex has field-specific versions
        global_params = ["S2", "tc", "te"]

        if param_name in global_params:
            # These parameters don't have field suffixes
            return self._find_column_with_field(param_name, None)

        if self.current_field == "field1":
            return self._find_column_with_field(param_name, "field1")
        elif self.current_field == "field2":
            return self._find_column_with_field(param_name, "field2")
        else:
            # For overlay, return field1 first (will handle both in plot)
            return self._find_column_with_field(param_name, "field1")

    def _show_blank_state(self):
        """Show blank plot with instruction message"""
        self.ax.clear()

        if self.df is None:
            # No data loaded - show instructions to load CSV
            message = "No data loaded.\n\nClick 'Load CSV File' to load results."
        else:
            # Data loaded - waiting for parameter selection
            message = "Select a parameter above to display results"

        self.ax.text(0.5, 0.5, message,
                    ha='center', va='center',
                    fontsize=16, color=SECONDARY_TEXT,
                    transform=self.ax.transAxes)
        self.ax.axis('off')
        self.canvas.draw()

    def _toggle_parameter(self, param):
        """Toggle parameter selection (multi-select)"""
        if param in self.selected_params:
            # Deselect
            self.selected_params.remove(param)
        else:
            # Select - check if from same group
            param_group = self._get_param_group(param)

            # Remove any selected params from different groups
            self.selected_params = [p for p in self.selected_params
                                   if self._get_param_group(p) == param_group]

            # Add new selection
            self.selected_params.append(param)

        # Update button states
        for p, btn in self.parameter_buttons.items():
            if p in self.selected_params:
                btn.configure(fg_color=PRIMARY_BUTTON_BG,
                            hover_color=PRIMARY_BUTTON_HOVER,
                            text_color="white")
            else:
                btn.configure(fg_color=SECONDARY_BUTTON_BG,
                            hover_color=SECONDARY_BUTTON_HOVER,
                            text_color=PRIMARY_TEXT)

        # Reset Show All button
        self.show_all_btn.configure(fg_color=PRIMARY_BUTTON_BG,
                                    hover_color=PRIMARY_BUTTON_HOVER)

        # Update plot
        if self.selected_params:
            self.current_parameter = self.selected_params[0]  # Keep for compatibility
            self._update_plot_stacked()
        else:
            self._show_blank_state()

    def _get_param_group(self, param):
        """Get the group name for a parameter"""
        for group_name, params in self.param_groups.items():
            if param in params:
                return group_name
        return None

    def _show_all_parameters(self):
        """Display all available parameters in subplots"""
        self.current_parameter = "ALL"
        self.selected_params = []  # Clear individual selections

        # Reset individual parameter buttons
        for p, btn in self.parameter_buttons.items():
            btn.configure(fg_color=SECONDARY_BUTTON_BG,
                        hover_color=SECONDARY_BUTTON_HOVER,
                        text_color=PRIMARY_TEXT)

        # Highlight Show All button
        self.show_all_btn.configure(fg_color=SUCCESS_GREEN,
                                    hover_color=SUCCESS_GREEN)

        # Update plot
        self._update_plot_all()

    def _on_field_change(self):
        """Handle field selection change"""
        self.current_field = self.field_var.get()
        if self.selected_params:
            self._update_plot_stacked()
        elif self.current_parameter == "ALL":
            self._update_plot_all()
        else:
            self._update_plot()

    def _update_plot(self):
        """Update the plot with current selection"""
        if self.current_parameter is None or self.current_parameter == "ALL":
            self._show_blank_state()
            return

        # Clear plot
        self.ax.clear()

        # Get residue column (try common names)
        residue_col = None
        for col in ['Residue', 'residue', 'Res', 'res', 'ResidueNum']:
            if col in self.df.columns:
                residue_col = col
                break

        if residue_col is None:
            # Use index as residue numbers
            residues = self.df.index.values
        else:
            residues = self._extract_residue_numbers(self.df[residue_col].values)

        # Handle overlay mode
        if self.is_dual_field and self.current_field == "overlay":
            self._plot_overlay(residues)
        else:
            self._plot_single_field(residues)

        # Adjust layout
        self.figure.tight_layout()

        # Redraw
        self.canvas.draw()

    def _update_plot_stacked(self):
        """Display multiple selected parameters in stacked vertical plots"""
        if not self.selected_params:
            self._show_blank_state()
            return

        # Clear figure and recreate with vertical subplots
        self.figure.clear()

        n_params = len(self.selected_params)

        # Get residue column
        residue_col = None
        for col in ['Residue', 'residue', 'Res', 'res', 'ResidueNum']:
            if col in self.df.columns:
                residue_col = col
                break

        if residue_col is None:
            residues = self.df.index.values
        else:
            residues = self._extract_residue_numbers(self.df[residue_col].values)

        # Create stacked subplots (vertical)
        for idx, param in enumerate(self.selected_params):
            ax = self.figure.add_subplot(n_params, 1, idx + 1)

            # Plot based on field selection
            if self.is_dual_field and self.current_field == "overlay":
                self._plot_param_overlay(ax, param, residues)
            else:
                self._plot_param_single(ax, param, residues)

            # Format x-axis only for bottom subplot
            if idx == n_params - 1:
                ax.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
                self._format_xaxis_for_ax(ax, residues)
                # Show x-tick labels
                ax.tick_params(labelbottom=True)
            else:
                # Hide x-tick labels for upper subplots
                ax.set_xlabel('')
                self._format_xaxis_for_ax(ax, residues)
                ax.tick_params(labelbottom=False)

            # Y-axis label
            ax.set_ylabel(self._get_ylabel(param), fontsize=10, fontweight='bold')

            # Grid and styling
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set y-axis to start at 0
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1] * 1.1)

        # Overall title
        group_name = self._get_param_group(self.selected_params[0])
        title = f"{group_name} Parameters vs Residue Sequence"
        if self.is_dual_field and self.current_field == "overlay":
            title += " (Overlay)"
        self.figure.suptitle(title, fontsize=14, fontweight='bold')

        # Adjust layout
        self.figure.tight_layout(rect=[0, 0, 1, 0.97])

        # Redraw
        self.canvas.draw()

    def _plot_param_single(self, ax, param, residues):
        """Plot a single parameter on given axis (single field)"""
        # Get column name
        col_name = self._find_column_with_field(param,
                    "field1" if self.current_field == "field1" else
                    "field2" if self.current_field == "field2" else None)

        if col_name is None:
            ax.text(0.5, 0.5, f"'{param}' not found",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Find error column
        error_col = None
        for suffix in ['err', '_err', '_error', 'Error']:
            potential_err = f"{col_name}{suffix}"
            if potential_err in self.df.columns:
                error_col = potential_err
                break

        # Get data
        values = self.df[col_name].values
        errors = self.df[error_col].values if error_col and self.show_errors.get() else None

        # Add grey bars for missing residues
        self._add_missing_residue_bars_for_ax(ax, residues, values)

        # Plot data
        if errors is not None and self.show_errors.get():
            ax.errorbar(residues, values, yerr=errors,
                       fmt='o', markersize=5, capsize=3,
                       color=PRIMARY_BUTTON_BG, ecolor=SECONDARY_TEXT, zorder=3)
        else:
            ax.plot(residues, values, 'o', markersize=5,
                   color=PRIMARY_BUTTON_BG, zorder=3)

    def _plot_param_overlay(self, ax, param, residues):
        """Plot a single parameter on given axis (overlay both fields)"""
        # Get columns for both fields
        col_field1 = self._find_column_with_field(param, "field1")
        col_field2 = self._find_column_with_field(param, "field2")

        # Collect values for grey bars
        all_values = []
        if col_field1:
            all_values.extend(self.df[col_field1].dropna().values)
        if col_field2:
            all_values.extend(self.df[col_field2].dropna().values)

        if all_values:
            self._add_missing_residue_bars_for_ax(ax, residues, np.array(all_values))

        colors = ['#5B9EE5', '#E8554E']  # Blue for field1, Red for field2

        for i, (col_name, field_label, color) in enumerate([
            (col_field1, f"F1 ({self.field1_freq} MHz)", colors[0]),
            (col_field2, f"F2 ({self.field2_freq} MHz)", colors[1])
        ]):
            if col_name is None:
                continue

            # Find error column
            error_col = None
            for suffix in ['err', '_err', '_error', 'Error']:
                potential_err = f"{col_name}{suffix}"
                if potential_err in self.df.columns:
                    error_col = potential_err
                    break

            # Get data
            values = self.df[col_name].values
            errors = self.df[error_col].values if error_col and self.show_errors.get() else None

            # Plot data
            if errors is not None and self.show_errors.get():
                ax.errorbar(residues, values, yerr=errors,
                           fmt='o', markersize=4, capsize=2,
                           color=color, ecolor=color, alpha=0.7,
                           label=field_label, zorder=3)
            else:
                ax.plot(residues, values, 'o', markersize=4,
                       color=color, alpha=0.7,
                       label=field_label, zorder=3)

        # Add legend for overlay
        ax.legend(loc='best', fontsize=8)

    def _plot_single_field(self, residues):
        """Plot data for a single field"""
        # Get column name
        col_name = self._find_column(self.current_parameter)
        if col_name is None:
            self.ax.text(0.5, 0.5,
                        f"Parameter '{self.current_parameter}' not found in results",
                        ha='center', va='center',
                        fontsize=14, color=ERROR_RED,
                        transform=self.ax.transAxes)
            self.ax.axis('off')
            return

        # Find error column
        error_col = None
        for suffix in ['err', '_err', '_error', 'Error']:
            potential_err = f"{col_name}{suffix}"
            if potential_err in self.df.columns:
                error_col = potential_err
                break

        # Get data
        values = self.df[col_name].values
        errors = self.df[error_col].values if error_col and self.show_errors.get() else None

        # Add grey bars for missing residues
        self._add_missing_residue_bars(residues, values)

        # Plot data
        field_label = f"{self.current_parameter}"
        if self.is_dual_field:
            field_name = f"Field {self.current_field[-1]}" if "field" in self.current_field else self.current_field
            field_label = f"{self.current_parameter} ({field_name})"

        if errors is not None and self.show_errors.get():
            self.ax.errorbar(residues, values, yerr=errors,
                           fmt='o', markersize=6, capsize=3,
                           color=PRIMARY_BUTTON_BG, ecolor=SECONDARY_TEXT,
                           label=field_label, zorder=3)
        else:
            self.ax.plot(residues, values, 'o', markersize=6,
                       color=PRIMARY_BUTTON_BG,
                       label=field_label, zorder=3)

        # Style plot
        self.ax.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
        self.ax.set_ylabel(self._get_ylabel(self.current_parameter),
                          fontsize=12, fontweight='bold')
        self.ax.set_title(f"{self.current_parameter} vs Residue Sequence",
                         fontsize=14, fontweight='bold', pad=20)
        self.ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Set x-axis limits and ticks
        self._format_xaxis(residues)

        # Set y-axis to start at 0
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(0, ylim[1] * 1.1)

    def _plot_overlay(self, residues):
        """Plot overlay of both fields"""
        # Get columns for both fields
        col_field1 = self._find_column_with_field(self.current_parameter, "field1")
        col_field2 = self._find_column_with_field(self.current_parameter, "field2")

        # Collect all values to determine y-range for grey bars
        all_values = []
        if col_field1:
            all_values.extend(self.df[col_field1].dropna().values)
        if col_field2:
            all_values.extend(self.df[col_field2].dropna().values)

        # Add grey bars for missing residues
        if all_values:
            self._add_missing_residue_bars(residues, np.array(all_values))

        colors = ['#5B9EE5', '#E8554E']  # Blue for field1, Red for field2

        for i, (col_name, field_label, color) in enumerate([
            (col_field1, f"Field 1 ({self.field1_freq} MHz)", colors[0]),
            (col_field2, f"Field 2 ({self.field2_freq} MHz)", colors[1])
        ]):
            if col_name is None:
                continue

            # Find error column
            error_col = None
            for suffix in ['err', '_err', '_error', 'Error']:
                potential_err = f"{col_name}{suffix}"
                if potential_err in self.df.columns:
                    error_col = potential_err
                    break

            # Get data
            values = self.df[col_name].values
            errors = self.df[error_col].values if error_col and self.show_errors.get() else None

            # Plot data
            if errors is not None and self.show_errors.get():
                self.ax.errorbar(residues, values, yerr=errors,
                               fmt='o', markersize=6, capsize=3,
                               color=color, ecolor=color, alpha=0.7,
                               label=field_label, zorder=3)
            else:
                self.ax.plot(residues, values, 'o', markersize=6,
                           color=color, alpha=0.7,
                           label=field_label, zorder=3)

        # Style plot
        self.ax.set_xlabel('Residue Number', fontsize=12, fontweight='bold')
        self.ax.set_ylabel(self._get_ylabel(self.current_parameter),
                          fontsize=12, fontweight='bold')
        self.ax.set_title(f"{self.current_parameter} vs Residue Sequence (Overlay)",
                         fontsize=14, fontweight='bold', pad=20)
        self.ax.grid(True, alpha=0.3, linestyle='--', zorder=1)
        self.ax.legend(loc='best')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Set x-axis limits and ticks
        self._format_xaxis(residues)

        # Set y-axis to start at 0
        ylim = self.ax.get_ylim()
        self.ax.set_ylim(0, ylim[1] * 1.1)

    def _update_plot_all(self):
        """Display all parameters in a grid of subplots"""
        # Clear figure and recreate with subplots
        self.figure.clear()

        # Get list of available parameters
        available_params = list(self.parameter_buttons.keys())
        n_params = len(available_params)

        if n_params == 0:
            self._show_blank_state()
            return

        # Calculate grid layout (prefer wider than tall)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        # Get residue column
        residue_col = None
        for col in ['Residue', 'residue', 'Res', 'res', 'ResidueNum']:
            if col in self.df.columns:
                residue_col = col
                break

        if residue_col is None:
            residues = self.df.index.values
        else:
            residues = self._extract_residue_numbers(self.df[residue_col].values)

        # Create subplots
        for idx, param in enumerate(available_params):
            ax = self.figure.add_subplot(n_rows, n_cols, idx + 1)

            # Plot based on field selection
            if self.is_dual_field and self.current_field == "overlay":
                self._plot_param_overlay_small(ax, param, residues)
            else:
                self._plot_param_single_small(ax, param, residues)

            # Set x-axis limits and ticks
            min_residue = 1
            max_residue = int(np.max(residues))
            ax.set_xlim(min_residue - 0.5, max_residue + 0.5)

            # Simplified tick formatting for small subplots
            if max_residue >= 20:
                # Show fewer ticks for small plots
                tick_step = 20 if max_residue > 50 else 10
                ticks = [1] + list(range(tick_step, max_residue + 1, tick_step))
                if max_residue not in ticks:
                    ticks.append(max_residue)
                ax.set_xticks(ticks)
                ax.set_xticklabels([str(t) for t in ticks])
            else:
                # For very small proteins, show every 5th
                ticks = [1] + list(range(5, max_residue + 1, 5))
                if max_residue not in ticks:
                    ticks.append(max_residue)
                ax.set_xticks(ticks)

            # Style
            ax.set_title(param, fontsize=10, fontweight='bold')
            ax.set_xlabel('Residue', fontsize=8)
            ax.set_ylabel(self._get_ylabel(param), fontsize=8)
            ax.grid(True, alpha=0.2, linestyle='--', zorder=1)
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set y-axis to start at 0
            ylim = ax.get_ylim()
            ax.set_ylim(0, ylim[1] * 1.1)

        # Adjust layout
        self.figure.tight_layout()

        # Redraw
        self.canvas.draw()

    def _plot_param_single_small(self, ax, param, residues):
        """Plot a single parameter on small subplot (single field)"""
        # Get column name
        col_name = self._find_column_with_field(param,
                    "field1" if self.current_field == "field1" else
                    "field2" if self.current_field == "field2" else None)

        if col_name is None:
            ax.text(0.5, 0.5, f"'{param}' not found",
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.axis('off')
            return

        # Find error column
        error_col = None
        for suffix in ['err', '_err', '_error', 'Error']:
            potential_err = f"{col_name}{suffix}"
            if potential_err in self.df.columns:
                error_col = potential_err
                break

        # Get data
        values = self.df[col_name].values
        errors = self.df[error_col].values if error_col and self.show_errors.get() else None

        # Add grey bars for missing residues
        self._add_missing_residue_bars_for_ax(ax, residues, values)

        # Plot data
        if errors is not None and self.show_errors.get():
            ax.errorbar(residues, values, yerr=errors,
                       fmt='o', markersize=3, capsize=2,
                       color=PRIMARY_BUTTON_BG, ecolor=SECONDARY_TEXT, zorder=3)
        else:
            ax.plot(residues, values, 'o', markersize=3,
                   color=PRIMARY_BUTTON_BG, zorder=3)

    def _plot_param_overlay_small(self, ax, param, residues):
        """Plot a single parameter on small subplot (overlay both fields)"""
        # Get columns for both fields
        col_field1 = self._find_column_with_field(param, "field1")
        col_field2 = self._find_column_with_field(param, "field2")

        # Collect values for grey bars
        all_values = []
        if col_field1:
            all_values.extend(self.df[col_field1].dropna().values)
        if col_field2:
            all_values.extend(self.df[col_field2].dropna().values)

        if all_values:
            self._add_missing_residue_bars_for_ax(ax, residues, np.array(all_values))

        colors = ['#5B9EE5', '#E8554E']  # Blue for field1, Red for field2

        for i, (col_name, color) in enumerate([
            (col_field1, colors[0]),
            (col_field2, colors[1])
        ]):
            if col_name is None:
                continue

            # Find error column
            error_col = None
            for suffix in ['err', '_err', '_error', 'Error']:
                potential_err = f"{col_name}{suffix}"
                if potential_err in self.df.columns:
                    error_col = potential_err
                    break

            # Get data
            values = self.df[col_name].values
            errors = self.df[error_col].values if error_col and self.show_errors.get() else None

            # Plot data (no labels for small plots to avoid clutter)
            if errors is not None and self.show_errors.get():
                ax.errorbar(residues, values, yerr=errors,
                           fmt='o', markersize=2, capsize=1,
                           color=color, ecolor=color, alpha=0.7, zorder=3)
            else:
                ax.plot(residues, values, 'o', markersize=2,
                       color=color, alpha=0.7, zorder=3)

    def _extract_residue_numbers(self, residue_labels):
        """
        Extract numeric residue numbers from residue labels

        Handles formats like:
        - "75.GLY" -> 75
        - "GLY75" -> 75
        - "75" -> 75
        - 75 -> 75

        Parameters:
        -----------
        residue_labels : array
            Array of residue labels (can be strings or numbers)

        Returns:
        --------
        array : Numeric residue numbers
        """
        numeric_residues = []

        for label in residue_labels:
            # If already numeric, use as-is
            if isinstance(label, (int, float, np.integer, np.floating)):
                numeric_residues.append(int(label))
                continue

            # Convert to string and try to extract number
            label_str = str(label)

            # Try to extract number from formats like "75.GLY", "GLY75", etc.
            import re
            numbers = re.findall(r'\d+', label_str)

            if numbers:
                # Take the first number found
                numeric_residues.append(int(numbers[0]))
            else:
                # Fallback: use index if no number found
                numeric_residues.append(len(numeric_residues) + 1)

        return np.array(numeric_residues)

    def _add_missing_residue_bars_for_ax(self, ax, residues, values):
        """Add grey bars for missing residues to a specific axis"""
        min_residue = 1
        max_residue = int(np.max(residues))

        present_residues = set(residues)
        all_residues = set(range(min_residue, max_residue + 1))
        missing_residues = sorted(all_residues - present_residues)

        if not missing_residues:
            return

        # Add grey bars for missing residues
        for res in missing_residues:
            ax.axvspan(res - 0.5, res + 0.5,
                      facecolor='lightgrey', alpha=0.3, zorder=0)

    def _format_xaxis_for_ax(self, ax, residues):
        """Format x-axis with ticks every 10 residues for a specific axis"""
        min_residue = 1
        max_residue = int(np.max(residues))

        ax.set_xlim(min_residue - 0.5, max_residue + 0.5)

        # Set ticks every 10 residues
        if max_residue >= 10:
            first_tick = 10
            ticks = list(range(first_tick, max_residue + 1, 10))
            if 1 not in ticks:
                ticks = [1] + ticks
            if max_residue not in ticks and max_residue - ticks[-1] > 2:
                ticks.append(max_residue)
        else:
            ticks = list(range(1, max_residue + 1))

        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    def _add_missing_residue_bars(self, residues, values):
        """
        Add grey bars for missing residues

        Parameters:
        -----------
        residues : array
            Array of residue numbers present in data
        values : array
            Array of values (used to determine y-axis range)
        """
        # Determine full residue range (1 to max)
        min_residue = 1
        max_residue = int(np.max(residues))

        # Find missing residues
        present_residues = set(residues)
        all_residues = set(range(min_residue, max_residue + 1))
        missing_residues = sorted(all_residues - present_residues)

        if not missing_residues:
            return

        # Get y-axis range from data
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return

        y_min = np.min(valid_values)
        y_max = np.max(valid_values)
        y_range = y_max - y_min
        y_buffer = y_range * 0.1  # 10% buffer

        # Add grey bars for missing residues
        for res in missing_residues:
            self.ax.axvspan(res - 0.5, res + 0.5,
                          facecolor='lightgrey', alpha=0.3, zorder=0)

    def _format_xaxis(self, residues):
        """
        Format x-axis with ticks every 10 residues

        Parameters:
        -----------
        residues : array
            Array of residue numbers present in data
        """
        # Set x-axis limits from 1 to max residue
        min_residue = 1
        max_residue = int(np.max(residues))

        self.ax.set_xlim(min_residue - 0.5, max_residue + 0.5)

        # Set ticks every 10 residues
        # Start at first multiple of 10, or at 1 if max < 10
        if max_residue >= 10:
            first_tick = 10
            ticks = list(range(first_tick, max_residue + 1, 10))
            # Add 1 at the beginning if not already there
            if 1 not in ticks:
                ticks = [1] + ticks
            # Add max residue at the end if not already there
            if max_residue not in ticks and max_residue - ticks[-1] > 2:
                ticks.append(max_residue)
        else:
            # For small proteins, show every residue
            ticks = list(range(1, max_residue + 1))

        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([str(t) for t in ticks])

    def _get_ylabel(self, param):
        """Get appropriate y-axis label for parameter"""
        labels = {
            "R1": "R₁ (s⁻¹)",
            "R2": "R₂ (s⁻¹)",
            "hetNOE": "hetNOE",
            "J(0)": "J(0) (ns/rad)",
            "J(wN)": "J(ωₙ) (ns/rad)",
            "J(wH)": "J(ωₕ) (ns/rad)",
            "J(0.87wH)": "J(0.87ωₕ) (ns/rad)",
            "S2": "S²",
            "tc": "τc (ns)",
            "te": "τₑ (ps)",
            "Rex": "Rₑₓ (s⁻¹)"
        }
        return labels.get(param, param)


def open_results_viewer(parent, results_file, field1_freq, field2_freq=None, is_dual_field=False):
    """
    Convenience function to open the results viewer

    Parameters:
    -----------
    parent : tk widget
        Parent window
    results_file : str
        Path to results CSV file
    field1_freq : float
        Field 1 frequency in MHz
    field2_freq : float, optional
        Field 2 frequency in MHz
    is_dual_field : bool
        Whether dual-field analysis was performed
    """
    viewer = ResultsViewer(parent, results_file, field1_freq, field2_freq, is_dual_field)
    return viewer
