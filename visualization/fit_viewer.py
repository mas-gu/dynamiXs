"""
T1/T2 Fit Viewer Module - Interactive visualization of T1/T2 fitting results

Displays exponential decay fits with measured data points and fitted curves
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import customtkinter as ctk
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from pathlib import Path

# Import GUI components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from gui_components import (
    BG_COLOR, PANEL_BG_COLOR, FRAME_BG_COLOR,
    PRIMARY_TEXT, SECONDARY_TEXT,
    PRIMARY_BUTTON_BG, PRIMARY_BUTTON_HOVER,
    SECONDARY_BUTTON_BG, SECONDARY_BUTTON_HOVER,
    SPACING_XS, SPACING_SM, SPACING_MD,
    FONT_SECTION_LABEL, FONT_BODY, FONT_SMALL,
    create_primary_button, create_secondary_button, create_label
)


class FitViewer(tk.Toplevel):
    """
    Interactive viewer for T1/T2 fitting results

    Features:
    - JSON folder selection
    - Multi-residue selection (max 4 plots)
    - Field selection (F1, F2, Overlay)
    - Measurement type selection (T1, T2)
    - 4x1 vertical subplot layout
    """

    def __init__(self, parent, json_folder=None):
        """
        Initialize the Fit Viewer

        Parameters:
        -----------
        parent : tk widget
            Parent window
        json_folder : str, optional
            Path to JSON data folder
        """
        super().__init__(parent)

        self.title("T1/T2 Fit Viewer")
        self.geometry("1400x900")
        self.configure(bg=BG_COLOR)

        # Data storage
        self.json_folder = json_folder
        self.data = {}  # {field_name_type: json_data}
        self.available_residues = []
        self.selected_residues = []

        # UI state
        self.field_mode = tk.StringVar(value="field1")
        self.show_t1 = tk.BooleanVar(value=True)
        self.show_t2 = tk.BooleanVar(value=True)
        self.residue_vars = {}  # {residue: BooleanVar}

        # Build UI
        self._create_ui()

        # Load data if folder provided
        if json_folder and os.path.exists(json_folder):
            self._load_json_folder(json_folder)

    def _create_ui(self):
        """Create the user interface"""
        # Main container (horizontal split)
        main_container = ctk.CTkFrame(self, fg_color=BG_COLOR, corner_radius=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Left panel - Plots (75% width)
        left_panel = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR, corner_radius=8)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, SPACING_SM))

        # Right panel - Navigator (25% width)
        right_panel = ctk.CTkFrame(main_container, fg_color=PANEL_BG_COLOR, corner_radius=8)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(SPACING_SM, 0))
        right_panel.configure(width=350)

        # Create panels
        self._create_plot_panel(left_panel)
        self._create_navigator_panel(right_panel)

    def _create_plot_panel(self, parent):
        """Create the plot display panel"""
        # Header
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill=tk.X, padx=SPACING_MD, pady=(SPACING_MD, 0))

        title_label = create_label(header_frame, text="Fit Visualization",
                                   font=("SF Pro", 18, "bold"))
        title_label.pack(side=tk.LEFT)

        # Plot area
        plot_frame = ctk.CTkFrame(parent, fg_color=FRAME_BG_COLOR, corner_radius=8)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=SPACING_MD)

        # Create matplotlib figure (4x1 layout)
        self.figure = Figure(figsize=(10, 12), facecolor=PANEL_BG_COLOR)
        self.axes = []

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar_frame = ctk.CTkFrame(plot_frame, fg_color="transparent")
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Initial blank state
        self._show_blank_state()

    def _create_navigator_panel(self, parent):
        """Create the navigator control panel"""
        # Header
        header_frame = ctk.CTkFrame(parent, fg_color="transparent")
        header_frame.pack(fill=tk.X, padx=SPACING_MD, pady=(SPACING_MD, SPACING_SM))

        title_label = create_label(header_frame, text="Peak Navigator",
                                   font=("SF Pro", 16, "bold"))
        title_label.pack(side=tk.LEFT)

        # Scrollable content
        scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=SPACING_MD, pady=(0, SPACING_MD))

        # === JSON Folder Selection ===
        folder_section = ctk.CTkFrame(scroll_frame, fg_color=FRAME_BG_COLOR, corner_radius=8)
        folder_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        create_label(folder_section, text="JSON Data Folder:",
                    font=FONT_SECTION_LABEL).pack(anchor=tk.W, padx=SPACING_SM, pady=(SPACING_SM, SPACING_XS))

        self.folder_var = tk.StringVar(value=self.json_folder if self.json_folder else "No folder selected")
        folder_label = ctk.CTkLabel(folder_section, textvariable=self.folder_var,
                                   fg_color=BG_COLOR, corner_radius=4,
                                   font=FONT_SMALL, text_color=SECONDARY_TEXT,
                                   anchor="w", wraplength=280)
        folder_label.pack(fill=tk.X, padx=SPACING_SM, pady=(0, SPACING_SM))

        browse_btn = create_primary_button(folder_section, text="Browse Folder",
                                          command=self._browse_json_folder, width=280)
        browse_btn.pack(padx=SPACING_SM, pady=(0, SPACING_SM))

        # === Field Selection ===
        field_section = ctk.CTkFrame(scroll_frame, fg_color=FRAME_BG_COLOR, corner_radius=8)
        field_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        create_label(field_section, text="Field:",
                    font=FONT_SECTION_LABEL).pack(anchor=tk.W, padx=SPACING_SM, pady=(SPACING_SM, SPACING_XS))

        self.field1_radio = ctk.CTkRadioButton(field_section, text="Field 1",
                                              variable=self.field_mode, value="field1",
                                              fg_color=PRIMARY_BUTTON_BG,
                                              hover_color=PRIMARY_BUTTON_HOVER,
                                              font=FONT_BODY)
        self.field1_radio.pack(anchor=tk.W, padx=SPACING_SM, pady=(0, SPACING_XS))

        self.field2_radio = ctk.CTkRadioButton(field_section, text="Field 2",
                                              variable=self.field_mode, value="field2",
                                              fg_color=PRIMARY_BUTTON_BG,
                                              hover_color=PRIMARY_BUTTON_HOVER,
                                              font=FONT_BODY)
        self.field2_radio.pack(anchor=tk.W, padx=SPACING_SM, pady=(0, SPACING_XS))

        self.overlay_radio = ctk.CTkRadioButton(field_section, text="Overlay Both",
                                               variable=self.field_mode, value="overlay",
                                               fg_color=PRIMARY_BUTTON_BG,
                                               hover_color=PRIMARY_BUTTON_HOVER,
                                               font=FONT_BODY)
        self.overlay_radio.pack(anchor=tk.W, padx=SPACING_SM, pady=(0, SPACING_SM))

        # === Measurement Type ===
        type_section = ctk.CTkFrame(scroll_frame, fg_color=FRAME_BG_COLOR, corner_radius=8)
        type_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        create_label(type_section, text="Measurement Type:",
                    font=FONT_SECTION_LABEL).pack(anchor=tk.W, padx=SPACING_SM, pady=(SPACING_SM, SPACING_XS))

        t1_check = ctk.CTkCheckBox(type_section, text="T1", variable=self.show_t1,
                                  command=self._update_max_residues,
                                  fg_color=PRIMARY_BUTTON_BG,
                                  hover_color=PRIMARY_BUTTON_HOVER,
                                  font=FONT_BODY)
        t1_check.pack(anchor=tk.W, padx=SPACING_SM, pady=(0, SPACING_XS))

        t2_check = ctk.CTkCheckBox(type_section, text="T2", variable=self.show_t2,
                                  command=self._update_max_residues,
                                  fg_color=PRIMARY_BUTTON_BG,
                                  hover_color=PRIMARY_BUTTON_HOVER,
                                  font=FONT_BODY)
        t2_check.pack(anchor=tk.W, padx=SPACING_SM, pady=(0, SPACING_SM))

        # === Residue Selection ===
        residue_section = ctk.CTkFrame(scroll_frame, fg_color=FRAME_BG_COLOR, corner_radius=8)
        residue_section.pack(fill=tk.X, pady=(0, SPACING_MD))

        create_label(residue_section, text="Select Residues (max 4 plots):",
                    font=FONT_SECTION_LABEL).pack(anchor=tk.W, padx=SPACING_SM, pady=(SPACING_SM, SPACING_XS))

        # Scrollable residue list
        self.residue_list_frame = ctk.CTkScrollableFrame(residue_section,
                                                         fg_color=BG_COLOR,
                                                         height=250)
        self.residue_list_frame.pack(fill=tk.BOTH, expand=True, padx=SPACING_SM, pady=(0, SPACING_SM))

        # === Action Buttons ===
        button_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        button_frame.pack(fill=tk.X, pady=(SPACING_MD, 0))

        clear_btn = create_secondary_button(button_frame, text="Clear Selection",
                                           command=self._clear_selection, width=135)
        clear_btn.pack(side=tk.LEFT, padx=(0, SPACING_XS))

        update_btn = create_primary_button(button_frame, text="Update Plots",
                                          command=self._update_plots, width=135)
        update_btn.pack(side=tk.LEFT)

    def _browse_json_folder(self):
        """Browse for JSON data folder"""
        folder_path = filedialog.askdirectory(
            title="Select JSON Data Folder",
            initialdir=self.json_folder if self.json_folder else os.getcwd()
        )

        if folder_path:
            self._load_json_folder(folder_path)

    def _load_json_folder(self, folder_path):
        """Load JSON files from folder"""
        try:
            self.json_folder = folder_path
            self.folder_var.set(folder_path)
            self.data = {}
            self.available_residues = []

            # Scan for JSON files matching pattern: field*_T*_fit_data.json
            json_files = list(Path(folder_path).glob("*_fit_data.json"))

            if not json_files:
                messagebox.showwarning("No Data", "No fit data JSON files found in selected folder.")
                self._populate_residue_list()
                return

            # Load each JSON file
            residues_set = set()
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    # Extract file key (e.g., "field1_T1")
                    filename = json_file.stem
                    # Remove "_fit_data" suffix
                    file_key = filename.replace("_fit_data", "")

                    self.data[file_key] = data

                    # Collect residues
                    for fit in data['fits']:
                        residues_set.add(fit['residue'])

                except Exception as e:
                    print(f"Error loading {json_file}: {e}")

            # Sort residues
            self.available_residues = sorted(list(residues_set), key=self._residue_sort_key)

            # Update UI
            self._update_field_controls()
            self._populate_residue_list()

            # Show status
            n_files = len(self.data)
            n_residues = len(self.available_residues)
            messagebox.showinfo("Data Loaded",
                              f"Loaded {n_files} dataset(s) with {n_residues} residue(s)")

        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading JSON folder:\n{str(e)}")

    def _residue_sort_key(self, residue_str):
        """Extract numeric part of residue for sorting"""
        import re
        numbers = re.findall(r'\d+', residue_str)
        return int(numbers[0]) if numbers else 0

    def _update_field_controls(self):
        """Enable/disable field controls based on available data"""
        has_field1 = any('field1' in key for key in self.data.keys())
        has_field2 = any('field2' in key for key in self.data.keys())

        # Enable/disable radio buttons
        self.field1_radio.configure(state="normal" if has_field1 else "disabled")
        self.field2_radio.configure(state="normal" if has_field2 else "disabled")
        self.overlay_radio.configure(state="normal" if (has_field1 and has_field2) else "disabled")

        # Set default selection
        if has_field1 and has_field2:
            self.field_mode.set("overlay")
        elif has_field1:
            self.field_mode.set("field1")
        elif has_field2:
            self.field_mode.set("field2")

    def _populate_residue_list(self):
        """Populate the residue selection list"""
        # Clear existing
        for widget in self.residue_list_frame.winfo_children():
            widget.destroy()

        self.residue_vars = {}

        if not self.available_residues:
            no_data_label = create_label(self.residue_list_frame,
                                        text="No data available.\nSelect a JSON data folder.",
                                        font=FONT_SMALL)
            no_data_label.pack(pady=SPACING_MD)
            return

        # Create checkboxes for each residue
        for residue in self.available_residues:
            var = tk.BooleanVar(value=False)
            self.residue_vars[residue] = var

            checkbox = ctk.CTkCheckBox(self.residue_list_frame, text=residue,
                                      variable=var,
                                      command=self._on_residue_selected,
                                      fg_color=PRIMARY_BUTTON_BG,
                                      hover_color=PRIMARY_BUTTON_HOVER,
                                      font=FONT_BODY)
            checkbox.pack(anchor=tk.W, pady=SPACING_XS)

    def _on_residue_selected(self):
        """Handle residue selection change"""
        # Count selected residues
        selected = [res for res, var in self.residue_vars.items() if var.get()]

        # Calculate max allowed based on measurement types
        n_types = int(self.show_t1.get()) + int(self.show_t2.get())
        max_residues = 4 // n_types if n_types > 0 else 4

        # If exceeded, disable unchecked boxes
        if len(selected) >= max_residues:
            for residue, var in self.residue_vars.items():
                if not var.get():
                    # Find the checkbox widget and disable it
                    for widget in self.residue_list_frame.winfo_children():
                        if isinstance(widget, ctk.CTkCheckBox) and widget.cget("text") == residue:
                            widget.configure(state="disabled")
        else:
            # Re-enable all checkboxes
            for widget in self.residue_list_frame.winfo_children():
                if isinstance(widget, ctk.CTkCheckBox):
                    widget.configure(state="normal")

    def _update_max_residues(self):
        """Update maximum residues based on measurement types"""
        # Recalculate limits and update checkboxes
        self._on_residue_selected()

    def _clear_selection(self):
        """Clear all residue selections"""
        for var in self.residue_vars.values():
            var.set(False)
        self._on_residue_selected()
        self._show_blank_state()

    def _update_plots(self):
        """Update plots with current selection"""
        # Get selected residues
        selected = [res for res, var in self.residue_vars.items() if var.get()]

        if not selected:
            messagebox.showinfo("No Selection", "Please select at least one residue.")
            return

        # Check measurement types
        if not self.show_t1.get() and not self.show_t2.get():
            messagebox.showinfo("No Measurement Type", "Please select at least one measurement type (T1 or T2).")
            return

        # Generate plots
        self._generate_plots(selected)

    def _generate_plots(self, residues):
        """Generate plots for selected residues"""
        self.figure.clear()
        self.axes = []

        # Determine plot count
        plots = []
        for residue in sorted(residues, key=self._residue_sort_key):
            if self.show_t1.get():
                plots.append((residue, "T1"))
            if self.show_t2.get():
                plots.append((residue, "T2"))

        n_plots = len(plots)
        if n_plots == 0:
            self._show_blank_state()
            return

        # Create subplots (4x1 layout)
        for idx, (residue, meas_type) in enumerate(plots):
            ax = self.figure.add_subplot(n_plots, 1, idx + 1)
            self.axes.append(ax)

            # Plot data
            self._plot_single_fit(ax, residue, meas_type)

        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_single_fit(self, ax, residue, meas_type):
        """Plot single fit on given axes"""
        field_mode = self.field_mode.get()

        if field_mode == "overlay":
            self._plot_overlay(ax, residue, meas_type)
        else:
            self._plot_single_field(ax, residue, meas_type, field_mode)

    def _plot_single_field(self, ax, residue, meas_type, field):
        """Plot single field data"""
        # Get data key
        data_key = f"{field}_{meas_type}"

        if data_key not in self.data:
            ax.text(0.5, 0.5, f"No {meas_type} data for {field}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Find residue in data
        fit_data = None
        for fit in self.data[data_key]['fits']:
            if fit['residue'] == residue:
                fit_data = fit
                break

        if not fit_data:
            ax.text(0.5, 0.5, f"No data for residue {residue}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Extract data
        metadata = self.data[data_key]['metadata']
        time_points = metadata['time_points']
        intensities = fit_data['intensities']
        fit_time = fit_data['fit_curve']['time']
        fit_intensity = fit_data['fit_curve']['intensity']
        t_value = fit_data['t2']
        t_error = fit_data['t2_err']

        # Plot
        ax.plot(time_points, intensities, 'bo', markersize=8, label='Data')
        ax.plot(fit_time, fit_intensity, 'b-', linewidth=2, label='Fit')

        # Annotation
        field_freq = metadata['field_freq']
        time_units = metadata['time_units']
        textstr = f"{meas_type} ({field_freq} MHz) = {t_value:.2f} ± {t_error:.2f} {time_units}"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels
        ax.set_xlabel(f"Time ({time_units})")
        ax.set_ylabel("Signal Intensity")
        ax.set_title(f"Residue {residue} - {meas_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_overlay(self, ax, residue, meas_type):
        """Plot overlay of both fields"""
        # Get data for both fields
        field1_key = f"field1_{meas_type}"
        field2_key = f"field2_{meas_type}"

        has_field1 = field1_key in self.data
        has_field2 = field2_key in self.data

        if not has_field1 and not has_field2:
            ax.text(0.5, 0.5, f"No {meas_type} data available",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Find residue data
        fit1 = None
        fit2 = None

        if has_field1:
            for fit in self.data[field1_key]['fits']:
                if fit['residue'] == residue:
                    fit1 = fit
                    break

        if has_field2:
            for fit in self.data[field2_key]['fits']:
                if fit['residue'] == residue:
                    fit2 = fit
                    break

        if not fit1 and not fit2:
            ax.text(0.5, 0.5, f"No data for residue {residue}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Plot Field 1 (blue)
        annotations = []
        time_units = "ms"

        if fit1:
            metadata1 = self.data[field1_key]['metadata']
            time_points1 = metadata1['time_points']
            intensities1 = fit1['intensities']
            fit_time1 = fit1['fit_curve']['time']
            fit_intensity1 = fit1['fit_curve']['intensity']

            ax.plot(time_points1, intensities1, 'bo', markersize=8, label=f"{metadata1['field_freq']} MHz data")
            ax.plot(fit_time1, fit_intensity1, 'b-', linewidth=2, label=f"{metadata1['field_freq']} MHz fit")

            annotations.append(f"{meas_type} ({metadata1['field_freq']} MHz) = {fit1['t2']:.2f} ± {fit1['t2_err']:.2f} {metadata1['time_units']}")
            time_units = metadata1['time_units']

        # Plot Field 2 (red)
        if fit2:
            metadata2 = self.data[field2_key]['metadata']
            time_points2 = metadata2['time_points']
            intensities2 = fit2['intensities']
            fit_time2 = fit2['fit_curve']['time']
            fit_intensity2 = fit2['fit_curve']['intensity']

            ax.plot(time_points2, intensities2, 'ro', markersize=8, label=f"{metadata2['field_freq']} MHz data")
            ax.plot(fit_time2, fit_intensity2, 'r-', linewidth=2, label=f"{metadata2['field_freq']} MHz fit")

            annotations.append(f"{meas_type} ({metadata2['field_freq']} MHz) = {fit2['t2']:.2f} ± {fit2['t2_err']:.2f} {metadata2['time_units']}")
            time_units = metadata2['time_units']

        # Annotation
        textstr = '\n'.join(annotations)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels
        ax.set_xlabel(f"Time ({time_units})")
        ax.set_ylabel("Signal Intensity")
        ax.set_title(f"Residue {residue} - {meas_type}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _show_blank_state(self):
        """Show blank plot with instruction message"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5,
               "Select residues and click 'Update Plots' to display fits",
               ha='center', va='center',
               fontsize=16, color=SECONDARY_TEXT,
               transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()
