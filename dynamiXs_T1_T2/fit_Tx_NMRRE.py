#!/usr/bin/env python3
"""
Enhanced T1/T2 NMR Relaxation Fitting Script with Configurable File Handling

This script performs exponential decay fitting for T1/T2 relaxation data with bootstrap error estimation.
RE VERSION: Enhanced with configurable input/output file management in main function.

Features:
- Exponential decay fitting using lmfit
- Bootstrap error estimation
- Configurable file paths in main function
- Multi-figure PDF output with customizable plots per figure
- Comprehensive results output
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from lmfit import Model
import pandas as pd
import os
from pathlib import Path


def binding(x, A, t2):
    """Exponential decay function for T1/T2 relaxation fitting"""
    return A * np.exp(-x / t2)


def bootstrap_errors(x, y, model, params, n_bootstrap=1000):
    """
    Bootstrap function to estimate parameter errors
    
    Parameters:
    -----------
    x : array
        Time points
    y : array
        Signal intensities
    model : lmfit.Model
        Fitting model
    params : lmfit.Parameters
        Initial parameters
    n_bootstrap : int
        Number of bootstrap iterations
        
    Returns:
    --------
    tuple : (A_error, t2_error)
    """
    a_values = []
    t2_values = []
    
    # Initial fit to get residuals
    result = model.fit(y, params, x=x)
    y_fit = result.best_fit
    residuals = y - y_fit
    
    for _ in range(n_bootstrap):      
        params_boot = model.make_params(A=params['A'].value, t2=params['t2'].value)
        params_boot['A'].min = 0
        params_boot['t2'].min = 0 
        
        # Resample residuals and add to fit
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        y_resampled = y_fit + resampled_residuals
        res = model.fit(y_resampled, params_boot, x=x)
        a_values.append(res.params['A'].value)
        t2_values.append(res.params['t2'].value)        
        
    return np.std(a_values), np.std(t2_values)


def fit_single_residue(x, y, residue_name, initial_A=5, initial_t2=100, n_bootstrap=1000):
    """
    Fit exponential decay to single residue data
    
    Parameters:
    -----------
    x : array
        Time points
    y : array
        Signal intensities
    residue_name : str
        Residue identifier
    initial_A : float
        Initial amplitude estimate
    initial_t2 : float
        Initial time constant estimate
    n_bootstrap : int
        Number of bootstrap iterations
        
    Returns:
    --------
    dict : Fitting results
    """
    print(f"Fitting residue: {residue_name}")
    
    model = Model(binding)
    params = model.make_params(A=initial_A, t2=initial_t2)
    params['A'].min = 0
    params['t2'].min = 0

    result = model.fit(y, params, x=x)

    a = result.params['A'].value
    t2 = result.params['t2'].value

    a_err, t2_err = bootstrap_errors(x, y, model, params, n_bootstrap)
    
    return {
        'residue': residue_name,
        'A': a,
        't2': t2,
        'A_err': a_err,
        't2_err': t2_err,
        'x': x,
        'y': y,
        'result': result
    }


def create_plots(results_list, output_prefix, n_plots_per_figure=20, 
                 experiment_type="T1", time_units="ms", signal_units="Intensity"):
    """
    Create multi-figure PDF plots of fitting results
    
    Parameters:
    -----------
    results_list : list
        List of fitting results dictionaries
    output_prefix : str
        Prefix for output PDF files
    n_plots_per_figure : int
        Number of plots per figure
    experiment_type : str
        Type of experiment (T1, T2, etc.)
    time_units : str
        Units for time axis
    signal_units : str
        Units for signal axis
    """
    n_datasets = len(results_list)
    n_figures = int(np.ceil(n_datasets / n_plots_per_figure))
    
    for fig_idx in range(n_figures):
        fig, axes = plt.subplots(5, 4, figsize=(20, 25), sharex=True)
        axes = axes.flatten()
        start_idx = fig_idx * n_plots_per_figure
        end_idx = min((fig_idx + 1) * n_plots_per_figure, n_datasets)
        
        for idx, result_idx in enumerate(range(start_idx, end_idx)):
            if result_idx >= len(results_list):
                break
                
            result = results_list[result_idx]
            x = result['x']
            y = result['y']
            residue = result['residue']
            a = result['A']
            t2 = result['t2']
            t2_err = result['t2_err']
            
            # Generate fit curve
            x_fit = np.linspace(0, max(x)*1.2, 50)
            y_fit = a * np.exp(-x_fit / t2)
            
            ax = axes[idx]
            ax.plot(x, y, 'ko', lw=2, ms=8, label="Data")
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label="Fit")
            
            # Add fitting results text
            textstr = f"{experiment_type} = {t2:.2f} ± {t2_err:.2f} {time_units}"
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_title(f"Residue: {residue}")
            ax.set_xlabel(f"Time ({time_units})")
            ax.set_ylabel(f"Signal ({signal_units})")
            ax.legend()
            ax.set_xlim(0, max(x)*1.1)
            ax.set_ylim(min(y)*0.9, max(y)*1.1)
        
        # Hide unused subplots
        for idx in range(end_idx - start_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        fig.savefig(f"{output_prefix}_fit_results_fig{fig_idx + 1}.pdf", format='pdf', dpi=300)
        plt.show()
        plt.close(fig)


def save_results(results_list, output_file, experiment_type="T1"):
    """
    Save fitting results to text file
    
    Parameters:
    -----------
    results_list : list
        List of fitting results dictionaries
    output_file : str
        Output filename
    experiment_type : str
        Type of experiment for headers
    """
    with open(output_file, "w") as f:
        f.write(f"Residue\tA\t{experiment_type}\tA_err\t{experiment_type}_err\n")
        for result in results_list:
            f.write(f"{result['residue']}\t{result['A']:.6e}\t{result['t2']:.6e}\t"
                   f"{result['A_err']:.6e}\t{result['t2_err']:.6e}\n")


def main():
    """
    Main function with configurable file handling
    """
    
    # ========================
    # CONFIGURATION SECTION
    # ========================
    
    # Input file configuration
    input_csv_file = "T1_WT_data_600.csv"  # Input CSV file path
    
    # Output file configuration
    output_prefix = "600_WT_T1"  # Prefix for output files
    results_txt_file = "600_WT_T1_fit_results.txt"  # Results text file
    
    # Experiment configuration
    experiment_type = "T1"  # T1, T2, etc. (for labels and headers)
    time_units = "ms"  # Time axis units
    signal_units = "Intensity"  # Signal axis units
    
    # Fitting parameters
    initial_A = 5  # Initial amplitude estimate
    initial_t2 = 100  # Initial time constant estimate
    n_bootstrap = 1000  # Number of bootstrap iterations for error estimation
    
    # Plot configuration
    n_plots_per_figure = 20  # Number of plots per PDF page
    
    # Optional: Time scaling factor (uncomment if needed)
    # time_scaling_factor = 1000  # Multiply time values by this factor
    
    # ========================
    # END CONFIGURATION
    # ========================
    
    # Validate input file exists
    if not os.path.exists(input_csv_file):
        raise FileNotFoundError(f"Input file not found: {input_csv_file}")
    
    print(f"Starting {experiment_type} relaxation fitting analysis...")
    print(f"Input file: {input_csv_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    
    # Load CSV using pandas to handle mixed data types
    print("Loading data...")
    raw_df = pd.read_csv(input_csv_file, header=None)
    
    residue_names = raw_df.iloc[1:, 0].to_numpy()  # Column 1 (residue names), skip header
    x = raw_df.iloc[0, 1:].astype(float).to_numpy()  # First row, columns 2+ (time delays)
    
    # Optional time scaling
    # if 'time_scaling_factor' in locals():
    #     x = x * time_scaling_factor
    
    y_data = raw_df.iloc[1:, 1:].astype(float).to_numpy()  # Values to fit (rows 2+, columns 2+)
    
    print(f"Loaded {len(residue_names)} residues with {len(x)} time points")
    print(f"Time range: {x.min():.3f} to {x.max():.3f} {time_units}")
    
    # Fit all residues
    print("Starting fitting process...")
    results_list = []
    
    for col_idx in range(len(residue_names)):
        y = y_data[col_idx, :]
        residue = residue_names[col_idx]
        
        result = fit_single_residue(x, y, residue, initial_A, initial_t2, n_bootstrap)
        results_list.append(result)
    
    print(f"Completed fitting for {len(results_list)} residues")
    
    # Create plots
    print("Generating plots...")
    create_plots(results_list, output_prefix, n_plots_per_figure, 
                 experiment_type, time_units, signal_units)
    
    # Save results
    print(f"Saving results to {results_txt_file}...")
    save_results(results_list, results_txt_file, experiment_type)
    
    # Summary statistics
    t2_values = [r['t2'] for r in results_list]
    t2_errors = [r['t2_err'] for r in results_list]
    
    print(f"\n{experiment_type} Analysis Summary:")
    print(f"  Number of residues fitted: {len(results_list)}")
    print(f"  {experiment_type} range: {min(t2_values):.2f} to {max(t2_values):.2f} {time_units}")
    print(f"  Mean {experiment_type}: {np.mean(t2_values):.2f} ± {np.std(t2_values):.2f} {time_units}")
    print(f"  Mean fitting error: {np.mean(t2_errors):.2f} {time_units}")
    print(f"  Results saved to: {results_txt_file}")
    print(f"  Plots saved with prefix: {output_prefix}")
    
    print("Analysis completed successfully!")


def run_analysis_with_params(params):
    """
    Run analysis with parameters provided by GUI
    
    Parameters:
    -----------
    params : dict
        Dictionary containing all analysis parameters
        
    Returns:
    --------
    dict : Results summary
    """
    
    # Input file configuration
    input_csv_file = params['input_csv_file']
    
    # Output file configuration
    output_prefix = params['output_prefix']
    results_txt_file = params['results_txt_file']
    
    # Experiment configuration
    experiment_type = params['experiment_type']
    time_units = params.get('time_units', 'ms')
    signal_units = params.get('signal_units', 'Intensity')
    
    # Fitting parameters
    initial_A = params.get('initial_A', 5)
    initial_t2 = params.get('initial_t2', 100)
    n_bootstrap = params.get('n_bootstrap', 1000)
    
    # Plot configuration
    n_plots_per_figure = params.get('n_plots_per_figure', 20)
    
    # Validate input file exists
    if not os.path.exists(input_csv_file):
        raise FileNotFoundError(f"Input file not found: {input_csv_file}")
    
    print(f"Starting {experiment_type} relaxation fitting analysis...")
    print(f"Input file: {input_csv_file}")
    print(f"Output prefix: {output_prefix}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    
    # Load CSV using pandas to handle mixed data types
    print("Loading data...")
    raw_df = pd.read_csv(input_csv_file, header=None)
    
    residue_names = raw_df.iloc[1:, 0].to_numpy()  # Column 1 (residue names), skip header
    x = raw_df.iloc[0, 1:].astype(float).to_numpy()  # First row, columns 2+ (time delays)
    y_data = raw_df.iloc[1:, 1:].astype(float).to_numpy()  # Values to fit (rows 2+, columns 2+)
    
    print(f"Loaded {len(residue_names)} residues with {len(x)} time points")
    print(f"Time range: {x.min():.3f} to {x.max():.3f} {time_units}")
    
    # Fit all residues
    print("Starting fitting process...")
    results_list = []
    
    for col_idx in range(len(residue_names)):
        y = y_data[col_idx, :]
        residue = residue_names[col_idx]
        
        result = fit_single_residue(x, y, residue, initial_A, initial_t2, n_bootstrap)
        results_list.append(result)
    
    print(f"Completed fitting for {len(results_list)} residues")
    
    # Create plots
    print("Generating plots...")
    create_plots(results_list, output_prefix, n_plots_per_figure, 
                 experiment_type, time_units, signal_units)
    
    # Save results
    print(f"Saving results to {results_txt_file}...")
    save_results(results_list, results_txt_file, experiment_type)
    
    # Summary statistics
    t2_values = [r['t2'] for r in results_list]
    t2_errors = [r['t2_err'] for r in results_list]
    
    print(f"\n{experiment_type} Analysis Summary:")
    print(f"  Number of residues fitted: {len(results_list)}")
    print(f"  {experiment_type} range: {min(t2_values):.2f} to {max(t2_values):.2f} {time_units}")
    print(f"  Mean {experiment_type}: {np.mean(t2_values):.2f} ± {np.std(t2_values):.2f} {time_units}")
    print(f"  Mean fitting error: {np.mean(t2_errors):.2f} {time_units}")
    print(f"  Results saved to: {results_txt_file}")
    print(f"  Plots saved with prefix: {output_prefix}")
    
    print("Analysis completed successfully!")
    
    # Return results summary
    return {
        'n_fitted': len(results_list),
        'results_file': results_txt_file,
        'plots_prefix': output_prefix,
        't2_range': (min(t2_values), max(t2_values)),
        'mean_t2': np.mean(t2_values),
        'std_t2': np.std(t2_values),
        'mean_error': np.mean(t2_errors)
    }


if __name__ == "__main__":
    main()