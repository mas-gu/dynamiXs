#!/usr/bin/env python3
"""
Simple test script to verify GUI functionality

This script checks if all required modules can be imported
and the GUI can be initialized without errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing module imports...")
    
    try:
        import tkinter as tk
        print("✓ tkinter imported successfully")
    except ImportError as e:
        print(f"✗ tkinter import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ matplotlib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ pandas import failed: {e}")
        return False
    
    try:
        import lmfit
        print("✓ lmfit imported successfully")
    except ImportError as e:
        print(f"✗ lmfit import failed: {e}")
        return False
    
    try:
        import scipy
        print("✓ scipy imported successfully")
    except ImportError as e:
        print(f"✗ scipy import failed: {e}")
        return False
    
    return True

def test_gui_init():
    """Test if GUI can be initialized"""
    print("\nTesting GUI initialization...")
    
    try:
        # Add current directory to path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from dynamiXs_gui import DynamiXsGUI
        import tkinter as tk
        
        # Create a test root window
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Initialize the GUI
        app = DynamiXsGUI(root)
        print("✓ GUI initialized successfully")
        
        # Clean up
        root.destroy()
        return True
        
    except Exception as e:
        print(f"✗ GUI initialization failed: {e}")
        return False

def test_script_access():
    """Test if analysis scripts can be accessed"""
    print("\nTesting analysis script access...")
    
    current_dir = Path(__file__).parent
    
    # Check T1/T2 scripts
    t1_t2_path = current_dir / "dynamiXs_T1_T2"
    if t1_t2_path.exists():
        fit_script = t1_t2_path / "fit_Tx_NMRRE.py"
        multi_script = t1_t2_path / "fitmulti__Tx_NMRRE.py"
        
        if fit_script.exists():
            print("✓ T1/T2 single-core script found")
        else:
            print("✗ T1/T2 single-core script not found")
            
        if multi_script.exists():
            print("✓ T1/T2 multi-core script found")
        else:
            print("✗ T1/T2 multi-core script not found")
    else:
        print("✗ dynamiXs_T1_T2 directory not found")
    
    # Check spectral density scripts
    density_path = current_dir / "dynamiXs_density_functions"
    if density_path.exists():
        density_script = density_path / "ZZ_2fields_density_claude_rex_mcmc_error.py"
        
        if density_script.exists():
            print("✓ Spectral density analysis script found")
        else:
            print("✗ Spectral density analysis script not found")
    else:
        print("✗ dynamiXs_density_functions directory not found")
    
    # Check plotting scripts
    plot_path = current_dir / "dynamiXs_plot"
    if plot_path.exists():
        compare_script = plot_path / "compare_datasets.py"
        plot_script = plot_path / "plot_dual_field_datasets.py"
        
        if compare_script.exists():
            print("✓ Dataset comparison script found")
        else:
            print("✗ Dataset comparison script not found")
            
        if plot_script.exists():
            print("✓ Dual field plotting script found")
        else:
            print("✗ Dual field plotting script not found")
    else:
        print("✗ dynamiXs_plot directory not found")
    
    # Check CPMG scripts
    cpmg_path = current_dir / "dynamiXs_cpmg"
    if cpmg_path.exists():
        cpmg_script = cpmg_path / "cpmg_RD.py"
        
        if cpmg_script.exists():
            print("✓ CPMG analysis script found")
        else:
            print("✗ CPMG analysis script not found")
    else:
        print("✗ dynamiXs_cpmg directory not found")

def main():
    """Run all tests"""
    print("DynamiXs GUI Test Suite")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test GUI initialization (only if imports work)
    if imports_ok:
        gui_ok = test_gui_init()
    else:
        print("\nSkipping GUI test due to import failures")
        gui_ok = False
    
    # Test script access
    test_script_access()
    
    print("\n" + "=" * 40)
    if imports_ok and gui_ok:
        print("✓ All tests passed! GUI should work correctly.")
        print("Run 'python run_dynamixs_gui.py' to start the interface.")
    else:
        print("✗ Some tests failed. Check the errors above.")
        if not imports_ok:
            print("Install missing packages with: pip install numpy matplotlib pandas lmfit scipy")

if __name__ == "__main__":
    main()