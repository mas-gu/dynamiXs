#!/usr/bin/env python3
"""
Launch script for DynamiXs GUI

This script launches the DynamiXs NMR relaxation analysis GUI.
Simply run this script to start the graphical interface.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from dynamiXs_gui import main
    
    if __name__ == "__main__":
        print("Starting DynamiXs GUI...")
        print("GUI should open in a new window.")
        print("If the window doesn't appear, check your Python tkinter installation.")
        main()
        
except ImportError as e:
    print(f"Error importing GUI modules: {e}")
    print("Please ensure all required packages are installed:")
    print("- tkinter (usually included with Python)")
    print("- customtkinter (pip install customtkinter)")
    print("- numpy")
    print("- matplotlib")
    print("- pandas")
    print("- lmfit")
    print("- scipy")
    sys.exit(1)
    
except Exception as e:
    print(f"Error starting GUI: {e}")
    sys.exit(1)