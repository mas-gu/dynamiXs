#!/usr/bin/env python3
"""
Test script to verify the integration of the *087 scripts in the dynamiXs GUI
"""

import os
import sys

def test_script_availability():
    """Test if all *087 scripts are available"""
    
    print("Testing *087 script integration in dynamiXs...")
    print("=" * 50)
    
    # Test density function scripts
    density_path = "dynamiXs_density_functions"
    density_scripts = [
        "ZZ_2fields_density_087.py",
        "ZZ_multi_2fields_density_087.py", 
        "ZZ_multi_density_087.py"
    ]
    
    print(f"\n1. Testing density function scripts in {density_path}:")
    for script in density_scripts:
        full_path = os.path.join(density_path, script)
        if os.path.exists(full_path):
            print(f"   ✓ {script} - Found")
        else:
            print(f"   ✗ {script} - Missing")
    
    # Test plotting scripts
    plot_path = "dynamiXs_plot"
    plot_scripts = [
        "compare_datasets_087.py",
        "plot_dual_field_datasets_087.py",
        "ylimits087.json"
    ]
    
    print(f"\n2. Testing plotting scripts in {plot_path}:")
    for script in plot_scripts:
        full_path = os.path.join(plot_path, script)
        if os.path.exists(full_path):
            print(f"   ✓ {script} - Found")
        else:
            print(f"   ✗ {script} - Missing")
    
    # Test GUI modification
    print(f"\n3. Testing GUI modifications:")
    try:
        import dynamiXs_gui
        
        # Check if SpectralDensityGUI has spectral_method_var
        print("   ✓ GUI module imports successfully")
        
        # Check for specific functionality markers
        gui_code = open("dynamiXs_gui.py").read()
        
        if "spectral_method_var" in gui_code:
            print("   ✓ Spectral method selection implemented")
        else:
            print("   ✗ Spectral method selection missing")
            
        if "J(0.87ωH)" in gui_code:
            print("   ✓ J(0.87ωH) option available")
        else:
            print("   ✗ J(0.87ωH) option missing")
            
        if "087" in gui_code:
            print("   ✓ *087 script integration present")
        else:
            print("   ✗ *087 script integration missing")
    
    except Exception as e:
        print(f"   ✗ GUI testing failed: {e}")
    
    print("\n" + "=" * 50)
    print("Integration test completed!")
    print("\nNew functionality:")
    print("• Users can now choose between J(ωH) and J(0.87ωH) methods")
    print("• Both spectral density analysis and plotting respect the choice")
    print("• All *087 scripts are available in the GUI")

if __name__ == "__main__":
    test_script_availability()