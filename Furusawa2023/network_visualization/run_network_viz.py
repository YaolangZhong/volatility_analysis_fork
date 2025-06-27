#!/usr/bin/env python3
"""
Simple launcher for the 3D Network Visualization
"""

import subprocess
import sys
import os
from pathlib import Path

def run_network_visualization():
    """Run the 3D network visualization."""
    
    # Change to network_visualization directory
    viz_dir = Path("network_visualization")
    
    if not viz_dir.exists():
        print("❌ Error: network_visualization directory not found!")
        return
    
    # Change to the visualization directory
    os.chdir(viz_dir)
    
    print("🚀 Launching 3D Network Visualization...")
    print("📁 Working directory:", os.getcwd())
    
    try:
        # Run the open_3d_viz.py script
        subprocess.run([sys.executable, "open_3d_viz.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running visualization: {e}")
    except FileNotFoundError:
        print("❌ Error: open_3d_viz.py not found in network_visualization directory")

if __name__ == "__main__":
    run_network_visualization() 