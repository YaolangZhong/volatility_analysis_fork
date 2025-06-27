#!/usr/bin/env python3
"""
Simple script to open the network visualization in the default browser.
"""

import webbrowser
import os
from pathlib import Path

def open_network_visualization():
    """Open the network visualization in the default browser."""
    html_file = Path("network_trade_visualization.html")
    
    if html_file.exists():
        # Get absolute path
        abs_path = html_file.absolute()
        file_url = f"file://{abs_path}"
        
        print(f"Opening network visualization in browser...")
        print(f"URL: {file_url}")
        
        # Open in default browser
        webbrowser.open(file_url)
        
        print("\nNetwork Visualization Features:")
        print("✓ Smooth zoom and pan (mouse wheel + drag)")
        print("✓ Country nodes sized by aggregate expenditure") 
        print("✓ Color-coded by continent")
        print("✓ Multiple layout algorithms:")
        print("  - Force-directed (default)")
        print("  - Circle arrangement")
        print("  - Grid layout")
        print("✓ Adjustable node sizes with slider")
        print("✓ Rich tooltips on hover")
        print("✓ Export to SVG functionality")
        print("✓ Scenario switching (baseline/counterfactual/comparison)")
        
    else:
        print("Error: network_trade_visualization.html not found!")
        print("Run 'python network_data_generator.py' first to generate the visualization.")

if __name__ == "__main__":
    open_network_visualization() 