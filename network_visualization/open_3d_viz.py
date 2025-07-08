#!/usr/bin/env python3
"""
Launch the 3D network visualization in the default browser.
"""

import webbrowser
import os
from pathlib import Path

def open_3d_network_visualization():
    """Open the 3D network visualization in the default browser."""
    html_file = Path("simple_3d_viz.html")
    
    if html_file.exists():
        # Get absolute path
        abs_path = html_file.absolute()
        file_url = f"file://{abs_path}"
        
        print("🌐 Opening 3D Trade Network Visualization...")
        print(f"URL: {file_url}")
        
        # Open in default browser
        webbrowser.open(file_url)
        
        print("\n🎮 3D Navigation Controls:")
        print("🖱️  Mouse Controls:")
        print("   • Drag to orbit around the network")
        print("   • Scroll to zoom in/out")
        print("   • Click nodes for detailed information")
        
        print("\n🎛️  Interface Features:")
        print("   • Node Size represents aggregate expenditure")
        print("   • Colors indicate continent groupings")
        print("   • Interactive statistics panel")
        print("   • Smooth 3D navigation")
        
        print("\n🎨 Visual Elements:")
        print("   • 37 country nodes with real expenditure data")
        print("   • Color-coded by continent")
        print("   • Hover effects and tooltips")
        print("   • Real-time statistics display")
        
    else:
        print("❌ Error: simple_3d_viz.html not found!")
        print("Make sure the file exists in the current directory.")

def compare_visualizations():
    """Show comparison between 2D and 3D visualizations."""
    html_2d = Path("network_trade_2d.html")
    html_3d = Path("network_trade_3d.html")
    
    print("\n📊 Visualization Comparison:")
    print("="*50)
    
    if html_2d.exists():
        print("✅ 2D Version Available (network_trade_2d.html)")
        print("   • Traditional flat network view")
        print("   • Multiple layout algorithms")
        print("   • SVG export capability")
    else:
        print("❌ 2D Version Not Found")
    
    if html_3d.exists():
        print("✅ 3D Version Available (network_trade_3d.html)")
        print("   • Immersive 3D exploration")
        print("   • Advanced camera controls")
        print("   • Real-time 3D physics")
    else:
        print("❌ 3D Version Not Found")
    
    print("\n💡 Recommendation:")
    print("Use 3D version for:")
    print("   • Exploratory data analysis")
    print("   • Presentations and demos")
    print("   • Understanding spatial relationships")
    
    print("Use 2D version for:")
    print("   • Quick analysis")
    print("   • Printing/export needs")
    print("   • Lower performance devices")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_visualizations()
    else:
        open_3d_network_visualization() 