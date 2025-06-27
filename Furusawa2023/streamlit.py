#!/usr/bin/env python3
"""
Streamlit App Launcher
=====================

Simple launcher for the enhanced economic model Streamlit app.

Usage:
    python streamlit
    # or make it executable: chmod +x streamlit && ./streamlit
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the streamlit app in API directory
    app_path = script_dir / "API" / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    # Change to the API directory for proper imports
    os.chdir(script_dir / "API")
    
    # Launch streamlit
    try:
        print("🚀 Starting Enhanced Economic Model Streamlit App...")
        print(f"📁 Working directory: {os.getcwd()}")
        print(f"🎯 Running: streamlit run app.py")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start app: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 