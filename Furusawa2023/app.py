#!/usr/bin/env python
"""
Main Application Launcher
=========================

This script launches the enhanced Streamlit app with API support.
It provides a simple entry point while the actual implementation 
is organized in the API directory.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add API directory to path for imports
api_dir = Path(__file__).parent / "API"
sys.path.insert(0, str(api_dir))

# Execute the main app
exec(open(api_dir / "app_with_api.py").read()) 