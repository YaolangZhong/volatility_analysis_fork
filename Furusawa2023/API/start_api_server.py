#!/usr/bin/env python
"""
API Server Startup Script
=========================

This script starts the API server for economic model data generation.
It handles server configuration and provides helpful status information.
"""

import os
import sys
import logging
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['fastapi', 'uvicorn', 'pydantic', 'pandas', 'numpy', 'openpyxl']
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Please install them with: pip install " + " ".join(missing))
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_model_data():
    """Check if model data files exist."""
    data_file = Path("../data.npz")
    
    if not data_file.exists():
        print(f"❌ Model data file not found: {data_file}")
        print("Please ensure the data file exists in the correct location.")
        return False
    
    print(f"✅ Model data file found: {data_file}")
    return True

def start_server(host="0.0.0.0", port=8000, workers=1, reload=False):
    """Start the FastAPI server."""
    print(f"🚀 Starting API server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Workers: {workers}")
    print(f"   Reload: {reload}")
    print(f"   URL: http://localhost:{port}")
    print("-" * 50)
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api_server:app",
            f"--host={host}",
            f"--port={port}",
            f"--workers={workers}"
        ]
        
        if reload:
            cmd.append("--reload")
        
        # Run the server
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function to start the API server with checks."""
    print("Economic Model API Server Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model data
    if not check_model_data():
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Start the Economic Model API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Start the server
    success = start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 