#!/usr/bin/env python3
"""
Minimal Baseline Model Solver
=============================

Minimal script to solve baseline model and save to single pickle file.
Just click "run" to solve data.npz and save baseline_model.pkl
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models import ModelParams, Model
from solvers import ModelSolver


def solve_and_save_baseline(data_path: str = "data.npz", output_path: str = "baseline_model.pkl") -> str:
    """
    Solve baseline model and save complete model to single pickle file.
    
    Parameters
    ----------
    data_path : str, optional
        Path to .npz data file (default: "data.npz")
    output_path : str, optional
        Output path for model file (default: "baseline_model.pkl")
        
    Returns
    -------
    str
        Path to saved model file
    """
    # Load parameters (includes validation)
    params = ModelParams.load_from_npz(data_path)
    
    # Create model (auto-initializes baseline shocks and empty solution)
    model = Model(params)
    
    # Solve model with default configuration (shows solver progress)
    solver = ModelSolver(model)
    solver.solve()
    
    # Save complete model to single file
    model.save_to_pickle(output_path)
    
    return output_path


if __name__ == "__main__":
    # Default behavior: solve data.npz and save baseline_model.pkl
    data_path = "data.npz"
    output_path = "baseline_model.pkl"
    
    # Override with command line arguments if provided
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    
    try:
        solve_and_save_baseline(data_path, output_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 