"""
API Server for Economic Model Data Generation
=============================================

This server handles all model solving and data generation.
It exposes REST API endpoints for:
1. Solving benchmark models
2. Solving counterfactual models  
3. Downloading results as Excel files
4. Getting model metadata

The server maintains the same logic as model_pipeline.py but exposes it via HTTP.

=== VARIABLE DOWNLOAD INTEGRATION WITH SOLUTION FRAMEWORK ===

The download endpoints extract economic variables directly from your solution framework:

1. SOLUTION EXTRACTION:
   - Gets ModelSol objects from solved models (your solution framework output)
   - Accesses individual variables (w_hat, c_hat, sector_links, etc.) as NumPy arrays
   - Uses ModelParams for country/sector names for proper labeling

2. VARIABLE FORMATTING:
   - 1D Variables (Country-level): w_hat, real_w_hat, D_prime → Countries as rows
   - 2D Variables (Country-Sector): c_hat, Pf_hat, X_prime → Countries×Sectors matrix
   - 3D Variables (Trade flows): pif_hat, sector_links → Flattened to 2D for Excel
   - 4D Variables (sector_links): → Available as separate endpoint

3. DOWNLOAD FORMATS:
   - Single variable Excel: One variable per file with economic labels
   - All variables Excel: Multi-sheet workbook with all 1D/2D variables
   - Economic interpretation: Country names, sector names as row/column labels

This allows users to get Excel files directly from your economic model solutions
with proper economic variable names and geographic/sectoral labels.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import io
import tempfile
import os
import json
from datetime import datetime
import logging

# Import our existing model components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# model_pipeline is now in the same directory
from model_pipeline import (
    get_model_pipeline,
    solve_benchmark_cached,
    solve_counterfactual_cached
)
from models import ModelSol, ModelParams
from visualization import ModelVisualizationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Economic Model API", version="1.0.0")

# Pydantic models for API requests/responses
class CounterfactualRequest(BaseModel):
    importers: List[str]
    exporters: List[str]
    sectors: List[str]
    tariff_data: dict  # Fixed: Changed from tariff_rate: float to tariff_data: dict

class ModelMetadata(BaseModel):
    countries: List[str]
    sectors: List[str]
    N: int
    S: int

class SolutionSummary(BaseModel):
    scenario_key: str
    solved_at: str
    request_params: Dict[str, Any]
    variables_available: List[str]

# Global cache for storing solved models (in production, use Redis or similar)
_solved_models_cache = {}

@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Economic Model API is running", "timestamp": datetime.now().isoformat()}

@app.get("/metadata", response_model=ModelMetadata)
def get_model_metadata():
    """Get basic model metadata (countries, sectors, dimensions)."""
    try:
        _, params = solve_benchmark_cached()
        
        return ModelMetadata(
            countries=list(params.country_list),
            sectors=list(params.sector_list),
            N=params.N,
            S=params.S
        )
    except Exception as e:
        logger.error(f"Error getting model metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model metadata: {str(e)}")

@app.get("/benchmark/solve")
def solve_benchmark():
    """Solve the benchmark model and return summary."""
    try:
        sol, params = solve_benchmark_cached()
        
        # Store in cache with a special key for benchmark
        cache_key = "benchmark"
        _solved_models_cache[cache_key] = {
            "solution": sol,
            "params": params,
            "solved_at": datetime.now().isoformat(),
            "type": "benchmark"
        }
        
        return {
            "scenario_key": cache_key,
            "status": "solved",
            "solved_at": _solved_models_cache[cache_key]["solved_at"],
            "message": "Benchmark model solved successfully"
        }
    except Exception as e:
        logger.error(f"Error solving benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to solve benchmark model: {str(e)}")

@app.post("/counterfactual/solve")
def solve_counterfactual(request: CounterfactualRequest):
    """Solve a counterfactual model and return summary."""
    try:
        pipeline = get_model_pipeline()
        
        # Solve the counterfactual
        scenario_key = pipeline.solve_counterfactual(
            request.importers, 
            request.exporters, 
            request.sectors, 
            request.tariff_data
        )
        
        # Get the results
        sol, params = pipeline.get_counterfactual_results(scenario_key)
        
        # Store in cache
        _solved_models_cache[scenario_key] = {
            "solution": sol,
            "params": params,
            "solved_at": datetime.now().isoformat(),
            "type": "counterfactual",
            "request_params": request.dict()
        }
        
        return {
            "scenario_key": scenario_key,
            "status": "solved", 
            "solved_at": _solved_models_cache[scenario_key]["solved_at"],
            "message": f"Counterfactual model solved successfully"
        }
        
    except Exception as e:
        logger.error(f"Error solving counterfactual: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to solve counterfactual model: {str(e)}")

@app.get("/models", response_model=List[SolutionSummary])
def list_solved_models():
    """List all solved models in cache."""
    summaries = []
    for key, data in _solved_models_cache.items():
        summary = SolutionSummary(
            scenario_key=key,
            solved_at=data["solved_at"],
            request_params=data.get("request_params", {}),
            variables_available=list(data["solution"].__dict__.keys())
        )
        summaries.append(summary)
    return summaries

@app.get("/models/{scenario_key}/variables")
def get_model_variables(scenario_key: str):
    """Get all variable names and shapes for a solved model."""
    if scenario_key not in _solved_models_cache:
        raise HTTPException(status_code=404, detail=f"Model {scenario_key} not found")
    
    sol = _solved_models_cache[scenario_key]["solution"]
    params = _solved_models_cache[scenario_key]["params"]
    
    variables = {}
    
    # Add solution variables
    for attr_name in dir(sol):
        if not attr_name.startswith('_') and hasattr(sol, attr_name):
            attr_value = getattr(sol, attr_name)
            if isinstance(attr_value, np.ndarray):
                variables[attr_name] = {
                    "shape": attr_value.shape,
                    "dtype": str(attr_value.dtype),
                    "component": "solution"
                }
    
    return {
        "scenario_key": scenario_key,
        "variables": variables,
        "countries": list(params.country_list),
        "sectors": list(params.sector_list)
    }

@app.get("/models/{scenario_key}/variable/{variable_name}")
def get_variable_data(scenario_key: str, variable_name: str):
    """Get data for a specific variable from a solved model."""
    if scenario_key not in _solved_models_cache:
        raise HTTPException(status_code=404, detail=f"Model {scenario_key} not found")
    
    sol = _solved_models_cache[scenario_key]["solution"]
    params = _solved_models_cache[scenario_key]["params"]
    
    if not hasattr(sol, variable_name):
        raise HTTPException(status_code=404, detail=f"Variable {variable_name} not found")
    
    data = getattr(sol, variable_name)
    
    if not isinstance(data, np.ndarray):
        raise HTTPException(status_code=400, detail=f"Variable {variable_name} is not an array")
    
    return {
        "scenario_key": scenario_key,
        "variable_name": variable_name,
        "shape": data.shape,
        "data": data.tolist(),
        "countries": list(params.country_list),
        "sectors": list(params.sector_list)
    }

@app.get("/models/{scenario_key}/download/excel/{variable_name}")
def download_variable_excel(scenario_key: str, variable_name: str):
    """Download a specific variable as an Excel file."""
    if scenario_key not in _solved_models_cache:
        raise HTTPException(status_code=404, detail=f"Model {scenario_key} not found")
    
    # === EXTRACT SOLUTION VARIABLES FROM YOUR SOLUTION FRAMEWORK ===
    sol = _solved_models_cache[scenario_key]["solution"]      # Your ModelSol object from solution framework
    params = _solved_models_cache[scenario_key]["params"]     # Your ModelParams object with country/sector names
    
    if not hasattr(sol, variable_name):
        raise HTTPException(status_code=404, detail=f"Variable {variable_name} not found")
    
    # === GET SPECIFIC VARIABLE DATA FROM YOUR SOLUTION ===
    data = getattr(sol, variable_name)  # Extract NumPy array from your ModelSol object (e.g., w_hat, sector_links, etc.)
    
    if not isinstance(data, np.ndarray):
        raise HTTPException(status_code=400, detail=f"Variable {variable_name} is not an array")
    
    try:
        # === CREATE EXCEL FILE WITH PROPER ECONOMIC VARIABLE FORMATTING ===
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            
            # === 1D VARIABLES: Country-level data (e.g., w_hat, real_w_hat, D_prime) ===
            if data.ndim == 1:
                # Format: Countries as rows, single column for the variable
                df = pd.DataFrame({
                    variable_name: data
                }, index=params.country_list)  # Use your country names as row labels
                
            # === 2D VARIABLES: Country-Sector data (e.g., c_hat, Pf_hat, X_prime) ===
            elif data.ndim == 2:
                # Format: Countries as rows, sectors as columns
                df = pd.DataFrame(
                    data, 
                    index=list(params.country_list),    # Countries from your solution framework
                    columns=list(params.sector_list)    # Sectors from your solution framework
                )
                
            # === 3D VARIABLES: Trade flows (e.g., pif_hat, pim_hat) ===
            elif data.ndim == 3:
                # Flatten 3D trade data (N,N,S) to 2D for Excel compatibility
                # Reshape from (countries, countries, sectors) to (countries, countries*sectors)
                reshaped_data = data.reshape(data.shape[0], -1)
                
                # Create meaningful column names for flattened trade data
                if data.shape[1] == len(params.country_list):  # Trade flows: (importer, exporter, sector)
                    col_names = [f"{params.country_list[i]}_{params.sector_list[j]}" 
                               for i in range(data.shape[1]) for j in range(data.shape[2])]
                else:
                    col_names = [f"dim_{i}_{j}" for i in range(data.shape[1]) for j in range(data.shape[2])]
                
                df = pd.DataFrame(
                    reshaped_data,
                    index=params.country_list,           # Importing countries as rows
                    columns=col_names                    # Exporter_Sector combinations as columns
                )
            else:
                raise HTTPException(status_code=400, detail=f"Cannot export {data.ndim}D array")
            
            # === SAVE EXCEL FILE WITH ECONOMIC VARIABLE DATA ===
            df.to_excel(tmp_file.name, sheet_name=variable_name)
            
            # === RETURN DOWNLOADABLE FILE ===
            filename = f"{scenario_key}_{variable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            return FileResponse(
                tmp_file.name,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                filename=filename
            )
            
    except Exception as e:
        logger.error(f"Error creating Excel file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create Excel file: {str(e)}")

@app.get("/models/{scenario_key}/download/all")
def download_all_variables_excel(scenario_key: str):
    """Download all variables as a multi-sheet Excel file."""
    if scenario_key not in _solved_models_cache:
        raise HTTPException(status_code=404, detail=f"Model {scenario_key} not found")
    
    # === EXTRACT ALL SOLUTION VARIABLES FROM YOUR SOLUTION FRAMEWORK ===
    sol = _solved_models_cache[scenario_key]["solution"]      # Your complete ModelSol object
    params = _solved_models_cache[scenario_key]["params"]     # Your ModelParams with country/sector names
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            with pd.ExcelWriter(tmp_file.name, engine='openpyxl') as writer:
                
                # === ITERATE THROUGH ALL VARIABLES IN YOUR SOLUTION FRAMEWORK ===
                # This loops through all attributes of your ModelSol object (w_hat, c_hat, sector_links, etc.)
                for attr_name in dir(sol):
                    if not attr_name.startswith('_') and hasattr(sol, attr_name):
                        attr_value = getattr(sol, attr_name)  # Get each variable from your solution
                        
                        # === PROCESS ONLY NUMPY ARRAYS (YOUR ECONOMIC VARIABLES) ===
                        if isinstance(attr_value, np.ndarray):
                            try:
                                # === 1D VARIABLES: Country-level data ===
                                if attr_value.ndim == 1:
                                    # Variables like w_hat, real_w_hat, D_prime (shape: N)
                                    df = pd.DataFrame({
                                        attr_name: attr_value
                                    }, index=params.country_list)  # Countries as row labels
                                    
                                # === 2D VARIABLES: Country-Sector data ===
                                elif attr_value.ndim == 2:
                                    # Variables like c_hat, Pf_hat, X_prime (shape: N,S)
                                    df = pd.DataFrame(
                                        attr_value,
                                        index=params.country_list,    # Countries as rows
                                        columns=params.sector_list    # Sectors as columns
                                    )
                                else:
                                    # === SKIP 3D+ ARRAYS (e.g., sector_links) ===
                                    # 3D+ arrays like sector_links, pif_hat are too complex for simple Excel sheets
                                    continue
                                
                                # === CREATE EXCEL SHEET FOR EACH VARIABLE ===
                                # Each economic variable gets its own sheet in the Excel workbook
                                sheet_name = attr_name[:31] if len(attr_name) > 31 else attr_name  # Excel sheet name limit
                                df.to_excel(writer, sheet_name=sheet_name)
                                
                            except Exception as e:
                                # === SKIP VARIABLES THAT CAN'T BE EXPORTED ===
                                logger.warning(f"Could not export variable {attr_name}: {e}")
                                continue
            
            # === RETURN COMPLETE EXCEL WORKBOOK WITH ALL VARIABLES ===
            filename = f"{scenario_key}_all_variables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            return FileResponse(
                tmp_file.name,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                filename=filename
            )
            
    except Exception as e:
        logger.error(f"Error creating Excel file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create Excel file: {str(e)}")

@app.delete("/models/{scenario_key}")
def delete_model(scenario_key: str):
    """Delete a solved model from cache."""
    if scenario_key not in _solved_models_cache:
        raise HTTPException(status_code=404, detail=f"Model {scenario_key} not found")
    
    if scenario_key == "benchmark":
        raise HTTPException(status_code=400, detail="Cannot delete benchmark model")
    
    del _solved_models_cache[scenario_key]
    return {"message": f"Model {scenario_key} deleted successfully"}

@app.post("/models/clear")
def clear_all_models():
    """Clear all solved models from cache (except benchmark)."""
    keys_to_delete = [k for k in _solved_models_cache.keys() if k != "benchmark"]
    
    for key in keys_to_delete:
        del _solved_models_cache[key]
    
    return {"message": f"Cleared {len(keys_to_delete)} models from cache"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 