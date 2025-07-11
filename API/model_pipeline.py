"""
Simplified Model Pipeline: Direct counterfactual solving with hash-based caching
===============================================================================

Core functions:
1. solve_counterfactual() - directly solve and cache  
2. get_counterfactual_results() - directly retrieve from cache
3. Hash-based caching for performance

No unnecessary wrapper classes or complex abstractions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import hashlib
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import ModelParams, Model, ModelSol
from solvers import ModelSolver


# Global cache for counterfactual solutions
_solution_cache: Dict[str, ModelSol] = {}
_params_cache: Dict[str, ModelParams] = {}


def _hash_tariff_structure(tariff_data: dict) -> str:
    """Generate hash from tariff structure."""
    tariff_str = str(sorted(tariff_data.items()))
    return hashlib.md5(tariff_str.encode()).hexdigest()[:16]


def _solve_tariff_scenario(baseline_params: ModelParams,
                          importers: List[str], 
                          exporters: List[str], 
                          sectors: List[str], 
                          tariff_data: dict) -> ModelSol:
    """Core solving logic."""
    country_names = list(baseline_params.country_list)
    sector_names = list(baseline_params.sector_list)
    
    # Handle empty tariff case
    if not tariff_data:
        cf_model = Model(baseline_params)
        cf_solver = ModelSolver(cf_model)
        cf_solver.solve()
        return cf_model.sol
    
    # Modify tariff matrix
    tilde_tau_1 = baseline_params.tilde_tau.copy()
    
    for importer in importers:
        for exporter in exporters:
            for sector in sectors:
                i = country_names.index(importer)
                j = country_names.index(exporter)
                s = sector_names.index(sector)
                
                if i != j:  # Skip domestic trade
                    tariff_rate = tariff_data.get((importer, exporter, sector), 0.0)
                    tilde_tau_1[i, j, s] = (
                        baseline_params.tilde_tau[i, j, s] + (tariff_rate / 100.0)
                    )
    
    # Solve counterfactual
    cf_params = deepcopy(baseline_params)
    cf_params.tilde_tau = tilde_tau_1
    cf_model = Model(cf_params)
    cf_solver = ModelSolver(cf_model)
    cf_solver.solve()
    
    return cf_model.sol


def solve_counterfactual(baseline_params: ModelParams,
                        importers: List[str], 
                        exporters: List[str], 
                        sectors: List[str], 
                        tariff_data: dict) -> str:
    """
    Solve counterfactual model and cache result.
    
    Args:
        baseline_params: Baseline model parameters
        importers: List of importing country names
        exporters: List of exporting country names  
        sectors: List of sector names affected
        tariff_data: Dict mapping (importer, exporter, sector) to tariff rates
    
    Returns:
        str: Cache key for retrieving the solution later
    """
    # Generate cache key from tariff structure
    cache_key = _hash_tariff_structure(tariff_data)
    
    # Check cache first
    if cache_key in _solution_cache:
        return cache_key
    
    # Solve and cache
    solution = _solve_tariff_scenario(baseline_params, importers, exporters, sectors, tariff_data)
    _solution_cache[cache_key] = solution
    _params_cache[cache_key] = deepcopy(baseline_params)
    
    return cache_key


def get_tariff_cache_key(tariff_data: dict) -> str:
    """Get cache key for tariff configuration without solving."""
    return _hash_tariff_structure(tariff_data)


def get_counterfactual_results(cache_key: str) -> Tuple[Optional[ModelSol], Optional[ModelParams], str]:
    """
    Retrieve counterfactual results by cache key.
    
    Args:
        cache_key: Hash key returned by solve_counterfactual()
        
    Returns:
        Tuple[ModelSol, ModelParams, str]: Solution, parameters, and cache_key, or (None, None, cache_key) if not found
    """
    solution = _solution_cache.get(cache_key)
    params = _params_cache.get(cache_key)
    return solution, params, cache_key


def clear_counterfactual_cache() -> None:
    """Clear all cached counterfactual solutions."""
    global _solution_cache, _params_cache
    _solution_cache.clear()
    _params_cache.clear()


def list_cached_scenarios() -> List[str]:
    """List all cached scenario keys."""
    return list(_solution_cache.keys())


def get_metadata_cached() -> Tuple[List[str], List[str], int, int]:
    """Get model metadata from baseline model pickle."""
    try:
        from models import Model
        model = Model.load_from_pickle("baseline_model.pkl")
        country_names = list(model.params.country_list)
        sector_names = list(model.params.sector_list)
        return country_names, sector_names, len(country_names), len(sector_names)
    except Exception:
        # Fallback to direct data loading
        from models import ModelParams
        from pathlib import Path
        
        data_paths = ["data.npz", "../data.npz", "Furusawa2023/data.npz"]
        for path in data_paths:
            if Path(path).exists():
                params = ModelParams.load_from_npz(path)
                return list(params.country_list), list(params.sector_list), params.N, params.S
        
        raise FileNotFoundError("Could not find data.npz or baseline_model.pkl") 