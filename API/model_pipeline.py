"""
Model Pipeline: Separated model solving and result storage
=========================================================

This module provides a clear separation between:
1. Model solving (benchmark and counterfactual scenarios)
2. Result storage and retrieval
3. Visualization data preparation

This allows for easy reuse of solving logic for different visualization approaches
(traditional plotting vs network graphs).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, cast
from copy import deepcopy
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import ModelParams, Model, ModelSol
from solvers import ModelSolver


class ModelResultStorage:
    """Manages storage and retrieval of model solutions."""
    
    def __init__(self):
        self._benchmark_solution: Optional[ModelSol] = None
        self._benchmark_params: Optional[ModelParams] = None
        self._counterfactual_solutions: Dict[str, ModelSol] = {}
        self._counterfactual_params: Dict[str, ModelParams] = {}
    
    def store_benchmark(self, solution: ModelSol, params: ModelParams) -> None:
        """Store benchmark model solution and parameters."""
        self._benchmark_solution = solution
        self._benchmark_params = params
    
    def store_counterfactual(self, key: str, solution: ModelSol, params: ModelParams) -> None:
        """Store counterfactual model solution with a unique key."""
        self._counterfactual_solutions[key] = solution
        self._counterfactual_params[key] = params
    
    def get_benchmark(self) -> Tuple[Optional[ModelSol], Optional[ModelParams]]:
        """Retrieve benchmark solution and parameters."""
        return self._benchmark_solution, self._benchmark_params
    
    def get_counterfactual(self, key: str) -> Tuple[Optional[ModelSol], Optional[ModelParams]]:
        """Retrieve counterfactual solution and parameters by key."""
        return (self._counterfactual_solutions.get(key), 
                self._counterfactual_params.get(key))
    
    def list_counterfactuals(self) -> List[str]:
        """List all stored counterfactual keys."""
        return list(self._counterfactual_solutions.keys())
    
    def clear_counterfactuals(self) -> None:
        """Clear all stored counterfactual solutions."""
        self._counterfactual_solutions.clear()
        self._counterfactual_params.clear()


class BenchmarkModelSolver:
    """Handles solving and caching of the benchmark model."""
    
    def __init__(self, data_file_name: str = "../data.npz"):
        # Handle different deployment environments
        if not Path(data_file_name).exists():
            # Try alternative paths for deployment
            possible_paths = [
                data_file_name,
                "data.npz",
                "../data.npz", 
                "Furusawa2023/data.npz",
                str(Path(__file__).parent.parent / "data.npz")
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_file_name = path
                    break
            else:
                raise FileNotFoundError(f"Could not find data file. Tried: {possible_paths}")
        
        self.data_file_name = data_file_name
        self._solution: Optional[ModelSol] = None
        self._params: Optional[ModelParams] = None
        self._is_solved = False
    
    def solve(self) -> Tuple[ModelSol, ModelParams]:
        """Solve the benchmark model and cache results."""
        if not self._is_solved:
            self._params = ModelParams.load_from_npz(self.data_file_name)
            model = Model(self._params)
            solver = ModelSolver(model)
            solver.solve()
            self._solution = model.sol
            self._is_solved = True
        
        # Ensure we return non-None values
        if self._solution is None or self._params is None:
            raise RuntimeError("Failed to solve benchmark model")
        return self._solution, self._params
    
    def get_cached_solution(self) -> Tuple[ModelSol, ModelParams]:
        """Get cached solution without re-solving."""
        if not self._is_solved or self._solution is None or self._params is None:
            raise RuntimeError("Model must be solved before accessing cached solution")
        return cast(ModelSol, self._solution), cast(ModelParams, self._params)
    
    def is_solved(self) -> bool:
        """Check if benchmark model has been solved."""
        return self._is_solved


class CounterfactualModelSolver:
    """Handles solving of counterfactual models with various tariff scenarios."""
    
    def __init__(self, benchmark_params: ModelParams):
        self.benchmark_params = benchmark_params
        self.country_names = list(benchmark_params.country_list)
        self.sector_names = list(benchmark_params.sector_list)
    
    def solve_tariff_scenario(self, 
                            importers: List[str], 
                            exporters: List[str], 
                            sectors: List[str], 
                            tariff_data: dict) -> ModelSol:
        """
        Solve counterfactual model with specified tariff changes.
        
        Args:
            importers: List of importing country names
            exporters: List of exporting country names  
            sectors: List of sector names affected
            tariff_data: Dict mapping either:
                - (importer, exporter) pairs to tariff rates (for country-based)
                - (importer, exporter, sector) triplets to tariff rates (for country-sector-based)
                - (importer, sector) pairs to tariff rates (for simple sector-based)
        
        Returns:
            ModelSol: Solution of the counterfactual model
        """
        # Create modified tariff matrix
        tilde_tau_1 = self.benchmark_params.tilde_tau.copy()
        
        # Determine tariff data format based on key structure
        sample_key = next(iter(tariff_data.keys())) if tariff_data else None
        if sample_key is None:
            # No tariff changes, solve with original parameters
            cf_model = Model(self.benchmark_params)
            cf_solver = ModelSolver(cf_model)
            cf_solver.solve()
            return cf_model.sol
            
        key_length = len(sample_key)
        
        for importer in importers:
            for exporter in exporters:
                for sector in sectors:
                    i = self.country_names.index(importer)
                    j = self.country_names.index(exporter)
                    s = self.sector_names.index(sector)
                    
                    if i != j:  # Don't modify domestic trade
                        if key_length == 3:
                            # Country-sector-based: (importer, exporter, sector) → rate
                            tariff_rate = tariff_data.get((importer, exporter, sector), 0)
                        elif key_length == 2:
                            # Check if it's (importer, exporter) or (importer, sector)
                            if (importer, exporter) in tariff_data:
                                # Country-based: (importer, exporter) → rate (applied to all sectors)
                                tariff_rate = tariff_data.get((importer, exporter), 0)
                            else:
                                # Simple sector-based: (importer, sector) → rate (applied to all exporters)
                                tariff_rate = tariff_data.get((importer, sector), 0)
                        else:
                            tariff_rate = 0
                        
                        tilde_tau_1[i, j, s] = (
                            self.benchmark_params.tilde_tau[i, j, s] + (tariff_rate / 100.0)
                        )
        
        # Create counterfactual model with modified parameters
        cf_params = deepcopy(self.benchmark_params)
        cf_params.tilde_tau = tilde_tau_1
        cf_model = Model(cf_params)
        cf_solver = ModelSolver(cf_model)
        cf_solver.solve()
        
        return cf_model.sol
    
    def generate_scenario_key(self, 
                            importers: List[str], 
                            exporters: List[str], 
                            sectors: List[str], 
                            tariff_data: dict) -> str:
        """Generate a unique key for a counterfactual scenario."""
        imp_str = "_".join(sorted(importers))
        exp_str = "_".join(sorted(exporters))
        sec_str = "_".join(sorted(sectors))
        
        # Create a hash of the tariff data to make the key unique
        import hashlib
        tariff_str = str(sorted(tariff_data.items()))
        tariff_hash = hashlib.md5(tariff_str.encode()).hexdigest()[:8]
        
        return f"tariff_{tariff_hash}_{imp_str}_to_{exp_str}_sectors_{sec_str}"


class ModelPipeline:
    """
    Main pipeline class that orchestrates benchmark and counterfactual model solving.
    """
    
    def __init__(self, data_file_name: str = "../data.npz"):
        # Handle different deployment environments
        if not Path(data_file_name).exists():
            # Try alternative paths for deployment
            possible_paths = [
                data_file_name,
                "data.npz",
                "../data.npz", 
                "Furusawa2023/data.npz",
                str(Path(__file__).parent.parent / "data.npz")
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_file_name = path
                    break
            else:
                raise FileNotFoundError(f"Could not find data file. Tried: {possible_paths}")
        
        self.benchmark_solver = BenchmarkModelSolver(data_file_name)
        self.storage = ModelResultStorage()
        self._counterfactual_solver: Optional[CounterfactualModelSolver] = None
    
    def ensure_benchmark_solved(self) -> Tuple[ModelSol, ModelParams]:
        """Ensure benchmark model is solved and return results."""
        if not self.benchmark_solver.is_solved():
            sol, params = self.benchmark_solver.solve()
            self.storage.store_benchmark(sol, params)
            
            # Initialize counterfactual solver with benchmark params
            self._counterfactual_solver = CounterfactualModelSolver(params)
            return sol, params
        else:
            return self.benchmark_solver.get_cached_solution()
    
    def solve_counterfactual(self, 
                           importers: List[str], 
                           exporters: List[str], 
                           sectors: List[str], 
                           tariff_data: dict) -> str:
        """
        Solve counterfactual scenario and return storage key.
        
        Returns:
            str: Key that can be used to retrieve the solution later
        """
        # Ensure benchmark is solved first
        _, benchmark_params = self.ensure_benchmark_solved()
        
        if self._counterfactual_solver is None:
            self._counterfactual_solver = CounterfactualModelSolver(benchmark_params)
        
        # Generate unique key for this scenario
        key = self._counterfactual_solver.generate_scenario_key(
            importers, exporters, sectors, tariff_data
        )
        
        # Check if already solved
        existing_sol, _ = self.storage.get_counterfactual(key)
        if existing_sol is not None:
            return key
        
        # Solve and store
        cf_solution = self._counterfactual_solver.solve_tariff_scenario(
            importers, exporters, sectors, tariff_data
        )
        
        # Create modified params for storage (though solution is what matters)
        cf_params = deepcopy(benchmark_params)
        self.storage.store_counterfactual(key, cf_solution, cf_params)
        
        return key
    
    def get_benchmark_results(self) -> Tuple[Optional[ModelSol], Optional[ModelParams]]:
        """Get benchmark model results."""
        return self.storage.get_benchmark()
    
    def get_counterfactual_results(self, key: str) -> Tuple[Optional[ModelSol], Optional[ModelParams]]:
        """Get counterfactual model results by key."""
        return self.storage.get_counterfactual(key)
    
    def list_solved_counterfactuals(self) -> List[str]:
        """List all solved counterfactual scenario keys."""
        return self.storage.list_counterfactuals()
    
    def clear_counterfactuals(self) -> None:
        """Clear all counterfactual solutions (keep benchmark)."""
        self.storage.clear_counterfactuals()


# Global pipeline instance for Streamlit caching
@st.cache_resource
def get_model_pipeline() -> ModelPipeline:
    """Get cached model pipeline instance."""
    return ModelPipeline()


@st.cache_data
def solve_benchmark_cached() -> Tuple[ModelSol, ModelParams]:
    """Cached function to solve benchmark model."""
    pipeline = get_model_pipeline()
    return pipeline.ensure_benchmark_solved()


@st.cache_data
def get_metadata_cached() -> Tuple[List[str], List[str], int, int]:
    """Cached function to get model metadata without solving."""
    # Load just the parameters to get metadata
    from pathlib import Path
    data_paths = [
        "data.npz",
        "../data.npz", 
        "Furusawa2023/data.npz"
    ]
    data_file = None
    for path in data_paths:
        if Path(path).exists():
            data_file = path
            break
    
    if data_file is None:
        raise FileNotFoundError("Could not find data.npz file")
    
    from models import ModelParams
    params = ModelParams.load_from_npz(data_file)
    return list(params.country_list), list(params.sector_list), params.N, params.S


@st.cache_data  
def solve_counterfactual_cached(importers: List[str], 
                              exporters: List[str], 
                              sectors: List[str], 
                              tariff_data: dict) -> str:
    """Cached function to solve counterfactual model."""
    pipeline = get_model_pipeline()
    return pipeline.solve_counterfactual(importers, exporters, sectors, tariff_data) 