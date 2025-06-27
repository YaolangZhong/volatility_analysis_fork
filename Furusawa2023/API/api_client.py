"""
API Client for Economic Model Communication
==========================================

This client provides the same interface as model_pipeline.py but communicates
with the API server instead of solving models locally. This allows for 
seamless switching between local and API-based model solving.
"""

import requests
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models import ModelSol, ModelParams
import json

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class ModelAPIClient:
    """
    API Client that mimics the interface of model_pipeline.py.
    
    This allows the Streamlit app to work with either local or remote model solving
    by simply switching between ModelPipeline and ModelAPIClient.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._cached_metadata = None
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            raise APIError(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise APIError(f"Could not connect to API server at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = response.json().get('detail', str(e))
            except:
                error_detail = str(e)
            raise APIError(f"API error: {error_detail}")
        except Exception as e:
            raise APIError(f"Unexpected error: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if API server is healthy."""
        try:
            response = self._make_request("GET", "/")
            return response.status_code == 200
        except APIError:
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata (countries, sectors, dimensions)."""
        if self._cached_metadata is None:
            response = self._make_request("GET", "/metadata")
            self._cached_metadata = response.json()
        return self._cached_metadata
    
    def solve_benchmark(self) -> Tuple[ModelSol, ModelParams]:
        """
        Solve benchmark model via API.
        Returns same format as solve_benchmark_cached().
        """
        # First solve via API
        response = self._make_request("GET", "/benchmark/solve")
        result = response.json()
        scenario_key = result["scenario_key"]
        
        # Get the solution data
        return self._get_solution_and_params(scenario_key)
    
    def solve_counterfactual(self, importers: List[str], exporters: List[str], 
                           sectors: List[str], tariff_rate: float) -> str:
        """
        Solve counterfactual model via API.
        Returns scenario key for later retrieval.
        """
        request_data = {
            "importers": importers,
            "exporters": exporters,
            "sectors": sectors,
            "tariff_rate": tariff_rate
        }
        
        response = self._make_request("POST", "/counterfactual/solve", json=request_data)
        result = response.json()
        return result["scenario_key"]
    
    def get_counterfactual_results(self, scenario_key: str) -> Tuple[ModelSol, ModelParams]:
        """
        Get counterfactual results for a given scenario.
        Returns same format as model_pipeline.get_counterfactual_results().
        """
        return self._get_solution_and_params(scenario_key)
    
    def _get_solution_and_params(self, scenario_key: str) -> Tuple[ModelSol, ModelParams]:
        """Get solution and parameters for a scenario."""
        # Get variables info
        response = self._make_request("GET", f"/models/{scenario_key}/variables")
        var_info = response.json()
        
        countries = var_info["countries"]
        sectors = var_info["sectors"]
        N, S = len(countries), len(sectors)
        
        # Get all solution variables
        solution_data = {}
        for var_name in var_info["variables"]:
            var_response = self._make_request("GET", f"/models/{scenario_key}/variable/{var_name}")
            var_data = var_response.json()
            solution_data[var_name] = np.array(var_data["data"])
        
        # Create ModelSol object
        sol = ModelSol(**solution_data)
        
        # Create minimal ModelParams for compatibility
        # Note: We only have basic info, not full parameters
        params = self._create_minimal_params(countries, sectors)
        
        return sol, params
    
    def _create_minimal_params(self, countries: List[str], sectors: List[str]) -> ModelParams:
        """Create minimal ModelParams for API compatibility."""
        N, S = len(countries), len(sectors)
        
        # Create dummy params with minimal required structure
        params = ModelParams(
            N=N,
            S=S,
            alpha=np.ones((N, S)) / S,  # Dummy uniform distribution
            beta=np.ones((N, S)) * 0.5,  # Dummy 50% value added share
            gamma=np.ones((N, S, S)) / S,  # Dummy uniform IO coefficients 
            theta=np.ones(S) * 5.0,  # Dummy trade elasticity
            pif=np.ones((N, N, S)) / N,  # Dummy uniform trade shares
            pim=np.ones((N, N, S)) / N,  # Dummy uniform trade shares
            pi=np.ones((N, N, S)) / N,  # Dummy uniform trade shares
            tilde_tau=np.ones((N, N, S)),  # No trade costs
            Xf=np.ones((N, S)),  # Dummy expenditure
            Xm=np.ones((N, S)),  # Dummy expenditure
            X=np.ones((N, S)),  # Dummy expenditure
            V=np.ones(N),  # Dummy value added
            D=np.zeros(N),  # Zero trade deficit
            country_list=countries,
            sector_list=sectors
        )
        
        return params
    
    def list_solved_models(self) -> List[Dict[str, Any]]:
        """List all solved models on the server."""
        response = self._make_request("GET", "/models")
        return response.json()
    
    def download_variable_excel(self, scenario_key: str, variable_name: str) -> bytes:
        """Download variable as Excel file."""
        response = self._make_request("GET", f"/models/{scenario_key}/download/excel/{variable_name}")
        return response.content
    
    def download_all_variables_excel(self, scenario_key: str) -> bytes:
        """Download all variables as Excel file."""
        response = self._make_request("GET", f"/models/{scenario_key}/download/all")
        return response.content
    
    def clear_models(self) -> Dict[str, str]:
        """Clear all solved models (except benchmark)."""
        response = self._make_request("POST", "/models/clear")
        return response.json()


# Factory function to create appropriate pipeline based on configuration
def get_model_pipeline(use_api: bool = False, api_url: str = "http://localhost:8000"):
    """
    Factory function that returns either local ModelPipeline or API client.
    
    Args:
        use_api: If True, use API client; if False, use local pipeline
        api_url: URL of the API server (if using API)
    
    Returns:
        Pipeline object with consistent interface
    """
    if use_api:
        client = ModelAPIClient(api_url)
        if not client.health_check():
            raise APIError(f"API server at {api_url} is not available")
        return client
    else:
        # Import and return local pipeline
        from model_pipeline import get_model_pipeline as get_local_pipeline
        return get_local_pipeline()


# Compatibility functions that match the original model_pipeline.py interface
def solve_benchmark_cached(use_api: bool = False, api_url: str = "http://localhost:8000") -> Tuple[ModelSol, ModelParams]:
    """
    Solve benchmark model - API or local version.
    This maintains compatibility with the original function signature.
    """
    if use_api:
        client = ModelAPIClient(api_url)
        return client.solve_benchmark()
    else:
        from model_pipeline import solve_benchmark_cached as solve_local
        sol, params = solve_local()
        if sol is None or params is None:
            raise RuntimeError("Benchmark model solving failed")
        return sol, params


def solve_counterfactual_cached(importers: List[str], exporters: List[str], 
                              sectors: List[str], tariff_rate: float,
                              use_api: bool = False, api_url: str = "http://localhost:8000") -> Tuple[ModelSol, ModelParams]:
    """
    Solve counterfactual model - API or local version.
    This maintains compatibility with the original function signature.
    """
    if use_api:
        client = ModelAPIClient(api_url)
        scenario_key = client.solve_counterfactual(importers, exporters, sectors, tariff_rate)
        return client.get_counterfactual_results(scenario_key)
    else:
        # Use local pipeline
        from model_pipeline import get_model_pipeline as get_local_pipeline
        pipeline = get_local_pipeline()
        scenario_key = pipeline.solve_counterfactual(importers, exporters, sectors, tariff_rate)
        return pipeline.get_counterfactual_results(scenario_key)


# Test function to verify API connectivity
def test_api_connection(api_url: str = "http://localhost:8000") -> bool:
    """Test if API server is reachable and working."""
    try:
        client = ModelAPIClient(api_url)
        if client.health_check():
            logger.info(f"✅ API server at {api_url} is healthy")
            
            # Test metadata endpoint
            metadata = client.get_metadata()
            logger.info(f"📊 Model has {metadata['N']} countries and {metadata['S']} sectors")
            
            return True
        else:
            logger.error(f"❌ API server at {api_url} is not responding")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to connect to API server: {e}")
        return False


if __name__ == "__main__":
    # Test API connection
    import sys
    
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    if test_api_connection(api_url):
        print("API client is working correctly!")
    else:
        print("API client test failed!") 