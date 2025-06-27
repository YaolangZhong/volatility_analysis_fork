"""
Enhanced Streamlit App with API Support and Download Features
=============================================================

This enhanced app provides:
1. Toggle between local and API-based model solving
2. Excel download functionality for all model variables
3. Clean separation between data generation and visualization
4. Maintains all existing functionality

The app can work in two modes:
- Local Mode: Uses model_pipeline.py for direct solving (original behavior)
- API Mode: Communicates with api_server.py for remote solving
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import io
from typing import List, Optional, Tuple, Union
from datetime import datetime

# MUST be first Streamlit command
st.set_page_config(layout="wide")

# Configuration for API mode
USE_API = st.sidebar.checkbox("Use API Mode", value=False, help="Toggle between local and API-based model solving")
API_URL = st.sidebar.text_input("API Server URL", value="http://localhost:8000", help="URL of the API server")

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import appropriate modules based on mode
if USE_API:
    try:
        from api_client import (
            ModelAPIClient, 
            APIError, 
            test_api_connection
        )
        api_client = ModelAPIClient(API_URL)
        API_AVAILABLE = api_client.health_check()
    except ImportError:
        st.error("API client not available. Please install required dependencies.")
        API_AVAILABLE = False
    except Exception as e:
        st.error(f"API connection failed: {e}")
        API_AVAILABLE = False
else:
    # Use original local modules
    from model_pipeline import (
        get_model_pipeline, 
        solve_benchmark_cached, 
        solve_counterfactual_cached
    )
    from visualization import ModelVisualizationEngine
    from models import ModelSol, ModelParams
    API_AVAILABLE = True  # Local mode is always available

def show_api_status():
    """Show API connection status in sidebar."""
    if USE_API:
        if API_AVAILABLE:
            st.sidebar.success("✅ API Connected")
            try:
                metadata = api_client.get_metadata()
                st.sidebar.info(f"📊 {metadata['N']} countries, {metadata['S']} sectors")
            except:
                st.sidebar.warning("⚠️ API metadata unavailable")
        else:
            st.sidebar.error("❌ API Unavailable")
    else:
        st.sidebar.info("🏠 Local Mode")

def get_country_sector_names():
    """Get country and sector names - works with both API and local mode."""
    if USE_API and API_AVAILABLE:
        try:
            metadata = api_client.get_metadata()
            return metadata['countries'], metadata['sectors']
        except Exception as e:
            st.error(f"Failed to get metadata from API: {e}")
            return [], []
    else:
        _, params = solve_benchmark_cached()
        return list(params.country_list), list(params.sector_list)

def solve_benchmark_unified():
    """Solve benchmark model - unified interface for API/local."""
    if USE_API and API_AVAILABLE:
        try:
            return api_client.solve_benchmark()
        except Exception as e:
            st.error(f"API benchmark solving failed: {e}")
            return None, None
    else:
        return solve_benchmark_cached()

def solve_counterfactual_unified(importers, exporters, sectors, tariff_rate):
    """Solve counterfactual model - unified interface for API/local."""
    if USE_API and API_AVAILABLE:
        try:
            scenario_key = api_client.solve_counterfactual(importers, exporters, sectors, tariff_rate)
            return api_client.get_counterfactual_results(scenario_key), scenario_key
        except Exception as e:
            st.error(f"API counterfactual solving failed: {e}")
            return (None, None), None
    else:
        pipeline = get_model_pipeline()
        scenario_key = pipeline.solve_counterfactual(importers, exporters, sectors, tariff_rate)
        results = pipeline.get_counterfactual_results(scenario_key)
        return results, scenario_key

def create_excel_download_button(sol: ModelSol, params: ModelParams, scenario_key: Optional[str], variable_name: Optional[str] = None):
    """Create download button for Excel export."""
    try:
        if USE_API and API_AVAILABLE and scenario_key:
            # Use API download
            if variable_name:
                excel_data = api_client.download_variable_excel(scenario_key, variable_name)
                filename = f"{scenario_key}_{variable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            else:
                excel_data = api_client.download_all_variables_excel(scenario_key)
                filename = f"{scenario_key}_all_variables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label=f"📥 Download {variable_name or 'All Variables'} (Excel)",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            # Local Excel generation
            excel_buffer = create_excel_locally(sol, params, variable_name)
            
            if variable_name:
                filename = f"{variable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                label = f"📥 Download {variable_name} (Excel)"
            else:
                filename = f"all_variables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                label = "📥 Download All Variables (Excel)"
            
            st.download_button(
                label=label,
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def create_excel_locally(sol: ModelSol, params: ModelParams, variable_name: Optional[str] = None) -> io.BytesIO:
    """Create Excel file locally."""
    excel_buffer = io.BytesIO()
    
    if variable_name:
        # Single variable export
        if hasattr(sol, variable_name):
            data = getattr(sol, variable_name)
            
            if data.ndim == 1:
                df = pd.DataFrame({variable_name: data}, index=params.country_list)
            elif data.ndim == 2:
                df = pd.DataFrame(data, index=params.country_list, columns=params.sector_list)
            else:
                # For 3D+ arrays, create a flattened version
                reshaped_data = data.reshape(data.shape[0], -1)
                col_names = [f"dim_{i}_{j}" for i in range(data.shape[1]) for j in range(data.shape[2])]
                df = pd.DataFrame(reshaped_data, index=params.country_list, columns=col_names)
            
            df.to_excel(excel_buffer, sheet_name=variable_name, engine='openpyxl')
    else:
        # All variables export
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            for attr_name in dir(sol):
                if not attr_name.startswith('_') and hasattr(sol, attr_name):
                    attr_value = getattr(sol, attr_name)
                    
                    if isinstance(attr_value, np.ndarray):
                        try:
                            if attr_value.ndim == 1:
                                df = pd.DataFrame({attr_name: attr_value}, index=params.country_list)
                            elif attr_value.ndim == 2:
                                df = pd.DataFrame(attr_value, index=params.country_list, columns=params.sector_list)
                            else:
                                continue  # Skip 3D+ for now
                            
                            sheet_name = attr_name[:31] if len(attr_name) > 31 else attr_name
                            df.to_excel(writer, sheet_name=sheet_name)
                        except Exception:
                            continue
    
    excel_buffer.seek(0)
    return excel_buffer

def create_counterfactual_ui(suffix: str = "", description: str = "Counterfactual") -> tuple:
    """Create UI components for counterfactual model configuration."""
    country_names, sector_names = get_country_sector_names()
    
    if not country_names or not sector_names:
        st.error("Could not load country and sector data")
        return None, None, None, None
    
    # Priority countries for better UX
    priority_countries = [
        "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "ITA", "CAN", "KOR", "IND",
        "ESP", "NLD", "BEL", "SWE", "RUS", "BRA", "MEX", "AUS"
    ]
    country_names_sorted = (
        [c for c in priority_countries if c in country_names] + 
        [c for c in country_names if c not in priority_countries]
    )

    # Importer selection
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{description}: Select ALL Importers", key=f"cf_select_all_importers{suffix}"):
            st.session_state[f"cf_importer_multiselect{suffix}"] = country_names_sorted
    with cols[1]:
        if st.button(f"{description}: Remove ALL Importers", key=f"cf_remove_all_importers{suffix}"):
            st.session_state[f"cf_importer_multiselect{suffix}"] = []
    
    cf_importers = st.multiselect(
        f"{description}: Select Importer(s)", 
        country_names_sorted, 
        default=[], 
        key=f"cf_importer_multiselect{suffix}"
    )

    # Exporter selection
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{description}: Select ALL Exporters", key=f"cf_select_all_exporters{suffix}"):
            st.session_state[f"cf_exporter_multiselect{suffix}"] = country_names_sorted
    with cols[1]:
        if st.button(f"{description}: Remove ALL Exporters", key=f"cf_remove_all_exporters{suffix}"):
            st.session_state[f"cf_exporter_multiselect{suffix}"] = []
    
    cf_exporters = st.multiselect(
        f"{description}: Select Exporter(s)", 
        country_names_sorted, 
        default=[], 
        key=f"cf_exporter_multiselect{suffix}"
    )

    # Sector selection
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{description}: Select ALL Sectors", key=f"cf_select_all_sectors{suffix}"):
            st.session_state[f"cf_sector_multiselect{suffix}"] = sector_names
    with cols[1]:
        if st.button(f"{description}: Remove ALL Sectors", key=f"cf_remove_all_sectors{suffix}"):
            st.session_state[f"cf_sector_multiselect{suffix}"] = []
    
    cf_sectors = st.multiselect(
        f"{description}: Select Sector(s)", 
        sector_names, 
        default=sector_names, 
        key=f"cf_sector_multiselect{suffix}"
    )

    # Tariff rate
    tariff_rate = st.slider(
        f"{description}: Tariff Rate (%)", 
        min_value=0, max_value=100, 
        value=20, step=1, 
        key=f"tariff_rate{suffix}"
    )

    # Validate selection
    if cf_importers and cf_exporters and cf_sectors:
        cf_importers_in_model_order = [c for c in country_names if c in cf_importers]
        cf_exporters_in_model_order = [c for c in country_names if c in cf_exporters]
        return cf_importers_in_model_order, cf_exporters_in_model_order, cf_sectors, tariff_rate
    else:
        st.warning(f"Please select at least one importer, one exporter, and one sector for {description}.")
        return None, None, None, None

def show_variable_download_section(sol: ModelSol, params: ModelParams, scenario_key: Optional[str] = None):
    """Show variable download options."""
    st.subheader("📥 Download Results")
    
    # Get all available variables
    variable_names = [attr for attr in dir(sol) 
                     if not attr.startswith('_') and isinstance(getattr(sol, attr), np.ndarray)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Individual Variables:**")
        selected_var = st.selectbox("Select variable to download:", variable_names)
        if selected_var:
            var_data = getattr(sol, selected_var)
            st.write(f"Shape: {var_data.shape}")
            create_excel_download_button(sol, params, scenario_key, selected_var)
    
    with col2:
        st.write("**All Variables:**")
        st.write(f"Available variables: {len(variable_names)}")
        create_excel_download_button(sol, params, scenario_key)

def main():
    """Main application logic."""
    st.title("Enhanced Model Output Explorer")
    
    # Show API status
    show_api_status()
    
    # Check if system is ready
    if USE_API and not API_AVAILABLE:
        st.error("🚫 API mode selected but API server is not available. Please check the server or switch to local mode.")
        return
    
    # Initialize session state for solutions
    if 'cf1_solution' not in st.session_state:
        st.session_state['cf1_solution'] = None
        st.session_state['cf1_params'] = None
        st.session_state['cf1_scenario_key'] = None
    if 'cf2_solution' not in st.session_state:
        st.session_state['cf2_solution'] = None
        st.session_state['cf2_params'] = None
        st.session_state['cf2_scenario_key'] = None
    
    # Initialize visualization engine
    country_names, sector_names = get_country_sector_names()
    if country_names and sector_names:
        viz_engine = ModelVisualizationEngine(country_names, sector_names)
    else:
        st.error("Failed to initialize visualization engine")
        return
    
    # Model selection
    st.header("Model Selection")
    
    # Clear solutions button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Clear All Solutions", key="clear_solutions"):
            st.session_state['cf1_solution'] = None
            st.session_state['cf1_params'] = None
            st.session_state['cf1_scenario_key'] = None
            st.session_state['cf2_solution'] = None
            st.session_state['cf2_params'] = None
            st.session_state['cf2_scenario_key'] = None
            st.rerun()
    
    model_view = "Compare Two Models"  # Fixed to comparison mode for now
    
    if model_view == "Compare Two Models":
        # Model 1 selection
        st.subheader("Model 1")
        model1_type = st.radio("Model 1", ["Benchmark", "Counterfactual"], horizontal=True, key="model1_type")
        
        # Status indicator for Model 1
        if model1_type == "Benchmark":
            st.info("✅ Benchmark model is always available")
        else:
            if st.session_state.get('cf1_solution', None) is not None:
                st.success("✅ Counterfactual 1 solution available")
            else:
                st.warning("⏳ Counterfactual 1 not solved yet")
        
        sol1, params1, scenario_key1 = None, None, None
        if model1_type == "Benchmark":
            sol1, params1 = solve_benchmark_unified()
            scenario_key1 = "benchmark"
        else:
            cf1_config = create_counterfactual_ui("_1", "Counterfactual 1")
            if all(x is not None for x in cf1_config):
                if st.button("Run Counterfactual 1", key="run_cf1"):
                    importers, exporters, sectors, tariff_rate = cf1_config
                    try:
                        with st.spinner("Solving Counterfactual 1..."):
                            (sol1, params1), scenario_key1 = solve_counterfactual_unified(
                                importers, exporters, sectors, tariff_rate
                            )
                            if sol1 is not None:
                                st.session_state['cf1_solution'] = sol1
                                st.session_state['cf1_params'] = params1
                                st.session_state['cf1_scenario_key'] = scenario_key1
                                st.success("Counterfactual 1 solved successfully!")
                    except Exception as e:
                        st.error(f"Error solving counterfactual 1: {e}")
                
                # Get solution from session state if available
                sol1 = st.session_state.get('cf1_solution', None)
                params1 = st.session_state.get('cf1_params', None)
                scenario_key1 = st.session_state.get('cf1_scenario_key', None)
        
        # Model 2 selection
        st.subheader("Model 2")
        model2_type = st.radio("Model 2", ["Benchmark", "Counterfactual"], horizontal=True, key="model2_type")
        
        # Status indicator for Model 2
        if model2_type == "Benchmark":
            st.info("✅ Benchmark model is always available")
        else:
            if st.session_state.get('cf2_solution', None) is not None:
                st.success("✅ Counterfactual 2 solution available")
            else:
                st.warning("⏳ Counterfactual 2 not solved yet")
        
        sol2, params2, scenario_key2 = None, None, None
        if model2_type == "Benchmark":
            sol2, params2 = solve_benchmark_unified()
            scenario_key2 = "benchmark"
        else:
            cf2_config = create_counterfactual_ui("_2", "Counterfactual 2")
            if all(x is not None for x in cf2_config):
                if st.button("Run Counterfactual 2", key="run_cf2"):
                    importers, exporters, sectors, tariff_rate = cf2_config
                    try:
                        with st.spinner("Solving Counterfactual 2..."):
                            (sol2, params2), scenario_key2 = solve_counterfactual_unified(
                                importers, exporters, sectors, tariff_rate
                            )
                            if sol2 is not None:
                                st.session_state['cf2_solution'] = sol2
                                st.session_state['cf2_params'] = params2
                                st.session_state['cf2_scenario_key'] = scenario_key2
                                st.success("Counterfactual 2 solved successfully!")
                    except Exception as e:
                        st.error(f"Error solving counterfactual 2: {e}")
                
                # Get solution from session state if available
                sol2 = st.session_state.get('cf2_solution', None)
                params2 = st.session_state.get('cf2_params', None)
                scenario_key2 = st.session_state.get('cf2_scenario_key', None)
        
        # Visualization and Downloads
        if sol1 is not None and sol2 is not None:
            # Show comparison visualization
            viz_engine.visualize_comparison(sol1, sol2)
            
            # Download sections for both models
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📥 Download {model1_type} 1 Results")
                show_variable_download_section(sol1, params1, scenario_key1)
            
            with col2:
                st.subheader(f"📥 Download {model2_type} 2 Results")
                show_variable_download_section(sol2, params2, scenario_key2)
                
        else:
            st.info("Please configure both models to see comparison and download options.")
    
    else:
        # Single model view (for future extension)
        st.info("Single model view - feature not implemented yet")

if __name__ == "__main__":
    main() 