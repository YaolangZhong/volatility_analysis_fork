"""
Streamlit App with API Support and Download Features
===================================================

This app provides:
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
import io
import time
import hashlib
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import sys

# MUST be first Streamlit command
st.set_page_config(
    page_title="Model Output Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure parent directory is in path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import local modules
from model_pipeline import get_model_pipeline, get_metadata_cached
from visualization import ModelVisualizationEngine
from models import ModelSol, ModelParams, Model
from download_excel import create_excel_download_button, show_variable_download_section

def get_api_client(use_api: bool, api_url: str):
    """Initialize API client if needed."""
    if not use_api:
        return None, False
        
    try:
        from api_client import ModelAPIClient
        api_client = ModelAPIClient(api_url)
        # Test if API is available
        try:
            api_client.get_metadata()
            return api_client, True
        except:
            return api_client, False
    except ImportError:
        st.sidebar.error("API client not available. Install required dependencies.")
        return None, False

def show_api_status(use_api: bool, api_available: bool, api_url: str):
    """Show API connection status in sidebar."""
    if use_api:
        if api_available:
            st.sidebar.success(f"✅ API Connected: {api_url}")
        else:
            st.sidebar.error(f"❌ API Unavailable: {api_url}")
    else:
        st.sidebar.info("🔧 Local Mode Active")

def get_country_sector_names():
    """Get country and sector names for UI."""
    return get_metadata_cached()

def load_baseline_model(pickle_path: str = "baseline_model.pkl") -> tuple[ModelSol, ModelParams]:
    """Load pre-solved baseline model from pickle file."""
    if not Path(pickle_path).exists():
        raise FileNotFoundError(
            f"Baseline model file '{pickle_path}' not found. "
            f"Please run 'python solve_baseline_from_data.py' first to create it."
        )
    
    try:
        model = Model.load_from_pickle(pickle_path)
        if model.sol is None:
            raise RuntimeError("Loaded model has no solution. The baseline model may be corrupted.")
        return model.sol, model.params
    except Exception as e:
        raise RuntimeError(f"Failed to load baseline model from '{pickle_path}': {e}")

def solve_counterfactual_unified(importers, exporters, sectors, tariff_data, baseline_params, api_client=None, api_available=False):
    """Solve counterfactual model - unified interface for API/local."""
    if api_client is not None and api_available:
        try:
            scenario_key = api_client.solve_counterfactual(importers, exporters, sectors, tariff_data)
            return api_client.get_counterfactual_results(scenario_key), scenario_key
        except Exception as e:
            st.error(f"API counterfactual solving failed: {e}")
            return (None, None), None
    else:
        pipeline = get_model_pipeline()
        pipeline.initialize_with_baseline_params(baseline_params)
        scenario_key = pipeline.solve_counterfactual(importers, exporters, sectors, tariff_data)
        results = pipeline.get_counterfactual_results(scenario_key)
        return results, scenario_key

def create_counterfactual_ui(suffix: str = "", description: str = "Counterfactual") -> tuple:
    """Create UI components for counterfactual model configuration."""
    country_names, sector_names, _, _ = get_country_sector_names()
    
    if not country_names or not sector_names:
        st.error("Could not load country and sector data")
        return None, None, None, None

    # Importer selection
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{description}: Select ALL Importers", key=f"cf_select_all_importers{suffix}"):
            st.session_state[f"cf_importer_multiselect{suffix}"] = country_names
    with cols[1]:
        if st.button(f"{description}: Remove ALL Importers", key=f"cf_remove_all_importers{suffix}"):
            st.session_state[f"cf_importer_multiselect{suffix}"] = []
    
    cf_importers = st.multiselect(
        f"{description}: Select Importer(s)", 
        country_names, 
        default=[], 
        key=f"cf_importer_multiselect{suffix}"
    )

    # Exporter selection
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"{description}: Select ALL Exporters", key=f"cf_select_all_exporters{suffix}"):
            st.session_state[f"cf_exporter_multiselect{suffix}"] = country_names
    with cols[1]:
        if st.button(f"{description}: Remove ALL Exporters", key=f"cf_remove_all_exporters{suffix}"):
            st.session_state[f"cf_exporter_multiselect{suffix}"] = []
    
    cf_exporters = st.multiselect(
        f"{description}: Select Exporter(s)", 
        country_names, 
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

    # Advanced Tariff Rate Configuration
    st.subheader("🎛️ Tariff Rate Configuration")
    
    # Option to use uniform or custom tariffs
    tariff_mode = st.radio(
        "Choose tariff configuration mode:",
        ["Uniform Rate", "Custom Rates by Country", "Custom Rates by Sector", "Custom Rates by Country-Sector"],
        horizontal=True,
        key=f"tariff_mode{suffix}",
        help="Uniform: Same rate for all pairs. Country: Individual rates per country pair. Sector: One rate per sector (applied to all exporters). Country-Sector: Individual rates per country pair within each sector."
    )
    
    # Generate unified tariff_data based on the selected mode
    tariff_data = generate_unified_tariff_data(
        tariff_mode, cf_importers, cf_exporters, cf_sectors, suffix, description
    )

    # Validate selection
    if cf_importers and cf_exporters and cf_sectors and tariff_data:
        cf_importers_in_model_order = [c for c in country_names if c in cf_importers]
        cf_exporters_in_model_order = [c for c in country_names if c in cf_exporters]
        return cf_importers_in_model_order, cf_exporters_in_model_order, cf_sectors, tariff_data
    else:
        if not (cf_importers and cf_exporters and cf_sectors):
            st.warning(f"Please select at least one importer, one exporter, and one sector for {description}.")
        return None, None, None, None

def generate_unified_tariff_data(tariff_mode: str, cf_importers: list, cf_exporters: list, 
                               cf_sectors: list, suffix: str, description: str) -> dict:
    """Generate unified tariff_data format {(importer, exporter, sector): rate} for all modes."""
    tariff_data = {}
    
    if tariff_mode == "Uniform Rate":
        # Simple uniform tariff rate
        uniform_rate = st.slider(
            f"{description}: Uniform Tariff Rate (%)", 
            min_value=0, max_value=100, 
            value=20, step=1, 
            key=f"uniform_tariff_rate{suffix}"
        )
        
        # Apply uniform rate to all selected combinations
        if cf_importers and cf_exporters and cf_sectors:
            for importer in cf_importers:
                for exporter in cf_exporters:
                    if importer != exporter:  # No self-tariffs
                        for sector in cf_sectors:
                            tariff_data[(importer, exporter, sector)] = uniform_rate
    
    elif tariff_mode == "Custom Rates by Country":
        # Custom rates by country
        if cf_importers and cf_exporters:
            st.write("**Configure individual tariff rates for each importer-exporter pair:**")
            
            for i, importer in enumerate(cf_importers):
                st.write(f"**🏛️ {importer} (Importer) → Tariffs on imports from:**")
                
                # Add uniform adjustment slider for this importer
                col1, col2 = st.columns([2, 1])
                with col1:
                    uniform_importer_rate = st.slider(
                        f"⚡ Uniform adjustment for {importer} on ALL exporters (%)",
                        min_value=0, max_value=100,
                        value=20, step=1,
                        key=f"uniform_importer_{importer}{suffix}",
                        help=f"Set the same tariff rate for {importer} on imports from all selected exporters"
                    )
                with col2:
                    if st.button(f"Apply to All", key=f"apply_uniform_{importer}{suffix}"):
                        for exporter in cf_exporters:
                            if importer != exporter:
                                st.session_state[f"tariff_{importer}_{exporter}{suffix}"] = uniform_importer_rate
                        st.rerun()
                
                st.write("**Individual rates:**")
                
                # Create columns for better layout
                cols_per_row = 3
                exporter_chunks = [cf_exporters[j:j+cols_per_row] for j in range(0, len(cf_exporters), cols_per_row)]
                
                for chunk in exporter_chunks:
                    cols = st.columns(len(chunk))
                    for col_idx, exporter in enumerate(chunk):
                        with cols[col_idx]:
                            if importer != exporter:
                                tariff_rate = st.slider(
                                    f"🌍 {exporter}",
                                    min_value=0, max_value=100,
                                    value=20, step=1,
                                    key=f"tariff_{importer}_{exporter}{suffix}",
                                    help=f"Tariff rate imposed by {importer} on imports from {exporter}"
                                )
                                # Apply this rate to ALL sectors for this country pair
                                for sector in cf_sectors:
                                    tariff_data[(importer, exporter, sector)] = tariff_rate
                            else:
                                st.write(f"🚫 {exporter} (self)")
                                for sector in cf_sectors:
                                    tariff_data[(importer, exporter, sector)] = 0
                
                if i < len(cf_importers) - 1:
                    st.write("---")
    
    elif tariff_mode == "Custom Rates by Sector":
        # Sector-based tariffs
        if cf_importers and cf_sectors:
            st.write("**Configure tariff rates by sector for each importer:**")
            st.info("Each importer sets one tariff rate per sector, applied to ALL selected exporters.")
            
            for i, importer in enumerate(cf_importers):
                st.write(f"**🏛️ {importer} (Importer) → Sector-based tariffs:**")
                
                cols_per_row = 3
                sector_chunks = [cf_sectors[j:j+cols_per_row] for j in range(0, len(cf_sectors), cols_per_row)]
                
                for chunk in sector_chunks:
                    cols = st.columns(len(chunk))
                    for col_idx, sector in enumerate(chunk):
                        with cols[col_idx]:
                            tariff_rate = st.slider(
                                f"🏭 {sector}",
                                min_value=0, max_value=100,
                                value=20, step=1,
                                key=f"simple_sector_tariff_{importer}_{sector}{suffix}",
                                help=f"Tariff rate from {importer} on {sector} imports from ALL selected exporters"
                            )
                            for exporter in cf_exporters:
                                if importer != exporter:
                                    tariff_data[(importer, exporter, sector)] = tariff_rate
                
                if i < len(cf_importers) - 1:
                    st.write("---")
    
    else:  # Custom Rates by Country-Sector
        # Custom rates by country-sector
        if cf_importers and cf_exporters and cf_sectors:
            st.write("**Configure individual tariff rates by importer → exporter → sector:**")
            st.info("Each importer sets uniform rates per exporter, then customizes by individual sectors.")
            
            for i, importer in enumerate(cf_importers):
                st.write(f"**🏛️ {importer} (Importer)**")
                
                for j, exporter in enumerate(cf_exporters):
                    if importer != exporter:
                        st.write(f"**🌍 {exporter} (Exporter) ← Tariffs from {importer}:**")
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            uniform_importer_exporter_rate = st.slider(
                                f"⚡ Uniform adjustment for {importer} → {exporter} on ALL sectors (%)",
                                min_value=0, max_value=100,
                                value=20, step=1,
                                key=f"uniform_country_sector_{importer}_{exporter}{suffix}",
                                help=f"Set the same tariff rate for {importer} on imports from {exporter} across all selected sectors"
                            )
                        with col2:
                            if st.button(f"Apply to All", key=f"apply_uniform_country_sector_{importer}_{exporter}{suffix}"):
                                for sector in cf_sectors:
                                    st.session_state[f"country_sector_tariff_{importer}_{exporter}_{sector}{suffix}"] = uniform_importer_exporter_rate
                                st.rerun()
                        
                        st.write("**Individual sector rates:**")
                        
                        cols_per_row = 3
                        sector_chunks = [cf_sectors[k:k+cols_per_row] for k in range(0, len(cf_sectors), cols_per_row)]
                        
                        for chunk in sector_chunks:
                            cols = st.columns(len(chunk))
                            for col_idx, sector in enumerate(chunk):
                                with cols[col_idx]:
                                    sector_tariff_rate = st.slider(
                                        f"🏭 {sector}",
                                        min_value=0, max_value=100,
                                        value=20, step=1,
                                        key=f"country_sector_tariff_{importer}_{exporter}_{sector}{suffix}",
                                        help=f"Tariff rate imposed by {importer} on {sector} imports from {exporter}"
                                    )
                                    tariff_data[(importer, exporter, sector)] = sector_tariff_rate
                        
                        remaining_exporters = [e for e in cf_exporters[j+1:] if e != importer]
                        if remaining_exporters:
                            st.write("---")
                    else:
                        # Handle self-trade case
                        for sector in cf_sectors:
                            tariff_data[(importer, exporter, sector)] = 0
                
                if i < len(cf_importers) - 1:
                    st.write("=" * 50)
    
    return tariff_data

def main():
    """Main Streamlit app."""
    st.title("Model Output Explorer")
    
    # Configuration for API mode
    use_api = st.sidebar.checkbox("Use API Mode", value=False, help="Toggle between local and API-based model solving")
    api_url = st.sidebar.text_input("API Server URL", value="http://localhost:8000", help="URL of the API server")
    
    # Initialize API client if needed
    api_client, api_available = get_api_client(use_api, api_url)
    show_api_status(use_api, api_available, api_url)
    
    # Cache management
    st.sidebar.subheader("🔧 Cache Management")
    if st.sidebar.button("🗑️ Clear All Caches", help="Clear Streamlit caches to reload models with latest code changes"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # Clear visualization UI cache
            ui_keys_to_clear = [key for key in st.session_state.keys() 
                               if isinstance(key, str) and key.startswith(('ui_created_', 'selected_items_', 'fig_size_'))]
            for key in ui_keys_to_clear:
                del st.session_state[key]
            
            # Clear counterfactual session state
            if 'cf_solution' in st.session_state:
                st.session_state['cf_solution'] = None
                st.session_state['cf_params'] = None
                st.session_state['cf_scenario_key'] = None
                st.session_state['cf_config_hash'] = None
            st.sidebar.success("✅ Caches cleared! Models will be re-solved.")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Cache clearing failed: {e}")
    
    # Load baseline model
    try:
        with st.spinner("🔄 Loading baseline model..."):
            baseline_sol, baseline_params = load_baseline_model()
        st.success("✅ Baseline model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load baseline model: {e}")
        st.error("Please run 'python solve_baseline_from_data.py' first to create baseline_model.pkl")
        st.stop()

    # Initialize visualization engine
    country_names, sector_names, _, _ = get_country_sector_names()
    if country_names and sector_names:
        viz_engine = ModelVisualizationEngine(country_names, sector_names)
    else:
        st.error("Failed to initialize visualization engine")
        st.stop()
    
    st.markdown("---")
    
    # Model Type Selection
    model_type = st.radio(
        "📈 Select Model Type:",
        ["Baseline Model", "Counterfactual Model"],
        index=0,
        help="Baseline uses real-world tariff data, Counterfactual uses custom tariff scenarios"
    )
    
    if model_type == "Baseline Model":
        # Baseline Model Section
        st.header("🏛️ Baseline Model Analysis")
        st.write("Analysis of the baseline economic model with real-world tariff data.")
        
        if baseline_sol is not None and baseline_params is not None:
            st.success("✅ Baseline model ready for analysis!")
            
            st.markdown("---")
            show_variable_download_section(baseline_sol, baseline_params, "benchmark", "baseline", None, api_client, api_available)
            
            st.markdown("---")
            st.header("📊 Baseline Model Visualization")
            viz_engine.visualize_single_model(baseline_sol)
        else:
            st.error("❌ Baseline model failed to load. Please refresh the page or clear caches.")
    
    else:
        # Counterfactual Model Section
        st.header("🔧 Counterfactual Model Analysis")
        st.info("Configure custom tariff scenario for analysis")
        
        # Initialize session state for counterfactual results
        if 'cf_solution' not in st.session_state:
            st.session_state['cf_solution'] = None
            st.session_state['cf_params'] = None
            st.session_state['cf_scenario_key'] = None
            st.session_state['cf_config_hash'] = None
        
        # Clear solution button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Clear Solution & Reconfigure", key="clear_cf_solution"):
                st.session_state['cf_solution'] = None
                st.session_state['cf_params'] = None
                st.session_state['cf_scenario_key'] = None
                st.session_state['cf_config_hash'] = None
                st.rerun()
        
        # Show status indicator
        if st.session_state.get('cf_solution', None) is not None:
            st.success("✅ Counterfactual solution available - you can switch between view modes")
        else:
            st.warning("⏳ Configure and solve counterfactual below")
        
        # Counterfactual configuration
        cf_config = create_counterfactual_ui("", "Counterfactual")
        
        # Check if configuration has changed
        if cf_config and all(x is not None for x in cf_config):
            config_str = str(cf_config)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            if st.session_state.get('cf_config_hash') != config_hash:
                if st.session_state.get('cf_solution') is not None:
                    st.warning("⚠️ Configuration changed - previous solution cleared. Please solve again.")
                    st.session_state['cf_solution'] = None
                    st.session_state['cf_params'] = None
                    st.session_state['cf_scenario_key'] = None
                st.session_state['cf_config_hash'] = config_hash
        
        # Solving logic
        if all(x is not None for x in cf_config):
            if st.session_state.get('cf_solution') is None:
                if st.button("🚀 Solve Counterfactual Model", key="run_cf"):
                    importers, exporters, sectors, tariff_data = cf_config
                    try:
                        with st.spinner("Solving Counterfactual Model..."):
                            (cf_sol, cf_params), cf_scenario_key = solve_counterfactual_unified(
                                importers, exporters, sectors, tariff_data, baseline_params, api_client, api_available
                            )
                            if cf_sol is not None:
                                st.session_state['cf_solution'] = cf_sol
                                st.session_state['cf_params'] = cf_params
                                st.session_state['cf_scenario_key'] = cf_scenario_key
                                st.success("🎉 Counterfactual analysis completed successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Counterfactual solving failed")
                    except Exception as e:
                        st.error(f"❌ Error solving counterfactual: {e}")
            else:
                st.info("✅ Counterfactual already solved for current configuration. Use 'Clear Solution' to reconfigure.")
        
        # Get solutions from session state
        cf_sol = st.session_state.get('cf_solution', None)
        cf_params = st.session_state.get('cf_params', None)
        cf_scenario_key = st.session_state.get('cf_scenario_key', None)
        
        # Downloads and Visualization
        if cf_sol is not None and cf_params is not None:
            st.markdown("---")
            st.header("📥 Download Results & 📊 Visualization")
            
            view_mode = st.radio(
                "Choose view mode:",
                ["Level Values", "Percentage Change from Baseline"],
                horizontal=True,
                help="Level Values: See the actual values of variables under the counterfactual scenario. Percentage Change: See how much variables changed from baseline to counterfactual."
            )
            
            if view_mode == "Level Values":
                st.subheader("📥 Download Counterfactual Level Values")
                show_variable_download_section(cf_sol, cf_params, cf_scenario_key, "counterfactual_levels", None, api_client, api_available)
                
                st.subheader("📊 Counterfactual Model Visualization (Level Values)")
                viz_engine.visualize_single_model(cf_sol)
                
            else:  # Percentage Change from Baseline
                if baseline_sol is not None:
                    st.subheader("📥 Download Percentage Changes (Baseline → Counterfactual)")
                    show_variable_download_section(cf_sol, cf_params, cf_scenario_key, "counterfactual_changes", baseline_sol, api_client, api_available)
                    
                    st.subheader("📊 Percentage Change Visualization (Baseline → Counterfactual)")
                    viz_engine.visualize_comparison(baseline_sol, cf_sol)
                else:
                    st.warning("⚠️ Baseline model needed for percentage change comparison")
                    st.info("💡 The baseline model should be loaded automatically. If you see this message, there may have been an error.")
        elif all(x is not None for x in cf_config if cf_config):
            st.info("👆 Click 'Solve Counterfactual Model' to run the analysis and view results.")
        else:
            st.info("👆 Configure the tariff scenario above to proceed with counterfactual analysis.")

if __name__ == "__main__":
    main() 