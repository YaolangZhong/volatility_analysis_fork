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

# Ensure parent directory is in path for imports (works both locally and in deployment)
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Always import local modules for type hints and fallback
from model_pipeline import (
    get_model_pipeline, 
    solve_benchmark_cached, 
    solve_counterfactual_cached,
    get_metadata_cached
)
from visualization import ModelVisualizationEngine
from models import ModelSol, ModelParams

# Import API client for API mode
api_client = None
API_AVAILABLE = False
if USE_API:
    try:
        from api_client import ModelAPIClient
        api_client = ModelAPIClient(API_URL)
        # Test if API is available
        try:
            api_client.get_metadata()
            API_AVAILABLE = True
        except:
            API_AVAILABLE = False
    except ImportError:
        st.sidebar.error("API client not available. Install required dependencies.")
        API_AVAILABLE = False

def show_api_status():
    """Show API connection status in sidebar."""
    if USE_API:
        if API_AVAILABLE:
            st.sidebar.success(f"✅ API Connected: {API_URL}")
        else:
            st.sidebar.error(f"❌ API Unavailable: {API_URL}")
    else:
        st.sidebar.info("🔧 Local Mode Active")

def get_country_sector_names():
    """Get country and sector names - works with both API and local mode."""
    if USE_API and API_AVAILABLE and api_client is not None:
        try:
            metadata = api_client.get_metadata()  # type: ignore
            return metadata['countries'], metadata['sectors']
        except Exception as e:
            st.error(f"Failed to get metadata from API: {e}")
            return [], []
    else:
        # Use cached metadata that doesn't require solving the model
        countries, sectors, N, S = get_metadata_cached()
        return countries, sectors

def solve_benchmark_unified():
    """Solve benchmark model - unified interface for API/local."""
    if USE_API and API_AVAILABLE and api_client is not None:
        try:
            return api_client.solve_benchmark()  # type: ignore
        except Exception as e:
            st.error(f"API benchmark solving failed: {e}")
            return None, None
    else:
        return solve_benchmark_cached()

def solve_counterfactual_unified(importers, exporters, sectors, tariff_data):
    """Solve counterfactual model - unified interface for API/local."""
    if USE_API and API_AVAILABLE and api_client is not None:
        try:
            scenario_key = api_client.solve_counterfactual(importers, exporters, sectors, tariff_data)  # type: ignore
            return api_client.get_counterfactual_results(scenario_key), scenario_key  # type: ignore
        except Exception as e:
            st.error(f"API counterfactual solving failed: {e}")
            return (None, None), None
    else:
        pipeline = get_model_pipeline()
        scenario_key = pipeline.solve_counterfactual(importers, exporters, sectors, tariff_data)
        results = pipeline.get_counterfactual_results(scenario_key)
        return results, scenario_key

def create_excel_download_button(sol: ModelSol, params: ModelParams, scenario_key: Optional[str], variable_name: Optional[str] = None, unique_key: str = "", baseline_sol: Optional[ModelSol] = None):
    """Create download button for Excel export."""
    try:
        if USE_API and API_AVAILABLE and api_client is not None and scenario_key:
            # Use API download
            if variable_name:
                excel_data = api_client.download_variable_excel(scenario_key, variable_name)  # type: ignore
                filename = f"{scenario_key}_{variable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            else:
                excel_data = api_client.download_all_variables_excel(scenario_key)  # type: ignore
                filename = f"{scenario_key}_all_variables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            st.download_button(
                label=f"📥 Download {variable_name or 'All Variables'} (Excel)",
                data=excel_data,
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_btn_{unique_key}_{variable_name or 'all'}"
            )
        else:
            # Local Excel generation
            excel_buffer = create_excel_locally(sol, params, variable_name, baseline_sol)
            
            if variable_name:
                filename = f"{variable_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                label = f"📥 Download {variable_name} (Excel)"
            else:
                if baseline_sol is not None:
                    filename = f"percentage_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    label = "📥 Download Percentage Changes (Excel)"
                else:
                    filename = f"all_variables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    label = "📥 Download All Variables (Excel)"
            
            # Create download buttons side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label=label,
                    data=excel_buffer.getvalue(),
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_btn_excel_{unique_key}_{variable_name or 'all'}"
                )
            
            with col2:
                # CSV download (only for all variables, not single variable)
                if variable_name is None:
                    csv_buffer = create_csv_locally(sol, params, baseline_sol)
                    
                    if baseline_sol is not None:
                        csv_filename = f"percentage_changes_1D_2D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        csv_label = "📥 Download 1D & 2D Variables (CSV)"
                    else:
                        csv_filename = f"all_variables_1D_2D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        csv_label = "📥 Download 1D & 2D Variables (CSV)"
                    
                    st.download_button(
                        label=csv_label,
                        data=csv_buffer.getvalue(),
                        file_name=csv_filename,
                        mime="text/csv",
                        key=f"download_btn_csv_{unique_key}_{variable_name or 'all'}"
                    )
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def create_csv_locally(sol: ModelSol, params: ModelParams, baseline_sol: Optional[ModelSol] = None) -> io.BytesIO:
    """Create CSV file locally for 1D and 2D variables in long format."""
    csv_buffer = io.StringIO()
    
    # Prepare data rows
    rows = []
    
    for attr_name in dir(sol):
        if not attr_name.startswith('_') and hasattr(sol, attr_name):
            attr_value = getattr(sol, attr_name)
            
            if isinstance(attr_value, np.ndarray):
                # Calculate percentage change if baseline is provided
                data_to_use = attr_value
                if baseline_sol is not None and hasattr(baseline_sol, attr_name):
                    baseline_value = getattr(baseline_sol, attr_name)
                    # Calculate percentage change: 100 * (counterfactual - baseline) / baseline
                    data_to_use = 100 * (attr_value - baseline_value) / (np.abs(baseline_value) + 1e-8)
                
                if data_to_use.ndim == 1:
                    # 1D variable - treat as country-level with Sector = null
                    for i, country in enumerate(params.country_list):
                        if i < len(data_to_use):
                            rows.append({
                                'Country': country,
                                'Sector': 'null',
                                'Variable Name': attr_name,
                                'Value': data_to_use[i]
                            })
                elif data_to_use.ndim == 2:
                    # 2D variable - country x sector
                    for i, country in enumerate(params.country_list):
                        for j, sector in enumerate(params.sector_list):
                            if i < data_to_use.shape[0] and j < data_to_use.shape[1]:
                                rows.append({
                                    'Country': country,
                                    'Sector': sector,
                                    'Variable Name': attr_name,
                                    'Value': data_to_use[i, j]
                                })
                # Skip 3D+ variables as requested
    
    # Create DataFrame and write to CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_buffer, index=False)
    else:
        # Write header only if no data
        csv_buffer.write("Country,Sector,Variable Name,Value\n")
    
    # Convert to BytesIO for download
    csv_bytes = io.BytesIO()
    csv_bytes.write(csv_buffer.getvalue().encode('utf-8'))
    csv_bytes.seek(0)
    
    return csv_bytes

def create_excel_locally(sol: ModelSol, params: ModelParams, variable_name: Optional[str] = None, baseline_sol: Optional[ModelSol] = None) -> io.BytesIO:
    """Create Excel file locally."""
    excel_buffer = io.BytesIO()
    
    if variable_name:
        # Single variable export
        if hasattr(sol, variable_name):
            data = getattr(sol, variable_name)
            
            if data.ndim == 1:
                df = pd.DataFrame({variable_name: data}, index=params.country_list)  # type: ignore
            elif data.ndim == 2:
                df = pd.DataFrame(data, index=params.country_list, columns=params.sector_list)  # type: ignore
            elif data.ndim == 3:
                # Handle 3D arrays (e.g., trade flows)
                N, _, S = data.shape
                rows = []
                for n in range(N):
                    for i in range(N):
                        for s in range(S):
                            rows.append({
                                'Importer': params.country_list[n],
                                'Exporter': params.country_list[i], 
                                'Sector': params.sector_list[s],
                                variable_name: data[n, i, s]
                            })
                df = pd.DataFrame(rows)
            elif data.ndim == 4 and variable_name == 'sector_links':
                # Special handling for sector_links - flatten to 2D
                from API.visualization import flatten_sector_links_for_viz
                flattened_data, row_labels, col_labels = flatten_sector_links_for_viz(data, params)
                df = pd.DataFrame(flattened_data, index=pd.Index(row_labels), columns=pd.Index(col_labels))
            else:
                # For other 4D+ arrays, create a flattened version
                reshaped_data = data.reshape(data.shape[0], -1)
                col_names = [f"dim_{i}" for i in range(reshaped_data.shape[1])]
                df = pd.DataFrame(reshaped_data, index=params.country_list, columns=col_names)  # type: ignore
            
            df.to_excel(excel_buffer, sheet_name=variable_name, engine='openpyxl')  # type: ignore
    else:
        # All variables export
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:  # type: ignore
            for attr_name in dir(sol):
                if not attr_name.startswith('_') and hasattr(sol, attr_name):
                    attr_value = getattr(sol, attr_name)
                    
                    if isinstance(attr_value, np.ndarray):
                        try:
                            # Calculate percentage change if baseline is provided
                            data_to_use = attr_value
                            if baseline_sol is not None and hasattr(baseline_sol, attr_name):
                                baseline_value = getattr(baseline_sol, attr_name)
                                # Calculate percentage change: 100 * (counterfactual - baseline) / baseline
                                data_to_use = 100 * (attr_value - baseline_value) / (np.abs(baseline_value) + 1e-8)
                            
                            if data_to_use.ndim == 1:
                                df = pd.DataFrame({attr_name: data_to_use}, index=params.country_list)  # type: ignore
                            elif data_to_use.ndim == 2:
                                df = pd.DataFrame(data_to_use, index=params.country_list, columns=params.sector_list)  # type: ignore
                            elif data_to_use.ndim == 3:
                                # Handle 3D arrays (e.g., trade flows)
                                N, _, S = data_to_use.shape
                                rows = []
                                for n in range(N):
                                    for i in range(N):
                                        for s in range(S):
                                            rows.append({
                                                'Importer': params.country_list[n],
                                                'Exporter': params.country_list[i], 
                                                'Sector': params.sector_list[s],
                                                attr_name: data_to_use[n, i, s]
                                            })
                                df = pd.DataFrame(rows)
                            elif data_to_use.ndim == 4 and attr_name == 'sector_links':
                                # Special handling for sector_links - flatten to 2D
                                from API.visualization import flatten_sector_links_for_viz
                                flattened_data, row_labels, col_labels = flatten_sector_links_for_viz(data_to_use, params)
                                df = pd.DataFrame(flattened_data, index=pd.Index(row_labels), columns=pd.Index(col_labels))
                            else:
                                continue  # Skip other 4D+ variables
                            
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
    
    tariff_data = {}
    
    if tariff_mode == "Uniform Rate":
        # Simple uniform tariff rate
        uniform_rate = st.slider(
            f"{description}: Uniform Tariff Rate (%)", 
            min_value=0, max_value=100, 
            value=20, step=1, 
            key=f"uniform_tariff_rate{suffix}"
        )
        
        # Apply uniform rate to all selected pairs
        if cf_importers and cf_exporters:
            for importer in cf_importers:
                for exporter in cf_exporters:
                    tariff_data[(importer, exporter)] = uniform_rate
    
    elif tariff_mode == "Custom Rates by Country":
        # Custom rates by country
        if cf_importers and cf_exporters:
            st.write("**Configure individual tariff rates for each importer-exporter pair:**")
            
            # Create individual sliders for each importer-exporter pair
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
                        # Apply uniform rate to all exporters for this importer
                        for exporter in cf_exporters:
                            if importer != exporter:  # Don't apply to self
                                st.session_state[f"tariff_{importer}_{exporter}{suffix}"] = uniform_importer_rate
                        st.rerun()
                
                st.write("**Individual rates:**")
                
                # Create columns for better layout (max 3 exporters per row)
                cols_per_row = 3
                exporter_chunks = [cf_exporters[j:j+cols_per_row] for j in range(0, len(cf_exporters), cols_per_row)]
                
                for chunk in exporter_chunks:
                    cols = st.columns(len(chunk))
                    for col_idx, exporter in enumerate(chunk):
                        with cols[col_idx]:
                            if importer != exporter:  # Don't allow self-tariffs
                                tariff_rate = st.slider(
                                    f"🌍 {exporter}",
                                    min_value=0, max_value=100,
                                    value=20, step=1,
                                    key=f"tariff_{importer}_{exporter}{suffix}",
                                    help=f"Tariff rate imposed by {importer} on imports from {exporter}"
                                )
                                tariff_data[(importer, exporter)] = tariff_rate
                            else:
                                st.write(f"🚫 {exporter} (self)")
                                tariff_data[(importer, exporter)] = 0  # No self-tariffs
                
                if i < len(cf_importers) - 1:  # Add separator between importers
                    st.write("---")
    
    elif tariff_mode == "Custom Rates by Sector":
        # New simple sector-based tariffs: each importer sets one rate per sector for ALL exporters
        if cf_importers and cf_sectors:
            st.write("**Configure tariff rates by sector for each importer:**")
            st.info("Each importer sets one tariff rate per sector, applied to ALL selected exporters.")
            
            # Create individual sliders for each importer
            for i, importer in enumerate(cf_importers):
                st.write(f"**🏛️ {importer} (Importer) → Sector-based tariffs:**")
                
                # Create columns for better layout (max 3 sectors per row)
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
                            # Store using (importer, sector) as key - will be expanded to all exporters
                            tariff_data[(importer, sector)] = tariff_rate
                
                # Add spacing between importers
                if i < len(cf_importers) - 1:
                    st.write("---")
    
    else:
        # Custom rates by country-sector (detailed version)
        if cf_importers and cf_exporters and cf_sectors:
            st.write("**Configure individual tariff rates by importer → exporter → sector:**")
            st.info("Each importer sets uniform rates per exporter, then customizes by individual sectors.")
            
            # Create individual sliders organized by importer → exporter → sector
            for i, importer in enumerate(cf_importers):
                st.write(f"**🏛️ {importer} (Importer)**")
                
                # Create sliders for each exporter within this importer
                for j, exporter in enumerate(cf_exporters):
                    if importer != exporter:  # Don't allow self-tariffs
                        st.write(f"**🌍 {exporter} (Exporter) ← Tariffs from {importer}:**")
                        
                        # Add uniform adjustment slider for this importer-exporter combination across all sectors
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
                                # Apply uniform rate to all sectors for this importer-exporter combination
                                for sector in cf_sectors:
                                    st.session_state[f"country_sector_tariff_{importer}_{exporter}_{sector}{suffix}"] = uniform_importer_exporter_rate
                                st.rerun()
                        
                        st.write("**Individual sector rates:**")
                        
                        # Create columns for better layout (max 3 sectors per row)
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
                                    # Store with (importer, exporter, sector) triplet as key
                                    tariff_data[(importer, exporter, sector)] = sector_tariff_rate
                        
                        # Add separator between exporters within same importer (only if not the last valid exporter)
                        remaining_exporters = [e for e in cf_exporters[j+1:] if e != importer]
                        if remaining_exporters:
                            st.write("---")
                    else:
                        # Handle self-trade case
                        for sector in cf_sectors:
                            tariff_data[(importer, exporter, sector)] = 0  # No self-tariffs
                
                # Add separator between importers
                if i < len(cf_importers) - 1:
                    st.write("=" * 50)
    
    # Validate selection
    if cf_importers and cf_exporters and cf_sectors and tariff_data:
        cf_importers_in_model_order = [c for c in country_names if c in cf_importers]
        cf_exporters_in_model_order = [c for c in country_names if c in cf_exporters]
        return cf_importers_in_model_order, cf_exporters_in_model_order, cf_sectors, tariff_data
    else:
        if not (cf_importers and cf_exporters and cf_sectors):
            st.warning(f"Please select at least one importer, one exporter, and one sector for {description}.")
        return None, None, None, None

def show_variable_download_section(sol: ModelSol, params: ModelParams, scenario_key: Optional[str] = None, unique_key: str = "", baseline_sol: Optional[ModelSol] = None):
    """Show variable download options."""
    st.markdown("### Download Options")
    if baseline_sol is not None:
        st.info("📊 **Excel**: All variables in separate sheets (includes 3D/4D variables)")
        st.info("📋 **CSV**: 1D & 2D variables in long format (Country, Sector, Variable Name, Value)")
    else:
        st.info("📊 **Excel**: All variables in separate sheets (includes sector_links and trade flows)")  
        st.info("📋 **CSV**: 1D & 2D variables in standardized long format for easy analysis")
    
    create_excel_download_button(sol, params, scenario_key, None, unique_key, baseline_sol)

def main():
    """Main application logic."""
    st.title("Enhanced Model Output Explorer")
    
    # Show API status
    show_api_status()
    
    # Add cache clearing button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Cache Management")
    if st.sidebar.button("🗑️ Clear All Caches", help="Clear Streamlit caches to reload models with latest code changes"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
            # Also clear session state caches
            if 'baseline_solution' in st.session_state:
                st.session_state['baseline_solution'] = None
                st.session_state['baseline_params'] = None
            if 'cf_solution' in st.session_state:
                st.session_state['cf_solution'] = None
                st.session_state['cf_params'] = None
                st.session_state['cf_scenario_key'] = None
            st.sidebar.success("✅ Caches cleared! Models will be re-solved.")
            # Use rerun for Streamlit 1.18+
            if hasattr(st, 'rerun'):
                st.rerun()
            elif hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()  # type: ignore
            else:
                # Fallback for very old versions
                st.sidebar.info("Please refresh the page manually to see changes")
        except Exception as e:
            st.sidebar.error(f"Cache clearing failed: {e}")
            st.sidebar.info("Please refresh the page manually")
    
    # Check if system is ready
    if USE_API and not API_AVAILABLE:
        st.error("🚫 API mode selected but API server is not available. Please check the server or switch to local mode.")
        return
    
    # Initialize session state for counterfactual solution
    if 'cf_solution' not in st.session_state:
        st.session_state['cf_solution'] = None
        st.session_state['cf_params'] = None
        st.session_state['cf_scenario_key'] = None
    
    # Initialize session state for baseline solution (to avoid re-solving)
    if 'baseline_solution' not in st.session_state:
        st.session_state['baseline_solution'] = None
        st.session_state['baseline_params'] = None

    # 🚀 IMMEDIATE BASELINE LOADING: Load baseline model on startup
    # This ensures the model is ready for immediate use
    if st.session_state['baseline_solution'] is None:
        with st.spinner("🔄 Loading baseline model (this may take 15-30 seconds)..."):
            baseline_sol, baseline_params = solve_benchmark_unified()
            st.session_state['baseline_solution'] = baseline_sol
            st.session_state['baseline_params'] = baseline_params
            if baseline_sol is not None:
                st.success("✅ Baseline model loaded successfully!")
            else:
                st.error("❌ Failed to load baseline model")
                return

    # Get solutions from session state (they're now guaranteed to be loaded)
    baseline_sol = st.session_state['baseline_solution']
    baseline_params = st.session_state['baseline_params']
    
    # Initialize visualization engine
    country_names, sector_names = get_country_sector_names()
    if country_names and sector_names:
        viz_engine = ModelVisualizationEngine(country_names, sector_names)
    else:
        st.error("Failed to initialize visualization engine")
        return
    
    # Model selection
    st.header("Model Selection")
    
    # Model type selection
    st.header("🎯 Select Model Type")
    model_type = st.radio(
        "Choose which model to explore:",
        ["Baseline Model", "Counterfactual Model"],
        horizontal=True,
        help="Baseline uses real-world tariff data, Counterfactual uses custom tariff scenarios"
    )
    
    if model_type == "Baseline Model":
        # Baseline Model Section
        st.header("🏛️ Baseline Model Analysis")
        st.info("Using real-world tariff data from the dataset")
        
        # Model is already loaded on startup, so we can show results immediately
        baseline_scenario_key = "benchmark"
        
        # Show baseline model status
        if baseline_sol is not None and baseline_params is not None:
            st.success("✅ Baseline model ready for analysis!")
            
            # Downloads and Visualization for baseline
            st.markdown("---")
            st.header("📥 Download Results")
            show_variable_download_section(baseline_sol, baseline_params, baseline_scenario_key, "baseline")
            
            # Show baseline visualization  
            st.markdown("---")
            st.header("📊 Baseline Model Visualization")
            viz_engine.visualize_single_model(baseline_sol)  # Single model view
            
        else:
            st.error("❌ Baseline model failed to load. Please refresh the page or clear caches.")
            if st.button("🔄 Refresh Page", key="refresh_page"):
                st.rerun()
    
    else:
        # Counterfactual Model Section
        st.header("🔧 Counterfactual Model Analysis")
        st.info("Configure custom tariff scenario for analysis")
        
        # Clear solution button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Clear Solution & Reconfigure", key="clear_cf_solution"):
                st.session_state['cf_solution'] = None
                st.session_state['cf_params'] = None
                st.session_state['cf_scenario_key'] = None
                st.rerun()
        
        # Status indicator for counterfactual
        if st.session_state.get('cf_solution', None) is not None:
            st.success("✅ Counterfactual solution available")
        else:
            st.warning("⏳ Configure and solve counterfactual below")
        
        # Counterfactual configuration
        cf_config = create_counterfactual_ui("", "Counterfactual")
        cf_sol, cf_params, cf_scenario_key = None, None, None
        
        if all(x is not None for x in cf_config):
            if st.button("🚀 Solve Counterfactual Model", key="run_cf"):
                importers, exporters, sectors, tariff_data = cf_config
                try:
                    with st.spinner("Solving Counterfactual Model..."):
                        (cf_sol, cf_params), cf_scenario_key = solve_counterfactual_unified(
                            importers, exporters, sectors, tariff_data
                        )
                        if cf_sol is not None:
                            st.session_state['cf_solution'] = cf_sol
                            st.session_state['cf_params'] = cf_params
                            st.session_state['cf_scenario_key'] = cf_scenario_key
                            st.success("🎉 Counterfactual analysis completed successfully!")
                except Exception as e:
                    st.error(f"❌ Error solving counterfactual: {e}")
        
        # Get solution from session state if available
        cf_sol = st.session_state.get('cf_solution', None)
        cf_params = st.session_state.get('cf_params', None)
        cf_scenario_key = st.session_state.get('cf_scenario_key', None)
        
        # Downloads and Visualization for counterfactual
        if cf_sol is not None and cf_params is not None:
            st.markdown("---")
            st.header("📥 Download Results & 📊 Visualization")
            
            # Option to choose between level values or percentage change
            st.subheader("Choose view mode:")
            view_mode = st.radio(
                "",
                ["Level Values", "Percentage Change from Baseline"],
                horizontal=True,
                help="Level Values: See the actual values of variables under the counterfactual scenario. Percentage Change: See how much variables changed from baseline to counterfactual."
            )
            
            if view_mode == "Level Values":
                st.subheader("📥 Download Counterfactual Level Values")
                show_variable_download_section(cf_sol, cf_params, cf_scenario_key, "counterfactual_levels")
                
                # Show counterfactual visualization (level values)
                st.subheader("📊 Counterfactual Model Visualization (Level Values)")
                viz_engine.visualize_single_model(cf_sol)
                
            else:  # Percentage Change from Baseline
                # Check if baseline model is loaded for comparison
                if baseline_sol is not None:
                    st.subheader("📥 Download Percentage Changes (Baseline → Counterfactual)")
                    show_variable_download_section(cf_sol, cf_params, cf_scenario_key, "counterfactual_changes", baseline_sol)
                    
                    # Show percentage change visualization  
                    st.subheader("📊 Percentage Change Visualization (Baseline → Counterfactual)")
                    viz_engine.visualize_comparison(baseline_sol, cf_sol)
                else:
                    st.warning("⚠️ Baseline model needed for percentage change comparison")
                    st.info("💡 The baseline model was loaded on startup, but there may have been an error. Try refreshing the page.")
                    
                    if st.button("🔄 Refresh Page", key="refresh_for_baseline"):
                        st.rerun()
        elif all(x is not None for x in cf_config if cf_config):
            st.info("👆 Click 'Solve Counterfactual Model' to run the analysis and view results.")
        else:
            st.info("👆 Configure the tariff scenario above to proceed with counterfactual analysis.")

if __name__ == "__main__":
    main() 