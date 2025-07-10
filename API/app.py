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
import time # Added for cache clearing

# MUST be first Streamlit command
st.set_page_config(
    page_title="Enhanced Model Output Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    solve_counterfactual_cached,
    get_metadata_cached
)
from visualization import ModelVisualizationEngine
from models import ModelSol, ModelParams, Model

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
    # Use cached metadata that doesn't require solving the model
    countries, sectors, N, S = get_metadata_cached()
    return countries, sectors

def load_baseline_model(pickle_path: str = "baseline_model.pkl") -> tuple[ModelSol, ModelParams]:
    """
    Load pre-solved baseline model from pickle file.
    
    Parameters
    ----------
    pickle_path : str, optional
        Path to baseline model pickle file (default: "baseline_model.pkl")
        
    Returns
    -------
    tuple[ModelSol, ModelParams]
        Loaded baseline solution and parameters
        
    Raises
    ------
    FileNotFoundError
        If baseline_model.pkl is not found
    RuntimeError
        If model loading fails
    """
    from pathlib import Path
    
    # Check if pickle file exists
    if not Path(pickle_path).exists():
        raise FileNotFoundError(
            f"Baseline model file '{pickle_path}' not found. "
            f"Please run 'python solve_baseline_from_data.py' first to create it."
        )
    
    try:
        # Load the pre-solved model
        model = Model.load_from_pickle(pickle_path)
        
        if model.sol is None:
            raise RuntimeError("Loaded model has no solution. The baseline model may be corrupted.")
        
        return model.sol, model.params
        
    except Exception as e:
        raise RuntimeError(f"Failed to load baseline model from '{pickle_path}': {e}")

def solve_benchmark_unified(api_client: Optional[object], api_available: bool, api_url: str):
    """Load baseline model - unified interface for API/local (DEPRECATED - use load_baseline_model)."""
    if api_client is not None and api_available:
        try:
            return api_client.solve_benchmark()  # type: ignore
        except Exception as e:
            st.error(f"API baseline loading failed: {e}")
            return None, None
    else:
        return load_baseline_model()

def solve_counterfactual_unified(importers, exporters, sectors, tariff_data, api_client=None, api_available=False):
    """Solve counterfactual model - unified interface for API/local."""
    if api_client is not None and api_available:
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

def create_excel_download_button(sol: ModelSol, params: ModelParams, scenario_key: Optional[str], variable_name: Optional[str] = None, unique_key: str = "", baseline_sol: Optional[ModelSol] = None, api_client=None, api_available=False):
    """Create download button for Excel export."""
    try:
        if api_client is not None and api_available and scenario_key:
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
                # Network data download (only for all variables, not single variable)
                if variable_name is None:
                    csv_buffer = create_csv_locally(sol, params, baseline_sol)
                    
                    if baseline_sol is not None:
                        network_filename = f"network_data_percentage_changes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                        network_label = "📊 Download Network Data (4 CSV files)"
                    else:
                        network_filename = f"network_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                        network_label = "📊 Download Network Data (4 CSV files)"
                    
                    st.download_button(
                        label=network_label,
                        data=csv_buffer.getvalue(),
                        file_name=network_filename,
                        mime="application/zip",
                        key=f"download_btn_network_{unique_key}_{variable_name or 'all'}"
                    )
    except Exception as e:
        st.error(f"Failed to create download: {e}")

def create_csv_locally(sol: ModelSol, params: ModelParams, baseline_sol: Optional[ModelSol] = None) -> io.BytesIO:
    """Create CSV files for the 4 specialized network analysis sheets."""
    import zipfile
    
    # Create a ZIP file containing 4 separate CSV files
    zip_buffer = io.BytesIO()
    
    # Calculate percentage change if baseline is provided
    def get_data_to_use(attr_name):
        attr_value = getattr(sol, attr_name)
        if baseline_sol is not None and hasattr(baseline_sol, attr_name):
            baseline_value = getattr(baseline_sol, attr_name)
            # Calculate percentage change: 100 * (counterfactual - baseline) / baseline
            return 100 * (attr_value - baseline_value) / (np.abs(baseline_value) + 1e-8)
        return attr_value
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        
        # Sheet 1: Country Node - I_prime variable only
        if hasattr(sol, 'I_prime'):
            I_prime_data = get_data_to_use('I_prime')
            country_node_rows = []
            for i, country in enumerate(params.country_list):
                if i < len(I_prime_data):
                    country_node_rows.append({
                        'Country': country,
                        'Sector': 'null',
                        'Variable Name': 'country income',
                        'Value': I_prime_data[i]
                    })
            if country_node_rows:
                df_country_node = pd.DataFrame(country_node_rows)
                csv_content = df_country_node.to_csv(index=False)
                zipf.writestr('country_node.csv', csv_content)
        
        # Sheet 2: Sector Node - X_prod_prime variable only
        if hasattr(sol, 'X_prod_prime'):
            X_prod_data = get_data_to_use('X_prod_prime')
            sector_node_rows = []
            for i, country in enumerate(params.country_list):
                for j, sector in enumerate(params.sector_list):
                    if i < X_prod_data.shape[0] and j < X_prod_data.shape[1]:
                        sector_node_rows.append({
                            'Country': country,
                            'Sector': sector,
                            'Variable Name': 'sector output',
                            'Value': X_prod_data[i, j]
                        })
            if sector_node_rows:
                df_sector_node = pd.DataFrame(sector_node_rows)
                csv_content = df_sector_node.to_csv(index=False)
                zipf.writestr('sector_node.csv', csv_content)
        
        # Sheet 3: Country Edge - country_links variable only
        if hasattr(sol, 'country_links'):
            country_links_data = get_data_to_use('country_links')
            country_edge_rows = []
            for i, import_country in enumerate(params.country_list):
                for j, export_country in enumerate(params.country_list):
                    if i < country_links_data.shape[0] and j < country_links_data.shape[1]:
                        country_edge_rows.append({
                            'Import Country': import_country,
                            'Export Country': export_country,
                            'Variable Name': 'country level export',
                            'Value': country_links_data[i, j]
                        })
            if country_edge_rows:
                df_country_edge = pd.DataFrame(country_edge_rows)
                csv_content = df_country_edge.to_csv(index=False)
                zipf.writestr('country_edge.csv', csv_content)
        
        # Sheet 4: Sector Edge - sector_links variable only, flattened from (N,S,N,S) to (NS,NS)
        if hasattr(sol, 'sector_links'):
            sector_links_data = get_data_to_use('sector_links')
            sector_edge_rows = []
            
            # Create flattened sector names: "CountryName_SectorName"
            # ik pairs: importer_country + output_sector (rows)
            import_sector_labels = []
            # ns pairs: exporter_country + input_sector (columns)
            export_sector_labels = []
            
            # ik pairs (rows): importer_country + output_sector
            for country in params.country_list:
                for sector in params.sector_list:
                    import_sector_labels.append(f"{country}_{sector}")
            
            # ns pairs (columns): exporter_country + input_sector
            for country in params.country_list:
                for sector in params.sector_list:
                    export_sector_labels.append(f"{country}_{sector}")
            
            # Flatten sector_links from (N,S,N,S) to (NS,NS)
            # sector_links[i, k, n, s] -> reshape gives (ik, ns) indexing
            # Rows: ik pairs (importer_country + output_sector)
            # Columns: ns pairs (exporter_country + input_sector)
            N, S = len(params.country_list), len(params.sector_list)
            flattened_data = sector_links_data.reshape(N*S, N*S)
            
            for i, import_sector in enumerate(import_sector_labels):
                for j, export_sector in enumerate(export_sector_labels):
                    if i < flattened_data.shape[0] and j < flattened_data.shape[1]:
                        sector_edge_rows.append({
                            'Import Sector': import_sector,
                            'Export Sector': export_sector,
                            'Variable Name': 'sector level export',
                            'Value': flattened_data[i, j]
                        })
            
            if sector_edge_rows:
                df_sector_edge = pd.DataFrame(sector_edge_rows)
                csv_content = df_sector_edge.to_csv(index=False)
                zipf.writestr('sector_edge.csv', csv_content)
    
    zip_buffer.seek(0)
    return zip_buffer

def create_excel_locally(sol: ModelSol, params: ModelParams, variable_name: Optional[str] = None, baseline_sol: Optional[ModelSol] = None) -> io.BytesIO:
    """Create Excel file locally."""
    excel_buffer = io.BytesIO()
    
    if variable_name:
        # Single variable export
        if hasattr(sol, variable_name):
            data = getattr(sol, variable_name)
            
            # Calculate percentage change if baseline is provided
            data_to_use = data
            if baseline_sol is not None and hasattr(baseline_sol, variable_name):
                baseline_value = getattr(baseline_sol, variable_name)
                # Calculate percentage change: 100 * (counterfactual - baseline) / baseline
                data_to_use = 100 * (data - baseline_value) / (np.abs(baseline_value) + 1e-8)
            
            if data_to_use.ndim == 1:
                df = pd.DataFrame({variable_name: data_to_use}, index=params.country_list)  # type: ignore
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
                                variable_name: data_to_use[n, i, s]
                            })
                df = pd.DataFrame(rows)
            elif data_to_use.ndim == 4 and variable_name == 'sector_links':
                # Special handling for sector_links - flatten to 2D
                from API.visualization import flatten_sector_links_for_viz
                flattened_data, row_labels, col_labels = flatten_sector_links_for_viz(data_to_use, params)
                df = pd.DataFrame(flattened_data, index=pd.Index(row_labels), columns=pd.Index(col_labels))
            elif variable_name == 'country_links':
                # Handle country_links - country x country matrix
                df = pd.DataFrame(data_to_use, index=pd.Index(params.country_list), columns=pd.Index(params.country_list))
            else:
                # For other 4D+ arrays, create a flattened version
                reshaped_data = data_to_use.reshape(data_to_use.shape[0], -1)
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
                            elif attr_name == 'country_links':
                                # Handle country_links - country x country matrix
                                df = pd.DataFrame(data_to_use, index=pd.Index(params.country_list), columns=pd.Index(params.country_list))
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
    """
    Generate unified tariff_data format {(importer, exporter, sector): rate} for all modes.
    This simplifies downstream processing by always using the same data structure.
    """
    tariff_data = {}
    
    if tariff_mode == "Uniform Rate":
        # Simple uniform tariff rate
        uniform_rate = st.slider(
            f"{description}: Uniform Tariff Rate (%)", 
            min_value=0, max_value=100, 
            value=20, step=1, 
            key=f"uniform_tariff_rate{suffix}"
        )
        
        # Apply uniform rate to all selected (importer, exporter, sector) combinations
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
                                # Apply this rate to ALL sectors for this country pair
                                for sector in cf_sectors:
                                    tariff_data[(importer, exporter, sector)] = tariff_rate
                            else:
                                st.write(f"🚫 {exporter} (self)")
                                # No self-tariffs for any sector
                                for sector in cf_sectors:
                                    tariff_data[(importer, exporter, sector)] = 0
                
                if i < len(cf_importers) - 1:  # Add separator between importers
                    st.write("---")
    
    elif tariff_mode == "Custom Rates by Sector":
        # Sector-based tariffs: each importer sets one rate per sector for ALL exporters
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
                            # Apply this rate to ALL exporters for this importer-sector combination
                            for exporter in cf_exporters:
                                if importer != exporter:  # No self-tariffs
                                    tariff_data[(importer, exporter, sector)] = tariff_rate
                
                # Add spacing between importers
                if i < len(cf_importers) - 1:
                    st.write("---")
    
    else:  # Custom Rates by Country-Sector
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
    
    return tariff_data

def show_variable_download_section(sol: ModelSol, params: ModelParams, scenario_key: Optional[str] = None, unique_key: str = "", baseline_sol: Optional[ModelSol] = None, api_client=None, api_available=False):
    """Show variable download options."""
    st.markdown("### Download Options")
    if baseline_sol is not None:
        st.info("📊 **Excel**: All variables in separate sheets (includes 3D/4D variables)")
        st.info("📊 **Network Data**: ZIP file with 4 CSV files for network analysis (country/sector nodes & edges)")
    else:
        st.info("📊 **Excel**: All variables in separate sheets (includes sector_links, country_links and trade flows)")  
        st.info("📊 **Network Data**: ZIP file with 4 CSV files for network analysis (country/sector nodes & edges)")
    
    create_excel_download_button(sol, params, scenario_key, None, unique_key, baseline_sol, api_client, api_available)



def main():
    """Main Streamlit app."""
    # Setup UI
    st.title("Enhanced Model Output Explorer")
    
    # Configuration for API mode (moved to main function)
    use_api = st.sidebar.checkbox("Use API Mode", value=False, help="Toggle between local and API-based model solving")
    api_url = st.sidebar.text_input("API Server URL", value="http://localhost:8000", help="URL of the API server")
    
    # Initialize API client if needed
    api_client, api_available = get_api_client(use_api, api_url)
    
    # Show API status
    show_api_status(use_api, api_available, api_url)
    
    # Add cache clearing button in sidebar
    try:
        st.sidebar.subheader("🔧 Cache Management")
        if st.sidebar.button("🗑️ Clear All Caches", help="Clear Streamlit caches to reload models with latest code changes"):
            try:
                st.cache_data.clear()
                st.cache_resource.clear()
                # Also clear counterfactual session state
                if 'cf_solution' in st.session_state:
                    st.session_state['cf_solution'] = None
                    st.session_state['cf_params'] = None
                    st.session_state['cf_scenario_key'] = None
                    st.session_state['cf_config_hash'] = None
                st.sidebar.success("✅ Caches cleared! Models will be re-solved.")
                time.sleep(1)  # Brief pause for user feedback
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Cache clearing failed: {e}")
    except Exception as e:
        st.sidebar.error(f"Sidebar setup failed: {e}")
    
    # Load baseline model (simple loading, no caching complexity)
    try:
        with st.spinner("🔄 Loading baseline model..."):
            baseline_sol, baseline_params = load_baseline_model()
        st.success("✅ Baseline model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load baseline model: {e}")
        st.error("Please run 'python solve_baseline_from_data.py' first to create baseline_model.pkl")
        st.stop()  # Stop execution if baseline model can't be loaded

    # Initialize visualization engine
    country_names, sector_names = get_country_sector_names()
    if country_names and sector_names:
        viz_engine = ModelVisualizationEngine(country_names, sector_names)
    else:
        st.error("Failed to initialize visualization engine")
        st.stop()
    
    # baseline_sol and baseline_params are already loaded above
    
    # Main UI
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
        
        # Store scenario key for downloads
        baseline_scenario_key = "benchmark"
        
        # Show baseline model status
        if baseline_sol is not None and baseline_params is not None:
            st.success("✅ Baseline model ready for analysis!")
            
            # Downloads and Visualization for baseline
            st.markdown("---")
            
            show_variable_download_section(baseline_sol, baseline_params, baseline_scenario_key, "baseline", None, api_client, api_available)
            
            # Show baseline visualization
            st.markdown("---")
            st.header("📊 Baseline Model Visualization")
            viz_engine.visualize_single_model(baseline_sol)  # Single model view
            
        else:
            st.error("❌ Baseline model failed to load. Please refresh the page or clear caches.")
    
    else:
        # Counterfactual Model Section
        st.header("🔧 Counterfactual Model Analysis")
        st.info("Configure custom tariff scenario for analysis")
        
        # Initialize session state for counterfactual results (minimal persistence)
        if 'cf_solution' not in st.session_state:
            st.session_state['cf_solution'] = None
            st.session_state['cf_params'] = None
            st.session_state['cf_scenario_key'] = None
            st.session_state['cf_config_hash'] = None
        
        # Add a clear solution button for starting fresh
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
        
        # Check if configuration has changed (to detect if we need to re-solve)
        if cf_config and all(x is not None for x in cf_config):
            import hashlib
            config_str = str(cf_config)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # If configuration changed, clear the old solution
            if st.session_state.get('cf_config_hash') != config_hash:
                if st.session_state.get('cf_solution') is not None:
                    st.warning("⚠️ Configuration changed - previous solution cleared. Please solve again.")
                    st.session_state['cf_solution'] = None
                    st.session_state['cf_params'] = None
                    st.session_state['cf_scenario_key'] = None
                st.session_state['cf_config_hash'] = config_hash
        
        # Solving button and logic
        if all(x is not None for x in cf_config):
            # Check if we already have a solution for this configuration
            if st.session_state.get('cf_solution') is None:
                if st.button("🚀 Solve Counterfactual Model", key="run_cf"):
                    importers, exporters, sectors, tariff_data = cf_config
                    try:
                        with st.spinner("Solving Counterfactual Model..."):
                            (cf_sol, cf_params), cf_scenario_key = solve_counterfactual_unified(
                                importers, exporters, sectors, tariff_data, api_client, api_available
                            )
                            if cf_sol is not None:
                                # Store in session state
                                st.session_state['cf_solution'] = cf_sol
                                st.session_state['cf_params'] = cf_params
                                st.session_state['cf_scenario_key'] = cf_scenario_key
                                st.success("🎉 Counterfactual analysis completed successfully!")
                                st.rerun()  # Refresh to show results
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
                show_variable_download_section(cf_sol, cf_params, cf_scenario_key, "counterfactual_levels", None, api_client, api_available)
                
                # Show counterfactual visualization (level values)
                st.subheader("📊 Counterfactual Model Visualization (Level Values)")
                viz_engine.visualize_single_model(cf_sol)
                
            else:  # Percentage Change from Baseline
                # Check if baseline model is loaded for comparison
                if baseline_sol is not None:
                    st.subheader("📥 Download Percentage Changes (Baseline → Counterfactual)")
                    show_variable_download_section(cf_sol, cf_params, cf_scenario_key, "counterfactual_changes", baseline_sol, api_client, api_available)
                    
                    # Show percentage change visualization  
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