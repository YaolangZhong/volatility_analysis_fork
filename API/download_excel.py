"""
Excel and CSV Download Module
=============================

This module handles all data export functionality for the Streamlit app,
including Excel files with variable data and CSV network data files.
"""

import io
import zipfile
import numpy as np
import pandas as pd
from typing import Optional
from datetime import datetime
import streamlit as st
import sys
from pathlib import Path

# Ensure parent directory is in path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from models import ModelSol, ModelParams


def create_csv_locally(sol: ModelSol, params: ModelParams, baseline_sol: Optional[ModelSol] = None) -> io.BytesIO:
    """Create CSV files for the 4 specialized network analysis sheets."""
    zip_buffer = io.BytesIO()
    
    # Calculate percentage change if baseline is provided
    def get_data_to_use(attr_name):
        attr_value = getattr(sol, attr_name)
        if baseline_sol is not None and hasattr(baseline_sol, attr_name):
            baseline_value = getattr(baseline_sol, attr_name)
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
        
        # Sheet 4: Sector Edge - sector_links variable only
        if hasattr(sol, 'sector_links'):
            sector_links_data = get_data_to_use('sector_links')
            sector_edge_rows = []
            
            # Create flattened sector names
            import_sector_labels = []
            export_sector_labels = []
            
            for country in params.country_list:
                for sector in params.sector_list:
                    import_sector_labels.append(f"{country}_{sector}")
                    export_sector_labels.append(f"{country}_{sector}")
            
            # Flatten sector_links from (N,S,N,S) to (NS,NS)
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
    
    def get_percentage_change_data(attr_value, attr_name):
        """Helper to calculate percentage change if baseline provided."""
        if baseline_sol is not None and hasattr(baseline_sol, attr_name):
            baseline_value = getattr(baseline_sol, attr_name)
            return 100 * (attr_value - baseline_value) / (np.abs(baseline_value) + 1e-8)
        return attr_value
    
    def create_dataframe_for_variable(attr_name, data_to_use):
        """Helper to create DataFrame based on data dimensions."""
        if data_to_use.ndim == 1:
            return pd.DataFrame({attr_name: data_to_use}, index=pd.Index(params.country_list))
        elif data_to_use.ndim == 2:
            return pd.DataFrame(data_to_use, index=pd.Index(params.country_list), columns=pd.Index(params.sector_list))
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
            return pd.DataFrame(rows)
        elif data_to_use.ndim == 4 and attr_name == 'sector_links':
            # Special handling for sector_links - flatten to 2D
            from visualization import flatten_sector_links_for_viz
            flattened_data, row_labels, col_labels = flatten_sector_links_for_viz(data_to_use, params)
            return pd.DataFrame(flattened_data, index=pd.Index(row_labels), columns=pd.Index(col_labels))
        elif attr_name == 'country_links':
            return pd.DataFrame(data_to_use, index=pd.Index(params.country_list), columns=pd.Index(params.country_list))
        else:
            return None  # Skip other 4D+ variables
    
    if variable_name:
        # Single variable export
        if hasattr(sol, variable_name):
            data = getattr(sol, variable_name)
            data_to_use = get_percentage_change_data(data, variable_name)
            df = create_dataframe_for_variable(variable_name, data_to_use)
            if df is not None:
                # Use BytesIO directly for single sheet
                df.to_excel(excel_buffer, sheet_name=variable_name, engine='openpyxl', index=True)
    else:
        # All variables export
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            for attr_name in dir(sol):
                if not attr_name.startswith('_') and hasattr(sol, attr_name):
                    attr_value = getattr(sol, attr_name)
                    if isinstance(attr_value, np.ndarray):
                        try:
                            data_to_use = get_percentage_change_data(attr_value, attr_name)
                            df = create_dataframe_for_variable(attr_name, data_to_use)
                            if df is not None:
                                sheet_name = attr_name[:31] if len(attr_name) > 31 else attr_name
                                df.to_excel(writer, sheet_name=sheet_name)
                        except Exception:
                            continue
    
    excel_buffer.seek(0)
    return excel_buffer


def create_excel_download_button(sol: ModelSol, params: ModelParams, scenario_key: Optional[str], 
                                variable_name: Optional[str] = None, unique_key: str = "", 
                                baseline_sol: Optional[ModelSol] = None, api_client=None, api_available=False):
    """Create download button for Excel export."""
    try:
        if api_client is not None and api_available and scenario_key:
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


def show_variable_download_section(sol: ModelSol, params: ModelParams, scenario_key: Optional[str] = None, 
                                 unique_key: str = "", baseline_sol: Optional[ModelSol] = None, 
                                 api_client=None, api_available=False):
    """Show variable download options."""
    st.markdown("### Download Options")
    if baseline_sol is not None:
        st.info("📊 **Excel**: All variables in separate sheets (includes 3D/4D variables)")
        st.info("📊 **Network Data**: ZIP file with 4 CSV files for network analysis (country/sector nodes & edges)")
    else:
        st.info("📊 **Excel**: All variables in separate sheets (includes sector_links, country_links and trade flows)")  
        st.info("📊 **Network Data**: ZIP file with 4 CSV files for network analysis (country/sector nodes & edges)")
    
    create_excel_download_button(sol, params, scenario_key, None, unique_key, baseline_sol, api_client, api_available) 