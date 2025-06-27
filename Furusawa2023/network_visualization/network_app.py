"""
Network Graph Streamlit App
===========================

Dedicated Streamlit application for network graph visualization of the trade model.
This app focuses specifically on the network representation described in temp.md.
"""

import streamlit as st
import numpy as np
from typing import List, Optional

# Import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "API"))

from model_pipeline import (
    get_model_pipeline,
    solve_benchmark_cached,
    solve_counterfactual_cached
)
from network_graph import NetworkGraphEngine
from models import ModelSol, ModelParams

st.set_page_config(layout="wide", page_title="Trade Network Visualization")

def get_country_sector_names():
    """Get country and sector names from the benchmark model."""
    _, params = solve_benchmark_cached()
    country_names = list(params.country_list)
    sector_names = list(params.sector_list)
    return country_names, sector_names

def create_counterfactual_ui(suffix: str = "", description: str = "Counterfactual") -> tuple:
    """
    Create UI components for counterfactual model configuration.
    
    Args:
        suffix: Unique suffix for widget keys (e.g., "_1", "_2")
        description: Description for the counterfactual (e.g., "Counterfactual 1")
    
    Returns:
        Tuple of (importers, exporters, sectors, tariff_rate) or (None, None, None, None) if incomplete
    """
    country_names, sector_names = get_country_sector_names()
    
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
        # Ensure selected countries are in model order
        cf_importers_in_model_order = [c for c in country_names if c in cf_importers]
        cf_exporters_in_model_order = [c for c in country_names if c in cf_exporters]
        return cf_importers_in_model_order, cf_exporters_in_model_order, cf_sectors, tariff_rate
    else:
        st.warning(f"Please select at least one importer, one exporter, and one sector for {description}.")
        return None, None, None, None

def main():
    """Main application logic for network graph visualization."""
    st.title("Trade Network Graph Visualization")
    
    # Introduction
    st.markdown("""
    This application visualizes the trade model as a network graph where:
    - **Nodes** represent Country-Sector pairs (e.g., "USA_Manufacturing", "CHN_Electronics")
    - **Node size** represents total expenditure (X matrix for baseline, X_prime for counterfactual)
    - **Country clusters** show aggregated expenditure by country
    - **Colors** indicate changes: 🔴 Red for increases, 🟢 Green for decreases
    """)
    
    # Initialize session state for solutions
    if 'baseline_solution' not in st.session_state:
        st.session_state['baseline_solution'] = None
    if 'cf_solution' not in st.session_state:
        st.session_state['cf_solution'] = None
    
    # Initialize network graph engine
    _, params = solve_benchmark_cached()
    network_engine = NetworkGraphEngine(params)
    
    # Visualization mode selection
    st.header("Visualization Mode")
    viz_mode = st.radio(
        "Choose visualization mode:",
        ["Baseline Only", "Counterfactual Comparison", "Side-by-Side Comparison"],
        horizontal=True
    )
    
    if viz_mode == "Baseline Only":
        st.markdown("---")
        network_engine.create_baseline_graph_ui()
        
    elif viz_mode == "Counterfactual Comparison":
        st.markdown("---")
        
        # Configuration section
        st.header("Counterfactual Configuration")
        cf_config = create_counterfactual_ui("", "Counterfactual")
        
        if all(x is not None for x in cf_config):
            if st.button("Run Counterfactual Analysis", key="run_cf"):
                importers, exporters, sectors, tariff_rate = cf_config
                try:
                    with st.spinner("Solving Counterfactual Model..."):
                        pipeline = get_model_pipeline()
                        scenario_key = pipeline.solve_counterfactual(importers, exporters, sectors, tariff_rate)
                        sol, _ = pipeline.get_counterfactual_results(scenario_key)
                        st.session_state['cf_solution'] = sol
                        st.success("Counterfactual analysis completed!")
                except Exception as e:
                    st.error(f"Error solving counterfactual: {e}")
            
            # Show counterfactual graph if solution exists
            if st.session_state.get('cf_solution', None) is not None:
                st.markdown("---")
                network_engine.create_counterfactual_graph_ui(st.session_state['cf_solution'])
        
    elif viz_mode == "Side-by-Side Comparison":
        st.markdown("---")
        
        # Configuration section
        st.header("Counterfactual Configuration")
        cf_config = create_counterfactual_ui("", "Counterfactual")
        
        if all(x is not None for x in cf_config):
            if st.button("Run Counterfactual Analysis", key="run_cf_comparison"):
                importers, exporters, sectors, tariff_rate = cf_config
                try:
                    with st.spinner("Solving Counterfactual Model..."):
                        pipeline = get_model_pipeline()
                        scenario_key = pipeline.solve_counterfactual(importers, exporters, sectors, tariff_rate)
                        sol, _ = pipeline.get_counterfactual_results(scenario_key)
                        st.session_state['cf_solution'] = sol
                        st.success("Counterfactual analysis completed!")
                except Exception as e:
                    st.error(f"Error solving counterfactual: {e}")
            
            # Show comparison if solution exists
            if st.session_state.get('cf_solution', None) is not None:
                st.markdown("---")
                network_engine.create_comparison_view(st.session_state['cf_solution'])
    
    # Information sidebar
    with st.sidebar:
        st.header("Network Graph Guide")
        st.markdown("""
        ### Node Types
        - **Circles**: Sector nodes (Country-Sector pairs)
        - **Diamonds**: Country nodes (aggregated totals)
        
        ### Node Sizes
        - Size proportional to expenditure magnitude
        - Use the scale slider to adjust visibility
        
        ### Color Coding (Counterfactual)
        - **🔴 Red**: Expenditure increased
        - **🟢 Green**: Expenditure decreased  
        - **Intensity**: Darker = larger change
        
        ### Layout
        - Countries arranged in a circle
        - Sectors clustered around their country
        - Country labels shown on diamond nodes
        """)
        
        # Clear solutions button
        if st.button("Clear All Solutions"):
            st.session_state['baseline_solution'] = None
            st.session_state['cf_solution'] = None
            st.rerun()

if __name__ == "__main__":
    main() 