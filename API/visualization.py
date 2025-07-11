"""
Simplified Visualization Module
==============================

Core functions for economic model visualization:
1. visualize_single_model() - display one model's results
2. visualize_comparison() - display percentage changes between two models

No unnecessary classes or complex abstractions.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Tuple
from models import ModelSol, ModelParams
import hashlib
from plotly.subplots import make_subplots


# Global caches for performance optimization
_cached_single_model_data = {}
_cached_percentage_data = {}


@st.cache_data
def _calculate_percentage_change(baseline_data: np.ndarray, cf_data: np.ndarray) -> np.ndarray:
    """Calculate percentage change with caching."""
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_change = ((cf_data - baseline_data) / baseline_data) * 100
        percentage_change = np.where(np.isfinite(percentage_change), percentage_change, 0)
    return percentage_change


def _get_solution_hash(solution: ModelSol) -> str:
    """Generate a unique hash for a solution using object identity and basic attributes."""
    # Use object id as primary identifier (unique per object instance)
    obj_id = id(solution)
    
    # Add some basic attributes for additional uniqueness
    # Use array shapes and a few values instead of full arrays
    hash_components = [str(obj_id)]
    
    for attr_name in dir(solution):
        if not attr_name.startswith('_') and hasattr(solution, attr_name):
            attr_value = getattr(solution, attr_name)
            if isinstance(attr_value, np.ndarray):
                # Use shape and hash of first/last few elements instead of full array
                if attr_value.size > 0:
                    shape_str = str(attr_value.shape)
                    # Use a small sample of the array for hashing
                    flat_array = attr_value.flatten()
                    if len(flat_array) > 10:
                        sample = np.concatenate([flat_array[:5], flat_array[-5:]])
                    else:
                        sample = flat_array
                    sample_hash = hashlib.md5(sample.tobytes()).hexdigest()[:8]
                    hash_components.append(f"{attr_name}:{shape_str}:{sample_hash}")
                else:
                    hash_components.append(f"{attr_name}:empty")
            elif isinstance(attr_value, (int, float, str)):
                hash_components.append(f"{attr_name}:{attr_value}")
    
    combined_str = "|".join(hash_components)
    return hashlib.md5(combined_str.encode()).hexdigest()[:16]


@st.cache_data
def _extract_model_data(_solution_hash: str, sol_dict: dict) -> dict:
    """Extract and cache all variable data from a model solution."""
    return {var_name: var_value for var_name, var_value in sol_dict.items()
            if isinstance(var_value, np.ndarray)}


@st.cache_data 
def _compute_percentage_changes(_baseline_hash: str, _cf_hash: str, 
                               baseline_dict: dict, cf_dict: dict) -> dict:
    """Pre-compute percentage changes for all variables between two models."""
    variable_keys = set(baseline_dict.keys()) & set(cf_dict.keys())
    percentage_data = {}
    
    for var_name in variable_keys:
        if isinstance(baseline_dict[var_name], np.ndarray) and isinstance(cf_dict[var_name], np.ndarray):
            percentage_data[var_name] = _calculate_percentage_change(
                baseline_dict[var_name], cf_dict[var_name]
            )
    return percentage_data


def _get_variable_description(variable_name: str) -> str:
    """Get description for a variable."""
    descriptions = {
        "w_hat": r"""$\hat{w}$: shape $(N,)$, index ($i$)<br>Percentage change in nominal wage in country $i$.""",
        "c_hat": r"""$\hat{c}$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in the unit cost of input bundles for producing output in country $i$, sector $s$.""",
        "Pf_hat": r"""$\hat{P}_f$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of final goods in country $i$, sector $s$.""",
        "Pm_hat": r"""$\hat{P}_m$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of intermediate goods in country $i$, sector $s$.""",
        "pif_hat": r"""$\hat{\pi}_f$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used for final consumption.""",
        "pim_hat": r"""$\hat{\pi}_m$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used as intermediate inputs.""",
        "real_w_hat": r"""real_w_hat: shape $(N,)$, index ($i$)<br>Percentage change in real wage in country $i$ (i.e., nominal wage deflated by the price index).""",
        "D_prime": r"""$D'$: shape $(N,)$, index ($i$)<br>Trade deficit or surplus in country $i$ under model 2 (the counterfactual scenario).""",
        "I_prime": r"""$I'$: shape $(N,)$, index ($i$)<br>Total income in country $i$ under the counterfactual scenario (includes wage income and tariff revenue).""",
    }
    return descriptions.get(variable_name, f"Variable: {variable_name}")


def _create_multiselect_with_buttons(label: str, options: List[str], session_key: str) -> List[str]:
    """Create a multiselect with Select All and Clear All buttons."""
    # Initialize session state
    if session_key not in st.session_state:
        st.session_state[session_key] = []
    
    cols = st.columns(2)
    with cols[0]:
        if st.button(f"Select ALL {label}", key=f"select_all_{session_key}"):
            st.session_state[session_key] = options.copy()
            st.rerun()
    with cols[1]:
        if st.button(f"Clear ALL {label}", key=f"clear_all_{session_key}"):
            st.session_state[session_key] = []
            st.rerun()
    
    return st.multiselect(label, options, default=st.session_state[session_key], key=session_key)


def _create_bar_chart(x_data: List[str], y_data: List[float], title: str, 
                     x_label: str, y_label: str, fig_width: int, fig_height: int):
    """Create a standard bar chart."""
    fig = px.bar(
        x=x_data, y=y_data,
        labels={'x': x_label, 'y': y_label},
        title=title, height=fig_height, width=fig_width
    )
    fig.update_traces(hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y:.6f}}')
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title_font=dict(size=20, color='black'),
        yaxis_title_font=dict(size=20, color='black')
    )
    return fig


def _visualize_1d_data(value: np.ndarray, variable_name: str, country_names: List[str], 
                      sector_names: List[str], is_percentage_change: bool = False):
    """Visualize 1D variables (country or sector level)."""
    if value.shape[0] == len(country_names):
        names = country_names
        label_type = "Countries"
        session_key = "viz_countries"
    else:
        names = sector_names
        label_type = "Sectors"
        session_key = "viz_sectors"
    
    selected_items = _create_multiselect_with_buttons(label_type, names, session_key)
    
    # Figure size controls - USE CONSISTENT KEYS
    st.markdown("### Figure Size")
    col1, col2 = st.columns(2)
    with col1:
        fig_width = st.slider("Width", 400, 2000, 1200, 100, key="fig_width_1d")
    with col2:
        fig_height = st.slider("Height", 300, 1000, 600, 50, key="fig_height_1d")
    
    if selected_items:
        # Prepare data
        labels, bars = [], []
        for name in selected_items:
            if name in names:
                idx = names.index(name)
                labels.append(name)
                bars.append(value[idx])
        
        if bars:
            y_label = f"{variable_name} (% Change)" if is_percentage_change else variable_name
            fig = _create_bar_chart(labels, bars, "Selected Values", label_type, y_label, fig_width, fig_height)
            st.plotly_chart(fig, use_container_width=False)
    else:
        st.info(f"No {label_type.lower()} selected.")


def _visualize_2d_data(value: np.ndarray, variable_name: str, country_names: List[str], 
                      sector_names: List[str], is_percentage_change: bool = False):
    """Visualize 2D variables (country-sector level)."""
    selected_countries = _create_multiselect_with_buttons("Countries", country_names, "viz_countries_2d")
    selected_sectors = _create_multiselect_with_buttons("Sectors", sector_names, "viz_sectors_2d")
    
    # Figure size controls - USE CONSISTENT KEYS
    st.markdown("### Figure Size")
    col1, col2 = st.columns(2)
    with col1:
        fig_width = st.slider("Width", 400, 2000, 1200, 100, key="fig_width_2d")
    with col2:
        fig_height = st.slider("Height", 300, 1000, 600, 50, key="fig_height_2d")
    
    if selected_countries and selected_sectors:
        y_label = f"{variable_name} (% Change)" if is_percentage_change else variable_name
        
        for country in selected_countries:
            if country in country_names:
                c_idx = country_names.index(country)
                bars, labels = [], []
                for sector in selected_sectors:
                    if sector in sector_names:
                        s_idx = sector_names.index(sector)
                        bars.append(value[c_idx, s_idx])
                        labels.append(sector)
                
                if bars:
                    fig = _create_bar_chart(labels, bars, f"{country}: Selected Sectors", "Sector", y_label, fig_width, fig_height)
                    st.plotly_chart(fig, use_container_width=False)
    else:
        st.info("Select countries and sectors to visualize.")


def _visualize_3d_data(value: np.ndarray, variable_name: str, country_names: List[str], 
                      sector_names: List[str], is_percentage_change: bool = False):
    """Visualize 3D variables (importer-exporter-sector level)."""
    selected_importers = _create_multiselect_with_buttons("Importers", country_names, "viz_importers_3d")
    selected_exporters = _create_multiselect_with_buttons("Exporters", country_names, "viz_exporters_3d")
    selected_sectors = _create_multiselect_with_buttons("Sectors", sector_names, "viz_sectors_3d")
    
    # Figure size controls - USE CONSISTENT KEYS
    st.markdown("### Figure Size")
    col1, col2 = st.columns(2)
    with col1:
        fig_width = st.slider("Width", 400, 2000, 1400, 100, key="fig_width_3d")
    with col2:
        fig_height = st.slider("Height", 400, 1200, 800, 50, key="fig_height_3d")
    
    if selected_importers and selected_exporters and selected_sectors:
        # Limit combinations for performance
        max_combinations = 16
        total_combinations = len(selected_importers) * len(selected_exporters)
        
        if total_combinations > max_combinations:
            st.warning(f"⚠️ Too many combinations ({total_combinations}). Limiting to first {max_combinations} for performance.")
            limit = int(np.sqrt(max_combinations))
            selected_importers = selected_importers[:limit]
            selected_exporters = selected_exporters[:limit]
        
        # Calculate subplot layout
        rows = cols = int(np.ceil(np.sqrt(len(selected_importers) * len(selected_exporters))))
        
        # Create subplots
        subplot_titles = [f"{imp}→{exp}" for imp in selected_importers for exp in selected_exporters]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, 
                           vertical_spacing=0.15, horizontal_spacing=0.1)
        
        plot_idx = 0
        for importer in selected_importers:
            for exporter in selected_exporters:
                if plot_idx >= rows * cols:
                    break
                
                if importer in country_names and exporter in country_names:
                    i_idx = country_names.index(importer)
                    e_idx = country_names.index(exporter)
                    
                    # Prepare data for this pair
                    y_values, x_labels = [], []
                    for sector in selected_sectors:
                        if sector in sector_names:
                            s_idx = sector_names.index(sector)
                            y_values.append(value[i_idx, e_idx, s_idx])
                            x_labels.append(sector)
                    
                    # Add bar trace
                    row = (plot_idx // cols) + 1
                    col = (plot_idx % cols) + 1
                    
                    y_label = f"{variable_name} (% Change)" if is_percentage_change else variable_name
                    fig.add_trace(
                        go.Bar(x=x_labels, y=y_values, name=f"{importer}→{exporter}", showlegend=False,
                               hovertemplate=f"<b>{importer}→{exporter}</b><br>Sector: %{{x}}<br>{y_label}: %{{y:.6f}}<extra></extra>"),
                        row=row, col=col
                    )
                
                plot_idx += 1
        
        fig.update_layout(height=max(600, rows * 200), width=fig_width, showlegend=False,
                         title_text=f"{variable_name}: Trade Relationships by Sector", title_x=0.5)
        fig.update_xaxes(tickangle=-45, tickfont_size=10)
        fig.update_yaxes(tickfont_size=10)
        
        st.plotly_chart(fig, use_container_width=False)
    else:
        st.info("Select importers, exporters, and sectors to visualize.")


def visualize_single_model(solution: ModelSol, country_names: List[str], sector_names: List[str], pre_computed_hash: Optional[str] = None):
    """
    Visualize results from a single model.
    
    Args:
        solution: Model solution to visualize
        country_names: List of country names
        sector_names: List of sector names
        pre_computed_hash: Optional pre-computed hash to use instead of generating new one
    """
    st.header("Variables and Visualization")
    
    # Use cached data extraction for performance
    sol_hash = pre_computed_hash if pre_computed_hash else _get_solution_hash(solution)
    if sol_hash not in _cached_single_model_data:
        _cached_single_model_data[sol_hash] = _extract_model_data(sol_hash, solution.__dict__)
    
    cached_data = _cached_single_model_data[sol_hash]
    variable_keys = list(cached_data.keys())
    
    # Variable selection
    variable = st.selectbox("Choose an output variable", variable_keys)
    
    if variable:
        # Show variable description
        description = _get_variable_description(variable)
        if description:
            st.markdown(description, unsafe_allow_html=True)
        
        # Get variable data
        value = cached_data[variable]
        st.write(f"Variable shape: {np.shape(value)}")
        
        # Visualize based on dimensions
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                _visualize_1d_data(value, variable, country_names, sector_names)
            elif value.ndim == 2:
                if variable == 'country_links':
                    st.info("🔗 **Country Links** is available for Excel download but not for interactive visualization.")
                else:
                    _visualize_2d_data(value, variable, country_names, sector_names)
            elif value.ndim == 3:
                _visualize_3d_data(value, variable, country_names, sector_names)
            elif value.ndim == 4:
                if variable == 'sector_links':
                    st.info("🔗 **Sector Links** is available for Excel download but not for interactive visualization.")
                else:
                    st.write(f"4D Variable: {variable}, Shape: {value.shape}")
                    st.info("This 4D variable requires specialized visualization not yet implemented.")
            elif value.ndim == 0:
                st.write(f"Value: **{value.item():.4f}**")
            else:
                st.write("Value:")
                st.write(value)


def visualize_comparison(sol1: ModelSol, sol2: ModelSol, country_names: List[str], sector_names: List[str], 
                        baseline_hash: Optional[str] = None, cf_hash: Optional[str] = None):
    """
    Visualize comparison between two models (percentage changes).
    
    Args:
        sol1: Baseline model solution
        sol2: Counterfactual model solution
        country_names: List of country names
        sector_names: List of sector names
        baseline_hash: Optional pre-computed hash for baseline model
        cf_hash: Optional pre-computed hash for counterfactual model
    """
    st.header("Variables and Visualization")
    
    # Use pre-computed percentage changes for performance
    baseline_hash = baseline_hash if baseline_hash else _get_solution_hash(sol1)
    cf_hash = cf_hash if cf_hash else _get_solution_hash(sol2)
    
    cache_key = f"{baseline_hash}_{cf_hash}"
    if cache_key not in _cached_percentage_data:
        _cached_percentage_data[cache_key] = _compute_percentage_changes(
            baseline_hash, cf_hash, sol1.__dict__, sol2.__dict__
        )
    
    percentage_data = _cached_percentage_data[cache_key]
    variable_keys = list(percentage_data.keys())
    
    # Variable selection
    variable = st.selectbox("Choose an output variable", variable_keys)
    
    if variable:
        # Show variable description
        description = _get_variable_description(variable)
        if description:
            st.markdown(description, unsafe_allow_html=True)
        
        # Get percentage change data
        value = percentage_data[variable]
        st.write(f"Variable shape: {np.shape(value)} (showing % change from Baseline to Counterfactual)")
        
        # Visualize based on dimensions
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                _visualize_1d_data(value, variable, country_names, sector_names, is_percentage_change=True)
            elif value.ndim == 2:
                if variable == 'country_links':
                    st.info("🔗 **Country Links** is available for Excel download but not for interactive visualization.")
                else:
                    _visualize_2d_data(value, variable, country_names, sector_names, is_percentage_change=True)
            elif value.ndim == 3:
                _visualize_3d_data(value, variable, country_names, sector_names, is_percentage_change=True)
            else:
                st.write(f"Variable: {variable}, Shape: {value.shape}")
                st.info("This variable requires specialized visualization not yet implemented.") 