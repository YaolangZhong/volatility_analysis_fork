"""
Visualization Module: Separated plotting and data preparation
===========================================================

This module handles all visualization logic for model results,
working with the ModelPipeline to display benchmark and counterfactual
model solutions.

This separation allows for easy extension to different visualization types
(traditional plots, network graphs, etc.) while reusing the same model
solving logic.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Union, Tuple
from models import ModelSol, ModelParams
import pandas as pd
from io import BytesIO
import hashlib
from plotly.subplots import make_subplots

@st.cache_data
def calculate_percentage_change(baseline_data: np.ndarray, cf_data: np.ndarray) -> np.ndarray:
    """
    Calculate percentage change with caching to avoid redundant calculations.
    
    Args:
        baseline_data: Baseline model data
        cf_data: Counterfactual model data
        
    Returns:
        Percentage change array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_change = ((cf_data - baseline_data) / baseline_data) * 100
        # Handle division by zero cases
        percentage_change = np.where(np.isfinite(percentage_change), percentage_change, 0)
    return percentage_change

@st.cache_data  
def generate_data_hash(data: np.ndarray) -> str:
    """Generate a hash for data arrays to enable efficient caching."""
    return hashlib.md5(data.tobytes()).hexdigest()

class VisualizationDataProcessor:
    """Processes model solution data for visualization."""
    
    def __init__(self, country_names: List[str], sector_names: List[str]):
        self.country_names = country_names
        self.sector_names = sector_names
        
        # Priority order for UI display
        self.priority_countries = [
            "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "ITA", "CAN", "KOR", "IND",
            "ESP", "NLD", "BEL", "SWE", "RUS", "BRA", "MEX", "AUS"
        ]
        self.country_names_sorted = (
            [c for c in self.priority_countries if c in country_names] + 
            [c for c in country_names if c not in self.priority_countries]
        )
    
    def get_variable_description(self, variable_name: str) -> str:
        """Get description for a variable."""
        descriptions = {
            "w_hat": r"""$\hat{w}$: shape $(N,)$, index ($i$)<br>Percentage change in nominal wage in country $i$.""",
            "c_hat": r"""$\hat{c}$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in the unit cost of input bundles for producing output in country $i$, sector $s$.""",
            "Pf_hat": r"""$\hat{P}_f$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of final goods in country $i$, sector $s$.""",
            "Pm_hat": r"""$\hat{P}_m$: shape $(N, S)$, indices ($i$, $s$)<br>Percentage change in prices of intermediate goods in country $i$, sector $s$.""",
            "pif_hat": r"""$\hat{\pi}_f$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used for final consumption.""",
            "pim_hat": r"""$\hat{\pi}_m$: shape $(N, N, S)$, indices ($n$, $i$, $s$)<br>Percentage change in expenditure shares by importer $n$ on goods from exporter $i$ used in producing goods in sector $s$, which are then used as intermediate inputs.""",
            "Xf_prime": r"""$X_f'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods used in producing output in sector $s$, which are then used for final consumption, under model 2.""",
            "Xm_prime": r"""$X_m'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods used in producing output in sector $s$, which are then used as intermediate inputs, under model 2.""",
            "X_prime": r"""$X'$: shape $(N, S)$, indices ($n$, $s$)<br>Total expenditure by country $n$ on goods in sector $s$ under model 2, i.e., the sum of $X_f'$ and $X_m'$.""",
            "Xf_prod_prime": r"""$X_{f,\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Production by country $n$ of goods in sector $s$ used for final consumption, under model 2.""",
            "Xm_prod_prime": r"""$X_{m,\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Production by country $n$ of goods in sector $s$ used as intermediate inputs, under model 2.""",
            "X_prod_prime": r"""$X_{\text{prod}}'$: shape $(N, S)$, indices ($n$, $s$)<br>Total production by country $n$ of goods in sector $s$ under model 2, i.e., the sum of $X_{f,\text{prod}}'$ and $X_{m,\text{prod}}'$.""",
            "p_index": r"""$p_{\text{index}}$: shape $(N,)$, index ($i$)<br>Change in the aggregate price index in country $i$.""",
            "real_w_hat": r"""real_w_hat: shape $(N,)$, index ($i$)<br>Percentage change in real wage in country $i$ (i.e., nominal wage deflated by the price index).""",
            "D_prime": r"""$D'$: shape $(N,)$, index ($i$)<br>Trade deficit or surplus in country $i$ under model 2 (the counterfactual scenario).""",
            "I_prime": r"""$I'$: shape $(N,)$, index ($i$)<br>Total income in country $i$ under the counterfactual scenario (includes wage income and tariff revenue).""",
            "output_prime": r"""$\text{Output}'$: shape $(N, S)$, indices ($i$, $s$)<br>Output demand in country $i$, sector $s$ under the counterfactual scenario (intermediate variable from expenditure calculation).""",
            "real_I_prime": r"""$\text{Real I}'$: shape $(N,)$, index ($i$)<br>Real income in country $i$ under the counterfactual scenario (nominal income deflated by the price index).""",
            "sector_links": r"""$\text{Sector Links}$: shape $(N, S, N, S)$, indices ($i$, $k$, $n$, $s$)<br>Import linkages where sector $k$ of country $i$ imports from sector $s$ in country $n$.""",
            "country_links": r"""$\text{Country Links}$: shape $(N, N)$, indices ($i$, $n$)<br>Country-level import linkages where country $i$ imports from country $n$ (sum of all sector-level linkages)."""
        }
        return descriptions.get(variable_name, f"Variable: {variable_name}")


class VisualizationUI:
    """Handles Streamlit UI components for visualization."""
    
    def __init__(self, data_processor: VisualizationDataProcessor):
        self.data_processor = data_processor
        # Initialize persistent selection state
        self._initialize_persistent_state()
    
    def _initialize_persistent_state(self):
        """Initialize persistent selection state that survives variable changes."""
        if "viz_selected_countries" not in st.session_state:
            st.session_state["viz_selected_countries"] = []
        if "viz_selected_sectors" not in st.session_state:
            st.session_state["viz_selected_sectors"] = self.data_processor.sector_names.copy()
        if "viz_selected_importers" not in st.session_state:
            st.session_state["viz_selected_importers"] = []
        if "viz_selected_exporters" not in st.session_state:
            st.session_state["viz_selected_exporters"] = []
        if "viz_fig_width" not in st.session_state:
            st.session_state["viz_fig_width"] = 1600
        if "viz_fig_height" not in st.session_state:
            st.session_state["viz_fig_height"] = 700
    
    def create_multiselect_with_buttons(self, 
                                      label: str, 
                                      options: List[str], 
                                      default: List[str], 
                                      key: str,
                                      all_button_key: str,
                                      clear_button_key: str) -> List[str]:
        """Create a multiselect with Select All and Clear All buttons."""
        cols = st.columns(2)
        with cols[0]:
            if st.button(f"Select ALL {label}", key=all_button_key):
                st.session_state[key] = options.copy()
                st.rerun()
        with cols[1]:
            if st.button(f"Remove ALL {label}", key=clear_button_key):
                st.session_state[key] = []
                st.rerun()
        
        return st.multiselect(label, options, default=st.session_state.get(key, default), key=key)
    
    def create_stable_country_selector(self, label: str = "Countries") -> List[str]:
        """Create country selector with stable keys that persist across variable changes."""
        return self.create_multiselect_with_buttons(
            label, 
            self.data_processor.country_names_sorted, 
            st.session_state["viz_selected_countries"],
            "viz_selected_countries",
            "viz_select_all_countries",
            "viz_clear_all_countries"
        )
    
    def create_stable_sector_selector(self, label: str = "Sectors") -> List[str]:
        """Create sector selector with stable keys that persist across variable changes."""
        return self.create_multiselect_with_buttons(
            label,
            self.data_processor.sector_names,
            st.session_state["viz_selected_sectors"],
            "viz_selected_sectors", 
            "viz_select_all_sectors",
            "viz_clear_all_sectors"
        )
    
    def create_stable_importer_selector(self) -> List[str]:
        """Create importer selector with stable keys."""
        return self.create_multiselect_with_buttons(
            "Importer Countries",
            self.data_processor.country_names_sorted,
            st.session_state["viz_selected_importers"],
            "viz_selected_importers",
            "viz_select_all_importers", 
            "viz_clear_all_importers"
        )
    
    def create_stable_exporter_selector(self) -> List[str]:
        """Create exporter selector with stable keys."""
        return self.create_multiselect_with_buttons(
            "Exporter Countries",
            self.data_processor.country_names_sorted,
            st.session_state["viz_selected_exporters"],
            "viz_selected_exporters",
            "viz_select_all_exporters",
            "viz_clear_all_exporters"
        )

    def create_figure_size_controls(self) -> Tuple[int, int]:
        """Create figure size control sliders with persistent state."""
        st.markdown("### Figure Size Adjustment")
        fig_width = st.slider(
            "Figure Width", 
            min_value=400, max_value=2000, 
            value=st.session_state["viz_fig_width"], 
            step=100,
            key="viz_fig_width_slider"
        )
        fig_height = st.slider(
            "Figure Height", 
            min_value=300, max_value=1000, 
            value=st.session_state["viz_fig_height"], 
            step=50,
            key="viz_fig_height_slider"
        )
        
        # Update session state
        st.session_state["viz_fig_width"] = fig_width
        st.session_state["viz_fig_height"] = fig_height
        
        return fig_width, fig_height


class PlotlyVisualizer:
    """Creates Plotly visualizations for model results."""
    
    def __init__(self, data_processor: VisualizationDataProcessor):
        self.data_processor = data_processor
    
    @st.cache_data
    def _prepare_1d_plot_data(_self, value: np.ndarray, names: List[str], 
                             selected_items: List[str]) -> Tuple[List[str], List[float]]:
        """Pre-compute and cache 1D plot data preparation."""
        selected_items_in_order = [name for name in names if name in selected_items]
        
        bars = []
        labels = []
        for name in selected_items_in_order:
            idx = names.index(name)
            bars.append(value[idx])
            labels.append(name)
        
        return labels, bars
    
    def create_bar_chart(self, 
                        x_data: List[str], 
                        y_data: List[float], 
                        title: str, 
                        x_label: str, 
                        y_label: str,
                        fig_width: int = 1600,
                        fig_height: int = 700):
        """Create a standard bar chart."""
        fig = px.bar(
            x=x_data,
            y=y_data,
            labels={'x': x_label, 'y': y_label},
            title=title,
            height=fig_height,
            width=fig_width
        )
        fig.update_traces(hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y:.6f}}')
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title_font=dict(size=20, color='black'),
            yaxis_title_font=dict(size=20, color='black')
        )
        return fig
    
    def visualize_1d_variable(self, 
                             value: np.ndarray, 
                             variable_name: str, 
                             selected_items: List[str],
                             fig_width: int, 
                             fig_height: int,
                             is_percentage_change: bool = False):
        """Visualize 1D variables with optimized data preparation."""
        
        if value.shape[0] == len(self.data_processor.country_names):
            names = self.data_processor.country_names
            label_type = "Countries"
        else:
            names = self.data_processor.sector_names
            label_type = "Sectors"
        
        # NEW: Use cached data preparation
        labels, bars = self._prepare_1d_plot_data(value, names, selected_items)
        
        if bars:
            if is_percentage_change:
                y_label = f"{variable_name} (% Change)"
            else:
                y_label = f"{variable_name} (% Change)" if variable_name.endswith("_hat") else variable_name
            
            fig = self.create_bar_chart(
                labels, bars, "Selected Values", label_type, y_label, fig_width, fig_height
            )
            st.plotly_chart(fig, use_container_width=False)
        else:
            st.info(f"No {label_type.lower()} selected.")
    
    def visualize_2d_variable(self, 
                             value: np.ndarray, 
                             variable_name: str, 
                             selected_countries: List[str],
                             selected_sectors: List[str],
                             fig_width: int, 
                             fig_height: int,
                             is_percentage_change: bool = False):
        """Visualize 2D variables (country-sector level)."""
        selected_countries_in_order = [c for c in self.data_processor.country_names if c in selected_countries]
        
        if is_percentage_change:
            y_label = f"{variable_name} (% Change)"
        else:
            y_label = f"{variable_name} (% Change)" if variable_name.endswith("_hat") else variable_name
        
        for country in selected_countries_in_order:
            c_idx = self.data_processor.country_names.index(country)
            bars = []
            labels = []
            for sector in selected_sectors:
                s_idx = self.data_processor.sector_names.index(sector)
                val = value[c_idx, s_idx]
                bars.append(val)
                labels.append(sector)
            
            fig = self.create_bar_chart(
                labels, bars, f"{country}: Selected Sectors", "Sector", y_label, fig_width, fig_height
            )
            st.plotly_chart(fig, use_container_width=False)
    
    def visualize_3d_variable(self, 
                             value: np.ndarray, 
                             variable_name: str, 
                             selected_importers: List[str],
                             selected_exporters: List[str], 
                             selected_sectors: List[str],
                             fig_width: int, 
                             fig_height: int,
                             is_percentage_change: bool = False):
        """Visualize 3D variables efficiently using subplot grid instead of separate plots."""
        selected_importers_in_order = [c for c in self.data_processor.country_names if c in selected_importers]
        selected_exporters_in_order = [c for c in self.data_processor.country_names if c in selected_exporters]
        
        if is_percentage_change:
            y_label = f"{variable_name} (% Change)"
        else:
            y_label = f"{variable_name} (% Change)" if variable_name.endswith("_hat") else variable_name
        
        num_importers = len(selected_importers_in_order)
        num_exporters = len(selected_exporters_in_order)
        
        # Limit subplot grid size for performance
        max_plots = 16  # 4x4 grid maximum
        total_combinations = num_importers * num_exporters
        
        if total_combinations > max_plots:
            st.warning(f"⚠️ Too many combinations ({total_combinations}). Showing only first {max_plots} combinations for performance. Consider selecting fewer countries.")
            # Limit to first few combinations
            combinations_limit = int(np.sqrt(max_plots))
            selected_importers_in_order = selected_importers_in_order[:combinations_limit]
            selected_exporters_in_order = selected_exporters_in_order[:combinations_limit]
            num_importers = len(selected_importers_in_order)
            num_exporters = len(selected_exporters_in_order)
            total_combinations = num_importers * num_exporters
        
        # Calculate optimal subplot layout
        if total_combinations <= 4:
            rows, cols = 2, 2
        elif total_combinations <= 9:
            rows, cols = 3, 3
        else:
            rows = cols = int(np.ceil(np.sqrt(total_combinations)))
        
        # Create subplot grid
        subplot_titles = []
        for importer in selected_importers_in_order:
            for exporter in selected_exporters_in_order:
                subplot_titles.append(f"{importer}→{exporter}")
        
        # Pad titles if needed
        while len(subplot_titles) < rows * cols:
            subplot_titles.append("")
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles[:rows * cols],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        plot_idx = 0
        for importer in selected_importers_in_order:
            for exporter in selected_exporters_in_order:
                if plot_idx >= rows * cols:
                    break
                    
                i_idx = self.data_processor.country_names.index(importer)
                e_idx = self.data_processor.country_names.index(exporter)
                
                # Prepare data for this importer-exporter pair
                y_values = []
                x_labels = []
                for sector in selected_sectors:
                    s_idx = self.data_processor.sector_names.index(sector)
                    y_values.append(value[i_idx, e_idx, s_idx])
                    x_labels.append(sector)
                
                # Calculate subplot position
                row = (plot_idx // cols) + 1
                col = (plot_idx % cols) + 1
                
                # Add bar trace to subplot
                fig.add_trace(
                    go.Bar(
                        x=x_labels,
                        y=y_values,
                        name=f"{importer}→{exporter}",
                        showlegend=False,
                        hovertemplate=f"<b>{importer}→{exporter}</b><br>Sector: %{{x}}<br>{y_label}: %{{y:.6f}}<extra></extra>"
                    ),
                    row=row, col=col
                )
                
                plot_idx += 1
        
        # Update layout for better appearance
        fig.update_layout(
            height=max(600, rows * 200),  # Dynamic height based on rows
            width=fig_width,
            title_text=f"{variable_name}: Trade Relationships by Sector",
            title_x=0.5,
            showlegend=False
        )
        
        # Update x-axes to show sector labels at angle for readability
        fig.update_xaxes(tickangle=-45, tickfont_size=10)
        fig.update_yaxes(title_text=y_label, tickfont_size=10)
        
        st.plotly_chart(fig, use_container_width=False)


class ModelVisualizationEngine:
    """Main engine for model visualization that coordinates all components."""
    
    def __init__(self, country_names: List[str], sector_names: List[str]):
        self.data_processor = VisualizationDataProcessor(country_names, sector_names)
        self.ui = VisualizationUI(self.data_processor)
        self.visualizer = PlotlyVisualizer(self.data_processor)
        
        # ADD NEW: Caching infrastructure
        self._cached_single_model_data = {}
        self._cached_percentage_data = {}
        self._cached_sol_hashes = {}
    
    def _get_solution_hash(self, solution) -> str:
        """Generate a unique hash for a solution to enable caching."""
        solution_str = str(solution.__dict__)
        return hashlib.md5(solution_str.encode()).hexdigest()[:16]
    
    @st.cache_data
    def _precompute_single_model_data(_self, solution_hash: str, sol_dict: dict) -> dict:
        """Pre-extract and cache all variable data from a single model solution."""
        return {var_name: var_value for var_name, var_value in sol_dict.items()
                if isinstance(var_value, np.ndarray)}
    
    @st.cache_data 
    def _precompute_percentage_changes(_self, baseline_hash: str, cf_hash: str, 
                                     baseline_dict: dict, cf_dict: dict) -> dict:
        """Pre-compute percentage changes for ALL variables between two models."""
        variable_keys = set(baseline_dict.keys()) & set(cf_dict.keys())
        percentage_data = {}
        
        for var_name in variable_keys:
            if isinstance(baseline_dict[var_name], np.ndarray) and isinstance(cf_dict[var_name], np.ndarray):
                percentage_data[var_name] = calculate_percentage_change(
                    baseline_dict[var_name], cf_dict[var_name]
                )
        return percentage_data
    
    def display_variable_description(self, variable_name: str):
        """Display variable description if available."""
        description = self.data_processor.get_variable_description(variable_name)
        if description:
            st.markdown(description, unsafe_allow_html=True)
    
    def visualize_single_model(self, solution: ModelSol):
        """Visualize results from a single model with optimized data caching."""
        st.header("Variables and Visualization")
        
        # NEW: Use cached data extraction
        sol_hash = self._get_solution_hash(solution)
        if sol_hash not in self._cached_single_model_data:
            self._cached_single_model_data[sol_hash] = self._precompute_single_model_data(
                sol_hash, solution.__dict__
            )
        
        cached_data = self._cached_single_model_data[sol_hash]
        variable_keys = list(cached_data.keys())
        
        # Variable selection
        variable = st.selectbox("Choose an output variable", variable_keys)
        
        if variable:
            self.display_variable_description(variable)
            
            # NEW: Direct lookup instead of attribute access
            value = cached_data[variable]
            st.write(f"Variable shape: {np.shape(value)}")
            
            self._visualize_variable(value, variable)
    
    def visualize_comparison(self, sol1: ModelSol, sol2: ModelSol):
        """Visualize comparison between two models."""
        st.header("Variables and Visualization")
        
        # NEW: Use pre-computed percentage changes instead of on-demand calculation
        baseline_hash = self._get_solution_hash(sol1)  
        cf_hash = self._get_solution_hash(sol2)
        
        # Get pre-computed percentage data (cached)
        cache_key = f"{baseline_hash}_{cf_hash}"
        if cache_key not in self._cached_percentage_data:
            self._cached_percentage_data[cache_key] = self._precompute_percentage_changes(
                baseline_hash, cf_hash, sol1.__dict__, sol2.__dict__
            )
        
        percentage_data = self._cached_percentage_data[cache_key]
        variable_keys = list(percentage_data.keys())
        
        # Variable selection (same as before)
        variable = st.selectbox("Choose an output variable", variable_keys)
        
        if variable:
            self.display_variable_description(variable)
            
            # NEW: Direct lookup instead of calculation
            value = percentage_data[variable]
            st.write(f"Variable shape: {np.shape(value)} (showing % change from Baseline to Counterfactual)")
            
            self._visualize_variable(value, variable, is_percentage_change=True)
    
    def _visualize_variable(self, value: np.ndarray, variable_name: str, is_percentage_change: bool = False):
        """Internal method to handle visualization based on variable dimensions."""
        if isinstance(value, np.ndarray) and value.ndim == 1:
            self._visualize_1d_variable(value, variable_name, is_percentage_change)
        elif isinstance(value, np.ndarray) and value.ndim == 2:
            if variable_name == 'country_links':
                # country_links should not be visualized, only available for download
                st.info("🔗 **Country Links** is available for Excel download but not for interactive visualization.")
                st.markdown("**To access country_links data:**")
                st.markdown("- Use the Excel download button to get the country-country matrix")
                st.markdown("- Rows and columns represent countries") 
                st.markdown("- Values show import linkages between countries")
            else:
                self._visualize_2d_variable(value, variable_name, is_percentage_change)
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            self._visualize_3d_variable(value, variable_name, is_percentage_change)
        elif isinstance(value, np.ndarray) and value.ndim == 4:
            # sector_links should not be visualized, only available for download
            if variable_name == 'sector_links':
                st.info("🔗 **Sector Links** is available for Excel download but not for interactive visualization.")
                st.markdown("**To access sector_links data:**")
                st.markdown("- Use the Excel download button to get the flattened country-sector matrix")
                st.markdown("- Rows and columns represent country-sector combinations") 
                st.markdown("- Values show import linkages between country-sector pairs")

            else:
                # Generic 4D handling - show basic statistics
                st.write(f"4D Variable: {variable_name}")
                st.write(f"Shape: {value.shape}")
                st.write(f"Min: {np.min(value):.6f}, Max: {np.max(value):.6f}, Mean: {np.mean(value):.6f}")
                st.info("This 4D variable requires specialized visualization not yet implemented.")
        elif isinstance(value, np.ndarray) and value.ndim == 0:
            st.write(f"Value: **{value.item():.4f}**")
        else:
            st.write("Value:")
            st.write(value)
    


    def _visualize_1d_variable(self, value: np.ndarray, variable_name: str, is_percentage_change: bool = False):
        """Handle 1D variable visualization with UI controls."""
        if value.shape[0] == len(self.data_processor.country_names):
            # Country-level data - use stable country selector
            selected_items = self.ui.create_stable_country_selector()
            names = self.data_processor.country_names
            label_type = "Countries"
        else:
            # Sector-level data - use stable sector selector  
            selected_items = self.ui.create_stable_sector_selector()
            names = self.data_processor.sector_names
            label_type = "Sectors"

        fig_width, fig_height = self.ui.create_figure_size_controls()
        
        if selected_items:
            self.visualizer.visualize_1d_variable(
                value, variable_name, selected_items, fig_width, fig_height, is_percentage_change
            )
        else:
            st.info(f"No {label_type.lower()} selected.")
    
    def _visualize_2d_variable(self, value: np.ndarray, variable_name: str, is_percentage_change: bool = False):
        """Handle 2D variable visualization with UI controls."""
        selected_countries = self.ui.create_stable_country_selector()
        
        selected_sectors = self.ui.create_stable_sector_selector()
        
        if selected_countries and selected_sectors:
            fig_width, fig_height = self.ui.create_figure_size_controls()
            self.visualizer.visualize_2d_variable(
                value, variable_name, selected_countries, selected_sectors, fig_width, fig_height, is_percentage_change
            )
        else:
            st.info("No countries or sectors selected.")
    
    def _visualize_3d_variable(self, value: np.ndarray, variable_name: str, is_percentage_change: bool = False):
        """Handle 3D variable visualization with UI controls."""
        selected_importers = self.ui.create_stable_importer_selector()
        
        selected_exporters = self.ui.create_stable_exporter_selector()
        
        selected_sectors = self.ui.create_stable_sector_selector()
        
        if selected_importers and selected_exporters and selected_sectors:
            fig_width, fig_height = self.ui.create_figure_size_controls()
            self.visualizer.visualize_3d_variable(
                value, variable_name, selected_importers, selected_exporters, 
                selected_sectors, fig_width, fig_height, is_percentage_change
            )
        else:
            st.info("No importers, exporters, or sectors selected for 3D variable.")


def create_excel_download_button(sol: ModelSol, params: ModelParams, scenario_key: Optional[str] = None, 
                                variable_name: Optional[str] = None, unique_key: str = "", 
                                baseline_sol: Optional[ModelSol] = None):
    """Create Excel download button for model solution data."""
    
    def generate_excel_data():
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if variable_name:
                # Download specific variable
                data = getattr(sol, variable_name)
                
                # Calculate percentage change if baseline is provided
                data_to_use = data
                if baseline_sol is not None and hasattr(baseline_sol, variable_name):
                    baseline_value = getattr(baseline_sol, variable_name)
                    # Calculate percentage change: 100 * (counterfactual - baseline) / baseline
                    data_to_use = 100 * (data - baseline_value) / (np.abs(baseline_value) + 1e-8)
                
                if variable_name == 'sector_links':
                    # Flatten sector_links for visualization
                    flattened_data, row_labels, col_labels = flatten_sector_links_for_viz(data_to_use, params)
                    df = pd.DataFrame(flattened_data, index=row_labels, columns=col_labels)
                    df.to_excel(writer, sheet_name=variable_name)
                elif variable_name == 'country_links':
                    # Flatten country_links for visualization
                    flattened_data, row_labels, col_labels = flatten_country_links_for_viz(data_to_use, params)
                    df = pd.DataFrame(flattened_data, index=row_labels, columns=col_labels)
                    df.to_excel(writer, sheet_name=variable_name)
                elif len(data_to_use.shape) == 1:
                    # 1D array - country-level data
                    df = pd.DataFrame({
                        'Country': list(params.country_list),
                        variable_name: data_to_use
                    })
                    df.to_excel(writer, sheet_name=variable_name, index=False)
                elif len(data_to_use.shape) == 2:
                    # 2D array - country x sector data
                    df = pd.DataFrame(
                        data_to_use, 
                        index=list(params.country_list),
                        columns=list(params.sector_list)
                    )
                    df.to_excel(writer, sheet_name=variable_name)
                elif len(data_to_use.shape) == 3:
                    # 3D array - handle trade data specially
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
                    df.to_excel(writer, sheet_name=variable_name, index=False)
            else:
                # Download all variables
                for var_name in dir(sol):
                    if not var_name.startswith('_') and hasattr(sol, var_name):
                        data = getattr(sol, var_name)
                        if isinstance(data, np.ndarray):
                            try:
                                # Calculate percentage change if baseline is provided
                                data_to_use = data
                                if baseline_sol is not None and hasattr(baseline_sol, var_name):
                                    baseline_value = getattr(baseline_sol, var_name)
                                    # Calculate percentage change: 100 * (counterfactual - baseline) / baseline
                                    data_to_use = 100 * (data - baseline_value) / (np.abs(baseline_value) + 1e-8)
                                
                                if var_name == 'sector_links':
                                    # Flatten sector_links for visualization
                                    flattened_data, row_labels, col_labels = flatten_sector_links_for_viz(data_to_use, params)
                                    df = pd.DataFrame(flattened_data, index=row_labels, columns=col_labels)
                                    df.to_excel(writer, sheet_name=var_name)
                                elif var_name == 'country_links':
                                    # Flatten country_links for visualization
                                    flattened_data, row_labels, col_labels = flatten_country_links_for_viz(data_to_use, params)
                                    df = pd.DataFrame(flattened_data, index=row_labels, columns=col_labels)
                                    df.to_excel(writer, sheet_name=var_name)
                                elif len(data_to_use.shape) == 1:
                                    df = pd.DataFrame({
                                        'Country': list(params.country_list),
                                        var_name: data_to_use
                                    })
                                    df.to_excel(writer, sheet_name=var_name, index=False)
                                elif len(data_to_use.shape) == 2:
                                    df = pd.DataFrame(
                                        data_to_use, 
                                        index=list(params.country_list),
                                        columns=list(params.sector_list)
                                    )
                                    df.to_excel(writer, sheet_name=var_name)
                                elif len(data_to_use.shape) == 3:
                                    N, _, S = data_to_use.shape
                                    rows = []
                                    for n in range(N):
                                        for i in range(N):
                                            for s in range(S):
                                                rows.append({
                                                    'Importer': params.country_list[n],
                                                    'Exporter': params.country_list[i], 
                                                    'Sector': params.sector_list[s],
                                                    var_name: data_to_use[n, i, s]
                                                })
                                    df = pd.DataFrame(rows)
                                    df.to_excel(writer, sheet_name=var_name, index=False)
                            except Exception as e:
                                st.warning(f"Could not export {var_name}: {e}")
        
        output.seek(0)
        return output.getvalue()
    
    if variable_name:
        filename = f"{scenario_key}_{variable_name}.xlsx" if scenario_key else f"{variable_name}.xlsx"
        button_text = f"📊 Download {variable_name} as Excel"
    else:
        filename = f"{scenario_key}_all_variables.xlsx" if scenario_key else "all_variables.xlsx"
        button_text = "📊 Download All Variables as Excel"
    
    st.download_button(
        label=button_text,
        data=generate_excel_data(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"excel_download_{unique_key}_{variable_name or 'all'}"
    )


def flatten_sector_links_for_viz(sector_links: np.ndarray, params: ModelParams) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Flatten sector_links from 4D (N, S, N, S) to 2D (NS, NS) for visualization.
    
    Args:
        sector_links: 4D array with shape (i, k, n, s) where:
        - i: importer country
        - k: output sector
        - n: exporter country  
        - s: input sector
        params: Model parameters containing country and sector lists
        
    Returns:
        Tuple of (flattened_2d_array, row_labels, col_labels)
        - Rows: ik pairs (importer_country + output_sector)
        - Columns: ns pairs (exporter_country + input_sector)
    """
    N, S, _, _ = sector_links.shape
    NS = N * S
    
    # Create labels using actual country and sector names from params
    row_labels = []  # ik pairs: importer_country + output_sector
    col_labels = []  # ns pairs: exporter_country + input_sector
    
    # Row labels: importer_country + output_sector (ik pairs)
    for i in range(N):
        for k in range(S):
            country_name = params.country_list[i] if i < len(params.country_list) else f"Country_{i}"
            sector_name = params.sector_list[k] if k < len(params.sector_list) else f"Sector_{k}"
            row_labels.append(f"{country_name}_{sector_name}")
    
    # Column labels: exporter_country + input_sector (ns pairs)
    for n in range(N): 
        for s in range(S):
            country_name = params.country_list[n] if n < len(params.country_list) else f"Country_{n}"
            sector_name = params.sector_list[s] if s < len(params.sector_list) else f"Sector_{s}"
            col_labels.append(f"{country_name}_{sector_name}")
    
    # Reshape 4D (N, S, N, S) to 2D (NS, NS)
    # sector_links[i, k, n, s] -> reshape gives (ik, ns) indexing
    # Rows: ik pairs (importer_country + output_sector)
    # Columns: ns pairs (exporter_country + input_sector)
    flattened = sector_links.reshape(NS, NS)
    
    return flattened, row_labels, col_labels


def flatten_country_links_for_viz(country_links: np.ndarray, params: ModelParams) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Format country_links for visualization as a country × country matrix.
    
    Args:
        country_links: 2D array with shape (importer_country, exporter_country)
        params: Model parameters containing country list
        
    Returns:
        Tuple of (country_links_array, row_labels, col_labels)
    """
    N, _ = country_links.shape
    
    # Create labels using actual country names from params
    row_labels = []  # Importer countries (rows)
    col_labels = []  # Exporter countries (columns)
    
    # Row labels: importer countries
    for i in range(N):
        country_name = params.country_list[i] if i < len(params.country_list) else f"Country_{i}"
        row_labels.append(country_name)
    
    # Column labels: exporter countries
    for j in range(N):
        country_name = params.country_list[j] if j < len(params.country_list) else f"Country_{j}"
        col_labels.append(country_name)
    
    # country_links is already in the correct 2D format (N, N)
    return country_links, row_labels, col_labels


def create_comparison_plots_section(baseline_sol: ModelSol, cf_sol: ModelSol, params: ModelParams):
    # ... existing code ...
    pass 