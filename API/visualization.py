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
    
    def calculate_percentage_change(self, val1: np.ndarray, val2: np.ndarray) -> np.ndarray:
        """Calculate percentage change from val1 to val2."""
        return 100 * (val2 - val1) / (np.abs(val1) + 1e-8)
    
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
        with cols[1]:
            if st.button(f"Remove ALL {label}", key=clear_button_key):
                st.session_state[key] = []
        
        return st.multiselect(label, options, default=default, key=key)
    
    def create_figure_size_controls(self) -> Tuple[int, int]:
        """Create figure size control sliders."""
        st.markdown("### Figure Size Adjustment")
        fig_width = st.slider(
            "Figure Width", 
            min_value=400, max_value=2000, 
            value=st.session_state.get("fig_width", 1600), 
            step=100
        )
        fig_height = st.slider(
            "Figure Height", 
            min_value=300, max_value=1000, 
            value=st.session_state.get("fig_height", 700), 
            step=50
        )
        st.session_state["fig_width"] = fig_width
        st.session_state["fig_height"] = fig_height
        return fig_width, fig_height


class PlotlyVisualizer:
    """Creates Plotly visualizations for model results."""
    
    def __init__(self, data_processor: VisualizationDataProcessor):
        self.data_processor = data_processor
    
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
        """Visualize 1D variables (country or sector level)."""
        if value.shape[0] == len(self.data_processor.country_names):
            # Country-level data
            names = self.data_processor.country_names
            label_type = "Countries"
            selected_items_in_order = [c for c in names if c in selected_items]
        else:
            # Sector-level data
            names = self.data_processor.sector_names
            label_type = "Sectors"  
            selected_items_in_order = selected_items.copy()
        
        bars = []
        labels = []
        for name in selected_items_in_order:
            idx = names.index(name)
            bars.append(value[idx])
            labels.append(name)
        
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
        """Visualize 3D variables (importer-exporter-sector level)."""
        selected_importers_in_order = [c for c in self.data_processor.country_names if c in selected_importers]
        selected_exporters_in_order = [c for c in self.data_processor.country_names if c in selected_exporters]
        
        if is_percentage_change:
            y_label = f"{variable_name} (% Change)"
        else:
            y_label = f"{variable_name} (% Change)" if variable_name.endswith("_hat") else variable_name
        
        for importer in selected_importers_in_order:
            for exporter in selected_exporters_in_order:
                i_idx = self.data_processor.country_names.index(importer)
                e_idx = self.data_processor.country_names.index(exporter)
                bars = []
                labels = []
                for sector in selected_sectors:
                    s_idx = self.data_processor.sector_names.index(sector)
                    bars.append(value[i_idx, e_idx, s_idx])
                    labels.append(sector)
                
                fig = self.create_bar_chart(
                    labels, bars, 
                    f"{importer} (Importer) — {exporter} (Exporter): Selected Sectors",
                    "Sector", y_label, fig_width, fig_height
                )
                st.plotly_chart(fig, use_container_width=False)


class ModelVisualizationEngine:
    """Main engine for model visualization that coordinates all components."""
    
    def __init__(self, country_names: List[str], sector_names: List[str]):
        self.data_processor = VisualizationDataProcessor(country_names, sector_names)
        self.ui = VisualizationUI(self.data_processor)
        self.visualizer = PlotlyVisualizer(self.data_processor)
    
    def display_variable_description(self, variable_name: str):
        """Display variable description if available."""
        description = self.data_processor.get_variable_description(variable_name)
        if description:
            st.markdown(description, unsafe_allow_html=True)
    
    def visualize_single_model(self, solution: ModelSol):
        """Visualize results from a single model."""
        st.header("Variables and Visualization")
        
        # Get available variables
        sol_dict = solution.__dict__
        variable = st.selectbox("Choose an output variable", list(sol_dict.keys()))
        
        if variable:
            self.display_variable_description(variable)
            value = sol_dict[variable]
            st.write(f"Variable shape: {np.shape(value)}")
            
            self._visualize_variable(value, variable)
    
    def visualize_comparison(self, sol1: ModelSol, sol2: ModelSol):
        """Visualize comparison between two models."""
        st.header("Variables and Visualization")
        
        # Get common variables
        sol1_dict = sol1.__dict__
        sol2_dict = sol2.__dict__
        variable_keys = list(set(sol1_dict.keys()) & set(sol2_dict.keys()))
        variable = st.selectbox("Choose an output variable", variable_keys)
        
        if variable:
            self.display_variable_description(variable)
            
            val1 = sol1_dict[variable]
            val2 = sol2_dict[variable]
            
            # Calculate percentage change for ALL variables
            value = self.data_processor.calculate_percentage_change(val1, val2)
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
            names = self.data_processor.country_names_sorted
            label = "Countries"
            key_prefix = "country"
            default_list = []
        else:
            names = self.data_processor.sector_names
            label = "Sectors"
            key_prefix = "sector"
            default_list = names

        selected_items = self.ui.create_multiselect_with_buttons(
            label, names, default_list, 
            f"{key_prefix}_multiselect_{variable_name}",
            f"select_all_{key_prefix}_{variable_name}",
            f"remove_all_{key_prefix}_{variable_name}"
        )

        fig_width, fig_height = self.ui.create_figure_size_controls()
        
        if selected_items:
            self.visualizer.visualize_1d_variable(
                value, variable_name, selected_items, fig_width, fig_height, is_percentage_change
            )
        else:
            st.info(f"No {label.lower()} selected.")
    
    def _visualize_2d_variable(self, value: np.ndarray, variable_name: str, is_percentage_change: bool = False):
        """Handle 2D variable visualization with UI controls."""
        selected_countries = self.ui.create_multiselect_with_buttons(
            "Countries", self.data_processor.country_names_sorted, [],
            f"country_multiselect_{variable_name}", f"select_all_countries_{variable_name}", f"remove_all_countries_{variable_name}"
        )
        
        selected_sectors = self.ui.create_multiselect_with_buttons(
            "Sectors", self.data_processor.sector_names, self.data_processor.sector_names,
            f"sector_multiselect_{variable_name}", f"select_all_sectors_{variable_name}", f"remove_all_sectors_{variable_name}"
        )
        
        if selected_countries and selected_sectors:
            fig_width, fig_height = self.ui.create_figure_size_controls()
            self.visualizer.visualize_2d_variable(
                value, variable_name, selected_countries, selected_sectors, fig_width, fig_height, is_percentage_change
            )
        else:
            st.info("No countries or sectors selected.")
    
    def _visualize_3d_variable(self, value: np.ndarray, variable_name: str, is_percentage_change: bool = False):
        """Handle 3D variable visualization with UI controls."""
        selected_importers = self.ui.create_multiselect_with_buttons(
            "Importer Countries", self.data_processor.country_names_sorted, [],
            f"importer_multiselect_{variable_name}", f"select_all_importers_{variable_name}", f"remove_all_importers_{variable_name}"
        )
        
        selected_exporters = self.ui.create_multiselect_with_buttons(
            "Exporter Countries", self.data_processor.country_names_sorted, [],
            f"exporter_multiselect_{variable_name}", f"select_all_exporters_{variable_name}", f"remove_all_exporters_{variable_name}"
        )
        
        selected_sectors = self.ui.create_multiselect_with_buttons(
            "Sectors", self.data_processor.sector_names, self.data_processor.sector_names,
            f"sector_multiselect_3d_{variable_name}", f"select_all_sectors_3d_{variable_name}", f"remove_all_sectors_3d_{variable_name}"
        )
        
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