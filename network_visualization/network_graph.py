"""
Network Graph Visualization Module
==================================

This module implements network graph visualization for the trade model as described in temp.md:
- Nodes represent Country-Sector pairs (e.g., "USA_Manufacturing", "CHN_Electronics")
- Node size represents total expenditure X (baseline) or X_prime (counterfactual)
- Country-level clusters represent aggregated expenditure by country
- Color encoding: Red for increases, Green for decreases, intensity shows magnitude
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
from models import ModelParams, ModelSol


class NetworkDataProcessor:
    """Processes model data for network graph visualization."""
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.N = params.N
        self.S = params.S
        self.country_names = list(params.country_list)
        self.sector_names = list(params.sector_list)
    
    def create_baseline_nodes(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Create baseline network nodes from X matrix.
        
        Returns:
            Tuple of (sector_nodes, country_nodes)
        """
        sector_nodes = []
        country_nodes = []
        
        # Create sector nodes (Country-Sector pairs)
        for n in range(self.N):
            for s in range(self.S):
                country_name = self.country_names[n]
                sector_name = self.sector_names[s]
                node_id = f"{country_name}_{sector_name}"
                
                sector_nodes.append({
                    'id': node_id,
                    'country': country_name,
                    'sector': sector_name,
                    'country_idx': n,
                    'sector_idx': s,
                    'size': self.params.X[n, s],
                    'type': 'sector'
                })
        
        # Create country nodes (aggregated by country)
        for n in range(self.N):
            country_name = self.country_names[n]
            total_expenditure = np.sum(self.params.X[n, :])
            
            country_nodes.append({
                'id': f"{country_name}_TOTAL",
                'country': country_name,
                'country_idx': n,
                'size': total_expenditure,
                'type': 'country'
            })
        
        return sector_nodes, country_nodes
    
    def create_counterfactual_nodes(self, 
                                   baseline_sector_nodes: List[Dict],
                                   baseline_country_nodes: List[Dict],
                                   sol: ModelSol) -> Tuple[List[Dict], List[Dict]]:
        """
        Create counterfactual network nodes with change information.
        
        Args:
            baseline_sector_nodes: Baseline sector nodes
            baseline_country_nodes: Baseline country nodes
            sol: Model solution containing X_prime
        
        Returns:
            Tuple of (cf_sector_nodes, cf_country_nodes)
        """
        cf_sector_nodes = []
        cf_country_nodes = []
        
        # Create counterfactual sector nodes
        for node in baseline_sector_nodes:
            n = node['country_idx']
            s = node['sector_idx']
            
            baseline_size = self.params.X[n, s]
            cf_size = sol.X_prime[n, s]
            
            # Calculate percentage change
            pct_change = float(((cf_size - baseline_size) / baseline_size) * 100) if baseline_size > 0 else 0.0
            
            cf_node = node.copy()
            cf_node.update({
                'size': cf_size,
                'baseline_size': baseline_size,
                'change': cf_size - baseline_size,
                'pct_change': pct_change,
                'color_intensity': self._calculate_color_intensity(pct_change)
            })
            
            cf_sector_nodes.append(cf_node)
        
        # Create counterfactual country nodes
        for node in baseline_country_nodes:
            n = node['country_idx']
            
            baseline_size = np.sum(self.params.X[n, :])
            cf_size = np.sum(sol.X_prime[n, :])
            
            # Calculate percentage change
            pct_change = float(((cf_size - baseline_size) / baseline_size) * 100) if baseline_size > 0 else 0.0
            
            cf_node = node.copy()
            cf_node.update({
                'size': cf_size,
                'baseline_size': baseline_size,
                'change': cf_size - baseline_size,
                'pct_change': pct_change,
                'color_intensity': self._calculate_color_intensity(pct_change)
            })
            
            cf_country_nodes.append(cf_node)
        
        return cf_sector_nodes, cf_country_nodes
    
    def _calculate_color_intensity(self, pct_change: float) -> float:
        """
        Calculate color intensity based on percentage change.
        
        Args:
            pct_change: Percentage change value
        
        Returns:
            Intensity value between 0 and 1
        """
        # Cap the intensity at reasonable values (e.g., ±50% change = max intensity)
        max_change = 50.0
        capped_change = np.clip(abs(pct_change), 0, max_change)
        return capped_change / max_change
    
    def get_node_color(self, pct_change: float, intensity: float) -> str:
        """
        Get color for a node based on percentage change and intensity.
        
        Args:
            pct_change: Percentage change value
            intensity: Color intensity (0-1)
        
        Returns:
            Color string in RGB format
        """
        if pct_change > 0:
            # Red spectrum for increases
            red_intensity = int(255 * intensity)
            return f'rgb({red_intensity}, 0, 0)'
        elif pct_change < 0:
            # Green spectrum for decreases
            green_intensity = int(255 * intensity)
            return f'rgb(0, {green_intensity}, 0)'
        else:
            # Neutral gray for no change
            return 'rgb(128, 128, 128)'


class NetworkLayoutEngine:
    """Handles network layout and positioning."""
    
    def __init__(self, country_names: List[str]):
        self.country_names = country_names
        self.N = len(country_names)
    
    def create_sector_layout(self, sector_nodes: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """
        Create layout positions for sector nodes clustered by country.
        
        Args:
            sector_nodes: List of sector node dictionaries
        
        Returns:
            Dictionary mapping node IDs to (x, y) coordinates
        """
        positions = {}
        
        # Calculate country cluster centers in a circle
        country_centers = self._get_country_cluster_centers()
        
        # Position sector nodes within each country cluster
        country_sectors = {}
        for node in sector_nodes:
            country = node['country']
            if country not in country_sectors:
                country_sectors[country] = []
            country_sectors[country].append(node)
        
        for country, nodes in country_sectors.items():
            center_x, center_y = country_centers[country]
            sector_positions = self._arrange_sectors_in_cluster(nodes, center_x, center_y)
            positions.update(sector_positions)
        
        return positions
    
    def create_country_layout(self, country_nodes: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """
        Create layout positions for country nodes.
        
        Args:
            country_nodes: List of country node dictionaries
        
        Returns:
            Dictionary mapping node IDs to (x, y) coordinates
        """
        positions = {}
        country_centers = self._get_country_cluster_centers()
        
        for node in country_nodes:
            country = node['country']
            x, y = country_centers[country]
            positions[node['id']] = (x, y)
        
        return positions
    
    def _get_country_cluster_centers(self) -> Dict[str, Tuple[float, float]]:
        """Get cluster center positions for countries arranged in a circle."""
        centers = {}
        
        for i, country in enumerate(self.country_names):
            angle = 2 * np.pi * i / self.N
            x = 10 * np.cos(angle)  # Scale up for better spacing
            y = 10 * np.sin(angle)
            centers[country] = (x, y)
        
        return centers
    
    def _arrange_sectors_in_cluster(self, 
                                   sector_nodes: List[Dict],
                                   center_x: float,
                                   center_y: float) -> Dict[str, Tuple[float, float]]:
        """Arrange sector nodes in a cluster around country center."""
        positions = {}
        n_sectors = len(sector_nodes)
        
        if n_sectors == 1:
            positions[sector_nodes[0]['id']] = (center_x, center_y)
        else:
            cluster_radius = 2.0  # Radius of sector cluster within country
            
            for i, node in enumerate(sector_nodes):
                angle = 2 * np.pi * i / n_sectors
                x = center_x + cluster_radius * np.cos(angle)
                y = center_y + cluster_radius * np.sin(angle)
                positions[node['id']] = (x, y)
        
        return positions


class NetworkGraphVisualizer:
    """Creates interactive network graph visualizations using Plotly."""
    
    def __init__(self, params: ModelParams):
        self.data_processor = NetworkDataProcessor(params)
        self.layout_engine = NetworkLayoutEngine(list(params.country_list))
    
    def create_baseline_graph(self, 
                            show_country_nodes: bool = True,
                            show_sector_nodes: bool = True,
                            node_size_scale: float = 0.1) -> go.Figure:
        """
        Create baseline network graph visualization.
        
        Args:
            show_country_nodes: Whether to show country-level nodes
            show_sector_nodes: Whether to show sector-level nodes
            node_size_scale: Scaling factor for node sizes
        
        Returns:
            Plotly figure object
        """
        sector_nodes, country_nodes = self.data_processor.create_baseline_nodes()
        
        fig = go.Figure()
        
        if show_sector_nodes:
            self._add_sector_nodes_to_figure(fig, sector_nodes, node_size_scale, is_baseline=True)
        
        if show_country_nodes:
            self._add_country_nodes_to_figure(fig, country_nodes, node_size_scale, is_baseline=True)
        
        fig.update_layout(
            title="Baseline Trade Network",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000,
            # Enable better zooming
            dragmode='pan'
        )
        
        return fig
    
    def create_counterfactual_graph(self, 
                                   sol: ModelSol,
                                   show_country_nodes: bool = True,
                                   show_sector_nodes: bool = True,
                                   node_size_scale: float = 0.1) -> go.Figure:
        """
        Create counterfactual network graph visualization with change colors.
        
        Args:
            sol: Model solution containing counterfactual results
            show_country_nodes: Whether to show country-level nodes
            show_sector_nodes: Whether to show sector-level nodes
            node_size_scale: Scaling factor for node sizes
        
        Returns:
            Plotly figure object
        """
        baseline_sector_nodes, baseline_country_nodes = self.data_processor.create_baseline_nodes()
        cf_sector_nodes, cf_country_nodes = self.data_processor.create_counterfactual_nodes(
            baseline_sector_nodes, baseline_country_nodes, sol
        )
        
        fig = go.Figure()
        
        if show_sector_nodes:
            self._add_sector_nodes_to_figure(fig, cf_sector_nodes, node_size_scale, is_baseline=False)
        
        if show_country_nodes:
            self._add_country_nodes_to_figure(fig, cf_country_nodes, node_size_scale, is_baseline=False)
        
        fig.update_layout(
            title="Counterfactual Trade Network (Red=Increase, Green=Decrease)",
            showlegend=True,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1400,
            height=1000,
            # Enable better zooming
            dragmode='pan'
        )
        
        return fig
    
    def _add_sector_nodes_to_figure(self, 
                                   fig: go.Figure,
                                   nodes: List[Dict],
                                   size_scale: float,
                                   is_baseline: bool):
        """Add sector nodes to the figure."""
        positions = self.layout_engine.create_sector_layout(nodes)
        
        # Group nodes by country for better legend organization
        country_groups = {}
        for node in nodes:
            country = node['country']
            if country not in country_groups:
                country_groups[country] = []
            country_groups[country].append(node)
        
        # Add nodes for each country
        for country, country_nodes in country_groups.items():
            x_coords = []
            y_coords = []
            sizes = []
            colors = []
            hover_texts = []
            
            for node in country_nodes:
                x, y = positions[node['id']]
                x_coords.append(x)
                y_coords.append(y)
                sizes.append(max(8, node['size'] * size_scale))  # Ensure minimum visible size
                
                if is_baseline:
                    colors.append('blue')
                    hover_text = (f"{node['id']}<br>"
                                f"Expenditure: {node['size']:.2e}")
                else:
                    color = self.data_processor.get_node_color(
                        node['pct_change'], node['color_intensity']
                    )
                    colors.append(color)
                    hover_text = (f"{node['id']}<br>"
                                f"Baseline: {node['baseline_size']:.2e}<br>"
                                f"Counterfactual: {node['size']:.2e}<br>"
                                f"Change: {node['pct_change']:.2f}%")
                
                hover_texts.append(hover_text)
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=1, color='black'),
                    opacity=0.7
                ),
                text=hover_texts,
                hoverinfo='text',
                name=f"{country} (Sectors)",
                legendgroup=country
            ))
    
    def _add_country_nodes_to_figure(self, 
                                   fig: go.Figure,
                                   nodes: List[Dict],
                                   size_scale: float,
                                   is_baseline: bool):
        """Add country nodes to the figure."""
        positions = self.layout_engine.create_country_layout(nodes)
        
        x_coords = []
        y_coords = []
        sizes = []
        colors = []
        hover_texts = []
        country_names = []
        
        for node in nodes:
            x, y = positions[node['id']]
            x_coords.append(x)
            y_coords.append(y)
            sizes.append(max(50, node['size'] * size_scale * 0.8))  # Ensure minimum visible size
            country_names.append(node['country'])
            
            if is_baseline:
                colors.append('darkblue')
                hover_text = (f"{node['country']} (Total)<br>"
                            f"Total Expenditure: {node['size']:.2e}")
            else:
                color = self.data_processor.get_node_color(
                    node['pct_change'], node['color_intensity']
                )
                colors.append(color)
                hover_text = (f"{node['country']} (Total)<br>"
                            f"Baseline: {node['baseline_size']:.2e}<br>"
                            f"Counterfactual: {node['size']:.2e}<br>"
                            f"Change: {node['pct_change']:.2f}%")
            
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=2, color='black'),
                opacity=0.8,
                symbol='diamond'
            ),
            text=country_names,
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            hovertext=hover_texts,
            hoverinfo='text',
            name="Countries (Total)",
            showlegend=True
        ))


class NetworkGraphEngine:
    """Main engine for network graph visualization."""
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.visualizer = NetworkGraphVisualizer(params)
    
    def create_baseline_graph_ui(self):
        """Create Streamlit UI for baseline graph."""
        st.header("Baseline Network Graph")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            show_sectors = st.checkbox("Show Sector Nodes", value=True, key="baseline_sectors")
        with col2:
            show_countries = st.checkbox("Show Country Nodes", value=True, key="baseline_countries")
        with col3:
            size_scale = st.slider("Node Size Scale", 0.01, 1.0, 0.1, 0.01, key="baseline_scale")
        
        if show_sectors or show_countries:
            fig = self.visualizer.create_baseline_graph(
                show_country_nodes=show_countries,
                show_sector_nodes=show_sectors,
                node_size_scale=size_scale
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please select at least one type of node to display.")
    
    def create_counterfactual_graph_ui(self, sol: ModelSol):
        """Create Streamlit UI for counterfactual graph."""
        st.header("Counterfactual Network Graph")
        
        # Controls
        col1, col2, col3 = st.columns(3)
        with col1:
            show_sectors = st.checkbox("Show Sector Nodes", value=True, key="cf_sectors")
        with col2:
            show_countries = st.checkbox("Show Country Nodes", value=True, key="cf_countries")
        with col3:
            size_scale = st.slider("Node Size Scale", 0.01, 1.0, 0.1, 0.01, key="cf_scale")
        
        if show_sectors or show_countries:
            fig = self.visualizer.create_counterfactual_graph(
                sol=sol,
                show_country_nodes=show_countries,
                show_sector_nodes=show_sectors,
                node_size_scale=size_scale
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.info("""
            **Color Coding:**
            - 🔴 Red: Expenditure increased from baseline
            - 🟢 Green: Expenditure decreased from baseline
            - Darker colors indicate larger percentage changes
            """)
        else:
            st.info("Please select at least one type of node to display.")
    
    def create_comparison_view(self, sol: ModelSol):
        """Create side-by-side comparison of baseline and counterfactual graphs."""
        st.header("Network Graph Comparison")
        
        # Shared controls
        col1, col2, col3 = st.columns(3)
        with col1:
            show_sectors = st.checkbox("Show Sector Nodes", value=True, key="comp_sectors")
        with col2:
            show_countries = st.checkbox("Show Country Nodes", value=True, key="comp_countries")
        with col3:
            size_scale = st.slider("Node Size Scale", 0.01, 1.0, 0.1, 0.01, key="comp_scale")
        
        if show_sectors or show_countries:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Baseline")
                fig_baseline = self.visualizer.create_baseline_graph(
                    show_country_nodes=show_countries,
                    show_sector_nodes=show_sectors,
                    node_size_scale=size_scale
                )
                st.plotly_chart(fig_baseline, use_container_width=True)
            
            with col_right:
                st.subheader("Counterfactual")
                fig_cf = self.visualizer.create_counterfactual_graph(
                    sol=sol,
                    show_country_nodes=show_countries,
                    show_sector_nodes=show_sectors,
                    node_size_scale=size_scale
                )
                st.plotly_chart(fig_cf, use_container_width=True)
        else:
            st.info("Please select at least one type of node to display.") 