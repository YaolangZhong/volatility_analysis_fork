"""
Network Data Generator
======================

This script extracts real trade data from the model and generates
an HTML file with the D3.js visualization populated with actual data.
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional
from models import ModelParams, ModelSol
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "API"))

from model_pipeline import ModelPipeline


class NetworkDataExtractor:
    """Extracts network data from model parameters and solutions."""
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.country_names = list(params.country_list)
        self.sector_names = list(params.sector_list)
        
        # Country continent mapping (you can customize this)
        self.country_continents = {
            'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
            'BRA': 'South America', 'ARG': 'South America', 'CHL': 'South America',
            'DEU': 'Europe', 'FRA': 'Europe', 'GBR': 'Europe', 'ITA': 'Europe',
            'ESP': 'Europe', 'NLD': 'Europe', 'BEL': 'Europe', 'SWE': 'Europe',
            'NOR': 'Europe', 'DNK': 'Europe', 'FIN': 'Europe', 'AUT': 'Europe',
            'CHE': 'Europe', 'POL': 'Europe', 'RUS': 'Europe', 'CZE': 'Europe',
            'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia', 'IND': 'Asia',
            'IDN': 'Asia', 'THA': 'Asia', 'VNM': 'Asia', 'MYS': 'Asia',
            'SGP': 'Asia', 'PHL': 'Asia', 'TWN': 'Asia', 'HKG': 'Asia',
            'AUS': 'Oceania', 'NZL': 'Oceania',
            'ZAF': 'Africa', 'EGY': 'Africa', 'MAR': 'Africa', 'NGA': 'Africa',
            'TUR': 'Asia', 'ISR': 'Asia', 'SAU': 'Asia', 'ARE': 'Asia'
        }
    
    def extract_baseline_data(self) -> Dict[str, Any]:
        """Extract baseline country data from model parameters."""
        countries = []
        
        for i, country_code in enumerate(self.country_names):
            # Calculate aggregate expenditure for this country
            total_expenditure = float(np.sum(self.params.X[i, :]))
            
            # Get country full name (fallback to code if not available)
            country_name = self._get_country_name(country_code)
            
            # Get continent
            continent = self.country_continents.get(country_code, 'Other')
            
            countries.append({
                'id': country_code,
                'name': country_name,
                'expenditure': total_expenditure,
                'continent': continent,
                'sectors': {
                    sector_name: float(self.params.X[i, j])
                    for j, sector_name in enumerate(self.sector_names)
                }
            })
        
        return {'countries': countries}
    
    def extract_counterfactual_data(self, solution: ModelSol) -> Dict[str, Any]:
        """Extract counterfactual country data from model solution."""
        countries = []
        
        for i, country_code in enumerate(self.country_names):
            # Calculate aggregate expenditure for this country in counterfactual
            total_expenditure = float(np.sum(solution.X_prime[i, :]))
            
            # Get country full name
            country_name = self._get_country_name(country_code)
            
            # Get continent
            continent = self.country_continents.get(country_code, 'Other')
            
            countries.append({
                'id': country_code,
                'name': country_name,
                'expenditure': total_expenditure,
                'continent': continent,
                'sectors': {
                    sector_name: float(solution.X_prime[i, j])
                    for j, sector_name in enumerate(self.sector_names)
                }
            })
        
        return {'countries': countries}
    
    def _get_country_name(self, country_code: str) -> str:
        """Convert country code to full name."""
        country_names_map = {
            'USA': 'United States', 'CAN': 'Canada', 'MEX': 'Mexico',
            'BRA': 'Brazil', 'ARG': 'Argentina', 'CHL': 'Chile',
            'DEU': 'Germany', 'FRA': 'France', 'GBR': 'United Kingdom',
            'ITA': 'Italy', 'ESP': 'Spain', 'NLD': 'Netherlands',
            'BEL': 'Belgium', 'SWE': 'Sweden', 'NOR': 'Norway',
            'DNK': 'Denmark', 'FIN': 'Finland', 'AUT': 'Austria',
            'CHE': 'Switzerland', 'POL': 'Poland', 'RUS': 'Russia',
            'CZE': 'Czech Republic', 'CHN': 'China', 'JPN': 'Japan',
            'KOR': 'South Korea', 'IND': 'India', 'IDN': 'Indonesia',
            'THA': 'Thailand', 'VNM': 'Vietnam', 'MYS': 'Malaysia',
            'SGP': 'Singapore', 'PHL': 'Philippines', 'TWN': 'Taiwan',
            'HKG': 'Hong Kong', 'AUS': 'Australia', 'NZL': 'New Zealand',
            'ZAF': 'South Africa', 'EGY': 'Egypt', 'MAR': 'Morocco',
            'NGA': 'Nigeria', 'TUR': 'Turkey', 'ISR': 'Israel',
            'SAU': 'Saudi Arabia', 'ARE': 'United Arab Emirates'
        }
        return country_names_map.get(country_code, country_code)


class NetworkHTMLGenerator:
    """Generates HTML file with embedded D3.js visualization and real data."""
    
    def __init__(self, extractor: NetworkDataExtractor):
        self.extractor = extractor
    
    def generate_html_with_data(self, 
                              baseline_data: Dict[str, Any],
                              counterfactual_data: Optional[Dict[str, Any]] = None,
                              output_file: str = "network_d3_with_data.html",
                              use_3d: bool = False) -> str:
        """Generate HTML file with real data embedded."""
        
        # Read the template HTML
        template_file = "network_3d.html" if use_3d else "network_d3.html"
        with open(template_file, "r") as f:
            template_html = f.read()
        
        # Prepare data for embedding
        data_to_embed = {
            'baseline': baseline_data,
            'counterfactual': counterfactual_data if counterfactual_data else baseline_data
        }
        
        # Create JavaScript code to replace sample data
        if use_3d:
            data_js = f"""
        // Real data from trade model
        let realData = {json.dumps(data_to_embed, indent=2)};
        
        // Replace sample data with real data and update graph
        if (window.loadTradeData3D) {{
            window.loadTradeData3D(realData);
        }} else {{
            // Fallback: replace graphData directly
            graphData = {{
                baseline: {{
                    nodes: realData.baseline.countries.map(country => ({{
                        id: country.id,
                        name: country.name,
                        expenditure: country.expenditure,
                        continent: country.continent,
                        sectors: country.sectors
                    }})),
                    links: []
                }},
                counterfactual: {{
                    nodes: realData.counterfactual.countries.map(country => ({{
                        id: country.id,
                        name: country.name,
                        expenditure: country.expenditure,
                        continent: country.continent,
                        sectors: country.sectors
                    }})),
                    links: []
                }}
            }};
        }}
        """
        else:
            data_js = f"""
        // Real data from trade model
        let realData = {json.dumps(data_to_embed, indent=2)};
        
        // Replace sample data with real data
        sampleData = realData;
        """
        
        # Insert the real data into the HTML
        if use_3d:
            # For 3D version, find graphData and replace it
            sample_data_start = template_html.find("let graphData = {")
            sample_data_end = template_html.find("};", sample_data_start) + 2
            
            if sample_data_start != -1 and sample_data_end != -1:
                # Replace graphData with real data
                new_html = (template_html[:sample_data_start] + 
                           data_js + 
                           template_html[sample_data_end:])
            else:
                # Append before initGraph() call
                init_call = template_html.find("initGraph();")
                new_html = (template_html[:init_call] + 
                           data_js + "\n        " + 
                           template_html[init_call:])
        else:
            # For 2D version, find sampleData and replace it
            sample_data_start = template_html.find("let sampleData = {")
            sample_data_end = template_html.find("};", sample_data_start) + 2
            
            if sample_data_start != -1 and sample_data_end != -1:
                # Replace sample data with real data
                new_html = (template_html[:sample_data_start] + 
                           data_js + 
                           template_html[sample_data_end:])
            else:
                # If we can't find the sample data, append the real data
                script_end = template_html.rfind("</script>")
                new_html = (template_html[:script_end] + 
                           "\n" + data_js + "\n" + 
                           template_html[script_end:])
        
        # Write the new HTML file
        with open(output_file, "w") as f:
            f.write(new_html)
        
        return output_file


def generate_network_visualization(data_file: str = "../data.npz",
                                 counterfactual_scenario: Optional[Dict[str, Any]] = None,
                                 output_file: str = "network_d3_with_data.html",
                                 use_3d: bool = False) -> str:
    """
    Generate a complete D3.js network visualization with real trade data.
    
    Args:
        data_file: Path to the model data file
        counterfactual_scenario: Dict with 'importers', 'exporters', 'sectors', 'tariff_rate'
        output_file: Output HTML file name
    
    Returns:
        Path to generated HTML file
    """
    # Load model parameters
    params = ModelParams.load_from_npz(data_file)
    extractor = NetworkDataExtractor(params)
    
    # Extract baseline data
    baseline_data = extractor.extract_baseline_data()
    
    # Extract counterfactual data if scenario provided
    counterfactual_data = None
    if counterfactual_scenario:
        pipeline = ModelPipeline(data_file)
        
        # Solve counterfactual
        cf_key = pipeline.solve_counterfactual(
            importers=counterfactual_scenario.get('importers', ['USA']),
            exporters=counterfactual_scenario.get('exporters', ['CHN']),
            sectors=counterfactual_scenario.get('sectors', ['Manufacturing']),
            tariff_rate=counterfactual_scenario.get('tariff_rate', 20.0)
        )
        
        # Get solution
        cf_solution, _ = pipeline.get_counterfactual_results(cf_key)
        if cf_solution:
            counterfactual_data = extractor.extract_counterfactual_data(cf_solution)
    
    # Generate HTML
    generator = NetworkHTMLGenerator(extractor)
    output_path = generator.generate_html_with_data(
        baseline_data, counterfactual_data, output_file, use_3d
    )
    
    print(f"Generated network visualization: {output_path}")
    print(f"Countries: {len(baseline_data['countries'])}")
    print(f"Total baseline expenditure: {sum(c['expenditure'] for c in baseline_data['countries']):.2e}")
    
    if counterfactual_data:
        total_cf = sum(c['expenditure'] for c in counterfactual_data['countries'])
        total_base = sum(c['expenditure'] for c in baseline_data['countries'])
        change_pct = ((total_cf - total_base) / total_base) * 100
        print(f"Total counterfactual expenditure: {total_cf:.2e}")
        print(f"Overall change: {change_pct:+.2f}%")
    
    return output_path


if __name__ == "__main__":
    # Generate both 2D and 3D visualizations
    print("Generating 2D network visualization...")
    html_file_2d = generate_network_visualization(
        counterfactual_scenario=None,
        output_file="network_trade_2d.html",
        use_3d=False
    )
    
    print("Generating 3D network visualization...")
    html_file_3d = generate_network_visualization(
        counterfactual_scenario=None,
        output_file="network_trade_3d.html",
        use_3d=True
    )
    
    print(f"\n🎉 Generated both visualizations!")
    print(f"2D Version: {html_file_2d}")
    print(f"3D Version: {html_file_3d}")
    
    print("\n📋 Features Comparison:")
    print("2D Version:")
    print("- ✅ Smooth zoom and pan")
    print("- ✅ Multiple layout algorithms")
    print("- ✅ Export to SVG")
    
    print("\n3D Version:")
    print("- ✅ Full 3D navigation (orbit, zoom, pan)")
    print("- ✅ WASD keyboard controls")
    print("- ✅ Auto-rotate mode")
    print("- ✅ Immersive experience")
    print("- ✅ Click-to-focus on nodes")
    
    # Example of how to generate with counterfactual (using real sector names)
    print("\n" + "="*60)
    print("To generate with counterfactual, use a scenario like:")
    print("counterfactual_scenario = {")
    print("    'importers': ['USA', 'CAN'],")
    print("    'exporters': ['CHN'],")
    print("    'sectors': ['Textiles', 'Metal Products'],")
    print("    'tariff_rate': 25.0")
    print("}")
    print("And add use_3d=True for 3D version") 