# API-Based Economic Model Analysis

This document explains how to use the new API-based architecture for economic model analysis, which separates data generation from visualization.

## Architecture Overview

The system is now split into two main components:

1. **Data Generation API Server** (`api_server.py`): Handles all model solving and data generation
2. **Visualization Client** (`app_with_api.py`): Handles user interface and data visualization

This separation allows for:
- Scalable model solving on dedicated servers
- Clean separation of concerns
- Excel download functionality for all model variables
- Both local and remote operation modes

## Simplified Tariff Configuration

**NEW**: The system now uses a unified tariff data structure across all 4 tariff modes for simplified processing:

### Unified Data Format
All tariff modes now produce the same data structure:
```python
tariff_data = {
    (importer, exporter, sector): rate,
    # Example:
    ("USA", "CHN", "Agriculture"): 25.0,
    ("USA", "CHN", "Manufacturing"): 25.0,
    ("USA", "DEU", "Agriculture"): 15.0,
    # ...
}
```

### Tariff Modes
1. **Uniform Rate**: Same rate applied to all (importer, exporter, sector) combinations
2. **Custom Rates by Country**: Different rates per country pair, applied to all sectors  
3. **Custom Rates by Sector**: Different rates per sector, applied to all country pairs
4. **Custom Rates by Country-Sector**: Individual rates for specific combinations

All modes expand to the unified `{(importer, exporter, sector): rate}` format before backend processing.

### Benefits
- **Simplified Backend**: Single lookup logic instead of 4 different cases
- **Easier Testing**: One data format to test and debug
- **Better Maintainability**: Cleaner, more predictable code structure
- **Future Extensibility**: Easy to add new tariff policy types

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Simple start
python start_api_server.py

# With custom configuration
python start_api_server.py --host 0.0.0.0 --port 8000 --reload
```

### 3. Run the Enhanced Streamlit App

```bash
streamlit run app_with_api.py
```

### 4. Toggle Between Local and API Modes

In the Streamlit app sidebar:
- **Local Mode**: Uses original `model_pipeline.py` (default)
- **API Mode**: Communicates with the API server

## API Endpoints

The API server exposes the following endpoints:

### Model Solving
- `GET /benchmark/solve` - Solve benchmark model
- `POST /counterfactual/solve` - Solve counterfactual model
- `GET /metadata` - Get model metadata (countries, sectors)

### Data Access
- `GET /models` - List all solved models
- `GET /models/{scenario_key}/variables` - Get variable info for a model
- `GET /models/{scenario_key}/variable/{variable_name}` - Get specific variable data

### Download Endpoints
- `GET /models/{scenario_key}/download/excel/{variable_name}` - Download single variable as Excel
- `GET /models/{scenario_key}/download/all` - Download all variables as Excel

### Management
- `DELETE /models/{scenario_key}` - Delete a solved model
- `POST /models/clear` - Clear all models (except benchmark)

## Using the Enhanced App

### Features Added

1. **API Mode Toggle**: Switch between local and API-based solving
2. **Excel Downloads**: Download any model variable as Excel files
3. **Status Indicators**: Clear visual feedback on model solving status
4. **Error Handling**: Robust error handling for API connectivity

### Download Functionality

For any solved model, you can:

1. **Download Individual Variables**: 
   - Select any variable (e.g., `w_hat`, `real_w_hat`, `X_prime`)
   - Download as Excel with country/sector labels
   - Files include metadata and proper formatting

2. **Download All Variables**:
   - Get all model variables in a single Excel file
   - Each variable on a separate sheet
   - Comprehensive model output export

### Excel File Format

Downloaded Excel files include:
- **1D arrays**: Countries as rows, variable values as columns
- **2D arrays**: Countries as rows, sectors as columns
- **3D arrays**: Flattened with descriptive column names
- **Proper indexing**: Country and sector names as row/column labels

## API Client Usage

You can also use the API client directly in Python:

```python
from api_client import ModelAPIClient

# Create client
client = ModelAPIClient("http://localhost:8000")

# Check connection
if client.health_check():
    print("API is available!")

# Solve benchmark
sol, params = client.solve_benchmark()

# Solve counterfactual
scenario_key = client.solve_counterfactual(
    importers=["USA"], 
    exporters=["CHN"], 
    sectors=["Manufacturing"], 
    tariff_rate=25.0
)

# Get results
sol, params = client.get_counterfactual_results(scenario_key)

# Download Excel
excel_data = client.download_variable_excel(scenario_key, "w_hat")
with open("wages.xlsx", "wb") as f:
    f.write(excel_data)
```

## Deployment Considerations

### Local Development
- Use `--reload` flag for automatic code reloading
- Single worker process is sufficient
- API server runs on `localhost:8000`

### Production Deployment
- Use multiple workers: `--workers 4`
- Configure appropriate host/port
- Consider using a reverse proxy (nginx)
- Set up proper logging and monitoring

### Security
- The API currently has no authentication
- For production, add API keys or other authentication
- Consider rate limiting for public deployments

## Troubleshooting

### API Connection Issues
1. Ensure API server is running: `curl http://localhost:8000/`
2. Check firewall settings for port 8000
3. Verify all dependencies are installed

### Model Solving Errors
1. Check that data files exist in `data/` directory
2. Verify model parameters are valid
3. Monitor API server logs for detailed error messages

### Download Issues
1. Ensure openpyxl is installed: `pip install openpyxl`
2. Check that the model has been solved successfully
3. Verify adequate disk space for large files

## Migration from Original App

The original `app.py` functionality is fully preserved:

1. **Existing workflows** work exactly the same in Local Mode
2. **All visualizations** remain unchanged
3. **New features** are additive - nothing is removed

To migrate:
1. Install new dependencies: `pip install fastapi uvicorn pydantic openpyxl requests`
2. Use `app_with_api.py` instead of `app.py`
3. Choose Local Mode for identical behavior to original app
4. Enable API Mode when ready to use the new architecture

## API Development

### Adding New Endpoints

To add new API endpoints:

1. Add endpoint function to `api_server.py`
2. Update `api_client.py` to include client method
3. Modify `app_with_api.py` to use new functionality

### Custom Variables

To add custom variable downloads:

1. Ensure the variable is included in `ModelSol`
2. Add any special formatting logic to `download_variable_excel`
3. Test with both API and local modes

## Performance Notes

- **API Mode**: Suitable for heavy computations on dedicated servers
- **Local Mode**: Better for development and small models
- **Caching**: API server caches solved models for fast retrieval
- **Memory**: API server keeps models in memory - monitor usage with many scenarios

## Support

For issues or questions:
1. Check this README first
2. Verify all dependencies are correctly installed
3. Test with Local Mode to isolate API-related issues
4. Check API server logs for detailed error information 