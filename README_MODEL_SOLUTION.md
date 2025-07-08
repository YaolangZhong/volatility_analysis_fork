# Economic Model Solution Framework

## Overview

This project implements a sophisticated multi-country, multi-sector trade model based on the Furusawa et al. (2023) framework. The model analyzes international trade flows, tariff impacts, and economic equilibrium under various policy scenarios.

## Table of Contents
- [Model Theory](#model-theory)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Mathematical Framework](#mathematical-framework)
- [Data Structure](#data-structure)
- [Solution Process](#solution-process)
- [Usage Examples](#usage-examples)
- [Technical Implementation](#technical-implementation)

## Model Theory

### Economic Framework
The model implements a general equilibrium trade model with the following key features:

1. **Multi-Country, Multi-Sector**: Handles N countries and S sectors simultaneously
2. **Input-Output Linkages**: Sectors use outputs from other sectors as intermediate inputs
3. **Trade Costs and Tariffs**: Incorporates both iceberg trade costs and ad-valorem tariffs
4. **Endogenous Prices**: Wages, sector prices, and trade shares determined endogenously
5. **Counterfactual Analysis**: Compare equilibrium outcomes under different policy scenarios

### Key Variables
- **w**: Wages by country (N×1)
- **Pf, Pm**: Final and intermediate good prices by country-sector (N×S)
- **π (pi)**: Trade shares - probability of sourcing from each country (N×N×S)
- **X**: Total expenditure by country-sector (N×S)
- **τ (tau)**: Tariff factors (1 + tariff rate) by importer-exporter-sector (N×N×S)

## System Architecture

The solution framework consists of several interconnected modules:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │ Solution Layer  │
│                 │    │                 │    │                 │
│ • Parameters    │───▶│ • Equations     │───▶│ • Solvers       │
│ • Calibration   │    │ • Equilibrium   │    │ • Convergence   │
│ • Validation    │    │ • Constraints   │    │ • Results       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### 1. Models Module (`models.py`)
**Purpose**: Defines data structures and model parameters
- `ModelParams`: Container for all model parameters and calibrated values
- `ModelSol`: Container for equilibrium solution variables
- `Model`: Main model class that ties parameters and solutions together

**Key Features**:
- Type-safe parameter handling
- Built-in validation and consistency checks
- Serialization support (save/load from .npz files)
- Comprehensive variable registry with metadata

### 2. Equations Module (`equations.py`)
**Purpose**: Implements the mathematical core of the model
- **Core Functions**:
  - `generate_equilibrium()`: Main equilibrium calculation function
  - `calc_sector_links()`: Computes 4D import linkage matrix
  - Various utility functions for price calculations, trade flows

**Mathematical Operations**:
- Trade share calculations using gravity-style equations
- Price index computations with CES aggregation
- Market clearing conditions
- Income and expenditure balance equations

### 3. Solvers Module (`solvers.py`)
**Purpose**: Numerical solution algorithms
- `ModelSolver`: Main solver class with iteration control
- **Algorithm**: Fixed-point iteration with adaptive convergence
- **Features**:
  - Configurable tolerance and maximum iterations
  - Convergence diagnostics and monitoring
  - Robust handling of numerical edge cases

## Mathematical Framework

### Core Equilibrium System

The model solves a system of nonlinear equations to find equilibrium values:

#### 1. Trade Shares (Gravity Equation)
```
π[n,i,s] = (τ[n,i,s] * c[i,s])^(-θ[s]) / Σ_k (τ[n,k,s] * c[k,s])^(-θ[s])
```

#### 2. Price Indices
```
Pf[n,s] = [Σ_i (τ[n,i,s] * c[i,s])^(-θ[s])]^(-1/θ[s])
Pm[n,s] = [Σ_i (τ[n,i,s] * c[i,s])^(-θ[s])]^(-1/θ[s])
```

#### 3. Unit Costs
```
c[i,s] = (w[i]^β[i,s]) * Π_k (Pm[i,k]^γ[i,s,k])
```

#### 4. Market Clearing
```
X[i,s] = Σ_n π[n,i,s] * (D[n,s] + Σ_k γ[n,k,s] * X[n,k])
```

#### 5. Trade Balance
```
Σ_s X[i,s] = w[i] * L[i] + D[i]
```

### Extended Variables

The model also computes additional economic indicators:

#### Sector Links (4D Matrix)
```
sector_links[n,s,i,j] = π[n,i,j] * γ[n,s,j] * X[n,j] / (1 + τ[n,i,j])
```
This represents imports by country n for output sector s from country i's sector j.

## Data Structure

### Input Data Requirements
- **Countries**: List of country codes/names (N countries)
- **Sectors**: List of sector names (S sectors)
- **Parameters**:
  - `α[n,s]`: Final demand shares (N×S)
  - `β[n,s]`: Labor cost shares (N×S)
  - `γ[n,s,k]`: Input-output coefficients (N×S×S)
  - `θ[s]`: Trade elasticity parameters (S×1)
  - `τ[n,i,s]`: Baseline tariff factors (N×N×S)

### Output Data Structure
The solution contains 19+ equilibrium variables including:
- **Price Variables**: `Pf_hat`, `Pm_hat`, `c_hat`, `p_index`
- **Quantity Variables**: `X_prime`, `Xf_prime`, `Xm_prime`, `output_prime`
- **Trade Variables**: `pif_hat`, `pim_hat`, `pif_prime`, `pim_prime`
- **Welfare Variables**: `real_w_hat`, `I_prime`, `real_I_prime`
- **Linkage Variables**: `sector_links` (4D)

## Solution Process

### Algorithm Flow

1. **Initialization**
   - Load parameters and validate consistency
   - Set initial guess for wages (typically w = 1 for all countries)
   - Initialize price indices

2. **Iteration Loop**
   ```
   do {
       // Update equilibrium variables
       (prices, trade_shares, expenditures, ...) = generate_equilibrium(wages, params)
       
       // Check convergence
       convergence_check = max(|new_wages - old_wages|)
       
       // Update wages for next iteration
       wages = new_wages
       
   } while (convergence_check > tolerance && iterations < max_iter)
   ```

3. **Convergence Criteria**
   - Default tolerance: 1e-8
   - Maximum iterations: 1000
   - Monitors both wage changes and price index stability

4. **Post-Processing**
   - Calculate additional derived variables
   - Compute welfare and trade flow metrics
   - Generate sector linkage matrices

### Performance Characteristics
- **Typical Convergence**: 10-50 iterations for well-calibrated models
- **Computational Complexity**: O(N²S) per iteration
- **Memory Usage**: Scales as O(N²S) for trade share matrices

## Usage Examples

### Basic Model Solving
```python
from models import ModelParams, Model
from solvers import ModelSolver

# Load parameters
params = ModelParams.load_from_npz("data_2017.npz")

# Create and solve model
model = Model(params)
solver = ModelSolver(model)
solver.solve()

# Access results
solution = model.sol
wages = solution.real_w_hat
trade_shares = solution.pif_prime
```

### Counterfactual Analysis
```python
# Modify tariffs for counterfactual
cf_params = params.copy()
cf_params.tilde_tau[usa_idx, chn_idx, :] *= 1.25  # 25% tariff increase

# Solve counterfactual
cf_model = Model(cf_params)
cf_solver = ModelSolver(cf_model)
cf_solver.solve()

# Compare outcomes
welfare_change = cf_model.sol.real_w_hat / model.sol.real_w_hat
```

## Technical Implementation

### Key Design Principles
1. **Modularity**: Clean separation between data, equations, and solvers
2. **Type Safety**: Extensive use of NumPy arrays with shape validation
3. **Performance**: Vectorized operations for computational efficiency
4. **Extensibility**: Easy to add new variables or modify equations
5. **Reproducibility**: Deterministic results with proper random seed management

### Dependencies
- **NumPy**: Core numerical computations
- **SciPy**: Advanced mathematical functions
- **Pandas**: Data manipulation and I/O
- **Matplotlib**: Visualization support

### Error Handling
- Parameter validation with detailed error messages
- Convergence failure diagnostics
- Numerical stability checks (NaN, infinity detection)
- Shape compatibility verification

### Testing Framework
- Unit tests for individual equation functions
- Integration tests for full model solutions
- Regression tests against known benchmark results
- Performance benchmarking suite

---

## Development and Contribution

### Code Organization
```
volatility_analysis/
├── API/                    # Streamlit visualization interface
├── data/                   # Input data and calibration files
├── network_visualization/  # Network analysis tools
├── models.py              # Core data structures
├── equations.py           # Mathematical implementations
├── solvers.py            # Numerical solution algorithms
└── README_MODEL_SOLUTION.md # This documentation
```

### Future Enhancements
- [ ] GPU acceleration for large-scale models
- [ ] Alternative solution algorithms (Newton-Raphson, etc.)
- [ ] Dynamic model extensions
- [ ] Uncertainty quantification methods
- [ ] Advanced visualization capabilities

---

*Documentation Version: 1.0*  
*Last Updated: January 2025*  
*Compatible with: Economic Model v2.0+* 