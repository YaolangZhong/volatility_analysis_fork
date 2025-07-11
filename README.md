# 📋 Project Structure

This project implements a comprehensive economic analysis framework organized into four main components:

### 1. Mathematical Model
The foundation is a multi-country, multi-sector general equilibrium trade model based on Caliendo and Parro (2015), with key extensions including separate treatment of final goods and intermediate inputs. The model analyzes how economic variables respond to tariff policy.

#### Model Parameter Variables

The model uses calibrated parameters from input-output tables and trade data to represent the structure of the global economy. These parameters capture key economic relationships including production technologies, consumption patterns, and trade linkages.

**Usage Parameters:**
- **`N`** (scalar): Number of countries 
- **`S`** (scalar): Number of sectors 
- **`country_list`**: (N) List of country names
- **`sector_list`**: (S) List of sector names

**Structural Parameters:**
- **`alpha[n,s]`** (N×S): Share of sector s in country n's final consumption expenditure
- **`beta[n,s]`** (N×S): Share of value added (labor) in total production costs for sector s in country n
- **`gamma[n,j,k]`** (N×S×S): Share of sector k inputs used in sector j production in country n
- **`theta[s]`** (S): Trade elasticity parameter for sector s (governs substitution between sources)

- **`pif[n,i,s]`** (N×N×S): Final goods trade share - fraction of country n's final demand in sector s sourced from country i
- **`pim[n,i,s]`** (N×N×S): Intermediate goods trade share - fraction of country n's intermediate demand in sector s sourced from country i  
- **`pi[n,i,s]`** (N×N×S): Total trade share combining final and intermediate goods flows
- **`tilde_tau[n,i,s]`** (N×N×S): Iceberg trade costs from country i to country n in sector s (≥1, with diagonal = 1)

- **`Xf[n,s]`** (N×S): Final goods expenditure in sector s by country n
- **`Xm[n,s]`** (N×S): Intermediate goods expenditure in sector s by country n
- **`X[n,s]`** (N×S): Total expenditure in sector s by country n (Xf + Xm)
- **`V[n]`** (N): Total value added (wage bill) in country n
- **`D[n]`** (N): Trade deficit in country n (imports - exports)

#### Policy Shock Variables (Hat Algebra)

Policy experiments are represented as multiplicative shocks relative to baseline equilibrium. All shock variables use "hat" notation where X_hat = X_new / X_baseline.

**Productivity Shocks:**
- **`lambda_hat[n,s]`** (N×S): Productivity shock in sector s of country n (>0, baseline = 1)

**Trade Cost Shocks:**  
- **`df_hat[n,i,s]`** (N×N×S): Final goods trade cost shock from country i to country n in sector s (>0, diagonal = 1)
- **`dm_hat[n,i,s]`** (N×N×S): Intermediate goods trade cost shock from country i to country n in sector s (>0, diagonal = 1)  
- **`tilde_tau_hat[n,i,s]`** (N×N×S): Tariff shock from country i to country n in sector s (≥1, diagonal = 1)

*Note: Baseline shocks are all ones, representing no change from calibrated equilibrium.*

To do (not this stage): System of equations featuring the model.(equations.py)

### 2. Real Data
Model parameters are calibrated from real-world input-output tables and trade data, stored in compressed NumPy format (`data.npz`) for efficient loading and processing.

To do (not this stage):
- A seperate note describing the data-cleaning process to transform the raw data to the IO tables
- A seperate note describing the formula to transform the IO tables to the model parameters


### 3. Model Solution Engine
The computational core solves the equilibrium under baseline/counterfactual policy scenarios using fixed-point iteration algorithms.

#### Numerical Solution Process (solvers.py)

The model employs a **fixed-point iteration solver** to find general equilibrium where all markets clear simultaneously. The solution algorithm:

**Core Algorithm:**
1. **Initialization**: Start with initial wage guess and previous price indices
2. **Equilibrium Calculation**: Solve for all endogenous variables given current wage vector using economic equations
3. **Market Clearing Check**: Verify trade balance conditions through wage gradient computation
4. **Wage Update**: Adjust wages based on trade deficit deviations from target using gradient descent
5. **Convergence Test**: Continue until wage changes and price changes fall below tolerance thresholds
6. **Solution Storage**: Save converged equilibrium to ModelSol object

**Solver Configuration:**
- **`max_iter`**: Maximum iterations (default: 1,000,000)
- **`tol`**: Convergence tolerance (default: 1e-6) 
- **`vfactor`**: Wage adjustment factor (default: -0.2)
- **`bound_eps`**: Lower bound for wages (default: 1e-6)

**Convergence Monitoring**: Tracks maximum wage gradient and price index changes, with detailed iteration logging showing wage bounds, expenditure ranges, and convergence metrics.

#### Model Output Variables (Equilibrium Solution)

The solution represents the new economic equilibrium after applying policy shocks. All variables capture changes relative to the baseline calibrated economy.

**Price and Cost Variables:**
- **`w_hat[n]`** (N): Wage change in country n relative to baseline equilibrium
- **`c_hat[n,s]`** (N×S): Unit cost change in sector s of country n (incorporates labor and intermediate input costs)
- **`Pf_hat[n,s]`** (N×S): Final goods price index change in sector s of country n
- **`Pm_hat[n,s]`** (N×S): Intermediate goods price index change in sector s of country n  
- **`p_index[n]`** (N): Consumer price index change in country n (overall price level)

**Trade Flow Variables:**
- **`pif_hat[n,i,s]`** (N×N×S): Final goods trade share change from exporter i to importer n in sector s
- **`pim_hat[n,i,s]`** (N×N×S): Intermediate goods trade share change from exporter i to importer n in sector s
- **`pif_prime[n,i,s]`** (N×N×S): New final goods trade share from exporter i to importer n in sector s
- **`pim_prime[n,i,s]`** (N×N×S): New intermediate goods trade share from exporter i to importer n in sector s

**Expenditure and Production Variables:**
- **`Xf_prime[n,s]`** (N×S): New final goods expenditure in sector s by country n
- **`Xm_prime[n,s]`** (N×S): New intermediate goods expenditure in sector s by country n
- **`X_prime[n,s]`** (N×S): New total expenditure in sector s by country n
- **`Xf_prod_prime[n,s]`** (N×S): New final goods production value in sector s of country n
- **`Xm_prod_prime[n,s]`** (N×S): New intermediate goods production value in sector s of country n  
- **`X_prod_prime[n,s]`** (N×S): New total production value in sector s of country n
- **`output_prime[n,s]`** (N×S): New output demand in sector s of country n

**Welfare and Income Variables:**
- **`real_w_hat[n]`** (N): Real wage change in country n (nominal wage deflated by consumer price index)
- **`I_prime[n]`** (N): New nominal income in country n
- **`real_I_prime[n]`** (N): New real income in country n (welfare measure)
- **`D_prime[n]`** (N): New trade deficit in country n

**Network Linkage Variables:**
- **`sector_links[n,s,i,k]`** (N×S×N×S): Import linkages showing how country n's sector s depends on inputs from country i's sector k
- **`country_links[n,i]`** (N×N): Aggregate import linkages between countries n and i across all sectors

*Note: Variables with "_hat" suffix represent percentage changes (new/baseline), while "_prime" suffix represents new absolute levels after policy shocks.*

### 4. Interactive Visualization Platform
A Streamlit-based web application provides real-time analysis capabilities, allowing users to configure custom tariff scenarios, visualize results across multiple dimensions, and export data for further research.

#### **`data.npz`** - Model Dataset
**Purpose**: Calibrated parameters from input-output tables

### ⚙️ Model Computation Layer

The computational engine handles economic modeling through specialized mathematical algorithms:

#### **`equations.py`** - Mathematical Framework
**Purpose**: Core economic equations and equilibrium computation

**Key Functions**:
- `generate_equilibrium()` - Main equilibrium calculation
- `solve_price_and_cost()` - Unit costs and price indices computation
- `calc_expenditure_share()` - Trade share calculations
- `calc_X_prime()` - Expenditure equilibrium solving
- `calc_sector_links()` - Compute import linkage matrices
- `calc_c_hat()` - Unit cost changes
- `calc_Pu_hat()` - Price index changes
- `calc_pi_hat()` - Expenditure share changes
- `calc_Xf_prime()` - Final expenditure calculations
- `calc_Xm_prime()` - Intermediate expenditure calculations
- `calc_td_prime()` - Trade deficit calculations

#### **`solvers.py`** - Numerical Algorithms
**Purpose**: Fixed-point iteration solver with convergence monitoring

**Key Functions**:
- `ModelSolver` class - Main solver with configurable parameters
- `solve()` - Execute model solving with convergence monitoring
- `_check_convergence()` - Monitor solution convergence
- `_update_wages()` - Wage adjustment iterations

#### **`API/model_pipeline.py`** - Model Solving Pipeline
**Purpose**: Direct counterfactual solving with hash-based caching

**Key Functions**:
- `solve_counterfactual()` - Main solving function
- `get_counterfactual_results()` - Retrieve cached solutions
- `get_metadata_cached()` - Fast country/sector name lookup
- `clear_counterfactual_cache()` - Cache management
- `list_cached_scenarios()` - View cached scenarios

### 🎯 Streamlit Application Layer

The user-facing layer handles interaction, visualization, and data export through three core scripts:

#### **`API/app.py`** - Main Streamlit Interface
**Purpose**: Primary user interface and application orchestration

**Key Functions**:
- `main()` - Application entry point and UI coordination
- `load_baseline_model()` - Loads pre-solved baseline from pickle file
- `create_counterfactual_ui()` - Dynamic tariff configuration interface
- `generate_unified_tariff_data()` - Handles multiple tariff input modes
- `solve_counterfactual_model()` - Coordinates model solving and caching

**Features**: Model type selection, four tariff configuration modes, session state management, and cache control.

#### **`API/visualization.py`** - Interactive Visualization Engine
**Purpose**: Simplified visualization with performance optimization

**Key Functions**:
- `visualize_single_model()` - Display individual model results
- `visualize_comparison()` - Show percentage changes between models

**Capabilities**: Multi-dimensional variable support, performance caching, interactive controls for country/sector selection, and responsive design with adjustable layouts.

#### **`API/download_excel.py`** - Data Export Module
**Purpose**: Excel and CSV data export functionality

**Key Functions**:
- `create_excel_locally()` - Generate Excel files with variable data
- `create_csv_locally()` - Create CSV network analysis files
- `show_variable_download_section()` - UI for download options
- `flatten_sector_links_for_viz()` - Convert complex linkages to exportable format

**Formats**: Excel files with proper indexing, CSV network data files, and percentage change comparisons.

**Contents**: Trade data with bilateral flows, input-output coefficients between sectors, trade elasticity parameters, and economic aggregates including value added and expenditures.
