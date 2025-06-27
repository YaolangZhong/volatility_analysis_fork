# Volatility Analysis
This project analyzes how economic variables ((real/nominal) wages, price indices, expenditures,...) react to exogenous productivity shocks and trade cost shocks. The model is an extended version of Caliendo and Parro (2015), which includes (i) distinction of final goods and intermediate inputs (they have different transportation costs), (ii) analysis of both productivity shocks and trade cost shocks.

![Structure Image](./figures_readme/structure.svg)

## Core Architecture

### models.py - Model Data Management
This file defines three main dataclasses for organizing all model variables with comprehensive validation and I/O capabilities:

#### ModelParams
Stores all structural parameters of the trade model:
- **Dimensions**: `N` (countries), `S` (sectors) 
- **Preference parameters**: `alpha` (final consumption shares)
- **Technology parameters**: `beta` (value-added shares), `gamma` (intermediate input shares), `theta` (trade elasticities)
- **Trade data**: `pif`, `pim`, `pi` (trade shares), `tilde_tau` (trade costs)
- **Expenditure data**: `Xf`, `Xm`, `X` (final, intermediate, total expenditures)
- **Economic aggregates**: `V` (value added), `D` (trade deficits)
- **Identifiers**: `country_list`, `sector_list`

**Key Methods**:
- `check_consistency()`: Validates economic constraints (shares sum to 1, non-negativity, etc.)
- `save_to_npz()` / `load_from_npz()`: Persistent storage in NumPy format
- `unpack()`: Returns all parameters as tuple for easy unpacking

#### ModelShocks  
Stores exogenous shocks in "hat algebra" (multiplicative changes from baseline):
- `lambda_hat`: Productivity shocks (N×S)
- `df_hat`: Final goods trade cost shocks (N×N×S) 
- `dm_hat`: Intermediate goods trade cost shocks (N×N×S)
- `tilde_tau_hat`: Tariff shocks (N×N×S)

**Key Methods**:
- `check_consistency()`: Validates shock constraints (positivity, diagonal elements = 1)
- `is_baseline()`: Checks if all shocks equal 1 (no change)
- `reset_to_baseline()`: Sets all shocks to baseline values
- `save_to_npz()` / `load_from_npz()`: Persistent storage

#### ModelSol
Stores equilibrium solution variables:
- **Price changes**: `w_hat` (wages), `c_hat` (unit costs), `Pf_hat`, `Pm_hat` (price indices)
- **Trade changes**: `pif_hat`, `pim_hat` (trade share changes)
- **Expenditure outcomes**: `Xf_prime`, `Xm_prime`, `X_prime` (new expenditure levels)
- **Welfare measures**: `p_index` (consumer price index), `real_w` (real wages)
- **Trade balances**: `D_prime` (new trade deficits)
- **Production values**: `Xf_prod_prime`, `Xm_prod_prime`, `X_prod_prime`

**Key Methods**:
- `save_to_npz()` / `load_from_npz()`: Persistent storage
- `unpack()`: Returns all solution variables as tuple

#### Model
Main orchestrating class that combines parameters, shocks, and solutions:
- **Properties**: `N`, `S` for easy access to dimensions
- **State management**: `is_optimized` flag, `validate_compatibility()`
- **Initialization**: `reset_shocks()`, `reset_sols()` 
- **Documentation**: `summary()`, variable registry integration
- **I/O**: `from_npz()` class method for loading complete models

#### ModelRegistry System
Comprehensive documentation system for all 33 model variables:
- **VarInfo**: Named tuple with variable metadata (name, shape, meaning, indexing, component)
- **MODEL_VARIABLES**: Complete registry of all model variables
- **ModelRegistry**: Static methods for variable lookup, validation, and documentation
- **Utilities**: `print_variable_summary()`, `validate_variable_shape()`, `get_variables_by_dimension()`

### equations.py - Mathematical Core
This file implements the core mathematical equations for computing general equilibrium solutions.

#### Core Functions

**solve_price_and_cost()**
Computes unit costs and price indices using vectorized operations:
```python
# Unit costs (Equation 7): ĉ[n,s] = ŵ[n]^β[n,s] * ∏(P̂m[n,k]^γ[n,s,k])
c_hat = np.exp(beta * log_w_hat[:, np.newaxis] + 
               np.einsum('nkj,nk->nj', gamma, log_Pm_hat))

# Price indices (Equation 8): P̂[n,s] = (Σ π[n,i,s] * (ĉ[i,s] * κ̂[n,i,s])^(-θ[s]))^(-1/θ[s])
```
*Returns*: `(c_hat, Pf_hat, Pm_hat)`

**calc_expenditure_share()**
Computes trade share changes using vectorized broadcasting:
```python
# Expenditure shares (Equation 9): π̂[i,n,s] = λ̂[n,s] * (ĉ[n,s]*d̂[i,n,s])^(-θ[s]) / P̂[i,s]^(-θ[s])
```
*Returns*: Trade share changes (I×N×S)

**calc_X_prime()** [Numba-accelerated]
Solves expenditure equilibrium using iterative algorithm:
- **Income computation**: Includes wages, trade deficits, and tariff revenues
- **Final expenditure**: `Xf_prime[n,s] = α[n,s] * I_prime[n]`
- **Intermediate expenditure**: Uses gamma coefficients and trade linkages
- **Convergence**: Iterates until changes < tolerance

*Returns*: `(Xf_prime, Xm_prime)`

**generate_equilibrium()**
Main equilibrium computation orchestrating all components:
1. Pre-computes shock-adjusted trade costs
2. Solves prices and costs
3. Computes trade share changes  
4. Solves expenditure equilibrium
5. Calculates trade balances and welfare measures

*Returns*: Complete tuple of all 14 solution variables

#### Performance Features
- **Vectorized operations**: Uses NumPy broadcasting and `einsum` for efficiency
- **Numba acceleration**: Critical functions compiled for ~5-10x speedup
- **Memory optimization**: Reduced array copying, efficient convergence checking
- **Numerical stability**: Built-in bounds checking and tolerance handling

#### Mathematical Consistency
- **Economic constraints**: Maintains all theoretical relationships from Caliendo & Parro (2015)
- **Hat algebra**: Consistent multiplicative change framework throughout
- **Equilibrium conditions**: Ensures market clearing and balanced trade

## Legacy Documentation (equations.py detailed)

### calc_c_hat
Function to calculate unit cost change (equation (7) of the paper).

```math
\hat{c}_{i}^{s} = \hat{w}_{i}^{\beta_{i}^{s}} \prod_{k=1}^{s} \left( \hat{P}_{i}^{km'} \right)^{\beta_{i}^{sk}}
```

### calc_Pu_hat
Function to calculate price index change (equation (8) of the paper).
It requires "usage" as an input (usage = "f" for final goods, = "m" for intermediate inputs)

```math
\hat{P}_{n}^{su} = \left( \sum_{h=1}^{N} \pi_{nh0}^{su} \hat{\lambda}_{h}^{s} \left( \hat{c}_{h}^{s} \hat{d}_{nh}^{su} \right)^{- \theta^{s}} \right)^{- \frac{1}{\theta^{s}}}
```

### calc_pi_hat
Function to calculate expenditure share change (equation (9) of the paper).
It requires "usage" as an input (usage = "f" for final goods, = "m" for intermediate inputs)

```math
\hat{\pi}_{ni}^{su} = \frac{\hat{\lambda}_{i}^{s} \left( \hat{c}_{h}^{s} \hat{d}_{nh}^{su} \right)^{- \theta^{s}}}{\left( \hat{P}_{n}^{su} \right)^{- \theta^{s}}}
```

### calc_Xf_prime
Function to calculate expenditure to the final goods (equation (10) of the paper).

```math
X_{n}^{sf'} = \alpha_{n}^{s} \left[ \hat{w}_{n} w_{n0} L_{n0} + \sum_{s=1}^S \sum_{i=1}^N \frac{\tau_{ni}^{s'}}{1 + \tau_{ni}^{s'}} \left( \pi_{ni}^{sf'} X_{n}^{sf'} + \pi_{ni}^{sm'} X_{n}^{sm'} \right) + TD_{n}^{'} \right]
```

### calc_Xm_prime
Function to calculate expenditure to the intermediate inputs (equation (11) of the paper).

```math
X_{n}^{sm'} = \sum_{k=1}^S \beta_{n}^{ks} \left( \sum_{i=1}^{N} \frac{\pi_{in}^{kf'}}{1 + \tau_{in}^{kf'}} X_{i}^{kf'} + \sum_{i=1}^N \frac{\pi_{in}^{km'}}{1 + \tau_{in}^{km'}} X_{i}^{km'} \right)
```

### calc_td_prime
Function to calculate trade deficit after the shock (equation (12) of the paper).

```math
TD_{n}^{'} = \sum_{s=1}^S \sum_{i=1}^N \left( \underbrace{\frac{\pi_{ni}^{sf'} X_{nt}^{sf'} + \pi_{ni}^{sm'} X_{nt}^{sm'}}{1 + \tau_{ni}^{s'}}}_{\text{Import}} - \underbrace{\frac{\pi_{in}^{sf'} X_{i}^{sf'} + \pi_{in}^{sm'} X_{i}^{sm'}}{1 + \tau_{in}^{s'}}}_{\text{Export}} \right)
```

## solvers.py
This file contains some functions to solve the model using loops.

### solve_price_and_cost
Function to solve `c_hat` and `Pm_hat` (inner loop). For any given value of $\{ \hat{w}_{n} \}$, it calculates $\{ \hat{c}_{n}^{s} \}$ and $\{ \hat{P}_{n}^{s,m} \}$ that simultaneously satisfy equations (7) and (8). It simply uses loops.

### solve_X_prime
Function to solve $\{ X_{n}^{s, f '} \}$ and $\{ X_{n}^{s, m '} \}$. For any given value of $\{ \hat{w}_n \}$, $\{ \hat{\pi}_{ni}^{s, f} \}$, $\{ \hat{\pi}_{ni}^{s, m} \}$, $TD_{n}^{'}$. Takes the initial guess of $\{ X_{n}^{s, f '} \}$ and $\{ X_{n}^{s, m '} \}$ and solve by using loops.

### <span style="color: grey; ">solve_equilibrium</span>
Function to solve the entire equilibrium using loops. It guesses $\{ \hat{w}_n \}$ and calculate endogenous variables and the difference of world-GDP-normalized trade deficit

```math
ZW2_n = \frac{TD_{n}^{'}}{\sum_{i=1}^{N} \hat{w}_{n} w_{n0} L_{n0}} - \frac{TD_{n0}}{\sum_{i=1}^{N} w_{n0} L_{n0}},
```

and updates the guess $\{ \hat{w}_n \}$ by

```math
\hat{w}_{n}^{\text{new}} = \hat{w}_{n}^{\text{old}} \exp \left(-\frac{ZW2_n}{10} \right).
```

It doesn't work for some reason, and is no longer used.

## equations_matrix.py
This file contains the function to solve $X_f$ and $X_m$ by usin linear algebgra.

### calc_X
Function to solve $\{ X_{n}^{s, f '} \}$ and $\{ X_{n}^{s, m '} \}$ by simultaneously solving equations (10) and (11). Since they are linear equations, it should be faster than using loop method in solve_X_prime in solvers.py and solve_X_prime can be replaced with it.

## optimization.py
This file defines the objective function for the mathematical optimization, which will be used to solve the equilibrium.

### reconstruct_w_hat
Function to reconstruct a $N$-length vector of $\hat{w}_{n}$ from the $(N-1)$-length vector without the numeraire country.

### objective_w_hat_reduced
Function to calculate the normalized trade deficit $ZW2_n$ for countries other than the numeraire country and returns the max-absolute value of the $\{ZW2_n\}$ vector. Used as the objective function for solving equilibrium.

## functions.py
This file defines some utility functions.

### generate_rand_params
Function to randomly generate parameters. Receives the number of countries and sectors and returns the ModelParams object.

### <span style="color: grey; ">generate_simple_params</span>
Function to construct ModelParams object with the fixed value of parameter sets. $N=2$ and $J=1$. It was used to compare the model's analytical solution and the output of Python programs here.

## toy_model.py
This file solves the model for randomly generated parameters.

## main.py
This file solves the equilibrium from the parameters read from the original data.