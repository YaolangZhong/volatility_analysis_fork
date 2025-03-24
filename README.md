# Volatility Analysis
This project analyzes how economic variables ((real/nominal) wages, price indices, expenditures,...) reacts exogenous productivity shocks and trade cost shocks. The model is an extended version of Caliendo and Parro (2015), which include (i) distinction of final goods intermediate inputs (they have different transportation costs), (ii) analysis of both productivity shocks and trade cost shocks.

![Structure Image](./figures_readme/structure.png)

## models.py
This file defines classes for model parameters, exogenous shocks, and model solutions. The classes are designed to handle parameter consistency checks, store data, and be used in solving the model.

### ModelParams
ModelParams holds all the core parameters used in the model—such as the number of countries and sectors, expenditure shares, and initial values for wages and labor supply. It also provides the following methods:
- **Consistency Checks**: Ensures parameters align with the model’s theoretical requirements (e.g., that expenditure shares sum to one).
- **Data Storage**: Can save parameters to a file for future reference in a npz file format.

### ModelShocks
ModelShocks stores exogenous shocks—specifically, productivity (lambda_hat) and trade cost shocks (df_hat, dm_hat). It includes the following methods:
- **Consistency Checks**: Verifies that shock values are valid (e.g., positivity, self-trade is not affected by shocks, etc.).
- **Data Storage**: Can save shock variables to a file for future reference in a npz file format.

### ModelSol
ModelSol encapsulates the model’s solution results after applying the model-solving algorithm. This class contains the following methods:
- **Data Storage**: Can save shock variables to a file for future reference in a npz file format.

### Usage
Usage specifies the usage of goods, final consumption or intermediate inputs. This class is used as an input for some functions (to calculate price index changes or expenditure share after the shock, final and intermediate goods sharing the functional form but using different variables)

## equations.py
This file defines model's equations to calculate equlibrium.

### calc_c_hat
Function to calculate unit cost change (equation (7) of the paper).

### calc_Pu_hat
Function to calculate price index change (equation (8) of the paper.)
It requires "usage" as an input (usage = "f" for final goods, = "m" for intermediate inputs)

### calc_pi_hat
Function to calculate expenditure share change (equation (9) of the paper).
It requires "usage" as an input (usage = "f" for final goods, = "m" for intermediate inputs)

### calc_Xf_prime
Function to calculate expenditure to the final goods (equation (10) of the paper).

### calc_Xm_prime
Function to calculate expenditure to the intermediate inputs (equation (11) of the paper).

### calc_td_prime
Function to calculate trade deficit after the shock (equation (12) of the paper).

## solvers.py
This file contains some functions to solve the model using loops.

### solve_price_and_cost
Function to solve c_hat and Pm_hat (inner loop). For any given value of w_hat, it works

## functions.py
This file defines some utility functions.

### generate_rand_params
Function to randomly generate parameters. Receives the number of countries and sectors and returns the ModelParams object.

### generate_symmetric_params
Function to randomly generate parameters, but each country should be identical. Doesn't work for some reason.