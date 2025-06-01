import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy

from models import *
from equations import *
from solvers import *


## Directory to store the experiment
experiment_dir_name = "test"
dir = os.path.join("experiments", experiment_dir_name)
os.makedirs(dir, exist_ok=True)

## Generate the parameters
#### load from the npz.file to get from saved data or saved instance
real_data_file_name = "real_data/real_data_2017.npz"
params = ModelParams.load_from_npz(real_data_file_name)
#### Or, specify the parameters manually by the generate_simple_params function
# params = generate_simple_params(N=5,J=2)

## Save the parameters
params_file_name = os.path.join(dir, "params.npz")
params.save_to_npz(params_file_name)

## Generate the model with baseline shock
base_model = Model(params)

## create the solver and solve the model 
config = SolverConfig()
solver = ModelSolver(base_model, config)
solver.solve()

## save the baseline solution
sol_file_name = os.path.join(dir, "baseline_sol.npz")
base_model.sol.save_to_npz()

shocks1 = deepcopy(base_model.shock)
shocks2 = deepcopy(base_model.shock)
shocks_list = [shocks1, shocks2]
shocks_name_list = ["shock_1", "sector_shock_2"]

results = []
with ProcessPoolExecutor() as executor:
    # submit one solve job per shock
    futures = [
        executor.submit(solve_with_shock, base_model, shock)
        for shock in shocks_list
    ]
    # collect as they complete
    for future in as_completed(futures):
        sol = future.result()       # this is a ModelSol
    

