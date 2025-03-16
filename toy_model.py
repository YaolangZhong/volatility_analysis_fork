import os
import numpy as np
import time
from models import ModelShocks
from solvers import solve_equilibrium
from functions import generate_rand_params


def main():
    # =========================================================================
    # Step 1. Setup and generate random parameters

    out_dir = "toymodel_output"

    # Change the working directory to the current file's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create directories to store the results
    os.makedirs(out_dir, exist_ok=True)

    N, J = 2, 2

    # Generate random parameters
    mp = generate_rand_params(N, J)
    if mp is None:
        print("Failed to generate random parameters")
        return None
    else:
        mp.save_to_npz(f"{out_dir}/model_params.npz")

    # =========================================================================
    # Step 2. Solve for the benchmark equilibrium

    bench_dir = f"{out_dir}/benchmark"
    os.makedirs(bench_dir, exist_ok=True)

    # Set the numeraire country
    numeraire_index = 0

    # Set shocks
    lambda_hat = np.ones((N, J))
    df_hat = np.ones((N, N, J))
    dm_hat = np.ones((N, N, J))
    tilde_tau_prime = np.ones((N, N, J))

    shocks = ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)

    if shocks.check_consistency():
        shocks.save_to_npz(f"{bench_dir}/model_shocks.npz")
    else:
        print("Shocks are inconsistent")
        return None

    # Sove for the benchmark equilibrium
    Xf_init = mp.Xf.copy()
    Xm_init = mp.Xm.copy()

    bench_sol = solve_equilibrium(
        mp, shocks, numeraire_index, Xf_init, Xm_init, mute=False
    )

    # Save the results
    if bench_sol is not None:
        bench_sol.save_to_npz(f"{bench_dir}/equilibrium.npz")
        print("Benchmark equilibrium saved.")


if __name__ == "__main__":
    start_time = time.time()
    np.seterr(over="raise")

    main()

    # End time of the script
    end_time = time.time()

    # Format the elapsed time to mm:ss format
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    print(f"Elapsed time: {elapsed_time_str}")
