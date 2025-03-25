import os
import numpy as np
import time
from models import ModelShocks, ModelSol
from equations import calc_Pu_hat, calc_piu_hat
from equations_matrix import calc_X
from solvers import solve_price_and_cost
from functions import generate_rand_params
from optimization import reconstruct_w_hat, objective_w_hat_reduced
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed


GLOBAL_mp = None
GLOBAL_bench_sol = None
GLOBAL_numeraire_index = None


class EarlyStopException(Exception):
    """Optimization early stop signal."""

    pass


def init_worker(mp, bench_sol, numeraire_index):
    """Initialize worker processes."""
    global GLOBAL_mp
    global GLOBAL_bench_sol
    global GLOBAL_numeraire_index
    GLOBAL_mp = mp
    GLOBAL_bench_sol = bench_sol
    GLOBAL_numeraire_index = numeraire_index


def run_counterfactual(i, out_dir, shocks):
    """
    Solve i-th counterfactual equilibrium.

    Arguments:
        i: int, index of the counterfactual equilibrium
        out_dir: str, path to output directory
        shocks: ModelShocks, model shocks
    Global variables:
        mp: ModelParams, model parameters
        bench_sol: ModelSol, benchmark equilibrium
        numeraire_index: int, index of the numeraire country
    Returns:
        ModelSol, counterfactual equilibrium
    """
    # Load the global variables
    mp = GLOBAL_mp
    bench_sol = GLOBAL_bench_sol
    numeraire_index = GLOBAL_numeraire_index
    N = mp.N
    J = mp.J

    # Set the initial guess
    Xf_init = bench_sol.Xf_prime.copy()
    Xm_init = bench_sol.Xm_prime.copy()

    # (A) Dimention of the optimization problem is N-1
    dim_reduced = N - 1
    # Initial guess for the reduced problem
    x0_guess = np.ones(dim_reduced)

    # (B) Example of early stop with callback
    best_x = [None]
    res = None

    def callback_func(xk):
        """Callback function to check the objective value and stop the optimization."""
        val = objective_w_hat_reduced(
            xk, mp, shocks, Xf_init, Xm_init, numeraire_index
        )
        threshold = 1e-6
        if val < threshold:
            best_x[0] = xk.copy()
            raise EarlyStopException(
                f"Residual {val} < threshold {threshold}. Early stopping."
            )

    try:
        # (C) Optimize w_hat by using Nelder-Mead method
        res = minimize(
            objective_w_hat_reduced,
            x0_guess,
            args=(mp, shocks, Xf_init, Xm_init, numeraire_index),
            method="Nelder-Mead",
            callback=callback_func,
            options={"maxiter": 10000, "disp": False},
        )
    except EarlyStopException as e:
        pass

    # (D) Extract the solution: res is the official solution
    if res is not None and hasattr(res, "x"):
        x_reduced_opt = res.x
    else:
        # If res is None, use the best solution so far
        x_reduced_opt = best_x[0]

    # (E) Reconstruct w_hat (numeraire is automatically set to 1)
    w_hat_opt = reconstruct_w_hat(x_reduced_opt, numeraire_index, N)

    # (F) Calculate the equilibrium
    Pm_init = np.ones((N, J))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_opt,
        Pm_init,
        mp,
        shocks,
        max_iter=1000,
        tol=1e-7,
        mute=True,
    )
    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)
    Xf_prime, Xm_prime = calc_X(w_hat_opt, pif_hat, pim_hat, mp.td, mp, shocks)

    # (G) Save the results
    sol = ModelSol(
        params=mp,
        shocks=shocks,
        w_hat=w_hat_opt,
        c_hat=c_hat,
        Pf_hat=Pf_hat,
        Pm_hat=Pm_hat,
        pif_hat=pif_hat,
        pim_hat=pim_hat,
        Xf_prime=Xf_prime,
        Xm_prime=Xm_prime,
    )

    sol.save_to_npz(f"{out_dir}/counterfactual_{i}.npz")

    return f"Counterfactual equilibrium {i} saved."


def main():
    # =========================================================================
    # Step 1. Setup and load parameters
    out_dir = "output"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)

    # ========== For now, generate random parameters ==========
    N, J = 5, 3
    mp = generate_rand_params(N, J)
    if mp is None:
        print("Failed to generate random parameters")
        return None
    else:
        mp.save_to_npz(f"{out_dir}/model_params.npz")
    # ===== Replace this part with loading parameters from a file =====

    # =========================================================================
    # Step 2. Solve for the benchmark equilibrium
    bench_dir = f"{out_dir}/benchmark"
    os.makedirs(bench_dir, exist_ok=True)

    numeraire_index = 0

    # Generate shocks for the benchmark equilibrium
    # For benchmark, set all shocks to 1 (no shocks)
    lambda_hat = np.ones((N, J))
    df_hat = np.ones((N, N, J))
    dm_hat = np.ones((N, N, J))
    tilde_tau_prime = np.ones((N, N, J))

    bench_shocks = ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)
    if bench_shocks.check_consistency():
        bench_shocks.save_to_npz(f"{bench_dir}/bench_shocks.npz")
    else:
        print("Shocks are inconsistent")
        return None

    # ----------------------------------------------------
    # Setup for optimization
    # ----------------------------------------------------
    Xf_init = mp.Xf.copy()
    Xm_init = mp.Xm.copy()

    # (A) Dimention of the optimization problem is N-1
    dim_reduced = N - 1
    # Initial guess for the reduced problem
    x0_guess = np.ones(dim_reduced)

    # (B) Example of early stop with callback
    best_x = [None]
    res = None

    def callback_func(xk):
        """Callback function to check the objective value and stop the optimization."""
        val = objective_w_hat_reduced(
            xk, mp, bench_shocks, Xf_init, Xm_init, numeraire_index
        )
        threshold = 1e-6
        if val < threshold:
            best_x[0] = xk.copy()
            raise EarlyStopException(
                f"Residual {val} < threshold {threshold}. Early stopping."
            )

    try:
        # (C) Optimize w_hat by using Nelder-Mead method
        res = minimize(
            objective_w_hat_reduced,
            x0_guess,
            args=(mp, bench_shocks, Xf_init, Xm_init, numeraire_index),
            method="Nelder-Mead",
            callback=callback_func,
            options={"maxiter": 10000, "disp": True},
        )
    except EarlyStopException as e:
        print("Early stop triggered:", e)

    # (D) Extract the solution: res is the official solution
    if res is not None and hasattr(res, "x"):
        print("Optimization finished. Scipy result:")
        print(res)
        x_reduced_opt = res.x
    else:
        # If res is None, use the best solution so far
        x_reduced_opt = best_x[0]

    # (E) Reconstruct w_hat (numeraire is automatically set to 1)
    w_hat_opt = reconstruct_w_hat(x_reduced_opt, numeraire_index, N)
    print("Final wage changes (including numeraire=1):", w_hat_opt)

    # (F) Calculate the equilibrium
    Pm_init = np.ones((N, J))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_opt,
        Pm_init,
        mp,
        bench_shocks,
        max_iter=1000,
        tol=1e-7,
        mute=True,
    )
    Pf_hat = calc_Pu_hat(c_hat, "f", mp, bench_shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, bench_shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, bench_shocks)
    Xf_prime, Xm_prime = calc_X(
        w_hat_opt, pif_hat, pim_hat, mp.td, mp, bench_shocks
    )

    # (G) Save the results
    bench_sol = ModelSol(
        params=mp,
        shocks=bench_shocks,
        w_hat=w_hat_opt,
        c_hat=c_hat,
        Pf_hat=Pf_hat,
        Pm_hat=Pm_hat,
        pif_hat=pif_hat,
        pim_hat=pim_hat,
        Xf_prime=Xf_prime,
        Xm_prime=Xm_prime,
    )
    bench_sol.save_to_npz(f"{bench_dir}/equilibrium.npz")
    print("Benchmark equilibrium saved.")

    # # =========================================================================
    # # Step 3. Solve for counterfactual equilibria
    # num = 100

    # # ========== For now, generate random shocks ==========
    # shock_list = []
    # for i in range(num):
    #     lambda_hat = np.random.rand(N, J) * 0.2 + 1.0
    #     df_hat = np.random.rand(N, N, J) * 0.2 + 1.0
    #     dm_hat = np.random.rand(N, N, J) * 0.2 + 1.0
    #     tilde_tau_prime = np.random.rand(N, N, J) * 0.2 + 1.0
    #     shock_list.append(
    #         ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)
    #     )
    # # ========== Replace this part with generating shocks from estimated parameters ==========

    # with ProcessPoolExecutor(
    #     max_workers=os.cpu_count() - 2,
    #     initializer=init_worker,
    #     initargs=(mp, bench_sol, numeraire_index),
    # ) as executor:
    #     futures = []
    #     for i in range(num):
    #         fut = executor.submit(
    #             run_counterfactual, i + 1, out_dir, shock_list[i]
    #         )
    #         futures.append(fut)

    #     for fut in as_completed(futures):
    #         print(fut.result())

    # print("All counterfactual equilibria are solved.")

    # =========================================================================
    # Step. 4 Run simulations for different sigmas
    num = 100
    sigmas = [0.1, 0.2, 0.3]

    for sigma in sigmas:
        # Generate random shocks for corresponding sigma
        sigma_dir = f"{out_dir}/sigma_{sigma}"
        os.makedirs(sigma_dir, exist_ok=True)

        shock_list = []
        for i in range(num):
            lambda_hat = np.random.rand(N, J) * 0.2 + 1.0
            df_hat = np.ones((N, N, J))  # No shocks on trade cost
            dm_hat = np.ones((N, N, J))  # No shocks on trade cost
            tilde_tau_prime = np.ones((N, N, J))  # No shocks on trade cost
            shock_list.append(
                ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)
            )

        with ProcessPoolExecutor(
            max_workers=os.cpu_count() - 2,
            initializer=init_worker,
            initargs=(mp, bench_sol, numeraire_index),
        ) as executor:
            futures = []
            for i in range(num):
                fut = executor.submit(
                    run_counterfactual, i + 1, sigma_dir, shock_list[i]
                )
                futures.append(fut)

            for fut in as_completed(futures):
                print(fut.result())

        print(f"All counterfactual equilibria for sigma = {sigma} are saved.")


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
