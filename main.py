import os
import numpy as np
import time
from models import ModelParams, ModelShocks, ModelSol
from equations import calc_Pu_hat, calc_piu_hat
from equations_autograd import calc_X
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


def run_counterfactual(shocks):
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

    iter_count = [0]

    n = len(x0_guess)
    eps = 1e-12  # Positive value close to 0
    bnds = [(eps, None)] * n  # Lower bound: eps, Upper bound: None

    def callback_func(xk):
        """Callback function to check the objective value and stop the optimization."""
        iter_count[0] += 1  # Increment the iteration counter
        val = objective_w_hat_reduced(
            xk, mp, shocks, Xf_init, Xm_init, numeraire_index
        )
        # Print the current loss value for each iteration
        print(f"Iteration {iter_count[0]}: loss = {val}")
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
            # method="Nelder-Mead",
            method="L-BFGS-B",
            callback=callback_func,
            bounds=bnds,
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

    return sol


def main():
    # =========================================================================
    # Step 1. Setup and load parameters
    # =========================================================================
    out_dir = "output"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)

    # data = np.load("real_data.npz")
    data = np.load("real_data_2017.npz")
    N, J = data["N"], data["J"]
    mp = ModelParams(
        N=N,
        J=J,
        alpha=data["alpha"],
        beta=data["beta"],
        gamma=data["gamma"],
        theta=data["theta"],
        pif=data["pi_f"],
        pim=data["pi_m"],
        tilde_tau=data["tilde_tau"],
        Xf=np.ones((N, J)),
        Xm=np.ones((N, J)),
        w0=data["VA"],
        L0=np.ones_like(data["VA"]),
        td=data["D"],
    )
    print("Loaded the parameters from the real data")
    mp.save_to_npz(f"{out_dir}/params.npz")

    # =========================================================================
    # Step 2. Solve for the benchmark equilibrium
    # =========================================================================
    bench_dir = f"{out_dir}/benchmark"
    os.makedirs(bench_dir, exist_ok=True)

    numeraire_index = 0

    # # Generate shocks for the benchmark equilibrium
    # # For benchmark, set all shocks to 1 (no shocks)
    # lambda_hat = np.ones((N, J))
    # df_hat = np.ones((N, N, J))
    # dm_hat = np.ones((N, N, J))
    # tilde_tau_prime = np.ones((N, N, J))

    # bench_shocks = ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)
    # if bench_shocks.check_consistency():
    #     bench_shocks.save_to_npz(f"{bench_dir}/shocks.npz")
    # else:
    #     print("Shocks are inconsistent")
    #     return None

    # # ----------------------------------------------------
    # # Setup for optimization
    # # ----------------------------------------------------
    # Xf_init = mp.Xf.copy()
    # Xm_init = mp.Xm.copy()

    # # (A) Dimention of the optimization problem is N-1
    # dim_reduced = N - 1
    # # Initial guess for the reduced problem
    # x0_guess = np.ones(dim_reduced)

    # # (B) Example of early stop with callback
    # best_x = [None]
    # res = None

    # iter_count = [0]

    # n = len(x0_guess)
    # eps = 1e-12  # 0に限りなく近い正の値を設定
    # bnds = [(eps, None)] * n  # 下限：eps, 上限：制限なし

    # def callback_func(xk):
    #     """Callback function to check the objective value and stop the optimization."""
    #     iter_count[0] += 1  # Increment the iteration counter
    #     val = objective_w_hat_reduced(
    #         xk, mp, bench_shocks, Xf_init, Xm_init, numeraire_index
    #     )
    #     # Print the current loss value for each iteration
    #     print(f"Iteration {iter_count[0]}: loss = {val}")
    #     threshold = 1e-6
    #     if val < threshold:
    #         best_x[0] = xk.copy()
    #         raise EarlyStopException(
    #             f"Residual {val} < threshold {threshold}. Early stopping."
    #         )

    # try:
    #     # (C) Optimize w_hat by using Nelder-Mead method
    #     res = minimize(
    #         objective_w_hat_reduced,
    #         x0_guess,
    #         args=(mp, bench_shocks, Xf_init, Xm_init, numeraire_index),
    #         # method="Nelder-Mead",
    #         method="L-BFGS-B",
    #         bounds=bnds,
    #         callback=callback_func,
    #         options={"maxiter": 10000, "disp": True},
    #     )
    # except EarlyStopException as e:
    #     print("Early stop triggered:", e)

    # # (D) Extract the solution: res is the official solution
    # if res is not None and hasattr(res, "x"):
    #     print("Optimization finished. Scipy result:")
    #     print(res)
    #     x_reduced_opt = res.x
    # else:
    #     # If res is None, use the best solution so far
    #     x_reduced_opt = best_x[0]

    # # (E) Reconstruct w_hat (numeraire is automatically set to 1)
    # w_hat_opt = reconstruct_w_hat(x_reduced_opt, numeraire_index, N)
    # print("Final wage changes (including numeraire=1):", w_hat_opt)

    # # (F) Calculate the equilibrium
    # Pm_init = np.ones((N, J))
    # c_hat, Pm_hat = solve_price_and_cost(
    #     w_hat_opt,
    #     Pm_init,
    #     mp,
    #     bench_shocks,
    #     max_iter=1000,
    #     tol=1e-7,
    #     mute=True,
    # )
    # Pf_hat = calc_Pu_hat(c_hat, "f", mp, bench_shocks)
    # pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, bench_shocks)
    # pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, bench_shocks)
    # Xf_prime, Xm_prime = calc_X(
    #     w_hat_opt, pif_hat, pim_hat, mp.td, mp, bench_shocks
    # )

    # # (G) Save the results
    # bench_sol = ModelSol(
    #     params=mp,
    #     shocks=bench_shocks,
    #     w_hat=w_hat_opt,
    #     c_hat=c_hat,
    #     Pf_hat=Pf_hat,
    #     Pm_hat=Pm_hat,
    #     pif_hat=pif_hat,
    #     pim_hat=pim_hat,
    #     Xf_prime=Xf_prime,
    #     Xm_prime=Xm_prime,
    # )
    # bench_sol.save_to_npz(f"{bench_dir}/numeraire1.npz")
    # print("Benchmark equilibrium saved.")

    bench_shocks = ModelShocks.load_from_npz(f"{bench_dir}/shocks.npz", mp)
    bench_sol = ModelSol.load_from_npz(
        f"{bench_dir}/numeraire1.npz", mp, bench_shocks
    )

    # =========================================================================
    # Step 3. Run counterfactuals
    # =========================================================================
    shocks_types = ["country", "sector", "idiosyncratic"]
    num_of_shocks = 10
    multipliers = [1, 2]
    sigma = 0.2
    counterfactual_dir = "output/counterfactual"

    for shock_type in shocks_types:
        # ---------------------------------------------------------------------
        # Generate lambda shocks
        # ---------------------------------------------------------------------
        if shock_type == "country":
            # For country-based shocks: one shock per country (shape: (num_of_shocks, N))
            # Then replicate it across sectors to get a final shape of (num_of_shocks, N, J)
            simulated_shocks = np.random.normal(
                loc=0.0, scale=sigma, size=(num_of_shocks, N)
            )
            lambda_hat_batch = np.exp(simulated_shocks)[:, :, np.newaxis]
            lambda_hat_batch = np.repeat(lambda_hat_batch, J, axis=2)
        elif shock_type == "sector":
            # For sector-based shocks: one shock per sector (shape: (num_of_shocks, J))
            # Then replicate it across countries to get a final shape of (num_of_shocks, N, J)
            simulated_shocks = np.random.normal(
                loc=0.0, scale=sigma, size=(num_of_shocks, J)
            )
            lambda_hat_batch = np.exp(simulated_shocks)[:, np.newaxis, :]
            lambda_hat_batch = np.repeat(lambda_hat_batch, N, axis=1)
        elif shock_type == "idiosyncratic":
            # For idiosyncratic shocks: independent shock for each (country, sector)
            # Shape is directly (num_of_shocks, N, J)
            simulated_shocks = np.random.normal(
                loc=0.0, scale=sigma, size=(num_of_shocks, N, J)
            )
            lambda_hat_batch = np.exp(simulated_shocks)

        # ---------------------------------------------------------------------
        # Run counterfactuals for different dm shocks
        # ---------------------------------------------------------------------
        for m in multipliers:
            # Generate dm shocks
            df_hat = np.ones((N, N, J))
            dm_hat = np.ones((N, N, J)) * m
            for i in range(N):
                for j in range(J):
                    dm_hat[i, i, j] = 1
            tilde_tau_prime = np.ones((N, N, J))  # No shocks on trade cost

            rel_path = "_".join((shock_type, f"dm_{m}"))
            dir = os.path.join(counterfactual_dir, rel_path)
            os.makedirs(dir, exist_ok=True)
            shock_list = []

            # Construct ModelShocks objects
            for i in range(num_of_shocks):
                lambda_hat = lambda_hat_batch[i, :, :]
                shock = ModelShocks(
                    mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime
                )
                shock_list.append(shock)

            # Run counterfactuals in parallel
            with ProcessPoolExecutor(
                max_workers=os.cpu_count() - 2,
                initializer=init_worker,
                initargs=(mp, bench_sol, numeraire_index),
            ) as executor:
                futures = [
                    executor.submit(run_counterfactual, shock)
                    for shock in shock_list
                ]
                for i, fut in enumerate(as_completed(futures)):
                    sol = fut.result()
                    # the save_start_idx is the number of existing results
                    save_start_idx = 0

                    shock_list[i].save_to_npz(
                        os.path.join(
                            dir, f"result_{i+save_start_idx}_shock.npz"
                        )
                    )
                    sol.save_to_npz(
                        os.path.join(dir, f"result_{i+save_start_idx}_sol.npz")
                    )

                    print(
                        f"Counterfactual equilibria for {shock_type} shock, dm = {m}, shock index {i} are saved."
                    )


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
