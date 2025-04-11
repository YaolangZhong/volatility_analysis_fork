import os
import numpy as np
import time
from models import ModelParams, ModelShocks, ModelSol
from equations import calc_Pu_hat, calc_piu_hat, calc_W
from equations_autograd import calc_X
from solvers import solve_price_and_cost
from functions import generate_rand_params, generate_simple_params
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

    iter_count = [0]

    def callback_func(xk):
        """Callback function to check the objective value and stop the optimization."""
        iter_count[0] += 1  # Increment the iteration counter
        val = objective_w_hat_reduced(
            xk, mp, shocks, Xf_init, Xm_init, numeraire_index
        )
        # Print the current loss value for each iteration
        print(f"Iteration {iter_count[0]}: loss = {val}")
        threshold = -1
        if val < threshold:
            best_x[0] = xk.copy()
            raise EarlyStopException(
                f"Residual {val} < threshold {threshold}. Early stopping."
            )

    eps = 1e-12  # 0に限りなく近い正の値を設定
    bnds = [(eps, None)] * (N - 1)  # 下限：eps, 上限：制限なし

    try:
        # (C) Optimize w_hat by using Nelder-Mead method
        res = minimize(
            objective_w_hat_reduced,
            x0_guess,
            args=(mp, shocks, Xf_init, Xm_init, numeraire_index),
            method="L-BFGS-B",
            bounds=bnds,
            callback=callback_func,
            options={"maxiter": 10000, "disp": True},
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

    shocks.save_to_npz(f"{out_dir}/shocks_{i}.npz")
    sol.save_to_npz(f"{out_dir}/counterfactual_{i}.npz")

    real_wage = calc_W(sol)
    print("Real wage changes:", real_wage)
    return f"Counterfactual equilibrium {i} saved."


def main():
    # =========================================================================
    # Step 1. Setup and load parameters
    out_dir = "test_output"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)

    # ========== For now, generate random parameters ==========
    N, J = 2, 1
    mp = generate_simple_params()
    if mp is None:
        print("Failed to generate random parameters")
        return None
    else:
        mp.save_to_npz(f"{out_dir}/model_params.npz")
    # ===== Replace this part with loading parameters from a file =====
    # data = np.load("real_data.npz")
    # N, J = data["N"], data["J"]
    # mp = ModelParams(
    #     N=N,
    #     J=J,
    #     alpha=data["alpha"],
    #     beta=data["beta"],
    #     gamma=data["gamma"],
    #     theta=data["theta"],
    #     pif=data["pi_f"],
    #     pim=data["pi_m"],
    #     tilde_tau=data["tilde_tau"],
    #     Xf=np.ones((N, J)),
    #     Xm=np.ones((N, J)),
    #     w0=data["VA"],
    #     L0=np.ones_like(data["VA"]),
    #     td=data["D"],
    # )
    # print("Loaded the parameters from the real data")
    mp.save_to_npz(f"{out_dir}/params.npz")
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
        bench_shocks.save_to_npz(f"{bench_dir}/shocks.npz")
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

    iter_count = [0]

    def callback_func(xk):
        """Callback function to check the objective value and stop the optimization."""
        iter_count[0] += 1  # Increment the iteration counter
        val = objective_w_hat_reduced(
            xk, mp, bench_shocks, Xf_init, Xm_init, numeraire_index
        )
        # Print the current loss value for each iteration
        print(f"Iteration {iter_count[0]}: loss = {val}")
        threshold = -1
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
            # method="L-BFGS-B",
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
    bench_sol.save_to_npz(f"{bench_dir}/numeraire1.npz")
    real_wage = calc_W(bench_sol)
    print("Real wage changes:", real_wage)
    print("Benchmark equilibrium saved.")

    # bench_sol = ModelSol.load_from_npz(f"{bench_dir}/equilibrium.npz", mp, bench_shocks)

    # # =========================================================================
    # # Step 3. Solve for counterfactual equilibria
    # num_of_shocks = 100

    # # ========== For now, generate random shocks ==========
    # shock_list = []
    # for i in range(num_of_shocks):
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
    #     for i in range(num_of_shocks):
    #         fut = executor.submit(
    #             run_counterfactual, i + 1, out_dir, shock_list[i]
    #         )
    #         futures.append(fut)

    #     for fut in as_completed(futures):
    #         print(fut.result())

    # print("All counterfactual equilibria are solved.")

    # =========================================================================
    # Step. 4 Run simulations for different sigmas

    num_of_shocks = 1
    multipliers = [1, 2]
    sigma = 0
    for i in range(num_of_shocks):
        lambda_hat = np.exp(
            np.random.normal(loc=0.0, scale=sigma, size=(N, J))
        )
        for m in multipliers:
            multiplier_dir = f"{out_dir}/d_{m}"
            os.makedirs(multiplier_dir, exist_ok=True)
            shock_list = []
            df_hat = np.ones((N, N, J))
            dm_hat = np.ones((N, N, J)) * m
            for i in range(N):
                for j in range(J):
                    dm_hat[i, i, j] = 1
            tilde_tau_prime = np.ones((N, N, J))  # No shocks on tariffs
            shock_list.append(
                ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)
            )

            with ProcessPoolExecutor(
                max_workers=os.cpu_count() - 2,
                initializer=init_worker,
                initargs=(mp, bench_sol, numeraire_index),
            ) as executor:
                futures = []
                for i in range(num_of_shocks):
                    fut = executor.submit(
                        run_counterfactual,
                        i + 1,
                        multiplier_dir,
                        shock_list[i],
                    )
                    futures.append(fut)

                for fut in as_completed(futures):
                    print(fut.result())

        print(f"All counterfactual equilibria for multiplier = {m} are saved.")


def print_counterfactuals():
    def print_npz(file_path):
        with np.load(file_path) as data:
            for key, array in data.items():
                print(f"Array '{key}' in {file_path}:")
                print(array)
                print()

    file1 = r"output\d_1\counterfactual_1.npz"
    file2 = r"output\d_2\counterfactual_1.npz"

    print_npz(file1)
    print_npz(file2)


def calculate_real_wage_from_npz(mp_path, shocks_path, sol_path):
    mp = ModelParams.load_from_npz(mp_path)
    shocks = ModelShocks.load_from_npz(shocks_path, mp)
    sol = ModelSol.load_from_npz(sol_path, mp, shocks)
    real_wage = calc_W(sol)
    print(f"Real wage changes from {sol_path}: {real_wage}")

    return real_wage


if __name__ == "__main__":
    start_time = time.time()
    np.seterr(over="raise")

    # main()
    print_counterfactuals()
    real_wage_1 = calculate_real_wage_from_npz(
        r"output\model_params.npz",
        r"output\d_1\shocks_1.npz",
        r"output\d_1\counterfactual_1.npz",
    )
    real_wage_2 = calculate_real_wage_from_npz(
        r"output\model_params.npz",
        r"output\d_2\shocks_1.npz",
        r"output\d_2\counterfactual_1.npz",
    )

    print(f"Real wage changes ratio: {real_wage_2 / real_wage_1}")

    # End time of the script
    end_time = time.time()

    # Format the elapsed time to mm:ss format
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    print(f"Elapsed time: {elapsed_time_str}")
