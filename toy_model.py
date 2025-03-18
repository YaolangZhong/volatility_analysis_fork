import os
import numpy as np
import time
from models import ModelShocks, ModelSol
from equations import calc_Pu_hat, calc_piu_hat
from solvers import solve_price_and_cost, solve_X_prime
from functions import generate_rand_params
from optimization import objective_w_hat, objective_w_hat_reduced
from scipy.optimize import minimize


class EarlyStopException(Exception):
    """Optimization early stop signal."""

    pass


def reconstruct_w_hat(x_reduced, numeraire_index, N):
    """
    長さ (N-1) のベクトル x_reduced を受け取り、
    長さ N の w_hat_full を返す。
    numeraire_index の要素は常に 1.0 に固定。
    """
    w_hat_full = np.zeros(N)
    idx_red = 0
    for i in range(N):
        if i == numeraire_index:
            w_hat_full[i] = 1.0  # numeraire 国を 1 に固定
        else:
            w_hat_full[i] = x_reduced[idx_red]
            idx_red += 1
    return w_hat_full


def callback_early_stop(
    xk, threshold, mp, shocks, Xf_init, Xm_init, numeraire_index
):
    """
    Callback function to check the objective value and stop the optimization
    if the value is below the threshold.
    """
    # Calculate the objective value
    val = objective_w_hat(xk, mp, shocks, Xf_init, Xm_init, numeraire_index)

    if val < threshold:
        raise EarlyStopException(
            f"Residual {val} < threshold {threshold}. Early stopping."
        )


def main():
    # =========================================================================
    # Step 1. Setup and generate random parameters
    out_dir = "toymodel_output"

    # Change the working directory to the current file's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create directories to store the results
    os.makedirs(out_dir, exist_ok=True)

    N, J = 5, 3

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

    # ----------------------------------------------------
    # Solve for the benchmark equilibrium
    # by minimizing the difference between the model and data trade deficit
    # ----------------------------------------------------

    # Initialize Xf and Xm by the value in the data
    Xf_init = mp.Xf.copy()
    Xm_init = mp.Xm.copy()

    res = None  # Variable to store the optimization result
    best_x = [None]  # Variable to store the best solution so far

    def callback_early_stop(
        xk, threshold, mp, shocks, Xf_init, Xm_init, numeraire_index
    ):
        val = objective_w_hat(
            xk, mp, shocks, Xf_init, Xm_init, numeraire_index
        )
        if val < threshold:
            best_x[0] = xk.copy()  # Save the best solution so far
            raise EarlyStopException(
                f"Residual {val} < threshold {threshold}. Early stopping."
            )

    # (a) Initialize w_hat
    w0_guess = np.ones(N)
    w0_guess[numeraire_index] = 1.0  # Wage rate of the numeraire country

    callback_threshold = 0  # Threshold to stop the optimization

    def callback_func(xk):
        # (b) Callback function to check the objective value
        # and stop the optimization
        callback_early_stop(
            xk,
            callback_threshold,
            mp,
            shocks,
            Xf_init,
            Xm_init,
            numeraire_index,
        )

    # (c) Optimize w_hat
    try:
        res = minimize(
            objective_w_hat,
            w0_guess,
            args=(mp, shocks, Xf_init, Xm_init, numeraire_index),
            method="Nelder-Mead",
            callback=callback_func,
            options={
                "maxiter": 10000,
                "disp": True,
            },
        )
    except EarlyStopException as e:
        print("Early stop triggered:", e)

    if res is not None:
        print("Optimization result:", res)
        w_hat_opt = res.x  # Optimal wage rate changes
    else:
        print("Optimization stopped. Using the best solution so far.")
        print("best_x so far:", best_x[0])
        w_hat_opt = best_x[0]

    # (d) Calculate the equilibrium prices and quantities
    Pm_init = np.ones((N, J))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_opt, Pm_init, mp, shocks, max_iter=1000, tol=1e-7, mute=True
    )
    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)
    Xf_prime, Xm_prime = solve_X_prime(
        w_hat_opt,
        pif_hat,
        pim_hat,
        mp.td,
        Xf_init,
        Xm_init,
        mp,
        shocks,
        max_iter=1000,
        tol=1e-7,
        mute=True,
    )

    # (e) Construct ModelSol object
    bench_sol = ModelSol(
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

    # Save the results
    if bench_sol is not None:
        bench_sol.save_to_npz(f"{bench_dir}/equilibrium.npz")
        print("Benchmark equilibrium saved.")
    else:
        print("Failed to build benchmark solution")


def main_two():
    # =========================================================================
    # Step 1. Setup and generate random parameters
    out_dir = "toymodel_output"
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out_dir, exist_ok=True)

    N, J = 5, 3
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

    numeraire_index = 0

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

    # ----------------------------------------------------
    # Setup for optimization
    # ----------------------------------------------------
    Xf_init = mp.Xf.copy()
    Xm_init = mp.Xm.copy()

    # (A) 最適化の変数次元は N-1
    dim_reduced = N - 1
    # 適当な初期値（すべて1）
    x0_guess = np.ones(dim_reduced)

    # (B) コールバックで early stop したい場合の例
    best_x = [None]
    res = None

    def callback_func(xk):
        """例: 残差がある閾値以下なら途中終了させる."""
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
        # (C) Nelder-Mead で (N-1) 次元の問題を最適化
        res = minimize(
            objective_w_hat_reduced,
            x0_guess,
            args=(mp, shocks, Xf_init, Xm_init, numeraire_index),
            method="Nelder-Mead",
            callback=callback_func,
            options={"maxiter": 10000, "disp": True},
        )
    except EarlyStopException as e:
        print("Early stop triggered:", e)

    # (D) 解の取り出し: res が None でなければ official な解
    if res is not None and hasattr(res, "x"):
        print("Optimization finished. Scipy result:")
        print(res)
        x_reduced_opt = res.x
    else:
        # early stop, best_x[0] に記録
        x_reduced_opt = best_x[0]

    # (E) 最終的な w_hat を復元（numeraire は自動的に1）
    w_hat_opt = reconstruct_w_hat(x_reduced_opt, numeraire_index, N)
    print("Final wage changes (including numeraire=1):", w_hat_opt)

    # (F) 均衡を計算
    Pm_init = np.ones((N, J))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_opt, Pm_init, mp, shocks, max_iter=1000, tol=1e-7, mute=True
    )
    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)
    Xf_prime, Xm_prime = solve_X_prime(
        w_hat_opt,
        pif_hat,
        pim_hat,
        mp.td,
        Xf_init,
        Xm_init,
        mp,
        shocks,
        max_iter=1000,
        tol=1e-7,
        mute=True,
    )

    # (G) 結果を保存
    bench_sol = ModelSol(
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
    bench_sol.save_to_npz(f"{bench_dir}/equilibrium.npz")
    print("Benchmark equilibrium saved.")


if __name__ == "__main__":
    # Start time of the script
    start_time = time.time()
    np.seterr(over="raise")

    main_two()

    # End time of the script
    end_time = time.time()

    # Format the elapsed time to mm:ss format
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    print(f"Elapsed time: {elapsed_time_str}")
