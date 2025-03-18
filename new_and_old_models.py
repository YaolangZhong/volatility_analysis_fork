import os
import numpy as np
import time
from scipy.optimize import minimize

from models import ModelShocks, ModelSol
from equations import calc_Pu_hat, calc_piu_hat
from solvers import solve_price_and_cost, solve_X_prime
from functions import generate_rand_params_without_usage
from optimization import objective_w_hat_reduced
from toy_model import (
    EarlyStopException,
    callback_early_stop,
    reconstruct_w_hat,
)

from original_cp.original_models import (
    OldModelParams,
    OldModelShocks,
    OldModelSol,
)
from original_cp.original_equations import (
    equilibrium,
)


def main():
    # =========================================================================
    # Step 1. Setup and generate random parameters without usage
    out_dir = "new_and_old_models_output"

    # Change the working directory to the current file's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create directories to store the results
    os.makedirs(out_dir, exist_ok=True)

    N, J = 5, 3

    # Generate random parameters
    mp = generate_rand_params_without_usage(N, J)
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

    # =========================================================================
    # Step 3. Solve for the equilibrium using the original model
    orig_dir = f"{out_dir}/original_model"
    os.makedirs(orig_dir, exist_ok=True)

    # Convert the new model's parameters to the original model's parameters
    omp = OldModelParams(
        N=mp.N,
        J=mp.J,
        alpha=mp.alpha,
        beta=mp.beta,
        gamma=mp.gamma,
        theta=mp.theta,
        pi=mp.pif,
        tilde_tau=mp.tilde_tau,
        X=mp.Xf + mp.Xm,
        VA=mp.w0 * mp.L0,
        D=-mp.td,
    )

    omp.check_consistency()

    # ----------------------------------------------------
    # Solve for the equilibrium using the original model
    # ----------------------------------------------------
    oshocks = OldModelShocks(omp, np.ones((N, N, J)))
    oshocks.check_consistency()

    # Solve for the equilibrium
    osol = equilibrium(omp, oshocks, numeraire_index, omp.X, vfactor=-0.2)

    # Print and save the results
    if osol is not None:
        print(f"w_hat for the original model: {osol.w_hat}")
        osol.save_to_npz(f"{orig_dir}/equilibrium.npz")
        print("Original model's equilibrium saved.")
    else:
        print("Failed to build solution")


if __name__ == "__main__":
    # Start time of the script
    start_time = time.time()
    np.seterr(over="raise")

    main()

    # End time of the script
    end_time = time.time()

    # Format the elapsed time to mm:ss format
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
    print(f"Elapsed time: {elapsed_time_str}")
