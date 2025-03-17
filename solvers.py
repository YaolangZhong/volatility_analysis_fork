import numpy as np
from models import ModelParams, ModelShocks, ModelSol
from equations import (
    calc_c_hat,
    calc_Pu_hat,
    calc_piu_hat,
    calc_Xf_prime,
    calc_Xm_prime,
    calc_td_prime,
)


def solve_price_and_cost(
    w_hat,
    Pm_hat_init,
    mp: ModelParams,
    shocks: ModelShocks,
    max_iter=1000,
    tol=1e-6,
    mute=True,
):
    """
    Solve for the price index changes of intermediate goods
    """
    Pm_hat = Pm_hat_init.copy()
    for i in range(max_iter):
        c_hat = calc_c_hat(w_hat, Pm_hat, mp)
        Pm_hat_new = calc_Pu_hat(c_hat, "m", mp, shocks)
        diff = np.max(np.abs(Pm_hat_new - Pm_hat))
        if diff < tol:
            if not mute:
                print(f"Pm_hat converged in {i+1} iterations")
            c_hat = calc_c_hat(w_hat, Pm_hat_new, mp)
            break
        Pm_hat = Pm_hat_new

    return c_hat, Pm_hat


def solve_X_prime(
    w_hat,
    pif_hat,
    pim_hat,
    td_prime,
    Xf_init,
    Xm_init,
    mp: ModelParams,
    shocks: ModelShocks,
    max_iter=1000,
    tol=1e-6,
    mute=True,
):
    """
    Solve for Xf_prime and Xm_prime
    given w_hat, pif_hat, pim_hat, tilde_tau_prime, td_prime

    Arguments for the model:
        w_hat: (N,) array of wage rate changes
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        tilde_tau_prime: (N, N, J) array of tariff after the shock
        td_prime: (N, N, J) array of trade deficit after the shock
        mp: ModelParams object containing model parameters
    Arguments for the solver:
        Xf_init: (N, J) array of initial guess for Xf_prime
        Xm_init: (N, J) array of initial guess for Xm_prime
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        mute: whether to print the progress of the solver
    Returns:
        Xf_prime: (N, J) array of final goods expenditure after the shock
        Xm_prime: (N, J) array of intermediate goods expenditure after the shock
    """
    Xf_prime = Xf_init.copy()
    Xm_prime = Xm_init.copy()
    for i in range(max_iter):
        Xf_prime_new = calc_Xf_prime(
            w_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, td_prime, mp, shocks
        )
        Xm_prime_new = calc_Xm_prime(
            pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks
        )
        diff_Xf = np.max(np.abs(Xf_prime_new - Xf_prime))
        diff_Xm = np.max(np.abs(Xm_prime_new - Xm_prime))
        if diff_Xf < tol and diff_Xm < tol:
            if not mute:
                print(f"X_prime converged in {i+1} iterations")
            break
        Xf_prime = Xf_prime_new
        Xm_prime = Xm_prime_new

    return Xf_prime, Xm_prime


def solve_equilibrium(
    mp: ModelParams,
    shocks: ModelShocks,
    numeraire_index: int,
    Xf_init: np.ndarray,
    Xm_init: np.ndarray,
    vfactor=-0.2,
    tol=1e-3,
    max_iter=1000,
    mute=True,
):
    """
    Solve for the equilibrium of the model

    Arguments:
        mp: ModelParams object containing model parameters
        shocks: ModelShocks object containing model shocks
        numeraire_index: index of the numeraire country
        Xf_init: (N, J) array of initial guess for Xf_prime
        Xm_init: (N, J) array of initial guess for Xm_prime
        vfactor: adjustment factor for the iteration process
        tol: tolerance for convergence
        max_iter: maximum number of iterations
        mute: whether to print the progress of the solver
    Returns:
        sol: ModelSol object
            containing the equilibrium values of endogenous variables
    """
    # Set the adjustment factor for the iteration process
    alpha = 0.1

    # Initialize the variables
    N, J = mp.N, mp.J
    VA = np.sum(mp.w0 * mp.L0)

    # Initlial guess for wage rate changes
    w_hat = np.ones(N)

    # Initial guess for the price index changes of intermediate goods
    Pm_hat = np.ones((N, J))

    # Initialize trade deficit
    td_prime = mp.td

    wfmax = 1.0

    for i in range(max_iter):
        # Calculate endogenous variables
        c_hat, Pm_hat = solve_price_and_cost(
            w_hat, Pm_hat, mp, shocks, max_iter=1000, tol=1e-6, mute=True
        )
        Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
        pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
        pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)
        Xf_prime, Xm_prime = solve_X_prime(
            w_hat,
            pif_hat,
            pim_hat,
            td_prime,
            Xf_init,
            Xm_init,
            mp,
            shocks,
            max_iter=1000,
            tol=1e-6,
            mute=True,
        )

        # Calculate the trade deficit
        td_prime = calc_td_prime(
            pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks
        )

        # ZW2 captures the difference
        # between the calculated trade deficit and the target trade deficit
        # normalized by the initial value added

        VA_prime = np.sum(mp.w0 * mp.L0 * w_hat)

        ZW2 = td_prime / VA_prime - mp.td / VA

        # Update the wage changes (numeraire country is excluded)
        w_hat_new = np.ones(N)
        w_hat_new = w_hat * np.exp(-alpha * ZW2)

        # w_hat_new = np.clip(w_hat_new, 1e-3, 1e3)

        # Check convergence (exclude numeraire country)
        wfmax = np.max(np.abs(w_hat_new - w_hat))

        # Update wage changes for the next iteration
        w_hat = w_hat_new / w_hat_new[numeraire_index]

        min_Xf_prime = np.min(Xf_prime)
        max_Xf_prime = np.max(Xf_prime)
        min_Xm_prime = np.min(Xm_prime)
        max_Xm_prime = np.max(Xm_prime)

        min_w_hat = np.min(w_hat)
        max_w_hat = np.max(w_hat)

        if not mute:
            print(
                f"Round {i+1}: wfmax={wfmax:.4f}, min_w_hat={min_w_hat:.4f}, max_w_hat={max_w_hat:.4f}, min_Xf_prime={min_Xf_prime:.4f}, max_Xf_prime={max_Xf_prime:.4f}, min_Xm_prime={min_Xm_prime:.4f}, max_Xm_prime={max_Xm_prime:.4f}"
            )

        if wfmax < tol:
            if not mute:
                print(f"Converged in {i+1} iterations")
            # Return the ModelSol object
            return ModelSol(
                mp,
                shocks,
                w_hat,
                c_hat,
                Pf_hat,
                Pm_hat,
                pif_hat,
                pim_hat,
                Xf_prime,
                Xm_prime,
            )

    if not mute:
        print("Failed to converge")
    return None
