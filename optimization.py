import numpy as np
from solvers import solve_price_and_cost, solve_X_prime
from equations import (
    calc_Pu_hat,
    calc_piu_hat,
    calc_td_prime,
)
from equations_matrix import calc_X


# Old objective function (no longer used)
# def objective_w_hat(w_hat, mp, shocks, Xf_init, Xm_init, numeraire_index=0):
#     """
#     Function to calculate the difference
#     between the model and data trade deficit and return its squared sum.
#     Corresponds to equation (13) in the paper

#     Set the wage rate of the numeraire country to 1.0
#     """
#     # 0. Fix wage rate of the numeraire country to 1.0
#     w_hat_mod = w_hat.copy()
#     w_hat_mod[numeraire_index] = 1.0

#     # -------------------------------------------------------------
#     # 1. Calculate c_hat, Pm_hat, Pf_hat, pif_hat, pim_hat
#     # -------------------------------------------------------------
#     N, J = mp.N, mp.J
#     Pm_init = np.ones((N, J))

#     c_hat, Pm_hat = solve_price_and_cost(
#         w_hat_mod,
#         Pm_init,
#         mp,
#         shocks,
#         max_iter=500,
#         tol=1e-6,
#         mute=True,
#     )

#     Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
#     pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
#     pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)

#     # -------------------------------------------------------------
#     # 2. Calculate Xf_prime, Xm_prime
#     # -------------------------------------------------------------
#     Xf_prime, Xm_prime = solve_X_prime(
#         w_hat_mod,
#         pif_hat,
#         pim_hat,
#         mp.td,
#         Xf_init,
#         Xm_init,
#         mp,
#         shocks,
#         max_iter=500,
#         tol=1e-6,
#         mute=True,
#     )

#     # -------------------------------------------------------------
#     # 3. Calculate trade deficit and value added
#     # -------------------------------------------------------------
#     td_prime = calc_td_prime(pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks)

#     VA_prime = np.sum(mp.w0 * mp.L0 * w_hat_mod)
#     VA0 = np.sum(mp.w0 * mp.L0)

#     diff = td_prime / VA_prime - mp.td / VA0

#     # -------------------------------------------------------------
#     # 4. Return the squared sum of the differences
#     # -------------------------------------------------------------
#     return np.max(abs(diff))  # return the maximum difference


def objective_w_hat_reduced(
    x_reduced, mp, shocks, Xf_init, Xm_init, numeraire_index
):
    """
    Function to calculate the difference
    between the model and data trade deficit and return its squared sum.
    """
    # 1. reconstruct w_hat_full
    w_hat_full = reconstruct_w_hat(x_reduced, numeraire_index, mp.N)

    # 2. Calculate endogenous variables for given w_hat_full
    N, J = mp.N, mp.J
    Pm_init = np.ones((N, J))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_full, Pm_init, mp, shocks, max_iter=500, tol=1e-6, mute=True
    )
    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)

    # 3. Calculate Xf_prime, Xm_prime
    Xf_prime, Xm_prime = calc_X(
        w_hat_full, pif_hat, pim_hat, mp.td, mp, shocks
    )

    # 4. Calculate trade deficit and value added
    td_prime = calc_td_prime(pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks)
    VA_prime = np.sum(mp.w0 * mp.L0 * w_hat_full)
    VA0 = np.sum(mp.w0 * mp.L0)

    diff = td_prime / VA_prime - mp.td / VA0
    return np.max(abs(diff))  # return the maximum difference


def reconstruct_w_hat(x_reduced, numeraire_index, N):
    """
    Function to receive a vector x_reduced of length (N-1) and
    and return a vector w_hat_full of length N.
    The element of numeraire_index is always fixed to 1.0.
    """
    w_hat_full = np.zeros(N)
    idx_red = 0
    for i in range(N):
        if i == numeraire_index:
            w_hat_full[i] = 1.0  # Fix the numeraire country to 1
        else:
            w_hat_full[i] = x_reduced[idx_red]
            idx_red += 1
    return w_hat_full
