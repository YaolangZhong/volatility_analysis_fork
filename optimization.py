import numpy as np
from solvers import solve_price_and_cost, solve_X_prime
from equations import (
    calc_Pu_hat,
    calc_piu_hat,
    calc_td_prime,
)


def objective_w_hat(w_hat, mp, shocks, Xf_init, Xm_init, numeraire_index=0):
    """
    Function to calculate the difference
    between the model and data trade deficit and return its squared sum.
    Corresponds to equation (13) in the paper

    Set the wage rate of the numeraire country to 1.0
    """
    # 0. Fix wage rate of the numeraire country to 1.0
    w_hat_mod = w_hat.copy()
    w_hat_mod[numeraire_index] = 1.0

    # -------------------------------------------------------------
    # 1. Calculate c_hat, Pm_hat, Pf_hat, pif_hat, pim_hat
    # -------------------------------------------------------------
    N, J = mp.N, mp.J
    Pm_init = np.ones((N, J))

    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_mod,
        Pm_init,
        mp,
        shocks,
        max_iter=500,
        tol=1e-6,
        mute=True,
    )

    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)

    # -------------------------------------------------------------
    # 2. Calculate Xf_prime, Xm_prime
    # -------------------------------------------------------------
    Xf_prime, Xm_prime = solve_X_prime(
        w_hat_mod,
        pif_hat,
        pim_hat,
        mp.td,
        Xf_init,
        Xm_init,
        mp,
        shocks,
        max_iter=500,
        tol=1e-6,
        mute=True,
    )

    # -------------------------------------------------------------
    # 3. Calculate trade deficit and value added
    # -------------------------------------------------------------
    td_prime = calc_td_prime(pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks)

    VA_prime = np.sum(mp.w0 * mp.L0 * w_hat_mod)
    VA0 = np.sum(mp.w0 * mp.L0)

    diff = td_prime / VA_prime - mp.td / VA0

    # -------------------------------------------------------------
    # 4. Return the squared sum of the differences
    # -------------------------------------------------------------
    return np.sum(diff**2)


def objective_w_hat_reduced(
    x_reduced, mp, shocks, Xf_init, Xm_init, numeraire_index
):
    """
    numeraire 国を除いた (N-1) 次元の w_hat ベクトル (x_reduced) を受け取り、
    フルサイズの w_hat を再構築 → (13) の残差（二乗）を返す。
    """
    # 1. numeraire 国を 1 に固定してフルベクトルに
    w_hat_full = reconstruct_w_hat(x_reduced, numeraire_index, mp.N)

    # 2. solve_price_and_cost 等を呼んで内生変数を計算
    N, J = mp.N, mp.J
    Pm_init = np.ones((N, J))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat_full, Pm_init, mp, shocks, max_iter=500, tol=1e-6, mute=True
    )
    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
    pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
    pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)

    # 3. Xf_prime, Xm_prime 計算
    Xf_prime, Xm_prime = solve_X_prime(
        w_hat_full,
        pif_hat,
        pim_hat,
        mp.td,
        Xf_init,
        Xm_init,
        mp,
        shocks,
        max_iter=500,
        tol=1e-6,
        mute=True,
    )

    # 4. 貿易収支の残差を計算
    td_prime = calc_td_prime(pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks)
    VA_prime = np.sum(mp.w0 * mp.L0 * w_hat_full)
    VA0 = np.sum(mp.w0 * mp.L0)

    diff = td_prime / VA_prime - mp.td / VA0
    return np.max(abs(diff))


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
