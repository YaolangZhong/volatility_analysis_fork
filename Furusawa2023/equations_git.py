import numpy as np
from numba import njit
from models import ModelParams, ModelShocks

@njit(cache=True)
def calc_c_hat_jit(
    w_hat: np.ndarray,      # shape (N,)
    Pm_hat: np.ndarray,     # shape (N, J)
    beta: np.ndarray,       # shape (N, J)
    gamma: np.ndarray       # shape (N, J, J)
) -> np.ndarray:
    """
    JIT version of calc_c_hat (Equation 7).

    Parameters
    ----------
    w_hat   : (N,)   Wage change vector
    Pm_hat  : (N,J)  Intermediate‐goods price index changes
    beta    : (N,J)  Labor‐share parameters
    gamma   : (N,J,J) Intermediate input‐share parameters

    Returns
    -------
    c_hat   : (N,J)  Unit‐cost index changes
    """
    N, J = w_hat.shape[0], Pm_hat.shape[1]
    c_hat = np.empty((N, J))
    # precompute logs
    log_w  = np.log(w_hat)
    log_Pm = np.log(Pm_hat)

    for n in range(N):
        for j in range(J):
            # wage component: beta[n,j] * log_w[n]
            wc = beta[n, j] * log_w[n]
            # input component: sum_k gamma[n,j,k] * log_Pm[n,k]
            ic = 0.0
            for k in range(J):
                ic += gamma[n, j, k] * log_Pm[n, k]
            c_hat[n, j] = np.exp(wc + ic)

    return c_hat


@njit(cache=True)
def calc_price_index_jit(
    N: int,
    J: int,
    theta: np.ndarray,          # shape (J,)
    pi: np.ndarray,             # shape (N, N, J)
    lambda_hat: np.ndarray,     # shape (N, J)
    c_hat: np.ndarray,          # shape (N, J)
    d_hat: np.ndarray           # shape (N, N, J)
) -> np.ndarray:
    """
    JIT version of calc_price_index (Equation 8).
    """
    P_hat = np.empty((N, J))
    for n in range(N):
        for j in range(J):
            acc = 0.0
            inv_theta = -theta[j]
            for i in range(N):
                acc += pi[n, i, j] \
                     * lambda_hat[n, j] \
                     * (c_hat[n, j] * d_hat[n, i, j])**inv_theta
            P_hat[n, j] = acc ** (-1.0 / theta[j])
    return P_hat


@njit(cache=True)
def calc_Pf_hat_jit(
    N: int,
    J: int,
    theta: np.ndarray,          # shape (J,)
    pif: np.ndarray,            # shape (N, N, J)
    lambda_hat: np.ndarray,     # shape (N, J)
    c_hat: np.ndarray,          # shape (N, J)
    df_hat: np.ndarray          # shape (N, N, J)
) -> np.ndarray:
    """
    JIT wrapper for final–goods price index P̂ᶠ (uses calc_price_index_jit).
    """
    return calc_price_index_jit(N, J, theta, pif, lambda_hat, c_hat, df_hat)


@njit(cache=True)
def calc_Pm_hat_jit(
    N: int,
    J: int,
    theta: np.ndarray,          # shape (J,)
    pim: np.ndarray,            # shape (N, N, J)
    lambda_hat: np.ndarray,     # shape (N, J)
    c_hat: np.ndarray,          # shape (N, J)
    dm_hat: np.ndarray          # shape (N, N, J)
) -> np.ndarray:
    """
    JIT wrapper for intermediate–goods price index P̂ᵐ (uses calc_price_index_jit).
    """
    return calc_price_index_jit(N, J, theta, pim, lambda_hat, c_hat, dm_hat)


@njit(cache=True)
def calc_expenditure_share_jit(
    N: int,
    J: int,
    theta: np.ndarray,        # shape (J,)
    lambda_hat: np.ndarray,   # shape (N, J)
    c_hat: np.ndarray,        # shape (N, J)
    P_hat: np.ndarray,        # shape (N, J)
    d_hat: np.ndarray         # shape (N, N, J)
) -> np.ndarray:
    """
    Equation (9) JIT: Expenditure‐share changes π̂ given cost‐ and price‐index changes.
    """
    piu = np.empty((N, N, J))
    for n in range(N):
        for i in range(N):
            for j in range(J):
                num = lambda_hat[n, j] * (c_hat[n, j] * d_hat[n, i, j]) ** (-theta[j])
                den = P_hat[n, j] ** (-theta[j])
                piu[n, i, j] = num / den
    return piu

@njit(cache=True)
def calc_pif_hat_jit(
    N: int,
    J: int,
    theta: np.ndarray,        # shape (J,)
    lambda_hat: np.ndarray,   # shape (N, J)
    c_hat: np.ndarray,        # shape (N, J)
    Pf_hat: np.ndarray,       # shape (N, J)
    df_hat: np.ndarray        # shape (N, N, J)
) -> np.ndarray:
    """
    Equation (9) JIT wrapper: final‐goods expenditure‐share changes π̂ᶠ.
    """
    return calc_expenditure_share_jit(N, J, theta, lambda_hat, c_hat, Pf_hat, df_hat)

@njit(cache=True)
def calc_pim_hat_jit(
    N: int,
    J: int,
    theta: np.ndarray,        # shape (J,)
    lambda_hat: np.ndarray,   # shape (N, J)
    c_hat: np.ndarray,        # shape (N, J)
    Pm_hat: np.ndarray,       # shape (N, J)
    dm_hat: np.ndarray        # shape (N, N, J)
) -> np.ndarray:
    """
    Equation (9) JIT wrapper: intermediate‐goods expenditure‐share changes π̂ᵐ.
    """
    return calc_expenditure_share_jit(N, J, theta, lambda_hat, c_hat, Pm_hat, dm_hat)






@njit(cache=True)
def _build_A_jit(
    N: int,
    J: int,
    w_hat: np.ndarray,    # (N,)
    td_prime: np.ndarray, # (N,)
    VA: np.ndarray,       # (N,)
    alpha: np.ndarray     # (N, J)
) -> np.ndarray:
    # produce a length‐2*N*J vector [vec(Af); vec(Am)]
    NJ = N * J
    A = np.empty(2 * NJ)
    idx = 0
    for n in range(N):
        base = w_hat[n] * VA[n] + td_prime[n]
        for j in range(J):
            A[idx] = alpha[n, j] * base
            idx += 1
    # Am is zero
    for _ in range(NJ):
        A[idx] = 0.0
        idx += 1
    return A

@njit(cache=True)
def _build_B_jit(
    N: int,
    J: int,
    alpha: np.ndarray,          # (N, J)
    gamma: np.ndarray,          # (N, J, J)
    pif: np.ndarray,            # (N, N, J)
    pim: np.ndarray,            # (N, N, J)
    tilde_tau: np.ndarray,      # (N, N, J)
    pif_hat: np.ndarray,        # (N, N, J)
    pim_hat: np.ndarray         # (N, N, J)
) -> np.ndarray:
    NJ  = N * J
    size = 2 * NJ
    B   = np.empty((size, size))
    # top blocks
    for r in range(NJ):
        n, j = divmod(r, J)
        v = alpha[n, j]
        for c in range(NJ):
            i, k = divmod(c, J)
            fac = (tilde_tau[n, i, k] - 1.0) / tilde_tau[n, i, k]
            B[r, c]    = v * fac * pif_hat[n, i, k] * pif[n, i, k]
            B[r, NJ+c] = v * fac * pim_hat[n, i, k] * pim[n, i, k]
    # bottom blocks
    for r in range(NJ, size):
        out = r - NJ
        n, s = divmod(out, J)
        for c in range(NJ):
            i, k = divmod(c, J)
            fac = tilde_tau[n, i, k]
            pf  = pif_hat[n, i, k] * pif[n, i, k] / fac
            pm  = pim_hat[n, i, k] * pim[n, i, k] / fac
            B[r, c]    = gamma[n, k, s] * pf
            B[r, NJ+c] = gamma[n, k, s] * pm
    return B

@njit(cache=True)
def calc_X_jit(
    N: int,
    J: int,
    w_hat: np.ndarray,
    pif_hat: np.ndarray,
    pim_hat: np.ndarray,
    td_prime: np.ndarray,
    VA: np.ndarray,
    alpha: np.ndarray,
    gamma: np.ndarray,
    pif: np.ndarray,
    pim: np.ndarray,
    tilde_tau: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    NJ   = N * J
    size = 2 * NJ

    # 1) build A, B
    A = _build_A_jit(N, J, w_hat, td_prime, VA, alpha)
    B = _build_B_jit(N, J, alpha, gamma, pif, pim, tilde_tau, pif_hat, pim_hat)

    # 2) solve (I - B) X = A
    M = np.eye(size)
    for i in range(size):
        for j in range(size):
            M[i, j] -= B[i, j]
    Xv = np.linalg.solve(M, A)

    # 3) unpack
    Xf = Xv[:NJ].reshape((N, J))
    Xm = Xv[NJ:].reshape((N, J))
    return Xf, Xm


@njit(cache=True)
def calc_td_prime_jit(
    N: int,
    J: int,
    pif_hat: np.ndarray,        # shape (N, N, J)
    pim_hat: np.ndarray,        # shape (N, N, J)
    Xf_prime: np.ndarray,       # shape (N, J)
    Xm_prime: np.ndarray,       # shape (N, J)
    tilde_tau_prime: np.ndarray  # shape (N, N, J)
) -> np.ndarray:
    """
    JIT version of calc_td_prime (Equation 12).
    """
    td = np.zeros(N)
    for n in range(N):
        tot = 0.0
        for i in range(N):
            for j in range(J):
                imp = (pif_hat[n, i, j] * Xf_prime[n, j] + pim_hat[n, i, j] * Xm_prime[n, j]) \
                      / tilde_tau_prime[n, i, j]
                exp = (pif_hat[i, n, j] * Xf_prime[i, j] + pim_hat[i, n, j] * Xm_prime[i, j]) \
                      / tilde_tau_prime[i, n, j]
                tot += imp - exp
        td[n] = tot
    return td



@njit(cache=True)
def solve_price_and_cost_jit(
    N: int,
    J: int,
    w_hat: np.ndarray,        # shape (N,)
    beta: np.ndarray,         # shape (N, J)
    gamma: np.ndarray,        # shape (N, J, J)
    theta: np.ndarray,        # shape (J,)
    pim: np.ndarray,          # shape (N, N, J)
    lambda_hat: np.ndarray,   # shape (N, J)
    dm_hat: np.ndarray,       # shape (N, N, J)
    max_iter: int = 1000,
    tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """
    JIT‐compiled version of solve_price_and_cost (Eqns 8–9),
    now delegating to calc_c_hat_jit and calc_Pm_hat_jit.
    """
    Pm_hat = np.ones((N, J))
    c_hat  = np.empty((N, J))

    for _ in range(max_iter):
        # 1) cost index via JIT helper
        c_hat = calc_c_hat_jit(w_hat, Pm_hat, beta, gamma)
        # 2) intermediate‐goods price index via JIT helper
        Pm_new = calc_Pm_hat_jit(N, J, theta, pim, lambda_hat, c_hat, dm_hat)
        # 3) convergence check
        if np.max(np.abs(Pm_new - Pm_hat)) < tol:
            Pm_hat = Pm_new
            # final cost re‐compute
            c_hat = calc_c_hat_jit(w_hat, Pm_hat, beta, gamma)
            break
        Pm_hat = Pm_new

    return c_hat, Pm_hat


@njit(cache=True)
def generate_equilibrium_jit(
    N: int,
    J: int,
    w_hat: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    theta: np.ndarray,
    pif: np.ndarray,
    pim: np.ndarray,
    lambda_hat: np.ndarray,
    df_hat: np.ndarray,
    dm_hat: np.ndarray,
    tilde_tau_prime: np.ndarray,
    td_prime: np.ndarray,
    VA: np.ndarray
):
    """
    JIT version of generate_equilibrium.
    """
    c_hat, Pm_hat = solve_price_and_cost_jit(
        N, J, w_hat,
        alpha, beta, gamma,
        theta, pif, pim,
        lambda_hat, df_hat, dm_hat, tilde_tau_prime
    )
    Pf_hat = calc_Pf_hat_jit(N, J, theta, pif, lambda_hat, c_hat, df_hat)
    pif_hat = calc_pif_hat_jit(N, J, theta, lambda_hat, c_hat, Pf_hat, df_hat)
    pim_hat = calc_pim_hat_jit(N, J, theta, lambda_hat, c_hat, Pm_hat, dm_hat)
    Xf_prime, Xm_prime = calc_X_jit(
        N, J, w_hat, pif_hat, pim_hat, td_prime,
        VA, alpha, gamma,
        pif, pim, tilde_tau_prime
    )
    td = calc_td_prime_jit(N, J, pif_hat, pim_hat, Xf_prime, Xm_prime, tilde_tau_prime)
    # price index and real wage are post‐processing, done in Python
    return c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, td