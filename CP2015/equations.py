import numpy as np
from scipy.linalg import block_diag
from typing import Tuple
from numba import njit
from models import ModelParams, ModelShocks

@njit
def calc_c_hat(
    w_hat:  np.ndarray,   # (N,)   wage changes  ẇ_i
    P_hat: np.ndarray,   # (N,S)  input-price changes P̂_{ik}
    beta:   np.ndarray,   # (N,S)  labour shares β_{is}
    gamma:  np.ndarray    # (N,S,S) input shares γ_{isk}
) -> np.ndarray:
    """
    Equation (E1):  ĉ_{is} = exp[ β_{is} ln ŵ_i  +  Σ_k γ_{isk} ln P̂_{ik} ].
    """
    N, S = beta.shape
    log_w  = np.log(w_hat)      # (N,)
    log_P = np.log(P_hat)     # (N,S)

    out = np.empty((N, S))
    for i in range(N):
        for s in range(S):
            acc = 0.0
            for k in range(S):
                acc += gamma[i, s, k] * log_P[i, k]
            out[i, s] = np.exp(beta[i, s] * log_w[i] + acc)
    return out

@njit
def calc_price_index(
    theta: np.ndarray,         # (S,)      trade elasticities θ_s
    pi: np.ndarray,            # (N,N,S)   bilateral shares π_{ins}^{·}
    lambda_hat: np.ndarray,    # (N,S)     productivity shocks Λ̂_{ns}
    c_hat: np.ndarray,         # (N,S)     cost changes ĉ_{ns}
    d_hat: np.ndarray          # (N,N,S)   wedge changes d̂_{ins}^{·}
) -> np.ndarray:
    """
    Equation (E2):
        P̂_{is}^{−θ_s} = Σ_n  π^{0}_{ins} · Λ̂_{ns} · (ĉ_{ns} d̂_{ins})^{−θ_s}.
    Works for both final and intermediate price indexes, depending on π and d.
    """
    I, N, S = pi.shape
    P_inv_pow = np.empty((I, S))   # store P̂^{−θ}

    for i in range(I):
        for s in range(S):
            acc = 0.0
            t  = theta[s]
            for n in range(N):
                acc += (
                    pi[i, n, s]
                    * lambda_hat[n, s]
                    * (c_hat[n, s] * d_hat[i, n, s]) ** (-t)
                )
            P_inv_pow[i, s] = acc

    # Convert P̂^{−θ} to P̂  :  P̂ = (P̂^{−θ})^{−1/θ}
    P_hat = np.empty_like(P_inv_pow)
    for i in range(I):
        for s in range(S):
            P_hat[i, s] = P_inv_pow[i, s] ** (-1.0 / theta[s])

    return P_hat


@njit
def calc_P_hat(
    c_hat: np.ndarray,
    theta: np.ndarray,
    pi: np.ndarray,
    lambda_hat: np.ndarray,
    d_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (8): Intermediate-goods price index changes P̂ᵐ.
    """
    return calc_price_index(
        theta,
        pi,
        lambda_hat,
        c_hat,
        d_hat
    )

@njit
def calc_expenditure_share(
    theta: np.ndarray,
    lambda_hat: np.ndarray,
    c_hat: np.ndarray,
    P_hat: np.ndarray,
    d_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (9): Expenditure‐share changes π̂ given cost and price index changes.

      π̂[i,n,j] = λ[n,j] · (ĉ[n,j]·d̂[i,n,j])^(−θ_j) / (P̂[i,j]^(−θ_j))
    Returns
    -------
    pi_hat : array (importer i, exporter n, sector j)
    """
    # λ[n,j]  → expand across importer dimension  → shape (1, N, S)
    lam_ex = lambda_hat[np.newaxis, :, :]
    # (ĉ[n,j] · d̂[i,n,j])^(−θ_j)  → shape (I, N, S)
    cost_factor = (c_hat[np.newaxis, :, :] * d_hat) ** (-theta[np.newaxis, np.newaxis, :])
    # numerator: λ · cost_factor → shape (I, N, S)
    num = lam_ex * cost_factor
    # denominator: P̂[i,j]^(−θ_j)  → shape (I, 1, S), broadcasts over exporter axis
    den = P_hat[:, np.newaxis, :] ** (-theta[np.newaxis, np.newaxis, :])
    # final (I, N, S) array – already ordered as (importer, exporter, sector)
    return num / den

@njit
def calc_pi_hat(
    c_hat: np.ndarray,
    P_hat: np.ndarray,
    theta: np.ndarray,
    lambda_hat: np.ndarray,
    d_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (9): Intermediate-goods expenditure-share changes π̂ᵐ.
    """
    return calc_expenditure_share(
        theta,
        lambda_hat,
        c_hat,
        P_hat,
        d_hat
    )



@njit
def solve_price_and_cost(
    w_hat:      np.ndarray,  # (N,)   wage changes
    beta:       np.ndarray,  # (N,S)  labour shares β_{is}
    gamma:      np.ndarray,  # (N,S,S) IO coefficients γ_{isk}
    theta:      np.ndarray,  # (S,)   trade elasticities θ_s
    pi:        np.ndarray,  # (N,N,S) baseline π^{0m}_{ins}
    lambda_hat: np.ndarray,  # (N,S)   productivity shocks Λ̂_{ns}
    d_hat:     np.ndarray   # (N,N,S) trade‑cost shocks d̂^{m}_{ins}
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed‑point solver for unit‑cost changes (c_hat) and intermediate
    price‑index changes (P_hat).  Runs fully in Numba nopython mode.
    """
    MAX_ITER = 1000
    TOL      = 1e-6
    N = w_hat.shape[0]
    S = theta.shape[0]
    # --- initial guess ---------------------------------------------------
    P_hat = np.ones((N, S))
    for _ in range(MAX_ITER):
        # 1. unit costs given current Pm_hat
        c_hat = calc_c_hat(w_hat, P_hat, beta, gamma)
        # 2. update intermediate price index
        P_new = calc_P_hat(c_hat, theta, pi, lambda_hat, d_hat)
        # 3. convergence
        if np.abs(P_new - P_hat).max() < TOL:
            P_hat = P_new
            break
        P_hat = P_new
    # final c_hat consistent with the converged Pm_hat
    c_hat = calc_c_hat(w_hat, P_hat, beta, gamma)
    return c_hat, P_hat


@njit
def calc_X_prime(
    alpha: np.ndarray,             # (N,S)
    gamma: np.ndarray,             # (N,S,S)
    pi_prime: np.ndarray,          # (N,N,S)
    tilde_tau_prime: np.ndarray,   # (N,N,S)
    w_hat: np.ndarray,             # (N,)
    V: np.ndarray,                 # (N,)
    X_prev: np.ndarray,            # (N,S), previous guess for X
    D: np.ndarray                  # (N,)
) -> np.ndarray:
    """
    Fully consistent CP2015 Eq. (13):

    X_ns' = sum_k gamma_nsk sum_i [pi_ink' / (1+tau_ink')] X_ik'
            + alpha_ns [ w_hat_n V_n + sum_k sum_i [tau_nik' pi_nik' / (1+tau_nik')] X_nk' + D_n ]
    """
    N, S = alpha.shape

    # Precompute import shares adjusted by tariffs
    pi_over_tau = pi_prime / tilde_tau_prime                 # shape (N,N,S)
    tau_ratio = (tilde_tau_prime - 1) / tilde_tau_prime      # shape (N,N,S)

    # Compute intermediate goods term
    intermed_term = np.zeros((N, S))
    for n in range(N):
        for s in range(S):
            sum_k = 0.0
            for k in range(S):
                sum_i = 0.0
                for i in range(N):
                    sum_i += pi_over_tau[i, n, k] * X_prev[i, k]
                sum_k += gamma[n, s, k] * sum_i
            intermed_term[n, s] = sum_k

    # Compute tariff revenue term
    tariff_term = np.zeros(N)
    for n in range(N):
        sum_k = 0.0
        for k in range(S):
            sum_i = 0.0
            for i in range(N):
                sum_i += tau_ratio[n, i, k] * pi_prime[n, i, k] * X_prev[n, k]
            sum_k += sum_i
        tariff_term[n] = sum_k

    # Final computation of expenditures X_prime
    X_prime = np.empty((N, S))
    for n in range(N):
        income = w_hat[n] * V[n] + tariff_term[n] + D[n]
        for s in range(S):
            X_prime[n, s] = intermed_term[n, s] + alpha[n, s] * income

    return X_prime


@njit
def calc_D_prime(
    pi_prime: np.ndarray,          # shape (N,N,S), importer n, exporter i, sector s
    tilde_tau_prime: np.ndarray,   # shape (N,N,S), importer n, exporter i, sector s
    X_prime: np.ndarray            # shape (N,S), importer n, sector s
) -> np.ndarray:
    """
    Corrected CP2015 Trade Balance (Eq. 14):

    D_n' = sum_{s,i}[π_nis' / (1+τ_nis')] * X_ns' - sum_{s,i}[π_ins' / (1+τ_ins')] * X_is'
    """
    N, _, S = pi_prime.shape

    D_prime = np.zeros(N)

    for n in range(N):
        imports = 0.0
        exports = 0.0

        for s in range(S):
            # Imports: sum over exporters i
            for i in range(N):
                imports += (pi_prime[n, i, s] / tilde_tau_prime[n, i, s]) * X_prime[n, s]

            # Exports: sum over importers i
            for i in range(N):
                exports += (pi_prime[i, n, s] / tilde_tau_prime[i, n, s]) * X_prime[i, s]

        D_prime[n] = imports - exports

    return D_prime


@njit
def generate_equilibrium(
    w_hat:           np.ndarray,   # (N,)
    beta:            np.ndarray,   # (N,S)
    gamma:           np.ndarray,   # (N,S,S)
    theta:           np.ndarray,   # (S,)
    pi:              np.ndarray,   # (N,N,S) baseline import shares π⁰_{nis}
    lambda_hat:      np.ndarray,   # (N,S)   productivity shocks Λ̂_{ns}
    d_hat:           np.ndarray,   # (N,N,S) trade-cost shocks d̂_{nis}
    tilde_tau_prime: np.ndarray,   # (N,N,S) 1+τ′_{nis}
    alpha:           np.ndarray,   # (N,S)   Cobb–Douglas α_{ns}
    V:               np.ndarray,   # (N,)    value-added deflator V_n
    D:               np.ndarray,   # (N,)    observed trade deficits (exogenous)
    X:               np.ndarray    # real data X, used as initial guess
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified CP-2015 equilibrium wrapper (single-expenditure version).

    Returns
    -------
    (c_hat, P_hat, pi_hat, X_prime, D_prime, p_index, real_w)
    """
    c_hat, P_hat = solve_price_and_cost(w_hat, beta, gamma, theta, pi, lambda_hat,d_hat)
    pi_hat   = calc_pi_hat(c_hat, P_hat, theta, lambda_hat, d_hat)
    pi_prime = pi * pi_hat
    X_prime = calc_X_prime(alpha, gamma, pi_prime, tilde_tau_prime, w_hat, V, X, D)
    D_prime = calc_D_prime(pi_prime, tilde_tau_prime, X_prime)
    p_index = np.exp((alpha * np.log(P_hat)).sum(axis=1))  # Cobb-Douglas CPI
    real_w  = w_hat / p_index
    return (c_hat, P_hat, pi_hat, X_prime, D_prime, p_index, real_w)
