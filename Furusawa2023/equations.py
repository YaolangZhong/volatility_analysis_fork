import numpy as np
from scipy.linalg import block_diag
from typing import Tuple
from numba import njit
from models import ModelParams, ModelShocks

def solve_price_and_cost(
    w_hat:      np.ndarray,  # (N,)   wage changes
    Pm_hat:     np.ndarray, # (N,S)  initial guess from previous outer loop
    beta:       np.ndarray,  # (N,S)  labour shares β_{is}
    gamma:      np.ndarray,  # (N,S,S) IO coefficients γ_{isk}
    theta:      np.ndarray,  # (S,)   trade elasticities θ_s
    pif:        np.ndarray,  # (N,N,S) baseline πf^{0m}_{ins}
    pim:        np.ndarray,  # (N,N,S) baseline πm^{0m}_{ins}
    kf_hat:     np.ndarray,     #  (N,N,S)
    km_hat:     np.ndarray,     #  (N,N,S)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Step 1: Compute cost of input bundles (c_hat) (Equation (10) in CP(2015))
    log_w_hat = np.log(w_hat)  # shape: (N,)
    log_Pm_hat = np.log(Pm_hat)  # shape: (N, J)
    ### Compute: beta[n,j] * log(w_hat[n])
    term1 = beta * log_w_hat[:, np.newaxis]  # shape: (N, J)
    ### Compute: sum over k of gamma[n, k, j] * log_P_hat[n, k]
    term2 = np.einsum('nkj,nk->nj', gamma, log_Pm_hat)  # shape: (N, J)
    log_c_hat = term1 + term2  # shape: (N, J)
    c_hat = np.exp(log_c_hat)  # shape: (N, J)
    
    # Step 2: Compute price indices (Pf_hat, Pm_hat)
    ### Compute: pi[n, i, j] * (c_hat[i, j] * kappa_hat[n, i, j]) ** -theta[j]
    weighted_costs = pif * (c_hat[np.newaxis, :, :] * kf_hat) ** -theta[np.newaxis, np.newaxis, :]  # shape: (N, N, J)
    Pf_hat = np.sum(weighted_costs, axis=1) ** (-1 / theta)  # shape: (N, J)
    weighted_costs = pim * (c_hat[np.newaxis, :, :] * km_hat) ** -theta[np.newaxis, np.newaxis, :]  # shape: (N, N, J)
    Pm_hat = np.sum(weighted_costs, axis=1) ** (-1 / theta)  # shape: (N, J)
    return c_hat, Pf_hat, Pm_hat


def calc_expenditure_share(
    theta:       np.ndarray,  # (S,)
    lambda_hat:  np.ndarray,  # (N,S)
    c_hat:       np.ndarray,  # (N,S)
    P_hat:       np.ndarray,  # (I,S)
    d_hat:       np.ndarray   # (I,N,S)
) -> np.ndarray:
    """
    π̂[i,n,s] = λ̂[n,s] · (ĉ[n,s]·d̂[i,n,s])^(−θ_s) / P̂[i,s]^(−θ_s)
    Broadcast version – no Python loops; nopython-safe.
    """
    inv_theta = -theta[np.newaxis, np.newaxis, :]            # (1,1,S)

    # (ĉ * d̂)^(−θ)   →  (I,N,S)
    cost_ratio = (c_hat[np.newaxis, :, :] * d_hat) ** inv_theta

    # λ̂ expand to (1,N,S) then multiply
    num = lambda_hat[np.newaxis, :, :] * cost_ratio          # (I,N,S)

    # denominator P̂^(−θ)  →  (I,1,S)
    den = P_hat[:, np.newaxis, :] ** inv_theta               # (I,1,S)

    return num / den                                         # (I,N,S)

def calc_pi_hat(
    c_hat:      np.ndarray,
    P_hat:      np.ndarray,
    theta:      np.ndarray,
    lambda_hat: np.ndarray,
    d_hat:      np.ndarray
) -> np.ndarray:
    """
    Expenditure-share change π̂.  Thin wrapper for compatibility.
    """
    return calc_expenditure_share(theta, lambda_hat, c_hat, P_hat, d_hat)



@njit
def calc_X_prime(
    alpha: np.ndarray,             # (N,S)
    gamma: np.ndarray,             # (N,S,S)
    pif_prime: np.ndarray,          # (N,N,S)
    pim_prime: np.ndarray,          # (N,N,S)
    tilde_tau_prime: np.ndarray,   # (N,N,S)
    w_hat: np.ndarray,             # (N,)
    V: np.ndarray,                 # (N,)
    Xf_init: np.ndarray,            # (N,S)  initial guess
    Xm_init: np.ndarray,            # (N,S)  initial guess
    D: np.ndarray,                 # (N,)
):
    max_iter = 1000000
    tol = 1e-6

    # Copy initial guess so we don't modify the input.
    Xf_prime = Xf_init.copy()
    Xm_prime = Xm_init.copy()
    N, S = alpha.shape
    # Clip tilde_tau_prime and pi_prime to avoid division by zero or extreme values.
    for n in range(N):
        for i in range(N):
            for s in range(S):
                if tilde_tau_prime[n, i, s] < 1e-10:
                    tilde_tau_prime[n, i, s] = 1e-10
                if pif_prime[n, i, s] < 1e-10:
                    pif_prime[n, i, s] = 1e-10
                elif pif_prime[n, i, s] > 1e10:
                    pif_prime[n, i, s] = 1e10
                if pim_prime[n, i, s] < 1e-10:
                    pim_prime[n, i, s] = 1e-10
                elif pim_prime[n, i, s] > 1e10:
                    pim_prime[n, i, s] = 1e10

    for n in range(N):
        for s in range(S):
            if Xf_prime[n, s] < 1e-10:
                Xf_prime[n, s] = 1e-10
            elif Xf_prime[n, s] > 1e10:
                Xf_prime[n, s] = 1e10
    for n in range(N):
        for s in range(S):
            if Xm_prime[n, s] < 1e-10:
                Xm_prime[n, s] = 1e-10
            elif Xm_prime[n, s] > 1e10:
                Xm_prime[n, s] = 1e10

    for iteration in range(max_iter):
        # Compute I_prime for each country n:
        # I_prime[n] = w_hat[n]*VA[n] - D[n] + sum_{i,j} [pi_prime[n, i, j]*(1 - 1/tilde_tau_prime[n, i, j])*X_prime[n,j]]
        I_prime = np.empty(N)
        for n in range(N):
            sum_term = 0.0
            for i in range(N):
                for s in range(S):
                    sum_term += (pif_prime[n, i, s] * (1.0 - (1.0 / tilde_tau_prime[n, i, s])) * Xf_prime[n, s] \
                                 + pim_prime[n, i, s] * (1.0 - (1.0 / tilde_tau_prime[n, i, s])) * Xm_prime[n, s])
            I_prime[n] = w_hat[n] * V[n] + D[n] + sum_term

        Xf_prime_new = np.empty((N, S))
        for n in range(N):
            for s in range(S):
                Xf_prime_new[n, s] =  alpha[n, s] * I_prime[n]

        # Compute Term_1 with corrected indices:
        # Term_1[n, k] = sum_{m=0}^{N-1} [pi_prime[m, n, k] / tilde_tau_prime[m, n, k] * X_prime[m, k]]
        Term_1 = np.empty((N, S))
        for n in range(N):
            for k in range(S):
                s_sum = 0.0
                for m in range(N):
                    s_sum += (pif_prime[m, n, k] / tilde_tau_prime[m, n, k] * Xf_prime[m, k]) + \
                            (pim_prime[m, n, k] / tilde_tau_prime[m, n, k] * Xm_prime[m, k])
                Term_1[n, k] = s_sum

        # Compute gamma_term: gamma_term[n, s] = sum_{k=0}^{S-1} gamma[n, s, k] * Term_1[n, k]
        Xm_prime_new = np.empty((N, S))
        for n in range(N):
            for s in range(S):
                s_sum = 0.0
                for k in range(S):
                    s_sum += gamma[n, s, k] * Term_1[n, k]
                Xm_prime_new[n, s] = s_sum


        # Check for convergence: if max absolute change is below tol, return.
        diff = 0.0
        for n in range(N):
            for s in range(S):
                tmp1 = Xf_prime_new[n, s] - Xf_prime[n, s]
                tmp2 = Xm_prime_new[n, s] - Xm_prime[n, s]
                tmp = max(abs(tmp1), abs(tmp2))
                if tmp > diff:
                    diff = tmp

        if diff < tol:
            return (Xf_prime_new, Xm_prime_new, I_prime, Term_1)

        # Update for next iteration
        Xf_prime = Xf_prime_new.copy()
        Xm_prime = Xm_prime_new.copy()

    print("Max iterations reached")
    return (Xf_prime_new, Xm_prime_new, I_prime, Term_1)



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

def calc_production(X, pi):
    return np.sum(X[:, np.newaxis, :] * pi, axis=0)


# @njit
def generate_equilibrium(
    w_hat:           np.ndarray, 
    Pm_hat:          np.ndarray,  # (N,)
    alpha:           np.ndarray,   # (N,S)   Cobb–Douglas α_{ns}
    beta:            np.ndarray,   # (N,S)
    gamma:           np.ndarray,   # (N,S,S)
    theta:           np.ndarray,   # (S,)
    pif:             np.ndarray,   # (N,N,S) baseline import shares π⁰_{nis}
    pim:             np.ndarray,   # (N,N,S) baseline import shares π⁰_{nis}
    tilde_tau:       np.ndarray,   # (N,N,S) 1+τ′_{nis}
    V:               np.ndarray,   # (N,)    value-added deflator V_n
    D:               np.ndarray,   # (N,)    observed trade deficits (exogenous)
    Xf:               np.ndarray,  # real data Xf, used as initial guess
    Xm:               np.ndarray,  # real data Xm, used as initial guess
    lambda_hat:      np.ndarray,   # (N,S)   productivity shocks Λ̂_{ns}
    df_hat:          np.ndarray,   # (N,N,S) trade-cost shocks d̂_{nis}
    dm_hat:          np.ndarray,   # (N,N,S) trade-cost shocks d̂_{nis}
    tilde_tau_hat:   np.ndarray,     # (N,N,S) 1+τ′_{nis}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kf_hat = lambda_hat**(-1/theta) * (df_hat * tilde_tau_hat)
    km_hat = lambda_hat**(-1/theta) * (dm_hat * tilde_tau_hat)
    c_hat, Pf_hat, Pm_hat = solve_price_and_cost(w_hat, Pm_hat, beta, gamma, theta, pif, pim, kf_hat, km_hat)
    pif_hat   = calc_pi_hat(c_hat, Pf_hat, theta, lambda_hat, df_hat)
    pim_hat   = calc_pi_hat(c_hat, Pm_hat, theta, lambda_hat, dm_hat)
    pif_prime = pif * pif_hat
    pim_prime = pim * pim_hat
    tilde_tau_prime = tilde_tau * tilde_tau_hat
    Xf_prime, Xm_prime, I_prime, output_prime = calc_X_prime(alpha, gamma, pif_prime, pim_prime, tilde_tau_prime , w_hat, V, Xf, Xm, D)
    EX = np.einsum('inj,inj,ij->n', pif_prime, 1 / tilde_tau_prime, Xf_prime) + \
        np.einsum('inj,inj,ij->n', pim_prime, 1 / tilde_tau_prime, Xm_prime)
    IM = np.einsum('nij,nij,nj->n', pif_prime, 1 / tilde_tau_prime, Xf_prime) + \
        np.einsum('nij,nij,nj->n', pim_prime, 1 / tilde_tau_prime, Xm_prime)
    D_prime = IM - EX
    #D_prime = calc_D_prime(pi_prime, tilde_tau_prime, X_prime)
    p_index = np.exp((alpha * np.log(Pf_hat)).sum(axis=1))  # Cobb-Douglas CPI
    real_w_hat  = w_hat / p_index
    real_I_prime = I_prime / p_index
    X_prime = Xf_prime + Xm_prime
    Xf_prod_prime = calc_production(Xf_prime, pif)
    Xm_prod_prime = calc_production(Xm_prime, pim)
    X_prod_prime = Xf_prod_prime + Xm_prod_prime

    return (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, D_prime, p_index, real_w_hat, X_prime, Xf_prod_prime, Xm_prod_prime, X_prod_prime, I_prime, output_prime, real_I_prime)
