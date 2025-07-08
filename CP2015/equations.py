import numpy as np
from scipy.linalg import block_diag
from typing import Tuple
from numba import njit
from models import ModelParams, ModelShocks

def solve_price_and_cost(
    w_hat:      np.ndarray,  # (N,)   wage changes
    P_hat:      np.ndarray,  # (N,S)  initial guess from previous outer loop
    beta:       np.ndarray,  # (N,S)  labour shares β_{is}
    gamma:      np.ndarray,  # (N,S,S) IO coefficients γ_{isk}
    theta:      np.ndarray,  # (S,)   trade elasticities θ_s
    pi:         np.ndarray,  # (N,N,S) baseline π^{0m}_{ins}
    kappa_hat:  np.ndarray, #  (N,N,S), the relative trade cost vector under policy tau_prime and tau
) -> Tuple[np.ndarray, np.ndarray]:
    # Step 1: Compute cost of input bundles (c_hat) (Equation (10) in CP(2015))
    log_w_hat = np.log(w_hat)  # shape: (N,)
    log_P_hat = np.log(P_hat)  # shape: (N, J)
    ### Compute: beta[n,j] * log(w_hat[n])
    term1 = beta * log_w_hat[:, np.newaxis]  # shape: (N, J)
    ### Compute: sum over k of gamma[n, k, j] * log_P_hat[n, k]
    term2 = np.einsum('nkj,nk->nj', gamma, log_P_hat)  # shape: (N, J)
    log_c_hat = term1 + term2  # shape: (N, J)
    c_hat = np.exp(log_c_hat)  # shape: (N, J)
    
    # Step 2: Compute price indices (P_hat) (Equation (11) in CP(2015))
    ### Compute: pi[n, i, j] * (c_hat[i, j] * kappa_hat[n, i, j]) ** -theta[j]
    weighted_costs = pi * (c_hat[np.newaxis, :, :] * kappa_hat) ** -theta[np.newaxis, np.newaxis, :]  # shape: (N, N, J)
    P_hat = np.sum(weighted_costs, axis=1) ** (-1 / theta)  # shape: (N, J)
    return c_hat, P_hat


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
    c_hat: np.ndarray,      # (N, S)
    P_hat: np.ndarray,      # (N, S)
    theta: np.ndarray,      # (S,)
    lambda_hat: np.ndarray, # (N, S)
    d_hat: np.ndarray       # (N, N, S)
) -> np.ndarray:
    """
    Calculate expenditure share changes π̂.
    π̂[n,i,s] = λ̂[i,s] · (ĉ[i,s]·d̂[n,i,s])^(−θ_s) / P̂[n,s]^(−θ_s)
    """
    inv_theta = -theta[np.newaxis, np.newaxis, :]            # (1,1,S)

    # (ĉ * d̂)^(−θ)   →  (N,N,S)
    cost_ratio = (c_hat[np.newaxis, :, :] * d_hat) ** inv_theta

    # λ̂ expand to (1,N,S) then multiply
    num = lambda_hat[np.newaxis, :, :] * cost_ratio          # (N,N,S)

    # denominator P̂^(−θ)  →  (N,1,S)
    den = P_hat[:, np.newaxis, :] ** inv_theta               # (N,1,S)

    return num / den                                         # (N,N,S)


@njit
def calc_X_prime(
    alpha: np.ndarray,             # (N,S)
    gamma: np.ndarray,             # (N,S,S)
    pi_prime: np.ndarray,          # (N,N,S)
    tilde_tau_prime: np.ndarray,   # (N,N,S)
    w_hat: np.ndarray,             # (N,)
    V: np.ndarray,                 # (N,)
    X_init: np.ndarray,            # (N,S)  initial guess
    D: np.ndarray,                 # (N,)
):
    max_iter = 1000000
    tol = 1e-6

    # Copy initial guess so we don't modify the input.
    X_prime = X_init.copy()
    N, J = alpha.shape
    # Clip tilde_tau_prime and pi_prime to avoid division by zero or extreme values.
    for n in range(N):
        for i in range(N):
            for j in range(J):
                if tilde_tau_prime[n, i, j] < 1e-10:
                    tilde_tau_prime[n, i, j] = 1e-10
                if pi_prime[n, i, j] < 1e-10:
                    pi_prime[n, i, j] = 1e-10
                elif pi_prime[n, i, j] > 1e10:
                    pi_prime[n, i, j] = 1e10

    for n in range(N):
        for j in range(J):
            if X_prime[n, j] < 1e-10:
                X_prime[n, j] = 1e-10
            elif X_prime[n, j] > 1e10:
                X_prime[n, j] = 1e10

    for iteration in range(max_iter):
        # Compute I_prime for each country n:
        # I_prime[n] = w_hat[n]*VA[n] - D[n] + sum_{i,j} [pi_prime[n, i, j]*(1 - 1/tilde_tau_prime[n, i, j])*X_prime[n,j]]
        I_prime = np.empty(N)
        for n in range(N):
            sum_term = 0.0
            for i in range(N):
                for j in range(J):
                    sum_term += pi_prime[n, i, j] * (1.0 - (1.0 / tilde_tau_prime[n, i, j])) * X_prime[n, j]
            I_prime[n] = w_hat[n] * V[n] + D[n] + sum_term

        # Compute Term_1 with corrected indices:
        # Term_1[n, k] = sum_{m=0}^{N-1} [pi_prime[m, n, k] / tilde_tau_prime[m, n, k] * X_prime[m, k]]
        Term_1 = np.empty((N, J))
        for n in range(N):
            for k in range(J):
                s = 0.0
                for m in range(N):
                    s += pi_prime[m, n, k] / tilde_tau_prime[m, n, k] * X_prime[m, k]
                Term_1[n, k] = s

        # Compute gamma_term: gamma_term[n, j] = sum_{k=0}^{J-1} gamma[n, j, k] * Term_1[n, k]
        gamma_term = np.empty((N, J))
        for n in range(N):
            for j in range(J):
                s = 0.0
                for k in range(J):
                    s += gamma[n, j, k] * Term_1[n, k]
                gamma_term[n, j] = s

        # Compute new X_prime: X_prime_new[n, j] = gamma_term[n, j] + alpha[n, j]*I_prime[n]
        X_prime_new = np.empty((N, J))
        for n in range(N):
            for j in range(J):
                X_prime_new[n, j] = gamma_term[n, j] + alpha[n, j] * I_prime[n]

        # Check for convergence: if max absolute change is below tol, return.
        diff = 0.0
        for n in range(N):
            for j in range(J):
                tmp = X_prime_new[n, j] - X_prime[n, j]
                if tmp < 0:
                    tmp = -tmp
                if tmp > diff:
                    diff = tmp

        if diff < tol:
            return X_prime_new

        # Update for next iteration
        X_prime = X_prime_new.copy()

    print("Max iterations reached")
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


def calc_sector_links(
    X_prime: np.ndarray,           # shape (N, S), expenditure by country n in sector j  
    pi_prime: np.ndarray,          # shape (N, N, S), trade shares from country i to n in sector j
    tilde_tau_prime: np.ndarray,   # shape (N, N, S), tariff factors
    gamma: np.ndarray              # shape (N, S, S), input-output coefficients
) -> np.ndarray:
    """
    Calculate sector linkages: imports of country n for its output sector k, from country i sector j.
    
    Returns:
        sector_links: np.ndarray with shape (N, S, N, S) and index (n, k, i, j)
        sector_links[n, k, i, j] = imports of country n for output sector k from country i sector j
    """
    N, S = X_prime.shape
    
    # Step 1: Calculate imports: X[n,j] * pi[n,i,j] / (1+tau[n,i,j])
    # imports[n,i,j] = imports of country n from country i in sector j
    imports = X_prime[:, np.newaxis, :] * pi_prime / tilde_tau_prime  # shape (N, N, S)
    
    # Step 2: Expand imports to [n,i,j,:] (add fourth dimension)
    # imports_expanded[n,i,j,k] = imports[n,i,j] for all k
    imports_expanded = imports[:, :, :, np.newaxis]  # shape (N, N, S, 1)
    imports_expanded = np.broadcast_to(imports_expanded, (N, N, S, S))  # shape (N, N, S, S)
    
    # Step 3: Expand gamma from [n,j,k] to [n,:,j,k] (expand second dimension)
    # gamma_expanded[n,i,j,k] = gamma[n,j,k] for all i
    gamma_expanded = gamma[:, np.newaxis, :, :]  # shape (N, 1, S, S)
    gamma_expanded = np.broadcast_to(gamma_expanded, (N, N, S, S))  # shape (N, N, S, S)
    
    # Step 4: Multiply imports by gamma coefficients
    # sector_links_temp[n,i,j,k] = imports[n,i,j] * gamma[n,j,k]
    sector_links_temp = imports_expanded * gamma_expanded  # shape (N, N, S, S)
    
    # Step 5: Reshape from (n,i,j,k) to (n,k,i,j) for interpretation
    # sector_links[n,k,i,j] = country n's imports for output sector k from country i sector j
    sector_links = sector_links_temp.transpose(0, 3, 1, 2)  # shape (N, S, N, S)
    
    return sector_links


# @njit
def generate_equilibrium(
    w_hat:           np.ndarray, 
    P_hat:           np.ndarray,   # (N,)
    alpha:           np.ndarray,   # (N,S)   Cobb–Douglas α_{ns}
    beta:            np.ndarray,   # (N,S)
    gamma:           np.ndarray,   # (N,S,S)
    theta:           np.ndarray,   # (S,)
    pi:              np.ndarray,   # (N,N,S) baseline import shares π⁰_{nis}
    tilde_tau:       np.ndarray,   # (N,N,S) 1+τ′_{nis}
    V:               np.ndarray,   # (N,)    value-added deflator V_n
    D:               np.ndarray,   # (N,)    observed trade deficits (exogenous)
    X:               np.ndarray,    # real data X, used as initial guess
    lambda_hat:      np.ndarray,   # (N,S)   productivity shocks Λ̂_{ns}
    d_hat:           np.ndarray,   # (N,N,S) trade-cost shocks d̂_{nis}
    tilde_tau_hat:   np.ndarray,     # (N,N,S) 1+τ′_{nis}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kappa_hat = lambda_hat**(-1/theta) * (d_hat * tilde_tau_hat)
    c_hat, P_hat = solve_price_and_cost(w_hat, P_hat, beta, gamma, theta, pi, kappa_hat)
    pi_hat   = calc_pi_hat(c_hat, P_hat, theta, lambda_hat, d_hat)
    pi_prime = pi * pi_hat
    tilde_tau_prime = tilde_tau * tilde_tau_hat
    X_prime = calc_X_prime(alpha, gamma, pi_prime, tilde_tau_prime , w_hat, V, X, D)
    EX = np.einsum('inj,inj,ij->n', pi_prime, 1 / tilde_tau_prime, X_prime)
    IM = np.einsum('nij,nij,nj->n', pi_prime, 1 / tilde_tau_prime, X_prime)
    D_prime = IM - EX
    #D_prime = calc_D_prime(pi_prime, tilde_tau_prime, X_prime)
    p_index = np.exp((alpha * np.log(P_hat)).sum(axis=1))  # Cobb-Douglas CPI
    real_w  = w_hat / p_index
    sector_links = calc_sector_links(X_prime, pi_prime, tilde_tau_prime, gamma)
    return (c_hat, P_hat, pi_hat, X_prime, D_prime, p_index, real_w, sector_links)
