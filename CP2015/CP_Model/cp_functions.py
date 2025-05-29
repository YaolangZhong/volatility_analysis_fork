import numpy as np
from numpy.typing import NDArray
from numba import njit

def solve_price_and_cost(
        w_hat: NDArray[np.float64], # shape (N, ), the relative wage vector under policy tau_prime and tau
        P_hat: NDArray[np.float64], # shape (N, J), the relative price vector under policy tau_prime and tau
        pi: NDArray[np.float64],    # shape (N, N, J), the expenditure shape vector, the (n, i, j) element denotes the country n's share of expenditure of goods from sector j from country i
        gamma: NDArray[np.float64], # shape (N, J, J), the (n, j, k) element denotes sector k's share in producing goods in sector j, country n 
        beta: NDArray[np.float64],  # shape (N, J), the (n, j) element denotes the country n's value-added share in sector j
        theta: NDArray[np.float64], # shape (J, ) the Frechet distribution parameters of dispersion
        kappa_hat: NDArray[np.float64], # shape (N, N, J), the relative trade cost vector under policy tau_prime and tau
        N, J) -> set[NDArray[np.float64], NDArray[np.float64]]: 

    
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
    
    return P_hat, c_hat


# (Equation (12) in CP(2015))
def solve_piprime(
        c_hat: NDArray[np.float64], # shape (N, J), the cost of input bundle vecotr, the (n, j) element denotes the cost of producing good in sector j in country n 
        P_hat: NDArray[np.float64], # shape (N, J), the price index vector, the (n, j) element denotes the price index of sector j in country n
        pi: NDArray[np.float64], # shape (N, N, J), the expenditure share vector under policy tau, the (n, i, j) element denotes the country n's share of expenditure of goods from sector j from country i
        theta: NDArray[np.float64], # shape (J, ) the Frechet distribution parameters of dispersion
        kappa_hat: NDArray[np.float64], # shape (N, N, J), the relative trade cost vector under policy tau_prime and tau
        N, J) -> NDArray[np.float64]: # pi_prime: shape (N, N, J), the expenditure share vector under policy tau_prime
    
    numerator = c_hat[np.newaxis, :, :] * kappa_hat  # shape: (N, N, J)
    # Compute the denominator (P_hat[n, j])
    denominator = P_hat[:, np.newaxis, :]  # shape: (N, 1, J)
    # Compute the expression inside the brackets
    inside_brackets = numerator / denominator  # shape: (N, N, J) 
    # Raise to the power of -theta[j]
    pi_hat = inside_brackets ** -theta[np.newaxis, np.newaxis, :]  # shape: (N, N, J)
    pi_prime = pi_hat * pi
    # Normalize
    # pi_hat_sum = np.sum(pi_hat, axis=1, keepdims=True)  # shape: (N, 1, J)
    # pi_hat_normalized = pi_hat / pi_hat_sum  # shape: (N, N, J)
    # pi_prime = pi_hat_normalized * pi  # shape: (N, N, J)
    return pi_prime


@njit
def solve_X_prime(
    w_hat,           # shape (N,), the relative wage vector under policy tau_prime and tau
    alpha,           # shape (N, J), utility share of goods from sector j for country n
    gamma,           # shape (N, J, J), the (n, j, k) element denotes sector k's share in producing goods in sector j, country n
    pi_prime,        # shape (N, N, J), the (n, i, j) element denotes country n's share of expenditure of goods of sector j from country i
    VA,              # shape (N,), value added for each country
    tilde_tau_prime, # shape (N, N, J), the tariff vector under policy tau_prime (1+tariff rate)
    D,               # shape (N,), country n's trade deficit (data is export - import, so -D is used)
    N,
    J,
    X_prime_initial  # shape (N, J), the initial guess of X_prime
):
    max_iter = 1000000
    tol = 1e-6

    # Copy initial guess so we don't modify the input.
    X_prime = X_prime_initial.copy()

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
            I_prime[n] = w_hat[n] * VA[n] - D[n] + sum_term

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



# Real wage

def calc_real_w(w, P, alpha):

    N = P.shape[0]  
    P_index_hat = np.ones(N)
    
    for n in range(N):
        for j in range(P.shape[1]):
            P_index_hat[n] *= P[n, j] ** alpha[n, j]
    
    real_w_hat = w / P_index_hat
    return real_w_hat



# Real income

def calc_real_I(pi, tilde_tau, X, D, P, alpha, beta):

    VA_nj = np.einsum('ij,nij,nij,nj->ij', beta, pi, 1 / tilde_tau, X)  # shape: (N,J)
    VA_n = np.sum(VA_nj, axis=1)

    sum_term  = np.sum(                                          # (N,)
        pi * (1.0 - 1.0 / tilde_tau) * X[:, None, :],
        axis=(1, 2)
    )
    I_base = VA_n - D + sum_term                            # (N,)

    P_index_base = np.prod(P ** alpha, axis=1)               # (N,)

    real_I_base = I_base / P_index_base                          # (N,)

    return real_I_base




