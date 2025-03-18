import numpy as np
from .original_models import OldModelParams, OldModelShocks, OldModelSol


def solve_price_and_cost(
    w_hat, P_hat, mp: OldModelParams, shocks: OldModelShocks
):
    """
    Equation (10) and (11) in CP(2015)
    Solve for the price index (P_hat) and cost of input bundles (c_hat)
    given the wage index (w_hat), price index (P_hat) and exogenous parameters.

    Endogenous variables:
        w_hat: Wage change (N,)
        P_hat: Price index change (N, J)
    Returns:
        P_hat: Price index change (N, J)
        c_hat: Cost of input bundles change (N, J)
    """
    # Step 1: Compute cost of input bundles (c_hat) (Equation (10) in CP(2015))
    log_w_hat = np.log(w_hat)  # shape: (N,)
    log_P_hat = np.log(P_hat)  # shape: (N, J)

    # Compute: beta[n,j] * log(w_hat[n])
    term1 = mp.beta * log_w_hat[:, np.newaxis]  # shape: (N, J)

    # Compute: sum over k of gamma[n, k, j] * log_P_hat[n, k]
    term2 = np.einsum("nkj,nk->nj", mp.gamma, log_P_hat)  # shape: (N, J)

    log_c_hat = term1 + term2  # shape: (N, J)

    c_hat = np.exp(log_c_hat)  # shape: (N, J)

    # Step 2: Compute price indices (P_hat) (Equation (11) in CP(2015))
    # Compute: pi[n, i, j] * (c_hat[i, j] * kappa_hat[n, i, j]) ** -theta[j]
    weighted_costs = (
        mp.pi
        * (c_hat[np.newaxis, :, :] * shocks.kappa_hat)
        ** -mp.theta[np.newaxis, np.newaxis, :]
    )  # shape: (N, N, J)

    P_hat = np.sum(weighted_costs, axis=1) ** (-1 / mp.theta)  # shape: (N, J)

    return P_hat, c_hat


# (Equation (12) in CP(2015))
def solve_piprime(c_hat, P_hat, mp: OldModelParams, shocks: OldModelShocks):
    """
    Equation (12) in CP(2015)
    Solve for the new trade shares (pi_prime)
    given the cost of input bundles (c_hat), price index (P_hat) and exogenous parameters.

    Arguments:
        c_hat: Cost of input bundles change (N, J)
        P_hat: Price index change (N, J)
    Returns:
        pi_prime: New trade shares (N, N, J)
    """
    # Compute (c_hat[i, j] * kappa_hat[n, i, j])
    numerator = c_hat[np.newaxis, :, :] * shocks.kappa_hat  # shape: (N, N, J)

    # Compute the denominator (P_hat[n, j])
    denominator = P_hat[:, np.newaxis, :]  # shape: (N, 1, J)

    # Compute the expression inside the brackets
    inside_brackets = numerator / denominator  # shape: (N, N, J)

    # Raise to the power of -theta[j]
    pi_hat = (
        inside_brackets ** -mp.theta[np.newaxis, np.newaxis, :]
    )  # shape: (N, N, J)

    # print(pi_hat)

    pi_prime = pi_hat * mp.pi

    return pi_prime


def solve_X_prime(
    w_hat,
    pi_prime,
    mp: OldModelParams,
    shocks: OldModelShocks,
    X_prime_initial,
    max_iter=1e6,
    tol=1e-6,
    mute=True,
):
    """
    Equation (13) in CP(2015)
    Solve for the new expenditures (X_prime)
    given the wage index (w_hat), exogenous parameters and initial guess of expenditures.

    Arguments:
        w_hat: Wage change (N,)
        pi_prime: Trade share after shock (N, N, J)
    Arguments for iterations:
        X_prime_initial: Initial guess of expenditure (N, J)
        max_iter: Maximum number of iterations
        tol: Tolerance level for convergence

    Returns:
        X_prime: New expenditures (N, J)
    """
    # initialize X_prime
    X_prime = X_prime_initial
    tilde_tau_prime = shocks.kappa_hat * mp.tilde_tau
    tilde_tau_prime = np.maximum(tilde_tau_prime, 1e-10)
    pi_prime = np.clip(pi_prime, 1e-10, 1e10)
    X_prime = np.clip(X_prime, 1e-10, 1e10)

    for iteration in range(int(max_iter)):
        # I_prime
        # I_n = w_n_hat * VA_n + sum_{j=1}^J sum_{i=1}^N (pi_prime[n,i,j] * (1 - 1/tilde_tau_prime[n,i,j]) * X_prime[n,j]) + D_n

        sum_term = np.sum(
            pi_prime * (1 - 1 / tilde_tau_prime) * X_prime[:, np.newaxis, :],
            axis=(1, 2),
        )
        I_prime = w_hat * mp.VA - mp.D + sum_term  # (N,)

        # Term_1
        # Term_1[i,k] = sum_{n=1}^N (pi_prime[i,n,k] / tilde_tau_prime[i,n,k} * X_prime[i,k})
        pi_tau_ratio = pi_prime / tilde_tau_prime  # (N, N, J)
        Term_1 = np.sum(
            pi_tau_ratio * X_prime[:, np.newaxis, :], axis=0
        )  # (N, J)

        # gamma_term
        # gamma_term[n,j] = sum_{k=1}^J gamma[n,j,k] * Term_1[n,k}
        gamma_term = np.einsum("njk,nk->nj", mp.gamma, Term_1)  # (N, J)

        # X_prime_new
        X_prime_new = gamma_term + mp.alpha * I_prime[:, np.newaxis]  # (N, J)

        # Ensure X_prime_new is positive
        # X_prime_new = np.maximum(X_prime_new, 1e-10)

        if np.max(np.abs(X_prime_new - X_prime)) < tol:
            if not mute:
                print(f"Converged in {iteration} iterations")
            return X_prime_new

        X_prime = X_prime_new
        iteration += 1

    if not mute:
        print("Max iterations reached")
        print(X_prime)

    return X_prime


def equilibrium(
    mp: OldModelParams,
    shocks: OldModelShocks,
    numeraire_index,
    X_prime_initial,
    vfactor=-0.2,
    tol=1e-6,
    max_iter=1000,
    mute=True,
):
    """
    Solve for the equilibrium given the exogenous parameters and initial guess of expenditures.

    Arguments for models:
        mp: OldModelParams
        shocks: OldModelShocks
    Arguments for iterations:
        numerair_index: Index of the numeraire country
        X_prime_initial: Initial guess of expenditure (N, J)
        vfactor: Learning rate
        tol: Tolerance level for convergence
        max_iter: Maximum number of iterations
    """
    # Initialize variables
    N = mp.N
    J = mp.J
    w_hat = np.ones(N)
    P_hat = np.ones((N, J))

    mask = np.ones(N, dtype=bool)
    mask[numeraire_index] = False

    wfmax = 1
    Pfmax = 1
    e = 1

    while e <= max_iter and wfmax > tol:

        # P_hat_new, c_hat = solve_price_and_cost(w_hat, P_hat, pi, gamma, beta, theta, kappa_hat)
        P_hat_new, c_hat = solve_price_and_cost(w_hat, P_hat, mp, shocks)

        pi_prime = solve_piprime(c_hat, P_hat_new, mp, shocks)

        X_prime = solve_X_prime(
            w_hat,
            pi_prime,
            mp,
            shocks,
            X_prime_initial,
            max_iter=1e6,
            tol=1e-6,
            mute=mute,
        )
        # X_prime_initial = X_prime

        EX = np.zeros(N)
        IM = np.zeros(N)
        EX = np.einsum(
            "inj,inj,ij->n",
            pi_prime,
            1 / shocks.kappa_hat * mp.tilde_tau,
            X_prime,
        )  # shape: (N,)
        IM = np.einsum(
            "nij,nij,nj->n",
            pi_prime,
            1 / shocks.kappa_hat * mp.tilde_tau,
            X_prime,
        )  # shape: (N,)

        # ZW2 = (IM- EX - D) / VA
        ZW2 = -(IM - EX + mp.D) / mp.VA

        # Update wages (skip the first country using the mask)
        w_hat_new = np.ones(N)
        w_hat_new[mask] = w_hat[mask] * (1 - vfactor * ZW2[mask] / w_hat[mask])

        # Check convergence (exclude the first country using the mask)
        wfmax = np.max(np.abs(w_hat_new[mask] - w_hat[mask]))
        Pfmax = np.max(np.abs(P_hat_new - P_hat))

        # Update wages for the next iteration
        w_hat = w_hat_new

        P_hat = P_hat_new

        # vfactor = 0.2 * vfactor  # Reduce learning rate

        min_X_prime = np.min(X_prime)
        max_X_prime = np.max(X_prime)

        min_w_hat = np.min(w_hat)
        max_w_hat = np.max(w_hat)

        if not mute:
            print(
                f"Round {e}: w_hat_min = {min_w_hat}, w_hat_max = {max_w_hat}, min_X_prime = {min_X_prime}, max_X_prime = {max_X_prime}, wfmax = {wfmax}, Pfmax = {Pfmax} "
            )

        e += 1

    # Return MoodelSol object
    c_hat, P_hat = solve_price_and_cost(w_hat, P_hat, mp, shocks)
    return OldModelSol(mp, shocks, w_hat, c_hat, P_hat, X_prime, pi_prime)
