import numpy as np
from cp_functions import solve_price_and_cost, solve_piprime, solve_X_prime


def equilibrium(gamma, beta, theta, tilde_tau_prime, kappa_hat, pi, alpha, VA, D, N, J, X_initial,w_hat_initial,P_hat_initial):
    vfactor = -0.2  
    tol = 1e-6    
    max_iter = 1e+6  

    # w_hat = np.ones(N)
    # P_hat = np.ones((N, J)) 

    w_hat = w_hat_initial.copy()  # Initialize w_hat with the provided initial guess
    P_hat = P_hat_initial.copy()  # Initialize P_hat with the provided initial guess

    wfmax = 1  
    Pfmax = 1
    e = 1  

    while e <= max_iter and wfmax > tol:
        
        P_hat_new, c_hat = solve_price_and_cost(w_hat, P_hat, pi, gamma, beta, theta, kappa_hat, N, J)
        pi_prime = solve_piprime(c_hat, P_hat_new, pi, theta, kappa_hat, N, J)
        X_prime = solve_X_prime(w_hat, alpha, gamma, pi_prime, VA, tilde_tau_prime, D, N, J, X_initial)
        
        X_initial = X_prime.copy()  # Update the initial guess for X_prime
        
        EX = np.zeros(N) 
        IM = np.zeros(N) 
        EX = np.einsum('inj,inj,ij->n', pi_prime, 1 / tilde_tau_prime, X_prime)  # shape: (N,)
        IM = np.einsum('nij,nij,nj->n', pi_prime, 1 / tilde_tau_prime, X_prime)  # shape: (N,)
    
        ZW2 = -(IM - EX + D) / VA
        # Update wages (skip the first country using the mask)
        w_hat_new = np.ones(N)  
        w_hat_new = w_hat * (1 - vfactor * ZW2 / w_hat)

        # Check convergence (exclude the first country using the mask)
        wfmax = np.max(np.abs(w_hat_new - w_hat))
        Pfmax = np.max(np.abs(P_hat_new - P_hat))

        # Update wages for the next iteration
        w_hat = w_hat_new

        P_hat = P_hat_new 

        # vfactor = 0.2 * vfactor  # Reduce learning rate
        
        min_X_prime = np.min(X_prime)
        max_X_prime = np.max(X_prime)

        min_w_hat = np.min(w_hat)
        max_w_hat = np.max(w_hat)

        print(f"Round {e}: w_hat_min = {min_w_hat}, w_hat_max = {max_w_hat}, min_X_prime = {min_X_prime}, max_X_prime = {max_X_prime}, wfmax = {wfmax}, Pfmax = {Pfmax} ")

        e += 1


    return w_hat, P_hat, X_prime, pi_prime