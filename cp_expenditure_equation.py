import numpy as np

def calc_X_cp(w_hat, alpha, gamma, pi_prime, tilde_tau_prime, D, VA):
    N, J = alpha.shape

    def calc_A():
        Af = alpha * (w_hat * VA - D)[:, None]
        return np.concatenate([Af.ravel(), np.zeros(Af.size)])

    def calc_B():
        # Precompute common components for Bff and Bfm
        factorff = (tilde_tau_prime - 1) / tilde_tau_prime
        U_ff = np.sum(factorff * pi_prime, axis=1)
        u_ff = U_ff.ravel()
        Du_ff = np.diag(u_ff)
        
        v = alpha.ravel()
        Dv = np.diag(v)
        R = np.kron(np.eye(N), np.ones((1, J)))
        P = np.kron(np.eye(N), np.ones((J, 1)))
        
        Bff = Dv @ P @ R @ Du_ff
        Bfm = Bff  

        U_m = pi_prime / tilde_tau_prime
        B = np.einsum('nks,ink->nsik', gamma, U_m).reshape(N*J, N*J)
        Bmf = Bmm = B  

        return np.vstack([
            np.hstack([Bff, Bfm]),
            np.hstack([Bmf, Bmm])
        ])

    I = np.eye(2 * N * J)
    A = calc_A()
    B = calc_B()
    
    X_total = np.linalg.solve(I - B, A)
    X = (X_total[:N*J] + X_total[N*J:]).reshape(N, J)
    
    return X

