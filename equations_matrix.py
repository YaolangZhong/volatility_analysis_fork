import numpy as np
from numba import njit, prange
from models import ModelParams, ModelShocks, Usage




def calc_X(
    w_hat,
    pif_hat,
    pim_hat,
    td_prime,
    mp: ModelParams,
    shocks: ModelShocks,
):
    """
    X = A + B X
    where 
        X = [vec(Xf); vec(Xm)]
        A = [Af; Am]
        B = [Bff  Bfm]
            [Bmf  Bmm]
    
    In the vectorized system:
    - Xf and Xm are originally (N, J) matrices, and when vectorized, the reshape(-1) function by default using the "C-order" 
        and they become vectors of length N*J in the order [X_11, ..., X_1J,..., X_N1,...,X_NJ].
        Therefore, X = [vec(Xf); vec(Xm)] is a vector of shape (2*N*J, ).
    - A is defined similarly: A = [vec(Af); vec(Am)], with Af and Am each of shape (N, J),
      so A has shape (2*N*J, ).
    - The B blocks (Bff, Bfm, Bmf, Bmm) each is of the shape (N*J, N*J) constructed as below

    """

    N, J = mp.alpha.shape

    def calc_A():
        # Af is the [alpha x (w_hat x w0 x L0 + TD)], alpha is of shape (N, J) and all terms in the parathesis is of shape (N, )
        # Am is simply zero of shape (N, J)
        Af = mp.alpha * (w_hat * mp.w0 * mp.L0 + td_prime)[:, np.newaxis]
        # Intermediate goods exogenous component (Am) is zero: (N, J)
        Am = np.zeros_like(Af)
        # Vectorize to get vectors of length N*J
        Af_vec = Af.reshape(-1)  # (N*J,)
        Am_vec = Am.reshape(-1)  # (N*J,)
        # Stack vertically to form A of shape (2*N*J,)
        A = np.concatenate([Af_vec, Am_vec])
        return A
    
    def calc_B():
        # --- Compute Bff  ---
        """
        - sum_i^N {tau/(1+tau) * pif} is irrelavant to X so we first calculate this term and call the resulting matrix U, 
        which is of the shape (N, J). 
        - Similarly, sum_i^N {tau/(1+tau) * pim} for the case of Bfm
        - we also call mp.alpha the vector V for short
        - the vectorization of U and V is called u and v respectively
        - similarly, the vectorization of Xff is called x
        """
        factorff = (shocks.tilde_tau_prime - 1) / shocks.tilde_tau_prime
        pif_prime = pif_hat * mp.pif
        U = np.sum(factorff * pif_prime, axis=1)   # shape (N, J)
        V = mp.alpha                               # shape (N, J)
        u, v = U.reshape(-1), V.reshape(-1)        # shape (NJ, )
        """
        - consider S = sum_j^J Unj*Xnj, which has shape (N, )
        - the elementwise product Unj*Xnj is computed by Du @ x where Du (NJxNJ) is the diagonalization of u
        - the sum over index j is achieve by a matrix R which is the Kronecker product of diag(N) and ones(1xJ)
        - hence S = R @ Du @ x
        """
        Du = np.diag(u)
        R = np.kron(np.eye(N), np.ones((1, J)))
        """
        - next, consider replicating S by J times, by a matrix P which is the Kronecker product of diag(N) and ones(Jx1)
        - the resulting PS is of shape (NJ, )
        - we then calculate the elementwise product of V and PS by Dv @ PS where Dv (NJxNJ) is the diagonalization of v
        """
        P = np.kron(np.eye(N), np.ones((J, 1)))
        Dv = np.diag(v)
        Bff = Dv @ P @ R @ Du

        # --- Compute Bfm  ---
        # only U (and therefore u and Du) is re-calculated
        pim_prime = pim_hat * mp.pim
        U = np.sum(factorff * pim_prime, axis=1)   # shape (N, J)
        u = U.reshape(-1)
        Du = np.diag(u)
        Bfm = Dv @ P @ R @ Du
            

        # --- Compute Bmf  ---
        """
        - let us call pif/(1+tau) the U matrix for short, and call mp.gamma the V matrix, the result matrix Z
        - the index is then X(i,k), U(i,n,k), V(n,k,s), Z(n,s) such that 
        - Z(n,s) = sum_k sum_i V(n,k,s)*U(i,n,k)*X(i,k)
        - let x=vec(X) and z=vec(Z) such that X(i,k) corresponds to the index(i-1)J+k and Z(n,s) corresponds to the index(n-1)J+s
        - then we have z((n-1)J+s) = sum_k sum_i B(n,s)(i,k)*x((i-1)J+k) 
        - with the matrix form z = B x
        where we first have a 4-D tensor B(n,s,i,k) = V(n,k,s)*U(i,n,k)
        and then reshape into (NJ, NJ)
        """
        U = pif_prime / shocks.tilde_tau_prime
        V = mp.gamma
        B = np.einsum('nks,ink->nsik', V, U)
        Bmf = B.reshape((N*J, N*J))

        # --- Compute Bmf  ---
        U = pim_prime / shocks.tilde_tau_prime
        V = mp.gamma
        B = np.einsum('nks,ink->nsik', V, U)
        Bmm = B.reshape((N*J, N*J))
            
        # --- Assemble full B ---
        # Each of Bff, Bfm, Bmf, Bmm has shape (N*J, N*J)
        # Assemble into a 2×2 block matrix:
        B_top = np.hstack((Bff, Bfm))      # shape: (N*J, 2*N*J)
        B_bottom = np.hstack((Bmf, Bmm))     # shape: (N*J, 2*N*J)
        B = np.vstack((B_top, B_bottom))   # shape: (2*N*J, 2*N*J)
        return B
    
    I = np.eye(2*N*J)
    A_vec = calc_A()
    B_vec = calc_B()
    # Solve the linear system: (I - B_vec) * X_vec = A_vec
    X_vec = np.linalg.solve(I - B_vec, A_vec)  # X_vec will have shape (2*N*J,)
    # Now, extract Xf and Xm from X_vec. The first NJ elements correspond to Xf, and the rest to Xm
    Xf_vec = X_vec[:N*J]
    Xm_vec = X_vec[N*J:]

    # Xf_vec, Xm_vec = np.zeros((N, J)), np.zeros((N, J))
    # for j in range(J):
    #     # Determine the indices for sector j.
    #     indices_f = np.arange(j * N, (j + 1) * N)                 # indices for X_f of sector j (length: N)
    #     indices_m = np.arange(N * J + j * N, N * J + (j + 1) * N)     # indices for X_m of sector j (length: N)
    #     # Combine indices to form the full vector indices for sector j (length: 2N)
    #     sector_indices = np.concatenate([indices_f, indices_m])
        
    #     # Extract the corresponding subvector of A (A_sector) and submatrix of B (B_sector)
    #     A_vec_j = A_vec[sector_indices]                # shape: (2N,)
    #     B_vec_j = B_vec[np.ix_(sector_indices, sector_indices)]  # shape: (2N, 2N)
        
    #     # Define the identity matrix for the sector, I_sector of shape (2N, 2N)
    #     I_j = np.eye(2 * N)
        
    #     # Solve the smaller system: (I_sector - B_sector) * X_sector = A_sector
    #     X_vec_j = np.linalg.solve(I_j - B_vec_j, A_vec_j)  # shape: (2N,)
        
    #     # The first N elements correspond to X_f and the next N elements correspond to X_m for sector j.
    #     Xf_vec[:, j] = X_vec_j[:N]
    #     Xm_vec[:, j] = X_vec_j[N:]

    # Reshape the vectors back to (N, J)
    Xf = Xf_vec.reshape(N, J)  # Final goods, shape: (N, J)
    Xm = Xm_vec.reshape(N, J)  # Intermediate goods, shape: (N, J)
    return Xf, Xm



@njit
def calc_c_hat_numba(w_hat, Pm_hat, beta, gamma, N, J):
    """
    Equation (7) in the paper
    Calculate the unit cost index changes (c_hat)
    given wage changes (w_hat) and intermediate input price changes (Pm_hat).
    Endogenous variables:
        w_hat: (N,) array of wage changes
        Pm_hat: (N, J) array of intermediate input price changes
    Returns:
        c_hat: (N, J) array of unit cost index changes
    This is a Numba-accelerated version of calc_c_hat.
    """
    log_w_hat = np.log(w_hat)         # shape: (N,)
    log_Pm_hat = np.log(Pm_hat)         # shape: (N, J)
    
    c_hat = np.empty((N, J))
    # Compute wage component: beta[n,j]*log_w_hat[n]
    for n in prange(N):
        for j in range(J):
            # wage_component for country n and sector j
            wage_comp = beta[n, j] * log_w_hat[n]
            # input_component: sum over k of gamma[n,k,j] * log_Pm_hat[n,k]
            input_comp = 0.0
            for k in range(J):   # assuming gamma is (N, J, J) and log_Pm_hat is (N, J)
                input_comp += gamma[n, k, j] * log_Pm_hat[n, k]
            log_c = wage_comp + input_comp
            c_hat[n, j] = np.exp(log_c)
    return c_hat