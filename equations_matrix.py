import numpy as np
from scipy.linalg import block_diag
from models import ModelParams, ModelShocks, Usage

def calc_X(
    w_hat,
    pif_hat,
    pim_hat,
    Xf_prime,
    Xm_prime,
    td_prime,
    mp: ModelParams,
    shocks: ModelShocks,
):
    """
    X = A + B X
    where 
        X = [Xf; Xm]
        A = [Af; Am]
        B = [Bff  Bfm]
            [Bmf  Bmm]
    
    In the vectorized system:
    - Xf and Xm are originally (N, J) matrices, and when vectorized they become vectors of length N*J.
    - Therefore, X = [vec(Xf); vec(Xm)] is a vector of shape (2*N*J, ).
    - A is defined similarly: A = [vec(Af); vec(Am)], with Af and Am each of shape (N, J),
      so A has shape (2*N*J, ).
    - The B blocks (Bff, Bfm, Bmf, Bmm) are first computed as 3D arrays of shape (N, N, J)
      (with the two country dimensions and one sector dimension). These are then converted (vectorized)
      into 2D matrices of shape (N*J, N*J) (typically as block diagonal matrices if sectors are independent).
    - The full B matrix is assembled as a 2x2 block matrix of shape (2*N*J, 2*N*J).
    """

    def calc_A():
        # Wage component: for each country, shape (N,)
        wage_component = w_hat * mp.w0 * mp.L0
        # Exogenous expenditure: wage component plus total demand shock, shape (N,)
        expenditure_exo = wage_component + td_prime
        # Af: final goods exogenous component: (N, J)
        Af = mp.alpha * expenditure_exo[:, np.newaxis]
        # Am: intermediate goods exogenous component is assumed zero: (N, J)
        Am = np.zeros_like(Af)
        # Vectorize each (reshape to a vector of length N*J)
        Af_vec = Af.reshape(-1)  # shape: (N*J,)
        Am_vec = Am.reshape(-1)  # shape: (N*J,)
        # Stack Af and Am vertically to get A of shape (2*N*J,)
        A = np.concatenate([Af_vec, Am_vec])
        return A

    def calc_B():
        # --- Compute Bff ---
        # pif_hat has shape (N, N, J) and mp.pif is broadcastable to (N, N, J)
        pif_prime = pif_hat * mp.pif  # shape: (N, N, J)
        # Tariff factor: (tilde_tau_prime - 1) / tilde_tau_prime, shape: (N, N, J)
        factorff = (shocks.tilde_tau_prime - 1) / shocks.tilde_tau_prime
        # Compute Bff in its natural 3D form:
        # mp.alpha is (N, J). To multiply with (N, N, J), we add a new axis for the exporter dimension.
        # The result is a 3D array of shape (N, N, J)
        Bff_3D = mp.alpha[:, None, :] * factorff * pif_prime  # shape: (N, N, J)
        # Now, for each sector (the third dimension), extract the (N, N) matrix and form a block-diagonal matrix.
        # This will convert the 3D array into a 2D matrix of shape (N*J, N*J).
        blocks = [Bff_3D[:, :, j] for j in range(Bff_3D.shape[2])]
        from scipy.linalg import block_diag
        Bff = block_diag(*blocks)  # shape: (N*J, N*J)


        # --- Compute Bfm ---
        # pif_prime already computed for Bff; now compute pim_prime for final goods equation:
        pim_prime = pim_hat * mp.pim  # shape: (N, N, J)
        # Use the same tariff factor as in Bff:
        factorff = (shocks.tilde_tau_prime - 1) / shocks.tilde_tau_prime  # shape: (N, N, J)
        # Compute the 3D array for Bfm (final goods equation, intermediate goods part)
        Bfm_3D = mp.alpha[:, None, :] * factorff * pim_prime  # shape: (N, N, J)
        # Convert to a 2D block-diagonal matrix (each block is an (N, N) matrix per sector)
        blocks_Bfm = [Bfm_3D[:, :, j] for j in range(Bfm_3D.shape[2])]
        from scipy.linalg import block_diag
        Bfm = block_diag(*blocks_Bfm)  # shape: (N*J, N*J)

        # --- Compute Bmf ---
        # For the intermediate goods equation, the roles of importer and exporter reverse.
        # Compute pif_prime for intermediate goods:
        pif_prime = pif_hat * mp.pif  # shape: (N, N, J)
        # Swap the importer and exporter indices to get pif_prime(i, n, j)
        pif_prime_trans = pif_prime.transpose(1, 0, 2)  # shape: (N, N, J)
        # For the intermediate equation, the tariff factor is 1/tilde_tau_prime, with swapped indices:
        factor_m = 1 / shocks.tilde_tau_prime.transpose(1, 0, 2)  # shape: (N, N, J)
        # Compute the 3D array for Bmf:
        Bmf_3D = mp.beta[:, None, :] * (pif_prime_trans * factor_m)  # shape: (N, N, J)
        # Convert to a 2D block-diagonal matrix:
        blocks_Bmf = [Bmf_3D[:, :, j] for j in range(Bmf_3D.shape[2])]
        Bmf = block_diag(*blocks_Bmf)  # shape: (N*J, N*J)

        # --- Compute Bmm ---
        # Compute pim_prime for intermediate goods (reuse pim_prime from above)
        # Swap importer and exporter indices to get pim_prime(i, n, j)
        pim_prime_trans = pim_prime.transpose(1, 0, 2)  # shape: (N, N, J)
        # Use the same swapped tariff factor: 1/tilde_tau_prime with swapped indices
        factor_m = 1 / shocks.tilde_tau_prime.transpose(1, 0, 2)  # shape: (N, N, J)
        # Compute the 3D array for Bmm:
        Bmm_3D = mp.beta[:, None, :] * (pim_prime_trans * factor_m)  # shape: (N, N, J)
        # Convert to a 2D block-diagonal matrix:
        blocks_Bmm = [Bmm_3D[:, :, j] for j in range(Bmm_3D.shape[2])]
        Bmm = block_diag(*blocks_Bmm)  # shape: (N*J, N*J)
        
        # Concatenate the blocks to form the full B matrix.
        # Bff_2D, B_fm_2D, B_mf_2D, B_mm_2D are all of shape (N*J, N*J)

        # Concatenate horizontally to form the top and bottom rows:
        B_top = np.hstack((Bff, Bfm))   # shape: (N*J, 2*N*J)
        B_bottom = np.hstack((Bmf, Bmm))  # shape: (N*J, 2*N*J)

        # Now vertically stack them to get the full B matrix:
        B = np.vstack((B_top, B_bottom))         # shape: (2*N*J, 2*N*J)
        return B
    
    A_vec = calc_A()
    B_vec = calc_B()
    N, J = mp.alpha.shape
    I = np.eye(2*N*J)

    # Solve the linear system: (I - B_vec) * X_vec = A_vec
    X_vec = np.linalg.solve(I - B_vec, A_vec)  # X_vec will have shape (2*N*J,)

    # Now, extract Xf and Xm from X_vec. The first NJ elements correspond to Xf, and the rest to Xm
    Xf_vec = X_vec[:N*J]
    Xm_vec = X_vec[N*J:]

    # Reshape the vectors back to (N, J)
    Xf = Xf_vec.reshape(mp.alpha.shape)  # Final goods, shape: (N, J)
    Xm = Xm_vec.reshape(mp.alpha.shape)  # Intermediate goods, shape: (N, J)
    return Xf, Xm
