import numpy as np
import autograd.numpy as anp
from models import ModelParams, ModelShocks, ModelSol, Usage


def calc_c_hat(w_hat, Pm_hat, mp):
    """
    Equation (7) in the paper

    Calculate the unit cost index changes (c_hat)
    given wage changes (w_hat) and intermediate input price changes (Pm_hat).

    Endogenous variables:
        w_hat: (N,) array of wage changes
        Pm_hat: (N, J) array of intermediate input price changes
    Returns:
        c_hat: (N, J) array of unit cost index changes

    Note: This version uses autograd.numpy (anp) for automatic differentiation.
    """
    log_w_hat = anp.log(w_hat)
    log_Pm_hat = anp.log(Pm_hat)

    # beta[n,j] * log(w_hat[n])
    wage_component = mp.beta * log_w_hat[:, anp.newaxis]

    # \sum_k gamma[n,k,j] * log(Pm_hat[n,k])
    input_component = anp.einsum("njk,nk->nj", mp.gamma, log_Pm_hat)

    log_c_hat = wage_component + input_component
    c_hat = anp.exp(log_c_hat)

    return c_hat



def calc_Pu_hat(c_hat, usage, mp, shocks):
    """
    Equation (8) in the paper

    Calculate price index changes (Pu_hat) given cost index changes (c_hat).

    usage: "f" for final demand, "m" for intermediate input demand.
           Raises ValueError if usage is not "f" or "m".

    Endogenous variables:
        c_hat: (N, J) array of unit cost index changes
    Returns:
        Pu_hat: (N, J) array of price index changes
    """
    if usage == "f":
        pi = mp.pif
        d_hat = shocks.df_hat
    elif usage == "m":
        pi = mp.pim
        d_hat = shocks.dm_hat
    else:
        raise ValueError(f"Invalid usage: {usage}")

    # Compute cost_index: For each (n, i, s):
    #   cost_index[n,i,s] = pi[n,i,s] * lambda_hat[n,s] * ( c_hat[n,s]*d_hat[n,i,s] )^(-theta[s])
    cost_index = (
        pi
        * shocks.lambda_hat[np.newaxis, :, :]  # broadcast lambda_hat: (1, N, J)
        * (c_hat[np.newaxis, :, :] * d_hat) ** (-mp.theta[np.newaxis, np.newaxis, :])
    )
    
    # Sum over the importer index (axis=1), then take power (-1/theta) elementwise
    Pu_hat = anp.sum(cost_index, axis=1) ** (-1 / mp.theta)
    return Pu_hat



def calc_piu_hat(c_hat, P_hat, usage, mp, shocks):
    """
    Equation (9) in the paper

    Calculate the expenditure share after the shock (piu_hat)
    given cost index changes (c_hat) and price index changes (P_hat).

    Parameters:
      c_hat : (N, J) array of unit cost index changes
      P_hat : (N, J) array of price index changes corresponding to the usage
      usage : "f" for final demand or "m" for intermediate input demand
      mp : ModelParams instance containing parameters such as theta and pif/pim (if needed)
      shocks : ModelShocks instance containing lambda_hat, df_hat, dm_hat, etc.

    Returns:
      piu_hat : (N, N, J) array of expenditure share after the shock
    """
    if usage == "f":
        d_hat = shocks.df_hat
    elif usage == "m":
        d_hat = shocks.dm_hat
    else:
        raise ValueError(f"Invalid usage: {usage}")

    # Compute cost_term.
    # Here c_hat has shape (N, J), and d_hat has shape (N, N, J).
    # We add an axis to c_hat so that the multiplication broadcasts correctly:
    #   c_hat[anp.newaxis, :, :] has shape (1, N, J)
    # Then the product (c_hat[anp.newaxis, :, :] * d_hat) has shape (N, N, J).
    # Raise it to the power -theta. Here mp.theta is assumed to have shape (J,).
    # We add two new axes so that mp.theta broadcasts with shape (1, 1, J).
    cost_term = (c_hat[anp.newaxis, :, :] * d_hat) ** (-mp.theta[anp.newaxis, anp.newaxis, :])

    # Numerator: multiply lambda_hat (shape (N, J)) with cost_term.
    # shocks.lambda_hat is assumed to be (N, J); add an axis so it becomes (1, N, J)
    numerator = shocks.lambda_hat[anp.newaxis, :, :] * cost_term

    # Denominator: P_hat is (N, J). Compute P_hat**(-theta) with mp.theta (shape (J,))
    # P_hat**(-mp.theta) has shape (N, J); add a new axis to make it (N, 1, J)
    denominator = P_hat ** (-mp.theta)
    denominator = denominator[:, anp.newaxis, :]

    # Expenditure share after the shock:
    piu_hat = numerator / denominator
    return piu_hat


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
        A = [vec(Af); vec(Am)]
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
        U = np.sum(factorff * pif_prime, axis=1)  # shape (N, J)
        V = mp.alpha  # shape (N, J)
        u, v = U.reshape(-1), V.reshape(-1)  # shape (NJ, )
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
        U = np.sum(factorff * pim_prime, axis=1)  # shape (N, J)
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
        B = np.einsum("nks,ink->nsik", V, U)
        Bmf = B.reshape((N * J, N * J))

        # --- Compute Bmf  ---
        U = pim_prime / shocks.tilde_tau_prime
        V = mp.gamma
        B = np.einsum("nks,ink->nsik", V, U)
        Bmm = B.reshape((N * J, N * J))

        # --- Assemble full B ---
        # Each of Bff, Bfm, Bmf, Bmm has shape (N*J, N*J)
        # Assemble into a 2×2 block matrix:
        B_top = np.hstack((Bff, Bfm))  # shape: (N*J, 2*N*J)
        B_bottom = np.hstack((Bmf, Bmm))  # shape: (N*J, 2*N*J)
        B = np.vstack((B_top, B_bottom))  # shape: (2*N*J, 2*N*J)
        return B

    I = np.eye(2 * N * J)
    A_vec = calc_A()
    B_vec = calc_B()
    # Solve the linear system: (I - B_vec) * X_vec = A_vec
    X_vec = np.linalg.solve(I - B_vec, A_vec)  # X_vec will have shape (2*N*J,)
    # Now, extract Xf and Xm from X_vec. The first NJ elements correspond to Xf, and the rest to Xm
    Xf_vec = X_vec[: N * J]
    Xm_vec = X_vec[N * J :]

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





def calc_td_prime(
    pif_hat, pim_hat, Xf_prime, Xm_prime, mp: ModelParams, shocks: ModelShocks
):
    """
    Equation (12) in the paper

    Calculate trade SURPLUS after the shock (td_prime)
    given expenditure share changes for final goods (pif_hat),
    expenditure share changes for intermediate goods (pim_hat),
    expenditure for final goods after the shock (Xf_prime),
    expenditure for intermediate goods after the shock (Xm_prime),
    and tariff rates after the shock (tilde_tau_prime).

    Endogenous variables:
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        Xf_prime: (N, J) array of expenditure for final goods after the shock
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
        tilde_tau_prime: (N, N, J) array of tariff rates after the shock
    Returns:
        td_prime: (N,) array of trade deficit after the shock
    """

    # Calculate pif_prime and pim_prime
    pif_prime = pif_hat * mp.pif
    pim_prime = pim_hat * mp.pim

    # import_volume array for each (importer, exporter, sector) triplet
    import_volume = (
        pif_prime * Xf_prime[:, np.newaxis, :]
        + pim_prime * Xm_prime[:, np.newaxis, :]
    ) / shocks.tilde_tau_prime

    # transpose import_volume to make (exporter, importer, sector) triplet
    export_volume = import_volume.transpose(1, 0, 2)

    # Calculate trade deficit for each (exporter, importer, sector) triplet
    trade_balance = import_volume - export_volume

    # Sum over exporting countries and sectors to obtain trade deficit for each importer
    td_prime = np.sum(trade_balance, axis=(1, 2))

    return td_prime

