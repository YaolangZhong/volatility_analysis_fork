import numpy as np
from models import ModelParams, ModelShocks, ModelSol, Usage


def calc_c_hat(w_hat, Pm_hat, mp: ModelParams):
    """
    Equation (7) in the paper

    Calculate the unit cost index changes (c_hat)
    given wage changes (w_hat) and intermediate input price changes (Pm_hat).

    Endogenous variables:
        w_hat: (N,) array of wage changes
        Pm_hat: (N, J) array of intermediate input price changes
    Returns:
        c_hat: (N, J) array of unit cost index changes
    """
    log_w_hat = np.log(w_hat)
    log_Pm_hat = np.log(Pm_hat)

    # beta[n,j] * log(w_hat[n])
    wage_component = mp.beta * log_w_hat[:, np.newaxis]

    # \sum_k gamma[n,k,j] * log(Pm_hat[n,k])
    input_component = np.einsum("njk,nk->nj", mp.gamma, log_Pm_hat)

    log_c_hat = wage_component + input_component
    c_hat = np.exp(log_c_hat)

    return c_hat


def calc_Pu_hat(c_hat, usage: Usage, mp: ModelParams, shocks: ModelShocks):
    """
    Equation (8) in the paper

    Calculate price index changes (Pu_hat) given cost index changes (c_hat).

    usage = "f" for final demand, "m" for intermediate input demand
    Raise ValueError if usage is not "f" or "m".

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

    cost_index = (
        pi
        * shocks.lambda_hat[np.newaxis, :, :]
        * (c_hat[np.newaxis, :, :] * d_hat)
        ** -mp.theta[np.newaxis, np.newaxis, :]
    )
    Pu_hat = np.sum(cost_index, axis=1) ** (-1 / mp.theta)

    return Pu_hat


def calc_piu_hat(
    c_hat, P_hat, usage: Usage, mp: ModelParams, shocks: ModelShocks
):
    """
    Equation (9) in the paper

    Calculate the expenditure share after the shock (piu_prime)
    given cost index changes (c_hat) and price index changes (P_hat).

    usage = "f" for final demand, "m" for intermediate input demand
    Raise ValueError if usage is not "f" or "m".

    Endogenous variables:
        c_hat: (N, J) array of unit cost index changes
        P_hat: (N, J) array of price index changes corresponding to the usage
    Returns:
        piu_hat: (N, N, J) array of expenditure share after the shock
    """
    if usage == "f":
        d_hat = shocks.df_hat
    elif usage == "m":
        d_hat = shocks.dm_hat
    else:
        raise ValueError(f"Invalid usage: {usage}")

    # cost_term: (c_hat[n,j] * d_hat[n,i,j]) ** -theta[j] => [n,i,j]
    cost_term = (c_hat[np.newaxis, :, :] * d_hat) ** -mp.theta[
        np.newaxis, np.newaxis, :
    ]

    # numerator: lambda_hat[n,j] * cost_term[n,h,j] => [n,i,j]
    numerator = shocks.lambda_hat[np.newaxis, :, :] * cost_term

    # denominator: (P_hat[n,j]) ** -theta[j] => [n,i,j]
    denominator = P_hat**-mp.theta  # shape (N,J)
    denominator = denominator[:, np.newaxis, :]  # shape (N,N,J)

    # piu_prime[n,i,j] = numerator[n,i,j] / denominator[n,i,j]
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


def calc_Xf_prime(
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
    Equation (10) in the paper

    Calculate expenditure for final goods after the shock (Xf_prime)
    given wage changes (w_hat),
    expenditure share changes for final goods (pif_hat),
    expenditure share changes for intermediate goods (pim_hat),
    expenditure for final goods after the shock (Xf_prime),
    expenditure for intermediate goods after the shock (Xm_prime),
    tariff rates after the shock (tilde_tau_prime),
    and total demand after the shock (td_prime).

    Endogenous variables:
        w_hat: (N,) array of wage changes
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        Xf_prime: (N, J) array of expenditure for final goods after the shock
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
        tilde_tau_prime: (N, N, J) array of tariff rates after the shock
        td_prime: (N,) array of total demand after the shock
    Returns:
        Xf_prime: (N, J) array of expenditure for final goods after the shock
    """
    # Calculate pif_prime and pim_prime
    pif_prime = pif_hat * mp.pif
    pim_prime = pim_hat * mp.pim

    # Wage component: w_hat[n] * w0[n] * L0[n]
    wage_component = w_hat * mp.w0 * mp.L0

    # Tariff revenue component:
    # ((tau_tilde[n, i, j] - 1) / tau_tilde[n, i, j])
    # * (pim_prime[n, i, j] * Xm_prime[n, j]
    #   + pif_prime[n, i, j] * Xf_prime[n, j])
    # => Tariff revenue for each (country, sector) dyad
    tariff_revenue = (
        (shocks.tilde_tau_prime - 1) / shocks.tilde_tau_prime
    ) * (
        pim_prime * Xm_prime[:, np.newaxis, :]
        + pif_prime * Xf_prime[:, np.newaxis, :]
    )

    tariff_component = np.sum(tariff_revenue, axis=(1, 2))

    # Expenditure: wage_component[n] + tariff_revenue[n] + td_prime[n]
    expenditure = wage_component + tariff_component + td_prime

    # Xf_prime[n, j] = alpha[n, j] * expenditure[n]
    Xf_prime = mp.alpha * expenditure[:, np.newaxis]

    return Xf_prime


def calc_Xm_prime(
    pif_hat, pim_hat, Xf_prime, Xm_prime, mp: ModelParams, shocks: ModelShocks
):
    """
    Equation (11) in the paper

    Calculate expenditure for intermediate goods after the shock (Xm_prime)
    given expenditure share changes for final goods (pif_hat),
    expenditure share changes for intermediate goods (pim_hat),
    expenditure for final goods after the shock (Xf_prime),
    expenditure for intermediate goods after the shock (Xm_prime).

    Endogenous variables:
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        Xf_prime: (N, J) array of expenditure for final goods after the shock
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
        tilde_tau_prime: (N, N, J) array of tariff rates after the shock
    Returns:
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
    """
    # Calculate pif_prime and pim_prime
    pif_prime = pif_hat * mp.pif
    pim_prime = pim_hat * mp.pim

    # Output as final goods from country n:
    # (pif_prime[i, n, j] / tilde_tau_prime[i, n, j]) * Xf_prime[i, j]
    # => Output each (importer, exporter, sector) triplet
    output_final = (pif_prime / shocks.tilde_tau_prime) * Xf_prime[
        :, np.newaxis, :
    ]
    output_final = np.sum(output_final, axis=0)  # Sum over importers => [n, j]

    # Output as intermediate goods from country n:
    # (pim_prime[i, n, j] / tilde_tau_prime[i, n, j]) * Xm_prime[i, j]
    # => Output for each (importer, exporter, sector) triplet
    output_intermediate = (pim_prime / shocks.tilde_tau_prime) * Xm_prime[
        :, np.newaxis, :
    ]
    output_intermediate = np.sum(
        output_intermediate, axis=0
    )  # Sum over importers => [n, j]

    # Total output from country n:
    output = output_final + output_intermediate

    # Expenditure for intermediate goods:
    # gamma[n, h, j] * output[n, h]
    # => Sum over input sector => [n, j]
    Xm_component = mp.gamma * output[:, :, np.newaxis]
    Xm_prime = np.sum(Xm_component, axis=1)

    return Xm_prime


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


# ------------------------------------------------------------------------------
# Test the functions
def generate_test_parameters(N, J):
    # N = 2
    # J = 2
    alpha = np.ones((N, J)) / J
    beta = np.ones((N, J)) * 0.3
    gamma = np.ones((N, J, J)) * 0.7 / J
    theta = np.ones(J) * 8.0
    pif = np.ones((N, N, J)) / N
    pim = np.ones((N, N, J)) / N
    tilde_tau = np.ones((N, N, J)) * 1.1
    indices = np.arange(N)
    tilde_tau[indices, indices, :] = 1.0
    Xf = np.ones((N, J)) * 100
    Xm = np.ones((N, J)) * 100
    w0 = np.ones(N)
    L0 = np.ones(N) * 100
    td = np.zeros(N)

    params = ModelParams(
        N=N,
        J=J,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta=theta,
        pif=pif,
        pim=pim,
        tilde_tau=tilde_tau,
        Xf=Xf,
        Xm=Xm,
        w0=w0,
        L0=L0,
        td=td,
    )

    # Check consistency
    if params.check_consistency(mute=True):
        return params
    else:
        return None


def generate_test_shocks(params):
    N = params.N
    J = params.J
    df_hat = np.ones((N, J))
    dm_hat = np.ones((N, J))
    lambda_hat = np.ones((N, J))
    tilde_tau_prime = np.ones((N, N, J))
    shocks = ModelShocks(
        params=params,
        df_hat=df_hat,
        dm_hat=dm_hat,
        lambda_hat=lambda_hat,
        tilde_tau_prime=tilde_tau_prime,
    )
    return shocks


def test_equations(params):
    """
    Function to test the equations
    """
    N = params.N
    J = params.J

    # ========== Test calc_c_hat ==========
    w_hat = np.ones(N)
    w_hat[0] = 1.5 ** (1 / 0.3)

    Pm_hat = np.ones((N, J))
    Pm_hat[0,] = 1.5 ** (1 / 0.7)
    c_hat = calc_c_hat(w_hat, Pm_hat, params)

    # c_hat[0,] should be 1.5 * 1.5 = 2.25
    print("c_hat:", c_hat)

    # ========== Test calc_Pu_hat ==========

    # ...


"""
$$Q_n^s \equiv \sum_{i}\frac{(\pi_{in}^{sf}X_{i}^{sf}+\pi_{in}^{sm}X_{i}^{sm})}{1+\tau^s_{in}}$$
Then,HHI can be calculated as
$$HHI_n=\sum_k\left(\frac{Q_n^k}{\sum_s Q_n^s}\right)^2$$
"""


def calc_HHI(pim, pif, Xm, Xf, tau_tilde):
    """
    - pif, pim, tau_tilde: arrays of shape (i, n, s)
        where:
            i = importing country index,
            n = exporting country index,
            s = sector index.
    - Xm, Xf: arrays of shape (i, s) corresponding to intermediate and final goods expenditures
        for importing countries.
    The formula is:
      Q_n^s = sum_{i} [ (pif[i,n,s] * Xf[i,s] + pim[i,n,s] * Xm[i,s]) / (1 + tau_tilde[i,n,s]) ]
    and then the HHI for each exporting country n is:
      HHI_n = sum_s ( Q_n^s / (sum_{s'} Q_n^{s'}) )^2.

    Returns:
      HHI: an array of shape (n,) giving the HHI for each exporting country.
    """
    numerator = pif * Xf[:, None, :] + pim * Xm[:, None, :]  # shape: (N, N, J)
    term = numerator / (1 + tau_tilde)  # shape: (N, N, J)
    Q = term.sum(axis=0)  # shape: (N, J)
    # For each exporting country, sum over sectors to get the total Q:
    Q_total = Q.sum(axis=1)  # shape: (N,)
    # Compute HHI for each exporting country:
    HHI = np.sum((Q / Q_total[:, None]) ** 2, axis=1)  # shape: (N,)
    return HHI


def calc_W(sol: ModelSol):
    Pf_hat = sol.Pf_hat
    alpha = sol.params.alpha
    w_hat = sol.w_hat
    P_index = np.prod(Pf_hat**alpha, axis=1)
    W_hat = w_hat / P_index
    return W_hat




if __name__ == "__main__":
    params = generate_test_parameters(2, 1)
    shocks = generate_test_shocks(params)
    test_equations(params)
