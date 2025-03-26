import numpy as np
from models import ModelParams, ModelShocks, ModelSol
from equations import (
    calc_c_hat,
    calc_Pu_hat,
    calc_piu_hat,
    calc_Xf_prime,
    calc_Xm_prime,
    calc_td_prime,
)


def solve_price_and_cost(
    w_hat,
    Pm_hat_init,
    mp: ModelParams,
    shocks: ModelShocks,
    max_iter=1000,
    tol=1e-6,
    mute=True,
):
    """
    Solve for the price index changes of intermediate goods
    """
    Pm_hat = Pm_hat_init.copy()
    for i in range(max_iter):
        c_hat = calc_c_hat(w_hat, Pm_hat, mp)
        Pm_hat_new = calc_Pu_hat(c_hat, "m", mp, shocks)
        diff = np.max(np.abs(Pm_hat_new - Pm_hat))
        if diff < tol:
            if not mute:
                print(f"Pm_hat converged in {i+1} iterations")
            c_hat = calc_c_hat(w_hat, Pm_hat_new, mp)
            break
        Pm_hat = Pm_hat_new

    return c_hat, Pm_hat


def solve_X_prime_analytical(
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

    # Reshape the vectors back to (N, J)
    Xf = Xf_vec.reshape(N, J)  # Final goods, shape: (N, J)
    Xm = Xm_vec.reshape(N, J)  # Intermediate goods, shape: (N, J)
    return Xf, Xm


def solve_X_prime(
    w_hat,
    pif_hat,
    pim_hat,
    td_prime,
    Xf_init,
    Xm_init,
    mp: ModelParams,
    shocks: ModelShocks,
    max_iter=1000,
    tol=1e-6,
    mute=True,
):
    """
    Solve for Xf_prime and Xm_prime
    given w_hat, pif_hat, pim_hat, tilde_tau_prime, td_prime

    Arguments for the model:
        w_hat: (N,) array of wage rate changes
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        tilde_tau_prime: (N, N, J) array of tariff after the shock
        td_prime: (N, N, J) array of trade deficit after the shock
        mp: ModelParams object containing model parameters
    Arguments for the solver:
        Xf_init: (N, J) array of initial guess for Xf_prime
        Xm_init: (N, J) array of initial guess for Xm_prime
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        mute: whether to print the progress of the solver
    Returns:
        Xf_prime: (N, J) array of final goods expenditure after the shock
        Xm_prime: (N, J) array of intermediate goods expenditure after the shock
    """
    Xf_prime = Xf_init.copy()
    Xm_prime = Xm_init.copy()
    for i in range(max_iter):
        Xf_prime_new = calc_Xf_prime(
            w_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, td_prime, mp, shocks
        )
        Xm_prime_new = calc_Xm_prime(
            pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks
        )
        diff_Xf = np.max(np.abs(Xf_prime_new - Xf_prime))
        diff_Xm = np.max(np.abs(Xm_prime_new - Xm_prime))
        if diff_Xf < tol and diff_Xm < tol:
            if not mute:
                print(f"X_prime converged in {i+1} iterations")
            break
        Xf_prime = Xf_prime_new
        Xm_prime = Xm_prime_new

    return Xf_prime, Xm_prime


def solve_equilibrium(
    mp: ModelParams,
    shocks: ModelShocks,
    numeraire_index: int,
    Xf_init: np.ndarray = None,
    Xm_init: np.ndarray = None,
    vfactor=-0.2,
    tol=1e-3,
    max_iter=1000,
    mute=True,
):
    """
    Solve for the equilibrium of the model

    Arguments:
        mp: ModelParams object containing model parameters
        shocks: ModelShocks object containing model shocks
        numeraire_index: index of the numeraire country
        Xf_init: (N, J) array of initial guess for Xf_prime
        Xm_init: (N, J) array of initial guess for Xm_prime
        vfactor: adjustment factor for the iteration process
        tol: tolerance for convergence
        max_iter: maximum number of iterations
        mute: whether to print the progress of the solver
    Returns:
        sol: ModelSol object
            containing the equilibrium values of endogenous variables
    """
    # Set the adjustment factor for the iteration process
    alpha = 0.1

    # Initialize the variables
    N, J = mp.N, mp.J
    VA = np.sum(mp.w0 * mp.L0)

    # Initlial guess for wage rate changes
    w_hat = np.ones(N)

    # Initial guess for the price index changes of intermediate goods
    Pm_hat = np.ones((N, J))

    # Initialize trade deficit
    td_prime = mp.td

    wfmax = 1.0

    for i in range(max_iter):
        # Calculate endogenous variables
        c_hat, Pm_hat = solve_price_and_cost(
            w_hat, Pm_hat, mp, shocks, max_iter=1000, tol=1e-6, mute=True
        )
        Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)
        pif_hat = calc_piu_hat(c_hat, Pf_hat, "f", mp, shocks)
        pim_hat = calc_piu_hat(c_hat, Pm_hat, "m", mp, shocks)
        Xf_prime, Xm_prime = solve_X_prime_analytical(
            w_hat, pif_hat, pim_hat, td_prime, mp, shocks
        )
        # Xf_prime, Xm_prime = solve_X_prime(
        #     w_hat,
        #     pif_hat,
        #     pim_hat,
        #     td_prime,
        #     Xf_init,
        #     Xm_init,
        #     mp,
        #     shocks,
        #     max_iter=1000,
        #     tol=1e-6,
        #     mute=True,
        # )

        # Calculate the trade deficit
        td_prime = calc_td_prime(
            pif_hat, pim_hat, Xf_prime, Xm_prime, mp, shocks
        )

        # ZW2 captures the difference
        # between the calculated trade deficit and the target trade deficit
        # normalized by the initial value added

        VA_prime = np.sum(mp.w0 * mp.L0 * w_hat)

        ZW2 = td_prime / VA_prime - mp.td / VA

        # Update the wage changes (numeraire country is excluded)
        w_hat_new = np.ones(N)
        w_hat_new = w_hat * np.exp(-alpha * ZW2)

        # w_hat_new = np.clip(w_hat_new, 1e-3, 1e3)

        # Check convergence (exclude numeraire country)
        wfmax = np.max(np.abs(w_hat_new - w_hat))

        # Update wage changes for the next iteration
        w_hat = w_hat_new / w_hat_new[numeraire_index]

        min_Xf_prime = np.min(Xf_prime)
        max_Xf_prime = np.max(Xf_prime)
        min_Xm_prime = np.min(Xm_prime)
        max_Xm_prime = np.max(Xm_prime)

        min_w_hat = np.min(w_hat)
        max_w_hat = np.max(w_hat)

        if not mute:
            print(
                f"Round {i+1}: wfmax={wfmax:.4f}, min_w_hat={min_w_hat:.4f}, max_w_hat={max_w_hat:.4f}, min_Xf_prime={min_Xf_prime:.4f}, max_Xf_prime={max_Xf_prime:.4f}, min_Xm_prime={min_Xm_prime:.4f}, max_Xm_prime={max_Xm_prime:.4f}"
            )

        if wfmax < tol:
            if not mute:
                print(f"Converged in {i+1} iterations")
            # Return the ModelSol object
            return ModelSol(
                mp,
                shocks,
                w_hat,
                c_hat,
                Pf_hat,
                Pm_hat,
                pif_hat,
                pim_hat,
                Xf_prime,
                Xm_prime,
            )

    if not mute:
        print("Failed to converge")
    return None


if __name__ == "__main__":
    from functions import generate_simple_params

    N, J = 2, 1
    mp = generate_simple_params()
    lambda_hat = np.ones((N, J))
    df_hat = np.ones((N, N, J)) * 1
    for i in range(N):
        for j in range(J):
            df_hat[i, i, j] = 1
    dm_hat = np.ones((N, N, J)) * 2
    for i in range(N):
        for j in range(J):
            dm_hat[i, i, j] = 1
    tilde_tau_prime = np.ones((N, N, J))

    shocks = ModelShocks(mp, lambda_hat, df_hat, dm_hat, tilde_tau_prime)

    w_hat = np.array([1.0, 1.0])
    Pm_hat_init = np.ones((2, 1))
    c_hat, Pm_hat = solve_price_and_cost(
        w_hat,
        Pm_hat_init,
        mp,
        shocks,
        max_iter=1000,
        tol=1e-6,
        mute=False,
    )

    Pf_hat = calc_Pu_hat(c_hat, "f", mp, shocks)

    print(f"c_hat: {c_hat}")
    print(f"Pm_hat: {Pm_hat}")
    print(f"Pf_hat: {Pf_hat}")
