import numpy as np


# ------------------------------------------------------------------------------
class OldModelParams:
    """
    Parameter class for the model.
    """

    def __init__(
        self,
        N: int,
        J: int,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        theta: np.ndarray,
        pi: np.ndarray,
        tilde_tau: np.ndarray,
        X: np.ndarray,
        VA: np.ndarray,
        D: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        N : int
            Number of countries
        J : int
            Number of sectors
        alpha : (N, J) array
            Share of expenditure on good j in country n
        beta : (N, J) array
            Share of labor inputs in production of good j in country n
        gamma : (N, J, J) array
            Intermediate input share
        theta : (J,) array
            Trade elasticity parameter for each sector
        pi : (N, N, J) array
            Expenditure share of imports from country i in sector j of country n
        tilde_tau : (N, N, J) array
            1 + tariff
        X : (N, J) array
            Expenditure on good j in country n
        VA : (N,) array
            Value added in country n
        D : (N,) array
            Exogenous trade deficit in country n
        """
        self.N = N
        self.J = J
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.pi = pi
        self.tilde_tau = tilde_tau
        self.X = X
        self.VA = VA
        self.D = D

    def check_consistency(self, tol=1e-6):
        """
        Check if the parameters are consistent
        """
        inconsistent = False

        print("Checking the consistency of parameters of the model...")

        # 1. Check the dimension of the data
        N = self.N
        J = self.J

        assert self.alpha.shape == (N, J)
        assert self.beta.shape == (N, J)
        assert self.gamma.shape == (N, J, J)
        assert self.theta.shape == (J,)
        assert self.pi.shape == (N, N, J)
        assert self.tilde_tau.shape == (N, N, J)
        assert self.VA.shape == (N,)
        assert self.D.shape == (N,)
        assert self.X.shape == (N, J)

        print("Check 1: The dimension of the data is consistent. ✅")

        # 2. Check consistency of alpha
        alpha = self.alpha
        # Check 2-1: For each n, the sum over j equals 1
        sum_by_n = np.sum(alpha, axis=1)
        check_sum = np.allclose(
            sum_by_n, 1, atol=tol
        )  # Allowing a small numerical tolerance
        if check_sum:
            print(
                "Check 2-1: For each country, the sum over secotrs equals 1 ✅"
            )
        else:
            print(
                "There are some country where the sum over sectors is not 1 ❌"
            )
            print(
                "The issue occurs at the following n indices:",
                np.where(np.abs(sum_by_n - 1) > tol)[0],
            )
            print(
                "The corresponding sums are:",
                sum_by_n[np.where(np.abs(sum_by_n - 1) > tol)[0]],
            )
            inconsistent = True

        # Check 2-2: Every value in alpha is between 0 and 1
        check_range = np.all((alpha >= 0) & (alpha <= 1))
        if check_range:
            print("Check 2-2: Every value in alpha is between 0 and 1 ✅")
        else:
            print("There are values in alpha that are not between 0 and 1 ❌")
            print(
                "These values are at positions:",
                np.where((alpha < 0) | (alpha > 1)),
            )
            print("The values are:", alpha[(alpha < 0) | (alpha > 1)])
            inconsistent = True

        # 3. Check consistency of beta
        beta = self.beta
        # Check 3-1: Every value in beta is between 0 and 1
        check_range = np.all((beta >= 0) & (beta <= 1))
        if check_range:
            print("Check 3: very value in beta is between 0 and 1 ✅")
        else:
            print("There are values in beta that are not between 0 and 1 ❌")
            print(
                "These values are at positions:",
                np.where((beta < 0) | (beta > 1)),
            )
            print("The values are:", beta[(beta < 0) | (beta > 1)])
            inconsistent = True

        # 4. Check consistency of gamma
        gamma = self.gamma
        # Check 4-1: Every value in gamma is between 0 and 1
        invalid_values = (gamma < 0) | (gamma > 1)

        if np.any(invalid_values):
            print("There are values in gamma that are not between 0 and 1 ❌")
            print("These values are at positions:", np.where(invalid_values))
            print("The values are:", gamma[invalid_values])
            inconsistent = True
        else:
            print("Check 4-1: Every value in gamma is between 0 and 1 ✅")

        # Check 4-2: sum(k) gamma[n,k,j] + beta[n,j] = 1
        temp = np.sum(gamma, axis=1) + beta
        is_valid = np.allclose(temp, 1, atol=tol)

        if is_valid:
            print(
                "Check 4-2: Condition satisfied: sum(k) gamma[n, k, j] + beta[n, j] = 1 ✅"
            )
        else:
            print(
                "Condition not satisfied: sum(k) gamma[n, k, j] + beta[n, j] ≠ 1 ❌"
            )
            print(
                "Positions where the condition fails:",
                np.where(~np.isclose(temp, 1, atol=tol)),
            )
            print(
                "Values that do not satisfy the condition:",
                temp[~np.isclose(temp, 1, atol=tol)],
            )
            inconsistent = True

        # 5. Check consistency of tilde_tau
        # Check 5-1: Diagonal elements of tilde_tau are one
        diag_tilde_tau = self.tilde_tau[range(self.N), range(self.N), :]
        if not np.allclose(diag_tilde_tau, 1.0, atol=tol):
            print(
                "Condition not satisfied: some diagonal elements of tilde_tau are not ones ❌",
                diag_tilde_tau,
            )
            inconsistent = True
        else:
            print(
                "Check 5: Condition satisfied: diagonal elements of tilde_tau are ones ✅"
            )

        # 6. Check consistency of pi
        pi = self.pi
        # Check 6-1: Every value in pi is between 0 and 1
        temp = np.sum(pi, axis=1)
        is_valid = np.allclose(temp, 1, atol=tol)

        if is_valid:
            print("Check 6: Condition satisfied: sum(i) pi[n, i, j] = 1 ✅")
        else:
            print("Condition not satisfied: sum(i) pi[n, i, j] ≠ 1 ❌")
            print(
                "Positions where the condition fails:",
                np.where(~np.isclose(temp, 1, atol=tol)),
            )
            print(
                "Values that do not satisfy the condition:",
                temp[~np.isclose(temp, 1, atol=tol)],
            )
            inconsistent = True

        # 7. Check consistency of VA
        VA = self.VA
        # Check 7-1: Every value in tilde_tau is greater than 0
        check_positive = np.all(VA > 0)
        if check_positive:
            print("Check 7: Every country's value added is greater than 0 ✅")
        else:
            print("There are values in VA that are less than or equal to 0 ❌")
            print("These values are at positions:", np.where(VA <= 0))
            print("The values are:", VA[VA <= 0])
            inconsistent = True

        if not inconsistent:
            print("The data is consistent. ✅")
            return True
        else:
            print("The data is inconsistent. ❌")
            return False

    def save_to_npz(self, filename: str) -> None:
        """
        Save the parameters to a np
        """
        np.savez(
            filename,
            N=self.N,
            J=self.J,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            theta=self.theta,
            pi=self.pi,
            tilde_tau=self.tilde_tau,
            X=self.X,
            VA=self.VA,
            D=self.D,
        )


class OldModelShocks(object):
    def __init__(self, params: OldModelParams, kappa_hat: np.ndarray) -> None:
        """
        Shock class for the model
        """
        self.params = params
        N = self.params.N
        J = self.params.J

        # kappa_hat_nij: trade cost shock
        self.kappa_hat = kappa_hat

    def check_consistency(self, tol=1e-6):
        """
        Check if the shocks are consistent
        """
        inconsistent = False
        print("Checking the consistency of shocks of the model...")

        # 1. Check the dimension of the data
        N = self.params.N
        J = self.params.J

        assert self.kappa_hat.shape == (N, N, J)

        print("Check 1: The dimension of the data is consistent. ✅")

        # 2. Check consistency of kappa_hat
        kappa_hat = self.kappa_hat
        # Check 2-1: Every value in kappa_hat is greater than 0
        check_positive = np.all(kappa_hat > 0)
        if check_positive:
            print("Check 2-1: Every value in kappa_hat is greater than 0 ✅")
        else:
            print(
                "There are values in kappa_hat that are less than or equal to 0 ❌"
            )
            print("These values are at positions:", np.where(kappa_hat <= 0))
            print("The values are:", kappa_hat[kappa_hat <= 0])
            inconsistent = True

        # Check 2-2: Diagonal elements of kappa_hat are one
        diag_kappa_hat = kappa_hat[range(N), range(N), :]
        if not np.allclose(diag_kappa_hat, 1.0, atol=tol):
            print(
                "Condition not satisfied: some diagonal elements of kappa_hat are not ones ❌",
                diag_kappa_hat,
            )
            inconsistent = True
        else:
            print(
                "Check 2-2: Condition satisfied: diagonal elements of kappa_hat are ones ✅"
            )

        if not inconsistent:
            print("kappa_hat is consistent. ✅")
            return True
        else:
            return False

    def save_to_npz(self, filename: str) -> None:
        """
        Save the shocks to a np
        """
        np.savez(
            filename,
            kappa_hat=self.kappa_hat,
        )


class OldModelSol(object):
    def __init__(
        self,
        params: OldModelParams,
        shocks: OldModelShocks,
        w_hat: np.ndarray,
        c_hat: np.ndarray,
        P_hat: np.ndarray,
        X_prime: np.ndarray,
        pi_prime: np.ndarray,
    ) -> None:
        """
        Solution class for the model
        """
        self.params = params
        self.shocks = shocks

        self.w_hat = w_hat
        self.c_hat = c_hat
        self.P_hat = P_hat
        self.X_prime = X_prime
        self.pi_prime = pi_prime

    def check_consistency(self, tol=1e-6):
        inconsistent = False
        print(
            "Checking the consistency of endogenous variables of the model..."
        )

        # 1. Check the dimension of the data
        N = self.params.N
        J = self.params.J

        assert self.w_hat.shape == (N,)
        assert self.c_hat.shape == (N, J)
        assert self.P_hat.shape == (N, J)
        assert self.X_prime.shape == (N, J)
        assert self.pi_prime.shape == (N, N, J)

        print("Check 1: The dimension of the data is consistent. ✅")

        # 2. Check consistency of w_hat
        w_hat = self.w_hat
        # Check 2: Every value in w_hat is greater than 0
        check_positive = np.all(w_hat > 0)
        if check_positive:
            print("Check 2: Every value in w_hat is greater than 0 ✅")
        else:
            print(
                "There are values in w_hat that are less than or equal to 0 ❌"
            )
            print("These values are at positions:", np.where(w_hat <= 0))
            print("The values are:", w_hat[w_hat <= 0])
            inconsistent = True

        # 3. Check consistency of c_hat
        c_hat = self.c_hat
        # Check 3: Every value in c_hat is greater than 0
        check_positive = np.all(c_hat > 0)
        if check_positive:
            print("Check 3: Every value in c_hat is greater than 0 ✅")
        else:
            print(
                "There are values in c_hat that are less than or equal to 0 ❌"
            )
            print("These values are at positions:", np.where(c_hat <= 0))
            print("The values are:", c_hat[c_hat <= 0])
            inconsistent = True

        # 4. Check consistency of P_hat
        P_hat = self.P_hat
        # Check 4: Every value in P_hat is greater than 0
        check_positive = np.all(P_hat > 0)
        if check_positive:
            print("Check 4: Every value in P_hat is greater than 0 ✅")
        else:
            print(
                "There are values in P_hat that are less than or equal to 0 ❌"
            )
            print("These values are at positions:", np.where(P_hat <= 0))
            print("The values are:", P_hat[P_hat <= 0])
            inconsistent = True

        # 5. Check consistency of X_prime
        X_prime = self.X_prime
        # Check 5: Every value in X_prime is greater than 0
        check_positive = np.all(X_prime > 0)
        if check_positive:
            print("Check 5: Every value in X_prime is greater than 0 ✅")
        else:
            print(
                "There are values in X_prime that are less than or equal to 0 ❌"
            )
            print("These values are at positions:", np.where(X_prime <= 0))
            print("The values are:", X_prime[X_prime <= 0])
            inconsistent = True

        # 6. Check consistency of pi_prime
        pi_prime = self.pi_prime
        # Check 6: Every value in pi_prime is between 0 and 1
        check_range = np.all((pi_prime >= 0) & (pi_prime <= 1))
        if check_range:
            print("Check 6: Every value in pi_prime is between 0 and 1 ✅")
        else:
            print(
                "There are values in pi_prime that are not between 0 and 1 ❌"
            )
            print(
                "These values are at positions:",
                np.where((pi_prime < 0) | (pi_prime > 1)),
            )
            print("The values are:", pi_prime[(pi_prime < 0) | (pi_prime > 1)])
            inconsistent = True

        # Check 7: sum(i) pi_prime[n, i, j] = 1
        temp = np.sum(pi_prime, axis=1)
        is_valid = np.allclose(temp, 1, atol=tol)

        if is_valid:
            print(
                "Check 7: Condition satisfied: sum(i) pi_prime[n, i, j] = 1 ✅"
            )
        else:
            print("Condition not satisfied: sum(i) pi_prime[n, i, j] ≠ 1 ❌")
            print(
                "Positions where the condition fails:",
                np.where(~np.isclose(temp, 1, atol=tol)),
            )
            print(
                "Values that do not satisfy the condition:",
                temp[~np.isclose(temp, 1, atol=tol)],
            )
            inconsistent = True

        if not inconsistent:
            print("The solution is consistent. ✅")
            return True
        else:
            print("The solution is inconsistent. ❌")
            return False

    def save_to_npz(self, filename: str) -> None:
        """
        Save the solution to a np
        """
        np.savez(
            filename,
            w_hat=self.w_hat,
            c_hat=self.c_hat,
            P_hat=self.P_hat,
            X_prime=self.X_prime,
            pi_prime=self.pi_prime,
        )


# ------------------------------------------------------------------------------
# Test consistency by using example parameters
if __name__ == "__main__":
    N = 3
    J = 2

    # alpha_nj: share of expenditure on good j in country n
    alpha = np.ones((N, J)) / J

    # beta_nj: share of labor inputs in production of good j in country n
    beta = np.ones((N, J)) * 0.3

    # gamma_nkj: share of intermediate inputs from sector k
    # in the production of good j in country n
    gamma = np.ones((N, J, J)) * 0.7 / J

    # theta_j: trade elasticity parameter for sector j
    theta = np.ones(J) * 8.0

    # tau_nij: tariff on imports of good j in country n from country i
    tau = np.ones((N, N, J)) * 0.1
    indices = np.arange(N)
    tau[indices, indices, :] = 0.0  # self-trade is tariff free

    # tilde_tau_nij: 1+tau_nij
    tilde_tau = tau + 1

    # pi_nij: expenditure share of imports from country i
    # within sector j of country n
    pi = np.ones((N, N, J)) / N

    # VA_n: value added in country n (w * L)
    VA = np.ones(N) * 100

    # D_n: exogenous trade deficit of country n
    D = np.zeros(N)

    # X_nj: expenditure on good j in country n
    X = np.ones((N, J)) * 100

    mp = OldModelParams(
        N, J, alpha, beta, gamma, theta, pi, tilde_tau, X, VA, D
    )
    mp.check_consistency()
