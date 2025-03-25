import numpy as np
from enum import Enum


# ------------------------------------------------------------------------------
class ModelParams(object):
    """
    Class to store model parameters
    """

    def __init__(
        self,
        N: int,
        J: int,
        alpha: np.ndarray,
        beta: np.ndarray,
        gamma: np.ndarray,
        theta: np.ndarray,
        pif: np.ndarray,
        pim: np.ndarray,
        tilde_tau: np.ndarray,
        Xf: np.ndarray,
        Xm: np.ndarray,
        w0: np.ndarray,
        L0: np.ndarray,
        td: np.ndarray,
    ) -> None:
        """
        ---------- Parameters ----------
        N : int
            Number of countries
        J : int
            Number of sectors
        alpha : (N, J) array (country, sector)
            Share of expenditure on good j in country n
        beta : (N, J) array (country, sector)
            Share of labor inputs in production of good j in country n
        gamma : (N, J, J) array (country n, using sector j, producing sector k)
            Share of intermediate inputs
            from sector k, in production of good j in country n
        theta : (J,) array (sector)
            Trade elasticity parameter for each sector
        pif : (N, N, J) array (importer, exporter, sector)
            Expenditure share of final goods imports
            from country i in sector j of country n
        pim : (N, N, J) array (importer, exporter, sector)
            Expenditure share of intermediate goods imports
            from country i in sector j of country n
        tilde_tau : (N, N, J) array (importer, exporter, sector)
            1 + tariff rate on imports from country i in sector j of country n
        Xf : (N, J) array (country, sector)
            Expenditure as final goods on good j in country n
        Xm : (N, J) array (country, sector)
            Expenditure as intermediate goods on good j in country n
        w0 : (N,) array (country)
            Initial wage in country n
        L0 : (N,) array (country)
            Initial labor supply in country n
        td : (N,) array (country)
            Initial trade deficit in country n

        ---------- Methods ----------
        check_consistency(tol=1e-6)
            Check consistency of model parameters
        save_to_npz(filename: str)
            Save model parameters to a .npz file

        """
        self.N = N
        self.J = J
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.pif = pif
        self.pim = pim
        self.tilde_tau = tilde_tau
        self.Xf = Xf
        self.Xm = Xm
        self.w0 = w0
        self.L0 = L0
        self.td = td

    def check_consistency(self, mute=False, tol=1e-6):
        """
        Check consistency of model parameters
        """
        inconsistent = False

        if not mute:
            print("Checking consistency of model parameters...")

        # 1. Check dimensions of arrays
        N = self.N
        J = self.J

        assert self.alpha.shape == (N, J)
        assert self.beta.shape == (N, J)
        assert self.gamma.shape == (N, J, J)
        assert self.theta.shape == (J,)
        assert self.pif.shape == (N, N, J)
        assert self.pim.shape == (N, N, J)
        assert self.tilde_tau.shape == (N, N, J)
        assert self.Xf.shape == (N, J)
        assert self.Xm.shape == (N, J)
        assert self.w0.shape == (N,)
        assert self.L0.shape == (N,)
        assert self.td.shape == (N,)

        if not mute:
            print("Check 1:   ✅ Dimensions of arrays are consistent.")

        # 2. Check consistency of alpha
        alpha = self.alpha

        # Check 2-1: For each n, the sum over j equals 1
        sum_by_n = np.sum(alpha, axis=1)
        check_sum = np.allclose(
            sum_by_n, 1, atol=tol
        )  # Allowing a small numerical tolerance
        if check_sum:
            if not mute:
                print(
                    "Check 2-1: ✅ For each country, the sum over secotrs equals 1"
                )
        else:
            print(
                "Check 2-1: ❌ There are some country where the sum over sectors is not 1"
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
            if not mute:
                print("Check 2-2: ✅ Every value in alpha is between 0 and 1")
        else:
            print(
                "Check 2-2: ❌ There are values in alpha that are not between 0 and 1"
            )
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
            if not mute:
                print("Check 3:   ✅ Every value in beta is between 0 and 1")
        else:
            print(
                "Check 3:   ❌ There are values in beta that are not between 0 and 1"
            )
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
            print(
                "Check 4-1: ❌ There are values in gamma that are not between 0 and 1"
            )
            print("These values are at positions:", np.where(invalid_values))
            print("The values are:", gamma[invalid_values])
            inconsistent = True
        else:
            if not mute:
                print("Check 4-1: ✅ Every value in gamma is between 0 and 1")

        # Check 4-2: sum(k) gamma[n,k,j] + beta[n,j] = 1
        temp = np.sum(gamma, axis=1) + beta
        is_valid = np.allclose(temp, 1, atol=tol)

        if is_valid:
            if not mute:
                print(
                    "Check 4-2: ✅ Condition satisfied: sum(k) gamma[n, k, j] + beta[n, j] = 1"
                )
        else:
            print(
                "Check 4-2: ❌ Condition not satisfied: sum(k) gamma[n, k, j] + beta[n, j] ≠ 1"
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
        tilde_tau = self.tilde_tau
        # Check 5-1: Every value in tilde_tau is no less than 1
        invalid_values = tilde_tau < 1
        if np.any(invalid_values):
            print(
                "Check 5-1: ❌ There are values in tilde_tau that are less than 1"
            )
            print("These values are at positions:", np.where(invalid_values))
            print("The values are:", tilde_tau[invalid_values])
            inconsistent = True
        else:
            if not mute:
                print(
                    "Check 5-1: ✅ Every value in tilde_tau is greater than 1"
                )

        # Check 5-2: Diagonal elements of tilde_tau are one
        diag_tilde_tau = self.tilde_tau[range(self.N), range(self.N), :]
        if not np.allclose(diag_tilde_tau, 1.0, atol=tol):
            print(
                "Condition not satisfied: some diagonal elements of tilde_tau are not ones ❌",
                diag_tilde_tau,
            )
            inconsistent = True
        else:
            if not mute:
                print(
                    "Check 5-2: ✅ Condition satisfied: diagonal elements of tilde_tau are ones"
                )

        # 6. Check consistency of pi
        pif = self.pif
        # Check 6-1: Every value in pif is between 0 and 1
        temp = np.sum(pif, axis=1)
        is_valid = np.allclose(temp, 1, atol=tol)

        if is_valid:
            if not mute:
                print(
                    "Check 6-1: ✅ Condition satisfied: sum(i) pif[n, i, j] = 1"
                )
        else:
            print(
                "Check 6-1: ❌ Condition not satisfied: sum(i) pif[n, i, j] ≠ 1"
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

        # Check 6-2: Every value in pim is between 0 and 1
        temp = np.sum(pif, axis=1)
        is_valid = np.allclose(temp, 1, atol=tol)

        if is_valid:
            if not mute:
                print(
                    "Check 6-2: ✅ Condition satisfied: sum(i) pim[n, i, j] = 1"
                )
        else:
            print(
                "Check 6-2: ❌ Condition not satisfied: sum(i) pim[n, i, j] ≠ 1"
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

        # 7. Check consistency of w0 and L0
        w0 = self.w0
        L0 = self.L0
        # Check 7-1: Every value in w0 is greater than 0
        check_positive = np.all(w0 > 0)
        if check_positive:
            if not mute:
                print("Check 7-1: ✅ Every country's wage is greater than 0")
        else:
            print(
                "Check 7-1: ❌ There are values in w0 that are less than or equal to 0"
            )
            print("These values are at positions:", np.where(w0 <= 0))
            print("The values are:", w0[w0 <= 0])
            inconsistent = True

        # Check 7-2: Every value in L0 is greater than 0
        check_positive = np.all(L0 > 0)
        if check_positive:
            if not mute:
                print(
                    "Check 7-2: ✅ Every country's labor endowment is greater than 0"
                )
        else:
            print(
                "Check 7-2: ❌ There are values in L0 that are less than or equal to 0"
            )
            print("These values are at positions:", np.where(L0 <= 0))
            print("The values are:", L0[L0 <= 0])
            inconsistent = True

        if not inconsistent:
            if not mute:
                print("Check OK:  ✅ The data is consistent.")
            return True
        else:
            print("Check NG:  ❌The data is inconsistent.")
            return False

    def save_to_npz(self, filename: str) -> None:
        """
        Save model parameters to a .npz file
        """
        np.savez(
            filename,
            N=self.N,
            J=self.J,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            theta=self.theta,
            pif=self.pif,
            pim=self.pim,
            tilde_tau=self.tilde_tau,
            Xf=self.Xf,
            Xm=self.Xm,
            w0=self.w0,
            L0=self.L0,
            td=self.td,
        )

    @classmethod
    def load_from_npz(cls, filename: str) -> "ModelParams":
        """
        Load model parameters from a .npz file and return a ModelParams instance.
        
        The .npz file must contain the keys:
          'N', 'J', 'alpha', 'beta', 'gamma', 'theta',
          'pif', 'pim', 'tilde_tau', 'Xf', 'Xm', 'w0', 'L0', 'td'
        """
        data = np.load(filename, allow_pickle=True)
        # Make sure to extract values from data.
        N = int(data["N"])
        J = int(data["J"])
        alpha = data["alpha"]
        beta = data["beta"]
        gamma = data["gamma"]
        theta = data["theta"]
        pif = data["pif"]
        pim = data["pim"]
        tilde_tau = data["tilde_tau"]
        Xf = data["Xf"]
        Xm = data["Xm"]
        w0 = data["w0"]
        L0 = data["L0"]
        td = data["td"]
        
        return cls(N, J, alpha, beta, gamma, theta, pif, pim, tilde_tau, Xf, Xm, w0, L0, td)


class ModelShocks(object):
    def __init__(
        self,
        params: ModelParams,
        lambda_hat: np.ndarray,
        df_hat: np.ndarray,
        dm_hat: np.ndarray,
        tilde_tau_prime: np.ndarray,
    ) -> None:
        """
        ---------- Parameters ----------
        params : ModelParams
            Model parameters
        lambda_hat : (N, J) array (country, sector)
            Productivity shocks
            in sector j in country n
        df_hat : (N, N, J) array (importer, exporter, sector)
            Final goods trade cost shocks
            in sector j, from country i to country n
        dm_hat : (N, N, J) array (importer, exporter, sector)
            Intermediate goods trade cost shocks
            in sector j, from country i to country n
        tilde_tau_prime : (N, N, J) array (importer, exporter, sector)
            1 + tariff rate on imports
            from country i in sector j of country n after the shock

        ---------- Methods ----------
        check_consistency(tol=1e-6)
            Check consistency of model shocks
        save_to_npz(filename: str)
            Save model shocks to a .npz file
        """
        self.params = params

        self.lambda_hat = lambda_hat
        self.df_hat = df_hat
        self.dm_hat = dm_hat
        self.tilde_tau_prime = tilde_tau_prime

    def check_consistency(self, tol=1e-6):
        """
        Check consistency of model shocks
        """
        inconsistent = False
        print("Checking consistency of model shocks...")

        # 1. Check dimensions of arrays
        N = self.params.N
        J = self.params.J

        assert self.lambda_hat.shape == (N, J)
        assert self.df_hat.shape == (N, N, J)
        assert self.dm_hat.shape == (N, N, J)

        print("Check 1:   ✅ Dimensions of arrays are consistent.")

        # 2. Check consistency of lambda_hat
        lambda_hat = self.lambda_hat
        # Check 2-1: Every value in lambda_hat is greater than 0
        check_positive = np.all(lambda_hat > 0)
        if check_positive:
            print("Check 2:   ✅ Every value in lambda_hat is greater than 0")
        else:
            print(
                "Check 2:   ❌ There are values in lambda_hat that are less than or equal to 0"
            )
            print("These values are at positions:", np.where(lambda_hat <= 0))
            print("The values are:", lambda_hat[lambda_hat <= 0])
            inconsistent = True

        # 3. Check consistency of df_hat
        df_hat = self.df_hat
        # Check 3-1: Every value in df_hat is greater than 0
        check_positive = np.all(df_hat > 0)
        if check_positive:
            print("Check 3-1: ✅ Every value in df_hat is greater than 0")
        else:
            print(
                "Check 3-1: ❌ There are values in df_hat that are less than or equal to 0"
            )
            print("These values are at positions:", np.where(df_hat <= 0))
            print("The values are:", df_hat[df_hat <= 0])
            inconsistent = True

        # Check 3-2: Diagonal elements of df_hat are ones
        diag_df_hat = df_hat[range(N), range(N), :]
        if not np.allclose(diag_df_hat, 1.0, atol=tol):
            print(
                "Check 3-2: ❌ Condition not satisfied: some diagonal elements of df_hat are not ones",
                diag_df_hat,
            )
            inconsistent = True
        else:
            print(
                "Check 3-2: ✅ Condition satisfied: diagonal elements of df_hat are ones"
            )

        # 4. Check consistency of dm_hat
        dm_hat = self.dm_hat
        # Check 4-1: Every value in dm_hat is greater than 0
        check_positive = np.all(dm_hat > 0)
        if check_positive:
            print("Check 4-1: ✅ Every value in dm_hat is greater than 0")
        else:
            print(
                "Check 4-1: ❌ There are values in dm_hat that are less than or equal to 0"
            )
            print("These values are at positions:", np.where(dm_hat <= 0))
            print("The values are:", dm_hat[dm_hat <= 0])
            inconsistent = True

        # Check 4-2: Diagonal elements of dm_hat are ones
        diag_dm_hat = dm_hat[range(N), range(N), :]
        if not np.allclose(diag_dm_hat, 1.0, atol=tol):
            print(
                "Check 4-2: ❌ Condition not satisfied: some diagonal elements of dm_hat are not ones",
                diag_dm_hat,
            )
            inconsistent = True
        else:
            print(
                "Check 4-2: ✅ Condition satisfied: diagonal elements of dm_hat are ones"
            )

        # 5. Check consistency of tilde_tau_prime
        tilde_tau_prime = self.tilde_tau_prime
        # Check 5-1: Every value in tilde_tau_prime is no less than 1
        invalid_values = tilde_tau_prime < 1
        if np.any(invalid_values):
            print(
                "Check 5-1: ❌ There are values in tilde_tau_prime that are less than 1"
            )
            print("These values are at positions:", np.where(invalid_values))
            print("The values are:", tilde_tau_prime[invalid_values])
            inconsistent = True
        else:
            print(
                "Check 5-1: ✅ Every value in tilde_tau_prime is greater than 1"
            )

        # Check 5-2: Diagonal elements of tilde_tau_prime are one
        diag_tilde_tau_prime = tilde_tau_prime[range(N), range(N), :]
        if not np.allclose(diag_tilde_tau_prime, 1.0, atol=tol):
            print(
                "Check 5-2: ❌ Condition not satisfied: some diagonal elements of tilde_tau_prime are not ones",
                diag_tilde_tau_prime,
            )
            inconsistent = True
        else:
            print(
                "Check 5-2: ✅ Condition satisfied: diagonal elements of tilde_tau_prime are ones"
            )

        if not inconsistent:
            print("Check OK:  ✅ Shocks are consistent.")
            return True
        else:
            print("Check NG:  ❌ Shocks are inconsistent.")
            return False

    def save_to_npz(self, filename: str) -> None:
        """
        Save model shocks to a .npz file
        """
        np.savez(
            filename,
            lambda_hat=self.lambda_hat,
            df_hat=self.df_hat,
            dm_hat=self.dm_hat,
            tilde_tau_prime=self.tilde_tau_prime,
        )

    @classmethod
    def load_from_npz(cls, filename: str, params: ModelParams) -> "ModelShocks":
        """
        Load model shocks from a .npz file and return a ModelShocks instance.

        Parameters:
          filename : str
              The path to the .npz file.
          params : ModelParams
              The ModelParams instance associated with these shocks.

        Returns:
          An instance of ModelShocks with the loaded data.
        """
        data = np.load(filename, allow_pickle=True)
        lambda_hat = data["lambda_hat"]
        df_hat = data["df_hat"]
        dm_hat = data["dm_hat"]
        tilde_tau_prime = data["tilde_tau_prime"]
        return cls(params, lambda_hat, df_hat, dm_hat, tilde_tau_prime)

class ModelSol(object):
    def __init__(
        self,
        params: ModelParams,
        shocks: ModelShocks,
        w_hat: np.ndarray,
        c_hat: np.ndarray,
        Pf_hat: np.ndarray,
        Pm_hat: np.ndarray,
        pif_hat: np.ndarray,
        pim_hat: np.ndarray,
        Xf_prime: np.ndarray,
        Xm_prime: np.ndarray,
    ) -> None:
        """
        ---------- Parameters ----------
        params : ModelParams
            Model parameters
        shocks : ModelShocks
            Model shocks
        w_hat : (N,) array (country,)
            Wage change in each country
        c_hat : (N, J) array (country, sector)
            Cost change in each sector in each country
        Pf_hat : (N, J) array (country, sector)
            Final goods price change in each sector in each country
        Pm_hat : (N, J) array (country, sector)
            Intermediate goods price change in each sector in each country
        pif_hat : (N, N, J) array (importer, exporter, sector)
            Final goods import expenditure changes in each sector in each country
        pim_hat : (N, N, J) array (importer, exporter, sector)
            Intermediate goods import expenditure changes in each sector in each country
        Xf_prime : (N, J) array (country, sector)
            Final goods expenditure in each sector in each country
        Xm_prime : (N, J) array (country, sector)
            Intermediate goods expenditure in each sector in each country

        ---------- Methods ----------
        save_to_npz(filename: str)
            Save model solution to a .npz file
        """
        self.params = params
        self.shocks = shocks

        self.w_hat = w_hat
        self.c_hat = c_hat
        self.Pf_hat = Pf_hat
        self.Pm_hat = Pm_hat
        self.pif_hat = pif_hat
        self.pim_hat = pim_hat
        self.Xf_prime = Xf_prime
        self.Xm_prime = Xm_prime

    def save_to_npz(self, filename: str) -> None:
        """
        Save model solution to a .npz file
        """
        np.savez(
            filename,
            w_hat=self.w_hat,
            c_hat=self.c_hat,
            Pf_hat=self.Pf_hat,
            Pm_hat=self.Pm_hat,
            pif_hat=self.pif_hat,
            pim_hat=self.pim_hat,
            Xf_prime=self.Xf_prime,
            Xm_prime=self.Xm_prime,
        )

    @classmethod
    def load_from_npz(cls, filename: str, params: ModelParams, shocks: ModelShocks) -> "ModelSol":
        """
        Load a model solution from a .npz file and return a ModelSol instance.
        This function assumes that the npz file contains the following keys:
          'params', 'shocks', 'w_hat', 'c_hat', 'Pf_hat', 'Pm_hat',
          'pif_hat', 'pim_hat', 'Xf_prime', 'Xm_prime'
        """
        data = np.load(filename, allow_pickle=True)
        # If the parameters and shocks were saved, load them;
        # otherwise, set them to None or handle appropriately.
        params = params
        shocks = shocks
        w_hat = data["w_hat"]
        c_hat = data["c_hat"]
        Pf_hat = data["Pf_hat"]
        Pm_hat = data["Pm_hat"]
        pif_hat = data["pif_hat"]
        pim_hat = data["pim_hat"]
        Xf_prime = data["Xf_prime"]
        Xm_prime = data["Xm_prime"]

        return cls(params, shocks, w_hat, c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime)

class Usage(Enum):
    """
    Class to define the usage of the product
    """

    F = "f"
    M = "m"


# --------------------------------------------------------------------------------
# Test consistency function by using example parameters
def main():
    N = 3
    J = 2
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
    params.check_consistency()


def compute_W(w_hat, Pf_hat, alpha):
    
    P_index = np.prod(Pf_hat ** alpha, axis=1)
    W_hat = w_hat/ P_index
    
    return W_hat


if __name__ == "__main__":
    main()
