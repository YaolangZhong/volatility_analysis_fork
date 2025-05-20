import numpy as np
from dataclasses import dataclass, field, fields
from typing import Any
import logging
from dataclasses import fields

# Configure logging; this will print INFO-level logs to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"   # only the actual log message
)
logger = logging.getLogger(__name__)
CHECK_CONSISTENCY_MUTE = True

# --- Mixin for .npz save/load for dataclasses ---
class NpzMixin:
    """Mixin to save/load all init dataclass fields to/from .npz."""
    def save_to_npz(self, filename: str) -> None:
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        np.savez(filename, **data)

    @classmethod
    def load_from_npz(cls, filename: str) -> "NpzMixin":
        data = np.load(filename, allow_pickle=True)
        kwargs = {}
        for f in fields(cls):
            if not f.init: 
                continue
            val = data[f.name]
            # cast ints if needed
            if f.type is int:
                val = int(val)
            kwargs[f.name] = val
        return cls(**kwargs)
    

@dataclass
class ModelParams(NpzMixin):
    """
    Class to store model parameters.
    N and S must match alpha.shape.
    """
    N: int
    S: int
    alpha:     np.ndarray  # shape (N, S)
    beta:      np.ndarray  # shape (N, S)
    gamma:     np.ndarray  # shape (N, S, S)
    theta:     np.ndarray  # shape (S,)
    pif:       np.ndarray  # shape (N, N, S)
    pim:       np.ndarray  # shape (N, N, S)
    tilde_tau: np.ndarray  # shape (N, N, S)
    Xf:        np.ndarray  # shape (N, S)
    Xm:        np.ndarray  # shape (N, S)
    V:        np.ndarray  # shape (N,)
    D:         np.ndarray  # shape (N,)
    country_list:  list[str] = field(default_factory=list)
    sector_list:   list[str] = field(default_factory=list)

    def __post_init__(self):
        # dimensions must match N, J
        if self.alpha.ndim != 2:
            raise ValueError("alpha must be 2D")
        if self.alpha.shape != (self.N, self.S):
            raise ValueError(f"alpha.shape {self.alpha.shape} does not match (N, S)=({self.N},{self.S})")
        self.check_consistency(mute=CHECK_CONSISTENCY_MUTE)

    def check_consistency(self, mute: bool = False, tol: float = 1e-6) -> bool:
        """
        Check consistency of model parameters.
        Returns True if all checks pass, False otherwise.
        """
        inconsistent = False
        if not mute:
            logger.info("Checking consistency of model parameters...")

        N, S = self.N, self.S

        # 1. Check dimensions
        try:
            assert self.alpha.shape == (N, S), f"alpha shape {self.alpha.shape} != ({N}, {S})"
            assert self.beta.shape == (N, S)
            assert self.gamma.shape == (N, S, S)
            assert self.theta.shape == (S,)
            assert self.pif.shape == (N, N, S)
            assert self.pim.shape == (N, N, S)
            assert self.tilde_tau.shape == (N, N, S)
            assert self.V.shape == (N,)
            assert self.D.shape == (N,)
        except AssertionError as e:
            logger.error("Dimension check failed: %s", e)
            inconsistent = True

        if not mute and not inconsistent:
            logger.info("Check 1: ✅ Dimensions of arrays are consistent.")

        # 2. Check consistency of alpha
        alpha_sum = np.sum(self.alpha, axis=1)
        if not np.allclose(alpha_sum, 1, atol=tol):
            indices = np.where(np.abs(alpha_sum - 1) > tol)[0]
            logger.error("Check 2-1: ❌ For some countries, alpha does not sum to 1. Offending indices: %s", indices)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 2-1: ✅ For each country, alpha sums to 1.")
        if not np.all((self.alpha >= 0) & (self.alpha <= 1)):
            logger.error("Check 2-2: ❌ Some values in alpha are not between 0 and 1.")
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 2-2: ✅ All alpha values are between 0 and 1.")

        # 3. Check consistency of beta
        if not np.all((self.beta >= 0) & (self.beta <= 1)):
            logger.error("Check 3: ❌ Some beta values are not between 0 and 1.")
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 3: ✅ All beta values are between 0 and 1.")

        # 4. Check consistency of gamma
        if np.any((self.gamma < 0) | (self.gamma > 1)):
            logger.error("Check 4-1: ❌ Some gamma values are not between 0 and 1.")
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 4-1: ✅ All gamma values are between 0 and 1.")
        # Verify that for each (n,j): sum_k gamma[n,j,k] + beta[n,j] = 1.
        gamma_beta_sum = np.sum(self.gamma, axis=2) + self.beta
        if not np.allclose(gamma_beta_sum, 1, atol=tol):
            problem_indices = np.where(~np.isclose(gamma_beta_sum, 1, atol=tol))
            logger.error("Check 4-2: ❌ For some (n,j), sum(gamma) + beta != 1 at indices: %s", problem_indices)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 4-2: ✅ For each (n,j), sum(gamma) + beta equals 1.")

        # 5. Check consistency of tilde_tau.
        if np.any(self.tilde_tau < 1):
            logger.error("Check 5-1: ❌ Some tilde_tau values are less than 1.")
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 5-1: ✅ All tilde_tau values are >= 1.")
        # Diagonal elements of tilde_tau must be exactly 1.
        diag_tilde_tau = self.tilde_tau[np.arange(N), np.arange(N), :]
        if not np.allclose(diag_tilde_tau, 1, atol=tol):
            logger.error("Check 5-2: ❌ Diagonal elements of tilde_tau are not equal to 1. Values: %s", diag_tilde_tau)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 5-2: ✅ Diagonal elements of tilde_tau are equal to 1.")

        # 6. Check consistency of pif and pim.
        pif_sum = np.sum(self.pif, axis=1)
        if not np.allclose(pif_sum, 1, atol=tol):
            logger.error("Check 6-1: ❌ For some (n,j), sum over exporters in pif is not 1.")
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 6-1: ✅ For each (n,j), pif sums to 1.")
        pim_sum = np.sum(self.pim, axis=1)
        if not np.allclose(pim_sum, 1, atol=tol):
            logger.error("Check 6-2: ❌ For some (n,j), sum over exporters in pim is not 1.")
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 6-2: ✅ For each (n,j), pim sums to 1.")

    # ------------------------------------------------------------------
    # Utility: unpack all init attributes in declaration order
    # ------------------------------------------------------------------
    def unpack(self):
        """
        Return a tuple of all init fields in their declared order.
        Useful for one‑liner assignments such as:
            (N, S, alpha, beta, gamma, theta, pif, pim, tilde_tau,
             Xf, Xm, V, D, countries, sectors) = params.unpack()
        """
        return tuple(getattr(self, f.name) for f in fields(self) if f.init)


@dataclass
class ModelShocks(NpzMixin):
    """
    ModelShocks holds the shock arrays; derives N,S from lambda_hat.shape.
    """
    lambda_hat:       np.ndarray  # shape (N, S)
    df_hat:           np.ndarray  # shape (N, N, S)
    dm_hat:           np.ndarray  # shape (N, N, S)
    tilde_tau_prime:  np.ndarray  # shape (N, N, S)

    _initialized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_initialized", True)
        self.check_consistency(tol=1e-6, mute=CHECK_CONSISTENCY_MUTE)

    @property
    def N(self) -> int:
        return self.lambda_hat.shape[0]

    @property
    def S(self) -> int:
        return self.lambda_hat.shape[1]

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # only re-check core arrays after init
        if getattr(self, "_initialized", False) and name in (
            "lambda_hat", "df_hat", "dm_hat", "tilde_tau_prime"
        ):
            self.check_consistency(tol=1e-6, mute=CHECK_CONSISTENCY_MUTE)

    def check_consistency(self, tol: float = 1e-6, mute: bool = False) -> bool:
        """
        Check consistency of model shocks.
        """
        inconsistent = False
        if not mute:
            logger.info("Checking consistency of model shocks...")

        N, S = self.N, self.S

        # 1. Check dimensions.
        try:
            assert self.lambda_hat.shape == (N, S), f"lambda_hat shape {self.lambda_hat.shape} != ({N}, {S})"
            assert self.df_hat.shape == (N, N, S), f"df_hat shape {self.df_hat.shape} != ({N}, {N}, {S})"
            assert self.dm_hat.shape == (N, N, S), f"dm_hat shape {self.dm_hat.shape} != ({N}, {N}, {S})"
            assert self.tilde_tau_prime.shape == (N, N, S), f"tilde_tau_prime shape {self.tilde_tau_prime.shape} != ({N}, {N}, {S})"
        except AssertionError as e:
            logger.error("Dimension check failed: %s", e)
            inconsistent = True

        if not mute and not inconsistent:
            logger.info("Check 1:   ✅ Dimensions of arrays are consistent.")

        # 2. Check consistency of lambda_hat (each value must be > 0).
        if not np.all(self.lambda_hat > 0):
            invalid_positions = np.where(self.lambda_hat <= 0)
            logger.error("Check 2:   ❌ Some lambda_hat values are <= 0 at positions %s. Values: %s",
                         invalid_positions, self.lambda_hat[invalid_positions])
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 2:   ✅ All lambda_hat values are > 0.")

        # 3. Check consistency of df_hat.
        # 3-1: All values > 0.
        if not np.all(self.df_hat > 0):
            invalid_positions = np.where(self.df_hat <= 0)
            logger.error("Check 3-1: ❌ Some df_hat values are <= 0 at positions %s. Values: %s",
                         invalid_positions, self.df_hat[invalid_positions])
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 3-1: ✅ All df_hat values are > 0.")
        # 3-2: Diagonal elements must be ones.
        diag_df_hat = self.df_hat[np.arange(N), np.arange(N), :]
        if not np.allclose(diag_df_hat, 1.0, atol=tol):
            logger.error("Check 3-2: ❌ Diagonal elements of df_hat are not equal to 1. They are: %s", diag_df_hat)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 3-2: ✅ Diagonal elements of df_hat are 1.")

        # 4. Check consistency of dm_hat.
        # 4-1: All values > 0.
        if not np.all(self.dm_hat > 0):
            invalid_positions = np.where(self.dm_hat <= 0)
            logger.error("Check 4-1: ❌ Some dm_hat values are <= 0 at positions %s. Values: %s", 
                         invalid_positions, self.dm_hat[invalid_positions])
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 4-1: ✅ All dm_hat values are > 0.")
        # 4-2: Diagonal elements must be ones.
        diag_dm_hat = self.dm_hat[np.arange(N), np.arange(N), :]
        if not np.allclose(diag_dm_hat, 1.0, atol=tol):
            logger.error("Check 4-2: ❌ Diagonal elements of dm_hat are not equal to 1. They are: %s", diag_dm_hat)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 4-2: ✅ Diagonal elements of dm_hat are 1.")

        # 5. Check consistency of tilde_tau_prime.
        # 5-1: Every value must be >= 1.
        if np.any(self.tilde_tau_prime < 1):
            invalid_indices = np.where(self.tilde_tau_prime < 1)
            logger.error("Check 5-1: ❌ Some tilde_tau_prime values are < 1 at positions %s. Values: %s", 
                         invalid_indices, self.tilde_tau_prime[invalid_indices])
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 5-1: ✅ All tilde_tau_prime values are >= 1.")
        # 5-2: Diagonal elements must be 1.
        diag_tau = self.tilde_tau_prime[np.arange(N), np.arange(N), :]
        if not np.allclose(diag_tau, 1.0, atol=tol):
            logger.error("Check 5-2: ❌ Diagonal elements of tilde_tau_prime are not equal to 1. They are: %s", diag_tau)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 5-2: ✅ Diagonal elements of tilde_tau_prime are 1.")

        if not inconsistent:
            if not mute:
                logger.info("Check OK:   ✅ The shocks are consistent.")
            return True
        else:
            logger.error("Check NG:   ❌ Ths shocks are inconsistent.")
            return False

    # ------------------------------------------------------------------
    # Utility: unpack all init attributes in declaration order
    # ------------------------------------------------------------------
    def unpack(self):
        """
        Return a tuple of all init fields in their declared order.
        Example:
            (lambda_hat, df_hat, dm_hat, tilde_tau_prime) = shocks.unpack()
        """
        return tuple(getattr(self, f.name) for f in fields(self) if f.init)


@dataclass
class ModelSol(NpzMixin):
    """
    Solution object for the model.
     w_hat : (N,) array (country,) 
        Wage change in country i
    c_hat : (N, S) array (country, sector)
        Cost change in sector s in country i
    Pf_hat : (N, S) array (country, sector)
        Final goods price change in sector s in each country i
    Pm_hat : (N, S) array (country, sector)
        Intermediate goods price change in sector s in country i
    pif_hat : (N, N, S) array (importer, exporter, sector)
        Final goods import expenditure changes in each sector in each country
    pim_hat : (N, N, S) array (importer, exporter, sector)
        Intermediate goods import expenditure changes in each sector in each country
    Xf_prime : (N, S) array (country, sector)
        Final goods expenditure in sector s in country i
    Xm_prime : (N, S) array (country, sector)
        Intermediate goods expenditure in sector i in country s

    """
    w_hat:     np.ndarray  # shape (N,)
    c_hat:     np.ndarray  # shape (N, S)
    Pf_hat:    np.ndarray  # shape (N, S)
    Pm_hat:    np.ndarray  # shape (N, S)
    pif_hat:   np.ndarray  # shape (N, N, S)
    pim_hat:   np.ndarray  # shape (N, N, S)
    Xf_prime:  np.ndarray  # shape (N, S)
    Xm_prime:  np.ndarray  # shape (N, S)
    p_index:   np.ndarray  # shape (N,)
    real_w:    np.ndarray  # shape (N,)
    D_prime:  np.ndarray  # shape (N,)
    

    # ------------------------------------------------------------------
    # Utility: unpack all init attributes in declaration order
    # ------------------------------------------------------------------
    def unpack(self):
        """
        Return a tuple of all init fields in their declared order.
        Example usage:
            (w_hat, c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime,
             Xm_prime, p_index, real_w, D_prime) = sol.unpack()
        """
        return tuple(getattr(self, f.name) for f in fields(self) if f.init)

    # NpzMixin provides save_to_npz and load_from_npz


@dataclass
class Model:
    params:      ModelParams
    shocks:      ModelShocks = field(init=False)
    sol:         ModelSol    = field(init=False)
    is_optimized: bool       = field(init=False, default=False)

    def __post_init__(self):
        self.reset_shocks()
        self.reset_sols()

    def reset_shocks(self) -> None:
        N,S = self.params.N, self.params.S
        self.shocks = ModelShocks(
            lambda_hat      = np.ones((N,S)),
            df_hat          = np.ones((N,N,S)),
            dm_hat          = np.ones((N,N,S)),
            tilde_tau_prime = np.ones((N,N,S)),
        )
        self.is_optimized = False

    def reset_sols(self) -> None:
        N,S = self.params.N, self.params.S
        self.sol = ModelSol(
            w_hat    = np.ones(N),
            c_hat    = np.ones((N,S)),
            Pf_hat   = np.ones((N,S)),
            Pm_hat   = np.ones((N,S)),
            pif_hat  = np.ones((N,N,S)),
            pim_hat  = np.ones((N,N,S)),
            Xf_prime = np.ones((N,S)),
            Xm_prime = np.ones((N,S)),
            p_index  = np.ones(N),
            real_w   = np.ones(N),
            D_prime = np.ones(N),
        )

    @classmethod
    def simple(cls, N:int=2, S:int=1) -> "Model":
        params = generate_simple_params(N=N, S=S)
        return cls(params)

    @classmethod
    def from_npz(cls, params_file, shocks_file, sols_file) -> "Model":
        p = ModelParams.load_from_npz(params_file)
        m = cls(p)
        m.shocks.reload_from_npz(shocks_file)
        m.sol.reload_from_npz(sols_file)
        return m






def generate_simple_params(N: int = 2, S: int = 1) -> ModelParams:
    """
    Generate fixed parameters for a simple test case with dimensions determined by N and S.
    
    Args:
        N (int, optional): Number of countries. Defaults to 2.
        S (int, optional): Number of sectors. Defaults to 1.
    
    Returns:
        ModelParams: An instance of ModelParams with values chosen so that the sum-to-one constraints are satisfied.
    """
    # For each country, set alpha so that the sum over sectors is 1.
    alpha = np.ones((N, S)) / S

    # Set beta to 0.4 for every (n, s).
    beta = np.full((N, S), 0.4)
    
    # For gamma, ensure that for each (n, s): sum_k gamma[n, s, k] = 0.6 (since 0.4 + 0.6 = 1).
    # Set each element in the (n,s) slice to 0.6/S.
    gamma = np.full((N, S, S), 0.6 / S)
    
    # Set theta as before.
    theta = np.full((S,), 8.0)
    
    # For pif and pim, each (n, s) slice over exporters must sum to 1, so each value is 1/N.
    pif = np.ones((N, N, S)) / N
    pim = np.ones((N, N, S)) / N
    
    # tilde_tau remains as ones (indicating no shock on tariffs).
    tilde_tau = np.ones((N, N, S))

    # Expenditures as final and intermediate goods initially all ones
    Xf = np.ones((N, S))
    Xm = np.ones((N, S))

    # Wage and labor supply are constant.
    V = np.full((N,), 500.0)

    # Trade deficits; zero for simplicity.
    D = np.zeros((N,))

    country_list = [f"country_{i+1}" for i in range(N)]
    sector_list  = [f"sector_{s+1}" for s in range(S)]

    params = ModelParams(
        N=N,
        S=S,
        country_list=country_list,
        sector_list=sector_list,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta=theta,
        pif=pif,
        pim=pim,
        tilde_tau=tilde_tau,
        Xf=Xf,
        Xm=Xm,
        V=V,
        D=D,
    )

    return params
