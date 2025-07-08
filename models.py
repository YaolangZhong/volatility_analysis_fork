import numpy as np
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Tuple, NamedTuple, Optional
import logging

# Configure logging; this will print INFO-level logs to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"   # only the actual log message
)
logger = logging.getLogger(__name__)
CHECK_CONSISTENCY_MUTE = True

# --- Variable Registry System ---
class VarInfo(NamedTuple):
    """Information about a model variable."""
    name: str
    shape: str
    meaning: str
    indexing: str
    component: str  # 'params', 'shocks', or 'sol'
    data_type: str = "float"

# Comprehensive registry of all model variables
MODEL_VARIABLES = {
    # === ModelParams Variables ===
    'N': VarInfo('N', 'scalar', 'Number of countries', 'single value', 'params', 'int'),
    'S': VarInfo('S', 'scalar', 'Number of sectors', 'single value', 'params', 'int'),
    'alpha': VarInfo('alpha', '(N, S)', 'Share of sector s in country n final consumption', 'alpha[country, sector]', 'params'),
    'beta': VarInfo('beta', '(N, S)', 'Share of value added in sector s production in country n', 'beta[country, sector]', 'params'),
    'gamma': VarInfo('gamma', '(N, S, S)', 'Share of sector k in sector s intermediate inputs in country n', 'gamma[country, using_sector, input_sector]', 'params'),
    'theta': VarInfo('theta', '(S,)', 'Trade elasticity parameter for sector s', 'theta[sector]', 'params'),
    'pif': VarInfo('pif', '(N, N, S)', 'Final goods trade share from exporter i to importer n in sector s', 'pif[importer, exporter, sector]', 'params'),
    'pim': VarInfo('pim', '(N, N, S)', 'Intermediate goods trade share from exporter i to importer n in sector s', 'pim[importer, exporter, sector]', 'params'),
    'pi': VarInfo('pi', '(N, N, S)', 'Total trade share from exporter i to importer n in sector s', 'pi[importer, exporter, sector]', 'params'),
    'tilde_tau': VarInfo('tilde_tau', '(N, N, S)', 'Iceberg trade costs from exporter i to importer n in sector s', 'tilde_tau[importer, exporter, sector]', 'params'),
    'Xf': VarInfo('Xf', '(N, S)', 'Final goods expenditure in sector s in country n', 'Xf[country, sector]', 'params'),
    'Xm': VarInfo('Xm', '(N, S)', 'Intermediate goods expenditure in sector s in country n', 'Xm[country, sector]', 'params'),
    'X': VarInfo('X', '(N, S)', 'Total expenditure in sector s in country n (Xf + Xm)', 'X[country, sector]', 'params'),
    'V': VarInfo('V', '(N,)', 'Value added (wage bill) in country n', 'V[country]', 'params'),
    'D': VarInfo('D', '(N,)', 'Trade deficit in country n', 'D[country]', 'params'),
    'country_list': VarInfo('country_list', 'List[str]', 'List of country names', 'country_list[country_index]', 'params', 'str'),
    'sector_list': VarInfo('sector_list', 'List[str]', 'List of sector names', 'sector_list[sector_index]', 'params', 'str'),
    
    # === ModelShocks Variables (Hat Algebra) ===
    'lambda_hat': VarInfo('lambda_hat', '(N, S)', 'Productivity shock in sector s in country n (multiplicative)', 'lambda_hat[country, sector]', 'shocks'),
    'df_hat': VarInfo('df_hat', '(N, N, S)', 'Final goods trade cost shock from exporter i to importer n in sector s', 'df_hat[importer, exporter, sector]', 'shocks'),
    'dm_hat': VarInfo('dm_hat', '(N, N, S)', 'Intermediate goods trade cost shock from exporter i to importer n in sector s', 'dm_hat[importer, exporter, sector]', 'shocks'),
    'tilde_tau_hat': VarInfo('tilde_tau_hat', '(N, N, S)', 'Tariff shock from exporter i to importer n in sector s', 'tilde_tau_hat[importer, exporter, sector]', 'shocks'),
    
    # === ModelSol Variables (Equilibrium Solutions) ===
    'w_hat': VarInfo('w_hat', '(N,)', 'Wage change in country n relative to baseline', 'w_hat[country]', 'sol'),
    'c_hat': VarInfo('c_hat', '(N, S)', 'Unit cost change in sector s in country n', 'c_hat[country, sector]', 'sol'),
    'Pf_hat': VarInfo('Pf_hat', '(N, S)', 'Final goods price index change in sector s in country n', 'Pf_hat[country, sector]', 'sol'),
    'Pm_hat': VarInfo('Pm_hat', '(N, S)', 'Intermediate goods price index change in sector s in country n', 'Pm_hat[country, sector]', 'sol'),
    'pif_hat': VarInfo('pif_hat', '(N, N, S)', 'Final goods trade share change from exporter i to importer n in sector s', 'pif_hat[importer, exporter, sector]', 'sol'),
    'pim_hat': VarInfo('pim_hat', '(N, N, S)', 'Intermediate goods trade share change from exporter i to importer n in sector s', 'pim_hat[importer, exporter, sector]', 'sol'),
    'pif_prime': VarInfo('pif_prime', '(N, N, S)', 'New final goods trade share from exporter i to importer n in sector s', 'pif_prime[importer, exporter, sector]', 'sol'),
    'pim_prime': VarInfo('pim_prime', '(N, N, S)', 'New intermediate goods trade share from exporter i to importer n in sector s', 'pim_prime[importer, exporter, sector]', 'sol'),
    'Xf_prime': VarInfo('Xf_prime', '(N, S)', 'New final goods expenditure in sector s in country n', 'Xf_prime[country, sector]', 'sol'),
    'Xm_prime': VarInfo('Xm_prime', '(N, S)', 'New intermediate goods expenditure in sector s in country n', 'Xm_prime[country, sector]', 'sol'),
    'X_prime': VarInfo('X_prime', '(N, S)', 'New total expenditure in sector s in country n', 'X_prime[country, sector]', 'sol'),
    'p_index': VarInfo('p_index', '(N,)', 'Consumer price index change in country n', 'p_index[country]', 'sol'),
    'real_w_hat': VarInfo('real_w_hat', '(N,)', 'Real wage change in country n', 'real_w_hat[country]', 'sol'),
    'D_prime': VarInfo('D_prime', '(N,)', 'New trade deficit in country n', 'D_prime[country]', 'sol'),
    'Xf_prod_prime': VarInfo('Xf_prod_prime', '(N, S)', 'New final goods production value in sector s in country n', 'Xf_prod_prime[country, sector]', 'sol'),
    'Xm_prod_prime': VarInfo('Xm_prod_prime', '(N, S)', 'New intermediate goods production value in sector s in country n', 'Xm_prod_prime[country, sector]', 'sol'),
    'X_prod_prime': VarInfo('X_prod_prime', '(N, S)', 'New total production value in sector s in country n', 'X_prod_prime[country, sector]', 'sol'),
    'I_prime': VarInfo('I_prime', '(N,)', 'New income in country n', 'I_prime[country]', 'sol'),
    'output_prime': VarInfo('output_prime', '(N, S)', 'New output demand in sector s in country n', 'output_prime[country, sector]', 'sol'),
    'real_I_prime': VarInfo('real_I_prime', '(N,)', 'New real income in country n', 'real_I_prime[country]', 'sol'),
    'sector_links': VarInfo('sector_links', '(N, S, N, S)', 'Import linkages between countries and sectors', 'sector_links[importer_country, output_sector, exporter_country, input_sector]', 'sol'),
}

class ModelRegistry:
    """Registry for managing and documenting all model variables."""
    
    @staticmethod
    def get_variable_info(var_name: str) -> VarInfo:
        """Get information about a specific variable."""
        if var_name not in MODEL_VARIABLES:
            raise ValueError(f"Variable '{var_name}' not found in registry")
        return MODEL_VARIABLES[var_name]
    
    @staticmethod
    def list_variables(component: Optional[str] = None) -> List[VarInfo]:
        """List all variables, optionally filtered by component."""
        if component is None:
            return list(MODEL_VARIABLES.values())
        return [info for info in MODEL_VARIABLES.values() if info.component == component]
    
    @staticmethod
    def print_variable_summary(component: Optional[str] = None, show_indexing: bool = True):
        """Print a formatted summary of variables."""
        vars_to_show = ModelRegistry.list_variables(component)
        
        if component:
            print(f"\n=== {component.upper()} Variables ===")
        else:
            print("\n=== ALL MODEL Variables ===")
        
        print(f"{'Variable':<15} {'Shape':<12} {'Economic Meaning':<50} {'Indexing':<30}")
        print("-" * 107)
        
        for var in vars_to_show:
            indexing_str = var.indexing if show_indexing else ""
            print(f"{var.name:<15} {var.shape:<12} {var.meaning:<50} {indexing_str:<30}")
    
    @staticmethod
    def get_shape_info(var_name: str, N: int, S: int) -> Tuple[str, tuple]:
        """Get actual shape tuple for a variable given N and S."""
        info = ModelRegistry.get_variable_info(var_name)
        
        # Convert shape string to actual tuple
        shape_map = {
            'scalar': (),
            '(N,)': (N,),
            '(S,)': (S,),
            '(N, S)': (N, S),
            '(N, N, S)': (N, N, S),
            '(N, S, S)': (N, S, S),
            'List[str]': (N,) if 'country' in var_name else (S,)
        }
        
        actual_shape = shape_map.get(info.shape, info.shape)
        return info.shape, actual_shape
    
    @staticmethod
    def validate_variable_shape(var_name: str, array: np.ndarray, N: int, S: int) -> bool:
        """Validate that an array has the correct shape for a variable."""
        if var_name not in MODEL_VARIABLES:
            return True  # Skip validation for unknown variables
        
        _, expected_shape = ModelRegistry.get_shape_info(var_name, N, S)
        if isinstance(expected_shape, tuple) and hasattr(array, 'shape'):
            return array.shape == expected_shape
        return True
    
    @staticmethod
    def get_variables_by_dimension() -> Dict[str, List[str]]:
        """Group variables by their dimensionality."""
        dimension_groups = {
            'Scalars': [],
            'Country-level (N,)': [],
            'Sector-level (S,)': [],
            'Country-Sector (N,S)': [],
            'Trade flows (N,N,S)': [],
            'Input-Output (N,S,S)': [],
            'Other': []
        }
        
        for var_name, info in MODEL_VARIABLES.items():
            if info.shape == 'scalar':
                dimension_groups['Scalars'].append(var_name)
            elif info.shape == '(N,)':
                dimension_groups['Country-level (N,)'].append(var_name)
            elif info.shape == '(S,)':
                dimension_groups['Sector-level (S,)'].append(var_name)
            elif info.shape == '(N, S)':
                dimension_groups['Country-Sector (N,S)'].append(var_name)
            elif info.shape == '(N, N, S)':
                dimension_groups['Trade flows (N,N,S)'].append(var_name)
            elif info.shape == '(N, S, S)':
                dimension_groups['Input-Output (N,S,S)'].append(var_name)
            else:
                dimension_groups['Other'].append(var_name)
        
        return dimension_groups

# --- Mixin for .npz save/load for dataclasses ---
class NpzMixin:
    """Mixin to save/load all init dataclass fields to/from .npz."""
    def save_to_npz(self, filename: str) -> None:
        # Type ignore because self will be a dataclass when this mixin is used
        data = {f.name: getattr(self, f.name) for f in fields(self) if f.init}  # type: ignore
        np.savez(filename, **data)

    @classmethod
    def load_from_npz(cls, filename: str):
        """Load from .npz file and return instance of the concrete class."""
        data = np.load(filename, allow_pickle=True)
        kwargs = {}
        # Type ignore because cls will be a dataclass when this mixin is used
        for f in fields(cls):  # type: ignore
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
    pi:       np.ndarray  # shape (N, N, S)
    tilde_tau: np.ndarray  # shape (N, N, S)
    Xf:        np.ndarray  # shape (N, S)
    Xm:        np.ndarray  # shape (N, S)
    X:        np.ndarray  # shape (N, S)
    V:        np.ndarray  # shape (N,)
    D:         np.ndarray  # shape (N,)
    country_list:  list[str] = field(default_factory=list)
    sector_list:   list[str] = field(default_factory=list)

    def __post_init__(self):
        # Validate basic array structure
        self._validate_basic_structure()
        # Run full consistency check
        self.check_consistency(mute=CHECK_CONSISTENCY_MUTE)
    
    def _validate_basic_structure(self) -> None:
        """Validate basic array structure and dimensions."""
        if self.alpha.ndim != 2:
            raise ValueError("alpha must be 2D")
        if self.alpha.shape != (self.N, self.S):
            raise ValueError(f"alpha.shape {self.alpha.shape} does not match (N, S)=({self.N},{self.S})")
        
        # Validate non-negativity for key economic variables
        if np.any(self.V <= 0):
            raise ValueError("All V (value added) values must be positive")
        if np.any(self.Xf < 0) or np.any(self.Xm < 0):
            raise ValueError("All expenditure values (Xf, Xm) must be non-negative")

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
            assert self.pi.shape == (N, N, S)
            assert self.tilde_tau.shape == (N, N, S)
            assert self.Xf.shape == (N, S)
            assert self.Xm.shape == (N, S)
            assert self.X.shape == (N, S)
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
        gamma_beta_sum = np.sum(self.gamma, axis=1) + self.beta
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
        
        # Return the final consistency check result
        if not inconsistent:
            if not mute:
                logger.info("✅ All parameter consistency checks passed.")
            return True
        else:
            if not mute:
                logger.error("❌ Parameter consistency checks failed.")
            return False

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
    All shock values represent multiplicative changes from baseline (hat algebra).
    """
    lambda_hat:       np.ndarray  # shape (N, S) - Productivity shocks
    df_hat:           np.ndarray  # shape (N, N, S) - Final goods trade cost shocks
    dm_hat:           np.ndarray  # shape (N, N, S) - Intermediate goods trade cost shocks
    tilde_tau_hat:  np.ndarray  # shape (N, N, S) - Tariff shocks

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
    
    def is_baseline(self) -> bool:
        """Check if all shocks are at baseline (all ones)."""
        return (np.allclose(self.lambda_hat, 1.0) and 
                np.allclose(self.df_hat, 1.0) and
                np.allclose(self.dm_hat, 1.0) and
                np.allclose(self.tilde_tau_hat, 1.0))
    
    def reset_to_baseline(self) -> None:
        """Reset all shocks to baseline values (ones)."""
        N, S = self.N, self.S
        self.lambda_hat = np.ones((N, S))
        self.df_hat = np.ones((N, N, S))
        self.dm_hat = np.ones((N, N, S))
        self.tilde_tau_hat = np.ones((N, N, S))

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
            assert self.tilde_tau_hat.shape == (N, N, S), f"tilde_tau_hat shape {self.tilde_tau_hat.shape} != ({N}, {N}, {S})"
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

        # 5. Check consistency of tilde_tau_hat.
        # 5-1: Every value must be >= 1.
        if np.any(self.tilde_tau_hat < 1):
            invalid_indices = np.where(self.tilde_tau_hat < 1)
            logger.error("Check 5-1: ❌ Some tilde_tau_hat values are < 1 at positions %s. Values: %s", 
                         invalid_indices, self.tilde_tau_hat[invalid_indices])
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 5-1: ✅ All tilde_tau_hat values are >= 1.")
        # 5-2: Diagonal elements must be 1.
        diag_tau = self.tilde_tau_hat[np.arange(N), np.arange(N), :]
        if not np.allclose(diag_tau, 1.0, atol=tol):
            logger.error("Check 5-2: ❌ Diagonal elements of tilde_tau_hat are not equal to 1. They are: %s", diag_tau)
            inconsistent = True
        else:
            if not mute:
                logger.info("Check 5-2: ✅ Diagonal elements of tilde_tau_hat are 1.")

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
            (lambda_hat, df_hat, dm_hat, tilde_tau_hat) = shocks.unpack()
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
    pif_prime: np.ndarray  # shape (N, N, S)
    pim_prime: np.ndarray  # shape (N, N, S)
    Xf_prime:  np.ndarray  # shape (N, S)
    Xm_prime:  np.ndarray  # shape (N, S)
    X_prime:   np.ndarray  # shape (N, S)
    p_index:   np.ndarray  # shape (N,)
    real_w_hat:    np.ndarray  # shape (N,)
    D_prime:  np.ndarray  # shape (N,)
    Xf_prod_prime: np.ndarray  # shape (N, S)
    Xm_prod_prime: np.ndarray  # shape (N, S)
    X_prod_prime: np.ndarray  # shape (N, S)
    I_prime: np.ndarray  # shape (N,)
    output_prime: np.ndarray  # shape (N, S)
    real_I_prime: np.ndarray  # shape (N,)
    sector_links: np.ndarray  # shape (N, S, N, S)
    

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
    """
    Main model class that orchestrates parameters, shocks, and solutions.
    Represents the Caliendo & Parro (2015) trade model.
    """
    params:      ModelParams
    shocks:      ModelShocks = field(init=False)
    sol:         ModelSol    = field(init=False)
    is_optimized: bool       = field(init=False, default=False)

    def __post_init__(self):
        self.reset_shocks()
        self.reset_sols()
    
    @property
    def N(self) -> int:
        """Number of countries."""
        return self.params.N
    
    @property 
    def S(self) -> int:
        """Number of sectors."""
        return self.params.S
    
    def validate_compatibility(self) -> bool:
        """Check if shocks and params have compatible dimensions."""
        return (self.N == self.shocks.N and 
                self.S == self.shocks.S)

    def reset_shocks(self) -> None:
        N, S = self.N, self.S
        self.shocks = ModelShocks(
            lambda_hat      = np.ones((N,S)),
            df_hat          = np.ones((N,N,S)),
            dm_hat          = np.ones((N,N,S)),
            tilde_tau_hat = np.ones((N,N,S)),
        )
        self.is_optimized = False

    def reset_sols(self) -> None:
        N, S = self.N, self.S
        self.sol = ModelSol(
            w_hat    = np.ones(N),
            c_hat    = np.ones((N,S)),
            Pf_hat   = np.ones((N,S)),
            Pm_hat   = np.ones((N,S)),
            pif_hat  = np.ones((N,N,S)),
            pim_hat  = np.ones((N,N,S)),
            pif_prime = np.ones((N,N,S)),
            pim_prime = np.ones((N,N,S)),
            Xf_prime = np.ones((N,S)),
            Xm_prime = np.ones((N,S)),
            X_prime = np.ones((N,S)),
            p_index  = np.ones(N),
            real_w_hat   = np.ones(N),
            D_prime = np.ones(N),
            Xf_prod_prime = np.ones((N,S)),
            Xm_prod_prime = np.ones((N,S)),
            X_prod_prime = np.ones((N,S)),
            I_prime = np.ones(N),
            output_prime = np.ones((N,S)),
            real_I_prime = np.ones(N),
            sector_links = np.ones((N, S, N, S)),
        )
    
    def summary(self) -> str:
        """Return a summary of the model configuration."""
        return (f"Model({self.N} countries, {self.S} sectors, "
                f"{'solved' if self.is_optimized else 'unsolved'}, "
                f"{'baseline' if self.shocks.is_baseline() else 'shocked'})")
    
    def print_variables_info(self, component: Optional[str] = None):
        """Print information about model variables."""
        ModelRegistry.print_variable_summary(component)
    
    def validate_all_shapes(self) -> bool:
        """Validate that all arrays have correct shapes according to registry."""
        N, S = self.N, self.S
        all_valid = True
        
        # Check params
        for field_info in fields(self.params):
            if field_info.init and hasattr(self.params, field_info.name):
                var_value = getattr(self.params, field_info.name)
                if isinstance(var_value, np.ndarray):
                    if not ModelRegistry.validate_variable_shape(field_info.name, var_value, N, S):
                        logger.error(f"Shape validation failed for params.{field_info.name}")
                        all_valid = False
        
        # Check shocks 
        for field_info in fields(self.shocks):
            if field_info.init and hasattr(self.shocks, field_info.name):
                var_value = getattr(self.shocks, field_info.name)
                if isinstance(var_value, np.ndarray):
                    if not ModelRegistry.validate_variable_shape(field_info.name, var_value, N, S):
                        logger.error(f"Shape validation failed for shocks.{field_info.name}")
                        all_valid = False
        
        # Check solutions
        for field_info in fields(self.sol):
            if field_info.init and hasattr(self.sol, field_info.name):
                var_value = getattr(self.sol, field_info.name)
                if isinstance(var_value, np.ndarray):
                    if not ModelRegistry.validate_variable_shape(field_info.name, var_value, N, S):
                        logger.error(f"Shape validation failed for sol.{field_info.name}")
                        all_valid = False
        
        if all_valid:
            logger.info("✅ All variable shapes are valid")
        return all_valid
    
    def get_variable_info(self, var_name: str) -> VarInfo:
        """Get registry information for a specific variable."""
        return ModelRegistry.get_variable_info(var_name)
    
    def list_variables_by_dimension(self):
        """Print variables grouped by dimension."""
        groups = ModelRegistry.get_variables_by_dimension()
        
        print("\n=== Variables by Dimension ===")
        for group_name, var_names in groups.items():
            if var_names:  # Only show non-empty groups
                print(f"\n{group_name}:")
                for var_name in var_names:
                    info = MODEL_VARIABLES[var_name]
                    print(f"  {var_name:<15} - {info.meaning}")

    @classmethod
    def from_npz(cls, params_file, shocks_file, sols_file) -> "Model":
        from typing import cast
        p = cast(ModelParams, ModelParams.load_from_npz(params_file))
        m = cls(p)
        m.shocks = cast(ModelShocks, ModelShocks.load_from_npz(shocks_file))
        m.sol = cast(ModelSol, ModelSol.load_from_npz(sols_file))
        return m


# === Demo Function for ModelRegistry Usage ===
def demo_model_registry():
    """Demonstrate how to use the ModelRegistry system."""
    print("=== ModelRegistry Demo ===\n")
    
    # 1. Print all variables
    ModelRegistry.print_variable_summary()
    
    # 2. Print variables by component
    print("\n" + "="*50)
    ModelRegistry.print_variable_summary('params')
    
    print("\n" + "="*50)  
    ModelRegistry.print_variable_summary('shocks')
    
    print("\n" + "="*50)
    ModelRegistry.print_variable_summary('sol')
    
    # 3. Get info about specific variables
    print("\n" + "="*50)
    print("\n=== Specific Variable Info ===")
    for var in ['alpha', 'lambda_hat', 'w_hat']:
        info = ModelRegistry.get_variable_info(var)
        print(f"{var}: {info.meaning}")
        print(f"  Shape: {info.shape}, Indexing: {info.indexing}")
    
    # 4. Show variables by dimension
    print("\n" + "="*50)
    groups = ModelRegistry.get_variables_by_dimension()
    for group_name, var_names in groups.items():
        if var_names:
            print(f"\n{group_name}: {', '.join(var_names)}")
    
    # 5. Shape validation example
    print("\n" + "="*50)
    print("\n=== Shape Info Examples (N=3, S=2) ===")
    N, S = 3, 2
    for var in ['alpha', 'gamma', 'pif', 'V']:
        shape_str, actual_shape = ModelRegistry.get_shape_info(var, N, S)
        print(f"{var}: {shape_str} -> {actual_shape}")


if __name__ == "__main__":
    demo_model_registry()



