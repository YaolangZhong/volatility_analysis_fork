import numpy as np
from scipy.optimize import minimize, OptimizeResult
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional

from models    import Model, ModelSol, ModelShocks
from equations import generate_equilibrium

# --------------------------------------------------------------------
#   Solver configuration (no numeraire, no reduced system)
# --------------------------------------------------------------------
@dataclass
class SolverConfig:
    method        : str   = "L-BFGS-B"
    bound_eps     : float = 1e-6      # lower bound on each wage element
    maxiter       : int   = 10_000
    gtol          : float = 1e-8
    ftol          : float = 1e-8
    display       : bool  = False
    mute_callback : bool  = False


# --------------------------------------------------------------------
#                       Main optimisation class
# --------------------------------------------------------------------
class ModelSolver:
    def __init__(self, model: Model, config: SolverConfig = SolverConfig()):
        self.model   = model
        self.config  = config
        self.bounds  = [(self.config.bound_eps, None)] * model.params.N
        self.iter_ct = 0
        self.res     : Optional[OptimizeResult] = None

    # ------------------------------------------------ objective
    def objective(self, w_hat: np.ndarray) -> float:
        params   = self.model.params
        shocks = self.model.shocks

        (N, S, alpha, beta, gamma, theta, pi, tilde_tau, X, V, D, countries, sectors) = params.unpack()
        (lambda_hat, d_hat, tilde_tau_prime) = shocks.unpack()
        (_, _, _, _, D_prime, _, _) = generate_equilibrium(w_hat, beta, gamma, theta, 
            pi, lambda_hat, d_hat, tilde_tau_prime, alpha, V, D, X)

        return np.linalg.norm(D_prime - D)

    # ------------------------------------------------ solve
    def solve(self):
        current_val = [None]

        def fun(x):
            v = self.objective(x)
            current_val[0] = v
            return v

        def cb(xk, state=None):
            self.iter_ct += 1
            if not self.config.mute_callback:
                print(f"Iter {self.iter_ct}: loss = {current_val[0]:.3e}")

        x0 = self.model.sol.w_hat.copy()

        self.res = minimize(
            fun=fun,
            x0=x0,
            method=self.config.method,
            callback=cb,
            bounds=self.bounds,
            options=dict(
                maxiter=self.config.maxiter,
                disp=self.config.display,
                gtol=self.config.gtol,
                ftol=self.config.ftol,
            ),
        )
        print(f"SciPy status {self.res.status}: {self.res.message}")
        w_hat = self.res.x

        # ------------- recompute full equilibrium with optimal wages
        params   = self.model.params
        shocks = self.model.shocks

        (N, S, alpha, beta, gamma, theta, pi, tilde_tau, X, V, D, countries, sectors) = params.unpack()
        (lambda_hat, d_hat, tilde_tau_prime) = shocks.unpack()
        (c_hat, P_hat, pi_hat, X_prime, D_prime, p_index, real_w) = generate_equilibrium(
            w_hat, beta, gamma, theta, pi, lambda_hat, d_hat, tilde_tau_prime, alpha, V, D, X)

        # ------------- store in Model.sol
        self.model.sol = ModelSol(
            w_hat   = w_hat,
            c_hat   = c_hat,
            P_hat   = P_hat,
            pi_hat  = pi_hat,
            X_prime = X_prime,
            p_index = p_index,
            real_w  = real_w,
            D_prime = D_prime,
        )
        self.model.is_optimized = True