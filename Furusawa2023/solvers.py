import numpy as np
from scipy.optimize import minimize, OptimizeResult
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional

from models import *
from equations import *

@dataclass
class SolverConfig:
    """
    Solver configuration.

    Attributes
    ----------
    method : str
        Optimiser name passed to ``scipy.optimize.minimize`` (default ``"L-BFGS-B"``).
    eps : float
        Lower bound for each element of the reduced wage vector.
    maxiter : int
        Maximum iterations allowed.
    gtol : float
        Convergence tolerance on the infinity‑norm of the gradient
        (‖∇F‖∞ ≤ gtol terminates the solver).
    display : bool
        If ``True``, enable SciPy’s internal iteration display.
    numeraire_index : int
        Index of the wage held fixed at 1.0.
    """
    method: str = "L-BFGS-B"
    bound_eps: float = 1e-6
    maxiter: int = 10_000
    gtol: float = 1e-8
    ftol: float = 1e-8
    fd_eps: float = 1e-3
    display: bool = False
    numeraire_index: int = 0
    numeraire_value: float = 1.0
    mute_callback: bool = False
    reduced: bool = True  # optimise reduced (N−1) system if True, else full N


class ModelSolver:
    def __init__(
        self,
        model: Model,
        config: SolverConfig = SolverConfig(),
    ):
        self.model = model
        self.config = config

        # bounds array
        self.bounds = [(self.config.bound_eps, None)] * (model.params.N - 1)
        # state for early stopping
        self.iter_count = 0
        self.res: Optional[OptimizeResult] = None  # store scipy minimize output

    def objective(
        self, w_reduced, params, shocks, numeraire_index, numeraire_value
    ):
        """Objective function for optimization."""
        # ------------------------------------------------------------
        # Build the full wage vector.
        # If numeraire_index >= 0 we are in reduced‑vector mode and must
        # insert the numeraire; if it is -1 the vector is already full.
        # ------------------------------------------------------------
        if numeraire_index >= 0:
            w_hat = np.insert(w_reduced, numeraire_index, numeraire_value)
        else:
            w_hat = w_reduced  # full length already
        (N, S, alpha, beta, gamma, theta, pif, pim, tilde_tau,
                    Xf, Xm, V, D, countries, sectors) = params.unpack()
        (lambda_hat, df_hat, dm_hat, tilde_tau_prime) = shocks.unpack()
        (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, D_prime, p_index, real_w) = generate_equilibrium(
            w_hat, beta, gamma, theta, pif, pim, lambda_hat, df_hat, dm_hat, tilde_tau_prime, alpha, V)

        gap_D  = (D_prime - params.D)                         # (N,)
        gap_Xf = (Xf_prime - Xf).ravel()                   # (N·S,)
        gap_Xm = (Xm_prime - Xm).ravel()                   # (N·S,)

        #diff_all = np.concatenate([gap_D, gap_Xf, gap_Xm])
        diff_all = gap_D
        return np.linalg.norm(diff_all)

    def solve_reduced(self) -> np.ndarray:
        """
        Optimize the (N−1)-dimensional wage vector and store the SciPy
        OptimizeResult in `self.res`.
        """
        current_val = [None]  # mutable cell to share the latest loss

        def fun(x):
            v = self.objective(
                x, self.model.params, self.model.shocks, self.config.numeraire_index, self.config.numeraire_value
            )
            current_val[0] = v
            return v

        def cb(xk, state=None):
            self.iter_count += 1
            if self.config.mute_callback == False:
                print(f"Iter {self.iter_count}: loss = {current_val[0]:.3e}")

        x0 = np.delete(self.model.sol.w_hat, self.config.numeraire_index)

        res = minimize(
            fun=fun,
            x0=x0,
            jac=None,
            method=self.config.method,
            callback=cb,
            bounds=self.bounds,
            options={
                "maxiter": self.config.maxiter,
                "disp":    self.config.display,
                "gtol":    self.config.gtol,
                #"ftol":    self.config.ftol,
                #"eps":    self.config.fd_eps,
            },
        )
        self.res = res
        # show SciPy termination message for diagnostics
        print(f"SciPy status {res.status}: {res.message}")
        return res.x
    


    def solve(self):
        """
        Full pipeline:
         1. Optimize reduced vector or full vector
         2. Reconstruct full w_hat
         3. Generate equilibrium
         4. Pack into ModelSol
        """
        if self.config.reduced:
            w_reduced_opt = self.solve_reduced()
            w_hat_opt = np.insert(
                w_reduced_opt,
                self.config.numeraire_index,
                self.config.numeraire_value,
            )
        else:
            w_hat_opt = self.solve_full()
        (N, S, alpha, beta, gamma, theta, pif, pim, tilde_tau,
                    Xf, Xm, V, D, countries, sectors) = self.model.params.unpack()
        (lambda_hat, df_hat, dm_hat, tilde_tau_prime) = self.model.shocks.unpack()
        (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, D_prime, p_index, real_w) = generate_equilibrium(
            w_hat_opt, beta, gamma, theta, pif, pim, lambda_hat, df_hat, dm_hat, tilde_tau_prime, alpha, V)
        self.model.sol = ModelSol(
            w_hat=w_hat_opt,
            c_hat=c_hat,
            Pf_hat=Pf_hat,
            Pm_hat=Pm_hat,
            pif_hat=pif_hat,
            pim_hat=pim_hat,
            Xf_prime=Xf_prime,
            Xm_prime=Xm_prime,
            p_index=p_index, 
            real_w=real_w,
            D_prime=D_prime,
        )
        self.model.is_optimized = True
        return


    # ------------------------------------------------------------------
    # Full‑length optimisation (N variables, no numeraire reduction)
    # ------------------------------------------------------------------
    def solve_full(self) -> np.ndarray:
        """
        Optimise the *full* wage vector (length N).  Uses the same SciPy
        settings but skips the numeraire deletion.
        """
        current_val = [None]

        def fun(x):
            v = self.objective(
                x,                           # full wage vector
                self.model.params,
                self.model.shocks,
                numeraire_index=-1,          # <- not used inside objective
                numeraire_value=1.0,
            )
            current_val[0] = v
            return v

        def cb(xk, state=None):
            self.iter_count += 1
            if not self.config.mute_callback:
                print(f"[FULL] Iter {self.iter_count}: loss = {current_val[0]:.3e}")

        x0 = self.model.sol.w_hat.copy()

        # bounds for full vector
        full_bounds = [(self.config.bound_eps, None)] * x0.size

        res = minimize(
            fun=fun,
            x0=x0,
            jac=None,
            method=self.config.method,
            callback=cb,
            bounds=full_bounds,
            options={
                "maxiter": self.config.maxiter,
                "disp":    self.config.display,
                "gtol":    self.config.gtol,
                #"ftol":    self.config.ftol,
                #"eps":     self.config.fd_eps,
            },
        )
        self.res = res
        print(f"SciPy status {res.status}: {res.message}")
        return res.x
    


    # def objective(
    #     self, w_reduced, params, shocks, numeraire_index, numeraire_value
    # ):
    #     """Objective function for optimization."""
    #     w_hat = w_reduced
    #     (N, S, alpha, beta, gamma, theta, pif, pim, tilde_tau,
    #                 Xf, Xm, V, D, countries, sectors) = params.unpack()
    #     (lambda_hat, df_hat, dm_hat, tilde_tau_prime) = shocks.unpack()
    #     (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, D_prime, p_index, real_w) = generate_equilibrium(
    #         w_hat, beta, gamma, theta, pif, pim, lambda_hat, df_hat, dm_hat, tilde_tau_prime, alpha, V)
    #     a = D_prime / np.sum(V * w_hat)
    #     b = params.D / np.sum(V)
    #     diff = (a - b) / b
    #     return np.linalg.norm(diff)

    # def solve_reduced(self) -> np.ndarray:
    #     """
    #     Optimize the (N−1)-dimensional wage vector and store the SciPy
    #     OptimizeResult in `self.res`.
    #     """
    #     current_val = [None]  # mutable cell to share the latest loss

    #     def fun(x):
    #         v = self.objective(
    #             x, self.model.params, self.model.shocks, self.config.numeraire_index, self.config.numeraire_value
    #         )
    #         current_val[0] = v
    #         return v

    #     def cb(xk):
    #         self.iter_count += 1
    #         if self.config.mute_callback == False:
    #             print(f"Iter {self.iter_count}: loss = {current_val[0]:.3e}")

    #     x0 = self.model.sol.w_hat

    #     res = minimize(
    #         fun=fun,
    #         x0=x0,
    #         jac=None,
    #         method=self.config.method,
    #         callback=cb,
    #         bounds=self.bounds,
    #         options={
    #             "maxiter": self.config.maxiter,
    #             "disp":    self.config.display,
    #             "gtol":    self.config.gtol,
    #             "ftol":    self.config.ftol,
    #         },
    #     )
    #     self.res = res
    #     return res.x
    


    # def solve(self):
    #     """
    #     Full pipeline:
    #      1. Optimize reduced vector
    #      2. Reconstruct full w_hat
    #      3. Generate equilibrium
    #      4. Pack into ModelSol
    #     """
    #     w_reduced_opt = self.solve_reduced()
    #     w_hat_opt = w_reduced_opt
    #     (N, S, alpha, beta, gamma, theta, pif, pim, tilde_tau,
    #                 Xf, Xm, V, D, countries, sectors) = self.model.params.unpack()
    #     (lambda_hat, df_hat, dm_hat, tilde_tau_prime) = self.model.shocks.unpack()
    #     (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, D_prime, p_index, real_w) = generate_equilibrium(
    #         w_hat_opt, beta, gamma, theta, pif, pim, lambda_hat, df_hat, dm_hat, tilde_tau_prime, alpha, V)
    #     self.model.sol = ModelSol(
    #         w_hat=w_hat_opt,
    #         c_hat=c_hat,
    #         Pf_hat=Pf_hat,
    #         Pm_hat=Pm_hat,
    #         pif_hat=pif_hat,
    #         pim_hat=pim_hat,
    #         Xf_prime=Xf_prime,
    #         Xm_prime=Xm_prime,
    #         p_index=p_index, 
    #         real_w=real_w,
    #         D_prime=D_prime,
    #     )
    #     self.model.is_optimized = True
    #     return
    



def solve_with_shock(base_model: Model, shock: ModelShocks, config: SolverConfig) -> ModelSol:
    """
    Worker function: clone the base model, swap in `shock`, solve, return the new solution.
    """
    # 1) copy the model so we don't clobber the original
    m = deepcopy(base_model)

    # 2) inject the new shocks
    m.shocks = shock
    m.is_optimized = False   # reset flag

    # 3) run the solver
    solver = ModelSolver(m, config)
    sol = solver.solve()

    # 4) return the completed solution object
    return sol
