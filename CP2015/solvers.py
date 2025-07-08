import numpy as np
from dataclasses import dataclass
from equations import *
from models import Model, ModelSol, ModelShocks

@dataclass
class SolverConfig:
    bound_eps : float = 1e-6
    max_iter  : int   = 1_000_000
    tol       : float = 1e-6
    vfactor   : float = -0.2
    mute      : bool  = False


class ModelSolver:
    def __init__(self, model: Model, config: SolverConfig = SolverConfig()):
        self.model   = model
        self.config  = config
        self.bounds  = [(self.config.bound_eps, None)] * model.params.N
        self.iter_ct = 0


    def solve(self):
        params  = self.model.params
        shocks  = self.model.shocks
        (N, S, alpha, beta, gamma, theta, pi, tilde_tau, X, V, D, countries, sectors) = params.unpack()
        (lambda_hat, d_hat, tilde_tau_hat) = shocks.unpack()

        w_hat = self.model.sol.w_hat.copy()
        P_hat_old = np.ones((N, S))

        vfactor   = self.config.vfactor
        tol       = self.config.tol
        max_iter  = self.config.max_iter
        wfmax = 1.0
        iteration = 1
        while iteration <= max_iter and wfmax > tol:
            (c_hat, P_hat, pi_hat, X_prime, D_prime, p_index, real_w, sector_links) = generate_equilibrium(
                w_hat, P_hat_old, alpha, beta, gamma, theta, pi, tilde_tau, V, D, X, lambda_hat, d_hat, tilde_tau_hat)
            w_grad = (D - D_prime) / V * vfactor
            X = X_prime        # latest expenditure becomes next-round initial guess
            # update wages
            w_hat = w_hat - w_grad

            # convergence check
            wfmax = np.max(np.abs(w_grad))
            Pfmax = np.max(np.abs(P_hat - P_hat_old))

            if not self.config.mute:
                mnX, mxX = X_prime.min(), X_prime.max()
                mnW, mxW = w_hat.min(), w_hat.max()
                print(f"Iter {iteration}: w_min={mnW:.3e}, w_max={mxW:.3e}, "
                      f"X_min={mnX:.3e}, X_max={mxX:.3e}, "
                      f"Δw={wfmax:.3e}, ΔP={Pfmax:.3e}")

            P_hat_old = P_hat
            iteration += 1

        # store
        self.model.sol = ModelSol(
            w_hat   = w_hat,
            c_hat   = c_hat,
            P_hat   = P_hat,
            pi_hat  = pi_hat,
            X_prime = X_prime,
            p_index = p_index,
            real_w  = real_w,
            D_prime = D_prime,
            sector_links = sector_links
        )
        self.model.is_optimized = True