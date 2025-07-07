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
        (N, S, alpha, beta, gamma, theta, pif, pim, pi, tilde_tau, Xf, Xm, X, V, D, countries, sectors) = params.unpack()
        (lambda_hat, df_hat, dm_hat, tilde_tau_hat) = shocks.unpack()

        w_hat = self.model.sol.w_hat.copy()
        Pm_hat_old = np.ones((N, S))

        vfactor   = self.config.vfactor
        tol       = self.config.tol
        max_iter  = self.config.max_iter
        wfmax = 1.0
        iteration = 1
        while iteration <= max_iter and wfmax > tol:
            (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, pif_prime, pim_prime, Xf_prime, Xm_prime, D_prime, p_index, real_w_hat, X_prime, Xf_prod_prime, Xm_prod_prime, X_prod_prime, I_prime, output_prime, real_I_prime) = generate_equilibrium(
                w_hat, Pm_hat_old, alpha, beta, gamma, theta, pif, pim, tilde_tau, V, D, Xf, Xm, lambda_hat, df_hat, dm_hat, tilde_tau_hat)
            w_grad = (D - D_prime) / V * vfactor
            Xf = Xf_prime        # latest expenditure becomes next-round initial guess
            Xm = Xm_prime
            # update wages
            w_hat = w_hat - w_grad

            # convergence check
            wfmax = np.max(np.abs(w_grad))
            Pmmax = np.max(np.abs(Pm_hat - Pm_hat_old))

            if not self.config.mute:
                mnX, mxX = Xf_prime.min(), Xf_prime.max()
                mnW, mxW = w_hat.min(), w_hat.max()
                print(f"Iter {iteration}: w_min={mnW:.3e}, w_max={mxW:.3e}, "
                      f"X_min={mnX:.3e}, X_max={mxX:.3e}, "
                      f"Δw={wfmax:.3e}, ΔP={Pmmax:.3e}")

            Pm_hat_old = Pm_hat
            iteration += 1

        # store
        self.model.sol = ModelSol(
            w_hat   = w_hat,
            c_hat   = c_hat,
            Pf_hat   = Pf_hat,
            Pm_hat   = Pm_hat,
            pif_hat  = pif_hat,
            pim_hat  = pim_hat,
            pif_prime = pif_prime,
            pim_prime = pim_prime,
            Xf_prime = Xf_prime,
            Xm_prime = Xm_prime,
            p_index = p_index,
            real_w_hat  = real_w_hat,
            D_prime = D_prime,
            X_prime = X_prime,
            Xf_prod_prime = Xf_prod_prime,
            Xm_prod_prime = Xm_prod_prime,
            X_prod_prime = X_prod_prime,
            I_prime = I_prime,
            output_prime = output_prime,
            real_I_prime = real_I_prime
        )
        self.model.is_optimized = True