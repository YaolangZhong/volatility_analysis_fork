"""
equations.py ― JAX version
--------------------------
All numerical routines now use `jax.numpy` (aliased as `jnp`) so they can
participate in JAX’s automatic differentiation and JIT compilation.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import jit

from models import ModelParams, ModelShocks

# ------------------------------------------------------------------
# (7) Unit-cost index
# ------------------------------------------------------------------
@jit
def calc_c_hat_jax(w_hat: jnp.ndarray,
                   Pm_hat: jnp.ndarray,
                   params: ModelParams) -> jnp.ndarray:
    log_w  = jnp.log(w_hat)[:, None]
    log_Pm = jnp.log(Pm_hat)
    wage   = params.beta * log_w
    inputs = jnp.einsum("nkj,nk->nj", params.gamma, log_Pm)
    return jnp.exp(wage + inputs)

# ------------------------------------------------------------------
# (8) Price-index changes
# ------------------------------------------------------------------
@jit
def calc_price_index_jax(theta: jnp.ndarray,
                         pi: jnp.ndarray,
                         lambda_hat: jnp.ndarray,
                         c_hat: jnp.ndarray,
                         d_hat: jnp.ndarray) -> jnp.ndarray:
    lam_ex = lambda_hat[jnp.newaxis, :, :]
    cost   = (c_hat[:, jnp.newaxis, :] * d_hat) ** (-theta[jnp.newaxis, None, :])
    term   = pi * lam_ex * cost
    return jnp.sum(term, axis=1) ** (-1.0 / theta)

def calc_Pf_hat_jax(c_hat, params, shocks):
    return calc_price_index_jax(params.theta, params.pif,
                                shocks.lambda_hat, c_hat, shocks.df_hat)

def calc_Pm_hat_jax(c_hat, params, shocks):
    return calc_price_index_jax(params.theta, params.pim,
                                shocks.lambda_hat, c_hat, shocks.dm_hat)

# ------------------------------------------------------------------
# (9) Expenditure-share changes
# ------------------------------------------------------------------
@jit
def calc_expenditure_share_jax(theta, lambda_hat, c_hat, P_hat, d_hat):
    lam_ex = lambda_hat[:, jnp.newaxis, :]
    cost   = (c_hat[:, jnp.newaxis, :] * d_hat) ** (-theta[jnp.newaxis, None, :])
    num    = lam_ex * cost
    den    = P_hat[:, jnp.newaxis, :] ** (-theta[jnp.newaxis, None, :])
    return num / den

def calc_pif_hat_jax(c_hat, Pf_hat, params, shocks):
    return calc_expenditure_share_jax(params.theta, shocks.lambda_hat,
                                      c_hat, Pf_hat, shocks.df_hat)

def calc_pim_hat_jax(c_hat, Pm_hat, params, shocks):
    return calc_expenditure_share_jax(params.theta, shocks.lambda_hat,
                                      c_hat, Pm_hat, shocks.dm_hat)

# ------------------------------------------------------------------
# (10-11) Linear system for X′
# ------------------------------------------------------------------
def _build_A_vec_jax(w_hat, td_prime, alpha, VA):
    Af = alpha * (w_hat * VA + td_prime)[:, None]
    Am = jnp.zeros_like(Af)
    return jnp.concatenate([Af.ravel(), Am.ravel()])

def _build_B_matrix_jax(N, J, alpha, gamma,
                        pif, pim,
                        tilde_tau_prime, pif_hat, pim_hat):
    factor = (tilde_tau_prime - 1.) / tilde_tau_prime
    Uff = jnp.sum(factor * pif_hat * pif, axis=1)
    Ufm = jnp.sum(factor * pim_hat * pim, axis=1)

    Du_ff = jnp.diag(Uff.ravel())
    Du_fm = jnp.diag(Ufm.ravel())
    Dv    = jnp.diag(alpha.ravel())

    R = jnp.kron(jnp.eye(N), jnp.ones((1, J)))
    P = jnp.kron(jnp.eye(N), jnp.ones((J, 1)))

    Bff = Dv @ P @ R @ Du_ff
    Bfm = Dv @ P @ R @ Du_fm

    pif_prime = pif_hat * pif / tilde_tau_prime
    pim_prime = pim_hat * pim / tilde_tau_prime

    Bmf = jnp.einsum("nks,ink->nsik", gamma, pif_prime).reshape(N*J, N*J)
    Bmm = jnp.einsum("nks,ink->nsik", gamma, pim_prime).reshape(N*J, N*J)

    return jnp.block([[Bff, Bfm],
                      [Bmf, Bmm]])

def calc_X_jax(w_hat, pif_hat, pim_hat, td_prime,
               params: ModelParams, shocks: ModelShocks):
    N, J = params.N, params.J
    A = _build_A_vec_jax(w_hat, td_prime, params.alpha, params.VA)
    B = _build_B_matrix_jax(N, J, params.alpha, params.gamma,
                            params.pif, params.pim,
                            shocks.tilde_tau_prime, pif_hat, pim_hat)
    X_vec = jnp.linalg.solve(jnp.eye(2*N*J) - B, A)
    Xf = X_vec[:N*J].reshape(N, J)
    Xm = X_vec[N*J:].reshape(N, J)
    return Xf, Xm

# ------------------------------------------------------------------
# (12) Trade-deficit update
# ------------------------------------------------------------------
@jit
def calc_td_prime_jax(pif_hat, pim_hat, Xf_prime, Xm_prime,
                      params: ModelParams, shocks: ModelShocks):
    imp = pif_hat*Xf_prime[:, None, :] + pim_hat*Xm_prime[:, None, :]
    imp_net = imp / shocks.tilde_tau_prime
    exp_net = jnp.swapaxes(imp_net, 0, 1)
    return jnp.sum(imp_net - exp_net, axis=(1, 2))

# ------------------------------------------------------------------
# Fixed-point iteration for P̂ᵐ
# ------------------------------------------------------------------
def solve_price_and_cost_jax(w_hat, params, shocks,
                             max_iter=1000, tol=1e-6, mute=True):
    N, J = params.N, params.J
    Pm = jnp.ones((N, J))
    for it in range(max_iter):
        c_hat = calc_c_hat_jax(w_hat, Pm, params)
        Pm_new = calc_Pm_hat_jax(c_hat, params, shocks)
        if jnp.max(jnp.abs(Pm_new-Pm)) < tol:
            if not mute:
                print(f"Pm̂ converged in {it+1} iterations")
            return c_hat, Pm_new
        Pm = Pm_new
    if not mute:
        print("Pm̂ did not converge")
    return calc_c_hat_jax(w_hat, Pm, params), Pm

# ------------------------------------------------------------------
# End-to-end equilibrium generator
# ------------------------------------------------------------------
def generate_equilibrium_jax(w_hat, params, shocks):
    c_hat, Pm_hat = solve_price_and_cost_jax(w_hat, params, shocks)
    Pf_hat  = calc_Pf_hat_jax(c_hat, params, shocks)
    pif_hat = calc_pif_hat_jax(c_hat, Pf_hat, params, shocks)
    pim_hat = calc_pim_hat_jax(c_hat, Pm_hat, params, shocks)
    Xf_prime, Xm_prime = calc_X_jax(w_hat, pif_hat, pim_hat,
                                    params.D, params, shocks)
    p_index = jnp.exp((params.alpha * jnp.log(Pf_hat)).sum(axis=1))
    real_w  = w_hat / p_index
    return (c_hat, Pf_hat, Pm_hat,
            pif_hat, pim_hat,
            Xf_prime, Xm_prime,
            p_index, real_w)