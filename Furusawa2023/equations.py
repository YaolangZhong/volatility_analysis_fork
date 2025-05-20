import numpy as np
from scipy.linalg import block_diag
from typing import Tuple
from numba import njit
from models import ModelParams, ModelShocks

@njit
def calc_c_hat(
    w_hat:  np.ndarray,   # (N,)   wage changes  ẇ_i
    Pm_hat: np.ndarray,   # (N,S)  input-price changes P̂ᵐ_{ik}
    beta:   np.ndarray,   # (N,S)  labour shares β_{is}
    gamma:  np.ndarray    # (N,S,S) input shares γ_{isk}
) -> np.ndarray:
    """
    Equation (E1):  ĉ_{is} = exp[ β_{is} ln ŵ_i  +  Σ_k γ_{isk} ln P̂ᵐ_{ik} ].
    """
    N, S = beta.shape
    log_w  = np.log(w_hat)      # (N,)
    log_Pm = np.log(Pm_hat)     # (N,S)

    out = np.empty((N, S))
    for i in range(N):
        for s in range(S):
            acc = 0.0
            for k in range(S):
                acc += gamma[i, s, k] * log_Pm[i, k]
            out[i, s] = np.exp(beta[i, s] * log_w[i] + acc)
    return out

@njit
def calc_price_index(
    theta: np.ndarray,         # (S,)      trade elasticities θ_s
    pi: np.ndarray,            # (N,N,S)   bilateral shares π_{ins}^{·}
    lambda_hat: np.ndarray,    # (N,S)     productivity shocks Λ̂_{ns}
    c_hat: np.ndarray,         # (N,S)     cost changes ĉ_{ns}
    d_hat: np.ndarray          # (N,N,S)   wedge changes d̂_{ins}^{·}
) -> np.ndarray:
    """
    Equation (E2):
        P̂_{is}^{−θ_s} = Σ_n  π^{0}_{ins} · Λ̂_{ns} · (ĉ_{ns} d̂_{ins})^{−θ_s}.
    Works for both final and intermediate price indexes, depending on π and d.
    """
    I, N, S = pi.shape
    P_inv_pow = np.empty((I, S))   # store P̂^{−θ}

    for i in range(I):
        for s in range(S):
            acc = 0.0
            t  = theta[s]
            for n in range(N):
                acc += (
                    pi[i, n, s]
                    * lambda_hat[n, s]
                    * (c_hat[n, s] * d_hat[i, n, s]) ** (-t)
                )
            P_inv_pow[i, s] = acc

    # Convert P̂^{−θ} to P̂  :  P̂ = (P̂^{−θ})^{−1/θ}
    P_hat = np.empty_like(P_inv_pow)
    for i in range(I):
        for s in range(S):
            P_hat[i, s] = P_inv_pow[i, s] ** (-1.0 / theta[s])

    return P_hat

@njit
def calc_Pf_hat(
    c_hat: np.ndarray,
    theta: np.ndarray,
    pif: np.ndarray,
    lambda_hat: np.ndarray,
    df_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (8): Final-goods price index changes P̂ᶠ.
    """
    return calc_price_index(
        theta,
        pif,
        lambda_hat,
        c_hat,
        df_hat
    )

@njit
def calc_Pm_hat(
    c_hat: np.ndarray,
    theta: np.ndarray,
    pim: np.ndarray,
    lambda_hat: np.ndarray,
    dm_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (8): Intermediate-goods price index changes P̂ᵐ.
    """
    return calc_price_index(
        theta,
        pim,
        lambda_hat,
        c_hat,
        dm_hat
    )

@njit
def calc_expenditure_share(
    theta: np.ndarray,
    lambda_hat: np.ndarray,
    c_hat: np.ndarray,
    P_hat: np.ndarray,
    d_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (9): Expenditure‐share changes π̂ given cost and price index changes.

      π̂[i,n,j] = λ[n,j] · (ĉ[n,j]·d̂[i,n,j])^(−θ_j) / (P̂[i,j]^(−θ_j))
    Returns
    -------
    pi_hat : array (importer i, exporter n, sector j)
    """
    # λ[n,j]  → expand across importer dimension  → shape (1, N, S)
    lam_ex = lambda_hat[np.newaxis, :, :]
    # (ĉ[n,j] · d̂[i,n,j])^(−θ_j)  → shape (I, N, S)
    cost_factor = (c_hat[np.newaxis, :, :] * d_hat) ** (-theta[np.newaxis, np.newaxis, :])
    # numerator: λ · cost_factor → shape (I, N, S)
    num = lam_ex * cost_factor
    # denominator: P̂[i,j]^(−θ_j)  → shape (I, 1, S), broadcasts over exporter axis
    den = P_hat[:, np.newaxis, :] ** (-theta[np.newaxis, np.newaxis, :])
    # final (I, N, S) array – already ordered as (importer, exporter, sector)
    return num / den

@njit
def calc_pif_hat(
    c_hat: np.ndarray,
    Pf_hat: np.ndarray,
    theta: np.ndarray,
    lambda_hat: np.ndarray,
    df_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (9): Final-goods expenditure-share changes π̂ᶠ.
    """
    return calc_expenditure_share(
        theta,
        lambda_hat,
        c_hat,
        Pf_hat,
        df_hat
    )

@njit
def calc_pim_hat(
    c_hat: np.ndarray,
    Pm_hat: np.ndarray,
    theta: np.ndarray,
    lambda_hat: np.ndarray,
    dm_hat: np.ndarray
) -> np.ndarray:
    """
    Equation (9): Intermediate-goods expenditure-share changes π̂ᵐ.
    """
    return calc_expenditure_share(
        theta,
        lambda_hat,
        c_hat,
        Pm_hat,
        dm_hat
    )



from numba import njit

@njit
def solve_price_and_cost(
    w_hat:      np.ndarray,  # (N,)   wage changes
    beta:       np.ndarray,  # (N,S)  labour shares β_{is}
    gamma:      np.ndarray,  # (N,S,S) IO coefficients γ_{isk}
    theta:      np.ndarray,  # (S,)   trade elasticities θ_s
    pim:        np.ndarray,  # (N,N,S) baseline π^{0m}_{ins}
    lambda_hat: np.ndarray,  # (N,S)   productivity shocks Λ̂_{ns}
    dm_hat:     np.ndarray   # (N,N,S) trade‑cost shocks d̂^{m}_{ins}
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixed‑point solver for unit‑cost changes (c_hat) and intermediate
    price‑index changes (Pm_hat).  Runs fully in Numba nopython mode.
    """
    MAX_ITER = 1000
    TOL      = 1e-6
    N = w_hat.shape[0]
    S = theta.shape[0]
    # --- initial guess ---------------------------------------------------
    Pm_hat = np.ones((N, S))
    for _ in range(MAX_ITER):
        # 1. unit costs given current Pm_hat
        c_hat = calc_c_hat(w_hat, Pm_hat, beta, gamma)
        # 2. update intermediate price index
        Pm_new = calc_Pm_hat(c_hat, theta, pim, lambda_hat, dm_hat)
        # 3. convergence
        if np.abs(Pm_new - Pm_hat).max() < TOL:
            Pm_hat = Pm_new
            break
        Pm_hat = Pm_new
    # final c_hat consistent with the converged Pm_hat
    c_hat = calc_c_hat(w_hat, Pm_hat, beta, gamma)
    return c_hat, Pm_hat


def build_linear_system(
    alpha:                  np.ndarray,   # (N,S)
    gamma:                  np.ndarray,   # (N,S,S)
    pif_prime:              np.ndarray,   # (N,N,S)
    pim_prime:              np.ndarray,   # (N,N,S)
    tilde_tau_prime:        np.ndarray,   # (N,N,S)
    w_hat:                  np.ndarray,   # (N,)
    V:                      np.ndarray    # (N,)
):
    """
    Construct A and B so that  (I - B) X = A  reproduces E4–E6 with
    cross-sector links (outer index k).

        X = [ vec(Xf) ; vec(Xm) ; D ]      length L = 2·N·S + N
    """
    N, S = alpha.shape
    L    = 2 * N * S + N

    A = np.zeros(L)
    B = np.zeros((L, L))

    # ------------------------------------------------------------------
    # 1.  λ-coefficients that depend on (i,k) only
    # ------------------------------------------------------------------
    tau      = tilde_tau_prime - 1.0            # τ′
    lam_ff_in = np.sum((tau / tilde_tau_prime) * pif_prime, axis=1)   # (N,S)
    lam_fm_in = np.sum((tau / tilde_tau_prime) * pim_prime, axis=1)   # (N,S)

    # ------------------------------------------------------------------
    # 2.  B_ff and B_fm  (block-diag by country, full within country)
    # ------------------------------------------------------------------
    ff_blocks, fm_blocks = [], []
    for i in range(N):
        # Λ^{ff}_i  and Λ^{fm}_i   (S × S)
        Λ_ff = alpha[i, :, None] * lam_ff_in[i][None, :]   # α_{is}·λ^{ff}_{ik}
        Λ_fm = alpha[i, :, None] * lam_fm_in[i][None, :]
        ff_blocks.append(Λ_ff)
        fm_blocks.append(Λ_fm)

    B_ff = block_diag(*ff_blocks)           # shape (N·S , N·S)
    B_fm = block_diag(*fm_blocks)

    # ------------------------------------------------------------------
    # 3.  B_fD   (each Xf-row gets +α_{is}·D_i)
    # ------------------------------------------------------------------
    J = N * S
    B_fD = np.zeros((J, N))
    for i in range(N):
        for s in range(S):
            B_fD[i * S + s, i] = alpha[i, s]

    # write the three sub-blocks into B
    B[:J, :J]           = B_ff
    B[:J, J:2 * J]      = B_fm
    B[:J, 2 * J:]       = B_fD

    # ------------------------------------------------------------------
    # 4.  B_mf and B_mm   (cross-country via γ_{isk})
    # ------------------------------------------------------------------
    def idx_f(i, s): return i * S + s
    def idx_m(i, s): return J + i * S + s
    def idx_D(i):    return 2 * J + i

    for i in range(N):
        for s in range(S):
            for k in range(S):
                for n in range(N):
                    w_ff = gamma[i, s, k] * pif_prime[i, n, k] / tilde_tau_prime[i, n, k]
                    w_fm = gamma[i, s, k] * pim_prime[i, n, k] / tilde_tau_prime[i, n, k]
                    B[idx_m(i, s), idx_f(n, k)] += w_ff
                    B[idx_m(i, s), idx_m(n, k)] += w_fm

    # ------------------------------------------------------------------
    # 5.  B_Df and B_Dm
    # ------------------------------------------------------------------
    for i in range(N):
        for s in range(S):
            for n in range(N):
                imp_f =  pif_prime[i, n, s] / tilde_tau_prime[i, n, s]
                imp_m =  pim_prime[i, n, s] / tilde_tau_prime[i, n, s]
                exp_f = -pif_prime[n, i, s] / tilde_tau_prime[n, i, s]
                exp_m = -pim_prime[n, i, s] / tilde_tau_prime[n, i, s]

                B[idx_D(i), idx_f(i, s)] += imp_f
                B[idx_D(i), idx_m(i, s)] += imp_m
                B[idx_D(i), idx_f(n, s)] += exp_f
                B[idx_D(i), idx_m(n, s)] += exp_m

    # ------------------------------------------------------------------
    # 6.  constant vector A   (only Xf rows)
    # ------------------------------------------------------------------
    for i in range(N):
        for s in range(S):
            A[idx_f(i, s)] = alpha[i, s] * w_hat[i] * V[i]

    return A, B


# ---------------------------------------------------------------------
# 2. Wrapper that returns (Xf, Xm, D)
# ---------------------------------------------------------------------
def calc_Xf_Xm_D(
    N: int,
    S: int,
    alpha: np.ndarray,
    gamma: np.ndarray,
    pif_prime: np.ndarray,
    pim_prime: np.ndarray,
    tilde_tau_prime:  np.ndarray,
    w_hat: np.ndarray,
    V: np.ndarray
):
    """
    Solve for (Xf, Xm, D) given all inputs.

    Returns
    -------
    Xf : ndarray (N, S)
    Xm : ndarray (N, S)
    D  : ndarray (N,)

    Parameters
    ----------
    tilde_tau_prime : array (N, N, S)
        = 1 + τ′, post-shock wedges
    """
    A, B = build_linear_system(alpha, gamma, pif_prime, pim_prime, tilde_tau_prime, w_hat, V)
    X = np.linalg.solve(np.eye(len(A)) - B, A)

    Xf = X[:N*S].reshape(N, S)
    Xm = X[N*S:2*N*S].reshape(N, S)
    D  = X[2*N*S:]

    return Xf, Xm, D


@njit
def calc_Xf_prime(
    alpha: np.ndarray,            # (N, S)
    pif_prime: np.ndarray,        # (i, n, k)
    pim_prime: np.ndarray,        # (i, n, k)
    tilde_tau_prime: np.ndarray,  # (i, n, k) = 1 + τ′
    w_hat: np.ndarray,            # (N,)
    V: np.ndarray,                # (N,)
    Xf_prev: np.ndarray,          # (N, S)  from previous iterate
    Xm_prev: np.ndarray,          # (N, S)  from previous iterate
    D_prev:  np.ndarray           # (N,)    from previous iterate
) -> np.ndarray:
    """
    Equation (E4): update final‑goods expenditure X̂ᶠ_{is}.
    """
    # τ′/(1+τ′)  — safe because tilde_tau_prime > 1 so denominator ≠ 0
    tau_ratio = (tilde_tau_prime - 1.0) / tilde_tau_prime  # (i,n,k)

    # λ^{ff}_{ik} and λ^{fm}_{ik}
    lam_ff = np.sum(tau_ratio * pif_prime, axis=1)         # (i,k)
    lam_fm = np.sum(tau_ratio * pim_prime, axis=1)         # (i,k)

    # inner income component: Σ_k [ λ^{ff}_{ik} Xf_prev + λ^{fm}_{ik} Xm_prev ]
    inner = np.sum(lam_ff * Xf_prev + lam_fm * Xm_prev, axis=1)  # (i,)

    # disposable income: wage + tariff rebate + trade balance
    B_i = w_hat * V + inner + D_prev      # (i,)

    # allocate α_{is} fraction to good s
    return alpha * B_i[:, None]           # (i,s)


@njit
def calc_Xm_prime(
    gamma: np.ndarray,            # (N, S, S)   γ_{i k s}
    pif_prime: np.ndarray,        # (i, n, k)   π̂ᶠ
    pim_prime: np.ndarray,        # (i, n, k)   π̂ᵐ
    tilde_tau_prime: np.ndarray,  # (i, n, k)   1 + τ′
    Xf_prime: np.ndarray,         # (N, S)      X̂ᶠ_{n k}
    Xm_prev: np.ndarray           # (N, S)      X̂ᵐ_{n k} from previous iter
) -> np.ndarray:
    """
    Consistent with
        X^{m′}_{is} = Σ_k γ_{iks} ·
                      Σ_n [ π̂^{f}_{nik} X̂^{f}_{nk} + π̂^{m}_{nik} X̂^{m}_{nk} ]
                            / (1+τ′_{nik})

    All tensors are stored in importer-first order (i, n, k).
    """
    # ------------------------------------------------------------------
    # 1. delivered value of each input type k that reaches importer i
    # ------------------------------------------------------------------
    delivered = (
        pif_prime * Xf_prime[np.newaxis, :, :] +
        pim_prime * Xm_prev[np.newaxis, :, :]
    ) / tilde_tau_prime
    # Σ_n  →  T_{ik}
    T = delivered.sum(axis=1)  # (i, k)
    # ------------------------------------------------------------------
    # 2. materials bill of sector‑s firms in importer i
    #    X^{m′}_{is} = Σ_k γ_{iks} T_{ik}
    # ------------------------------------------------------------------
    Xm_prime_new = (gamma * T[:, :, None]).sum(axis=1)   # (i, s)
    return Xm_prime_new


@njit
def calc_Xm_prime2(
    gamma: np.ndarray,            # (N, S, S)   γ_{i k s}
    pif_prime: np.ndarray,        # (i, n, k)   π̂ᶠ
    pim_prime: np.ndarray,        # (i, n, k)   π̂ᵐ
    tilde_tau_prime: np.ndarray,  # (i, n, k)   1 + τ′
    Xf_prime: np.ndarray,         # (N, S)      X̂ᶠ_{n k}
    Xm_prev:  np.ndarray          # (N, S)      X̂ᵐ_{n k}
) -> np.ndarray:
    """
    Implements
        X^{m′}_{is} = Σ_k γ_{iks} ·
                      Σ_n [ π̂^{f}_{nik} X̂^{f}_{nk} + π̂^{m}_{nik} X̂^{m}_{nk} ]
                            / (1+τ′_{nik})
    with importer-first tensors (i, n, k) and no np.einsum.
    """
    # ------------------------------------------------------------
    # 1. delivered value of each input good k that reaches importer i
    #    delivered[i,n,k] = (π̂^{f}_{ink} X̂ᶠ_{nk} + π̂ᵐ_{ink} X̂ᵐ_{nk}) / (1+τ′_{ink})
    # ------------------------------------------------------------
    delivered = (
        pif_prime * Xf_prime[np.newaxis, :, :] +  # broadcast over importer i
        pim_prime * Xm_prev[np.newaxis, :, :]
    ) / tilde_tau_prime                           # (i, n, k)

    # Σ_n  →  T_{ik}
    T = delivered.sum(axis=1)                     # (i, k)

    # ------------------------------------------------------------
    # 2. materials bill of sector-s firms in importer i
    #    X^{m′}_{is} = Σ_k γ_{iks} · T_{ik}
    # ------------------------------------------------------------
    Xm_prime_new = (gamma * T[:, :, None]).sum(axis=1)   # (i, s)
    return Xm_prime_new

@njit
def calc_D_prime(
    pif_prime: np.ndarray,        # (i, n, s)
    pim_prime: np.ndarray,        # (i, n, s)
    tilde_tau_prime: np.ndarray,  # (i, n, s)  = 1 + τ′
    Xf_prime: np.ndarray,         # (N, S)
    Xm_prime: np.ndarray          # (N, S)
) -> np.ndarray:
    """
    D′_i = Σ_{n,s} [imports_{ins} − exports_{nis}]
    with tensors stored (importer i, exporter n, sector s).

    * uses only Numba-supported ops (no tuple-axis reduction, no einsum)
    """
    # ------------ imports_{i n s}
    import_vol = (
        pif_prime * Xf_prime[:, np.newaxis, :] +
        pim_prime * Xm_prime[:, np.newaxis, :]
    ) / tilde_tau_prime                     # shape (i, n, s)

    # ------------ exports_{n i s}  (swap importer/exporter axes)
    export_vol = import_vol.transpose(1, 0, 2)   # (n, i, s)

    # ------------ trade balance tensor
    trade_bal = import_vol - export_vol          # (i, n, s)

    # Σ_s  then  Σ_n   (two single-axis sums → Numba friendly)
    D_prime = trade_bal.sum(axis=2).sum(axis=1)  # (i,)

    return D_prime


def calc_Xf_Xm_D_iterative_python(
    N: int,
    S: int,
    alpha: np.ndarray,            # (N, S)
    gamma: np.ndarray,            # (N, S, S)
    pif_prime: np.ndarray,        # (i, n, k)
    pim_prime: np.ndarray,        # (i, n, k)
    tilde_tau_prime: np.ndarray,  # (i, n, k) = 1 + τ′
    w_hat: np.ndarray,            # (N,)
    V: np.ndarray,                # (N,)
    *,
    Xf0: np.ndarray | None = None,
    Xm0: np.ndarray | None = None,
    D0:  np.ndarray | None = None,
    max_iter: int = 10_000,
    tol: float = 1e-3,
    damping: float = 0.3,
):
    """
    Fixed-point solver that matches equations (E4)–(E6).
    The heavy kernels it calls are JIT-compiled, but *this* wrapper runs
    in normal Python so you can pass optional initial guesses.
    """
    # ---------- initial guesses ----------------------------------------
    if Xf0 is None:
        Xf = alpha * (w_hat * V)[:, None]          # positive baseline
    else:
        Xf = Xf0.copy()

    if Xm0 is None:
        Xm = np.zeros_like(Xf)
    else:
        Xm = Xm0.copy()

    if D0 is None:
        D = np.zeros(N)
    else:
        D = D0.copy()

    eps = 1e-12  # avoid division by zero in relative error

    for _ in range(max_iter):
        Xf_prev, Xm_prev, D_prev = Xf.copy(), Xm.copy(), D.copy()

        # E-4  (final-goods expenditure)
        Xf_new = calc_Xf_prime(
            alpha, pif_prime, pim_prime, tilde_tau_prime,
            w_hat, V, Xf_prev, Xm_prev, D_prev
        )

        # E-5  (intermediate-goods expenditure)
        Xm_new = calc_Xm_prime(
            gamma, pif_prime, pim_prime, tilde_tau_prime,
            Xf_new, Xm_prev
        )

        # E-6  (trade balance)
        D_new = calc_D_prime(
            pif_prime, pim_prime, tilde_tau_prime,
            Xf_new, Xm_new
        )

        # Damping step
        Xf = damping * Xf_new + (1.0 - damping) * Xf_prev
        Xm = damping * Xm_new + (1.0 - damping) * Xm_prev
        D  = damping * D_new  + (1.0 - damping) * D_prev

        # Relative convergence check
        rel_Xf = np.max(np.abs(Xf - Xf_prev) / (np.abs(Xf_prev) + eps))
        rel_Xm = np.max(np.abs(Xm - Xm_prev) / (np.abs(Xm_prev) + eps))
        rel_D  = np.max(np.abs(D  - D_prev ) / (np.abs(D_prev ) + eps))

        if rel_Xf < tol and rel_Xm < tol and rel_D < tol:
            break

    return Xf, Xm, D

@njit
def calc_Xf_Xm_D_iterative(
    N: int,
    S: int,
    alpha: np.ndarray,            # (N, S)
    gamma: np.ndarray,            # (N, S, S)
    pif_prime: np.ndarray,        # (i, n, k)
    pim_prime: np.ndarray,        # (i, n, k)
    tilde_tau_prime: np.ndarray,  # (i, n, k) = 1 + τ′
    w_hat: np.ndarray,            # (N,)
    V: np.ndarray,                # (N,)
    max_iter: int = 10_000,
    tol: float = 1e-3,
    damping: float = 0.3,
) -> tuple:
    """
    Fixed-point solver for (Xf′, Xm′, D′) that matches equations (E4)–(E6).
    Fully Numba-compiled (nopython mode).
    """
    # ---------- initial guess -------------------------------------------
    Xf = alpha * (w_hat * V)[:, None]   # positive baseline
    Xm = np.zeros_like(Xf)
    D  = np.zeros(N)

    eps = 1e-12  # to avoid division by zero in relative convergence test

    for _ in range(max_iter):
        Xf_prev = Xf.copy()
        Xm_prev = Xm.copy()
        D_prev  = D.copy()
        # E4
        Xf_new = calc_Xf_prime(
            alpha, pif_prime, pim_prime, tilde_tau_prime,
            w_hat, V, Xf_prev, Xm_prev, D_prev
        )
        # E5
        Xm_new = calc_Xm_prime(
            gamma, pif_prime, pim_prime, tilde_tau_prime,
            Xf_new, Xm_prev
        )
        # E6
        D_new  = calc_D_prime(
            pif_prime, pim_prime, tilde_tau_prime,
            Xf_new, Xm_new
        )
        # Damping
        Xf = damping * Xf_new + (1.0 - damping) * Xf_prev
        Xm = damping * Xm_new + (1.0 - damping) * Xm_prev
        D  = damping * D_new  + (1.0 - damping) * D_prev
        # Convergence (relative)
        rel_Xf = np.max(np.abs(Xf - Xf_prev) / (np.abs(Xf_prev) + eps))
        rel_Xm = np.max(np.abs(Xm - Xm_prev) / (np.abs(Xm_prev) + eps))
        rel_D  = np.max(np.abs(D  - D_prev ) / (np.abs(D_prev ) + eps))
        if rel_Xf < tol and rel_Xm < tol and rel_D < tol:
            break
    return Xf, Xm, D


# ---------------------------------------------------------------------
# Master routine: build the full equilibrium from primitives and shocks
# ---------------------------------------------------------------------
@njit
def generate_equilibrium(
    w_hat:           np.ndarray,   # (N,)     wage changes
    beta:            np.ndarray,   # (N,S)    labour shares β_{is}
    gamma:           np.ndarray,   # (N,S,S)  IO coefficients γ_{isk}
    theta:           np.ndarray,   # (S,)     trade elasticities θ_s
    pif:            np.ndarray,   # (N,N,S)  baseline final-goods shares π^{0f}_{ins}
    pim:            np.ndarray,   # (N,N,S)  baseline intermediate-goods shares π^{0m}_{ins}
    lambda_hat:      np.ndarray,   # (N,S)    productivity shocks Λ̂_{ns}
    df_hat:          np.ndarray,   # (N,N,S)  trade-cost shocks d̂^{f}_{ins}
    dm_hat:          np.ndarray,   # (N,N,S)  trade-cost shocks d̂^{m}_{ins}
    tilde_tau_prime: np.ndarray,   # (N,N,S)  1 + τ′  (post-shock wedges)
    alpha:           np.ndarray,   # (N,S)    Cobb-Douglas α_{is}
    V:               np.ndarray    # (N,)     value-added deflators V_i
):
    """
    End-to-end equilibrium wrapper using only explicit arrays.

    Returns
    -------
    (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf, Xm, D_prime, p_index, real_w)
    """
    # 1. Solve for costs and intermediate-goods price index
    c_hat, Pm_hat = solve_price_and_cost(w_hat, beta, gamma, theta, pim, lambda_hat, dm_hat)
    # 2. Final-goods price index
    Pf_hat = calc_Pf_hat(c_hat, theta, pif, lambda_hat, df_hat)
    # 3. Expenditure-share changes
    pif_hat = calc_pif_hat(c_hat, Pf_hat, theta, lambda_hat, df_hat)
    pim_hat = calc_pim_hat(c_hat, Pm_hat, theta, lambda_hat, dm_hat)
    # 4. Updated bilateral shares
    pif_prime = pif * pif_hat
    pim_prime = pim * pim_hat
    # 5. τ (needed for linear system)
    N, S = alpha.shape
    # 6. Solve linear system for quantities and trade balance
    # Xf_prime, Xm_prime, D_prime = calc_Xf_Xm_D(N, S, alpha, gamma, pif_prime, pim_prime,tilde_tau_prime, w_hat, V)
    Xf_prime, Xm_prime, D_prime = calc_Xf_Xm_D_iterative(N, S, alpha, gamma, pif_prime, pim_prime,tilde_tau_prime, w_hat, V)
    # 7. Aggregate consumption price index (Cobb-Douglas)
    p_index = np.exp(np.sum(alpha * np.log(Pf_hat), axis=1))
    # 8. Real wages
    real_w = w_hat / p_index
    return (c_hat, Pf_hat, Pm_hat, pif_hat, pim_hat, Xf_prime, Xm_prime, D_prime, p_index, real_w)





# def calc_HHI(pim, pif, Xm, Xf, tau_tilde):
#     """
#     - pif, pim, tau_tilde: arrays of shape (i, n, s)
#         where:
#             i = importing country index,
#             n = exporting country index,
#             s = sector index.
#     - Xm, Xf: arrays of shape (i, s) corresponding to intermediate and final goods expenditures
#         for importing countries.
#     The formula is:
#       Q_n^s = sum_{i} [ (pif[i,n,s] * Xf[i,s] + pim[i,n,s] * Xm[i,s]) / (1 + tau_tilde[i,n,s]) ]
#     and then the HHI for each exporting country n is:
#       HHI_n = sum_s ( Q_n^s / (sum_{s'} Q_n^{s'}) )^2.

#     Returns:
#       HHI: an array of shape (n,) giving the HHI for each exporting country.
#     """
#     numerator = pif * Xf[:, None, :] + pim * Xm[:, None, :]  # shape: (N, N, J)
#     term = numerator / (1 + tau_tilde)  # shape: (N, N, J)
#     Q = term.sum(axis=0)  # shape: (N, J)
#     # For each exporting country, sum over sectors to get the total Q:
#     Q_total = Q.sum(axis=1)  # shape: (N,)
#     # Compute HHI for each exporting country:
#     HHI = np.sum((Q / Q_total[:, None]) ** 2, axis=1)  # shape: (N,)
#     return HHI







