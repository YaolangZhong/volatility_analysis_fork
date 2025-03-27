import numpy as np
from models import ModelParams


def generate_rand_params(N: int, J: int):
    """
    Generate random parameters for the model.

    ---------- Arguments ----------
    N : int
        Number of countries.
    J : int
        Number of sectors.

    ---------- Returns ----------
    model_params : ModelParams
        An instance of ModelParams filled with the generated parameters.
    """

    alpha = np.ones((N, J)) + np.random.rand(N, J)  # 1 ~ 2
    alpha = alpha / alpha.sum(axis=1, keepdims=True)

    beta = np.ones((N, J)) * 2 + np.random.rand(N, J)  # 2 ~ 3
    gamma = (
        np.ones((N, J, J)) * 7 / J + np.random.rand(N, J, J) / J
    )  # gamma_sum = 7 ~ 8

    # Calculate beta and gamma so that they sum to 1
    sum_gamma = gamma.sum(axis=1)
    beta_gamma_sum = beta + sum_gamma

    for n in range(N):
        for j_ in range(J):
            denom = beta_gamma_sum[n, j_]
            if denom < 1e-15:
                # if denom is too small, set beta=1 and gamma=0
                beta[n, j_] = 1.0
                gamma[n, :, j_] = 0.0
            else:
                # normalize beta and gamma
                beta[n, j_] /= denom
                gamma[n, :, j_] /= denom

    theta = np.random.rand(J)
    theta = theta * 4 + 6  # 6 ~ 10

    pif = np.ones((N, N, J)) + np.random.rand(N, N, J)  # 1 ~ 2
    pif_sum = pif.sum(axis=1, keepdims=True)
    pif = pif / pif_sum

    pim = np.ones((N, N, J)) + np.random.rand(N, N, J)  # 1 ~ 2
    pim_sum = pim.sum(axis=1, keepdims=True)
    pim = pim / pim_sum

    tilde_tau = np.random.rand(N, N, J) + 1  # 1 ~ 2
    for i in range(N):
        tilde_tau[i, i, :] = 1

    Xf = np.ones((N, J)) * 100 + np.random.rand(N, J) * 900  # 100 ~ 1000
    Xm = np.ones((N, J)) * 100 + np.random.rand(N, J) * 900  # 100 ~ 1000
    w0 = np.ones(N)  # 1
    L0 = np.ones(N) * 100 + np.random.rand(N) * 900  # 100 ~ 1000

    td = np.random.rand(N) * 10
    td = td - td.sum() / N

    mp = ModelParams(
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

    if mp.check_consistency():
        return mp
    else:
        print("Generated parameters are inconsistent.")
        return None


def generate_simple_params():
    """
    generate fixed parameters for N = 2, J = 1
    """
    N, J = 2, 1
    alpha = np.array([[1.0], [1.0]])
    beta = np.array([[0.4], [0.4]])
    gamma = np.array([[[0.6]], [[0.6]]])
    theta = np.array([8.0])
    pif = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
    pim = np.array([[[0.5], [0.5]], [[0.5], [0.5]]])
    tilde_tau = np.ones((N, N, J))
    Xf = np.array([[500.0], [500.0]])
    Xm = np.array([[500.0], [500.0]])
    w0 = np.array([1.0, 1.0])
    L0 = np.array([500.0, 500.0])
    td = np.array([0.0, 0.0])

    mp = ModelParams(
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

    mp.check_consistency(mute=False)
    return mp


# def generate_symmetric_params(N: int, J: int) -> ModelParams:
#     """
#     Generate symmetric parameters for the model. Every country receives
#     identical parameter values to ensure complete symmetry across countries.
#     """

#     # 1) alpha: pick a single random vector alpha_j, then replicate for all N
#     alpha_j = np.ones(J) + np.random.rand(J)  # ~ [1,2]
#     alpha_j /= alpha_j.sum()  # sums to 1 across sectors
#     alpha = np.tile(alpha_j, (N, 1))

#     # 2) beta and gamma: ensure sum_k gamma[n,k,j] + beta[n,j] = 1
#     #    for each row j, then replicate for all countries
#     #    We first generate a single row (beta_j, gamma_jj) that sums to 1,
#     #    then replicate that across N.
#     beta_j = (np.ones(J) * 2) + np.random.rand(J)  # ~ [2,3]
#     gamma_jj = (np.ones((J, J)) * (7.0 / J)) + (
#         np.random.rand(J, J) / J
#     )  # each row ~ [7,8]/J

#     # Normalize each row so that beta_j[j] + sum(gamma_jj[j, :]) = 1 exactly
#     for j_ in range(J):
#         row_sum = gamma_jj[j_, :].sum() + beta_j[j_]
#         # Safeguard for extremely small denominators
#         if row_sum < 1e-15:
#             beta_j[j_] = 1.0
#             gamma_jj[j_, :] = 0.0
#         else:
#             gamma_jj[j_, :] /= row_sum
#             beta_j[j_] /= row_sum

#     beta = np.tile(beta_j, (N, 1))  # shape (N,J)
#     gamma = np.tile(gamma_jj[np.newaxis, :, :], (N, 1, 1))  # shape (N,J,J)

#     # 3) theta: technology parameters (same for all countries but vary by sector)
#     theta = np.random.rand(J) * 4 + 6  # ~ [6,10]

#     # 4) pif, pim: final & intermediate input sourcing distribution
#     #    We pick a single distribution over N for each and replicate across i, j
#     dist_pif = np.random.rand(N)
#     dist_pif /= dist_pif.sum()
#     pif = np.zeros((N, N, J))
#     for i in range(N):
#         for j_ in range(J):
#             pif[i, :, j_] = dist_pif

#     dist_pim = np.random.rand(N)
#     dist_pim /= dist_pim.sum()
#     pim = np.zeros((N, N, J))
#     for i in range(N):
#         for j_ in range(J):
#             pim[i, :, j_] = dist_pim

#     # 5) tilde_tau: trade costs
#     #    For full symmetry, let off-diagonals = T in [1,2], diagonals = 1
#     T = 1.0 + np.random.rand()  # ~ [1,2]
#     tilde_tau = np.ones((N, N, J))
#     for i in range(N):
#         for n in range(N):
#             if i != n:
#                 tilde_tau[i, n, :] = T

#     # 6) Xf, Xm: final and intermediate demands, pick one random vector over J and replicate to each country
#     Xf_j = 100 + np.random.rand(J) * 900  # ~ [100,1000]
#     Xf = np.tile(Xf_j, (N, 1))

#     Xm_j = 100 + np.random.rand(J) * 900  # ~ [100,1000]
#     Xm = np.tile(Xm_j, (N, 1))

#     # 7) w0: wage for each country (set to 1 for symmetry)
#     w0 = np.ones(N)

#     # 8) L0: labor endowment, pick one random value, replicate for all countries
#     L = 100 + np.random.rand() * 900  # ~ [100,1000]
#     L0 = np.ones(N) * L

#     # 9) td: policy/tariff or dummy param, set to zero
#     td = np.zeros(N)

#     # Build ModelParams
#     mp = ModelParams(
#         N=N,
#         J=J,
#         alpha=alpha,
#         beta=beta,
#         gamma=gamma,
#         theta=theta,
#         pif=pif,
#         pim=pim,
#         tilde_tau=tilde_tau,
#         Xf=Xf,
#         Xm=Xm,
#         w0=w0,
#         L0=L0,
#         td=td,
#     )

#     # Final check
#     if mp.check_consistency():
#         return mp
#     else:
#         print("Generated symmetric parameters are inconsistent.")
#         return None


# def generate_rand_params_without_usage(N: int, J: int):
#     """
#     Generate random parameters for the model.
#     It generates the parameters without distinguishing between final and intermediate goods.

#     ---------- Arguments ----------
#     N : int
#         Number of countries.
#     J : int
#         Number of sectors.

#     ---------- Returns ----------
#     model_params : ModelParams
#         An instance of ModelParams filled with the generated parameters.
#     """

#     alpha = np.ones((N, J)) + np.random.rand(N, J)  # 1 ~ 2
#     alpha = alpha / alpha.sum(axis=1, keepdims=True)

#     beta = np.ones((N, J)) * 2 + np.random.rand(N, J)  # 2 ~ 3
#     gamma = (
#         np.ones((N, J, J)) * 7 / J + np.random.rand(N, J, J) / J
#     )  # gamma_sum = 7 ~ 8

#     # Calculate beta and gamma so that they sum to 1
#     sum_gamma = gamma.sum(axis=1)
#     beta_gamma_sum = beta + sum_gamma

#     for n in range(N):
#         for j_ in range(J):
#             denom = beta_gamma_sum[n, j_]
#             if denom < 1e-15:
#                 # if denom is too small, set beta=1 and gamma=0
#                 beta[n, j_] = 1.0
#                 gamma[n, :, j_] = 0.0
#             else:
#                 # normalize beta and gamma
#                 beta[n, j_] /= denom
#                 gamma[n, :, j_] /= denom

#     theta = np.random.rand(J)
#     theta = theta * 4 + 6  # 6 ~ 10

#     pi = np.ones((N, N, J)) + np.random.rand(N, N, J)  # 1 ~ 2
#     pi_sum = pi.sum(axis=1, keepdims=True)
#     pi = pi / pi_sum

#     pif, pim = pi, pi

#     tilde_tau = np.random.rand(N, N, J) + 1  # 1 ~ 2
#     for i in range(N):
#         tilde_tau[i, i, :] = 1

#     Xf = np.ones((N, J)) * 100 + np.random.rand(N, J) * 900  # 100 ~ 1000
#     Xm = np.ones((N, J)) * 100 + np.random.rand(N, J) * 900  # 100 ~ 1000
#     w0 = np.ones(N)  # 1
#     L0 = np.ones(N) * 100 + np.random.rand(N) * 900  # 100 ~ 1000

#     td = np.zeros(N)

#     mp = ModelParams(
#         N=N,
#         J=J,
#         alpha=alpha,
#         beta=beta,
#         gamma=gamma,
#         theta=theta,
#         pif=pif,
#         pim=pim,
#         tilde_tau=tilde_tau,
#         Xf=Xf,
#         Xm=Xm,
#         w0=w0,
#         L0=L0,
#         td=td,
#     )

#     if mp.check_consistency():
#         return mp
#     else:
#         print("Generated parameters are inconsistent.")
#         return None


# if __name__ == "__main__":
#     # Define the number of countries and sectors
#     N = 5
#     J = 3

#     # Generate symmetric parameters
#     mp = generate_simple_params()
#     print(mp)
