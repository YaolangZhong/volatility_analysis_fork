import numpy as np
from models import ModelParams


def define_theta():
    """
    Trade elasticity: take from "Global Value Chains and Aggregate Income Volatility"

    ---------- Tradable Sectors ----------
    1 Agriculture 2 Fishing
    -> Sector 1 -> 1. Agriculture, Hunting, Forestry and Fishing
    -> 6.26

    3 Mining and Quarrying
    -> Sector 2 -> 2. Mining and Quarrying
    -> 8.05

    4 Food & Beverages
    -> Sector 3 -> 3. Food, Beverages and Tobacco
    -> 7.31

    5 Textiles and Wearing Apparel
    -> Sector 4 -> 4. Textile Products, Leather Products and Footwear
    -> 6.31

    6 Wood and Paper
    -> Sector 5 -> 6. Wood and Products of Wood and Cork
                &  7. Pulp, Paper, Paper, Printing and Publishing
    -> (9.12 +11.37) / 2 = 10.245

    7 Petroleum, Chemical and Non-Metallic Mineral Products
    -> Sector 6 -> 8. Coke, Refined Petroleum and Nuclear Fuel
                &  9. Chemicals and Chemical Products
                & 10. Rubber and Plastics
                & 11. Other Non-Metallic Mineral
    -> (6.1 + 6.31 + 6.22 + 4.78) / 4 = 5.8525

    8 Metal Products
    -> Sector 7 -> 12. Basic Metals and Fabricated Metal
    -> 7.78

    9 Electrical and Machinery
    -> Sector 8 -> 13. Machinery, Nec
                &  14. Electrical and Optical Equipment
    -> (7.43 + 9.69) / 2 = 8.56

    10 Transport Equipment
    -> Sector 9 -> 15. Transport Equipment
    -> 7.13

    11 Other Manufacturing 12 Recycling
    -> Sector 10 -> 16. Manufacturing, Nec; Recycling
    -> 8.01

    ---------- Non Tradable Sectors ----------
    13 Electricity, Gas and Water;
    14 Construction;
    15 Maintenance and Repair;
    16 Wholesale Trade;
    17 Retail Trade;
    18 Hotels and Restaurants;
    19 Transport;
    20 Post and Telecommunications;
    21 Financial Intermediation and Business Activities;
    22 Public Administration;
    23 Education, Health and Other Services;
    24 Private Households;
    25 Others
    -> Use the average of the tradable sectors = 7.31
    """
    theta = np.array(
        [
            6.26,
            8.05,
            7.31,
            6.31,
            10.25,
            5.85,
            7.78,
            8.56,
            7.13,
            8.01,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
            7.31,
        ]
    )

    return theta


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

    td = np.zeros(N)

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
