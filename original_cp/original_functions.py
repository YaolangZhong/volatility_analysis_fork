import numpy as np
import pandas as pd
from .original_models import OldModelParams


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


def load_data(
    base_path,
    country_list_path="3_Result/results_Eora/country_list.csv",
    sector_list_path="3_Result/results_Eora/sector_list.csv",
    alpha_path="3_Result/results_parameters/Alpha_nj_2017.csv",
    beta_path="3_Result/results_parameters/Gamma_nj_2017.csv",
    gamma_path="3_Result/results_parameters/Gamma_njk_2017.csv",
    pi_path="3_Result/results_parameters/trade_share_2017.csv",
    tariff_path="3_Result/results_parameters/tariff_2017.csv",
    X_path="3_Result/results_parameters/X_nj_2017.csv",
):
    """
    Load the data from the CSV files and create OldModelParams.
    ---------- Arguments ----------
    base_path : str
        Path to the root directory of the project.
    country_list_path : str
        Path to the country list CSV file.
    sector_list_path : str
        Path to the sector list CSV file.
    intensity_path : str
        Path to the carbon intensity CSV file.
    alpha_path : str
        Path to the alpha CSV file.
    beta_path : str
        Path to the beta CSV file.
    gamma_path : str
        Path to the gamma CSV file.
    pi_path : str
        Path to the pi CSV file.
    tariff_path : str
        Path to the tariff CSV file.
    X_path : str
        Path to the X CSV file.

    ---------- Returns ----------
    model_params : OldModelParams
        An instance of OldModelParams filled with the loaded parameters.
    country_list : list of str
        List of country names.
    sector_list : list of str
        List of sector names.
    tradable_sector_list : list of str
        The first 10 sectors that are considered tradable (based on the code).
    intensity_matrix : (N, J) array
        Carbon intensity matrix [country x sector].
    """

    # 1. Load country and sector lists
    country_list_df = pd.read_csv(f"{base_path}/{country_list_path}")
    country_list = country_list_df.iloc[:, 0].tolist()
    N = len(country_list)

    sector_list_df = pd.read_csv(f"{base_path}/{sector_list_path}")
    sector_list = sector_list_df.iloc[:, 0].tolist()
    J = len(sector_list)

    tradable_sector_list = sector_list[0:10]

    print(
        f"We have {N} countries and {J} sectors. The first 10 sectors are tradable sectors."
    )

    # 2. Load alpha, beta, gamma, etc.
    alpha_df = pd.read_csv(f"{base_path}/{alpha_path}").iloc[:, 1:]
    alpha = alpha_df.to_numpy()  # (N, J)

    beta_df = pd.read_csv(f"{base_path}/{beta_path}").iloc[:, 1:]
    beta = beta_df.to_numpy()  # (N, J)

    gamma_df = pd.read_csv(f"{base_path}/{gamma_path}").iloc[:, 1:]
    gamma_np = gamma_df.to_numpy().reshape((N, J, J))
    # One need to transpose the gamma matrix to match the order of the indices
    gamma = np.transpose(gamma_np, (0, 2, 1))  # (N, J, J)

    # 3. Load pi (trade share)
    pi_df_temp = pd.read_csv(f"{base_path}/{pi_path}")
    pi_df = pi_df_temp.iloc[:, 1:]
    pi_df["Exporter_Sector"] = pi_df["Exporter"] + "_" + pi_df["Sector"]

    pi_df["Importer"] = pd.Categorical(
        pi_df["Importer"], categories=pi_df["Importer"].unique(), ordered=True
    )
    pi_df["Exporter_Sector"] = pd.Categorical(
        pi_df["Exporter_Sector"],
        categories=pi_df["Exporter_Sector"].unique(),
        ordered=True,
    )

    pi_matrix = pi_df.pivot_table(
        index="Importer",
        columns="Exporter_Sector",
        values="Share",
        aggfunc="first",
        observed=False,
    )
    pi_np = pi_matrix.to_numpy()
    pi = pi_np.reshape((N, N, J))

    # 4. Load tariffs
    tariff_raw = pd.read_csv(f"{base_path}/{tariff_path}")
    tariff_df = tariff_raw.iloc[:, 1:]
    tariff_df["Exporter_Sector"] = (
        tariff_df["Exporter"] + "_" + tariff_df["Sector"]
    )

    tariff_df["Importer"] = pd.Categorical(
        tariff_df["Importer"],
        categories=tariff_df["Importer"].unique(),
        ordered=True,
    )
    tariff_df["Exporter_Sector"] = pd.Categorical(
        tariff_df["Exporter_Sector"],
        categories=tariff_df["Exporter_Sector"].unique(),
        ordered=True,
    )

    tariff_matrix = tariff_df.pivot_table(
        index="Importer",
        columns="Exporter_Sector",
        values="Tariff",
        aggfunc="first",
        observed=False,
    )
    tariff_np = tariff_matrix.to_numpy()
    tariff_base = tariff_np.reshape((N, N, J))
    tilde_tau = tariff_base + 1

    # 5. Load X
    X_df = pd.read_csv(f"{base_path}/{X_path}").iloc[:, 1:]
    X = X_df.to_numpy()  # (N, J)

    # 6. Set theta
    theta = define_theta()

    # 7. Caluculate Value Added and Trade Deficit
    VA = np.zeros(N)
    for n in range(N):
        for j in range(J):
            inner_sum = 0.0
            for i in range(N):
                inner_sum += X[i, j] * (pi[i, n, j] / tilde_tau[i, n, j])
            VA[n] += beta[n, j] * inner_sum

    D = np.zeros(N)
    for n in range(N):
        for j in range(J):
            for i in range(N):
                IM = X[n, j] * pi[n, i, j] / tilde_tau[n, i, j]
                EX = X[i, j] * pi[i, n, j] / tilde_tau[i, n, j]
                D[n] += EX - IM

    # 9. Create OldModelParams
    model_params = OldModelParams(
        N=N,
        J=J,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta=theta,
        pi=pi,
        tilde_tau=tilde_tau,
        X=X,
        VA=VA,
        D=D,
    )

    # 10. Return OldModelParams and other data
    return (model_params, country_list, sector_list, tradable_sector_list)


def generate_rand_params(N: int, J: int):
    """
    Generate random parameters for the model.

    ---------- Arguments ----------
    N : int
        Number of countries.
    J : int
        Number of sectors.

    ---------- Returns ----------
    model_params : OldModelParams
        An instance of OldModelParams filled with the generated parameters.
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

    pi = np.ones((N, N, J)) + np.random.rand(N, N, J)  # 1 ~ 2
    pi_sum = pi.sum(axis=1, keepdims=True)
    pi = pi / pi_sum

    tilde_tau = np.random.rand(N, N, J) + 1  # 1 ~ 2
    for i in range(N):
        tilde_tau[i, i, :] = 1

    X = np.ones((N, J)) * 100 + np.random.rand(N, J) * 900  # 100 ~ 1000
    VA = np.ones(N) * 100 + np.random.rand(N) * 900  # 100 ~ 1000

    D = np.zeros(N)

    mp = OldModelParams(
        N=N,
        J=J,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta=theta,
        pi=pi,
        tilde_tau=tilde_tau,
        X=X,
        VA=VA,
        D=D,
    )

    if mp.check_consistency():
        return mp
    else:
        print("Generated parameters are inconsistent.")
        return None
