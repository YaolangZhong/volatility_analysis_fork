import numpy as np
from models import ModelParams, ModelShocks, Usage


def calc_c_hat(w_hat, Pm_hat, mp: ModelParams):
    """
    Equation (7) in the paper

    Calculate the unit cost index changes (c_hat)
    given wage changes (w_hat) and intermediate input price changes (Pm_hat).

    Endogenous variables:
        w_hat: (N,) array of wage changes
        Pm_hat: (N, J) array of intermediate input price changes
    Returns:
        c_hat: (N, J) array of unit cost index changes
    """
    log_w_hat = np.log(w_hat)
    log_Pm_hat = np.log(Pm_hat)

    # beta[n,j] * log(w_hat[n])
    wage_component = mp.beta * log_w_hat[:, np.newaxis]

    # \sum_k gamma[n,k,j] * log(Pm_hat[n,k])
    input_component = np.einsum("nkj,nk->nj", mp.gamma, log_Pm_hat)

    log_c_hat = wage_component + input_component
    c_hat = np.exp(log_c_hat)

    return c_hat


def calc_Pu_hat(c_hat, usage: Usage, mp: ModelParams, shocks: ModelShocks):
    """
    Equation (8) in the paper

    Calculate price index changes (Pu_hat) given cost index changes (c_hat).

    usage = "f" for final demand, "m" for intermediate input demand
    Raise ValueError if usage is not "f" or "m".

    Endogenous variables:
        c_hat: (N, J) array of unit cost index changes
    Returns:
        Pu_hat: (N, J) array of price index changes
    """
    if usage == "f":
        pi = mp.pif
        d_hat = shocks.df_hat
    elif usage == "m":
        pi = mp.pim
        d_hat = shocks.dm_hat
    else:
        raise ValueError(f"Invalid usage: {usage}")

    cost_index = (
        pi
        * shocks.lambda_hat[np.newaxis, :, :]
        * (c_hat[np.newaxis, :, :] * d_hat)
        ** -mp.theta[np.newaxis, np.newaxis, :]
    )
    Pu_hat = np.sum(cost_index, axis=1) ** (-1 / mp.theta)

    return Pu_hat


def calc_piu_hat(
    c_hat, P_hat, usage: Usage, mp: ModelParams, shocks: ModelShocks
):
    """
    Equation (9) in the paper

    Calculate the expenditure share after the shock (piu_prime)
    given cost index changes (c_hat) and price index changes (P_hat).

    usage = "f" for final demand, "m" for intermediate input demand
    Raise ValueError if usage is not "f" or "m".

    Endogenous variables:
        c_hat: (N, J) array of unit cost index changes
        P_hat: (N, J) array of price index changes corresponding to the usage
    Returns:
        piu_hat: (N, N, J) array of expenditure share after the shock
    """
    if usage == "f":
        d_hat = shocks.df_hat
    elif usage == "m":
        d_hat = shocks.dm_hat
    else:
        raise ValueError(f"Invalid usage: {usage}")

    # cost_term: (c_hat[n,j] * d_hat[n,i,j]) ** -theta[j] => [n,i,j]
    cost_term = (c_hat[np.newaxis, :, :] * d_hat) ** -mp.theta[
        np.newaxis, np.newaxis, :
    ]

    # numerator: lambda_hat[n,j] * cost_term[n,h,j] => [n,i,j]
    numerator = shocks.lambda_hat[np.newaxis, :, :] * cost_term

    # denominator: (P_hat[n,j]) ** -theta[j] => [n,i,j]
    denominator = P_hat**-mp.theta  # shape (N,J)
    denominator = denominator[:, np.newaxis, :]  # shape (N,N,J)

    # piu_prime[n,i,j] = numerator[n,i,j] / denominator[n,i,j]
    piu_hat = numerator / denominator

    return piu_hat


def calc_Xf_prime(
    w_hat,
    pif_hat,
    pim_hat,
    Xf_prime,
    Xm_prime,
    td_prime,
    mp: ModelParams,
    shocks: ModelShocks,
):
    """
    Equation (10) in the paper

    Calculate expenditure for final goods after the shock (Xf_prime)
    given wage changes (w_hat),
    expenditure share changes for final goods (pif_hat),
    expenditure share changes for intermediate goods (pim_hat),
    expenditure for final goods after the shock (Xf_prime),
    expenditure for intermediate goods after the shock (Xm_prime),
    tariff rates after the shock (tilde_tau_prime),
    and total demand after the shock (td_prime).

    Endogenous variables:
        w_hat: (N,) array of wage changes
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        Xf_prime: (N, J) array of expenditure for final goods after the shock
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
        tilde_tau_prime: (N, N, J) array of tariff rates after the shock
        td_prime: (N,) array of total demand after the shock
    Returns:
        Xf_prime: (N, J) array of expenditure for final goods after the shock
    """
    # Calculate pif_prime and pim_prime
    pif_prime = pif_hat * mp.pif
    pim_prime = pim_hat * mp.pim

    # Wage component: w_hat[n] * w0[n] * L0[n]
    wage_component = w_hat * mp.w0 * mp.L0

    # Tariff revenue component:
    # ((tau_tilde[n, i, j] - 1) / tau_tilde[n, i, j])
    # * (pim_prime[n, i, j] * Xm_prime[n, j]
    #   + pif_prime[n, i, j] * Xf_prime[n, j])
    # => Tariff revenue for each (country, sector) dyad
    tariff_revenue = (
        (shocks.tilde_tau_prime - 1) / shocks.tilde_tau_prime
    ) * (
        pim_prime * Xm_prime[:, np.newaxis, :]
        + pif_prime * Xf_prime[:, np.newaxis, :]
    )

    tariff_component = np.sum(tariff_revenue, axis=(1, 2))

    # Expenditure: wage_component[n] + tariff_revenue[n] + td_prime[n]
    expenditure = wage_component + tariff_component + td_prime

    # Xf_prime[n, j] = alpha[n, j] * expenditure[n]
    Xf_prime = mp.alpha * expenditure[:, np.newaxis]

    return Xf_prime


def calc_Xm_prime(
    pif_hat, pim_hat, Xf_prime, Xm_prime, mp: ModelParams, shocks: ModelShocks
):
    """
    Equation (11) in the paper

    Calculate expenditure for intermediate goods after the shock (Xm_prime)
    given expenditure share changes for final goods (pif_hat),
    expenditure share changes for intermediate goods (pim_hat),
    expenditure for final goods after the shock (Xf_prime),
    expenditure for intermediate goods after the shock (Xm_prime).

    Endogenous variables:
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        Xf_prime: (N, J) array of expenditure for final goods after the shock
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
        tilde_tau_prime: (N, N, J) array of tariff rates after the shock
    Returns:
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
    """
    # Calculate pif_prime and pim_prime
    pif_prime = pif_hat * mp.pif
    pim_prime = pim_hat * mp.pim

    # Output as final goods from country n:
    # (pif_prime[i, n, j] / tilde_tau_prime[i, n, j]) * Xf_prime[i, j]
    # => Output each (importer, exporter, sector) triplet
    output_final = (pif_prime / shocks.tilde_tau_prime) * Xf_prime[
        :, np.newaxis, :
    ]
    output_final = np.sum(output_final, axis=0)  # Sum over importers => [n, j]

    # Output as intermediate goods from country n:
    # (pim_prime[i, n, j] / tilde_tau_prime[i, n, j]) * Xm_prime[i, j]
    # => Output for each (importer, exporter, sector) triplet
    output_intermediate = (pim_prime / shocks.tilde_tau_prime) * Xm_prime[
        :, np.newaxis, :
    ]
    output_intermediate = np.sum(
        output_intermediate, axis=0
    )  # Sum over importers => [n, j]

    # Total output from country n:
    output = output_final + output_intermediate

    # Expenditure for intermediate goods:
    # gamma[n, h, j] * output[n, h]
    # => Sum over input sector => [n, j]
    Xm_component = mp.gamma * output[:, :, np.newaxis]
    Xm_prime = np.sum(Xm_component, axis=1)

    return Xm_prime


def calc_td_prime(
    pif_hat, pim_hat, Xf_prime, Xm_prime, mp: ModelParams, shocks: ModelShocks
):
    """
    Equation (12) in the paper

    Calculate trade SURPLUS after the shock (td_prime)
    given expenditure share changes for final goods (pif_hat),
    expenditure share changes for intermediate goods (pim_hat),
    expenditure for final goods after the shock (Xf_prime),
    expenditure for intermediate goods after the shock (Xm_prime),
    and tariff rates after the shock (tilde_tau_prime).

    Endogenous variables:
        pif_hat: (N, N, J) array of expenditure share changes for final goods
        pim_hat: (N, N, J) array of expenditure share changes for intermediate goods
        Xf_prime: (N, J) array of expenditure for final goods after the shock
        Xm_prime: (N, J) array of expenditure for intermediate goods after the shock
        tilde_tau_prime: (N, N, J) array of tariff rates after the shock
    Returns:
        td_prime: (N,) array of trade deficit after the shock
    """

    # Calculate pif_prime and pim_prime
    pif_prime = pif_hat * mp.pif
    pim_prime = pim_hat * mp.pim

    # import_volume array for each (importer, exporter, sector) triplet
    import_volume = (
        pif_prime * Xf_prime[:, np.newaxis, :]
        + pim_prime * Xm_prime[:, np.newaxis, :]
    ) / shocks.tilde_tau_prime

    # transpose import_volume to make (exporter, importer, sector) triplet
    export_volume = import_volume.transpose(1, 0, 2)

    # Calculate trade deficit for each (exporter, importer, sector) triplet
    trade_balance = import_volume - export_volume

    # Sum over exporting countries and sectors to obtain trade deficit for each importer
    td_prime = np.sum(trade_balance, axis=(1, 2))

    return td_prime


# ------------------------------------------------------------------------------
# Test the functions
def generate_test_parameters():
    N = 2
    J = 2
    alpha = np.ones((N, J)) / J
    beta = np.ones((N, J)) * 0.3
    gamma = np.ones((N, J, J)) * 0.7 / J
    theta = np.ones(J) * 8.0
    pif = np.ones((N, N, J)) / N
    pim = np.ones((N, N, J)) / N
    tilde_tau = np.ones((N, N, J)) * 1.1
    indices = np.arange(N)
    tilde_tau[indices, indices, :] = 1.0
    Xf = np.ones((N, J)) * 100
    Xm = np.ones((N, J)) * 100
    w0 = np.ones(N)
    L0 = np.ones(N) * 100
    td = np.zeros(N)

    params = ModelParams(
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

    # Check consistency
    if params.check_consistency(mute=True):
        return params
    else:
        return None


def generate_test_shocks(params):
    N = params.N
    J = params.J
    df_hat = np.ones((N, J))
    dm_hat = np.ones((N, J))
    lambda_hat = np.ones((N, J))
    tilde_tau_prime = np.ones((N, N, J))
    shocks = ModelShocks(
        params=params,
        df_hat=df_hat,
        dm_hat=dm_hat,
        lambda_hat=lambda_hat,
        tilde_tau_prime=tilde_tau_prime,
    )
    return shocks


def test_equations(params):
    """
    Function to test the equations
    """
    N = params.N
    J = params.J

    # ========== Test calc_c_hat ==========
    w_hat = np.ones(N)
    w_hat[0] = 1.5 ** (1 / 0.3)

    Pm_hat = np.ones((N, J))
    Pm_hat[0,] = 1.5 ** (1 / 0.7)
    c_hat = calc_c_hat(w_hat, Pm_hat, params)

    # c_hat[0,] should be 1.5 * 1.5 = 2.25
    print("c_hat:", c_hat)

    # ========== Test calc_Pu_hat ==========

    # ...


if __name__ == "__main__":
    params = generate_test_parameters()
    shocks = generate_test_shocks(params)
    test_equations(params)
