import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Tuple
from models import ModelParams

def read_io_excel(
    file_path: str | Path,
    N: int,
    S: int,
    *,
    sheets: Tuple[str, str, str] = ("M", "F", "V"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read an IO table stored in three sheets (M, F, V) of an Excel file.

    Parameters
    ----------
    file_path : str or Path
        Location of the workbook.
    N, S : int
        Number of countries and sectors.
    sheets : tuple of str, default ("M", "F", "V")
        The sheet names that contain the intermediate-goods matrix,
        the final-goods matrix, and the value-added vector.

    Returns
    -------
    M : ndarray, shape (N·S, N·S)
    F : ndarray, shape (N·S, N)
    V : ndarray, shape (N·S,)
    """
    file_path = Path(file_path)

    # ── 1. read the three sheets ────────────────────────────────────────────────
    M = pd.read_excel(file_path, sheet_name="M", header=None).to_numpy(float)
    F = pd.read_excel(file_path, sheet_name="F", header=None).to_numpy(float)
    V = pd.read_excel(file_path, sheet_name="V", header=None).to_numpy(float).ravel()


    # ── 2. sanity-check the raw shapes ─────────────────────────────────────────
    if M.shape != (N * S, N * S):
        raise ValueError(f"M sheet shape {M.shape} ≠ {(N*S, N*S)}")
    if F.shape != (N * S, N):
        raise ValueError(f"F sheet shape {F.shape} ≠ {(N*S, N)}")
    if V.size != N * S:
        raise ValueError(f"V sheet length {V.size} ≠ {N*S}")

    return M, F, V

def transform_io_tables(
    M_2d: np.ndarray,   # shape (N·S, N·S)   rows = exporter-sector (e,t), cols = importer-sector (i,s)
    F_2d: np.ndarray,   # shape (N·S, N)     rows = exporter-sector (e,t), cols = importer i
    V_2d: np.ndarray,   # shape (N·S,)       stacked by exporter-sector (e,t)
    N: int,
    S: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 2-D IO sheets to the tensor layout used by the model.

    Returns
    -------
    M : ndarray, shape (N, N, S, S)
        M[importer i, exporter e, import-sector s, export-sector t]
    F : ndarray, shape (N, N, S)
        F[importer i, exporter e, export-sector t]
    V : ndarray, shape (N, S)
        V[country i, sector s]
    """
    # --- M  -----------------------------------------------------------
    # 1)  reshape rows→(e,t), cols→(i,s)    → (exporter, export-sec, importer, import-sec)
    M_temp = M_2d.reshape(N, S, N, S)

    # 2)  reorder axes to (importer, exporter, import-sec, export-sec)
    M = M_temp.transpose(2, 0, 3, 1)        # (N, N, S, S)

    # --- F  -----------------------------------------------------------
    # rows = (exporter, export-sec), cols = importer
    F_temp = F_2d.reshape(N, S, N)          # (e, t, i)
    F = F_temp.transpose(2, 0, 1)           # (i, e, t)   → (N, N, S)

    # --- V  -----------------------------------------------------------
    V = V_2d.reshape(N, S)                  # (i, s)

    return M, F, V

def calc_D(
    M: np.ndarray,          # shape (N, N, S, S)
    F: np.ndarray           # shape (N, N, S)
) -> np.ndarray:
    """
    Compute imports, exports, and trade-deficit vector for each country.

    Parameters
    ----------
    M : ndarray (N, N, S, S)
        Intermediate-goods flows:
        (importer country, exporter country, import sector, export sector)
    F : ndarray (N, N, S)
        Final-goods flows:
        (importer country, exporter country, export sector)

    Returns
    -------
    imports  : ndarray (N,)   total goods absorbed by each country
    exports  : ndarray (N,)   total goods supplied by each country
    deficit  : ndarray (N,)   imports - exports  (positive ⇒ deficit)
    """
    # imports: keep importer axis 0, sum over exporters & sectors
    imports = M.sum(axis=(1, 2, 3)) + F.sum(axis=(1, 2))

    # exports: keep exporter axis 1, sum over importers & sectors
    exports = M.sum(axis=(0, 2, 3)) + F.sum(axis=(0, 2))

    # deficit vector (D)
    deficit = imports - exports
    return deficit

def calc_Y(
        V: np.ndarray,          # shape (N, S)
        D: np.ndarray,          # shape (N,)
        ) -> np.ndarray:   
    """
    Compute the total output vector for each country.

    Parameters
    ----------
    V : ndarray (N, S)
        Value-added vector:
        (country, sector)
    D : ndarray (N,)
        Trade-deficit vector:
        (country)

    Returns
    -------
    Y : ndarray (N,)   total output of each country
    """
    # total output = value added + trade deficit
    Y = V.sum(axis=1) + D
    return Y

def calc_alpha(F: np.ndarray) -> np.ndarray:
    """
    Compute α[n,s] = share of country-n final-goods spending that falls on good s.

    Parameters
    ----------
    F : ndarray, shape (N, S, N)
        Final-goods tensor  (importer-country, exporter-country, export-sector).

    Returns
    -------
    alpha : ndarray, shape (N, S)
        Rows sum to one (up to floating error); NaNs if a country has zero final demand.
    """
    # ── 1. total spending on each good s in importer n ─────────────────────────
    #     sum over exporter axis 1  ⇒  (importer, sector)
    exp_ns = F.sum(axis=1)               # (N,S)

    # ── 2. country totals ──────────────────────────────────────────────────────
    totals = exp_ns.sum(axis=1, keepdims=True)  # (N,1)

    # avoid divide-by-zero: if a country has no final demand, leave shares as 0
    with np.errstate(invalid="ignore", divide="ignore"):
        alpha = np.where(totals != 0, exp_ns / totals, 0.0)

    return alpha

def calc_beta(M: np.ndarray, F: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    β[n,s] = share of value-added in the gross output of
             exporter-country n, export-sector s.

    Parameters
    ----------
    M : ndarray (N, N, S, S)
        Intermediate-goods tensor
        (importer i, exporter n, import-sector s, export-sector t).
    F : ndarray (N, N, S)
        Final-goods tensor
        (importer i, exporter n, export-sector t).
    V : ndarray (N, S)
        Value-added by exporter-country and export-sector.

    Returns
    -------
    beta : ndarray (N, S)
           Value-added share by exporter-country and sector.
    """
    # ── Gross output by exporter-country n and export-sector t ────────────────
    #   • Intermediate deliveries: sum over importers i and import-sectors s
    interm_out = M.sum(axis=(0, 2))          # (N, S)

    #   • Final-goods deliveries: sum over importers i
    final_out  = F.sum(axis=0)               # (N, S)

    X = interm_out + final_out               # gross output (N, S)

    # ── Value-added share β = V / X, handling zero-output sectors ────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.where(X != 0, V / X, 0.0)

    return beta

def calc_gamma(M: np.ndarray) -> np.ndarray:
    """
    Compute γ[n,s,k] = share of intermediate inputs from sector k
    in the production of sector s in country n.

    Parameters
    ----------
    M : ndarray, shape (N, N, S, S)
        Intermediate-goods tensor:
        (importer country, exporter country, import sector, export sector).

    Returns
    -------
    gamma : ndarray, shape (N, S, S)
        Rows over k sum to one for each (n,s) pair.
    """
    # 1. Aggregate over exporters axis 1  →  flows[importer, import-sector, export-sector]
    flows = M.sum(axis=1)                # (N, S, S)

    # 2. No transpose needed (already (n, s, k))
    flows_nsk = flows

    # 3. Normalise by total inputs into each (n, s)
    denom = flows_nsk.sum(axis=2, keepdims=True)  # (N, S, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(denom != 0, flows_nsk / denom, 0.0)

    return gamma

def calc_pif(F: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    F : ndarray, shape (N, N, S)
        Final-goods tensor (importer, exporter, sector).

    Returns
    -------
    pif : ndarray, shape (N, N, S)
        π^F[importer n, exporter i, export-sector s]
    """
    flows = F                                # (N, N, S)

    denom = flows.sum(axis=1, keepdims=True) # (N, 1, S)

    with np.errstate(divide="ignore", invalid="ignore"):
        pif = np.where(denom != 0, flows / denom, 0.0)

    return pif

def calc_pim(M: np.ndarray) -> np.ndarray:
    """
    πᴹ[importer i, exporter e, export-sector t]

    M : (importer i, exporter e, import-sector s, export-sector t)
    """
    # 1. collapse the import-sector axis (s)  →  (i, e, t)
    flows = M.sum(axis=2)                      # shape (N, N, S)

    # 2. normalise across exporters e for each (i,t)
    denom = flows.sum(axis=1, keepdims=True)   # (N, 1, S)

    with np.errstate(divide="ignore", invalid="ignore"):
        pim = np.where(denom != 0, flows / denom, 0.0)

    return pim

def calc_piall(M: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Combined final- and intermediate-goods import shares.

    Parameters
    ----------
    M : ndarray, shape (N, N, S, S)
        Intermediate-goods tensor (importer, exporter, import-sector, export-sector)
    F : ndarray, shape (N, N, S)
        Final-goods tensor         (importer, exporter, export-sector)

    Returns
    -------
    piall : ndarray, shape (N, N, S)
        π_all[importer n, exporter i, export-sector s]
    """
    # --- total flows of export-sector s from exporter i to importer n ----------
    interm = M.sum(axis=2)          # sum over import-sector -> (N, N, S)
    total  = interm + F             # already (N, N, S)
    flows = total                   # (N, N, S)
    denom = flows.sum(axis=1, keepdims=True)  # (N, 1, S)

    with np.errstate(divide="ignore", invalid="ignore"):
        piall = np.where(denom != 0, flows / denom, 0.0)

    return piall

def calc_Xf_Xm(alpha: np.ndarray,
               Y: np.ndarray,
               M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Final- and intermediate-goods expenditure matrices.

    Parameters
    ----------
    alpha : ndarray, shape (N, S)
        Expenditure-share matrix.
    Y     : ndarray, shape (N,)
        Income vector by country.
    M     : ndarray, shape (N, N, S, S)
        Intermediate-goods tensor
        (importer-country, exporter-country, import-sector, export-sector).

    Returns
    -------
    Xf : ndarray, shape (N, S)
         Final-goods expenditure  (Xf = alpha * Y).
    Xm : ndarray, shape (N, S)
         Intermediate-goods expenditure.
    """
    # Final-goods expenditure: broadcast Y across sectors
    Xf = alpha * Y[:, None]          # (N, S)

    # Intermediate-goods expenditure: sum over exporters (1) and output-sector (3)
    interm_out = M.sum(axis=(1, 3))   # (N, S)

    Xm = interm_out

    return Xf, Xm


if __name__ == "__main__":

    data_path = "data/toy_IO.xlsx"
    N = 2
    S = 2
    M_2d, F_2d, V_2d = read_io_excel(data_path, N, S)
    M, F, V = transform_io_tables(M_2d, F_2d, V_2d, N, S)

    D = calc_D(M, F)
    Y = calc_Y(V, D)
    alpha = calc_alpha(F)
    beta = calc_beta(M, F, V)
    gamma = calc_gamma(M)
    pif = calc_pif(F)
    pim = calc_pim(M)
    piall = calc_piall(M, F)
    Xf, Xm = calc_Xf_Xm(alpha, Y, M)

    theta_constant = 7
    theta = np.ones(N) * theta_constant

    tau_constant = 0
    tilde_tau = np.ones((N, N, S)) + tau_constant

    V = np.sum(V, axis=1)  # (N,)
    gamma *= (1 - beta[:, None]) # (N, S, S)
    country_lists = {"A", "B"}
    sector_lists = {"1", "2"}

    params = ModelParams(
    N=N,
    S=S,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
    theta=theta,
    pif=pif,
    pim=pim,
    tilde_tau=tilde_tau,
    Xf=Xf,
    Xm=Xm,
    V=V,
    D=D,
    country_list=country_lists,
    sector_list=sector_lists,
)

    output_dir = "experiments/toy_IO"
    os.makedirs(output_dir, exist_ok=True)  # create directory if needed
    output_path = os.path.join(output_dir, "params.npz")
    params.save_to_npz(output_path)
