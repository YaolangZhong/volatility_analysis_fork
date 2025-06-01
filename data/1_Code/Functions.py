import pandas as pd
import numpy as np
import os

############################################################################################################### 
################################# These are functions using in 1_Clean_Eora.ipynb #############################
############################################################################################################### 


#############################################
# This is the function to read in Eora Data #
#############################################

def read_files(folder_path, years, file_template):
    """
    Read BP data files for each specified year.

    Parameters:
    - folder_path: base directory containing Eora26_<year>_bp subfolders.
    - years: list of years to read.
    - file_template: filename or template with 'XXXX' as year placeholder.

    Returns:
    - dict of DataFrames keyed by year as string.
    """
    data_dict = {}

    for year in years:
        # Replace placeholder if present
        if 'XXXX' in file_template:
            file_name = file_template.replace('XXXX', str(year))
        else:
            file_name = file_template

        # Construct directory and full file path (BP only)
        subfolder = f"Eora26_{year}_bp"
        file_path = os.path.join(folder_path, subfolder, file_name)

        if os.path.exists(file_path):
            # Read tab-delimited data without header
            df = pd.read_csv(file_path, sep='	', header=None, dtype=str)
            data_dict[str(year)] = df
        else:
            print(f"Warning: File not found: {file_path}")

    return data_dict


##############################################################
# This is the function to check the consistency of Eora Data #
##############################################################

def check_consistency(data_dict):
    """
    Check whether country and sector codes are consistent across years, while keeping the original order from the reference year.

    Parameters:
    - data_dict: dict of DataFrames keyed by year (strings).

    Returns:
    - consistency_df: pandas.DataFrame with Year, Country_Consistent, Sector_Consistent.
    - country_list: list of country codes from reference year (original order).
    - sector_list: list of sector codes from reference year (original order).
    """
    country_lists = {}
    sector_lists = {}

    # Collect country (col 0) and sector (col 3) codes for each year
    for year, df in data_dict.items():
        if df is not None:
            country_lists[year] = df.iloc[:, 0].unique().tolist()
            sector_lists[year] = df.iloc[:, 3].unique().tolist()

    # Use first year as reference
    ref_year = list(country_lists.keys())[0]
    country_list = country_lists[ref_year]  # keep original order, this is very important for our analysis
    sector_list = sector_lists[ref_year]    # keep original order, this is very important for our analysis

    # Prepare sorted versions for consistency checks
    sorted_ref_countries = sorted(country_list)
    sorted_ref_sectors = sorted(sector_list)

    consistency_records = []
    for year, clist in country_lists.items():
        sorted_countries = sorted(clist)
        sorted_sectors = sorted(sector_lists[year])
        countries_ok = sorted_countries == sorted_ref_countries
        sectors_ok = sorted_sectors == sorted_ref_sectors
        consistency_records.append({
            'Year': year,
            'Country_Consistent': countries_ok,
            'Sector_Consistent': sectors_ok
        })

    consistency_df = pd.DataFrame(consistency_records)

    return consistency_df, country_list, sector_list



##############################################################
# This is the function to merge countries in IO Talbe (T) ###
##############################################################


def IO_country_merge_function(io_matrix, N, J, code_my_country, code_row):
    """
    Merge countries in IO matrix and expand ROW.

    Parameters:
    - io_matrix: Original IO matrix (NumPy array).
    - N: Number of countries excluding ROW.
    - J: Number of sectors excluding Total.
    - code_my_country: Indices of selected countries.
    - code_row: Indices of countries to merge into ROW.

    Returns:
    - Modified IO matrix with merged countries and expanded ROW.
    """

    # Step 1: Add ROW_new country rows and columns
    row_new = np.zeros((J, N * J + 1))
    io_matrix = np.vstack((io_matrix, row_new))
    
    col_new = np.zeros(((N + 1) * J + 1, J))
    io_matrix = np.hstack((io_matrix, col_new))
    
    # Step 2: Merge selected countries into ROW_new
    cols_to_sum = []
    rows_to_sum = []
    for c in code_row:
        start_col = (c - 1) * J
        cols_to_sum.extend(range(start_col, start_col + J))
        
        start_row = (c - 1) * J
        rows_to_sum.extend(range(start_row, start_row + J))
        
    # Sum columns to ROW_new
    for sector in range(J):
        accumulated_sum = np.sum(io_matrix[:, [(c - 1) * J + sector for c in code_row]], axis=1)
        io_matrix[:, N * J + sector] = accumulated_sum
    
    # Sum rows to ROW_new
    for sector in range(J):
        accumulated_sum = np.sum(io_matrix[[(c - 1) * J + sector for c in code_row], :], axis=0)
        io_matrix[N * J + sector, :] = accumulated_sum
    
    # Remove merged columns and rows
    io_matrix = np.delete(io_matrix, cols_to_sum, axis=1)
    io_matrix = np.delete(io_matrix, rows_to_sum, axis=0)
    
    # Step 3: Add ROW_final rows and columns
    row_new = np.zeros((J, io_matrix.shape[1]))
    io_matrix = np.vstack((io_matrix, row_new))
    
    col_new = np.zeros((io_matrix.shape[0], J))
    io_matrix = np.hstack((io_matrix, col_new))
    
    # Steps 4: Expand ROW_Total (omitted detailed comments for brevity)
    N_my_country = len(code_my_country)
    main = N_my_country * J
    reference_rows = np.arange(main + 1, main + J + 1)
    expand_target_row = main
    target_rows = np.arange(io_matrix.shape[0] - J, io_matrix.shape[0])

    # Step 4-1 to Step 4-5: Expansion based on proportions
    for idx in range(J):
        # ROW_total_row expansions
        ref_sum = io_matrix[reference_rows, idx].sum()
        io_matrix[target_rows, idx] = io_matrix[expand_target_row, idx] * (io_matrix[reference_rows, idx] / ref_sum)
        
        # ROW_total_col expansions
        ref_sum_col = io_matrix[idx, reference_rows].sum()
        io_matrix[idx, target_rows] = io_matrix[idx, expand_target_row] * (io_matrix[idx, reference_rows] / ref_sum_col)
    
    # Expand ROW_ROW_Total
    reference_matrix = io_matrix[np.ix_(reference_rows, reference_rows)]
    expand_target = io_matrix[expand_target_row, expand_target_row]
    total_sum = reference_matrix.sum()
    io_matrix[np.ix_(target_rows, target_rows)] = expand_target * (reference_matrix / total_sum)
    
    # Remove ROW_Total row and column
    io_matrix = np.delete(io_matrix, expand_target_row, axis=0)
    io_matrix = np.delete(io_matrix, expand_target_row, axis=1)
    
    # Step 5: Sum ROW_new and expanded ROW, then remove duplicates
    row_start1, row_end1 = main, main + J
    row_start2, row_end2 = main + J, main + 2 * J
    io_matrix[row_start1:row_end1, :] += io_matrix[row_start2:row_end2, :]
    col_start1, col_end1 = row_start1, row_end1
    col_start2, col_end2 = row_start2, row_end2
    io_matrix[:, col_start1:col_end1] += io_matrix[:, col_start2:col_end2]
    
    io_matrix = np.delete(io_matrix, np.s_[row_start2:row_end2], axis=0)
    io_matrix = np.delete(io_matrix, np.s_[col_start2:col_end2], axis=1)

    return io_matrix


##############################################################
# This is the function to merge sectors in IO Talbe (T) ######
##############################################################


def IO_sector_merge_function(io_matrix, N, J_old, sector_mapping):
    """
    Merge sectors according to the provided mapping rules.

    Parameters:
    - io_matrix: Original IO matrix (numpy array).
    - N: Number of countries (including ROW).
    - J_old: Original number of sectors.
    - sector_mapping: Dictionary mapping new sectors to old sector indices.

    Returns:
    - Merged IO matrix.
    """
    J_new = len(sector_mapping)
    IO_temp_row = np.zeros((N * J_new, N * J_old))

    for n in range(N):
        for new_sec, old_secs in sector_mapping.items():
            target_row = n * J_new + new_sec
            source_rows = [n * J_old + sec for sec in old_secs]
            IO_temp_row[target_row, :] = io_matrix[source_rows, :].sum(axis=0)

    IO_merged = np.zeros((N * J_new, N * J_new))

    for n in range(N):
        for new_sec, old_secs in sector_mapping.items():
            target_col = n * J_new + new_sec
            source_cols = [n * J_old + sec for sec in old_secs]
            IO_merged[:, target_col] = IO_temp_row[:, source_cols].sum(axis=1)

    return IO_merged

########################################################
# This is the function to remove sector for IO table ###
########################################################

def IO_sector_remove_function(io_matrix, N, J, sector_to_remove):
    """
    Remove specific sector from IO matrix for each country.

    Parameters:
    - io_matrix: IO matrix after merging sectors.
    - N: Number of countries including ROW.
    - J: Total sectors before removal.
    - sector_to_remove: Index of sector to remove (0-based indexing).

    Returns:
    - IO matrix after removal of the specified sector.
    """
    delete_indices = [n * J + sector_to_remove for n in range(N)]
    io_matrix_reduced = np.delete(io_matrix, delete_indices, axis=0)
    io_matrix_reduced = np.delete(io_matrix_reduced, delete_indices, axis=1)

    return io_matrix_reduced



##############################################################
# This is the function to merge countries in FD Talbe (FD) ###
##############################################################

def FD_country_merge_function(fd_matrix, N, J, FD_num, code_my_country, code_row, index_ROW):
    """
    Merge countries in FD matrix and expand ROW final demand.

    Parameters:
    - fd_matrix: original FD matrix (numpy array)
    - N: number of countries excluding ROW
    - J: number of sectors excluding Total
    - FD_num: number of FD accounts (e.g. 6)
    - code_my_country: list of 1-based indices for selected countries
    - code_row: list of 1-based indices for countries to be merged into ROW
    - index_ROW: 1-based index of ROW in original country list

    Returns:
    - transformed FD matrix (numpy array)
    """
    fd_matrix = fd_matrix.copy()

    # Step 1: add ROW_new (to bottom and right)
    row_new = np.zeros((J, fd_matrix.shape[1]))
    fd_matrix = np.vstack([fd_matrix, row_new])

    col_new = np.zeros((fd_matrix.shape[0], FD_num))
    fd_matrix = np.hstack([fd_matrix, col_new])

    # Step 2: merge countries in code_row to ROW_new (bottom J rows)
    for sector in range(J):
        rows = [(c - 1) * J + sector for c in code_row]
        fd_matrix[N * J + sector, :] = fd_matrix[rows, :].sum(axis=0)

    # Merge FD columns for code_row + ROW into new FD column
    for fd_col in range(FD_num):
        cols = [(c - 1) * FD_num + fd_col for c in code_row + [index_ROW]]
        fd_matrix[:, (N + 1) * FD_num + fd_col] = fd_matrix[:, cols].sum(axis=1)

    # Step 2.5: delete rows/cols for code_row and ROW
    rows_to_delete = []
    for c in code_row:
        rows_to_delete.extend(range((c - 1) * J, c * J))
    fd_matrix = np.delete(fd_matrix, rows_to_delete, axis=0)

    cols_to_delete = []
    for c in code_row + [index_ROW]:
        cols_to_delete.extend(range((c - 1) * FD_num, c * FD_num))
    fd_matrix = np.delete(fd_matrix, cols_to_delete, axis=1)

    # Step 3: add ROW_expanded to the bottom
    row_new = np.zeros((J, fd_matrix.shape[1]))
    fd_matrix = np.vstack([fd_matrix, row_new])

    # Step 4: expand ROW_Total to ROW sectors
    N_my_country = len(code_my_country)
    main = N_my_country * J
    reference_rows = np.arange(main + 1, main + J + 1)  # ref: sector rows of ROW_new
    expand_target_row = main                           # the total row of ROW
    target_rows = np.arange(fd_matrix.shape[0] - J, fd_matrix.shape[0])

    for col in range(fd_matrix.shape[1]):
        reference_values = fd_matrix[reference_rows, col]
        reference_sum = reference_values.sum()
        if reference_sum != 0:
            fd_matrix[target_rows, col] = (
                fd_matrix[expand_target_row, col] * (reference_values / reference_sum)
            )

    # Delete ROW_Total row
    fd_matrix = np.delete(fd_matrix, expand_target_row, axis=0)

    # Step 5: Sum ROW_new + ROW_expanded, then delete ROW_expanded
    row_start1 = main
    row_end1 = main + J
    row_start2 = main + J
    row_end2 = main + 2 * J

    fd_matrix[row_start1:row_end1, :] += fd_matrix[row_start2:row_end2, :]
    fd_matrix = np.delete(fd_matrix, np.s_[row_start2:row_end2], axis=0)

    return fd_matrix



##############################################################
# This is the function to merge sectors in FD Talbe (FD) #####
##############################################################

def FD_sector_merge_function(fd_matrix, N, J_old, sector_mapping, FD_num):
    """
    Merge sectors in FD matrix for each country block.

    Parameters:
    - fd_matrix: FD matrix after country merge/expansion (np.ndarray), shape = (N * J_old, N * FD_num)
    - N: number of countries (including ROW)
    - J_old: original number of sectors (before merging)
    - sector_mapping: dictionary defining new sector groups (e.g. 24 to 23)
    - FD_num: number of final demand accounts per country

    Returns:
    - Merged FD matrix, shape = (N * J_new, N * FD_num)
    """
    J_new = len(sector_mapping)
    merged_fd = np.zeros((N * J_new, N * FD_num))

    for n in range(N):
        for new_sec, old_secs in sector_mapping.items():
            target_row = n * J_new + new_sec
            source_rows = [n * J_old + s for s in old_secs]
            merged_fd[target_row, :] = fd_matrix[source_rows, :].sum(axis=0)

    return merged_fd



########################################################
# This is the function to remove sector for FD table ###
########################################################
def FD_sector_remove_function(fd_matrix, N, J, sector_to_remove):
    """
    Remove sector from FD matrix (by row).

    Parameters:
    - fd_matrix: merged FD matrix, shape = (N * J, N * FD_num)
    - N: number of countries
    - J: number of sectors (including the one to remove)
    - sector_to_remove: 0-based index of the sector to remove

    Returns:
    - Reduced FD matrix with specified sector removed
    """
    delete_indices = [n * J + sector_to_remove for n in range(N)]
    return np.delete(fd_matrix, delete_indices, axis=0)



##############################################################
# This is the function to merge countries in VA Talbe (VA) ###
##############################################################


def VA_country_merge_function(va_matrix, N, J, VA_num, code_my_country, code_row):
    """
    Merge countries and expand ROW in the Value Added (VA) matrix (column-wise structure).

    Parameters:
    - va_matrix: original VA matrix, shape = (VA_num, N*J)
    - N: number of countries (excluding ROW)
    - J: number of sectors (excluding Total)
    - VA_num: number of value-added accounts (e.g., 6)
    - code_my_country: 1-based list of kept countries
    - code_row: 1-based list of countries to merge into ROW

    Returns:
    - merged VA matrix (numpy array), shape = (VA_num, (N_final * J))
    """
    va_matrix = va_matrix.copy()

    # Step 1: Add ROW_new columns at end
    col_new = np.zeros((va_matrix.shape[0], J))
    va_matrix = np.hstack((va_matrix, col_new))

    # Step 2: Merge all countries in code_row into ROW_new
    for sector in range(J):
        cols = [(c - 1) * J + sector for c in code_row]
        va_matrix[:, N * J + sector] = va_matrix[:, cols].sum(axis=1)

    # Delete the merged columns
    cols_to_delete = []
    for c in code_row:
        cols_to_delete.extend(range((c - 1) * J, c * J))
    va_matrix = np.delete(va_matrix, cols_to_delete, axis=1)

    # Step 3: Add ROW_expanded columns at end
    col_new = np.zeros((va_matrix.shape[0], J))
    va_matrix = np.hstack((va_matrix, col_new))

    # Expand ROW_total to ROW sectors
    N_my_country = len(code_my_country)
    main = N_my_country * J

    reference_cols = np.arange(main + 1, main + J + 1)     # original ROW_new sectors
    expand_target_col = main                               # ROW_total
    target_cols = np.arange(va_matrix.shape[1] - J, va_matrix.shape[1])

    for row in range(VA_num):
        reference_values = va_matrix[row, reference_cols]
        reference_sum = reference_values.sum()
        if reference_sum != 0:
            va_matrix[row, target_cols] = (
                va_matrix[row, expand_target_col] * (reference_values / reference_sum)
            )

    # Delete ROW_total column (expand_target_col)
    va_matrix = np.delete(va_matrix, expand_target_col, axis=1)

    # Step 4: Add ROW_new + ROW_expanded
    col_start1 = main       # after delete, this points to ROW_new
    col_end1 = col_start1 + J
    col_start2 = col_end1   # ROW_expanded
    col_end2 = col_start2 + J

    va_matrix[:, col_start1:col_end1] += va_matrix[:, col_start2:col_end2]
    va_matrix = np.delete(va_matrix, np.s_[col_start2:col_end2], axis=1)

    return va_matrix


##############################################################
# This is the function to merge sectors in VA Talbe (VA) ####
##############################################################

def VA_sector_merge_function(va_matrix, N, J_old, sector_mapping, VA_num):
    """
    Merge sectors in VA matrix (column-wise structure).

    Parameters:
    - va_matrix: VA matrix, shape = (VA_num, N * J_old)
    - N: number of countries
    - J_old: original number of sectors
    - sector_mapping: dict of new_sector -> list of old_sectors
    - VA_num: number of VA accounts

    Returns:
    - Merged VA matrix, shape = (VA_num, N * J_new)
    """
    J_new = len(sector_mapping)
    merged_va = np.zeros((VA_num, N * J_new))

    for n in range(N):
        for new_sec, old_secs in sector_mapping.items():
            target_col = n * J_new + new_sec
            source_cols = [n * J_old + s for s in old_secs]
            merged_va[:, target_col] = va_matrix[:, source_cols].sum(axis=1)

    return merged_va


########################################################
# This is the function to remove sector for VA table ###
########################################################


def VA_sector_remove_function(va_matrix, N, J, sector_to_remove):
    """
    Remove one sector (e.g. 'Re-export and Re-import') from VA matrix.

    Parameters:
    - va_matrix: shape = (VA_num, N * J)
    - N: number of countries
    - J: number of sectors
    - sector_to_remove: 0-based index of sector to remove

    Returns:
    - Reduced VA matrix
    """
    delete_indices = [n * J + sector_to_remove for n in range(N)]
    return np.delete(va_matrix, delete_indices, axis=1)


############################################################################################################### 
################################# These are functions using in 2_Calibrate_Paramteres.ipynb #####################
############################################################################################################### 


# Construct to long table
def save_trade_share_long(pi_matrix,
                        country_list,
                        sector_list,
                        save_path,
                        filename):
    """
    Reshape (N*J, N) pi matrix to long table and save to csv

    Parameters
    ----------
    pi_matrix    : np.ndarray  shape (N*J, N)
        row = Exporter * Sector (i,k); column = Import n; element = π_{ni}^k
    country_list : list[str]    N
    sector_list  : list[str]   J
    save_path    : str         
    filename     : str        

    Returns
    -------
    long_df : pandas.DataFrame (Importer, Exporter, Sector, Share)
    """
    N, J = len(country_list), len(sector_list)

    # 1) label of row
    exporter_col = np.repeat(country_list, J)   # N*J
    sector_col   = np.tile(sector_list,   N)    # N*J

    # 2) DataFrame
    pi_df = pd.DataFrame(pi_matrix, columns=country_list)
    pi_df.insert(0, "Sector",   sector_col)
    pi_df.insert(0, "Exporter", exporter_col)

    # 3) melt to long table
    long_df = (
        pi_df.melt(id_vars=["Exporter", "Sector"],
                var_name="Importer",
                value_name="Share")
            .loc[:, ["Importer", "Exporter", "Sector", "Share"]]
    )

    # 4) save
    os.makedirs(save_path, exist_ok=True)
    out_file = os.path.join(save_path, filename)
    long_df.to_csv(out_file, index=False)
    print(f"Saved trade-share file: {out_file}")

    return long_df


############################################################################################################### 
################################# These are functions using in 3_Calculate_TradeCost.ipynb #####################
############################################################################################################### 
def cal_trade_cost(pi_df: pd.DataFrame,
                tariff_df: pd.DataFrame,
                theta_map: dict[str, float],
                tariff_col:str = "tariff") -> pd.DataFrame:
    """
    Compute ln d_ni^{su} for all (Importer, Exporter, Sector) pairs in pi_df.
    The tariff column used is ‘tariff’; change the name if you need MFN, etc.
    The result has unique columns only: Importer, Exporter, Sector, ln_d.
    """

    # ------------------------------------------------------------------
    # 1. π_ni  (rename for clarity)
    # ------------------------------------------------------------------
    pi = (pi_df
        .rename(columns={"Importer": "imp",
                        "Exporter": "exp",
                        "Share":    "pi_ni"}))

    # ------------------------------------------------------------------
    # 2. π_in  (reverse direction once, then merge)
    # ------------------------------------------------------------------
    pi_rev = (pi[["imp", "exp", "Sector", "pi_ni"]]
                .rename(columns={"imp": "exp",
                                "exp": "imp",
                                "pi_ni": "pi_in"}))

    df = pi.merge(pi_rev, on=["imp", "exp", "Sector"], how="left")

    # ------------------------------------------------------------------
    # 3. π_nn  &  π_ii  (self-trade shares)
    # ------------------------------------------------------------------
    self_sh = (pi_df.loc[pi_df["Importer"] == pi_df["Exporter"],
                        ["Importer", "Sector", "Share"]]
            .rename(columns={"Importer": "cty",
                                "Share":    "pi_self"}))

    # π_nn
    df = (df
        .merge(self_sh,
                left_on=["imp", "Sector"],
                right_on=["cty", "Sector"],
                how="left")
        .rename(columns={"pi_self": "pi_nn"})
        .drop(columns="cty"))

    # π_ii
    df = (df
        .merge(self_sh,
                left_on=["exp", "Sector"],
                right_on=["cty", "Sector"],
                how="left",
                suffixes=("", "_dup"))          # avoid name clash
        .rename(columns={"pi_self": "pi_ii"})
        #.drop(columns=["cty", "pi_self_dup"])
        )

    # ------------------------------------------------------------------
    # 4. τ_ni  &  τ_in  — build once, merge once 
    # ------------------------------------------------------------------
    t = tariff_df[["Importer", "Exporter", "Sector", tariff_col]].copy()

    # stack ni / in, then pivot wider
    t_long = pd.concat(
        [
            t.assign(direction="tau_ni"),                          # original
            t.rename(columns={"Importer": "Exporter",
                            "Exporter": "Importer"})
            .assign(direction="tau_in")                           # reversed
        ],
        ignore_index=True
    )

    tau = (t_long
        .pivot_table(index=["Importer", "Exporter", "Sector"],
                        columns="direction", values="tariff")
        .reset_index())

    # merge once → both columns arrive together
    df = (df.merge(tau.rename(columns={"Importer": "imp",
                                    "Exporter": "exp"}),
                on=["imp", "exp", "Sector"],
                how="left"))

    # ------------------------------------------------------------------
    # 5. map θ and calculate ln d
    # ------------------------------------------------------------------
    df["theta"] = df["Sector"].map(theta_map)

    df["ln_d"] = (
        0.5 * np.log((1.0 + df["tau_ni"]) / (1.0 + df["tau_in"]))
        + 0.5 / df["theta"]
          * np.log((df["pi_nn"] * df["pi_ii"]) / (df["pi_ni"] * df["pi_in"]))
    )

    df["d"] = np.exp(df["ln_d"])

    # ------------------------------------------------------------------
    # 6. tidy return
    # ------------------------------------------------------------------
    return df.rename(columns={"imp": "Importer", "exp": "Exporter"})
    # return (df[["imp", "exp", "Sector", "ln_d",]]
    #         .rename(columns={"imp": "Importer",
    #                         "exp": "Exporter"}))




# To show all avaliable functions in this file

if __name__ == "__main__":
    import inspect
    import sys  
    functions = [name for name, obj in inspect.getmembers(sys.modules[__name__]) 
                if inspect.isfunction(obj)]
    print("Available functions:\n- " + "\n- ".join(functions))