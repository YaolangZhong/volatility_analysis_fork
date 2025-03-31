import os
import numpy as np
import pandas as pd
from models import ModelParams, ModelShocks, ModelSol
from equations import *

"""
Summary of the Data Generating Process

We consider an international trade model with N countries and J sectors. In this simulation, we study three types of productivity shocks:
	•	Country-specific shocks:
Each country receives one independent productivity shock (drawn from a log-normal distribution with mean 0 and standard deviation 0.2), and this shock is assumed to affect all sectors in that country equally.
	•	Sector-specific shocks:
Each sector receives one independent productivity shock (drawn from the same log-normal distribution) that is common to all countries.
	•	Idiosyncratic shocks:
Every country-sector pair receives its own independent productivity shock.

For each shock type, we generate B=10 simulations. For each simulation, we compute the global value chain effect for each country as the relative change in the real wage when intermediate goods trade barriers increase (comparing the case dm=2 versus dm=1). Finally, for each shock type, we summarize the results by computing the mean and standard deviation (across the 10 simulations) of the global value chain effect for each country. These summary statistics are stored in arrays of shape (N,).

The results are then compiled into a Pandas DataFrame with one row per country (using a provided country name list) and columns for:
	•	Country-specific shock effect: mean and standard deviation,
	•	Sector-specific shock effect: mean and standard deviation,
	•	Idiosyncratic shock effect: mean and standard deviation,

as well as overall averages across countries for each measure."
"""



# 1. Read the data of the ModelParams
mp = ModelParams.load_from_npz(f"output/params.npz")
data = np.load("real_data_2017.npz")

# 2. Reconstruct the ModelShocks and ModelSol from the result
out_dir = "output/counterfactual"

country_dm_1_shocks = []
country_dm_1_sols = []
country_dm_2_shocks = []
country_dm_2_sols = []
for i in range(9):
    country_dm_1_shocks.append(ModelShocks.load_from_npz(os.path.join(out_dir, f"country_dm_1", f"result_{i}_shock.npz"), mp))
    country_dm_1_sols.append(ModelSol.load_from_npz(os.path.join(out_dir, f"country_dm_1", f"result_{i}_sol.npz"), mp, country_dm_1_shocks[i]))
    country_dm_2_shocks.append(ModelShocks.load_from_npz(os.path.join(out_dir, f"country_dm_2", f"result_{i}_shock.npz"), mp))
    country_dm_2_sols.append(ModelSol.load_from_npz(os.path.join(out_dir, f"country_dm_2", f"result_{i}_sol.npz"), mp, country_dm_1_shocks[i]))

sector_dm_1_shocks = []
sector_dm_1_sols = []
sector_dm_2_shocks = []
sector_dm_2_sols = []
for i in range(9):
    sector_dm_1_shocks.append(ModelShocks.load_from_npz(os.path.join(out_dir, f"sector_dm_1", f"result_{i}_shock.npz"), mp))
    sector_dm_1_sols.append(ModelSol.load_from_npz(os.path.join(out_dir, f"sector_dm_1", f"result_{i}_sol.npz"), mp, sector_dm_1_shocks[i]))
    sector_dm_2_shocks.append(ModelShocks.load_from_npz(os.path.join(out_dir, f"sector_dm_2", f"result_{i}_shock.npz"), mp))
    sector_dm_2_sols.append(ModelSol.load_from_npz(os.path.join(out_dir, f"sector_dm_2", f"result_{i}_sol.npz"), mp, sector_dm_1_shocks[i]))

idiosyncratic_dm_1_shocks = []
idiosyncratic_dm_1_sols = []
idiosyncratic_dm_2_shocks = []
idiosyncratic_dm_2_sols = []
for i in range(9):
    idiosyncratic_dm_1_shocks.append(ModelShocks.load_from_npz(os.path.join(out_dir, f"idiosyncratic_dm_1", f"result_{i}_shock.npz"), mp))
    idiosyncratic_dm_1_sols.append(ModelSol.load_from_npz(os.path.join(out_dir, f"idiosyncratic_dm_1", f"result_{i}_sol.npz"), mp, idiosyncratic_dm_1_shocks[i]))
    idiosyncratic_dm_2_shocks.append(ModelShocks.load_from_npz(os.path.join(out_dir, f"idiosyncratic_dm_2", f"result_{i}_shock.npz"), mp))
    idiosyncratic_dm_2_sols.append(ModelSol.load_from_npz(os.path.join(out_dir, f"idiosyncratic_dm_2", f"result_{i}_sol.npz"), mp, idiosyncratic_dm_1_shocks[i]))

# 3. Calculate the treatment effects under different types of shocks
N = mp.N
country_effects = np.zeros((10, N))
sector_effects = np.zeros((10, N))
idiosyncratic_effects = np.zeros((10, N))

for i in range(9):
    country_effects[i, :] = calc_W(country_dm_2_sols[i]) / calc_W(country_dm_1_sols[i])
    sector_effects[i, :] = calc_W(sector_dm_2_sols[i]) / calc_W(sector_dm_1_sols[i])
    idiosyncratic_effects[i, :] = calc_W(idiosyncratic_dm_2_sols[i]) / calc_W(idiosyncratic_dm_1_sols[i])


mean_country_effects = np.mean(country_effects, axis=0)
std_country_effects = np.std(country_effects, axis=0)

print(
    f"mean of the country effect:\n{mean_country_effects}\n"
    f"std of the country effect:\n{std_country_effects}"
)

mean_sector_effects = np.mean(sector_effects, axis=0)
std_sector_effects = np.std(sector_effects, axis=0)

print(
    f"mean of the sector effect:\n{mean_sector_effects}\n"
    f"std of the sector effect:\n{std_sector_effects}"
)

mean_idiosyncratic_effects = np.mean(idiosyncratic_effects, axis=0)
std_idiosyncratic_effects = np.std(idiosyncratic_effects, axis=0)

print(
    f"mean of the idiosyncratic effect:\n{mean_idiosyncratic_effects}\n"
    f"std of the idiosyncratic effect:\n{std_idiosyncratic_effects}"
)

country_list = data["country_list"]


# Create a DataFrame with one row per country and columns for each shock type summary
df = pd.DataFrame({
    "Country-Specific Mean": mean_country_effects,
    "Country-Specific Std": std_country_effects,
    "Sector-Specific Mean": mean_sector_effects,
    "Sector-Specific Std": std_sector_effects,
    "Idiosyncratic Mean": mean_idiosyncratic_effects,
    "Idiosyncratic Std": std_idiosyncratic_effects
}, index=country_list)

# Compute overall averages across countries for each summary statistic:
overall_means = df.mean(axis=0)
overall_stds = df.std(axis=0)

# Append overall average rows to the DataFrame.
# One row for the mean over all countries, and one for the standard deviation.
df.loc["Overall Mean"] = overall_means
df.loc["Overall Std"] = overall_stds

# Display the DataFrame
print(df)

# Output the DataFrame to an Excel file.
output_filename = "global_value_chain_effects.xlsx"
df.to_excel(output_filename)