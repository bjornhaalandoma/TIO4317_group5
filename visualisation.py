import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Define directory and file names for daily data
data_dir = "financial_data"
file_names = {
    "Brent": "Brent_Crude.csv",
    "NaturalGas": "Natural_Gas.csv",
    "NOKUSD": "NOK_USD.csv",
    "OSEBX": "OSEBX.csv",
    "AkerBP": "Aker_BP.csv",
    "USTreasury": "US_10Yr_Treasury.csv"
}

# Load monthly datasets (CPI and Policy Rate)
cpi_df = pd.read_csv(os.path.join(data_dir, "CPI_rate.csv"), parse_dates=["Date"])
policy_df = pd.read_csv(os.path.join(data_dir, "Norwegian_Policy_Rate.csv"), parse_dates=["Date"])

# Sort monthly data
cpi_df.sort_values("Date", inplace=True)
policy_df.sort_values("Date", inplace=True)

# Load the daily data into a dictionary of DataFrames
data_dict = {}
for key, file in file_names.items():
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    data_dict[key] = df


# Visualisation of the data
# Function to plot a time series
def plot_series(df, date_col, value_col, title, ylabel):
    plt.figure(figsize=(10, 4))
    plt.plot(df[date_col], df[value_col], label=value_col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Visualize daily data: Plot both Close and Log Returns for each asset
for key, df in data_dict.items():
    plot_series(df, "Date", "Close", f"{key} Close Price", "Price")
    plot_series(df, "Date", "Log_Returns", f"{key} Log Returns", "Log Returns")

# Visualize monthly data: CPI and Policy Rate (raw)
plot_series(cpi_df, "Date", "Rate", "CPI Rate Over Time", "CPI Rate (%)")
plot_series(policy_df, "Date", "Rate", "Norwegian Policy Rate Over Time", "Policy Rate (%)")


def adf_test(series, title=''):
    result = adfuller(series.dropna())
    print(f"ADF Test for {title}")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print("=> Likely stationary.\n")
    else:
        print("=> Likely non-stationary.\n")

# Daily Data: Test stationarity for both Close prices and Log Returns
for key, df in data_dict.items():
    adf_test(df["Close"], title=f"{key} Close")
    adf_test(df["Log_Returns"], title=f"{key} Log Returns")

# Monthly Data: Test stationarity for the raw Rate series
adf_test(cpi_df["Rate"], title="CPI Rate")
adf_test(policy_df["Rate"], title="Norwegian Policy Rate")

# Since CPI and Policy Rate are non-stationary, we difference them.
cpi_df["Diff"] = cpi_df["Rate"].diff()
policy_df["Diff"] = policy_df["Rate"].diff()

# Test stationarity again on the differenced series
adf_test(cpi_df["Diff"], title="CPI Rate - 1st Difference")
adf_test(policy_df["Diff"], title="Policy Rate - 1st Difference")

# Visualize the differenced (stationary) monthly data
plot_series(cpi_df, "Date", "Diff", "CPI Rate - 1st Difference", "Diff of CPI Rate")
plot_series(policy_df, "Date", "Diff", "Norwegian Policy Rate - 1st Difference", "Diff of Policy Rate")

# For clarity in later merging, rename the differenced columns as the stationary versions
cpi_df.rename(columns={"Diff": "CPI_Stationary"}, inplace=True)
policy_df.rename(columns={"Diff": "PolicyRate_Stationary"}, inplace=True)


# Merge Daily and Monthly Data for Regression
# Use Aker BP's log returns as the base daily series.
df_reg = data_dict["AkerBP"][["Date", "Log_Returns"]].rename(
    columns={"Log_Returns": "AkerBP_Log_Returns"}
)

# Merge daily exogenous variables (log returns) for the assets
exog_keys = ["Brent", "NaturalGas", "NOKUSD", "OSEBX", "USTreasury"]
for key in exog_keys:
    temp_df = data_dict[key][["Date", "Log_Returns"]].rename(
        columns={"Log_Returns": f"{key}_Log_Returns"}
    )
    df_reg = pd.merge(df_reg, temp_df, on="Date", how="inner")

# Before merging, ensure that daily data and monthly data are sorted by Date.
df_reg.sort_values("Date", inplace=True)
cpi_df.sort_values("Date", inplace=True)
policy_df.sort_values("Date", inplace=True)

# Merge monthly data with daily data using merge_asof.
df_reg = pd.merge_asof(df_reg, cpi_df[["Date", "CPI_Stationary"]], on="Date", direction="backward")
df_reg = pd.merge_asof(df_reg, policy_df[["Date", "PolicyRate_Stationary"]], on="Date", direction="backward")

# Drop any rows with missing values (e.g., before the first available monthly observation)
df_reg.dropna(inplace=True)

# Regression Analysis
# Define dependent and independent variables
y = df_reg["AkerBP_Log_Returns"]
X = df_reg[["Brent_Log_Returns", "NaturalGas_Log_Returns", "NOKUSD_Log_Returns",
            "OSEBX_Log_Returns", "USTreasury_Log_Returns", "CPI_Stationary", "PolicyRate_Stationary"]]

# Add constant
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()
print(model.summary())
