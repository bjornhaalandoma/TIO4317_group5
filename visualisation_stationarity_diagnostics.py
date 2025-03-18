import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------
# 1. Load the Data
# -------------------------------

# Define directory and file names for daily data
data_dir = "financial_data"
file_names = {
    "Brent": "Brent_Crude.csv",
    "NaturalGas": "Natural_Gas.csv",
    "NOKUSD": "NOK_USD.csv",
    "OSEBX": "OSEBX.csv",
    "VarEnergi": "Var_Energi.csv",
    "USTreasury": "US_10Yr_Treasury.csv"
}

# Load monthly datasets (CPI and Policy Rate)
cpi_df = pd.read_csv(os.path.join(data_dir, "CPI_rate.csv"), parse_dates=["Month"])
policy_df = pd.read_csv(os.path.join(data_dir, "Norwegian_Policy_Rate.csv"), parse_dates=["Month"])

# Sort monthly data
cpi_df.sort_values("Month", inplace=True)
policy_df.sort_values("Month", inplace=True)

# Load the daily data into a dictionary of DataFrames
data_dict = {}
for key, file in file_names.items():
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    data_dict[key] = df

# -------------------------------
# 2. Visualize the Data
# -------------------------------

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
plot_series(cpi_df, "Month", "Rate", "CPI Rate Over Time", "CPI Rate (%)")
plot_series(policy_df, "Month", "Rate", "Norwegian Policy Rate Over Time", "Policy Rate (%)")

# -------------------------------
# 3. Check for Stationarity with ADF Test
# -------------------------------

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
plot_series(cpi_df, "Month", "Diff", "CPI Rate - 1st Difference", "Diff of CPI Rate")
plot_series(policy_df, "Month", "Diff", "Norwegian Policy Rate - 1st Difference", "Diff of Policy Rate")

# For clarity in later merging, rename the differenced columns as the stationary versions
cpi_df.rename(columns={"Month": "Date", "Diff": "CPI_Stationary"}, inplace=True)
policy_df.rename(columns={"Month": "Date", "Diff": "PolicyRate_Stationary"}, inplace=True)

# -------------------------------
# 4. Merge Daily and Monthly Data for Regression
# -------------------------------

# Use Vår Energi's log returns as the base daily series.
df_reg = data_dict["VarEnergi"][["Date", "Log_Returns"]].rename(
    columns={"Log_Returns": "VarEnergi_Log_Returns"}
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

# -------------------------------
# 5. Regression Analysis & Diagnostics
# -------------------------------

# Define dependent and independent variables
y = df_reg["VarEnergi_Log_Returns"]
X = df_reg[["Brent_Log_Returns", "NaturalGas_Log_Returns", "NOKUSD_Log_Returns",
            "OSEBX_Log_Returns", "USTreasury_Log_Returns", "CPI_Stationary", "PolicyRate_Stationary"]]

# Add constant
X = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X).fit()
print(model.summary())

# Diagnostics:
# Heteroskedasticity: Breusch–Pagan test
bp_test = het_breuschpagan(model.resid, model.model.exog)
print("Breusch–Pagan test p-value:", bp_test[1])
# Interpretation: p < 0.05 suggests heteroskedasticity, in which case consider robust errors.

# Autocorrelation: Durbin–Watson test
dw_stat = durbin_watson(model.resid)
print("Durbin–Watson statistic:", dw_stat)
# Interpretation: A value around 2 is ideal. Values far from 2 indicate autocorrelation.

# Multicollinearity: VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
# Interpretation: VIF values above 5 (or 10) suggest high multicollinearity.

# ======================
# 6. Next Steps / Modifications if Assumptions are Violated
# ======================
# - If the BP test indicates heteroskedasticity (p-value < 0.05), use robust standard errors:
#       model_robust = model.get_robustcov_results(cov_type='HC3')
# - If the Durbin–Watson statistic suggests autocorrelation, consider adding lagged variables
#   or switching to an ARIMAX framework.
# - If VIF values are high, check for highly correlated predictors and consider removing or combining themimport os