import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera

# Function to perform ADF test and check for stationarity
def check_stationarity(series, name):
    adf_result = adfuller(series.dropna())
    print(f"ADF Test for {name}:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f'   {key}: {value}')
    if adf_result[1] < 0.05:
        print(f"{name} is stationary (p-value < 0.05)\n")
    else:
        print(f"{name} is not stationary (p-value >= 0.05)\n")

# Load the CSV files into DataFrames
df_aker_bp = pd.read_csv("financial_data/Aker_BP_weekly.csv")
df_brent_crude = pd.read_csv("financial_data/Brent_Crude_weekly.csv")
df_cpi_rate = pd.read_csv("financial_data/CPI_Rate_weekly.csv")
df_natural_gas = pd.read_csv("financial_data/Natural_Gas_Weekly.csv")
df_nok_usd = pd.read_csv("financial_data/NOK_USD_weekly.csv")
df_norwegian_policy_rate = pd.read_csv("financial_data/Norwegian_Policy_Rate_weekly.csv")
df_osebx = pd.read_csv("financial_data/OSEBX_weekly.csv")
df_us_treasury = pd.read_csv("financial_data/US_10Yr_Treasury_weekly.csv")

# Rename the conflicting columns
df_aker_bp = df_aker_bp.rename(columns={'Log_Returns': 'Log_Returns_Aker_BP'})
df_brent_crude = df_brent_crude.rename(columns={'Log_Returns': 'Log_Returns_Brent_Crude'})
df_natural_gas = df_natural_gas.rename(columns={'Log_Returns': 'Log_Returns_Natural_Gas'})
df_nok_usd = df_nok_usd.rename(columns={'Log_Returns': 'Log_Returns_NOK_USD'})
df_osebx = df_osebx.rename(columns={'Log_Returns': 'Log_Returns_OSEBX'})
df_us_treasury = df_us_treasury.rename(columns={'Log_Returns': 'Log_Returns_US_Treasury'})
df_cpi_rate = df_cpi_rate.rename(columns={'Rate': 'CPI_Rate'})
df_norwegian_policy_rate = df_norwegian_policy_rate.rename(columns={'Rate': 'Policy_Rate'})

# Merge all the dataframes on 'Date' (assuming the 'Date' column exists in each file)
df = df_aker_bp[['Date', 'Log_Returns_Aker_BP']]  # Assuming 'Log_Returns' is the column name for the stock return
df = pd.merge(df, df_brent_crude[['Date', 'Log_Returns_Brent_Crude']], on='Date', how='inner')
df = pd.merge(df, df_cpi_rate[['Date', 'CPI_Rate']], on='Date', how='inner')
df = pd.merge(df, df_natural_gas[['Date', 'Log_Returns_Natural_Gas']], on='Date', how='inner')
df = pd.merge(df, df_nok_usd[['Date', 'Log_Returns_NOK_USD']], on='Date', how='inner')
df = pd.merge(df, df_norwegian_policy_rate[['Date', 'Policy_Rate']], on='Date', how='inner')
df = pd.merge(df, df_osebx[['Date', 'Log_Returns_OSEBX']], on='Date', how='inner')
df = pd.merge(df, df_us_treasury[['Date', 'Log_Returns_US_Treasury']], on='Date', how='inner')

# Drop any rows with missing values
df = df.dropna()

# Check stationarity for each of the key variables
check_stationarity(df['Log_Returns_Aker_BP'], 'Log_Returns_Aker_BP')
check_stationarity(df['Log_Returns_Brent_Crude'], 'Log_Returns_Brent_Crude')
check_stationarity(df['Log_Returns_Natural_Gas'], 'Log_Returns_Natural_Gas')
check_stationarity(df['Log_Returns_NOK_USD'], 'Log_Returns_NOK_USD')
check_stationarity(df['CPI_Rate'], 'CPI_Rate')
check_stationarity(df['Policy_Rate'], 'Policy_Rate')
check_stationarity(df['Log_Returns_OSEBX'], 'Log_Returns_OSEBX')
check_stationarity(df['Log_Returns_US_Treasury'], 'Log_Returns_US_Treasury')

# Differencing the Policy Rate and CPI Rate to make them stationary
df['CPI_Rate_diff'] = df['CPI_Rate'].diff()
df['Policy_Rate_diff'] = df['Policy_Rate'].diff()

# Drop any rows with missing values after differencing
df = df.dropna()
    
df = df.fillna(0)  # Replace NaN with 0
df.replace([float('inf'), -float('inf')], 1e10, inplace=True)  # Replace inf with 1e10

# Check for any remaining NaNs or Infs
if df.isnull().sum().any() or (df == float('inf')).sum().any():
    print("Warning: The data still contains missing or infinite values.")
else:
    print("No missing or infinite values found in the dataset.")

# Re-check stationarity for the differenced variables (if necessary)
check_stationarity(df['CPI_Rate_diff'], 'CPI_Rate_diff')
check_stationarity(df['Policy_Rate_diff'], 'Policy_Rate_diff')

# Define the dependent variable (log returns of Vår Energi stock)
y = df['Log_Returns_Aker_BP']

# Define the independent variables (macroeconomic factors)
X = df[['Log_Returns_Brent_Crude', 'Log_Returns_Natural_Gas', 'Log_Returns_NOK_USD', 
        'CPI_Rate_diff', 'Policy_Rate_diff', 'Log_Returns_OSEBX', 'Log_Returns_US_Treasury']]

# Add a constant to the independent variables matrix (for the intercept in the regression)
X = sm.add_constant(X)

# Check for any NaN or Inf values in X before running the regression
if X.isnull().sum().any() or (X == float('inf')).sum().any():
    print("Warning: The X matrix contains NaN or infinite values. Cleaning data...")
    X = X.fillna(0)  # Replace NaN with 0 in the independent variables matrix
    X.replace([float('inf'), -float('inf')], 1e10, inplace=True)  # Replace inf with 1e10

# Run the regression
model = sm.OLS(y, X).fit()

# Print the regression results
print(model.summary())

# Check for endogeneity: Calculate the VIF for each feature. No Perfect Collinearity
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)

# check for residulas being normally distributed by using Jarque-Bera test ut ~ N(0, σ²)
residuals = model.resid

# Jarque-Bera test
jb_stat, jb_p_value = jarque_bera(residuals)
print(f"Jarque-Bera Test: Statistic = {jb_stat}, p-value = {jb_p_value}")

if jb_p_value > 0.05:
    print("Residuals appear to be normally distributed (fail to reject H0).")
else:
    print("Residuals do not appear to be normally distributed (reject H0).")

# Plot the residuals to check for normality and heteroscedasticity. 
# Check for linearity
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, color="b")
plt.title('Residuals vs Fitted')
plt.show()
