import os
import pandas as pd
import numpy as np

# Lists of input CSV filenames and corresponding output CSV filenames
input_files = ['Aker_BP.csv', 'Brent_Crude.csv', 'Natural_Gas.csv', 'NOK_USD.csv', 'OSEBX.csv', 'US_10Yr_Treasury.csv']  # Replace with your list of input files
output_files = ['Aker_BP_weekly', 'Brent_Crude_weekly', 'Natural_Gas_weekly','NOK_USD_weekly', 'OSEBX_weekly', 'US_10Yr_Treasury_weekly']  # Replace with your list of output files

# Loop through each input file and process
for input_file, output_file in zip(input_files, output_files):
    # Load the CSV file into a DataFrame
    input_file_path = os.path.join('financial_data', input_file)
    df = pd.read_csv(input_file_path)

    # Convert the 'Date' column to datetime if not already
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Create new columns for the year and week number
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week

    # Resample the data to weekly frequency, using the desired aggregation function (e.g., 'mean', 'sum')
    df_weekly = df.resample('W').agg({
    'Close': 'mean',  # Apply mean to numeric data
    'High': 'mean', 
    'Low': 'mean',
    'Open': 'mean',
    'Volume': 'mean',
    'Log_Returns': 'mean',# Apply sum to another numeric column
    'Week': 'first',         # Take the first value for week number
    'Year': 'last'             # Take the first value for year
})
    
    df_weekly['Date'] = df_weekly['Year'].astype(int).astype(str) + '-' + df_weekly['Week'].astype(int).astype(str).str.zfill(2)
    df_weekly = df_weekly.drop(columns=['Year', 'Week'])
    columns = df_weekly.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df_weekly = df_weekly[columns]

    # Save the transformed weekly data to a new CSV file
    file_path = os.path.join('financial_data', f"{output_file}.csv")
    df_weekly.to_csv(file_path, index=False)

    print(f"Processed {input_file} and saved as {output_file}")
    
# Norwegian policy rate with different columns
input_file_path = os.path.join('financial_data', 'Norwegian_Policy_Rate.csv')
df = pd.read_csv(input_file_path)

# Convert the 'Date' column to datetime if not already
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Create new columns for the year and week number
df['Year'] = df.index.year
df['Week'] = df.index.isocalendar().week

# Resample the data to weekly frequency, using the desired aggregation function (e.g., 'mean', 'sum')
df_weekly = df.resample('W').agg({
'Rate': 'last',  # Apply mean to numeric data
'Week': 'first',         # Take the first value for week number
'Year': 'last'             # Take the first value for year
})

#df_weekly[['Year', 'Week']] = df_weekly[['Year', 'Week']].fillna(method='ffill')
df_weekly['Date'] = df_weekly['Year'].astype(int).astype(str) + '-' + df_weekly['Week'].astype(int).astype(str).str.zfill(2)
df_weekly = df_weekly.drop(columns=['Year', 'Week'])
columns = df_weekly.columns.tolist()
columns = columns[-1:] + columns[:-1]
df_weekly = df_weekly[columns]

# Save the transformed weekly data to a new CSV file
file_path = os.path.join('financial_data', f"Norwegian_Policy_Rate_weekly.csv")
df_weekly.to_csv(file_path, index=False)

print(f"Processed Norwegian_Policy_Rate and saved as Norwegian_Policy_Rate_weekly")

# CPI
# Norwegian policy rate with different columns
input_file_path = os.path.join('financial_data', 'CPI_Rate.csv')
df = pd.read_csv(input_file_path)

# Convert the 'Date' column to datetime if not already
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Function to compute the number of weeks in a month
def weeks_in_month(date):
    return len(pd.date_range(start=date.replace(day=1), 
                             end=date.replace(day=28) + pd.DateOffset(days=4),  # Ensures full month coverage
                             freq='W'))

# Compute the number of weeks for each month
df['Weeks_In_Month'] = df.index.to_series().apply(weeks_in_month)

# Convert Monthly Rate to Weekly Rate based on actual weeks in month
df['Rate'] = (1 + df['Rate'])**(1/df['Weeks_In_Month']) - 1

# Expand the monthly rate to all weeks within the month
df_weekly = df.resample('W').ffill()

# Reset index to retain the date column
df_weekly.reset_index(inplace=True)

# Create a 'Year' and 'Week' column
df_weekly['Year'] = df_weekly['Date'].dt.year
df_weekly['Week'] = df_weekly['Date'].dt.isocalendar().week

# Create a formatted 'Date' column in Year-Week format
df_weekly['Date'] = df_weekly['Year'].astype(str) + '-' + df_weekly['Week'].astype(str).str.zfill(2)

# Reorder columns
# Reorder columns to ensure 'Date' is the first column
columns = ['Date'] + [col for col in df_weekly.columns if col != 'Date']
df_weekly = df_weekly[columns]

# Drop unnecessary columns
df_weekly = df_weekly.drop(columns=['Weeks_In_Month', 'Year', 'Week'])

# Save the transformed weekly data to a new CSV file
file_path = os.path.join('financial_data', 'CPI_Rate_weekly.csv')
df_weekly.to_csv(file_path, index=False)

print("Processed CPI_Rate and saved as CPI_Rate_weekly")