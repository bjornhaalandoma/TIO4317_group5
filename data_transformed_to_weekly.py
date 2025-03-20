import os
import pandas as pd
import numpy as np

# Lists of CSV input filenames and corresponding output CSV filenames
input_files = ['Aker_BP.csv', 'Brent_Crude.csv', 'Natural_Gas.csv', 'NOK_USD.csv', 'OSEBX.csv', 'US_10Yr_Treasury.csv']  
output_files = ['Aker_BP_weekly', 'Brent_Crude_weekly', 'Natural_Gas_weekly','NOK_USD_weekly', 'OSEBX_weekly', 'US_10Yr_Treasury_weekly'] 

# Loop through each input file
for input_file, output_file in zip(input_files, output_files):
    # Load the CSV file into a DataFrame
    input_file_path = os.path.join('financial_data', input_file)
    df = pd.read_csv(input_file_path)

    # Convert the 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    
    # Create new columns for the year and week number
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week

    # Resample the data to weekly frequency, using mean for all the number columns
    df_weekly = df.resample('W').agg({
    'Close': 'mean',  
    'High': 'mean', 
    'Low': 'mean',
    'Open': 'mean',
    'Volume': 'mean',
    'Log_Returns': 'mean',
    'Week': 'first',         
    'Year': 'last'             
})
    # Formatting the csv. file
    df_weekly['Date'] = df_weekly['Year'].astype(int).astype(str) + '-' + df_weekly['Week'].astype(int).astype(str).str.zfill(2) # Change the 'Date' column to have the format yyyy-weekNo.
    df_weekly = df_weekly.drop(columns=['Year', 'Week']) # Remove columns 'Year' and 'Week that were used to format the 'Date' column
    columns = df_weekly.columns.tolist() 
    columns = columns[-1:] + columns[:-1] # Put the 'Date' column first
    df_weekly = df_weekly[columns] 

    # Save the transformed weekly data to a new CSV file
    file_path = os.path.join('financial_data', f"{output_file}.csv")
    df_weekly.to_csv(file_path, index=False)

    print(f"Processed {input_file} and saved as {output_file}")
    
# The same procedure as above for the Norwegian policy rate that has different columns that the csv files above
input_file_path = os.path.join('financial_data', 'Norwegian_Policy_Rate.csv')
df = pd.read_csv(input_file_path)

# Convert the 'Date' column to datetime if not already
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Create new columns for the year and week number
df['Year'] = df.index.year
df['Week'] = df.index.isocalendar().week

# Resample the data to weekly frequency, using the last rate of the week
df_weekly = df.resample('W').agg({
'Rate': 'last',  
'Week': 'first',        
'Year': 'last'             
})

# Formatting the csv. file
df_weekly['Date'] = df_weekly['Year'].astype(int).astype(str) + '-' + df_weekly['Week'].astype(int).astype(str).str.zfill(2) # Change the 'Date' column to have the format yyyy-weekNo.
df_weekly = df_weekly.drop(columns=['Year', 'Week']) # Remove columns 'Year' and 'Week that were used to format the 'Date' column
columns = df_weekly.columns.tolist()
columns = columns[-1:] + columns[:-1] # Put the 'Date' column first
df_weekly = df_weekly[columns] 

# Save the transformed weekly data to a new CSV file
file_path = os.path.join('financial_data', f"Norwegian_Policy_Rate_weekly.csv")
df_weekly.to_csv(file_path, index=False)

print(f"Processed Norwegian_Policy_Rate and saved as Norwegian_Policy_Rate_weekly")

# The same procedure as above for the CPI rate that has different columns that the csv files above and consist of monthly data and not daily
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

# Convert Monthly Rate to Weekly Rate using compound rate approximation and the number of weeks in the month
df['Rate'] = (1 + df['Rate'])**(1/df['Weeks_In_Month']) - 1

# Expand the monthly rate to all weeks within the month
df_weekly = df.resample('W').ffill()

# Reset index to retain the date column
df_weekly.reset_index(inplace=True)

# Create a 'Year' and 'Week' column
df_weekly['Year'] = df_weekly['Date'].dt.year
df_weekly['Week'] = df_weekly['Date'].dt.isocalendar().week

# Formatting the csv. file
df_weekly['Date'] = df_weekly['Year'].astype(str) + '-' + df_weekly['Week'].astype(str).str.zfill(2) # Change the 'Date' column to have the format yyyy-weekNo.
columns = ['Date'] + [col for col in df_weekly.columns if col != 'Date'] # Reorder columns to ensure 'Date' is the first column
df_weekly = df_weekly[columns]
df_weekly = df_weekly.drop(columns=['Weeks_In_Month', 'Year', 'Week']) # Remove columns 'Weeks_In_Month', 'Year' and 'Week that were used in calculations and formatting

# Save the transformed weekly data to a new CSV file
file_path = os.path.join('financial_data', 'CPI_Rate_weekly.csv')
df_weekly.to_csv(file_path, index=False)

print("Processed CPI_Rate and saved as CPI_Rate_weekly")