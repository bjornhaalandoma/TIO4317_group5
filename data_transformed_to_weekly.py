import os
import pandas as pd
import numpy as np
import os
import calendar

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
# Load CPI monthly data
input_file_path = os.path.join('financial_data', 'CPI_Rate.csv')
df_monthly = pd.read_csv(input_file_path)
df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
df_monthly.set_index('Date', inplace=True)

# Create a new weekly date range from the start to the end of data
start_date = df_monthly.index.min()
end_date = pd.to_datetime('2025-03-03')  # ensure it includes week 9
weekly_dates = pd.date_range(start=start_date, end=end_date, freq='W-WED')

# Prepare a new DataFrame to store weekly CPI rates
weekly_data = []

for date, row in df_monthly.iterrows():
    year = date.year
    month = date.month

    # Calculate number of weeks in the month
    first_day = pd.Timestamp(year=year, month=month, day=1)
    last_day = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
    weeks_in_month = len(pd.date_range(start=first_day, end=last_day, freq='W'))

    # Compute equivalent weekly rate using compound rate approximation
    monthly_rate = row['Rate']
    weekly_rate = (1 + monthly_rate)**(1 / weeks_in_month) - 1

    # Get all weekly index within this month
    weekly_dates_in_month = [d for d in weekly_dates if d >= first_day and d <= last_day]

    for d in weekly_dates_in_month:
        weekly_data.append({'Date': d, 'Rate': weekly_rate})

# Formatting the csv. file
df_weekly = pd.DataFrame(weekly_data) # Convert to DataFrame
df_weekly['Year'] = df_weekly['Date'].dt.year # Add year and ISO week number
df_weekly['Week'] = df_weekly['Date'].dt.isocalendar().week # Add year and ISO week number
df_weekly['Date'] = df_weekly['Year'].astype(str) + '-' + df_weekly['Week'].astype(str).str.zfill(2) # Format date to yyyy-weekNo.
df_weekly = df_weekly[['Date', 'Rate']] # Rearrange and remove extra columns

# Save the transformed weekly data to a new CSV file
output_file_path = os.path.join('financial_data', 'CPI_Rate_weekly.csv')
df_weekly.to_csv(output_file_path, index=False)

print("Processed CPI_Rate and saved as CPI_Rate_weekly")