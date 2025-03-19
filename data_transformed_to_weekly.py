import os
import pandas as pd

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