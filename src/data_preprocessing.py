import pandas as pd
import numpy as np
import os

def process_catchment(forcing_path, streamflow_path, output_path, report_file, catchment_name, start_year="1999", end_year="2013"):
    print(f"Processing {catchment_name}...")

    # Load data
    forcing_df = pd.read_csv(forcing_path, delim_whitespace=True, skiprows=3)
    forcing_df['Date'] = pd.to_datetime(forcing_df[['Year', 'Mnth', 'Day']].rename(columns={'Mnth': 'Month'}))
    forcing_df = forcing_df.set_index('Date').drop(columns=['Year', 'Mnth', 'Day', 'Hr'])

    streamflow_cols = ['gauge_id', 'Year', 'Mnth', 'Day', 'discharge', 'qc_flag']
    streamflow_df = pd.read_csv(streamflow_path, delim_whitespace=True, header=None, names=streamflow_cols)
    streamflow_df['Date'] = pd.to_datetime(streamflow_df[['Year', 'Mnth', 'Day']].rename(columns={'Mnth': 'Month'}))
    streamflow_df = streamflow_df.set_index('Date')
    
    streamflow_df['discharge'] = streamflow_df['discharge'].replace(-999.00, np.nan)
    streamflow_df = streamflow_df[['discharge']]

    # Merge and filter dates
    merged_df = forcing_df.join(streamflow_df, how='inner')
    merged_df = merged_df.loc[f'{start_year}-01-01':f'{end_year}-12-31']

    # Track missing values for the report
    missing_count = merged_df.isna().sum().sum()
    
    if missing_count > 0:
        merged_df = merged_df.interpolate(method='time').ffill().bfill()

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path)

    # Write to the summary report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "a") as f:
        f.write(f"--- {catchment_name} Catchment ---\n")
        f.write(f"Start Date: {merged_df.index.min().date()}\n")
        f.write(f"End Date: {merged_df.index.max().date()}\n")
        f.write(f"Total Days: {len(merged_df)}\n")
        f.write(f"Missing/Interpolated Values: {missing_count}\n\n")

    print(f"  Saved {catchment_name} data.\n")

if __name__ == "__main__":
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    report_path = "outputs/tables/preprocessing_report.txt"

    # Clear previous report if it exists
    if os.path.exists(report_path):
        os.remove(report_path)

    process_catchment(
        forcing_path=os.path.join(raw_dir, "snow_forcing.txt"),
        streamflow_path=os.path.join(raw_dir, "snow_streamflow.txt"),
        output_path=os.path.join(processed_dir, "snow_processed.csv"),
        report_file=report_path,
        catchment_name="Snow-Driven (Colorado)"
    )

    process_catchment(
        forcing_path=os.path.join(raw_dir, "rain_forcing.txt"),
        streamflow_path=os.path.join(raw_dir, "rain_streamflow.txt"),
        output_path=os.path.join(processed_dir, "rain_processed.csv"),
        report_file=report_path,
        catchment_name="Rain-Driven (Southeast US)"
    )