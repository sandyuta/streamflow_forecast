import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_hydrograph(csv_path, output_image_path, title):
    # Load the processed data
    df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
    
    # Create a figure with two subplots (Precipitation on top, Discharge on bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot Precipitation (usually plotted as a bar chart from the top down in hydrology)
    ax1.bar(df.index, df['prcp(mm/day)'], color='blue', width=1)
    ax1.invert_yaxis()  # Invert y-axis for precipitation
    ax1.set_ylabel('Precipitation (mm/day)')
    ax1.set_title(f'Hydrological Time Series: {title}')
    ax1.grid(True, alpha=0.3)

    # Plot Streamflow (Discharge)
    ax2.plot(df.index, df['discharge'], color='black', linewidth=1)
    ax2.set_ylabel('Discharge')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_image_path}")

if __name__ == "__main__":
    processed_dir = "data/processed"
    figures_dir = "outputs/figures"

    # Plot Snow Catchment
    plot_hydrograph(
        csv_path=os.path.join(processed_dir, "snow_processed.csv"),
        output_image_path=os.path.join(figures_dir, "snow_hydrograph.png"),
        title="Snow-Driven Catchment (Colorado)"
    )

    # Plot Rain Catchment
    plot_hydrograph(
        csv_path=os.path.join(processed_dir, "rain_processed.csv"),
        output_image_path=os.path.join(figures_dir, "rain_hydrograph.png"),
        title="Rain-Driven Catchment (Southeast US)"
    )