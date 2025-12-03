import pandas as pd
import matplotlib.pyplot as plt

# Define the filename
filename = '~/Documents/run_qwen.csv'  # <-- This is your path

try:
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename)

    # --- NEW: Clean up column names and data ---
    # Your output shows spaces around column names (e.g., ' Value')
    df.columns = df.columns.str.strip()
    
    # Check for required columns and strip spaces
    required_cols = ['Metric', 'Category']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found. Available columns:")
            print(df.columns)
            exit() # Exit if the key column is missing
        # Strip spaces from string columns
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    # --- End of new section ---


    # Print the first 5 rows (the 'head') of the DataFrame
    print("DataFrame loaded successfully. Here are the first 5 rows:")
    print(df.head())
    
    # You can also print info about the columns and data types
    print("\nDataFrame Info:")
    df.info()

    # --- NEW: Filter out unwanted categories ---
    print("\nOriginal categories found:")
    print(df['Category'].unique())
    
    category_to_exclude = 'DSP - Application'
    if category_to_exclude in df['Category'].unique():
        original_rows = len(df)
        
        # This is the line that does the filtering
        df = df[df['Category'] != category_to_exclude]
        
        print(f"\nFiltered out '{category_to_exclude}'. {original_rows - len(df)} rows removed.")
        print("Remaining categories:")
        print(df['Category'].unique())
    else:
        print(f"\nCategory '{category_to_exclude}' not found, no filtering needed.")
    # --- End of new section ---

    # Print unique metrics (from the *filtered* dataframe)
    print("\nUnique Metrics Found (after filtering):")
    print(df['Metric'].unique())

    # --- Plotting Section ---
    print("\nGenerating grouped metric graphs...")

    # Define 6 groups for the metrics
    metric_groups = [
        {'title': 'DSP & HMX Utilization', 
         'metrics': ['ADSP % Utilization', 'CDSP % Utilization', 'HMX Utilization']},
        
        {'title': 'Memory Usage', 
         'metrics': ['Available Memory', 'Free Memory', 'Total Memory', 'Used Memory']},
        
        {'title': 'Clock & Frequency', 
         'metrics': ['Bus Clock Vote', 'Core Clock', 'Eff Q6 Frequency']},
        
        {'title': 'AXI Bandwidth', 
         'metrics': ['AXI RD BW', 'AXI WR BW']},
        
        {'title': 'Performance (MPPS)', 
         'metrics': ['HVX MPPS', 'MPPS', 'pCPP']},
        
        {'title': 'Other (Temperature & L2 Miss)', 
         'metrics': ['Temperature', 'L2 FETCH DU MISS']}
    ]

    # Create a 3x2 grid of subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten() # Flatten the 2D array of axes for easy looping

    # Loop through each group and its assigned subplot
    for i, group in enumerate(metric_groups):
        ax = axes[i]
        ax.set_title(group['title'], fontsize=14, fontweight='bold')
        
        has_data = False
        for metric in group['metrics']:
            # Filter the DataFrame for the specific metric
            # This now searches in the *pre-filtered* DataFrame
            metric_df = df[df['Metric'] == metric]
            
            if not metric_df.empty:
                # Plot Timestamp vs. Value
                ax.plot(metric_df['Timestamp'], metric_df['Value'], label=metric)
                has_data = True
        
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.grid(True)
        
        # Only add a legend if data was plotted
        if has_data:
            ax.legend()
        else:
            # This will now correctly show 'No data' if a metric only
            # existed in the 'DSP - Application' category you removed
            ax.text(0.5, 0.5, 'No data for these metrics', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red')

    # Add a main title for the entire figure
    fig.suptitle('Metrics Dashboard (Filtered)', fontsize=22, fontweight='bold')
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # rect leaves space for suptitle
    
    # Save the figure to a file
    output_filename = 'metrics_graphs_filtered.png' # Changed filename
    plt.savefig(output_filename)
    
    print(f"\nSuccessfully saved filtered graphs to '{output_filename}'")
    # --- End of plotting section ---

except FileNotFoundError:
    print(f"Error: The file '{filename}' was not found. Please check the path.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{filename}' is empty.")
except Exception as e:
    print(f"An error occurred: {e}")