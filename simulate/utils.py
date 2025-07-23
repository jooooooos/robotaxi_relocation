import os
import collections
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
import holidays
from datetime import timedelta

from constants import (
    MAX_TAXI_ZONE_ID,
    location_ids,
    excluded_location_ids,
    location_id_to_index,
    num_locations,
    taxi_type,
    
    TIME_BLOCK_BOUNDARY,
    RIDER_ARRIVAL,
    RIDER_LOST,
    RIDE_COMPLETION,
    RELOCATION_COMPLETION,
    RELOCATION_START,
    RIDE_START,
    TAXI_INIT,
    
    location_ids,
)
from relocation_policies import *
from simulate import TaxiSimulator

Delta = 20 # in minutes
T_max = int(24 * (60 // Delta))
YEARS = list(range(2019, 2025))

us_holidays = holidays.US(years=YEARS)

def prepare_arrival_events_from_real_data(df, num_days=3):
    """
    Given a pre-filtered NYC trip dataframe (weekdays, valid IDs, etc.),
    extract a simulation-ready list of rider arrival events across `num_days` consecutive calendar days.

    Returns:
        events (List[Dict]): List of events with simulation time, origin, destination, and trip_time (in hours).
    """
    unique_dates = df.pickup_datetime.dt.date.unique()
    working_days = [date for date in unique_dates if date.weekday() < 5 and date not in us_holidays]

    # Filter for weekdays that are NOT US holidays
    df = df[
        (df.pickup_datetime.dt.weekday < 5) &  # Monday to Friday
        (~df.pickup_datetime.dt.date.isin(us_holidays))  # Exclude US holidays
    ]

    # filter for valid locatino IDs
    df = df[df['PULocationID'].isin(location_ids) & df['DOLocationID'].isin(location_ids)]
    df['time_bin'] = (df['pickup_datetime'].dt.hour * (60 // Delta) + df['pickup_datetime'].dt.minute // Delta).astype(int)

    # Map IDs to array indices
    df['pu_idx'] = df['PULocationID'].map(location_id_to_index)
    df['do_idx'] = df['DOLocationID'].map(location_id_to_index)

    # Round pickup_datetime to dates only
    pickup_dates = df['pickup_datetime'].dt.date.to_numpy()

    # Find earliest set of consecutive calendar days
    unique_dates = np.unique(pickup_dates)
    for i in range(len(unique_dates) - num_days + 1):
        base = unique_dates[i]
        if all((base + timedelta(days=j)) in unique_dates for j in range(num_days)):
            selected_dates = {base + timedelta(days=j) for j in range(num_days)}
            break
    else:
        raise ValueError(f"No consecutive {num_days}-day window found.")

    # Filter df just once
    mask = np.isin(pickup_dates, list(selected_dates))
    df_sel = df.loc[mask].copy()

    # Sort once, for time order
    df_sel.sort_values('pickup_datetime', inplace=True)

    # Compute simulation time in hours relative to min time
    min_time = df_sel['pickup_datetime'].min()
    df_sel['t_sim'] = (df_sel['pickup_datetime'] - min_time).dt.total_seconds() / 3600.0

    # Convert trip_time to hours
    df_sel['trip_time_hr'] = df_sel['trip_time'] / 3600.0

    # Pack into list of event dicts
    return df_sel[['t_sim', 'pu_idx', 'do_idx', 'trip_time_hr']].to_dict('records')

# --- Optimized Helper Function ---
def load_and_process_log(filepath, region_id, bin_minutes):
    """
    Loads log data, extracts relevant info, filters, calculates time bins,
    and performs a single group-by operation.
    """
    try:
        # 1. Load Data (Consider json.loads if applicable)
        # df = pd.read_csv(filepath, converters={'data': json.loads}) # Faster & safer if data is JSON
        df = pd.read_csv(filepath, converters={'data': eval}) # Keep eval if data is Python literal dicts

        # 2. Convert time and create datetime column *once*
        # Use a fixed start date for consistency if not present in logs
        # Adjust '2025-01-02' if your simulation has a different base date
        base_timestamp = pd.Timestamp('2025-01-02')
        df['datetime'] = pd.to_timedelta(df['time'], unit='h') + base_timestamp

        # 3. Filter relevant event types *early*
        event_types_needed = [RIDER_ARRIVAL, RIDE_START, RIDER_LOST]
        df_filtered = df[df['event_type'].isin(event_types_needed)].copy() # Copy slice to avoid SettingWithCopyWarning

        # 4. Extract relevant region/origin *once* based on event type
        def get_relevant_region(row):
            event = row['event_type']
            data = row['data']
            if event == RIDE_START:
                return data.get('origin')
            elif event in [RIDER_ARRIVAL, RIDER_LOST]:
                return data.get('region')
            return np.nan

        df_filtered['relevant_region'] = df_filtered.apply(get_relevant_region, axis=1)
        df_filtered['relevant_region'] = pd.to_numeric(df_filtered['relevant_region'], errors='coerce')


        # 5. Filter by the specific region_id *early*
        df_region = df_filtered[df_filtered['relevant_region'] == region_id].copy()

        if df_region.empty:
            return pd.DataFrame(columns=['time_bin', RIDER_ARRIVAL, RIDE_START, RIDER_LOST]).set_index('time_bin')


        # 6. Calculate time bins *once*
        df_region['time_bin'] = df_region['datetime'].dt.floor(f'{bin_minutes}min')

        # 7. Single Groupby and Unstack
        # Group by the calculated time bin and the event type, count occurrences
        # Then unstack to get event types as columns
        event_counts = df_region.groupby(['time_bin', 'event_type']).size().unstack(fill_value=0)

        # Ensure all required columns exist, adding them with 0 if necessary
        for etype in event_types_needed:
            if etype not in event_counts.columns:
                event_counts[etype] = 0

        return event_counts # Return the grouped counts indexed by time_bin

    except FileNotFoundError:
        print(f"Warning: File not found - {filepath}")
        return None
    except Exception as e:
        print(f"Warning: Error processing file {filepath} - {e}")
        # Optionally re-raise if debugging: raise e
        return None


# --- Refactored Plotting Function ---
def plot_simulation_grid_for_strategies(
    demand_mode: str,
    strategies: list,
    region_id: int,
    bin_minutes: int = 20,
    log_dir: str = "sim_outputs",
    filename: str = None,
    n: int = 0,
    ignore_days: int = 2, # Number of initial days to ignore for plotting
):
    """
    Plots rider arrival, ride start, and lost rider time series (binned)
    for multiple strategies using optimized data loading and processing.

    Args:
        demand_mode (str): 'real' or 'synthetic'
        strategies (list): list of strategy names used in file naming
        region_id (int): region ID to analyze
        bin_minutes (int): time bin size in minutes
        log_dir (str): directory where logs are saved
        filename (str): optional filename to save the plot
        n (int): run number (0, 1, 2, ...)
        ignore_days (int): number of initial days to skip in plot for stabilization
    """
    num_strategies = len(strategies)
    # Adjust grid size dynamically if needed, assuming 2 columns for now
    num_rows = (num_strategies + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 4 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten to easily index

    sns.set(style="whitegrid")
    plot_successful = False # Flag to check if any plot was made
    min_plot_time = None
    max_plot_time = None
    data_cache = {} # Cache data if needed across runs/modes (optional)

    # Determine common time range across strategies if needed (or use first valid one)
    overall_start_time = None
    overall_end_time = None

    # --- First pass: Load data and determine time range ---
    for i, strategy in enumerate(strategies):
        fname = f"{n}/{demand_mode}_demand__{strategy}.csv"
        path = os.path.join(log_dir, fname)

        processed_data = load_and_process_log(path, region_id, bin_minutes)
        data_cache[strategy] = processed_data # Store processed data

        if processed_data is not None and not processed_data.empty:
             # Determine time range from actual data
             current_min_time = processed_data.index.min()
             current_max_time = processed_data.index.max()
             # Update overall range
             if overall_start_time is None or current_min_time < overall_start_time:
                 overall_start_time = current_min_time
             if overall_end_time is None or current_max_time > overall_end_time:
                 overall_end_time = current_max_time


    if overall_start_time is None:
         print("Error: No valid data found for any strategy to determine time range.")
         plt.close(fig) # Close empty figure
         return

    # Calculate the actual start time for plotting (after ignoring initial days)
    plot_start_time = overall_start_time + timedelta(days=ignore_days)
    plot_end_time = overall_end_time

    # Create the full time index for reindexing
    full_time_index = pd.date_range(
        start=plot_start_time,
        end=plot_end_time,
        freq=f'{bin_minutes}min'
    )

    # --- Second pass: Reindex and Plot ---
    max_y_value = 0 # To potentially adjust y-limits later if needed

    for i, strategy in enumerate(strategies):
        event_counts = data_cache.get(strategy) # Retrieve from cache

        ax = axes[i] # Get the subplot axis

        if event_counts is None:
            print(f"Skipping plot for {strategy} (Error during processing)")
            ax.set_title(f"{strategy.replace('_', ' ').upper()} (No Data)")
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue
        elif event_counts.empty:
            print(f"No data for region {region_id} in strategy {strategy}.")
            ax.set_title(f"{strategy.replace('_', ' ').upper()} (No Region Data)")
            ax.text(0.5, 0.5, 'No Region Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue


        # Reindex with the full time range, filling missing bins with 0
        event_counts_filled = event_counts.reindex(full_time_index, fill_value=0)

        # Filter data to the desired plot range *after* reindexing
        event_counts_plot = event_counts_filled[(event_counts_filled.index >= plot_start_time) & (event_counts_filled.index <= plot_end_time)]

        if event_counts_plot.empty:
             print(f"No data within the plotting time range for strategy {strategy}.")
             ax.set_title(f"{strategy.replace('_', ' ').upper()} (No Plot Data)")
             ax.text(0.5, 0.5, 'No Plot Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             continue


        # Plotting
        ax.plot(event_counts_plot.index, event_counts_plot[RIDER_ARRIVAL], label='Arrivals', color='blue', marker='.', linestyle='-')
        ax.plot(event_counts_plot.index, event_counts_plot[RIDE_START], label='Ride Starts', color='orange', marker='.', linestyle='-')
        ax.plot(event_counts_plot.index, event_counts_plot[RIDER_LOST], label='Lost Riders', color='purple', marker='.', linestyle='-')
        ax.set_title(strategy.replace("_", " ").upper())
        ax.tick_params(axis='x', rotation=45)

        # Update max Y value seen for potential common Y limit setting
        max_y_value = max(max_y_value, event_counts_plot.max().max())

        plot_successful = True # Mark that at least one plot was generated


    # --- Final Touches ---
    if not plot_successful:
        print("No plots were generated.")
        plt.close(fig) # Close the empty figure
        return

    # Common Legend
    handles, labels = axes[0].get_legend_handles_labels() # Get legend from the first successful plot
    # Adjust anchor and position based on the number of rows
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.00 - (0.05 / num_rows)), ncol=3) # Dynamic positioning
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.91), ncol=3) # Dynamic positioning

    # Common Title
    fig.suptitle(
        f"Rider Arrivals, Ride Starts, and Lost Riders\nRegion {location_ids[region_id]} — {demand_mode.capitalize()} Demand",
        fontsize=16,
        y=0.98 # Adjust based on legend position
    )

    # Set shared Y-axis limit if desired (optional)
    # plt.setp(axes, ylim=(0, max_y_value * 1.1)) # Set ylim for all axes

    # Adjust layout
    plt.tight_layout(rect=[0.00, 0.00, 1, 0.95]) # Adjust rect based on title/legend

    # Save or Show
    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    plt.show()
    
    return data_cache

def calculate_regional_abandonment(df_log):
    """
    Calculates rider arrivals, lost riders, and abandonment rate per region.

    Args:
        df_log (pd.DataFrame): The pre-loaded log DataFrame.

    Returns:
        pd.DataFrame: Indexed by region ID, with columns 'arrivals', 'lost', 'abandonment_rate'.
                      Returns empty DataFrame on error or if no relevant data.
    """
    try:
        # Filter relevant events
        relevant_events = df_log[df_log['event_type'].isin([RIDER_ARRIVAL, RIDER_LOST])].copy()

        if relevant_events.empty:
            return pd.DataFrame(columns=['arrivals', 'lost', 'abandonment_rate'])

        # Extract regions (assuming 'data' contains 'region' for these events)
        # Use .get() for safety in case 'region' key is missing in some dicts
        relevant_events['region'] = relevant_events['data'].apply(lambda x: x.get('region', np.nan))

        # Drop rows where region extraction failed
        relevant_events.dropna(subset=['region'], inplace=True)

        # Convert region to numeric, handling potential errors
        relevant_events['region'] = pd.to_numeric(relevant_events['region'], errors='coerce')
        relevant_events.dropna(subset=['region'], inplace=True)
        relevant_events['region'] = relevant_events['region'].astype(int) # Convert to int after cleaning

        # Group by region and event type
        counts = relevant_events.groupby(['region', 'event_type']).size().unstack(fill_value=0)

        # Ensure standard columns exist
        if RIDER_ARRIVAL not in counts.columns: counts[RIDER_ARRIVAL] = 0
        if RIDER_LOST not in counts.columns: counts[RIDER_LOST] = 0

        # Rename for clarity
        counts = counts.rename(columns={RIDER_ARRIVAL: 'arrivals', RIDER_LOST: 'lost'})

        # Calculate abandonment rate (handle division by zero)
        counts['abandonment_rate'] = counts.apply(
            lambda row: row['lost'] / row['arrivals'] if row['arrivals'] > 0 else 0.0,
            axis=1
        )

        # Select relevant columns for output
        return counts[['arrivals', 'lost', 'abandonment_rate']]

    except Exception as e:
        print(f"Error calculating regional abandonment: {e}")
        return pd.DataFrame(columns=['arrivals', 'lost', 'abandonment_rate'])
    
def find_worst_region_for_strategy(demand_mode, strategy, n, log_dir="sim_outputs"):
    """
    Finds the region ID with the highest abandonment rate for a given strategy run.

    Args:
        demand_mode (str): 'real' or 'synthetic'.
        strategy (str): The strategy name (typically 'no_reloc').
        n (int): The run number.
        log_dir (str): Directory of log files.

    Returns:
        int or None: The region ID with the max abandonment rate, or None if an error occurs.
    """
    filepath = os.path.join(log_dir, f"{n}/{demand_mode}_demand__{strategy}.csv")
    print(f"Analyzing baseline strategy file: {filepath}")
    try:
        if not os.path.exists(filepath):
            print(f"Error: Baseline log file not found: {filepath}")
            return None

        # --- Load the log file ---
        # Use the appropriate converter (eval or json.loads)
        df_log = pd.read_csv(filepath, converters={'data': eval})
        # Add any other necessary preprocessing used by calculate_regional_abandonment if needed

        # --- Calculate regional stats ---
        regional_stats = calculate_regional_abandonment(df_log)

        if regional_stats.empty:
            print(f"Warning: No regional abandonment stats calculated for {strategy}.")
            return None

        if 'abandonment_rate' not in regional_stats.columns:
             print(f"Error: 'abandonment_rate' column missing.")
             return None

        # --- Find the region with the maximum abandonment rate ---
        # Ensure we don't select regions with 0 arrivals (rate is 0 by definition above)
        stats_with_arrivals = regional_stats[regional_stats['arrivals'] > 0]

        if stats_with_arrivals.empty:
             print(f"Warning: No regions with arrivals found for {strategy}.")
             return None

        # Find the index (region_id) of the row with the maximum rate
        worst_region_id = stats_with_arrivals['abandonment_rate'].idxmax()
        max_rate = stats_with_arrivals['abandonment_rate'].max()

        print(f"Identified Region {worst_region_id} as having the highest abandonment rate ({max_rate:.4f}) for strategy '{strategy}'.")
        return int(worst_region_id) # Return as integer

    except Exception as e:
        print(f"Error finding worst region for {strategy} (run {n}, demand {demand_mode}): {e}")
        # raise e # Uncomment to see full traceback during debugging
        return None

def run_all_simulations_for_seed(
    seed: int,
    lambda_: np.ndarray,
    mu_: np.ndarray,
    P: np.ndarray,
    Q_base: np.ndarray,
    arrival_events: list,
    T: int,
    R: int,
    N: int,
    max_time: float,
    eta: float = 0.5,
    output_dir: str = "sim_outputs"
):
    np.random.seed(seed)

    sampling_modes = ["real", "synthetic", ]
    relocation_modes = {
        "no_reloc": {"policy": relocation_policy_blind_sampling, "Q": Q_base},
        "JLCR": {"policy": relocation_policy_jlcr_eta, "Q": Q_base},
        "shortest_wait": {"policy": relocation_policy_shortest_wait, "Q": Q_base},
        "Q_2": {"policy": relocation_policy_blind_sampling, "Q_path": "../nyc_trip/Qs_2_clipping.npz"},
        "Q_4": {"policy": relocation_policy_blind_sampling, "Q_path": "../nyc_trip/Qs_4_clipping.npz"},
        "Q_8": {"policy": relocation_policy_blind_sampling, "Q_path": "../nyc_trip/Qs_8_clipping.npz"},
    }

    for sampling in sampling_modes:
        use_real = sampling == "real"
        for reloc_key, config in relocation_modes.items():            
            # Load Q matrix
            if "Q_path" in config:
                with np.load(config["Q_path"]) as data:
                    Q = data["Qs"]
            else:
                Q = config["Q"]

            # Setup relocation policy
            policy = config["policy"]
            kwargs = {"eta": eta} if reloc_key == "JLCR" else {}

            # Init simulator
            sim = TaxiSimulator(
                T=T,
                R=R,
                N=N,
                lambda_=lambda_,
                mu_=mu_,
                P=P,
                Q=Q,
                relocation_policy=policy,
                relocation_kwargs=kwargs,
                use_real_demand=use_real,
                demand_events=arrival_events if use_real else None,
            )

            # Run sim
            sim.run(max_time=max_time)
            df_log = pd.DataFrame(sim.logger)

            # Save
            seed_dir = os.path.join(output_dir, str(seed))
            os.makedirs(seed_dir, exist_ok=True)
            fname = f"{sampling}_demand__{reloc_key}.csv"
            df_log.to_csv(os.path.join(seed_dir, fname), index=False)
            
            print(f"[Seed {seed}] finished: {sampling} / {reloc_key}")
            
            
def safe_extract(data_dict, key):
    return data_dict.get(key, np.nan) # Use np.nan for missing numeric/object data


def preprocess_dflog(df_log):
    # Extract common fields (adjust based on actual keys present per event type)
    df_log['vehicle_id'] = df_log['data'].apply(lambda x: safe_extract(x, 'vehicle_id'))
    df_log['origin'] = df_log['data'].apply(lambda x: safe_extract(x, 'origin'))
    df_log['destination'] = df_log['data'].apply(lambda x: safe_extract(x, 'destination'))
    df_log['travel_time'] = df_log['data'].apply(lambda x: safe_extract(x, 'travel_time'))
    df_log['region'] = df_log['data'].apply(lambda x: safe_extract(x, 'region'))
    df_log['num_vehicles_init'] = df_log['data'].apply(lambda x: safe_extract(x, 'num_vehicles'))

    # --- Make sure numeric columns are numeric, converting errors to NaN ---
    numeric_cols = ['vehicle_id', 'origin', 'destination', 'travel_time', 'region', 'num_vehicles_init']
    for col in numeric_cols:
        # Check if column exists before trying to convert
        if col in df_log.columns:
            # Convert to float first to handle potential non-integers before Int64
            df_log[col] = pd.to_numeric(df_log[col], errors='coerce')
            # Use nullable integer type if appropriate (like for IDs)
            if col in ['vehicle_id', 'origin', 'destination', 'region', 'num_vehicles_init']:
                # Check for NaNs before converting to Int64, as Int64 cannot hold NaNs
                # If NaNs are present, might need to keep as float or fillna appropriately
                if not df_log[col].isnull().any():
                    df_log[col] = df_log[col].astype(pd.Int64Dtype()) # Use nullable integer
                # else: keep as float or handle NaNs

    # Ensure time is float
    df_log['time'] = pd.to_numeric(df_log['time'], errors='coerce')
    
    # Sort by time (important for queue calculations)
    df_log = df_log.sort_values('time').reset_index(drop=True)
    return df_log

def get_abandonment_rate(df_log):
    total_arrivals = df_log[df_log['event_type'] == RIDER_ARRIVAL].shape[0]
    total_lost = df_log[df_log['event_type'] == RIDER_LOST].shape[0]

    if total_arrivals > 0:
        abandonment_rate = total_lost / total_arrivals
    else:
        abandonment_rate = 0 # Or np.nan if you prefer

    # print(f"1. Overall Abandonment Rate: {abandonment_rate:.4f}")
    return abandonment_rate

def get_rates(df_log):
    # Calculate total ride time (sum of travel_time from RIDE_START)
    total_ride_time = df_log.loc[df_log['event_type'] == RIDE_START, 'travel_time'].sum()
    total_relocation_time = df_log.loc[df_log['event_type'] == RELOCATION_START, 'travel_time'].sum()

    # Get total number of vehicles
    total_vehicles = df_log.loc[df_log['event_type'] == TAXI_INIT, 'num_vehicles_init'].sum()

    # Get total simulation duration
    total_simulation_time = df_log['time'].max()

    if total_vehicles > 0 and total_simulation_time > 0:
        total_vehicle_hours = total_vehicles * total_simulation_time
        utilization_rate = total_ride_time / total_vehicle_hours
    else:
        utilization_rate = 0 # Or np.nan

    # Total vehicle hours calculated in step 2
    if total_vehicles > 0 and total_simulation_time > 0:
        relocation_time_ratio = total_relocation_time / total_vehicle_hours
    else:
        relocation_time_ratio = 0 # Or np.nan
        
    # Assuming total time = ride time + relocation time + idle time
    if total_vehicles > 0 and total_simulation_time > 0:
        idle_time_ratio = 1.0 - utilization_rate - relocation_time_ratio
        # Ensure it's not negative due to potential minor inaccuracies or overlaps if using actual durations
        idle_time_ratio = max(0, idle_time_ratio)
    else:
        idle_time_ratio = 0 # Or np.nan

    # print(f"Total Vehicles: {total_vehicles}")
    # print(f"Total Simulation Time (hours): {total_simulation_time:.2f}")
    # print(f"Total Ride Time (hours): {total_ride_time:.2f}")
    # print(f"Total Vehicle Hours: {total_vehicle_hours:.2f}")
    # print(f"2. Utilization Rate: {utilization_rate:.4f}")
    return utilization_rate, relocation_time_ratio, idle_time_ratio

def get_temporal_max_abandonment_rate(df_log, bin_size_hours=0.5):
    # Define bin size (0.5 hours for 30 minutes)
    max_time = df_log['time'].max()
    bins = np.arange(0, max_time + bin_size_hours, bin_size_hours)
    labels = bins[:-1] # Label bins by their start time

    # Assign time bin to each event
    df_log['time_bin'] = pd.cut(df_log['time'], bins=bins, labels=labels, right=False)

    # Count arrivals and lost riders per bin
    arrivals_per_bin = df_log[df_log['event_type'] == RIDER_ARRIVAL].groupby('time_bin', observed=False).size()
    lost_per_bin = df_log[df_log['event_type'] == RIDER_LOST].groupby('time_bin', observed=False).size()

    # Combine counts into a single DataFrame, filling missing bins with 0
    temporal_rates = pd.DataFrame({'arrivals': arrivals_per_bin, 'lost': lost_per_bin}).fillna(0)

    # Calculate rate per bin, handling division by zero
    temporal_rates['rate'] = temporal_rates.apply(
        lambda row: row['lost'] / row['arrivals'] if row['arrivals'] > 0 else 0,
        axis=1
    )

    max_temporal_abandonment_rate = temporal_rates['rate'].max()

    # print(f"\nTemporal Abandonment Rates per {bin_size_hours}hr Bin:")
    # print(temporal_rates)
    # print(f"\n5. Max Temporal Abandonment Rate: {max_temporal_abandonment_rate:.4f}")
    return max_temporal_abandonment_rate

def get_regional_max_abandonment_rate(df_log):
    # --- Ensure 'region' column exists and is suitable for grouping ---
    # The extraction step earlier should have created 'region'
    # Filter for relevant events
    arrival_events = df_log[df_log['event_type'] == RIDER_ARRIVAL].dropna(subset=['region'])
    lost_events = df_log[df_log['event_type'] == RIDER_LOST].dropna(subset=['region'])

    # Count arrivals and lost riders per region
    arrivals_per_region = arrival_events.groupby('region').size()
    lost_per_region = lost_events.groupby('region').size()

    # Combine counts, filling missing regions with 0
    regional_rates = pd.DataFrame({'arrivals': arrivals_per_region, 'lost': lost_per_region}).fillna(0)

    # Calculate rate per region, handling division by zero
    regional_rates['rate'] = regional_rates.apply(
        lambda row: row['lost'] / row['arrivals'] if row['arrivals'] > 0 else 0,
        axis=1
    )

    max_regional_abandonment_rate = regional_rates['rate'].max()

    # print(f"\nRegional Abandonment Rates:")
    # print(regional_rates)
    # print(f"\n6. Max Regional Abandonment Rate: {max_regional_abandonment_rate:.4f}")

    # Identify region(s) with the max rate
    max_rate_regions = regional_rates[regional_rates['rate'] == max_regional_abandonment_rate].index.tolist()
    # print(f"Region(s) with Max Rate: {max_rate_regions}")
    return max_regional_abandonment_rate, max_rate_regions

def get_maximal_average_queue_length(df_log):
    # Initialize queues based on TAXI_INIT events
    region_queues = collections.defaultdict(int)
    init_events = df_log[df_log['event_type'] == TAXI_INIT]
    for _, row in init_events.iterrows():
        # Use extracted columns if available, otherwise parse 'data' again
        region = row['region'] # Assumes region column exists after preparation
        num_vehicles = row['num_vehicles_init'] # Assumes num_vehicles_init column exists
        if pd.notna(region) and pd.notna(num_vehicles):
            region_queues[int(region)] += int(num_vehicles)

    # Data structures to store results
    queue_history = [] # Stores (time, region_id, queue_length)
    min_queues = region_queues.copy() # Initialize min with initial state
    max_queues = region_queues.copy() # Initialize max with initial state
    time_weighted_queues = collections.defaultdict(float) # For average calculation sum(queue_length * time_delta)

    # Get all unique regions involved in the simulation
    all_regions = set(region_queues.keys())
    activity_regions = df_log.dropna(subset=['origin'])['origin'].unique()
    destination_regions = df_log.dropna(subset=['destination'])['destination'].unique()
    rider_regions = df_log.dropna(subset=['region'])['region'].unique()

    all_regions.update(map(int, filter(np.isfinite, activity_regions)))
    all_regions.update(map(int, filter(np.isfinite, destination_regions)))
    all_regions.update(map(int, filter(np.isfinite, rider_regions)))

    # Initialize tracking for all regions
    for r in all_regions:
        r = int(r)
        if r not in region_queues: region_queues[r] = 0
        if r not in min_queues: min_queues[r] = 0
        if r not in max_queues: max_queues[r] = 0


    # Iterate through events chronologically
    last_event_time = 0.0
    total_simulation_time = df_log['time'].max() # Recalculate just in case

    # --- Add initial state to history ---
    current_time = 0.0
    for region_id, q_len in region_queues.items():
        queue_history.append((current_time, region_id, q_len))


    for index, event in df_log.iterrows():
        current_time = event['time']
        event_type = event['event_type']
        # Extract data needed for this event using prepared columns
        origin = event['origin']
        destination = event['destination']

        # Calculate time delta since last event
        time_delta = current_time - last_event_time

        # --- Update time-weighted sum for average calculation ---
        # Add contribution of the queue state *during* the interval [last_event_time, current_time)
        if time_delta > 0:
            for region_id, q_len in region_queues.items():
                # Ensure region_id is a basic type if necessary
                time_weighted_queues[region_id] += q_len * time_delta

        # --- Update queue lengths based on the *current* event ---
        region_affected = None
        change = 0
        if event_type == RIDE_START and pd.notna(origin):
            region_affected = int(origin)
            change = -1
        elif event_type == RELOCATION_START and pd.notna(origin):
            region_affected = int(origin)
            change = -1
        elif event_type == RIDE_COMPLETION and pd.notna(destination):
            region_affected = int(destination)
            change = 1
        elif event_type == RELOCATION_COMPLETION and pd.notna(destination):
            region_affected = int(destination)
            change = 1

        if region_affected is not None:
            # Update queue, ensuring it doesn't go negative (shouldn't happen in a valid sim)
            region_queues[region_affected] = max(0, region_queues.get(region_affected, 0) + change)
            new_q_len = region_queues[region_affected]

            # --- Record state change for history and min/max ---
            queue_history.append((current_time, region_affected, new_q_len))
            min_queues[region_affected] = min(min_queues.get(region_affected, float('inf')), new_q_len)
            max_queues[region_affected] = max(max_queues.get(region_affected, float('-inf')), new_q_len)


        # Update last event time
        last_event_time = current_time

    # --- Add contribution for the final interval after the last event ---
    final_time_delta = total_simulation_time - last_event_time
    if final_time_delta > 0:
        for region_id, q_len in region_queues.items():
            time_weighted_queues[region_id] += q_len * final_time_delta


    # --- Calculate Final Queue Metrics ---
    avg_queues = {}
    if total_simulation_time > 0:
        for region_id, weighted_sum in time_weighted_queues.items():
            avg_queues[region_id] = weighted_sum / total_simulation_time
    else:
        for region_id in all_regions:
            avg_queues[region_id] = 0 # Or NaN

    # Combine results into a DataFrame
    queue_stats = pd.DataFrame({
        'Average': pd.Series(avg_queues),
        'Min': pd.Series(min_queues),
        'Max': pd.Series(max_queues)
    }).fillna({'Min': 0, 'Max': 0}) # Fill NaN for regions with no activity if needed

    # Ensure all regions are present, even if they had 0 activity throughout
    queue_stats = queue_stats.reindex(list(all_regions)).fillna(0) # Fill potentially missing regions with 0

    # print(f"\n7. Queue Length Stats per Region (Idle Taxis Waiting):")
    # print(queue_stats)

    # You might also want the full history for plotting:
    # queue_history_df = pd.DataFrame(queue_history, columns=['time', 'region', 'queue_length'])
    # print("\nQueue History Sample:")
    # print(queue_history_df.head())
    return queue_stats.Average.max()

def compute_metrics(df_log):
    # remove rows with event_type == TIME_BLOCK_BOUNDARY
    df_log = df_log[df_log['event_type'] != TIME_BLOCK_BOUNDARY].reset_index(drop=True)
    df_log = preprocess_dflog(df_log)
    
    ur, rtr, itr = get_rates(df_log)
    
    result = {
        "abandonment_rate": get_abandonment_rate(df_log),
        "utilization_rate": ur,
        "relocation_time_ratio": rtr,
        "idle_time_ratio": itr,
        "temporal_max_abandonment_rate": get_temporal_max_abandonment_rate(df_log),
        "regional_max_abandonment_rate": get_regional_max_abandonment_rate(df_log),
        "maximal_average_queue_length": get_maximal_average_queue_length(df_log)
    }
    return result