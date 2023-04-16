import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_timeseries_raw(df, value_col: str, time_col: str, group_col: str, participant_id: str, save_path: str,
                        signal_type: str, figsize=(10, 12)):
    """
    Plots a timeseries of the input dataframe with one subplot per scenario.

    Args:

        df (pandas.DataFrame): A pandas DataFrame containing the timeseries data.
        value_col (str): The name of the column containing the data values to plot.
        time_col (str): The name of the column containing the timestamps.
        group_col (str): The name of the column containing the scenario IDs.
        participant_id (str): A string representing the participant ID for the plot title.
        save_path (str): path to save graphs
        signal_type (str): e.g., ECG, GSR etc
        figsize (tuple): The size of the plot.

    Returns:

        plot (matplotlib plot): A plot of the timeseries.
    """
    # Convert the ts_corr column back to timedelta format
    if pd.api.types.is_object_dtype(df[time_col]):
        df[time_col] = df[time_col].astype('int64')
        df[time_col] = pd.to_timedelta(df[time_col])

    # Create a list of the unique scenario IDs
    scenarios = df[group_col].unique()

    # Calculate the number of rows and columns needed for the plot
    num_plots = len(scenarios)
    num_cols = 1
    num_rows = num_plots

    # Set up the plot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)

    # Set the plot title
    if participant_id:
        fig.suptitle(f"Participant {participant_id} {signal_type} Data")

    # Iterate over the scenarios and plot each one
    for i, scenario in enumerate(scenarios):
        # Get the data for the current scenario
        scenario_data = df[df[group_col] == scenario]

        # Plot the data
        axes[i].plot(scenario_data[time_col].dt.total_seconds(), scenario_data[value_col])

        # Set the plot labels and tick marks
        axes[i].set_title(f"Scenario {scenario}")
        axes[i].set_xlabel("Time (seconds)")
        axes[i].set_ylabel(value_col)
        axes[i].set_xticks(np.arange(0, scenario_data[time_col].dt.total_seconds().max(), 10))

    # # Adjust the layout of the plot
    # plt.tight_layout()
    # plt.show()
    # Get the current date and time

    now = datetime.datetime.now()

    # Format the date and time string
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")

    # Create the filename
    filename = f"{participant_id}_{signal_type}_{date_str}_{time_str}.png"

    # Create the full save path
    full_path = os.path.join(save_path, filename)

    fig.savefig(full_path, dpi=300, bbox_inches='tight')

    return fig
