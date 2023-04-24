import datetime
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_timeseries_raw(df_list, value_col: str, time_col: str, scenario_col: str, mode_col: str, modes: Dict[str, str],
                        participant_id: str, save_path: str, signal_type: str, figsize=(10, 12)):
    """
    Plots a timeseries of the input dataframe with one subplot per scenario-mode combination.

    Args:

        df_list (list): A list of pandas DataFrames containing the timeseries data for each scenario-mode combination.
        value_col (str): The name of the column containing the data values to plot.
        time_col (str): The name of the column containing the timestamps.
        scenario_col (str): The name of the column containing the scenario IDs.
        mode_col (str): The name of the column containing the mode IDs.
        modes: A dictionary mapping mode codes to mode names, e.g., eco, sport.
        participant_id (str): A string representing the participant ID for the plot title.
        save_path (str): path to save graphs
        signal_type (str): e.g., ECG, GSR etc
        figsize (tuple): The size of the plot.

    Returns:

        plot (matplotlib plot): A plot of the timeseries.
    """
    num_plots = len(df_list)
    num_cols = 1
    num_rows = num_plots

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True)
    fig.suptitle(f"Participant {participant_id} {signal_type}", fontsize=16)

    for i, df in enumerate(df_list):
        scenario = df[scenario_col].iloc[0]
        mode = modes[f"{df[mode_col].iloc[0]}"]

        axs[i].plot(df[time_col], df[value_col])
        axs[i].set_title(f"Scenario: {scenario}, Mode: {mode}")
        axs[i].set_ylabel(value_col)

    axs[-1].set_xlabel(time_col + ' (s)')

    folder_name = f"SUBJ_{participant_id}_raw_signals"
    if not os.path.exists(save_path + '/' + folder_name):
        os.makedirs(save_path + '/' + folder_name)

    fig.savefig(f"{save_path}/{folder_name}/{participant_id}_{signal_type}.png")
    # plt.show()
