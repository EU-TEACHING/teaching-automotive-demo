import logging
import os
from typing import List, Dict, Union
import warnings

import numpy
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

from stress_preprocessor.utils.preprocessing_utils import prefix_columns


def extract_neuro_features(
    dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    sr_hz: int,
    signal_col: str,
    target_col: str,
    time_col: str,
    participant: str,
    scenario_col: str,
    mode_col: str,
    modes: Dict[str, str],
    graph_path: str,
    signal_type: str,
    offline,
) -> List[pd.DataFrame]:
    """
    Extracts ECG or EDA features from a list of dataframes.

    Args:
        dfs: A list of pandas dataframes containing ECG data.
        sr_hz: Sampling rate, in Hz
        signal_col: The name of the ECG or EDA/GSR column in each dataframe.
        target_col: The name of the prediction target column, e.g., stress.
        time_col: The timestamp column
        participant: The participant identifier string.
        scenario_col: The name of the scenario column in each dataframe.
        mode_col: The name of the scenario mode column in each dataframe.
        modes: A dictionary mapping mode codes to mode names, e.g., eco, sport.
        graph_path: The path to save the ECG feature plots.
        signal_type: "ECG" or "EDA"
        offline: True enables Neurokit plots for offline training. False during online inference

    Returns:
        A list of pandas dataframes containing the extracted ECG features.
    """
    if not isinstance(dfs, list):
        # If the input is not a list, convert it to a list with a single element
        dfs = [dfs]

    neuro_feats_dfs = []
    for idx, df in enumerate(dfs):
        scenario_id = df[scenario_col].iloc[0]
        mode = modes[f"{df[mode_col].iloc[0]}"]
        signal = df[signal_col]

        if signal_type == "ECG":
            # Compute ECG features using NeuroKit2 (Warnings are caught here!)
            with warnings.catch_warnings(record=True) as caught_warnings:
                try:
                    neuro_processed, info = nk.ecg_process(
                        signal, sampling_rate=int(sr_hz)
                    )
                    if offline:
                        nk.ecg_plot(neuro_processed, info, int(sr_hz))
                except Exception as e:
                    logging.error(
                        f"An error occured and signal features weren't extracted for Scenario {scenario_id} and mode {mode}."
                    )
                    neuro_processed = pd.DataFrame()
                    ecg_cols = ["ECG_Raw", "ECG_Clean"]
                    for col in ecg_cols:
                        neuro_processed[col] = numpy.nan
        elif signal_type == "EDA":
            try:
                # Compute EDA features using NeuroKit2
                with warnings.catch_warnings(record=True) as caught_warnings:
                    neuro_processed, info = nk.eda_process(
                        signal, sampling_rate=int(sr_hz)
                    )
                    neuro_processed["index_time"] = numpy.arange(len(df))
                    if offline:
                        # nk.eda_plot(neuro_processed, int(sr_hz))
                        # lines = neuro_processed[["EDA_Raw", "EDA_Clean"]].plot.line()
                        # Create figure and subplots
                        fig, (ax1, ax2, ax3) = plt.subplots(
                            3, 1, figsize=(10, 12), sharex=True
                        )

                        # Plot 1st plot: EDA_Raw and EDA_Clean
                        ax1.plot(
                            neuro_processed["index_time"],
                            neuro_processed["EDA_Raw"],
                            color="blue",
                            linewidth=2,
                            label="EDA_Raw",
                        )
                        ax1.plot(
                            neuro_processed["index_time"],
                            neuro_processed["EDA_Clean"],
                            color="red",
                            linewidth=2,
                            label="EDA_Clean",
                        )
                        ax1.set_ylabel("EDA (arbitrary units)")
                        ax1.set_title("EDA_Raw vs EDA_Clean")
                        ax1.legend()

                        # Plot 2nd plot: SCR_Onsets, SCR_Peaks, SCR_Recovery
                        peaks = neuro_processed[neuro_processed["SCR_Peaks"] == 1]
                        onsets = neuro_processed[neuro_processed["SCR_Onsets"] == 1]
                        recovery = neuro_processed[neuro_processed["SCR_Recovery"] == 1]

                        ax2.plot(
                            neuro_processed["index_time"],
                            neuro_processed["EDA_Clean"],
                            color="blue",
                            linewidth=2,
                            label="EDA_Clean",
                        )
                        ax2.scatter(
                            peaks.index,
                            peaks["EDA_Clean"],
                            color="red",
                            label="SCR_Peaks",
                        )
                        ax2.scatter(
                            onsets.index,
                            onsets["EDA_Clean"],
                            color="blue",
                            label="SCR_Onsets",
                        )
                        ax2.scatter(
                            recovery.index,
                            recovery["EDA_Clean"],
                            color="green",
                            label="SCR_Recovery",
                        )
                        ax2.set_ylabel("EDA (arbitrary units)")
                        ax2.set_title("SCR_Onsets, SCR_Peaks, SCR_Recovary")
                        ax2.legend()

                        # Plot 3rd plot: EDA_Tonic
                        ax3.plot(
                            neuro_processed["index_time"],
                            neuro_processed[["EDA_Tonic"]],
                            color="magenta",
                            linewidth=2,
                            label="EDA_Tonic",
                        )
                        ax3.set_xlabel("Time (s)")
                        ax3.set_ylabel("EDA Tonic (arbitrary units)")
                        ax3.set_title("EDA_Tonic")
                        ax3.legend()

                        # Adjust spacing between subplots
                        plt.subplots_adjust(hspace=0.3)

                        neuro_processed.drop(columns=["index_time"], inplace=True)
            except Exception as e:
                neuro_processed = pd.DataFrame()
                eda_cols = ["EDA_Raw", "EDA_Clean"]
                for col in eda_cols:
                    neuro_processed[col] = numpy.nan

        else:
            raise ValueError("Invalid value for signal_type. Must be 'ECG' or 'EDA'.")

        if offline:
            try:
                # Save neuro features plots
                subj_dir = os.path.join(
                    graph_path, f"SUBJ_{participant}_neurokit_features"
                )
                if not os.path.exists(subj_dir):
                    os.makedirs(subj_dir)

                fig = plt.gcf()
                # Create the filename and the full save path to save the plot
                filename = f"{signal_type}_FEATS_SUBJ_{participant}_SCEN_{scenario_id}_MODE_{mode}.png"
                full_path = os.path.join(subj_dir, filename)
                fig.savefig(full_path, dpi=300, bbox_inches="tight")
            except Exception as e:
                logging.error(
                    f"An error occured and no figures were saved for Scenario {scenario_id} and mode {mode}."
                )

        # Add prefixes to column names, to differentiate between ECG and EDA feats
        neuro_processed = prefix_columns(neuro_processed, f"{signal_type}_")

        # Add the rest of the features
        df_raw_part = df.loc[
            :, [time_col, scenario_col, mode_col, target_col, "Stress_Event"]
        ]

        neuro_processed = pd.concat([df_raw_part, neuro_processed], axis=1)

        neuro_feats_dfs.append(neuro_processed)

        # Log any warnings that were caught
        for warning in caught_warnings:
            logging.warning("Neurokit2: " + str(warning.message))

    return neuro_feats_dfs


def get_sampling_rate(dfs: List[pd.DataFrame], time_col: str) -> List[float]:
    """
    Estimates the sampling rate in Hz for each dataframe in a list of dataframes.

    Args:
        dfs: A list of pandas dataframes.
        time_col: The name of the timestamp column in the dataframes.

    Returns:
        A list of estimated sampling rates in Hz for each dataframe.

    Raises:
        ValueError: If the timestamp column is not found in a dataframe.
    """
    sampling_rates = []
    for df in dfs:
        if time_col not in df.columns:
            raise ValueError(f"Timestamp column '{time_col}' not found in dataframe.")
        time_series = df[time_col]
        time_interval = time_series.diff().mean()
        sampling_rate = 1 / time_interval
        sampling_rates.append(sampling_rate)
    return sampling_rates


# TODO: this is based on previous version, needs to be very refactored, if upsampling is absolutely necessary
def resample_dataframe(
    df: pd.DataFrame,
    timestamp_col: str,
    value_col: str,
    group_col: str,
    participant_col: str,
    target_freq: str,
) -> pd.DataFrame:
    """Resample a time series DataFrame to a specified frequency.

    Args:
        df (pandas.DataFrame): The DataFrame to resample.
        timestamp_col (str): The name of the column containing the timestamp data.
        value_col (str): The name of the column containing the data to be resampled.
        target_freq (str): The target frequency to which the DataFrame should be resampled.
            Must be a valid pandas frequency string (e.g., '1H', '1D', '1W').

    Returns:
        pandas.DataFrame: The resampled DataFrame with a DateTimeIndex and missing values filled forward.

    Raises:
        ValueError: If the specified timestamp column is not a valid datetime column.

    """
    # Set timestamp column as DataFrame index
    df.set_index(timestamp_col, inplace=True)

    # Resample DataFrame to target frequency
    resampled_df = df.resample(target_freq).agg(
        {
            participant_col: "first",
            # TODO add rest of the columns here
            group_col: "first",
            value_col: "mean",
        }
    )

    # Reset the index to timestamp_col column name
    resampled_df.reset_index(inplace=True)
    resampled_df.rename(columns={"index": "timestamp_col"}, inplace=True)

    return resampled_df
