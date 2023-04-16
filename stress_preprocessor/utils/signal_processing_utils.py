import datetime
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd
from scipy.signal import savgol_filter

from stress_preprocessor.utils.preprocessing_utils import prefix_columns

# TODO: Check if input is not list and added to a list for single dataframes input to be used for inference

def extract_ecg_features(dfs: List[pd.DataFrame], sr_hz_list: List[float], ecg_col: str, participant: str,
                         scenario_col: str, mode_col: str, modes: Dict[str, str], graph_path: str) -> List[
    pd.DataFrame]:
    """
    Extracts ECG features from a list of dataframes.

    Args:
        dfs: A list of pandas dataframes containing ECG data.
        sr_hz_list: A list of sampling rates, in Hz, corresponding to the dataframes in `dfs`.
        ecg_col: The name of the ECG column in each dataframe.
        participant: The participant identifier string.
        scenario_col: The name of the scenario column in each dataframe.
        mode_col: The name of the mode column in each dataframe.
        modes: A dictionary mapping mode codes to mode names.
        graph_path: The path to save the ECG feature plots.

    Returns:
        A list of pandas dataframes containing the extracted ECG features.
    """
    ecg_feats_dfs = []
    for idx, df in enumerate(dfs):
        scenario_id = df[scenario_col].iloc[0]
        mode = modes[df[mode_col].iloc[0]]
        ecg_signal = df[ecg_col]
        # Compute ECG features using NeuroKit2
        ecg_processed, info = nk.ecg_process(ecg_signal, sampling_rate=int(sr_hz_list[idx]))

        # Plot features
        nk.ecg_plot(ecg_processed, info, int(sr_hz_list[idx]))
        fig = plt.gcf()
        # Create the filename and the full save path to save the plot
        filename = f"ECG_FEATS_SUBJ_{participant}_SCEN_{scenario_id}_MODE_{mode}.png"
        full_path = os.path.join(graph_path, filename)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')

        # Add prefixes to column names, to differentiate between ECG and EDA feats
        ecg_processed = prefix_columns(ecg_processed, 'ECG_')

        ecg_feats_dfs.append(ecg_processed)

        # TODO: merge ECG/EDA functions into a single function with dynamic arg for dignal_type

    return ecg_feats_dfs


def extract_eda_features(dfs: List[pd.DataFrame], sr_hz_list: List[float], eda_col: str, participant: str,
                         scenario_col: str, mode_col: str, modes: Dict[str, str], graph_path: str) -> List[
    pd.DataFrame]:
    """
    Extracts EDA features from a list of dataframes.

    Args:
        dfs: A list of pandas dataframes containing EDA data.
        sr_hz_list: A list of sampling rates, in Hz, corresponding to the dataframes in `dfs`.
        eda_col: The name of the EDA column in each dataframe.
        participant: The participant identifier string.
        scenario_col: The name of the scenario column in each dataframe.
        mode_col: The name of the mode column in each dataframe.
        modes: A dictionary mapping mode codes to mode names.
        graph_path: The path to save the EDA feature plots.

    Returns:
        A list of pandas dataframes containing the extracted EDA features.
    """
    eda_feats_dfs = []
    for idx, df in enumerate(dfs):
        scenario_id = df[scenario_col].iloc[0]
        mode = modes[df[mode_col].iloc[0]]
        eda_signal = df[eda_col]
        # Compute EDA features using NeuroKit2
        eda_processed, info = nk.eda_process(eda_signal, sampling_rate=int(sr_hz_list[idx]))

        # Plot features
        nk.eda_plot(eda_processed, sampling_rate=int(sr_hz_list[idx]))
        fig = plt.gcf()
        # Create the filename and the full save path to save the plot
        filename = f"EDA_FEATS_SUBJ_{participant}_SCEN_{scenario_id}_MODE_{mode}.png"
        full_path = os.path.join(graph_path, filename)
        fig.savefig(full_path, dpi=300, bbox_inches='tight')

        # Add prefixes to column names, to differentiate between ECG and EDA feats
        eda_processed = prefix_columns(eda_processed, 'EDA_')

        eda_feats_dfs.append(eda_processed)

    return eda_feats_dfs


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


def remove_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the baseline signal from the ECG data using a Savitzky-Golay filter.

    Args:
        df: Pandas DataFrame with columns "ts_corr" (timestamp), "data" (ECG measurement),
            "prob_id" (participant id), and "scenario" (scenario category).

    Returns:
        Pandas DataFrame with baseline-corrected ECG data.
    """
    # Group the data by participant id and scenario category
    grouped = df.groupby(['prob_id', 'scenario'])

    # Loop through each group and remove the baseline signal
    for name, group in grouped:
        ecg_data = group['data'].values

        # Apply a Savitzky-Golay filter to the ECG data to remove the baseline signal
        ecg_data = ecg_data - savgol_filter(ecg_data, 201, 3)

        # Replace the original ECG data with the baseline-corrected data
        df.loc[group.index, 'data'] = ecg_data

    return df
