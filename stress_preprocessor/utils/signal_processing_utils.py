import logging
import os
from typing import List, Dict, Union
import warnings

import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

from stress_preprocessor.utils.preprocessing_utils import prefix_columns


def extract_neuro_features(dfs: Union[pd.DataFrame, List[pd.DataFrame]], sr_hz_list: List[float], signal_col: str,
                           target_col: str, time_col: str, participant: str, scenario_col: str, mode_col: str,
                           modes: Dict[str, str], graph_path: str, signal_type: str, offline) -> List[pd.DataFrame]:
    """
    Extracts ECG or EDA features from a list of dataframes.

    Args:
        dfs: A list of pandas dataframes containing ECG data.
        sr_hz_list: A list of sampling rates, in Hz, corresponding to the dataframes in `dfs`.
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
        mode = modes[df[mode_col].iloc[0]]
        signal = df[signal_col]

        if signal_type == 'ECG':
            # Compute ECG features using NeuroKit2
            with warnings.catch_warnings(record=True) as caught_warnings:
                neuro_processed, info = nk.ecg_process(signal, sampling_rate=int(sr_hz_list[idx]))
                if offline:
                    nk.ecg_plot(neuro_processed, info, int(sr_hz_list[idx]))
        elif signal_type == "EDA":
            # Compute EDA features using NeuroKit2
            with warnings.catch_warnings(record=True) as caught_warnings:
                neuro_processed, info = nk.eda_process(signal, sampling_rate=int(sr_hz_list[idx]))
                if offline:
                    nk.eda_plot(neuro_processed, int(sr_hz_list[idx]))
        else:
            raise ValueError("Invalid value for signal_type. Must be 'ECG' or 'EDA'.")

        if offline:
            # Save neuro features plots
            subj_dir = os.path.join(graph_path, f"SUBJ_{participant}")
            if not os.path.exists(subj_dir):
                os.makedirs(subj_dir)

            fig = plt.gcf()
            # Create the filename and the full save path to save the plot
            filename = f"{signal_type}_FEATS_SUBJ_{participant}_SCEN_{scenario_id}_MODE_{mode}.png"
            full_path = os.path.join(graph_path, filename)
            fig.savefig(full_path, dpi=300, bbox_inches='tight')

        # Add prefixes to column names, to differentiate between ECG and EDA feats
        neuro_processed = prefix_columns(neuro_processed, f'{signal_type}_')

        # Add the rest of the features
        df_raw_part = df.loc[:, [time_col, scenario_col, mode_col, target_col]]

        neuro_processed = pd.concat([df_raw_part, neuro_processed],
                                    axis=1)

        neuro_feats_dfs.append(neuro_processed)

        # Log any warnings that were caught
        for warning in caught_warnings:
            logging.warning(str(warning.message))

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


