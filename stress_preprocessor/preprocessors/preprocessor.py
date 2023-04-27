import logging
import os

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple
import warnings

from stress_preprocessor.utils.preprocessing_utils import (
    clean_duplicates,
    validate_timestamps,
    compute_diff,
    impute_null,
    mapper,
)
from stress_preprocessor.utils.signal_processing_utils import (
    extract_neuro_features,
    get_sampling_rate,
)
from stress_preprocessor.utils.visualization_utils import plot_timeseries_raw


class StressPreprocessor:
    def __init__(self, config, online: bool = False, **kwargs):
        self.subj_id = None
        self.config = config
        self.logger = logging.getLogger(__name__)
        if online:
            self.baseline_path = kwargs["baseline_path"]
            baseline_data = pd.read_csv(self.baseline_path, header=1)
            baseline_data = baseline_data.drop(0)
            self.last_timestamp = float((baseline_data["Time"].iloc[-1]))
            self.window = baseline_data.to_dict("records")[-kwargs["buffer_size"] :]
            self.last_returned = len(self.window)
            self.preprocessor = StressPreprocessor(self.config)

    def float_to_integer(self, dfs):
        formatted_types_dfs = []
        for df in dfs:
            df = df.astype(
                {
                    self.config.error_col: "Int32",
                    self.config.scenario_col: "Int32",
                    self.config.mode_col: "Int32",
                    self.config.participant_col: "Int32",
                }
            )
            df.reset_index(inplace=True, drop=True)
            formatted_types_dfs.append(df)
        return formatted_types_dfs

    def load_data(
        self,
        subpaths: Dict[str, str],
        baseline_subpath: str,
        subj_path: str,
        subj_id: str,
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Offline Load data from a subject's directory into a list of pandas DataFrames.

        Args:
            subpaths: A dictionary of file names for each scenario/mode dataset.
            baseline_subpath: The file name for the baseline scenario.
            subj_path: The path to the directory containing the subject's raw data files.
            subj_id: The participant's unique identification.

        Returns:
            The baseline dataframe and a list of pandas DataFrames, one for each scenario/mode dataset.
        """
        start = time.time()

        logging.info(f"Loading data from {subj_path} ...")
        # Check if the subject's directory exists
        if not os.path.exists(subj_path):
            logging.error(f"Directory not found: {subj_path}")
            raise FileNotFoundError(f"Directory not found: {subj_path}")

        baseline_df_list = []
        if baseline_subpath:
            baseline_subpath = baseline_subpath.replace("SUBJ_XX", f"SUBJ_{subj_id}")
            filepath = os.path.join(subj_path, baseline_subpath)
            # Check if the file exists
            if not os.path.exists(filepath):
                logging.error(f"Baseline data not found: {filepath}")
                raise FileNotFoundError(f"Baseline data not found: {filepath}")

            try:
                baseline_df = pd.read_csv(
                    filepath
                )  # , header=1) # Commented after manual cleaning
                # baseline_df = baseline_df.drop(0) # Commented after manual cleaning
                baseline_df = baseline_df[
                    [
                        self.config.time_col,
                        self.config.ecg_col,
                        self.config.gsr_col,
                        self.config.target_col,
                        self.config.error_col,
                        self.config.scenario_col,
                        self.config.mode_col,
                        self.config.participant_col,
                    ]
                ]
                baseline_df = baseline_df.astype(
                    {
                        self.config.time_col: "float32",
                        self.config.ecg_col: "float32",
                        self.config.gsr_col: "float32",
                        self.config.target_col: "float32",
                        self.config.error_col: "float32",
                        self.config.scenario_col: "float32",
                        self.config.mode_col: "float32",
                        self.config.participant_col: "float32",
                    }
                )
                baseline_df.reset_index(inplace=True, drop=True)
                baseline_df_list = [baseline_df]
                baseline_df_list = self.float_to_integer(baseline_df_list)
                logging.info(f"Successfully loaded {filepath}")

            except Exception as e:
                logging.error(f"Error loading file: {filepath}\n{str(e)}")
                raise ValueError(f"Error loading file: {filepath}\n{str(e)}")

        dfs = []
        for key, filename in subpaths.items():
            filename = filename.replace("SUBJ_XX", f"SUBJ_{subj_id}")
            filepath = os.path.join(subj_path, filename)

            # Check if the file exists
            if not os.path.exists(filepath):
                logging.warning(f"File not found: {filepath}")
                continue

            try:
                df = pd.read_csv(
                    filepath
                )  # , header=1) # Commented after manual cleaning
                # df = df.drop(0) # Commented after manual cleaning

                if self.config.error_col not in df.columns:
                    df[self.config.error_col] = 0
                df = df[
                    [
                        self.config.time_col,
                        self.config.ecg_col,
                        self.config.gsr_col,
                        self.config.target_col,
                        self.config.error_col,
                        self.config.scenario_col,
                        self.config.mode_col,
                        self.config.participant_col,
                    ]
                ]
                df = df.astype(
                    {
                        self.config.time_col: "float32",
                        self.config.ecg_col: "float32",
                        self.config.gsr_col: "float32",
                        self.config.target_col: "float32",
                        self.config.error_col: "float32",
                        self.config.scenario_col: "float32",
                        self.config.mode_col: "float32",
                        self.config.participant_col: "float32",
                    }
                )
                df.reset_index(inplace=True, drop=True)

                dfs.append(df)
                dfs = self.float_to_integer(dfs)
                logging.info(f"Successfully loaded {filepath}")

            except Exception as e:
                logging.error(f"Error loading file: {filepath}\n{str(e)}")
                continue

        stop = time.time()
        logging.info(f"Data loading latency (secs): {stop - start}")

        return baseline_df_list, dfs

    def load_data_online(self, window):
        # Dictionary to dataframe
        df = pd.DataFrame(window)

        df = df[
            [
                self.config.time_col,
                self.config.ecg_col,
                self.config.gsr_col,
                self.config.target_col,
                self.config.error_col,
                self.config.scenario_col,
                self.config.mode_col,
                self.config.participant_col,
            ]
        ]
        df = df.astype(
            {
                self.config.time_col: "float32",
                self.config.ecg_col: "float32",
                self.config.gsr_col: "float32",
                self.config.target_col: "float32",
                self.config.error_col: "float32",
                self.config.scenario_col: "float32",
                self.config.mode_col: "float32",
                self.config.participant_col: "float32",
            }
        )
        df.reset_index(inplace=True, drop=True)
        return [df]

    def clean_and_validate(
        self, baseline_df_list, dfs: List[pd.DataFrame]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Preprocesses a list of dataframes.

        Args:
            baseline_df_list: List containing the baseline dataframe.
            dfs: A list of pandas dataframes to preprocess.

        Returns:
            A list of preprocessed pandas dataframes.
        """
        start = time.time()
        imputed_baseline_df_list = []
        if baseline_df_list:
            # Remove duplicate rows from each dataframe in the input list.
            no_dup_baseline_df_list = clean_duplicates(baseline_df_list)
            # Validate the remaining timestamps to ensure that they are uniformly spaced.
            val_baseline_df_list = validate_timestamps(
                no_dup_baseline_df_list,
                self.config.time_col,
                self.config.sampling_rate_hz,
            )

            # Impute missing values: bfill and ffil for categorical, interpolation for numerical (neurokit default is ffill)
            # Before imputation, null values are added in error marked rows
            imputed_baseline_df_list = impute_null(
                val_baseline_df_list,
                self.config.error_col,
                self.config.ecg_col,
                self.config.gsr_col,
                self.config.target_col,
                self.config.scenario_col,
                self.config.mode_col,
                self.config.participant_col,
            )
        # Remove duplicate rows from each dataframe in the input list.
        no_dup_dfs = clean_duplicates(dfs)
        # Validate the remaining timestamps to ensure that they are uniformly spaced.
        val_dfs = validate_timestamps(
            no_dup_dfs, self.config.time_col, self.config.sampling_rate_hz
        )

        # Impute missing values: bfill and ffil for categorical, interpolation for numerical (neurokit default is ffill)
        # Before imputation, null values are added in error marked rows
        imputed_dfs = impute_null(
            val_dfs,
            self.config.error_col,
            self.config.ecg_col,
            self.config.gsr_col,
            self.config.target_col,
            self.config.scenario_col,
            self.config.mode_col,
            self.config.participant_col,
        )

        stop = time.time()
        logging.info(
            f"Data cleaning and timestamp validation latency (secs): {stop - start}"
        )

        return imputed_baseline_df_list, imputed_dfs

    def visualize(self, dfs: List[pd.DataFrame]):
        # Plot ECG raw
        plot_timeseries_raw(
            dfs,
            self.config.ecg_col,
            self.config.time_col,
            self.config.scenario_col,
            self.config.mode_col,
            self.config.modes,
            self.subj_id,
            self.config.graph_path,
            "ECG",
        )
        # Plot GSR raw
        plot_timeseries_raw(
            dfs,
            self.config.gsr_col,
            self.config.time_col,
            self.config.scenario_col,
            self.config.mode_col,
            self.config.modes,
            self.subj_id,
            self.config.graph_path,
            "GSR",
        )

    def extract_features(
        self, dfs: List[pd.DataFrame], offline=True
    ) -> List[pd.DataFrame]:
        """
        Extracts physiological features and first-order differences features from a list of dataframes.

        Args:
            dfs: A list of pandas dataframes containing physiological signals.
            offline: True enables Neurokit plotting functions

        Returns:
            A list of pandas dataframes containing extracted and original features.
        """

        start = time.time()

        # sr_list = get_sampling_rate(dfs, self.config.time_col)

        # Extract ECG features from each dataframe in the input list.
        ecg_feats_dfs = extract_neuro_features(
            dfs,
            self.config.sampling_rate_hz,
            self.config.ecg_col,
            self.config.target_col,
            self.config.time_col,
            self.subj_id,
            self.config.scenario_col,
            self.config.mode_col,
            self.config.modes,
            self.config.graph_path,
            "ECG",
            offline,
        )

        # Extract EDA features from each dataframe in the input list.
        eda_feats_dfs = extract_neuro_features(
            dfs,
            self.config.sampling_rate_hz,
            self.config.gsr_col,
            self.config.target_col,
            self.config.time_col,
            self.subj_id,
            self.config.scenario_col,
            self.config.mode_col,
            self.config.modes,
            self.config.graph_path,
            "EDA",
            offline,
        )

        new_feats_dfs = []
        for ecg_feats_df, eda_feats_df in zip(ecg_feats_dfs, eda_feats_dfs):
            # Concatenate ECG and EDA features
            concat_df = pd.concat([ecg_feats_df, eda_feats_df], axis=1)

            # Remove duplicated columns since both ECG and EDA dfs contained some of the initial feats, e.g., ScenarioID
            concat_df = concat_df.loc[:, ~concat_df.columns.duplicated()]

            new_feats_dfs.append(concat_df)

        # Compute first-order differences between consecutive values as additional features
        new_feats_dfs = compute_diff(new_feats_dfs, self.config.fod_feats)

        stop = time.time()
        logging.info(f"Feature extraction latency (secs): {stop - start}")
        return new_feats_dfs

    def save_preprocessed_data(
        self,
        baseline_df_list: List[pd.DataFrame],
        dfs: List[pd.DataFrame],
        subj_id: str,
    ) -> None:
        """
        Save the preprocessed data to the "processed_data_path" defined in config. If "save_single_df" is True,
        all scenarios/modes will be saved in a single file, with time counter reset to 0 for each scenario/mode.

        Args:
            dfs: A list of pandas DataFrames containing the preprocessed data to be saved.
            baseline_df_list: List containing the preprocessed baseline Dataframe to be saved.
            subj_id: A string representing the subject ID for which the data is being saved.

        Returns:
            None
        """
        subj_dir = os.path.join(self.config.processed_data_path, f"SUBJ_{subj_id}")
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        filename = None

        if baseline_df_list:
            filename = f"SUBJ_{subj_id}_SCEN_00_MODE_FreeDriving.csv"
            baseline_df = baseline_df_list[0]
            baseline_df.to_csv(os.path.join(subj_dir, filename), index=False)

        if self.config.save_single_df:
            final_df = pd.concat(dfs)
            filename = f"SUBJ_{subj_id}_ALL_SCENARIOS.csv"
            final_df.to_csv(os.path.join(subj_dir, filename), index=False)
        else:
            for df in dfs:
                scenario_id = df.loc[0, self.config.scenario_col]
                mode = df.loc[0, self.config.mode_col]
                filename = f"SCEN_{scenario_id}_MODE_{mode}.csv"
                df.to_csv(os.path.join(subj_dir, filename), index=False)

        logging.info(f"Preprocessed data saved at {os.path.join(subj_dir, filename)}")

    def run(
        self,
        subpaths: Dict[str, str],
        baseline_subpath: str,
        subject_path: str,
        subj_id: str,
    ) -> None:
        """
        Offline Preprocessing pipeline: data loading, cleaning and validation, feature extraction, store preprocessed.

        Args:
            subpaths: A dictionary of file names for each scenario/mode dataset.
            subject_path: The path to the directory containing the subject's raw data files.
            baseline_subpath: The file name for the baseline scenario.
            subj_id: The participant's unique identification

        Returns:
            None
        """
        self.subj_id = subj_id
        baseline_df_list, dfs = self.load_data(
            subpaths, baseline_subpath, subject_path, subj_id
        )

        self.visualize(dfs)
        prep_baseline_df_list, prep_dfs = self.clean_and_validate(baseline_df_list, dfs)
        new_feats_dfs = self.extract_features(prep_dfs)
        new_feats_baseline_df_list = self.extract_features(baseline_df_list)

        self.save_preprocessed_data(new_feats_baseline_df_list, new_feats_dfs, subj_id)

    def online_run(self, stream_dict: dict) -> np.array:
        start = time.time()
        stream_dict["ErrorCount"] = 0
        stream_dict["ScenarioID"] = 0
        stream_dict["Maneuvre_ID"] = 0
        stream_dict["Subject_ID"] = 0
        stream_dict_mapped = mapper(stream_dict)
        stream_dict_mapped["Time"] = (
            float(stream_dict_mapped["Time"]) + self.last_timestamp
        )
        self.window.append(stream_dict_mapped)
        self.window = self.window[1:]
        self.last_returned -= 1
        print(f"last_returned: {self.last_returned}")
        if self.last_returned <= len(self.window) - 100:
            dfs = self.load_data_online(self.window)
            _, prep_dfs = self.clean_and_validate(None, dfs)
            prep_dfs = self.float_to_integer(prep_dfs)
            proc_df = self.extract_features(prep_dfs, offline=False)[0]
            eda = proc_df["EDA_Clean"].values
            ecg = proc_df["ECG_Rate"].values
            to_return = np.stack([eda, ecg], axis=1)
            to_return = to_return[self.last_returned :]
            self.last_returned = len(self.window)
            print(f"length: {len(to_return)}")
            return to_return
        stop = time.time()
        logging.info(f"Overall latency (secs): {stop - start}")
