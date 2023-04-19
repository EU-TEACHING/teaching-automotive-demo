import logging
import os
import pandas as pd
import time
from typing import Dict, List
import warnings

from stress_preprocessor.utils.preprocessing_utils import clean_duplicates, validate_timestamps, compute_diff, \
    impute_null
from stress_preprocessor.utils.signal_processing_utils import extract_neuro_features, get_sampling_rate
from stress_preprocessor.utils.visualization_utils import plot_timeseries_raw


class StressPreprocessor:
    def __init__(self, config):
        self.subj_id = None
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_data(self, subpaths: Dict[str, str], subj_path: str) -> List[pd.DataFrame]:
        """
        Offline Load data from a subject's directory into a list of pandas DataFrames.

        Args:
            subpaths: A dictionary of file names for each scenario/mode dataset.
            subj_path: The path to the directory containing the subject's raw data files.

        Returns:
            A list of pandas DataFrames, one for each scenario/mode dataset.
        """
        start = time.time()

        logging.info(f"Loading data from {subj_path} ...")
        # Check if the subject's directory exists
        if not os.path.exists(subj_path):
            logging.error(f"Directory not found: {subj_path}")
            raise FileNotFoundError(f"Directory not found: {subj_path}")

        dfs = []
        for key, filename in subpaths.items():
            filepath = os.path.join(subj_path, filename)

            # Check if the file exists
            if not os.path.exists(filepath):
                logging.warning(f"File not found: {filepath}")
                continue

            try:
                df = pd.read_csv(filepath, header=1, sep=';')
                df = df.drop(0)
                df = df.astype({self.config.time_col: 'float32', self.config.ecg_col: 'float32',
                                self.config.gsr_col: 'float32', self.config.target_col: 'float32',
                                self.config.error_col: 'float32'})
                df.reset_index(inplace=True, drop=True)
                dfs.append(df)
                logging.info(f"Successfully loaded {filepath}")

            except Exception as e:
                logging.error(f"Error loading file: {filepath}\n{str(e)}")
                continue

        stop = time.time()
        logging.info(f"Data loading latency (secs): {stop - start}")

        return dfs

    def load_data_online(self, array):
        # Array to df
        df = pd.DataFrame(array, columns=self.config.online.array_schema)
        df = df.astype({self.config.time_col: 'float32', self.config.ecg_col: 'float32', self.config.gsr_col: 'float32',
                        self.config.target_col: 'float32', self.config.error_col: 'float32'})
        return [df]

    def clean_and_validate(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Preprocesses a list of dataframes.

        Args:
            dfs: A list of pandas dataframes to preprocess.

        Returns:
            A list of preprocessed pandas dataframes.
        """
        start = time.time()
        # Remove duplicate rows from each dataframe in the input list.
        no_dup_dfs = clean_duplicates(dfs)
        # Validate the remaining timestamps to ensure that they are uniformly spaced.
        val_dfs = validate_timestamps(no_dup_dfs, self.config.time_col, self.config.sampling_rate_hz)

        # Impute missing values: bfill and ffil for categorical, interpolation for numerical (neurokit default is ffill)
        # Before imputation, null values are added in error marked rows
        imputed_dfs = impute_null(val_dfs, self.config.error_col, self.config.ecg_col, self.config.gsr_col,
                                  self.config.target_col, self.config.scenario_col, self.config.mode_col,
                                  self.config.participant_col)

        stop = time.time()
        logging.info(f"Data cleaning and timestamp validation latency (secs): {stop - start}")

        return imputed_dfs

    def visualize(self):
        pass

    def extract_features(self, dfs: List[pd.DataFrame], offline=True) -> List[pd.DataFrame]:
        """
        Extracts physiological features and first-order differences features from a list of dataframes.

        Args:
            dfs: A list of pandas dataframes containing physiological signals.
            offline: True enables Neurokit plotting functions

        Returns:
            A list of pandas dataframes containing extracted and original features.
        """

        start = time.time()
        sr_list = get_sampling_rate(dfs, self.config.time_col)

        # Extract ECG features from each dataframe in the input list.
        ecg_feats_dfs = extract_neuro_features(dfs, sr_list, self.config.ecg_col, self.config.target_col,
                                               self.config.time_col, self.subj_id,
                                               self.config.scenario_col, self.config.mode_col, self.config.modes,
                                               self.config.graph_path, "ECG", offline)

        # Extract EDA features from each dataframe in the input list.
        eda_feats_dfs = extract_neuro_features(dfs, sr_list, self.config.gsr_col, self.config.target_col,
                                               self.config.time_col, self.subj_id,
                                               self.config.scenario_col, self.config.mode_col, self.config.modes,
                                               self.config.graph_path, "EDA", offline)

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

    def save_preprocessed_data(self, dfs: List[pd.DataFrame], subj_id: str) -> None:
        """
        Save the preprocessed data to the "processed_data_path" defined in config. If "save_single_df" is True,
        all scenarios/modes will be saved in a single file, with time counter reset to 0 for each scenario/mode.

        Args:
            dfs: A list of pandas DataFrames containing the preprocessed data to be saved.
            subj_id: A string representing the subject ID for which the data is being saved.

        Returns:
            None
        """
        subj_dir = os.path.join(self.config.processed_data_path, f"SUBJ_{subj_id}")
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        filename = None

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

    def run(self, subpaths: Dict[str, str], subject_path: str, subj_id: str) -> None:
        """
        Offline Preprocessing pipeline: data loading, cleaning and validation, feature extraction, store preprocessed.

        Args:
            subpaths: A dictionary of file names for each scenario/mode dataset.
            subject_path: The path to the directory containing the subject's raw data files.
            subj_id: The participant's unique identification

        Returns:
            None
        """
        self.subj_id = subj_id
        dfs = self.load_data(subpaths, subject_path)
        prep_dfs = self.clean_and_validate(dfs)
        new_feats_dfs = self.extract_features(prep_dfs)
        self.save_preprocessed_data(new_feats_dfs, subj_id)

    def online_run(self, array):
        dfs = self.load_data_online(array)
        prep_dfs = self.clean_and_validate(dfs)
        new_feats_dfs = self.extract_features(prep_dfs, offline=False)  # Using offline=False to indicate this is online

        return new_feats_dfs
