import logging
import os
import pandas as pd
import time
from typing import Dict, List

from stress_preprocessor.utils.preprocessing_utils import clean_duplicates, validate_timestamps
from stress_preprocessor.utils.signal_processing_utils import extract_ecg_features, extract_eda_features, \
    get_sampling_rate
from stress_preprocessor.utils.visualization_utils import plot_timeseries_raw


class StressPreprocessorOffline:
    def __init__(self, config):
        self.save_path = None
        self.subj_id = None
        self.config = config

    def load_data(self, subpaths: Dict[str, str], subj_path: str) -> List[pd.DataFrame]:
        """
        Load data from a subject's directory into a list of pandas DataFrames.

        Args:
            subpaths (Dict[str, str]): A dictionary of file names for each scenario/mode dataset.
            subj_path (str): The path to the directory containing the subject's data files.

        Returns:
            List[pd.DataFrame]: A list of pandas DataFrames, one for each scenario/mode dataset.
        """

        # Load data from the subject's path, header 0 is the separator info, so skip it
        s1_eco = pd.read_csv(os.path.join(subj_path, subpaths["s1_eco"]), header=1, sep=';')
        s1_sport = pd.read_csv(os.path.join(subj_path, subpaths["s1_sport"]), header=1, sep=';')
        s1_comfort = pd.read_csv(os.path.join(subj_path, subpaths["s1_comfort"]), header=1, sep=';')
        s2_eco = pd.read_csv(os.path.join(subj_path, subpaths["s2_eco"]), header=1, sep=';')
        s2_sport = pd.read_csv(os.path.join(subj_path, subpaths["s2_sport"]), header=1, sep=';')
        s2_comfort = pd.read_csv(os.path.join(subj_path, subpaths["s2_comfort"]), header=1, sep=';')
        s3_eco = pd.read_csv(os.path.join(subj_path, subpaths["s3_eco"]), header=1, sep=';')
        s3_sport = pd.read_csv(os.path.join(subj_path, subpaths["s3_sport"]), header=1, sep=';')
        s3_comfort = pd.read_csv(os.path.join(subj_path, subpaths["s3_comfort"]), header=1, sep=';')

        dfs = [s1_eco, s1_sport, s1_comfort, s2_eco, s2_sport, s2_comfort, s3_eco, s3_sport, s3_comfort]

        # Remove header with unit info
        dfs_final = []
        for df in dfs:
            df = df.drop(0)
            df = df.astype(
                {self.config.time_col: 'float32', self.config.ecg_col: 'float32', self.config.gsr_col: 'float32',
                 self.config.target_col: 'float32'})
            df = df.loc[0:3000, :]

            dfs_final.append(df)

        return dfs_final

    def preprocess_data(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Preprocesses a list of dataframes.

        Args:
            dfs: A list of pandas dataframes to preprocess.

        Returns:
            A list of preprocessed pandas dataframes.
        """
        # Remove duplicate rows from each dataframe in the input list.
        no_dup_dfs = clean_duplicates(dfs)
        # Validate the remaining timestamps to ensure that they are uniformly spaced.
        val_dfs = validate_timestamps(no_dup_dfs, self.config.time_col, self.config.sampling_rate_hz)

        # Impute missing values in the preprocessed dataframes.
        # TODO: Add imputation logic here. Update: NeuroKitWarning: There are 1 missing data points in your signal. Filling missing values by using the forward filling method.

        return val_dfs

    def visualize(self):
        pass

    def extract_features(self, dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Extracts physiological features and first-order differences features from a list of dataframes.

        Args:
            dfs: A list of pandas dataframes containing physiological signals.

        Returns:
            A list of pandas dataframes containing extracted features.
        """

        start = time.time()
        sr_list = get_sampling_rate(dfs, self.config.time_col)

        # Extract ECG features from each dataframe in the input list.
        ecg_feats_dfs = extract_ecg_features(dfs, sr_list, self.config.ecg_col, self.subj_id, self.config.scenario_col,
                                             self.config.mode_col, self.config.modes, self.config.graph_path)

        # Extract EDA features from each dataframe in the input list.
        eda_feats_dfs = extract_eda_features(dfs, sr_list, self.config.ecg_col, self.subj_id, self.config.scenario_col,
                                             self.config.mode_col, self.config.modes, self.config.graph_path)

        new_feats_dfs = []
        for ecg_feats_df, eda_feats_df in zip(ecg_feats_dfs, eda_feats_dfs):
            concat_df = pd.concat([ecg_feats_df, eda_feats_df], axis=1)
            new_feats_dfs.append(concat_df)

        # Compute first-order differences between consecutive values as additional features
        # TODO: Add code to compute differences

        new_feats_df = pd.concat()
        stop = time.time()
        logging.info(f"Feature extraction: {stop - start}")
        return ecg_feats_dfs

    def save_preprocessed_data(self, save_path):
        # Save the preprocessed data to the given path
        pass

    def run(self, subpaths, subject_path, subj_id, save_path):
        self.subj_id = subj_id
        self.save_path = save_path
        # Load, preprocess, and save the data
        dfs = self.load_data(subpaths, subject_path)
        prep_dfs = self.preprocess_data(dfs)
        new_feats_dfs = self.extract_features(prep_dfs)

        self.save_preprocessed_data(save_path)
