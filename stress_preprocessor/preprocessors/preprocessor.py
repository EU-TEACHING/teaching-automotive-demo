import logging
import os

import numpy as np
import pandas as pd
import time
from typing import Dict, List
from pandas import DataFrame
import warnings

from stress_preprocessor.utils.preprocessing_utils import (
    clean_duplicates,
    validate_timestamps,
    compute_diff,
    impute_null,
    mapper,
    is_in_time_interval,
    float_to_integer,
    load_csv_files,
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

    def load_data(
        self,
        subpaths: Dict[str, str],
        baseline_subpath: Dict[str, str],
        scenario_6_subpath: Dict[str, str],
        scenario_X_subpaths: Dict[str, str],
        subj_path: str,
        subj_id: str,
    ) -> [List[pd.DataFrame], List[pd.DataFrame]]:
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

        baseline_dfs = []
        if baseline_subpath:
            baseline_dfs = load_csv_files(
                baseline_dfs, baseline_subpath, subj_id, subj_path, self.config
            )

        scenario_6_dfs = []
        if scenario_6_subpath:
            scenario_6_dfs = load_csv_files(
                scenario_6_dfs, scenario_6_subpath, subj_id, subj_path, self.config
            )

        scenario_X_dfs = []
        if scenario_X_subpaths:
            scenario_X_dfs = load_csv_files(
                scenario_X_dfs, scenario_X_subpaths, subj_id, subj_path, self.config
            )

        dfs = []
        dfs = load_csv_files(dfs, subpaths, subj_id, subj_path, self.config)

        stop = time.time()
        logging.info(f"Data loading latency (secs): {stop - start}")

        return baseline_dfs, scenario_6_dfs, scenario_X_dfs, dfs

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

    def assign_stress_events(
        self,
        baseline_dfs: List[DataFrame],
        scenario_6_dfs: List[DataFrame],
        scenario_X_dfs: List[DataFrame],
        dfs: List[DataFrame],
    ) -> [
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
    ]:
        """Each scenario contains some crucial events. Create a new categorical feature in the dataframe with these
        events by assigning them based on a specific time range. x is the time lag to create the time ranges

        baseline_dfs: List containing the baseline dataframe or nothing.
            scenario_6_dfs: List containing the scenario 6 dataframe or nothing.
            scenario_X_dfs: List containing the scenario X dataframe or nothing.
            dfs: A list of pandas dataframes to preprocess.

        Returns:
            Lists of the raw pandas dataframes and an extra column containing the stress events.
        """

        start = time.time()
        logging.info(f"Annotation of stress events is starting ...")

        baseline_df_stress_events_list = []
        if baseline_dfs:
            for df in baseline_dfs:
                df["Stress_Event"] = None
                baseline_df_stress_events_list.append(df)

        # Dictionary of the crucial events that correspond to each scenario with their time ranges
        crucial_events = {
            1: {
                "accelerate_to_motorway": [[70 - 5, 70 + 5]],
                "cut_in_from_another_vehicle": [[92 - 5, 92 + 5]],
                "sharp_brake": [[98 - 5, 98 + 5]],
            },
            2: {
                "join_platoon": [[43 - 5, 43 + 5]],
                "platooning": [[75 - 5, 75 + 5], [117 - 29, 117 + 29]],
                "platoon_vehicle_cut_out": [[82 - 5, 82 + 5]],
            },
            3: {
                "traffic_light_sharp_break": [[43 - 10, 43 + 10]],
                "phantom_break": [[82 - 10, 82 + 10]],
                "road_crossing": [[113 - 10, 113 + 10]],
            },
            6: {
                "traffic_light": [
                    [f"{44 - 5}", f"{44 + 5}"],
                    [f"{109 - 5}", f"{109 + 5}"],
                    [f"{138 - 5}", f"{138 + 5}"],
                    [f"{207 - 5}", f"{207 + 5}"],
                ],
                "phantom_break": [
                    [f"{71 - 5}", f"{71 + 5}"],
                    [f"{181 - 5}", f"{181 + 5}"],
                ],
                "pedestrian_crossing": [f"{83 - 5}", f"{83 + 5}"],
                "cut_in_from_a_vehicle": [
                    [f"{313 - 5}", f"{313 + 5}"],
                    [f"{538 - 5}", f"{538 + 5}"],
                ],
                "join_platoon_at_motorway": [[f"{660 - 5}", f"{660 + 5}"]],
                "platoon_vehicle_cutting_out": [
                    [f"{666 - 5}", f"{666 + 5}"],
                    [f"{682 - 5}", f"{682 + 5}"],
                ],
            },
            "X": {
                "traffic_light_slow_down": [[f"{25 - 5}", f"{25 + 5}"]],
                "pedestrian_crossing": [[f"{64 - 5}", f"{64 + 5}"]],
            },
        }

        scenario_X_dfs_stress_events = []
        if scenario_X_dfs:
            for df in scenario_X_dfs:
                df["Stress_Event"] = df["Time"].apply(
                    lambda x: is_in_time_interval(
                        x, self.config.stress_events["X"].items()
                    )
                )
                df.reset_index(drop=True, inplace=True)
                scenario_X_dfs_stress_events.append(df)

        scenario_6_df_stress_events_list = []
        if scenario_6_dfs:
            for df in scenario_6_dfs:
                df["Stress_Event"] = df["Time"].apply(
                    lambda x: is_in_time_interval(
                        x, self.config.stress_events["6"].items()
                    )
                )
                df.reset_index(drop=True, inplace=True)
                scenario_6_df_stress_events_list.append(df)

        dfs_stress_events = []
        if dfs:
            for df in dfs:
                scenario_id = df[self.config.scenario_col].unique()[0]
                df["Stress_Event"] = df["Time"].apply(
                    lambda x: is_in_time_interval(
                        x,
                        self.config.stress_events[
                            str(self.config.scenario_ids[f"{scenario_id}"])
                        ].items(),
                    )
                )
                df.reset_index(drop=True, inplace=True)
                dfs_stress_events.append(df)

        stop = time.time()
        logging.info(f"Stress events annotation latency (secs): {stop - start}")

        return (
            baseline_df_stress_events_list,
            scenario_6_df_stress_events_list,
            scenario_X_dfs_stress_events,
            dfs_stress_events,
        )

    def __clean_and_validate_steps(self, dfs):
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
        return imputed_dfs

    def clean_and_validate(
        self,
        baseline_dfs: List[DataFrame],
        scenario_6_dfs: List[DataFrame],
        scenario_X_dfs: List[DataFrame],
        dfs: List[pd.DataFrame],
    ) -> [
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
    ]:
        """
        Preprocesses a list of dataframes.

        Args:
            baseline_dfs: List containing the baseline dataframe or nothing.
            scenario_6_dfs: List containing the scenario 6 dataframe or nothing.
            scenario_X_dfs: List containing the scenario X dataframe or nothing.
            dfs: A list of pandas dataframes to preprocess.

        Returns:
            List of preprocessed pandas dataframes.
        """

        start = time.time()
        logging.info(f"Starting data cleaning and timestamp validation ...")

        imputed_baseline_dfs = []
        if baseline_dfs:
            imputed_baseline_dfs = self.__clean_and_validate_steps(baseline_dfs)

        imputed_scenario_6_dfs = []
        if scenario_6_dfs:
            imputed_scenario_6_dfs = self.__clean_and_validate_steps(scenario_6_dfs)

        imputed_scenario_X_dfs = []
        if scenario_X_dfs:
            imputed_scenario_X_dfs = self.__clean_and_validate_steps(scenario_X_dfs)

        imputed_dfs = []
        if dfs:
            imputed_dfs = self.__clean_and_validate_steps(dfs)

        stop = time.time()
        logging.info(
            f"Data cleaning and timestamp validation latency (secs): {stop - start}"
        )

        return (
            imputed_baseline_dfs,
            imputed_scenario_6_dfs,
            imputed_scenario_X_dfs,
            imputed_dfs,
        )

    def visualize(self, dfs: List[pd.DataFrame]):
        start = time.time()
        logging.info(f" Starting data visualization ...")

        # Plot ECG raw
        if dfs:
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

        stop = time.time()
        logging.info(f" Data visualization latency (secs): {stop - start}")

    def __extract_features_steps(self, dfs, offline):
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
        return new_feats_dfs

    def extract_features(
        self,
        baseline_dfs: List[DataFrame],
        scenario_6_dfs: List[DataFrame],
        scenario_X_dfs: List[DataFrame],
        dfs: List[pd.DataFrame],
        offline=True,
    ) -> [
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
        List[pd.DataFrame],
    ]:
        """
        Extracts physiological features and first-order differences features from a list of dataframes.

        Args:
            dfs: A list of pandas dataframes containing physiological signals.
            offline: True enables Neurokit plotting functions

        Returns:
            A list of pandas dataframes containing extracted and original features.
        """

        start = time.time()
        logging.info(f"Starting feature extraction ...")

        baseline_new_feats_dfs = []
        if baseline_dfs:
            baseline_new_feats_dfs = self.__extract_features_steps(
                baseline_dfs, offline
            )

        scenario_6_new_feats_dfs = []
        if scenario_6_dfs:
            scenario_6_new_feats_dfs = self.__extract_features_steps(
                scenario_6_dfs, offline
            )

        scenario_X_new_feats_dfs = []
        if scenario_X_dfs:
            scenario_X_new_feats_dfs = self.__extract_features_steps(
                scenario_X_dfs, offline
            )

        new_feats_dfs = []
        if dfs:
            new_feats_dfs = self.__extract_features_steps(dfs, offline)

        stop = time.time()
        logging.info(f"Feature extraction latency (secs): {stop - start}")
        return (
            baseline_new_feats_dfs,
            scenario_6_new_feats_dfs,
            scenario_X_new_feats_dfs,
            new_feats_dfs,
        )

    def __save_data(self, dfs, subj_dir, filename):
        df = pd.concat(dfs)
        df.to_csv(os.path.join(subj_dir, filename), index=False)

    def save_raw_data_with_stress_events(
        self,
        baseline_dfs: List[pd.DataFrame],
        scenario_6_dfs: List[pd.DataFrame],
        scenario_X_dfs: List[pd.DataFrame],
        dfs: List[pd.DataFrame],
        subj_id: str,
    ) -> None:
        """
        Save the raw data to the "raw_data_with_stress_events_path" defined in config. If "save_single_df" is True,
        all scenarios/modes will be saved in a single file, with time counter reset to 0 for each scenario/mode.

        Args:
            dfs: A list of pandas DataFrames containing the raw data with the stress events to be saved.
            baseline_dfs: List containing the raw baseline Dataframe with the stress events to be saved.
            scenario_6_dfs: List containing the raw scenario 6 Dataframe with the stress events to be saved.
            scenario_X_dfs: A list of scenario X pandas DataFrames containing the raw data with the stress events to be saved.
            subj_id: A string representing the subject ID for which the data is being saved.

        Returns:
            None
        """

        start = time.time()
        logging.info(f"Saving the raw data with the stress events ...")

        subj_dir = os.path.join(
            self.config.raw_data_with_stress_events_path, f"SUBJ_{subj_id}"
        )
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        filename = None

        if baseline_dfs:
            self.__save_data(
                baseline_dfs, subj_dir, f"SUBJ_{subj_id}_SCEN_00_MODE_FreeDriving.csv"
            )

        if scenario_6_dfs:
            self.__save_data(scenario_6_dfs, subj_dir, f"SUBJ_{subj_id}_SCEN_06_AI.csv")

        if scenario_X_dfs:
            if self.config.save_single_df:
                self.__save_data(scenario_X_dfs, subj_dir, f"SUBJ_{subj_id}_SCEN_X.csv")
            else:
                for df in scenario_X_dfs:
                    mode = df.loc[0, self.config.mode_col]
                    filename = f"SCEN_X_MODE_{mode}.csv"
                    df.to_csv(os.path.join(subj_dir, filename), index=False)

        if dfs:
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

        stop = time.time()
        logging.info(
            f"Raw data with stress events saved at {subj_dir}, latency (secs): {stop - start}"
        )

    def save_preprocessed_data(
        self,
        baseline_dfs: List[pd.DataFrame],
        scenario_6_dfs: List[pd.DataFrame],
        scenario_X_dfs: List[pd.DataFrame],
        dfs: List[pd.DataFrame],
        subj_id: str,
    ) -> None:
        """
        Save the preprocessed data to the "processed_data_path" defined in config. If "save_single_df" is True,
        all scenarios/modes will be saved in a single file, with time counter reset to 0 for each scenario/mode.

        Args:
            dfs: A list of pandas DataFrames containing the preprocessed data to be saved.
            baseline_dfs: List containing the preprocessed baseline Dataframe to be saved.
            scenario_6_dfs: List containing the preprocessed scenario 6 Dataframe to be saved.
            scenario_X_dfs: List containing the preprocessed scenario X data to be saved.
            subj_id: A string representing the subject ID for which the data is being saved.

        Returns:
            None
        """

        start = time.time()
        logging.info(f"Saving the preprocessed data ...")

        subj_dir = os.path.join(self.config.processed_data_path, f"SUBJ_{subj_id}")
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)

        if baseline_dfs:
            self.__save_data(
                baseline_dfs, subj_dir, f"SUBJ_{subj_id}_SCEN_00_MODE_FreeDriving.csv"
            )

        if scenario_6_dfs:
            self.__save_data(scenario_6_dfs, subj_dir, f"SUBJ_{subj_id}_SCEN_06_AI.csv")

        if scenario_X_dfs:
            if self.config.save_single_df:
                self.__save_data(scenario_X_dfs, subj_dir, f"SUBJ_{subj_id}_SCEN_X.csv")
            else:
                for df in scenario_X_dfs:
                    mode = df.loc[0, self.config.mode_col]
                    filename = f"SCEN_X_MODE_{mode}.csv"
                    df.to_csv(os.path.join(subj_dir, filename), index=False)

        if dfs:
            if self.config.save_single_df:
                self.__save_data(dfs, subj_dir, f"SUBJ_{subj_id}_ALL_SCENARIOS.csv")
            else:
                for df in dfs:
                    scenario_id = df.loc[0, self.config.scenario_col]
                    mode = df.loc[0, self.config.mode_col]
                    filename = f"SCEN_{scenario_id}_MODE_{mode}.csv"
                    df.to_csv(os.path.join(subj_dir, filename), index=False)

        stop = time.time()
        logging.info(
            f"Preprocessed data saved at {subj_dir}, latency (secs): {stop - start}"
        )

    def run(
        self,
        subpaths: Dict[str, str],
        baseline_subpath: Dict[str, str],
        scenario_6_subpath: Dict[str, str],
        scenario_X_subpaths: Dict[str, str],
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
        baseline_dfs, scenario_6_dfs, scenario_X_dfs, dfs = self.load_data(
            subpaths,
            baseline_subpath,
            scenario_6_subpath,
            scenario_X_subpaths,
            subject_path,
            subj_id,
        )

        (
            baseline_dfs_stress_events,
            scenario_6_dfs_stress_events,
            scenario_X_dfs_stress_events,
            dfs_stress_events,
        ) = self.assign_stress_events(baseline_dfs, scenario_6_dfs, scenario_X_dfs, dfs)
        self.save_raw_data_with_stress_events(
            baseline_dfs_stress_events,
            scenario_6_dfs_stress_events,
            scenario_X_dfs_stress_events,
            dfs_stress_events,
            subj_id,
        )

        self.visualize(dfs)
        (
            prep_baseline_dfs,
            prep_scenario_6_dfs,
            prep_scenario_X_dfs,
            prep_dfs,
        ) = self.clean_and_validate(
            baseline_dfs_stress_events,
            scenario_6_dfs_stress_events,
            scenario_X_dfs_stress_events,
            dfs_stress_events,
        )
        (
            baseline_new_feats_dfs,
            scenario_6_new_feats_dfs,
            scenario_X_new_feats_dfs,
            new_feats_dfs,
        ) = self.extract_features(
            prep_baseline_dfs, prep_scenario_6_dfs, prep_scenario_X_dfs, prep_dfs
        )

        self.save_preprocessed_data(
            baseline_new_feats_dfs,
            scenario_6_new_feats_dfs,
            scenario_X_new_feats_dfs,
            new_feats_dfs,
            subj_id,
        )

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
            _, _, _, prep_dfs = self.clean_and_validate([], [], [], dfs)
            prep_dfs = float_to_integer(prep_dfs, self.config)
            _, _, _, proc_df = self.extract_features(
                [], [], [], prep_dfs, offline=False
            )[0]
            eda = proc_df["EDA_Clean"].values
            ecg = proc_df["ECG_Rate"].values
            to_return = np.stack([eda, ecg], axis=1)
            to_return = to_return[self.last_returned :]
            self.last_returned = len(self.window)
            print(f"length: {len(to_return)}")
            return to_return
        stop = time.time()
        logging.info(f"Overall latency (secs): {stop - start}")
