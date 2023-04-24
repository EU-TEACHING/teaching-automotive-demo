# import logging
# import sys
#
# import pandas as pd
#
# from stress_preprocessor.config import Config
# from stress_preprocessor.preprocessors.preprocessor import StressPreprocessor
#
#
# def main(streamed_array, subj_id):
#     config = Config('stress_preprocessor/config/config.json')
#     preprocessor = StressPreprocessor(config)
#     new_feats_df = preprocessor.online_run(streamed_array, subj_id)
#     print(new_feats_df)
#
#
# if __name__ == '__main__':
#     logging.root.handlers = []
#     logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO,
#                         datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler("online_logs.log"),
#                                                                logging.StreamHandler(sys.stdout)])
#     config = Config('stress_preprocessor/config/config.json')
#     # Simulate streaming
#     mock_filepath = 'stress_preprocessor/data/automotive_study_2/TEACHING_2023/Scenario_01.SUBJ_01.Scenario01_Eco/results/mcx/DATA_OUT_in.csv'
#     df = pd.read_csv(mock_filepath, header=1)
#     df = df.drop(0)
#     df = df[[config.time_col, config.ecg_col, config.gsr_col,
#                          config.target_col, config.error_col, config.scenario_col,
#                          config.mode_col, config.participant_col]]
#     df = df.astype({config.time_col: 'float32', config.ecg_col: 'float32',
#                     config.gsr_col: 'float32', config.target_col: 'float32',
#                     config.error_col: 'float32'})
#     df.reset_index(inplace=True, drop=True)
#
#     for i in range(len(df)):
#         df_chunk = df.iloc[i:(i + 1000), :]
#         array = df_chunk.values
#         main(array, "01")

from typing import Dict
import numpy as np
import pandas as pd
import neurokit2 as nk
import json

from stress_preprocessor.config import Config
from stress_preprocessor.preprocessors.preprocessor import StressPreprocessor


class SlidingWindowBuffer(object):
    def __init__(self, baseline_path: str, buffer_size: int) -> None:
        self.baseline_path = baseline_path
        self.config = Config('stress_preprocessor/config/config.json')
        baseline_data = pd.read_csv(baseline_path, sep=";")
        self.window = baseline_data.to_dict("records")[-buffer_size:]
        self.last_returned = len(self.window)
        self.preprocessor = StressPreprocessor(self.config)

    def __call__(self, data_dict: Dict) -> np.ndarray:
        self.window = self.window.append(data_dict)
        self.window = self.window[1:]
        self.last_returned -= 1

        if self.last_returned <= len(self.window) // 5:
            dfs = self.preprocessor.load_data_online(self.window)
            _, prep_dfs = self.preprocessor.clean_and_validate(None, dfs)
            prep_dfs = self.preprocessor.float_to_integer(prep_dfs)
            proc_df = self.preprocessor.extract_features(prep_dfs, offline=False)[0]
            eda = proc_df["EDA_Clean"].values
            ecg = proc_df["ECG_Rate"].values
            to_return = np.stack([eda, ecg], axis=1)
            to_return = to_return[self.last_returned :]
            self.last_returned = len(self.window)
            return to_return

