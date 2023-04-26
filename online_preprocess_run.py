import pandas as pd
import logging
import sys
from stress_preprocessor.config import Config
from stress_preprocessor.preprocessors.preprocessor import StressPreprocessor


if __name__ == "__main__":
    logging.root.handlers = []
    logging.basicConfig(
        format="%(asctime)s: %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler("online_logs.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    config = Config("stress_preprocessor/config/config.json")
    # Simulate streaming
    mock_filepath = "stress_preprocessor/data/automotive_study_2/TEACHING_2023/Scenario_01.SUBJ_01.Scenario01_Eco/results/mcx/DATA_OUT_in.csv"
    df = pd.read_csv(mock_filepath, header=1)
    df = df.drop(0)
    df = df[
        [
            config.time_col,
            config.ecg_col,
            config.gsr_col,
            config.target_col,
            config.error_col,
            config.scenario_col,
            config.mode_col,
            config.participant_col,
        ]
    ]
    df = df.astype(
        {
            config.ecg_col: "float32",
            config.gsr_col: "float32",
            config.target_col: "float32",
            config.error_col: "float32",
        }
    )
    preprocessor = StressPreprocessor(
        config=Config("stress_preprocessor/config/config.json"),
        baseline_path="stress_preprocessor/data/automotive_study_2/TEACHING_2023/Scenario_00.SUBJ_01.Scenario00_FreeDriving/results/mcx/DATA_OUT_in.csv",
        buffer_size=3000,
        online=True,
    )
    df.reset_index(inplace=True, drop=True)
    for i in range(len(df)):
        print(f"Row number: {i}")
        df_chunk = df.iloc[i]
        new_data = df_chunk.to_dict()
        print(new_data)
        preprocessor.online_run(new_data)
