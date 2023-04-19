import pandas as pd

from stress_preprocessor.config import Config
from stress_preprocessor.preprocessors.preprocessor import StressPreprocessor


def main(streamed_array):
    config = Config('stress_preprocessor/config/config.json')
    preprocessor = StressPreprocessor(config)
    new_feats_df = preprocessor.online_run(streamed_array)
    print(new_feats_df)


if __name__ == '__main__':
    config = Config('stress_preprocessor/config/config.json')
    # Simulate streaming
    mock_filepath = 'stress_preprocessor/data/automotive_study_2/SUBJ_01_DATA/Scenario_01.Case_set_1.Scenario01_Comfort/results/icos/SUBJ_01_MOCK_01.csv'
    df = pd.read_csv(mock_filepath, header=1, sep=';')
    df = df.drop(0)
    df = df.astype({config.time_col: 'float32', config.ecg_col: 'float32',
                    config.gsr_col: 'float32', config.target_col: 'float32'})
    df.reset_index(inplace=True, drop=True)

    for i in range(len(df)):
        df = df.iloc[i:3000, :]
        array = df.values
        main(array)


