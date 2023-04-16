import argparse
import logging
import sys

from stress_preprocessor.config import Config
from stress_preprocessor.preprocessors.offline_preprocessor import StressPreprocessorOffline


def main(subj_id, save_path):
    config = Config('stress_preprocessor/config/offline_config.json')

    subject_path = f"stress_preprocessor/automotive_study_2/SUBJ_{subj_id}_DATA"
    subpaths = {
        "s1_eco": "Scenario_01.Case_set_1.Scenario01_Eco/results/icos/SUBJ_01_MOCK_01.csv",
        "s1_sport": "Scenario_01.Case_set_1.Scenario01_Sport/results/icos/SUBJ_01_MOCK_01.csv",
        "s1_comfort": "Scenario_01.Case_set_1.Scenario01_Comfort/results/icos/SUBJ_01_MOCK_01.csv",
        "s2_eco": "Scenario_02.Case_set_1.Scenario01_Eco/results/icos/SUBJ_01_MOCK_01.csv",
        "s2_sport": "Scenario_02.Case_set_1.Scenario01_Sport/results/icos/SUBJ_01_MOCK_01.csv",
        "s2_comfort": "Scenario_02.Case_set_1.Scenario01_Comfort/results/icos/SUBJ_01_MOCK_01.csv",
        "s3_eco": "Scenario_03.Case_set_1.Scenario01_Eco/results/icos/SUBJ_01_MOCK_01.csv",
        "s3_sport": "Scenario_03.Case_set_1.Scenario01_Sport/results/icos/SUBJ_01_MOCK_01.csv",
        "s3_comfort": "Scenario_03.Case_set_1.Scenario01_Comfort/results/icos/SUBJ_01_MOCK_01.csv"
    }

    preprocessor = StressPreprocessorOffline(config)
    preprocessor.run(subpaths, subject_path, subj_id, save_path)


if __name__ == '__main__':
    logging.root.handlers = []
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler("logs.log"),
                                                               logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-sid', '--subj_id',
                        required=True,
                        type=str,
                        help="The participant's ID.")
    parser.add_argument('-sp', '--save_path',
                        required=False,
                        type=str,
                        help="The path to save the processed dataframes.")

    args = parser.parse_args()

    main(args.subj_id, args.save_path)
