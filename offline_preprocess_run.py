import argparse
import logging
import sys
import time

from stress_preprocessor.config import Config
from stress_preprocessor.preprocessors.preprocessor import StressPreprocessor


def main(subj_id):
    config = Config('stress_preprocessor/config/config.json')

    subject_path = f"stress_preprocessor/data/automotive_study_2/SUBJ_{subj_id}_DATA"
    subpaths = config.subpaths

    preprocessor = StressPreprocessor(config)
    preprocessor.run(subpaths, subject_path, subj_id)


if __name__ == '__main__':

    start = time.time()

    logging.root.handlers = []
    logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s", level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.FileHandler("logs.log"),
                                                               logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-sid', '--subj_id',
                        required=True,
                        type=str,
                        help="The participant's ID.")

    args = parser.parse_args()

    main(args.subj_id)

    stop = time.time()
    logging.info(f"Overall latency (secs): {stop - start}")
