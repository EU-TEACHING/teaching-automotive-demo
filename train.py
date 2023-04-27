import ray, argparse
from training.config import get_hparams_config, get_analysis, get_exp_dir
from training.phase import design

parser = argparse.ArgumentParser()
parser.add_argument("user_id", type=str)
parser.add_argument("method", type=str)
parser.add_argument("task", type=str)
parser.add_argument("phase", type=str, default="mrt")
parser.add_argument("gpus", type=float, default=0.1)


def main():
    args = parser.parse_args()
    user_id, method, task, gpus = args.user_id, args.method, args.task, args.gpus
    classification = task == "c"
    exp_dir = get_exp_dir(user_id, method, classification)
    if "m" in args.phase:
        ray.init(address="auto", ignore_reinit_error=True)
        config = get_hparams_config(user_id, method, classification)
        design.run("model_selection", exp_dir, config, gpus_per_trial=gpus)
    if "r" in args.phase:
        ray.init(address="auto", ignore_reinit_error=True)
        mode = "min" if True else "max"
        config = get_analysis("model_selection", exp_dir, mode).get_best_config()
        design.run("retraining", exp_dir, config, gpus_per_trial=gpus)


if __name__ == "__main__":
    main()
