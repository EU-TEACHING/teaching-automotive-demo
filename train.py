import ray, argparse
from training.exp.config import get_hparams_config, get_analysis, get_exp_dir
from training.exp.phase import design

parser = argparse.ArgumentParser()
parser.add_argument("method", type=str)
parser.add_argument("phase", type=str, default="mrt")
parser.add_argument("gpus", type=float, default=0.1)


def main():
    args = parser.parse_args()
    method, gpus = args.method, args.gpus
    exp_dir = get_exp_dir(method)
    if "m" in args.phase:
        ray.init(address="auto", ignore_reinit_error=True)
        config = get_hparams_config(method)
        design.run("model_selection", exp_dir, config, gpus_per_trial=gpus)
    if "r" in args.phase:
        ray.init(address="auto", ignore_reinit_error=True)
        mode = "min" if not True else "max"
        config = get_analysis("model_selection", exp_dir, mode).get_best_config()
        design.run("retraining", exp_dir, config, gpus_per_trial=gpus)


if __name__ == "__main__":
    main()
