import argparse
from src.algorithm_platform.utils.config import load_config
from src.algorithm_platform.training.train_videomae import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
