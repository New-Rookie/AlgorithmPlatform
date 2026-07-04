from __future__ import annotations

import argparse

from src.algorithm_platform.inference.single import run_single_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-video sliding-window action inference")
    parser.add_argument("--config", required=True, help="Path to infer_single.yaml")
    args = parser.parse_args()
    run_single_inference(args.config)


if __name__ == "__main__":
    main()
