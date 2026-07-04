from __future__ import annotations

import argparse

from src.algorithm_platform.inference.multi_sim import run_multi_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulated multi-stream concurrent action inference")
    parser.add_argument("--config", required=True, help="Path to infer_multi.yaml")
    args = parser.parse_args()
    run_multi_inference(args.config)


if __name__ == "__main__":
    main()
