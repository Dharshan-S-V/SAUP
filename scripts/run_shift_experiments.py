"""Run distribution shift experiments across datasets."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saup.config import ensure_directories, load_config, setup_logging
from saup.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


def run_shifts_for_dataset(dataset_name: str, config_path: str = None):
    cfg_file = config_path or f"configs/{'default' if dataset_name == 'adult' else dataset_name}.yaml"
    config = load_config(cfg_file)
    ensure_directories(config)

    print("\n" + "=" * 60)
    print(f"Shift Experiments for [{dataset_name.upper()}]")
    print("=" * 60)

    runner = ExperimentRunner(config, dataset_name=dataset_name)
    runner.download_data()
    runner.load_data()
    runner.split_data()
    runner.preprocess()
    runner.train_baselines()
    runner.calibrate()
    runner.setup_conformal()
    runner.setup_shift_detector()
    runner.setup_bbse()

    seeds = [config["random_seed"]] + config.get("additional_seeds", [])
    shift_results = runner.run_shift_experiments(seeds)
    print(f"\n✓ Shift experiments: {len(shift_results)} runs")

    saup_results = runner.run_saup_experiments([config["random_seed"]])
    print(f"\n✓ SAUP experiments: {len(saup_results)} runs")

    return shift_results, saup_results


def main():
    parser = argparse.ArgumentParser(description="Run distribution shift experiments")
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=["adult", "bank", "wine", "all"],
                        help="Dataset: 'adult', 'bank', 'wine', or 'all'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    args = parser.parse_args()

    setup_logging("INFO")
    datasets = ["adult", "bank", "wine"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_shifts_for_dataset(ds, config_path=args.config)


if __name__ == "__main__":
    main()
