"""Run baseline model training and evaluation across datasets."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saup.config import ensure_directories, load_config, setup_logging
from saup.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)


def run_baselines_for_dataset(dataset_name: str, config_path: str = None):
    cfg_file = config_path or f"configs/{'default' if dataset_name == 'adult' else dataset_name}.yaml"
    config = load_config(cfg_file)
    ensure_directories(config)

    print("\n" + "=" * 60)
    print(f"Baselines for [{dataset_name.upper()}]")
    print("=" * 60)

    runner = ExperimentRunner(config, dataset_name=dataset_name)
    runner.download_data()
    runner.load_data()
    runner.split_data()
    runner.preprocess()

    results = runner.train_baselines()
    print(f"\n✓ Baseline results ({len(results)} models):")
    print(results.to_string(index=False))

    cal_results = runner.calibrate()
    print("\n✓ Calibration complete")

    conf_results = runner.setup_conformal()
    print("\n✓ Conformal prediction calibrated")
    for name, metrics in conf_results.items():
        print(f"  {name}: coverage={metrics['empirical_coverage']:.4f}, "
              f"avg_set_size={metrics['avg_set_size']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=["adult", "bank", "wine", "all"],
                        help="Dataset: 'adult', 'bank', 'wine', or 'all'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    args = parser.parse_args()

    setup_logging("INFO")
    datasets = ["adult", "bank", "wine"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_baselines_for_dataset(ds, config_path=args.config)


if __name__ == "__main__":
    main()
