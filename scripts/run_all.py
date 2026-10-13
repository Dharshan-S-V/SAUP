"""Run the complete SAUP experimental pipeline for Adult, Bank Marketing, and/or Wine Quality."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saup.config import ensure_directories, load_config, setup_logging
from saup.experiment_runner import ExperimentRunner
from saup.reporting import (
    generate_cross_dataset_report,
    generate_dataset_report,
    generate_experimental_report,
    generate_limitations_doc,
    generate_methodology_doc,
    generate_unified_results_table,
)

logger = logging.getLogger(__name__)


def run_pipeline_for_dataset(dataset_name: str, config_path: str = None, zip_path: str = None) -> dict:
    """Run full pipeline for a single dataset."""
    default_configs = {
        "adult": "configs/default.yaml",
        "bank": "configs/bank.yaml",
        "wine": "configs/wine.yaml",
    }
    cfg_file = config_path or default_configs.get(dataset_name, "configs/default.yaml")
    config = load_config(cfg_file)
    ensure_directories(config)

    print("\n" + "=" * 60)
    print(f"SAUP Pipeline — [{dataset_name.upper()}]")
    print("=" * 60)

    runner = ExperimentRunner(config, dataset_name=dataset_name)
    outputs = runner.run_full_pipeline(zip_path=zip_path)

    if outputs.get("success"):
        generate_dataset_report(
            dataset_name,
            results_dir=config["output"]["results_dir"],
            reports_dir=config["output"]["reports_dir"],
        )
        print(f"[OK] {dataset_name.upper()} Pipeline Complete ({outputs.get('elapsed_s', 0):.1f}s)")
    else:
        print(f"[FAIL] {dataset_name.upper()} Pipeline Failed: {outputs.get('error')}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run complete SAUP pipeline")
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=["adult", "bank", "wine", "all"],
                        help="Dataset to run: 'adult', 'bank', 'wine', or 'all'")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config YAML")
    parser.add_argument("--zip-path", type=str, default=None,
                        help="Path to pre-downloaded dataset zip")
    args = parser.parse_args()

    setup_logging("INFO")

    print("=" * 60)
    print("SAUP — Shift-Aware Adaptive Uncertainty Prediction")
    print(f"Dataset target: {args.dataset.upper()}")
    print("=" * 60)

    if args.dataset == "all":
        datasets_to_run = ["bank", "wine"]
        # Note: Adult baseline already ran successfully; we do not re-run Adult unnecessarily
        for ds in datasets_to_run:
            run_pipeline_for_dataset(ds, config_path=args.config, zip_path=args.zip_path)

        # Generate reports across all datasets
        cfg = load_config()
        results_dir = cfg["output"]["results_dir"]
        reports_dir = cfg["output"]["reports_dir"]

        generate_dataset_report("adult", results_dir, reports_dir)
        generate_dataset_report("bank", results_dir, reports_dir)
        generate_dataset_report("wine", results_dir, reports_dir)
        generate_unified_results_table(results_dir)
        generate_cross_dataset_report(results_dir, reports_dir)
        generate_methodology_doc(reports_dir)
        generate_limitations_doc(reports_dir)
        print("\n" + "=" * 60)
        print("[OK] ALL DATASETS & CROSS-DATASET EVALUATION COMPLETE")
        print("=" * 60)

    else:
        outputs = run_pipeline_for_dataset(args.dataset, config_path=args.config, zip_path=args.zip_path)
        if outputs.get("success"):
            cfg = load_config(args.config or f"configs/{'default' if args.dataset == 'adult' else args.dataset}.yaml")
            results_dir = cfg["output"]["results_dir"]
            reports_dir = cfg["output"]["reports_dir"]
            generate_methodology_doc(reports_dir)
            generate_limitations_doc(reports_dir)
            generate_unified_results_table(results_dir)


if __name__ == "__main__":
    main()
