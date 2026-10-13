"""Download and validate datasets for SAUP: Adult, Bank Marketing, Wine Quality."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saup.config import ensure_directories, load_config, setup_logging
from saup.data_download import (
    download_adult_dataset,
    download_bank_dataset,
    download_dataset_by_name,
    download_wine_dataset,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download SAUP datasets (Adult, Bank, Wine)")
    parser.add_argument("--dataset", type=str, default="adult",
                        choices=["adult", "bank", "wine", "all"],
                        help="Dataset to download ('adult', 'bank', 'wine', 'all')")
    parser.add_argument("--zip-path", type=str, default=None,
                        help="Path to a pre-downloaded zip file")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    args = parser.parse_args()

    setup_logging("INFO")
    config = load_config(args.config)
    ensure_directories(config)

    datasets = ["adult", "bank", "wine"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        try:
            print(f"Downloading {ds}...")
            raw_path = download_dataset_by_name(ds, zip_path=args.zip_path)
            print(f"✓ {ds.upper()} dataset ready in: {raw_path}")
        except Exception as e:
            print(f"✗ Download failed for {ds}: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
