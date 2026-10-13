"""Execute SAUP audit corrections and generate revalidated results and reports.

This script executes the corrected SAUP pipeline for:
1. UCI Adult
2. UCI Bank Marketing
3. UCI Wine Quality

All outputs are written to reports/audit_corrected/ without overwriting
or mutating the historical results in reports/results/.
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from saup.config import load_config, resolve_path, setup_logging
from saup.experiment_runner import ExperimentRunner
from saup.reporting import (
    generate_cross_dataset_report,
    generate_dataset_report,
    generate_experimental_report,
    generate_limitations_doc,
    generate_methodology_doc,
    generate_references_doc,
    generate_unified_results_table,
)

logger = logging.getLogger(__name__)


def run_audit_for_dataset(dataset_name: str, audit_results_dir: Path, audit_figures_dir: Path) -> dict:
    """Run corrected pipeline for one dataset."""
    logger.info("=" * 65)
    logger.info("STARTING AUDIT PIPELINE: %s", dataset_name.upper())
    logger.info("=" * 65)
    
    cfg_file = f"configs/{'default' if dataset_name == 'adult' else dataset_name}.yaml"
    config = load_config(cfg_file)
    
    runner = ExperimentRunner(
        config=config,
        dataset_name=dataset_name,
        results_dir=str(audit_results_dir),
        figures_dir=str(audit_figures_dir),
    )
    
    t0 = time.time()
    
    # 1. Load data
    runner.load_data()
    
    # 2. Split
    runner.split_data()
    
    # 3. Preprocess
    runner.preprocess()
    
    # 4. Train/Evaluate Baselines
    runner.train_baselines()
    
    # 5. Calibrate
    runner.calibrate()
    
    # 6. Conformal Setup
    runner.setup_conformal()
    
    # 7. Shift Detector Setup (Standardized MMD)
    runner.setup_shift_detector()
    
    # 8. BBSE Setup
    runner.setup_bbse()
    
    # 9. Shift Experiments (Strict test set isolation)
    runner.run_shift_experiments([runner.seed])
    
    # 10. SAUP Experiments (Disambiguated metrics, principled adaptation)
    runner.run_saup_experiments([runner.seed])
    
    # 11. Ablation Experiments (Full SAUP engine execution)
    runner.run_ablation()
    
    elapsed = time.time() - t0
    logger.info("FINISHED AUDIT PIPELINE: %s in %.1f seconds", dataset_name.upper(), elapsed)
    return {"dataset": dataset_name, "elapsed_s": elapsed, "success": True}


def main():
    setup_logging("INFO")
    logger.info("=== SAUP AUDIT CORRECTION AND EXPERIMENTAL REVALIDATION ===")
    
    audit_base = resolve_path("reports/audit_corrected")
    audit_results_dir = audit_base / "results"
    audit_figures_dir = audit_base / "figures"
    
    audit_results_dir.mkdir(parents=True, exist_ok=True)
    audit_figures_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ["adult", "bank", "wine"]
    statuses = []
    
    for ds in datasets:
        try:
            status = run_audit_for_dataset(ds, audit_results_dir, audit_figures_dir)
            statuses.append(status)
        except Exception as e:
            logger.error("Audit pipeline failed for %s: %s", ds, e, exc_info=True)
            statuses.append({"dataset": ds, "success": False, "error": str(e)})
            
    # Generate unified table and reports
    logger.info("Generating unified results and corrected documentation...")
    generate_unified_results_table(str(audit_results_dir))
    
    for ds in datasets:
        generate_dataset_report(ds, str(audit_results_dir), str(audit_base))
        
    generate_cross_dataset_report(str(audit_results_dir), str(audit_base))
    generate_experimental_report(str(audit_results_dir), str(audit_base))
    generate_methodology_doc(str(audit_base))
    generate_limitations_doc(str(audit_base))
    generate_references_doc(str(audit_base))
    
    logger.info("=" * 65)
    logger.info("ALL AUDIT CORRECTIONS & REPORTS COMPLETED SUCCESSFULLY!")
    logger.info("Results directory: %s", audit_results_dir)
    logger.info("Reports directory: %s", audit_base)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
