"""Experiment runner for SAUP — orchestrates experimental workflows for Adult, Bank, and Wine."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from saup.config import ensure_directories, load_config, resolve_path, setup_logging
from saup.data_download import download_dataset_by_name
from saup.data_loader import (
    COLUMN_NAMES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    DatasetSpec,
    get_dataset_spec,
    load_dataset_by_name,
)
from saup.preprocessing import (
    build_preprocessor,
    fit_preprocessor,
    prepare_features_target,
    transform_data,
)
from saup.splitting import create_splits, save_split_metadata
from saup.models import (
    get_available_models,
    get_model,
    load_model,
    predict,
    save_model,
    train_model,
)
from saup.metrics import (
    area_under_risk_coverage,
    calibration_metrics,
    classification_metrics,
    compute_confusion_matrix,
    conformal_metrics,
    risk_coverage_curve,
    selective_prediction_metrics,
)
from saup.calibration import TemperatureScaler
from saup.conformal import (
    SplitConformalClassifier,
    WeightedConformalClassifier,
    estimate_density_ratios,
)
from saup.shift_generation import generate_shift
from saup.shift_detection import ShiftDetector
from saup.label_shift import BBSEEstimator
from saup.decision_engine import DecisionAction, SAUPDecisionEngine
from saup.visualization import (
    plot_ablation_study,
    plot_abstention_vs_severity,
    plot_accuracy_under_shift,
    plot_calibration_comparison,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_coverage_by_scenario,
    plot_missing_values,
    plot_model_comparison,
    plot_reliability_diagram,
    plot_risk_coverage_curve,
    plot_saup_vs_baseline,
    plot_set_size_by_scenario,
    plot_shift_analysis,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates SAUP experiments for a given dataset."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        dataset_name: Optional[str] = None,
        results_dir: Optional[str] = None,
        figures_dir: Optional[str] = None,
    ):
        if config is None:
            config = load_config()
        self.config = config
        self.seed = config.get("random_seed", 42)
        ensure_directories(config)

        # Output directory resolution (allows targeting audit_corrected without mutating original)
        self.results_dir = resolve_path(results_dir or config.get("output", {}).get("results_dir", "reports/results"))
        self.figures_dir = resolve_path(figures_dir or config.get("output", {}).get("figures_dir", "reports/figures"))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # Dataset resolution: explicit arg > config > default "adult"
        self.dataset_name = dataset_name or config.get("dataset", {}).get("name", "adult")
        self.dataset_spec: DatasetSpec = get_dataset_spec(self.dataset_name)

        # Will be populated during pipeline
        self.train_df = None
        self.test_df = None
        self.summary = None
        self.splits = None
        self.preprocessor = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.trained_models = {}
        self.calibrators = {}
        self.conformal_classifiers = {}
        self.shift_detector = None
        self.bbse = None
        self.default_model_name = None

    # ── Phase 1: Data ──────────────────────────────────────────────

    def download_data(self, zip_path: Optional[str] = None) -> Path:
        logger.info("=== Phase 1: Download Data (%s) ===", self.dataset_name)
        raw_dir = self.config.get("data", {}).get("raw_dir", "data/raw")
        if self.dataset_name == "bank":
            raw_dir = "data/raw/bank"
        elif self.dataset_name == "wine":
            raw_dir = "data/raw/wine"
        return download_dataset_by_name(
            dataset_name=self.dataset_name,
            raw_dir=raw_dir,
            zip_path=zip_path,
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=== Phase 2: Load & Clean Data (%s) ===", self.dataset_name)
        raw_dir = self.config.get("data", {}).get("raw_dir", "data/raw")
        if self.dataset_name == "bank":
            raw_dir = "data/raw/bank"
        elif self.dataset_name == "wine":
            raw_dir = "data/raw/wine"
            
        train_df, test_df, summary, spec = load_dataset_by_name(
            self.dataset_name,
            raw_dir=raw_dir,
            seed=self.seed,
        )
        self.train_df = train_df
        self.test_df = test_df
        self.summary = summary
        self.dataset_spec = spec

        # Save summary
        proc_dir = resolve_path(self.config["data"]["processed_dir"])
        proc_dir.mkdir(parents=True, exist_ok=True)
        summary_file = proc_dir / f"preprocessing_summary_{self.dataset_name}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        if self.dataset_name == "adult":
            with open(proc_dir / "preprocessing_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

        # Plots
        fig_dir = str(self.figures_dir)
        target = self.dataset_spec.target_col
        plot_class_distribution(
            train_df[target].values,
            f"Training Class Distribution ({self.dataset_name})",
            f"class_distribution_train_{self.dataset_name}",
            fig_dir,
            class_names=self.dataset_spec.class_names,
        )
        plot_class_distribution(
            test_df[target].values,
            f"Test Class Distribution ({self.dataset_name})",
            f"class_distribution_test_{self.dataset_name}",
            fig_dir,
            class_names=self.dataset_spec.class_names,
        )
        if summary.get("train_missing"):
            plot_missing_values(
                summary["train_missing"],
                f"Training Missing Values ({self.dataset_name})",
                f"missing_values_train_{self.dataset_name}",
                fig_dir,
            )

        return train_df, test_df

    def split_data(self) -> Dict[str, pd.DataFrame]:
        logger.info("=== Phase 3: Split Data (%s) ===", self.dataset_name)
        split_cfg = self.config.get("splitting", {})
        self.splits = create_splits(
            self.train_df,
            target_col=self.dataset_spec.target_col,
            train_ratio=split_cfg.get("train_ratio", 0.70),
            calibration_ratio=split_cfg.get("calibration_ratio", 0.10),
            conformal_ratio=split_cfg.get("conformal_ratio", 0.10),
            policy_ratio=split_cfg.get("policy_ratio", 0.10),
            random_seed=self.seed,
        )
        # Save metadata
        proc_dir = resolve_path(self.config["data"]["processed_dir"])
        meta_file = proc_dir / f"split_metadata_{self.dataset_name}.json"
        metadata = {
            "dataset": self.dataset_name,
            "random_seed": self.seed,
            "splits": {name: len(df) for name, df in self.splits.items()},
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        if self.dataset_name == "adult":
            save_split_metadata(self.splits, self.seed, self.config["data"]["processed_dir"])
            
        return self.splits

    def preprocess(self) -> None:
        logger.info("=== Phase 4: Preprocess (%s) ===", self.dataset_name)
        self.preprocessor = build_preprocessor(
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
            scale=True,
        )

        X_tr, self.y_train = prepare_features_target(
            self.splits["train"],
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        fit_preprocessor(self.preprocessor, X_tr)
        self.X_train = transform_data(self.preprocessor, X_tr)

        X_te, self.y_test = prepare_features_target(
            self.test_df,
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        self.X_test = transform_data(self.preprocessor, X_te)

        # Save preprocessor
        models_dir = resolve_path(self.config["output"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, models_dir / f"preprocessor_{self.dataset_name}.joblib")
        if self.dataset_name == "adult":
            joblib.dump(self.preprocessor, models_dir / "preprocessor.joblib")
            
        logger.info(
            "Preprocessor saved for %s. Train shape: %s, Test shape: %s",
            self.dataset_name, self.X_train.shape, self.X_test.shape
        )

    # ── Phase 2: Models ────────────────────────────────────────────

    def train_baselines(self) -> pd.DataFrame:
        logger.info("=== Phase 5: Train Baselines (%s) ===", self.dataset_name)
        model_names = get_available_models(self.config)
        results = []

        for name in model_names:
            try:
                model = get_model(name, self.config, self.seed)
                model, train_time = train_model(model, self.X_train, self.y_train, name)
                self.trained_models[name] = model
                
                # Save model
                models_dir = resolve_path(self.config["output"]["models_dir"])
                save_model(model, f"{self.dataset_name}_{name}", self.config["output"]["models_dir"])
                if self.dataset_name == "adult":
                    save_model(model, name, self.config["output"]["models_dir"])

                preds, probas = predict(model, self.X_test)
                cls_m = classification_metrics(self.y_test, preds, probas)
                cal_m = calibration_metrics(self.y_test, probas)

                row = {
                    "dataset": self.dataset_name,
                    "model_name": name,
                    "scenario": "source",
                    "severity": "none",
                    "seed": self.seed,
                    "n_test": len(self.y_test),
                    "train_time_s": round(train_time, 2),
                    **cls_m,
                    **cal_m,
                }
                results.append(row)

                # Confusion matrix
                cm = compute_confusion_matrix(self.y_test, preds)
                plot_confusion_matrix(
                    cm,
                    f"Confusion Matrix - {name} ({self.dataset_name})",
                    f"confusion_matrix_{self.dataset_name}_{name}",
                    str(self.figures_dir),
                )

                logger.info(
                    "[%s] %s: acc=%.4f, f1=%.4f, auc=%.4f, ece=%.4f",
                    self.dataset_name, name, cls_m["accuracy"], cls_m["f1"],
                    cls_m.get("roc_auc", 0), cal_m["ece"]
                )
            except Exception as e:
                logger.error("Failed to train %s on %s: %s", name, self.dataset_name, e)

        results_df = pd.DataFrame(results)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_dir / f"{self.dataset_name}_baseline_results.csv", index=False)
        if self.dataset_name == "adult":
            results_df.to_csv(self.results_dir / "baseline_results.csv", index=False)

        if len(results_df) > 0:
            plot_model_comparison(results_df, fig_dir=str(self.figures_dir))

        # Model selection using policy-validation split
        if self.splits is not None and len(results) > 0:
            X_pol, y_pol = prepare_features_target(
                self.splits["policy"],
                target_col=self.dataset_spec.target_col,
                numerical_features=self.dataset_spec.numerical_features,
                categorical_features=self.dataset_spec.categorical_features,
            )
            X_pol_t = transform_data(self.preprocessor, X_pol)
            best_f1 = -1
            for name, model in self.trained_models.items():
                preds_v, _ = predict(model, X_pol_t)
                f1_v = classification_metrics(y_pol, preds_v)["f1"]
                if f1_v > best_f1:
                    best_f1 = f1_v
                    self.default_model_name = name
            logger.info(
                "[%s] Default model selected: %s (policy-val F1=%.4f)",
                self.dataset_name, self.default_model_name, best_f1
            )

        return results_df

    # ── Phase 3: Calibration ───────────────────────────────────────

    def calibrate(self) -> Dict:
        logger.info("=== Phase 6: Calibration (%s) ===", self.dataset_name)
        cal_results = {}

        X_cal, y_cal = prepare_features_target(
            self.splits["calibration"],
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        X_cal_t = transform_data(self.preprocessor, X_cal)

        for name, model in self.trained_models.items():
            _, probas_cal = predict(model, X_cal_t)

            scaler = TemperatureScaler()
            scaler.fit(y_cal, probas_cal)
            self.calibrators[name] = scaler

            # Evaluate on held-out test set
            _, probas_test = predict(model, self.X_test)
            uncal_metrics = calibration_metrics(self.y_test, probas_test)
            cal_probas = scaler.transform(probas_test)
            cal_m = calibration_metrics(self.y_test, cal_probas)

            cal_results[name] = {
                "temperature": scaler.temperature,
                "uncalibrated": uncal_metrics,
                "calibrated": cal_m,
            }

            # Reliability diagrams
            fig_dir = str(self.figures_dir)
            plot_reliability_diagram(
                self.y_test, probas_test[:, 1],
                f"Reliability - {name} ({self.dataset_name}, Uncalibrated)",
                f"reliability_{self.dataset_name}_{name}_uncal", fig_dir
            )
            plot_reliability_diagram(
                self.y_test, cal_probas[:, 1],
                f"Reliability - {name} ({self.dataset_name}, Calibrated T={scaler.temperature:.3f})",
                f"reliability_{self.dataset_name}_{name}_cal", fig_dir
            )

            logger.info(
                "[%s] %s: T=%.4f, ECE %.4f→%.4f, Brier %.4f→%.4f",
                self.dataset_name, name, scaler.temperature,
                uncal_metrics["ece"], cal_m["ece"],
                uncal_metrics["brier_score"], cal_m["brier_score"]
            )

        # Save calibrators
        models_dir = resolve_path(self.config["output"]["models_dir"])
        for name, scaler in self.calibrators.items():
            joblib.dump(scaler, models_dir / f"calibrator_{self.dataset_name}_{name}.joblib")
            if self.dataset_name == "adult":
                joblib.dump(scaler, models_dir / f"calibrator_{name}.joblib")

        return cal_results

    # ── Phase 4: Conformal ─────────────────────────────────────────

    def setup_conformal(self) -> Dict:
        logger.info("=== Phase 7: Conformal Prediction (%s) ===", self.dataset_name)
        conf_results = {}

        X_conf, y_conf = prepare_features_target(
            self.splits["conformal"],
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        X_conf_t = transform_data(self.preprocessor, X_conf)
        alpha = self.config.get("conformal", {}).get("alpha", 0.10)

        for name, model in self.trained_models.items():
            _, probas_conf = predict(model, X_conf_t)

            if name in self.calibrators:
                probas_conf = self.calibrators[name].transform(probas_conf)

            cc = SplitConformalClassifier(alpha=alpha)
            cc.calibrate(y_conf, probas_conf)
            self.conformal_classifiers[name] = cc

            # Evaluate on held-out test
            _, probas_test = predict(model, self.X_test)
            if name in self.calibrators:
                probas_test = self.calibrators[name].transform(probas_test)

            pred_sets = cc.predict_sets(probas_test)
            cm = conformal_metrics(self.y_test, pred_sets, alpha)
            conf_results[name] = cm

            logger.info(
                "[%s] %s: coverage=%.4f (nominal=%.4f), avg_set=%.3f, empty=%.4f",
                self.dataset_name, name, cm["empirical_coverage"], cm["nominal_coverage"],
                cm["avg_set_size"], cm["empty_set_rate"]
            )

        return conf_results

    # ── Phase 5: Shift Detection Setup ─────────────────────────────

    def setup_shift_detector(self) -> None:
        logger.info("=== Phase 8: Shift Detector Setup (%s) ===", self.dataset_name)
        train_features_df = self.splits["train"][
            [c for c in (self.dataset_spec.numerical_features + self.dataset_spec.categorical_features)
             if c in self.splits["train"].columns]
        ]
        self.shift_detector = ShiftDetector(
            ks_alpha=self.config.get("shift", {}).get("ks_alpha", 0.05),
            psi_threshold=self.config.get("shift", {}).get("psi_thresholds", {}).get("moderate", 0.2),
            mmd_subsample=self.config.get("shift", {}).get("mmd_subsample", 2000),
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        self.shift_detector.fit(
            train_features_df,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )

    def setup_bbse(self) -> None:
        logger.info("=== Phase 9: BBSE Setup (%s) ===", self.dataset_name)
        if self.default_model_name is None:
            logger.warning("No default model; skipping BBSE")
            return

        X_pol, y_pol = prepare_features_target(
            self.splits["policy"],
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        X_pol_t = transform_data(self.preprocessor, X_pol)

        model = self.trained_models[self.default_model_name]
        preds_pol, _ = predict(model, X_pol_t)

        self.bbse = BBSEEstimator(n_classes=2)
        self.bbse.fit(y_pol, preds_pol)

    # ── Phase 6: Shift Experiments ─────────────────────────────────

    def run_shift_experiments(self, seeds: Optional[List[int]] = None) -> pd.DataFrame:
        logger.info("=== Phase 10: Shift Experiments (%s) ===", self.dataset_name)
        if seeds is None:
            seeds = [self.seed]

        model_name = self.default_model_name or "random_forest"
        model = self.trained_models.get(model_name)
        if model is None:
            logger.error("Model %s not found", model_name)
            return pd.DataFrame()

        calibrator = self.calibrators.get(model_name)
        cc = self.conformal_classifiers.get(model_name)
        alpha = self.config.get("conformal", {}).get("alpha", 0.10)

        scenarios = ["source", "covariate", "label", "mixed", "severe"]
        severities = ["mild", "moderate", "severe"]
        all_results = []

        for seed in seeds:
            for scenario in scenarios:
                sev_list = ["none"] if scenario == "source" else severities
                for severity in sev_list:
                    try:
                        result = self._run_single_experiment(
                            model, model_name, calibrator, cc, alpha,
                            scenario, severity, seed,
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            "Experiment failed [%s]: %s/%s/seed=%d: %s",
                            self.dataset_name, scenario, severity, seed, e, exc_info=True
                        )

        results_df = pd.DataFrame(all_results)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_dir / f"{self.dataset_name}_shift_experiment_results.csv", index=False)
        if self.dataset_name == "adult":
            results_df.to_csv(self.results_dir / "shift_experiment_results.csv", index=False)

        # Plots
        fig_dir = str(self.figures_dir)
        if len(results_df) > 0:
            plot_coverage_by_scenario(results_df, fig_dir=fig_dir)
            plot_set_size_by_scenario(results_df, fig_dir=fig_dir)
            plot_accuracy_under_shift(results_df, fig_dir=fig_dir)

        return results_df

    def _run_single_experiment(
        self, model, model_name, calibrator, cc, alpha,
        scenario, severity, seed,
    ) -> Dict:
        """Run one experiment configuration."""
        run_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Strict test isolation: ALL test-time scenarios (source and shifted) use held-out test_df
        # This guarantees 0% overlap with training/calibration splits.
        source_df = self.test_df
        shifted_df, shift_meta = generate_shift(
            source_df,
            scenario=scenario,
            severity=severity,
            seed=seed,
            target_col=self.dataset_spec.target_col,
            covariate_cols=self.dataset_spec.covariate_cols,
            perturb_cols=self.dataset_spec.perturb_cols,
        )

        X_shifted, y_shifted = prepare_features_target(
            shifted_df,
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        X_shifted_t = transform_data(self.preprocessor, X_shifted)

        # Predict
        preds, probas = predict(model, X_shifted_t)

        # Calibrate
        if calibrator is not None:
            cal_probas = calibrator.transform(probas)
        else:
            cal_probas = probas

        # Classification metrics
        cls_m = classification_metrics(y_shifted, preds, probas)
        cal_m = calibration_metrics(y_shifted, cal_probas)

        # Conformal
        conf_m = {}
        if cc is not None:
            pred_sets = cc.predict_sets(cal_probas)
            conf_m = conformal_metrics(y_shifted, pred_sets, alpha)

        # Shift detection (features only, no target labels)
        shift_report = self.shift_detector.detect(X_shifted) if self.shift_detector else {}

        # Selective prediction (using max confidence)
        confidence = np.max(cal_probas, axis=1)
        threshold = self.config.get("decision", {}).get("confidence_threshold", 0.70)
        accepted = confidence >= threshold
        sel_m = selective_prediction_metrics(y_shifted, preds, accepted)

        # Risk-coverage
        coverages, risks = risk_coverage_curve(y_shifted, preds, confidence)
        aurc = area_under_risk_coverage(coverages, risks)

        elapsed = time.time() - start

        result = {
            "run_id": run_id,
            "dataset": self.dataset_name,
            "model_name": model_name,
            "scenario": scenario,
            "severity": severity,
            "seed": seed,
            "n_samples": len(y_shifted),
            **cls_m,
            **cal_m,
            **conf_m,
            **sel_m,
            "aurc": aurc,
            "shift_detected": shift_report.get("overall", {}).get("shift_detected", False),
            "shift_severity_est": shift_report.get("overall", {}).get("severity_estimate", "unknown"),
            "shift_severity_score": shift_report.get("overall", {}).get("severity_score", 0),
            "mmd_value": shift_report.get("mmd", {}).get("value", 0),
            "max_psi": shift_report.get("overall", {}).get("max_psi", 0),
            "elapsed_s": round(elapsed, 2),
        }

        # Shift analysis plot for first seed
        if seed == self.seed and shift_report:
            fig_dir = str(self.figures_dir)
            plot_shift_analysis(
                shift_report,
                f"shift_analysis_{self.dataset_name}_{scenario}_{severity}",
                fig_dir,
            )

        logger.info(
            "[%s:%s] %s/%s seed=%d: acc=%.4f f1=%.4f cov=%.4f abs=%.3f",
            self.dataset_name, run_id, scenario, severity, seed,
            cls_m["accuracy"], cls_m["f1"],
            conf_m.get("empirical_coverage", 0),
            sel_m.get("abstention_rate", 0),
        )

        return result

    # ── Phase 7: SAUP Decision Experiments ─────────────────────────

    def run_saup_experiments(self, seeds: Optional[List[int]] = None) -> pd.DataFrame:
        logger.info("=== Phase 11: SAUP Decision Experiments (%s) ===", self.dataset_name)
        if seeds is None:
            seeds = [self.seed]

        model_name = self.default_model_name or "random_forest"
        model = self.trained_models.get(model_name)
        if model is None:
            return pd.DataFrame()

        calibrator = self.calibrators.get(model_name)
        cc = self.conformal_classifiers.get(model_name)

        # Build SAUP engine
        decision_cfg = self.config.get("decision", {})
        engine = SAUPDecisionEngine(
            model=model,
            preprocessor=self.preprocessor,
            calibrator=calibrator or TemperatureScaler(),
            conformal=cc or SplitConformalClassifier(),
            shift_detector=self.shift_detector or ShiftDetector(),
            bbse=self.bbse,
            confidence_threshold=decision_cfg.get("confidence_threshold", 0.70),
            set_size_threshold=decision_cfg.get("set_size_threshold", 1),
            abstain_shift_severity=decision_cfg.get("abstain_shift_severity", 0.8),
            min_ess_ratio=decision_cfg.get("min_ess_ratio", 0.1),
        )

        scenarios = ["source", "covariate", "label", "mixed", "severe"]
        severities = ["mild", "moderate", "severe"]
        all_results = []

        # Source predictions/labels for BBSE
        X_pol, y_pol = prepare_features_target(
            self.splits["policy"],
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        X_pol_t = transform_data(self.preprocessor, X_pol)
        preds_pol, _ = predict(model, X_pol_t)

        for seed in seeds:
            for scenario in scenarios:
                sev_list = ["none"] if scenario == "source" else severities
                for severity in sev_list:
                    try:
                        # Strict test isolation: sample from held-out test_df only
                        source_df = self.test_df
                        shifted_df, shift_meta = generate_shift(
                            source_df,
                            scenario=scenario,
                            severity=severity,
                            seed=seed,
                            target_col=self.dataset_spec.target_col,
                            covariate_cols=self.dataset_spec.covariate_cols,
                            perturb_cols=self.dataset_spec.perturb_cols,
                        )

                        feature_cols = [
                            c for c in (self.dataset_spec.numerical_features + self.dataset_spec.categorical_features)
                            if c in shifted_df.columns
                        ]
                        X_shifted_df = shifted_df[feature_cols]
                        y_shifted = shifted_df[self.dataset_spec.target_col].values.astype(int)

                        results, shift_report = engine.predict_batch(
                            X_shifted_df,
                            X_source_transformed=self.X_train,
                            source_preds=preds_pol,
                            source_labels=y_pol,
                        )

                        # Compute metrics
                        summary = engine.summarize_results(results)
                        predicted = np.array([r.predicted_class for r in results])
                        accepted_mask = np.array([r.action == DecisionAction.ACCEPT for r in results])

                        cls_m = classification_metrics(y_shifted, predicted)
                        sel_m = selective_prediction_metrics(y_shifted, predicted, accepted_mask)

                        pred_sets = [r.prediction_set for r in results]
                        alpha = self.config.get("conformal", {}).get("alpha", 0.10)
                        conf_m = conformal_metrics(y_shifted, pred_sets, alpha)

                        row = {
                            "variant": "full_saup",
                            "dataset": self.dataset_name,
                            "model_name": model_name,
                            "scenario": scenario,
                            "severity": severity,
                            "seed": seed,
                            "n_samples": len(y_shifted),
                            **cls_m,
                            **conf_m,
                            "accepted_fraction": summary["accepted_fraction"],
                            "prediction_set_fraction": summary["prediction_set_fraction"],
                            "abstention_rate": summary["abstention_rate"],
                            "accept_rate": summary["accepted_fraction"],
                            "set_rate": summary["prediction_set_fraction"],
                            "abstain_rate": summary["abstention_rate"],
                            "accepted_prediction_accuracy": sel_m["accepted_prediction_accuracy"],
                            "selective_accuracy": sel_m["selective_accuracy"],
                            "selective_risk": sel_m["selective_risk"],
                            "accepted_coverage": sel_m["accepted_coverage"],
                            "mean_confidence": summary["mean_confidence"],
                            "adaptation_method": summary["adaptation_method"],
                            "shift_severity_score": summary["shift_severity"],
                        }
                        all_results.append(row)

                        logger.info(
                            "SAUP [%s] %s/%s: acc=%.4f accept=%.2f abstain=%.2f",
                            self.dataset_name, scenario, severity,
                            cls_m["accuracy"], summary["accepted_fraction"], summary["abstention_rate"],
                        )
                    except Exception as e:
                        logger.error(
                            "SAUP experiment failed [%s] %s/%s: %s",
                            self.dataset_name, scenario, severity, e, exc_info=True
                        )

        results_df = pd.DataFrame(all_results)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_dir / f"{self.dataset_name}_saup_results.csv", index=False)
        if self.dataset_name == "adult":
            results_df.to_csv(self.results_dir / "saup_results.csv", index=False)

        fig_dir = str(self.figures_dir)
        if len(results_df) > 0 and "abstention_rate" in results_df.columns:
            plot_abstention_vs_severity(
                results_df,
                f"saup_abstention_vs_severity_{self.dataset_name}",
                fig_dir,
            )

        return results_df

    # ── Phase 8: Ablation ──────────────────────────────────────────

    def run_ablation(self) -> pd.DataFrame:
        logger.info("=== Phase 12: Ablation Study (%s) ===", self.dataset_name)
        model_name = self.default_model_name or "random_forest"
        model = self.trained_models.get(model_name)
        if model is None:
            return pd.DataFrame()

        calibrator = self.calibrators.get(model_name)
        cc = self.conformal_classifiers.get(model_name)
        alpha = self.config.get("conformal", {}).get("alpha", 0.10)

        # Build full SAUP decision engine for true ablation comparison
        decision_cfg = self.config.get("decision", {})
        engine = SAUPDecisionEngine(
            model=model,
            preprocessor=self.preprocessor,
            calibrator=calibrator or TemperatureScaler(),
            conformal=cc or SplitConformalClassifier(),
            shift_detector=self.shift_detector or ShiftDetector(),
            bbse=self.bbse,
            confidence_threshold=decision_cfg.get("confidence_threshold", 0.70),
            set_size_threshold=decision_cfg.get("set_size_threshold", 1),
            abstain_shift_severity=decision_cfg.get("abstain_shift_severity", 0.8),
            min_ess_ratio=decision_cfg.get("min_ess_ratio", 0.1),
        )

        X_pol, y_pol = prepare_features_target(
            self.splits["policy"],
            target_col=self.dataset_spec.target_col,
            numerical_features=self.dataset_spec.numerical_features,
            categorical_features=self.dataset_spec.categorical_features,
        )
        X_pol_t = transform_data(self.preprocessor, X_pol)
        preds_pol, _ = predict(model, X_pol_t)

        scenarios = ["covariate", "label", "mixed"]
        variants = ["uncalibrated", "calibrated_only", "conformal_only", "full_saup"]
        all_results = []

        for scenario in scenarios:
            # Shift generated from held-out test set
            shifted_df, _ = generate_shift(
                self.test_df,
                scenario=scenario,
                severity="moderate",
                seed=self.seed,
                target_col=self.dataset_spec.target_col,
                covariate_cols=self.dataset_spec.covariate_cols,
                perturb_cols=self.dataset_spec.perturb_cols,
            )
            X_sh, y_sh = prepare_features_target(
                shifted_df,
                target_col=self.dataset_spec.target_col,
                numerical_features=self.dataset_spec.numerical_features,
                categorical_features=self.dataset_spec.categorical_features,
            )
            X_sh_t = transform_data(self.preprocessor, X_sh)
            preds, probas = predict(model, X_sh_t)

            for variant in variants:
                row = {
                    "variant": variant,
                    "dataset": self.dataset_name,
                    "scenario": scenario,
                    "severity": "moderate",
                    "seed": self.seed,
                    "model_name": model_name,
                    "n_samples": len(y_sh),
                }

                if variant == "uncalibrated":
                    cls_m = classification_metrics(y_sh, preds, probas)
                    cal_m = calibration_metrics(y_sh, probas)
                    row.update(cls_m)
                    row.update(cal_m)
                    row.update({
                        "accepted_fraction": 1.0,
                        "prediction_set_fraction": 0.0,
                        "abstention_rate": 0.0,
                        "accepted_coverage": 1.0,
                        "accepted_prediction_accuracy": cls_m["accuracy"],
                        "selective_risk": 1.0 - cls_m["accuracy"],
                    })

                elif variant == "calibrated_only":
                    cal_probas = calibrator.transform(probas) if calibrator else probas
                    cal_preds = np.argmax(cal_probas, axis=1)
                    cls_m = classification_metrics(y_sh, cal_preds, cal_probas)
                    cal_m = calibration_metrics(y_sh, cal_probas)
                    row.update(cls_m)
                    row.update(cal_m)
                    row.update({
                        "accepted_fraction": 1.0,
                        "prediction_set_fraction": 0.0,
                        "abstention_rate": 0.0,
                        "accepted_coverage": 1.0,
                        "accepted_prediction_accuracy": cls_m["accuracy"],
                        "selective_risk": 1.0 - cls_m["accuracy"],
                    })

                elif variant == "conformal_only":
                    cal_probas = calibrator.transform(probas) if calibrator else probas
                    cal_preds = np.argmax(cal_probas, axis=1)
                    cls_m = classification_metrics(y_sh, cal_preds, cal_probas)
                    cal_m = calibration_metrics(y_sh, cal_probas)
                    row.update(cls_m)
                    row.update(cal_m)
                    if cc:
                        pred_sets = cc.predict_sets(cal_probas)
                        conf_m = conformal_metrics(y_sh, pred_sets, alpha)
                        row.update(conf_m)
                    row.update({
                        "accepted_fraction": 0.0,
                        "prediction_set_fraction": 1.0,
                        "abstention_rate": 0.0,
                        "accepted_coverage": 0.0,
                        "accepted_prediction_accuracy": float("nan"),
                        "selective_risk": float("nan"),
                    })

                elif variant == "full_saup":
                    feature_cols = [
                        c for c in (self.dataset_spec.numerical_features + self.dataset_spec.categorical_features)
                        if c in shifted_df.columns
                    ]
                    X_shifted_df = shifted_df[feature_cols]
                    results, shift_report = engine.predict_batch(
                        X_shifted_df,
                        X_source_transformed=self.X_train,
                        source_preds=preds_pol,
                        source_labels=y_pol,
                    )
                    summary = engine.summarize_results(results)
                    predicted = np.array([r.predicted_class for r in results])
                    accepted_mask = np.array([r.action == DecisionAction.ACCEPT for r in results])
                    adapted_probas = np.array([r.calibrated_probability for r in results])

                    cls_m = classification_metrics(y_sh, predicted, adapted_probas)
                    cal_m = calibration_metrics(y_sh, adapted_probas)
                    sel_m = selective_prediction_metrics(y_sh, predicted, accepted_mask)
                    pred_sets = [r.prediction_set for r in results]
                    conf_m = conformal_metrics(y_sh, pred_sets, alpha)

                    row.update(cls_m)
                    row.update(cal_m)
                    row.update(conf_m)
                    row.update({
                        "accepted_fraction": summary["accepted_fraction"],
                        "prediction_set_fraction": summary["prediction_set_fraction"],
                        "abstention_rate": summary["abstention_rate"],
                        "accept_rate": summary["accepted_fraction"],
                        "set_rate": summary["prediction_set_fraction"],
                        "abstain_rate": summary["abstention_rate"],
                        "accepted_prediction_accuracy": sel_m["accepted_prediction_accuracy"],
                        "selective_accuracy": sel_m["selective_accuracy"],
                        "selective_risk": sel_m["selective_risk"],
                        "accepted_coverage": sel_m["accepted_coverage"],
                        "mean_confidence": summary["mean_confidence"],
                        "adaptation_method": summary["adaptation_method"],
                        "shift_severity_score": summary["shift_severity"],
                    })

                all_results.append(row)

        results_df = pd.DataFrame(all_results)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_dir / f"{self.dataset_name}_ablation_results.csv", index=False)
        if self.dataset_name == "adult":
            results_df.to_csv(self.results_dir / "ablation_results.csv", index=False)

        if len(results_df) > 0:
            plot_ablation_study(results_df, fig_dir=str(self.figures_dir))

        return results_df

    # ── Full Pipeline ──────────────────────────────────────────────

    def run_full_pipeline(self, zip_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete SAUP experimental pipeline for this dataset."""
        setup_logging("INFO")
        logger.info("=" * 60)
        logger.info("SAUP Full Pipeline Starting for [%s]", self.dataset_name)
        logger.info("=" * 60)
        pipeline_start = time.time()
        outputs = {}

        try:
            # 1. Download
            self.download_data(zip_path)

            # 2. Load
            self.load_data()

            # 3. Split
            self.split_data()

            # 4. Preprocess
            self.preprocess()

            # 5. Baselines
            outputs["baseline_results"] = self.train_baselines()

            # 6. Calibrate
            outputs["calibration_results"] = self.calibrate()

            # 7. Conformal
            outputs["conformal_results"] = self.setup_conformal()

            # 8. Shift detector
            self.setup_shift_detector()

            # 9. BBSE
            self.setup_bbse()

            # 10. Shift experiments
            seeds = [self.seed] + self.config.get("additional_seeds", [])
            outputs["shift_results"] = self.run_shift_experiments(seeds)

            # 11. SAUP experiments
            outputs["saup_results"] = self.run_saup_experiments([self.seed])

            # 12. Ablation
            outputs["ablation_results"] = self.run_ablation()

            elapsed = time.time() - pipeline_start
            logger.info("=" * 60)
            logger.info("SAUP Pipeline Complete for [%s] in %.1f seconds", self.dataset_name, elapsed)
            logger.info("=" * 60)

            outputs["success"] = True
            outputs["elapsed_s"] = elapsed
            outputs["dataset"] = self.dataset_name

        except Exception as e:
            logger.error("Pipeline failed for [%s]: %s", self.dataset_name, e, exc_info=True)
            outputs["success"] = False
            outputs["error"] = str(e)
            outputs["dataset"] = self.dataset_name

        return outputs
