"""SAUP Adaptive Decision Engine.

Integrates shift detection, calibration, conformal prediction, and
selective abstention into a unified decision framework.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from saup.calibration import TemperatureScaler
from saup.conformal import (
    SplitConformalClassifier,
    WeightedConformalClassifier,
    estimate_density_ratios,
)
from saup.label_shift import BBSEEstimator
from saup.metrics import effective_sample_size
from saup.shift_detection import ShiftDetector

logger = logging.getLogger(__name__)


class DecisionAction(Enum):
    """Possible decision outcomes."""
    ACCEPT = "accept"
    SET_PREDICTION = "set_prediction"
    ABSTAIN = "abstain"


@dataclass
class PredictionResult:
    """Result of SAUP decision engine for a single prediction."""
    action: DecisionAction
    predicted_class: int
    calibrated_probability: np.ndarray
    prediction_set: List[int]
    confidence: float
    shift_severity: float
    ess_ratio: float
    adaptation_method: str
    warnings: List[str] = field(default_factory=list)
    reason: str = ""


class SAUPDecisionEngine:
    """SAUP Adaptive Decision Engine.
    
    Workflow:
    1. Detect shift and estimate severity.
    2. Obtain predicted probabilities.
    3. Apply appropriate calibration/correction.
    4. Generate conformal prediction set.
    5. Apply selective decision policy.
    
    The engine does NOT use true target labels at decision time.
    """
    
    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        calibrator: TemperatureScaler,
        conformal: SplitConformalClassifier,
        shift_detector: ShiftDetector,
        bbse: Optional[BBSEEstimator] = None,
        weighted_conformal: Optional[WeightedConformalClassifier] = None,
        confidence_threshold: float = 0.70,
        set_size_threshold: int = 1,
        abstain_shift_severity: float = 0.8,
        min_ess_ratio: float = 0.1,
    ):
        """
        Args:
            model: Fitted predictive model.
            preprocessor: Fitted preprocessor.
            calibrator: Fitted temperature scaler.
            conformal: Calibrated conformal classifier.
            shift_detector: Fitted shift detector.
            bbse: Optional BBSE estimator.
            weighted_conformal: Optional weighted conformal classifier.
            confidence_threshold: Min confidence for accept.
            set_size_threshold: Max set size for accept.
            abstain_shift_severity: Severity above which to abstain.
            min_ess_ratio: Min ESS ratio for weighted methods.
        """
        self.model = model
        self.preprocessor = preprocessor
        self.calibrator = calibrator
        self.conformal = conformal
        self.shift_detector = shift_detector
        self.bbse = bbse
        self.weighted_conformal = weighted_conformal
        
        self.confidence_threshold = confidence_threshold
        self.set_size_threshold = set_size_threshold
        self.abstain_shift_severity = abstain_shift_severity
        self.min_ess_ratio = min_ess_ratio
    
    def predict_batch(
        self,
        X_target_df: pd.DataFrame,
        X_source_transformed: Optional[np.ndarray] = None,
        source_preds: Optional[np.ndarray] = None,
        source_labels: Optional[np.ndarray] = None,
    ) -> Tuple[List[PredictionResult], Dict]:
        """Run the full SAUP pipeline on a batch of target data.
        
        Args:
            X_target_df: Target features DataFrame (raw, before preprocessing).
            X_source_transformed: Source transformed features (for density ratios).
            source_preds: Source predictions (for BBSE).
            source_labels: Source labels (for BBSE fitting only).
            
        Returns:
            Tuple of (list of PredictionResults, shift_report).
        """
        # Step 1: Shift detection (no target labels used)
        shift_report = self.shift_detector.detect(X_target_df)
        severity_score = shift_report["overall"]["severity_score"]
        shift_detected = shift_report["overall"]["shift_detected"]
        
        # Step 2: Preprocess and predict
        X_target_transformed = self.preprocessor.transform(X_target_df)
        raw_probas = self.model.predict_proba(X_target_transformed)
        
        # Step 3: Determine adaptation strategy
        adaptation_method = "source_calibrated"
        adapted_probas = self.calibrator.transform(raw_probas)
        ess_ratio = 1.0
        
        if shift_detected and severity_score < self.abstain_shift_severity:
            # Determine appropriate adaptation strategy
            ks_detected = shift_report["overall"]["ks_detected"]
            psi_detected = shift_report["overall"]["psi_detected"]
            
            if ks_detected and X_source_transformed is not None:
                # Covariate shift indicated -> importance weighting
                try:
                    _, target_weights, ess = estimate_density_ratios(
                        X_source_transformed, X_target_transformed
                    )
                    ess_ratio = ess / len(X_source_transformed)
                    
                    if ess_ratio >= self.min_ess_ratio:
                        adaptation_method = "importance_weighted"
                    else:
                        adaptation_method = "source_calibrated_cautious"
                except Exception as e:
                    logger.warning("Density ratio estimation failed: %s", e)
                    adaptation_method = "source_calibrated_cautious"
            elif (not ks_detected or not psi_detected) and (
                self.bbse is not None and source_preds is not None and source_labels is not None
            ):
                # Pure label shift indicated (features unchanged) -> BBSE prior correction
                try:
                    target_preds = np.argmax(adapted_probas, axis=1)
                    target_priors, bbse_info = self.bbse.estimate_target_priors(target_preds)
                    
                    if bbse_info.get("reliable", False):
                        corrected = self.bbse.correct_probabilities(
                            adapted_probas, target_priors
                        )
                        adapted_probas = corrected
                        adaptation_method = "bbse_corrected"
                except Exception as e:
                    logger.warning("BBSE correction failed: %s", e)
            else:
                # Mixed or complex shift -> cautious calibration
                adaptation_method = "source_calibrated_cautious"
        
        # Step 4: Generate conformal prediction sets
        prediction_sets = self.conformal.predict_sets(adapted_probas)
        
        # Step 5: Apply decision policy
        results = []
        for i in range(len(X_target_transformed)):
            result = self._make_decision(
                probabilities=adapted_probas[i],
                prediction_set=prediction_sets[i],
                severity_score=severity_score,
                ess_ratio=ess_ratio,
                adaptation_method=adaptation_method,
            )
            results.append(result)
        
        return results, shift_report
    
    def _make_decision(
        self,
        probabilities: np.ndarray,
        prediction_set: List[int],
        severity_score: float,
        ess_ratio: float,
        adaptation_method: str,
    ) -> PredictionResult:
        """Apply the decision policy for a single prediction.
        
        Args:
            probabilities: Calibrated/adapted probabilities for this sample.
            prediction_set: Conformal prediction set.
            severity_score: Estimated shift severity.
            ess_ratio: Effective sample size ratio.
            adaptation_method: Which adaptation was applied.
            
        Returns:
            PredictionResult with action and metadata.
        """
        predicted_class = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))
        set_size = len(prediction_set)
        warnings = []
        
        # Decision logic
        # Rule 1: Abstain under extreme shift
        if severity_score >= self.abstain_shift_severity:
            warnings.append("Extreme shift detected")
            return PredictionResult(
                action=DecisionAction.ABSTAIN,
                predicted_class=predicted_class,
                calibrated_probability=probabilities,
                prediction_set=prediction_set,
                confidence=confidence,
                shift_severity=severity_score,
                ess_ratio=ess_ratio,
                adaptation_method=adaptation_method,
                warnings=warnings,
                reason="Shift severity exceeds abstention threshold",
            )
        
        # Rule 2: Abstain if empty prediction set
        if set_size == 0:
            warnings.append("Empty conformal prediction set")
            return PredictionResult(
                action=DecisionAction.ABSTAIN,
                predicted_class=predicted_class,
                calibrated_probability=probabilities,
                prediction_set=prediction_set,
                confidence=confidence,
                shift_severity=severity_score,
                ess_ratio=ess_ratio,
                adaptation_method=adaptation_method,
                warnings=warnings,
                reason="Empty prediction set indicates high uncertainty",
            )
        
        # Rule 3: Abstain if ESS too low with weighted adaptation
        if "importance" in adaptation_method and ess_ratio < self.min_ess_ratio:
            warnings.append(f"Low ESS ratio: {ess_ratio:.3f}")
            return PredictionResult(
                action=DecisionAction.ABSTAIN,
                predicted_class=predicted_class,
                calibrated_probability=probabilities,
                prediction_set=prediction_set,
                confidence=confidence,
                shift_severity=severity_score,
                ess_ratio=ess_ratio,
                adaptation_method=adaptation_method,
                warnings=warnings,
                reason="Insufficient effective sample size for reliable weighted adaptation",
            )
        
        # Rule 4: Accept if confident and singleton set
        if confidence >= self.confidence_threshold and set_size <= self.set_size_threshold:
            if severity_score > 0:
                warnings.append(f"Some shift detected (severity={severity_score:.2f})")
            return PredictionResult(
                action=DecisionAction.ACCEPT,
                predicted_class=predicted_class,
                calibrated_probability=probabilities,
                prediction_set=prediction_set,
                confidence=confidence,
                shift_severity=severity_score,
                ess_ratio=ess_ratio,
                adaptation_method=adaptation_method,
                warnings=warnings,
                reason="High confidence with small prediction set",
            )
        
        # Rule 5: Return prediction set
        if set_size > self.set_size_threshold:
            warnings.append(f"Prediction set size={set_size}")
        if confidence < self.confidence_threshold:
            warnings.append(f"Low confidence: {confidence:.3f}")
        
        return PredictionResult(
            action=DecisionAction.SET_PREDICTION,
            predicted_class=predicted_class,
            calibrated_probability=probabilities,
            prediction_set=prediction_set,
            confidence=confidence,
            shift_severity=severity_score,
            ess_ratio=ess_ratio,
            adaptation_method=adaptation_method,
            warnings=warnings,
            reason="Moderate uncertainty; returning prediction set",
        )
    
    def summarize_results(
        self, results: List[PredictionResult]
    ) -> Dict[str, Any]:
        """Summarize batch prediction results with mutually exclusive outcomes.
        
        Guarantees: accepted_fraction + prediction_set_fraction + abstention_rate == 1.0.
        
        Args:
            results: List of PredictionResult objects.
            
        Returns:
            Summary dictionary.
        """
        n = len(results)
        if n == 0:
            return {
                "n": 0,
                "accepted_fraction": 0.0,
                "prediction_set_fraction": 0.0,
                "abstention_rate": 1.0,
                "accept_rate": 0.0,
                "set_rate": 0.0,
                "abstain_rate": 1.0,
            }
        
        actions = [r.action for r in results]
        n_accept = sum(1 for a in actions if a == DecisionAction.ACCEPT)
        n_set = sum(1 for a in actions if a == DecisionAction.SET_PREDICTION)
        n_abstain = sum(1 for a in actions if a == DecisionAction.ABSTAIN)
        
        accepted_fraction = n_accept / n
        prediction_set_fraction = n_set / n
        abstention_rate = n_abstain / n
        
        # Verify mutual exclusivity within numerical tolerance
        assert abs((accepted_fraction + prediction_set_fraction + abstention_rate) - 1.0) < 1e-6
        
        return {
            "n": n,
            "accepted_fraction": accepted_fraction,
            "prediction_set_fraction": prediction_set_fraction,
            "abstention_rate": abstention_rate,
            "accept_rate": accepted_fraction,
            "set_rate": prediction_set_fraction,
            "abstain_rate": abstention_rate,
            "mean_confidence": float(np.mean([r.confidence for r in results])),
            "mean_set_size": float(np.mean([len(r.prediction_set) for r in results])),
            "adaptation_method": results[0].adaptation_method if results else "none",
            "shift_severity": results[0].shift_severity if results else 0.0,
        }
