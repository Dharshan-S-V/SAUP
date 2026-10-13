"""Publication-quality visualization for SAUP experiments."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from saup.config import resolve_path
from saup.metrics import reliability_diagram_data

logger = logging.getLogger(__name__)

# Style
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
})
sns.set_style("whitegrid")


def save_fig(fig: plt.Figure, name: str, fig_dir: str = "reports/figures") -> Path:
    """Save figure and return path."""
    out_dir = resolve_path(fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Figure saved: %s", path)
    return path


def plot_class_distribution(
    labels: np.ndarray,
    title: str = "Class Distribution",
    name: str = "class_distribution",
    fig_dir: str = "reports/figures",
    class_names: Optional[Dict[int, str]] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    unique, counts = np.unique(labels, return_counts=True)
    colors = ["#3498db", "#e74c3c"]
    
    if class_names:
        tick_labels = [f"{class_names.get(int(u), f'Class {u}')} ({u})" for u in unique]
    elif len(unique) == 2 and list(unique) == [0, 1]:
        tick_labels = ["Negative (0)", "Positive (1)"]
    else:
        tick_labels = [f"Class {u}" for u in unique]
        
    bars = ax.bar(tick_labels, counts, color=colors[:len(unique)], edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{count}\n({count/sum(counts)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel("Count")
    return save_fig(fig, name, fig_dir)


def plot_missing_values(missing_dict: Dict[str, int], title: str = "Missing Values",
                        name: str = "missing_values", fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    cols = [k for k, v in missing_dict.items() if v > 0]
    vals = [missing_dict[k] for k in cols]
    if not cols:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14, transform=ax.transAxes)
    else:
        ax.barh(cols, vals, color="#e67e22", edgecolor="black", alpha=0.85)
        ax.set_xlabel("Count")
    ax.set_title(title)
    return save_fig(fig, name, fig_dir)


def plot_model_comparison(results_df, metric_cols: List[str] = None,
                          name: str = "model_comparison", fig_dir: str = "reports/figures") -> Path:
    if metric_cols is None:
        metric_cols = ["accuracy", "f1", "roc_auc", "brier_score"]
    
    available = [c for c in metric_cols if c in results_df.columns]
    n_metrics = len(available)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    models = results_df["model_name"].unique()
    colors = sns.color_palette("Set2", len(models))
    
    for i, metric in enumerate(available):
        for j, model in enumerate(models):
            val = results_df[results_df["model_name"] == model][metric].values
            if len(val) > 0:
                axes[i].bar(j, val[0], color=colors[j], edgecolor="black", alpha=0.85, label=model if i == 0 else "")
                axes[i].text(j, val[0] + 0.005, f"{val[0]:.3f}", ha="center", fontsize=8)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].set_xticks(range(len(models)))
        axes[i].set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    
    fig.suptitle("Baseline Model Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix",
                          name: str = "confusion_matrix", fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return save_fig(fig, name, fig_dir)


def plot_reliability_diagram(y_true: np.ndarray, prob_pos: np.ndarray,
                             title: str = "Reliability Diagram",
                             name: str = "reliability_diagram", fig_dir: str = "reports/figures") -> Path:
    centers, accuracies, counts = reliability_diagram_data(y_true, prob_pos)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})
    
    # Main diagram
    mask = counts > 0
    ax1.bar(centers[mask], accuracies[mask], width=1.0/len(centers)*0.8, alpha=0.7,
            color="#3498db", edgecolor="black", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7)
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(title)
    ax1.legend()
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    
    # Histogram
    ax2.bar(centers, counts, width=1.0/len(centers)*0.8, color="#95a5a6", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Mean Predicted Probability")
    ax2.set_ylabel("Count")
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_calibration_comparison(results: Dict[str, Dict], 
                                name: str = "calibration_comparison", fig_dir: str = "reports/figures") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["brier_score", "ece", "log_loss"]
    titles = ["Brier Score", "ECE", "Log Loss"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        labels = list(results.keys())
        values = [results[k].get(metric, 0) for k in labels]
        colors = ["#e74c3c" if "uncal" in k.lower() else "#2ecc71" for k in labels]
        axes[i].bar(labels, values, color=colors, edgecolor="black", alpha=0.85)
        for j, v in enumerate(values):
            axes[i].text(j, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
        axes[i].set_title(title)
        axes[i].tick_params(axis="x", rotation=30)
    
    fig.suptitle("Calibration Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_shift_analysis(shift_report: Dict, 
                        name: str = "shift_analysis", fig_dir: str = "reports/figures") -> Path:
    ks_data = shift_report.get("ks_results", {})
    psi_data = shift_report.get("psi_results", {})
    
    features = [k for k in ks_data if k not in ("n_rejected", "n_tested")]
    
    if not features:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No shift data", ha="center", va="center", transform=ax.transAxes)
        return save_fig(fig, name, fig_dir)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # KS statistics
    ks_vals = [ks_data[f]["statistic"] for f in features]
    colors = ["#e74c3c" if ks_data[f].get("rejected", False) else "#2ecc71" for f in features]
    ax1.barh(features, ks_vals, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xlabel("KS Statistic")
    ax1.set_title("KS Test by Feature")
    ax1.axvline(x=0.05, color="gray", linestyle="--", alpha=0.5, label="Reference")
    
    # PSI
    psi_features = [f for f in psi_data if isinstance(psi_data[f], dict)]
    psi_vals = [psi_data[f]["psi"] for f in psi_features]
    psi_colors = []
    for f in psi_features:
        flag = psi_data[f]["flag"]
        psi_colors.append("#e74c3c" if flag == "major" else "#f39c12" if flag == "moderate" else "#2ecc71")
    ax2.barh(psi_features, psi_vals, color=psi_colors, edgecolor="black", alpha=0.85)
    ax2.set_xlabel("PSI")
    ax2.set_title("PSI by Feature")
    ax2.axvline(x=0.1, color="orange", linestyle="--", alpha=0.5, label="Moderate")
    ax2.axvline(x=0.25, color="red", linestyle="--", alpha=0.5, label="Major")
    ax2.legend()
    
    fig.suptitle(f"Shift Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_coverage_by_scenario(results_df, name: str = "coverage_by_scenario",
                              fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "scenario" not in results_df.columns or "empirical_coverage" not in results_df.columns:
        ax.text(0.5, 0.5, "No coverage data", ha="center", transform=ax.transAxes)
        return save_fig(fig, name, fig_dir)
    
    scenarios = results_df.groupby(["scenario", "severity"])["empirical_coverage"].mean().reset_index()
    x_labels = [f"{r['scenario']}\n{r['severity']}" for _, r in scenarios.iterrows()]
    
    colors = sns.color_palette("Set2", len(x_labels))
    bars = ax.bar(range(len(x_labels)), scenarios["empirical_coverage"].values,
                  color=colors, edgecolor="black", alpha=0.85)
    
    ax.axhline(y=0.9, color="red", linestyle="--", label="90% nominal", alpha=0.7)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Conformal Coverage by Shift Scenario")
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_set_size_by_scenario(results_df, name: str = "set_size_by_scenario",
                              fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "scenario" not in results_df.columns or "avg_set_size" not in results_df.columns:
        ax.text(0.5, 0.5, "No set size data", ha="center", transform=ax.transAxes)
        return save_fig(fig, name, fig_dir)
    
    scenarios = results_df.groupby(["scenario", "severity"])["avg_set_size"].mean().reset_index()
    x_labels = [f"{r['scenario']}\n{r['severity']}" for _, r in scenarios.iterrows()]
    
    ax.bar(range(len(x_labels)), scenarios["avg_set_size"].values,
           color=sns.color_palette("Set3", len(x_labels)), edgecolor="black", alpha=0.85)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Average Prediction Set Size")
    ax.set_title("Prediction Set Size by Shift Scenario")
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_risk_coverage_curve(coverages: np.ndarray, risks: np.ndarray,
                             name: str = "risk_coverage", fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    order = np.argsort(coverages)
    ax.plot(coverages[order], risks[order], "b-", linewidth=2)
    ax.fill_between(coverages[order], risks[order], alpha=0.15, color="blue")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk (Error Rate)")
    ax.set_title("Risk-Coverage Curve")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(risks) * 1.1 + 0.01)
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_accuracy_under_shift(results_df, name: str = "accuracy_under_shift",
                              fig_dir: str = "reports/figures") -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for scenario in results_df["scenario"].unique():
        subset = results_df[results_df["scenario"] == scenario]
        sev_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        subset = subset.copy()
        subset["sev_order"] = subset["severity"].map(sev_order)
        subset = subset.sort_values("sev_order")
        
        if len(subset) > 0:
            ax1.plot(subset["severity"], subset["accuracy"], "o-", label=scenario, markersize=6)
            ax2.plot(subset["severity"], subset["f1"], "o-", label=scenario, markersize=6)
    
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy Under Increasing Shift")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=30)
    
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Under Increasing Shift")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_abstention_vs_severity(results_df, name: str = "abstention_vs_severity",
                                fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if "abstention_rate" not in results_df.columns:
        ax.text(0.5, 0.5, "No abstention data", ha="center", transform=ax.transAxes)
        return save_fig(fig, name, fig_dir)
    
    for scenario in results_df["scenario"].unique():
        subset = results_df[results_df["scenario"] == scenario]
        sev_order = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        subset = subset.copy()
        subset["sev_order"] = subset["severity"].map(sev_order)
        subset = subset.sort_values("sev_order")
        
        if len(subset) > 0:
            ax.plot(subset["severity"], subset["abstention_rate"], "o-", label=scenario, markersize=6)
    
    ax.set_ylabel("Abstention Rate")
    ax.set_title("SAUP Abstention Rate vs Shift Severity")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_saup_vs_baseline(results_df, name: str = "saup_vs_baseline",
                          fig_dir: str = "reports/figures") -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["accuracy", "empirical_coverage", "abstention_rate"]
    titles = ["Accuracy", "Conformal Coverage", "Abstention Rate"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if metric not in results_df.columns:
            axes[i].text(0.5, 0.5, f"No {metric} data", ha="center", transform=axes[i].transAxes)
            continue
        
        for variant in results_df["variant"].unique() if "variant" in results_df.columns else ["baseline"]:
            subset = results_df[results_df["variant"] == variant] if "variant" in results_df.columns else results_df
            if len(subset) > 0 and metric in subset.columns:
                vals = subset.groupby("scenario")[metric].mean()
                axes[i].bar(range(len(vals)), vals.values, alpha=0.7, label=variant)
        
        axes[i].set_title(title)
        axes[i].legend(fontsize=7)
        axes[i].tick_params(axis="x", rotation=30)
    
    fig.suptitle("SAUP vs Baseline Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def plot_ablation_study(results_df, name: str = "ablation_study",
                        fig_dir: str = "reports/figures") -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if "variant" not in results_df.columns:
        ax.text(0.5, 0.5, "No ablation data", ha="center", transform=ax.transAxes)
        return save_fig(fig, name, fig_dir)
    
    variants = results_df["variant"].unique()
    metrics = ["accuracy", "f1", "empirical_coverage"]
    available = [m for m in metrics if m in results_df.columns]
    
    x = np.arange(len(variants))
    width = 0.8 / max(len(available), 1)
    
    for i, metric in enumerate(available):
        vals = [results_df[results_df["variant"] == v][metric].mean() for v in variants]
        ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(), alpha=0.85)
    
    ax.set_xticks(x + width * len(available) / 2)
    ax.set_xticklabels([v.replace("_", "\n") for v in variants], fontsize=8)
    ax.set_title("Ablation Study")
    ax.legend()
    
    fig.tight_layout()
    return save_fig(fig, name, fig_dir)


def generate_all_figures(results_dir: str = "reports/results",
                         fig_dir: str = "reports/figures") -> List[Path]:
    """Generate all figures from saved CSV results."""
    import pandas as pd
    
    results_path = resolve_path(results_dir)
    figures = []
    
    # Try to load and plot each result file
    baseline_file = results_path / "baseline_results.csv"
    if baseline_file.exists():
        df = pd.read_csv(baseline_file)
        figures.append(plot_model_comparison(df, fig_dir=fig_dir))
    
    shift_file = results_path / "shift_experiment_results.csv"
    if shift_file.exists():
        df = pd.read_csv(shift_file)
        if len(df) > 0:
            figures.append(plot_coverage_by_scenario(df, fig_dir=fig_dir))
            figures.append(plot_set_size_by_scenario(df, fig_dir=fig_dir))
            figures.append(plot_accuracy_under_shift(df, fig_dir=fig_dir))
            if "abstention_rate" in df.columns:
                figures.append(plot_abstention_vs_severity(df, fig_dir=fig_dir))
    
    ablation_file = results_path / "ablation_results.csv"
    if ablation_file.exists():
        df = pd.read_csv(ablation_file)
        figures.append(plot_ablation_study(df, fig_dir=fig_dir))
    
    logger.info("Generated %d figures", len(figures))
    return figures
