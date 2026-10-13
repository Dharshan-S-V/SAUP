"""SAUP Research Dashboard — Streamlit application."""

import sys
import os
from pathlib import Path

# Add src to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import json

st.set_page_config(
    page_title="SAUP — Shift-Aware Adaptive Uncertainty Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_csv_safe(path: Path) -> pd.DataFrame:
    """Load a CSV file safely, returning empty DataFrame on failure."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {path.name}: {e}")
    return pd.DataFrame()


def main():
    # Sidebar navigation
    st.sidebar.title("🔬 SAUP Dashboard")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Overview",
            "📁 Dataset Explorer",
            "🤖 Model Evaluation",
            "🌊 Distribution Shift Lab",
            "🎯 Uncertainty & Conformal",
            "⚙️ SAUP Decision Explorer",
            "📥 Results & Downloads",
        ],
    )
    
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ **Notice:** The UCI Adult dataset is a public research dataset. "
        "This dashboard is an educational research prototype, NOT a system "
        "for making real-world decisions about individuals."
    )
    
    if "📊 Overview" in page:
        page_overview()
    elif "📁 Dataset" in page:
        page_dataset()
    elif "🤖 Model" in page:
        page_model_evaluation()
    elif "🌊 Distribution" in page:
        page_shift_lab()
    elif "🎯 Uncertainty" in page:
        page_uncertainty()
    elif "⚙️ SAUP" in page:
        page_saup_explorer()
    elif "📥 Results" in page:
        page_downloads()


def page_overview():
    st.title("🔬 SAUP — Shift-Aware Adaptive Uncertainty Prediction")
    st.markdown(
        "**An Adaptive Uncertainty-Aware ML Framework for Reliable Prediction "
        "Under Distribution Shift**"
    )
    
    col1, col2, col3 = st.columns(3)
    
    baseline_df = load_csv_safe(RESULTS_DIR / "baseline_results.csv")
    shift_df = load_csv_safe(RESULTS_DIR / "shift_experiment_results.csv")
    saup_df = load_csv_safe(RESULTS_DIR / "saup_results.csv")
    
    with col1:
        st.metric("Models Trained", len(baseline_df) if len(baseline_df) > 0 else "—")
        if len(baseline_df) > 0:
            best = baseline_df.loc[baseline_df["accuracy"].idxmax()]
            st.metric("Best Accuracy", f"{best['accuracy']:.4f}")
    
    with col2:
        st.metric("Shift Experiments", len(shift_df) if len(shift_df) > 0 else "—")
        if len(shift_df) > 0 and "empirical_coverage" in shift_df.columns:
            avg_cov = shift_df["empirical_coverage"].mean()
            st.metric("Mean Coverage", f"{avg_cov:.4f}")
    
    with col3:
        st.metric("SAUP Runs", len(saup_df) if len(saup_df) > 0 else "—")
        if len(saup_df) > 0 and "abstention_rate" in saup_df.columns:
            avg_abs = saup_df["abstention_rate"].mean()
            st.metric("Mean Abstention Rate", f"{avg_abs:.3f}")
    
    st.markdown("---")
    st.subheader("Research Goal")
    st.markdown(
        "Develop and evaluate a framework combining conventional ML prediction, "
        "distribution-shift detection, uncertainty estimation, probability calibration, "
        "conformal prediction, and adaptive selective abstention to improve "
        "prediction reliability under distribution shift."
    )
    
    # Show figures if available
    overview_fig = FIGURES_DIR / "model_comparison.png"
    if overview_fig.exists():
        st.image(str(overview_fig), caption="Baseline Model Comparison")


def page_dataset():
    st.title("📁 Dataset Explorer")
    
    summary_path = DATA_DIR / "processed" / "preprocessing_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Rows", summary.get("train_rows", "—"))
            st.metric("Test Rows", summary.get("test_rows", "—"))
        with col2:
            st.metric("Features", len(summary.get("numerical_features", [])) + 
                       len(summary.get("categorical_features", [])))
            st.metric("Train Duplicates", summary.get("train_duplicates", "—"))
        
        st.subheader("Feature Types")
        st.markdown(f"**Numerical:** {', '.join(summary.get('numerical_features', []))}")
        st.markdown(f"**Categorical:** {', '.join(summary.get('categorical_features', []))}")
        
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training:**")
            st.json(summary.get("train_label_dist", {}))
        with col2:
            st.markdown("**Test:**")
            st.json(summary.get("test_label_dist", {}))
        
        # Missing values
        st.subheader("Missing Values")
        train_missing = summary.get("train_missing", {})
        missing_cols = {k: v for k, v in train_missing.items() if v > 0}
        if missing_cols:
            st.dataframe(pd.DataFrame([missing_cols]).T.rename(columns={0: "Count"}))
        else:
            st.success("No missing values in training data")
        
        # Show figures
        for fig_name in ["class_distribution_train", "class_distribution_test", "missing_values_train"]:
            fig_path = FIGURES_DIR / f"{fig_name}.png"
            if fig_path.exists():
                st.image(str(fig_path))
        
        st.download_button(
            "📥 Download Preprocessing Summary",
            json.dumps(summary, indent=2),
            "preprocessing_summary.json",
            "application/json",
        )
    else:
        st.warning("No preprocessing summary found. Run the pipeline first.")


def page_model_evaluation():
    st.title("🤖 Model Evaluation")
    
    baseline_df = load_csv_safe(RESULTS_DIR / "baseline_results.csv")
    
    if len(baseline_df) > 0:
        st.subheader("Baseline Comparison")
        display_cols = ["model_name", "accuracy", "f1", "roc_auc", "brier_score", 
                        "ece", "log_loss", "train_time_s"]
        available = [c for c in display_cols if c in baseline_df.columns]
        st.dataframe(baseline_df[available].style.format(
            {c: "{:.4f}" for c in available if c != "model_name"}
        ), use_container_width=True)
        
        # Show figures
        fig_path = FIGURES_DIR / "model_comparison.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="Model Comparison")
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        for _, row in baseline_df.iterrows():
            fig_path = FIGURES_DIR / f"confusion_matrix_{row['model_name']}.png"
            if fig_path.exists():
                st.image(str(fig_path), width=400)
        
        # Calibration
        st.subheader("Calibration")
        cal_fig = FIGURES_DIR / "calibration_comparison.png"
        if cal_fig.exists():
            st.image(str(cal_fig))
        
        for _, row in baseline_df.iterrows():
            name = row["model_name"]
            col1, col2 = st.columns(2)
            uncal = FIGURES_DIR / f"reliability_{name}_uncal.png"
            cal = FIGURES_DIR / f"reliability_{name}_cal.png"
            if uncal.exists():
                with col1:
                    st.image(str(uncal), width=400)
            if cal.exists():
                with col2:
                    st.image(str(cal), width=400)
    else:
        st.warning("No baseline results found. Run `python scripts/run_baselines.py` first.")


def page_shift_lab():
    st.title("🌊 Distribution Shift Lab")
    
    shift_df = load_csv_safe(RESULTS_DIR / "shift_experiment_results.csv")
    
    if len(shift_df) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            scenarios = st.multiselect("Scenario", shift_df["scenario"].unique(),
                                       default=list(shift_df["scenario"].unique()))
        with col2:
            severities = st.multiselect("Severity", shift_df["severity"].unique(),
                                        default=list(shift_df["severity"].unique()))
        with col3:
            if "seed" in shift_df.columns:
                seeds = st.multiselect("Seed", shift_df["seed"].unique(),
                                       default=list(shift_df["seed"].unique()))
            else:
                seeds = []
        
        filtered = shift_df[
            shift_df["scenario"].isin(scenarios) & 
            shift_df["severity"].isin(severities)
        ]
        if seeds and "seed" in shift_df.columns:
            filtered = filtered[filtered["seed"].isin(seeds)]
        
        st.subheader("Results")
        display_cols = ["scenario", "severity", "seed", "accuracy", "f1", "roc_auc",
                        "empirical_coverage", "avg_set_size", "shift_detected",
                        "shift_severity_est", "mmd_value", "max_psi"]
        available = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available].style.format(
            {c: "{:.4f}" for c in available if c not in ("scenario", "severity", "seed", "shift_detected", "shift_severity_est")}
        ), use_container_width=True)
        
        # Figures
        for fig_name in ["coverage_by_scenario", "set_size_by_scenario", 
                          "accuracy_under_shift"]:
            fig_path = FIGURES_DIR / f"{fig_name}.png"
            if fig_path.exists():
                st.image(str(fig_path))
        
        # Shift analysis figures
        st.subheader("Feature-Level Shift Analysis")
        for scenario in scenarios:
            for severity in severities:
                fig_path = FIGURES_DIR / f"shift_analysis_{scenario}_{severity}.png"
                if fig_path.exists():
                    st.image(str(fig_path), caption=f"{scenario} - {severity}")
    else:
        st.warning("No shift experiment results found. Run `python scripts/run_shift_experiments.py` first.")


def page_uncertainty():
    st.title("🎯 Uncertainty & Conformal Prediction")
    
    shift_df = load_csv_safe(RESULTS_DIR / "shift_experiment_results.csv")
    
    if len(shift_df) > 0 and "empirical_coverage" in shift_df.columns:
        st.subheader("Conformal Coverage Across Scenarios")
        
        coverage_summary = shift_df.groupby(["scenario", "severity"]).agg({
            "empirical_coverage": ["mean", "std"],
            "avg_set_size": ["mean", "std"],
            "empty_set_rate": "mean" if "empty_set_rate" in shift_df.columns else "first",
        }).round(4)
        st.dataframe(coverage_summary, use_container_width=True)
        
        col1, col2 = st.columns(2)
        fig1 = FIGURES_DIR / "coverage_by_scenario.png"
        fig2 = FIGURES_DIR / "set_size_by_scenario.png"
        if fig1.exists():
            with col1:
                st.image(str(fig1))
        if fig2.exists():
            with col2:
                st.image(str(fig2))
        
        st.markdown(
            "**Note:** Conformal prediction provides finite-sample marginal coverage "
            "guarantees under exchangeability. Under distribution shift, the "
            "exchangeability assumption is violated, and coverage may degrade."
        )
    else:
        st.warning("No conformal results found. Run experiments first.")


def page_saup_explorer():
    st.title("⚙️ SAUP Decision Explorer")
    
    saup_df = load_csv_safe(RESULTS_DIR / "saup_results.csv")
    
    if len(saup_df) > 0:
        st.subheader("SAUP Results by Scenario")
        
        display_cols = ["scenario", "severity", "accuracy", "accept_rate",
                        "abstention_rate", "selective_accuracy", 
                        "empirical_coverage", "mean_confidence",
                        "adaptation_method", "shift_severity_score"]
        available = [c for c in display_cols if c in saup_df.columns]
        st.dataframe(saup_df[available], use_container_width=True)
        
        # Abstention figure
        fig_path = FIGURES_DIR / "saup_abstention_vs_severity.png"
        if fig_path.exists():
            st.image(str(fig_path))
        
        st.subheader("Decision Breakdown")
        for _, row in saup_df.iterrows():
            with st.expander(f"{row.get('scenario','')} / {row.get('severity','')}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accept Rate", f"{row.get('accept_rate',0):.3f}")
                with col2:
                    st.metric("Set Rate", f"{row.get('set_rate',0):.3f}")
                with col3:
                    st.metric("Abstain Rate", f"{row.get('abstention_rate',0):.3f}")
                
                st.markdown(f"**Adaptation:** {row.get('adaptation_method','—')}")
                st.markdown(f"**Shift Severity Score:** {row.get('shift_severity_score',0):.3f}")
                st.markdown(f"**Selective Accuracy:** {row.get('selective_accuracy',0):.4f}")
    else:
        st.warning("No SAUP results found. Run `python scripts/run_all.py` first.")
    
    # Ablation
    st.subheader("Ablation Study")
    ablation_df = load_csv_safe(RESULTS_DIR / "ablation_results.csv")
    if len(ablation_df) > 0:
        st.dataframe(ablation_df, use_container_width=True)
        fig_path = FIGURES_DIR / "ablation_study.png"
        if fig_path.exists():
            st.image(str(fig_path))


def page_downloads():
    st.title("📥 Results & Downloads")
    
    st.subheader("CSV Results")
    csv_files = [
        "baseline_results.csv",
        "shift_experiment_results.csv",
        "saup_results.csv",
        "ablation_results.csv",
    ]
    for fname in csv_files:
        path = RESULTS_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            st.markdown(f"**{fname}** ({len(df)} rows)")
            with open(path, "rb") as f:
                st.download_button(f"📥 Download {fname}", f, fname, "text/csv")
        else:
            st.caption(f"_{fname} — not yet generated_")
    
    st.subheader("Figures")
    if FIGURES_DIR.exists():
        figures = sorted(FIGURES_DIR.glob("*.png"))
        if figures:
            cols = st.columns(3)
            for i, fig in enumerate(figures):
                with cols[i % 3]:
                    st.image(str(fig), caption=fig.stem, width=300)
        else:
            st.caption("_No figures generated yet_")
    
    st.subheader("Reports")
    for rname in ["experimental_report.md", "methodology.md", "limitations.md"]:
        rpath = REPORTS_DIR / rname
        if rpath.exists():
            with open(rpath, "r", encoding="utf-8") as f:
                content = f.read()
            with st.expander(f"📄 {rname}"):
                st.markdown(content)
            st.download_button(f"📥 {rname}", content, rname, "text/markdown")


if __name__ == "__main__":
    main()
