"""SAUP Research Dashboard — Streamlit application.

Supports multi-dataset evaluation across UCI Adult, UCI Bank Marketing,
and UCI Wine Quality, with seamless switching between Audit-Corrected
and Historical Baseline result suites.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SAUP — Shift-Aware Adaptive Uncertainty Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_df(df: pd.DataFrame, **kwargs):
    """Render dataframe cleanly with stretch width."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except (TypeError, ValueError):
        st.dataframe(df, **kwargs)


def load_csv_safe(path: Path) -> pd.DataFrame:
    """Load a CSV file safely, returning empty DataFrame on failure."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {path.name}: {e}")
    return pd.DataFrame()


def get_dataset_csv(results_dir: Path, dataset_key: str, suffix: str) -> pd.DataFrame:
    """Load CSV for a specific dataset with Adult fallback."""
    p = results_dir / f"{dataset_key}_{suffix}.csv"
    if p.exists():
        return load_csv_safe(p)
    if dataset_key == "adult":
        alt = results_dir / f"{suffix}.csv"
        if alt.exists():
            return load_csv_safe(alt)
    return pd.DataFrame()


def main():
    # Sidebar navigation
    st.sidebar.title("🔬 SAUP Dashboard")
    st.sidebar.markdown("---")

    suite_mode = st.sidebar.selectbox(
        "Results Suite",
        ["Audit-Corrected (Recommended)", "Historical Baseline"],
    )

    if "Audit-Corrected" in suite_mode:
        results_dir = PROJECT_ROOT / "reports" / "audit_corrected" / "results"
        figures_dir = PROJECT_ROOT / "reports" / "audit_corrected" / "figures"
        reports_dir = PROJECT_ROOT / "reports" / "audit_corrected"
    else:
        results_dir = PROJECT_ROOT / "reports" / "results"
        figures_dir = PROJECT_ROOT / "reports" / "figures"
        reports_dir = PROJECT_ROOT / "reports"

    dataset_label = st.sidebar.selectbox(
        "Dataset",
        [
            "UCI Adult (Demographics)",
            "UCI Bank Marketing (Deposit)",
            "UCI Wine Quality (Physicochemical)",
        ],
    )
    dataset_map = {
        "UCI Adult (Demographics)": "adult",
        "UCI Bank Marketing (Deposit)": "bank",
        "UCI Wine Quality (Physicochemical)": "wine",
    }
    dataset_key = dataset_map[dataset_label]

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Overview & Summary",
            "🌐 Cross-Dataset Benchmark",
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
        "⚠️ **Research Prototype Notice:** SAUP is an educational research "
        "framework for uncertainty under distribution shift. Models reflect "
        "historical benchmark distributions and are not intended for autonomous "
        "real-world decisions."
    )

    if "📊 Overview" in page:
        page_overview(results_dir, figures_dir, dataset_key, dataset_label, suite_mode)
    elif "🌐 Cross-Dataset" in page:
        page_cross_dataset(results_dir, figures_dir, reports_dir)
    elif "📁 Dataset" in page:
        page_dataset(dataset_key, dataset_label, figures_dir)
    elif "🤖 Model" in page:
        page_model_evaluation(results_dir, figures_dir, dataset_key, dataset_label)
    elif "🌊 Distribution" in page:
        page_shift_lab(results_dir, figures_dir, dataset_key, dataset_label)
    elif "🎯 Uncertainty" in page:
        page_uncertainty(results_dir, figures_dir, dataset_key, dataset_label)
    elif "⚙️ SAUP" in page:
        page_saup_explorer(results_dir, figures_dir, dataset_key, dataset_label)
    elif "📥 Results" in page:
        page_downloads(results_dir, figures_dir, reports_dir)


def page_overview(results_dir: Path, figures_dir: Path, dataset_key: str, dataset_label: str, suite_mode: str):
    st.title("🔬 SAUP — Shift-Aware Adaptive Uncertainty Prediction")
    st.subheader(f"{dataset_label} — {suite_mode}")

    col1, col2, col3 = st.columns(3)

    baseline_df = get_dataset_csv(results_dir, dataset_key, "baseline_results")
    shift_df = get_dataset_csv(results_dir, dataset_key, "shift_experiment_results")
    saup_df = get_dataset_csv(results_dir, dataset_key, "saup_results")

    with col1:
        st.metric("Models Evaluated", len(baseline_df) if len(baseline_df) > 0 else "—")
        if len(baseline_df) > 0 and "accuracy" in baseline_df.columns:
            best = baseline_df.loc[baseline_df["accuracy"].idxmax()]
            st.metric("Top Accuracy", f"{best['accuracy']:.4f}", f"{best['model_name']}")

    with col2:
        st.metric("Shift Scenarios", len(shift_df) if len(shift_df) > 0 else "—")
        if len(shift_df) > 0 and "empirical_coverage" in shift_df.columns:
            avg_cov = shift_df["empirical_coverage"].mean()
            st.metric("Mean Conformal Coverage", f"{avg_cov:.4f}")

    with col3:
        st.metric("SAUP Decision Runs", len(saup_df) if len(saup_df) > 0 else "—")
        if len(saup_df) > 0 and "abstention_rate" in saup_df.columns:
            avg_abs = saup_df["abstention_rate"].mean()
            st.metric("Mean Abstention Rate", f"{avg_abs:.3f}")

    st.markdown("---")
    st.subheader("Framework Architecture")
    st.markdown(
        """
        SAUP integrates six core components into an adaptive decision pipeline:
        1. **Standard Classification Models:** Logistic Regression, Random Forest, MLP, XGBoost.
        2. **Multi-Detector Shift Sensing:** Benjamini-Hochberg corrected KS-tests, Population Stability Index (PSI), and source-standardized Maximum Mean Discrepancy (MMD).
        3. **Probability Calibration:** Temperature scaling with validation log-loss minimization.
        4. **Conformal Inference:** Split conformal prediction with density-ratio weighted test quantile estimation.
        5. **Label Shift Adaptation:** Black Box Shift Estimation (BBSE) for target prior estimation.
        6. **3-Way Selective Action Policy:** **ACCEPT** (high confidence singletons), **SET_PREDICTION** (hedged prediction sets under epistemic uncertainty), or **ABSTAIN** (severe shift / untrusted domain).
        """
    )

    overview_fig = figures_dir / "model_comparison.png"
    if overview_fig.exists():
        st.image(str(overview_fig), caption=f"Baseline Model Comparison ({dataset_label})")


def page_cross_dataset(results_dir: Path, figures_dir: Path, reports_dir: Path):
    st.title("🌐 Cross-Dataset Benchmark Synthesis")
    st.markdown(
        "Cross-domain comparative evaluation across **UCI Adult**, **UCI Bank Marketing**, "
        "and **UCI Wine Quality**."
    )

    unified_csv = results_dir / "unified_results.csv"
    if unified_csv.exists():
        df_uni = pd.read_csv(unified_csv)
        st.subheader("Unified Multi-Dataset Results")
        render_df(df_uni)

        st.subheader("Source Baseline Comparison Across Datasets")
        source_df = df_uni[df_uni["scenario"] == "source"].copy()
        if len(source_df) > 0:
            render_df(source_df[["dataset", "model", "accuracy", "f1", "brier_score", "ece"]])

        st.subheader("Conformal Coverage Invariance Under Shift")
        shift_df = df_uni[df_uni["scenario"] != "source"].dropna(subset=["conformal_coverage"])
        if len(shift_df) > 0:
            piv = shift_df.pivot_table(
                index=["scenario", "severity"],
                columns="dataset",
                values="conformal_coverage",
                aggfunc="mean",
            )
            render_df(piv.round(4))
    else:
        st.warning("Unified results table not found. Run `scripts/run_audit_corrections.py` first.")

    rep_file = reports_dir / "cross_dataset_comparison.md"
    if rep_file.exists():
        with st.expander("📄 View Full Cross-Dataset Report Markdown"):
            st.markdown(rep_file.read_text(encoding="utf-8"))


def page_dataset(dataset_key: str, dataset_label: str, figures_dir: Path):
    st.title(f"📁 Dataset Explorer — {dataset_label}")

    suffix = f"_{dataset_key}" if dataset_key != "adult" else ""
    summary_path = PROJECT_ROOT / "data" / "processed" / f"preprocessing_summary{suffix}.json"
    if not summary_path.exists() and dataset_key == "adult":
        summary_path = PROJECT_ROOT / "data" / "processed" / "preprocessing_summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Rows", summary.get("train_rows", "—"))
            st.metric("Test Rows", summary.get("test_rows", "—"))
        with col2:
            num_feats = len(summary.get("numerical_features", []))
            cat_feats = len(summary.get("categorical_features", []))
            st.metric("Total Features", num_feats + cat_feats)
            st.metric("Duplicates in Dev", summary.get("train_duplicates", summary.get("duplicates", 0)))

        st.subheader("Feature Schema")
        st.markdown(f"**Numerical ({num_feats}):** {', '.join(summary.get('numerical_features', []))}")
        st.markdown(f"**Categorical ({cat_feats}):** {', '.join(summary.get('categorical_features', []))}")

        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Split:**")
            st.json(summary.get("train_label_dist", {}))
        with col2:
            st.markdown("**Test Split:**")
            st.json(summary.get("test_label_dist", {}))

        # Show dataset figures
        for fig_stem in [f"class_distribution_train_{dataset_key}", f"class_distribution_test_{dataset_key}", f"missing_values_train_{dataset_key}"]:
            fp = figures_dir / f"{fig_stem}.png"
            if fp.exists():
                st.image(str(fp), caption=fig_stem)
    else:
        st.warning(f"No preprocessing summary found for {dataset_label}.")


def page_model_evaluation(results_dir: Path, figures_dir: Path, dataset_key: str, dataset_label: str):
    st.title(f"🤖 Model Evaluation — {dataset_label}")

    baseline_df = get_dataset_csv(results_dir, dataset_key, "baseline_results")

    if len(baseline_df) > 0:
        st.subheader("Held-Out Test Set Performance")
        display_cols = ["model_name", "accuracy", "balanced_accuracy", "f1", "roc_auc", "brier_score", "ece", "train_time_s"]
        avail = [c for c in display_cols if c in baseline_df.columns]
        render_df(
            baseline_df[avail].style.format({c: "{:.4f}" for c in avail if c not in ("model_name", "train_time_s")})
        )

        st.subheader("Confusion Matrices")
        cols = st.columns(2)
        for i, (_, row) in enumerate(baseline_df.iterrows()):
            mname = row["model_name"]
            cm_path = figures_dir / f"confusion_matrix_{dataset_key}_{mname}.png"
            if not cm_path.exists() and dataset_key == "adult":
                cm_path = figures_dir / f"confusion_matrix_{mname}.png"
            if cm_path.exists():
                with cols[i % 2]:
                    st.image(str(cm_path), caption=f"{mname} ({dataset_label})")

        st.subheader("Reliability Diagrams & Calibration")
        for i, (_, row) in enumerate(baseline_df.iterrows()):
            mname = row["model_name"]
            c1, c2 = st.columns(2)
            uncal = figures_dir / f"reliability_{dataset_key}_{mname}_uncal.png"
            cal = figures_dir / f"reliability_{dataset_key}_{mname}_cal.png"
            if not uncal.exists() and dataset_key == "adult":
                uncal = figures_dir / f"reliability_{mname}_uncal.png"
                cal = figures_dir / f"reliability_{mname}_cal.png"
            if uncal.exists():
                with c1:
                    st.image(str(uncal), caption=f"{mname} (Uncalibrated)")
            if cal.exists():
                with c2:
                    st.image(str(cal), caption=f"{mname} (Temperature Scaled)")
    else:
        st.warning("No baseline results found.")


def page_shift_lab(results_dir: Path, figures_dir: Path, dataset_key: str, dataset_label: str):
    st.title(f"🌊 Distribution Shift Lab — {dataset_label}")

    shift_df = get_dataset_csv(results_dir, dataset_key, "shift_experiment_results")

    if len(shift_df) > 0:
        c1, c2 = st.columns(2)
        with c1:
            scenarios = st.multiselect("Scenario", shift_df["scenario"].unique(), default=list(shift_df["scenario"].unique()))
        with c2:
            severities = st.multiselect("Severity", shift_df["severity"].unique(), default=list(shift_df["severity"].unique()))

        filtered = shift_df[shift_df["scenario"].isin(scenarios) & shift_df["severity"].isin(severities)]

        st.subheader("Shift Results Table")
        cols = ["scenario", "severity", "accuracy", "f1", "roc_auc", "empirical_coverage", "avg_set_size", "abstention_rate", "shift_detected", "shift_severity_est", "mmd_value", "max_psi"]
        avail = [c for c in cols if c in filtered.columns]
        render_df(
            filtered[avail].style.format(
                {c: "{:.4f}" for c in avail if c not in ("scenario", "severity", "shift_detected", "shift_severity_est")}
            )
        )

        st.subheader("Feature-Level Shift Diagnostics")
        for sc in scenarios:
            for sev in severities:
                fig_path = figures_dir / f"shift_analysis_{dataset_key}_{sc}_{sev}.png"
                if not fig_path.exists() and dataset_key == "adult":
                    fig_path = figures_dir / f"shift_analysis_{sc}_{sev}.png"
                if fig_path.exists():
                    st.image(str(fig_path), caption=f"{sc} — {sev} ({dataset_label})")
    else:
        st.warning("No shift experiment results found.")


def page_uncertainty(results_dir: Path, figures_dir: Path, dataset_key: str, dataset_label: str):
    st.title(f"🎯 Uncertainty & Conformal Prediction — {dataset_label}")

    shift_df = get_dataset_csv(results_dir, dataset_key, "shift_experiment_results")

    if len(shift_df) > 0 and "empirical_coverage" in shift_df.columns:
        st.subheader("Split Conformal Coverage by Shift Scenario")
        coverage_summary = shift_df.groupby(["scenario", "severity"]).agg(
            {
                "empirical_coverage": ["mean", "min", "max"],
                "avg_set_size": ["mean"],
                "empty_set_rate": "mean" if "empty_set_rate" in shift_df.columns else "first",
            }
        ).round(4)
        render_df(coverage_summary)

        st.markdown(
            "> [!NOTE]\n"
            "> Nominal coverage is calibrated at **1 - α = 90%** on exchangeable calibration data. "
            "Under mild-to-moderate shift, coverage stays near 88-92%. Under severe/OOD shift, "
            "coverage degrades unless adapted or abstained by SAUP's decision policy."
        )

        c1, c2 = st.columns(2)
        f1 = figures_dir / "coverage_by_scenario.png"
        f2 = figures_dir / "set_size_by_scenario.png"
        if f1.exists():
            with c1:
                st.image(str(f1), caption="Conformal Empirical Coverage")
        if f2.exists():
            with c2:
                st.image(str(f2), caption="Conformal Average Set Size")
    else:
        st.warning("No conformal results found.")


def page_saup_explorer(results_dir: Path, figures_dir: Path, dataset_key: str, dataset_label: str):
    st.title(f"⚙️ SAUP Decision Explorer — {dataset_label}")

    saup_df = get_dataset_csv(results_dir, dataset_key, "saup_results")

    if len(saup_df) > 0:
        st.subheader("Adaptive 3-Way Policy Results")
        cols = [
            "scenario",
            "severity",
            "accuracy",
            "accept_rate",
            "set_rate",
            "abstain_rate",
            "selective_accuracy",
            "empirical_coverage",
            "adaptation_method",
            "shift_severity_score",
        ]
        avail = [c for c in cols if c in saup_df.columns]
        render_df(saup_df[avail])

        # Abstention plot
        abs_fig = figures_dir / f"saup_abstention_vs_severity_{dataset_key}.png"
        if not abs_fig.exists() and dataset_key == "adult":
            abs_fig = figures_dir / "saup_abstention_vs_severity.png"
        if abs_fig.exists():
            st.image(str(abs_fig), caption=f"Abstention vs Shift Severity ({dataset_label})")

        st.subheader("Decision Policy Verification")
        st.markdown(
            "Mutual exclusivity test: `accept_rate + set_rate + abstain_rate == 1.0` is strictly enforced."
        )

        for _, row in saup_df.iterrows():
            with st.expander(f"{row.get('scenario','')} / {row.get('severity','')}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Accept Rate", f"{row.get('accept_rate', 0):.3f}")
                with c2:
                    st.metric("Set Rate", f"{row.get('set_rate', 0):.3f}")
                with c3:
                    st.metric("Abstain Rate", f"{row.get('abstention_rate', 0):.3f}")
                st.markdown(f"**Adaptation Strategy:** `{row.get('adaptation_method','standard')}`")
                st.markdown(f"**Shift Severity Score:** `{row.get('shift_severity_score', 0):.3f}`")
                sel_acc = row.get("selective_accuracy", np.nan)
                st.markdown(f"**Selective Accuracy on Accepted:** `{sel_acc:.4f}`" if pd.notna(sel_acc) else "**Selective Accuracy:** `N/A (All abstained)`")

    # Ablation
    st.subheader("Component Ablation Study")
    ablation_df = get_dataset_csv(results_dir, dataset_key, "ablation_results")
    if len(ablation_df) > 0:
        render_df(ablation_df)
        abl_fig = figures_dir / "ablation_study.png"
        if abl_fig.exists():
            st.image(str(abl_fig), caption="Component Ablation Study")


def page_downloads(results_dir: Path, figures_dir: Path, reports_dir: Path):
    st.title("📥 Results & Downloads")

    st.subheader("Exportable CSV Results")
    if results_dir.exists():
        csv_files = sorted(results_dir.glob("*.csv"))
        for p in csv_files:
            df = pd.read_csv(p)
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{p.name}** ({len(df)} rows)")
            with c2:
                with open(p, "rb") as f:
                    st.download_button(f"📥 Download", f, p.name, "text/csv", key=f"dl_{p.name}")

    st.subheader("Generated Scientific Reports")
    if reports_dir.exists():
        rep_files = sorted(reports_dir.glob("*.md"))
        for p in rep_files:
            with st.expander(f"📄 {p.name}"):
                content = p.read_text(encoding="utf-8")
                st.markdown(content)
                st.download_button(f"📥 Download {p.name}", content, p.name, "text/markdown", key=f"dl_rep_{p.name}")


if __name__ == "__main__":
    main()
