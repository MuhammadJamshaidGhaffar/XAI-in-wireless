"""Publication-grade visualization utilities for continual 6G experiments."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
    }
)
sns.set_palette("colorblind")

ACTIVE_PLOTS = [
    "global_convergence",
    "independent_phase",
    "sum_rate",
    "ee_progression",
    "interference_scatter",
]


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save figure with IEEE-friendly defaults.

    Args:
        fig: Matplotlib figure.
        output_path: Destination path.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO - Visualizer] Saved figure: {output_path}")


def plot_global_convergence(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot scenario-level convergence with phase-zone shading.

    Args:
        df: DataFrame containing scenario_index, phase_name, iterations_to_converge.
        ax: Optional axis.

    Returns:
        Axis with rendered chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    x = df["scenario_index"].to_numpy()
    y = df["iterations_to_converge"].to_numpy()

    zones = [
        ("Phase 0", "#f2f2f2"),
        ("Phase 1/2", "#eaf5ff"),
        ("Phase 3", "#fff4e6"),
        ("Phase 4", "#e9f7ef"),
    ]

    phase_bins = df.groupby("phase_name")["scenario_index"].agg(["min", "max"]).reset_index()
    for i, (_, row) in enumerate(phase_bins.iterrows()):
        label = zones[min(i, len(zones) - 1)][0]
        color = zones[min(i, len(zones) - 1)][1]
        ax.axvspan(row["min"] - 0.5, row["max"] + 0.5, color=color, alpha=0.6, label=label if i < 4 else None)

    ax.plot(x, y, marker="o", linewidth=2, label="Convergence Iterations")
    ax.set_xlabel("Scenario Index")
    ax.set_ylabel("Iterations to Converge")
    ax.set_ylim(0.8, 10.5)
    ax.set_title("Global Convergence Behavior Across Continual Phases")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    return ax


def plot_independent_phase(df: pd.DataFrame, phase_name: str, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a single phase learning curve.

    Args:
        df: Full metrics DataFrame.
        phase_name: Target phase name.
        ax: Optional axis.

    Returns:
        Axis with rendered chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    subset = df[df["phase_name"] == phase_name].copy()
    if subset.empty:
        ax.text(0.5, 0.5, f"No data for {phase_name}", ha="center", va="center")
        return ax

    ax.plot(subset["scenario_index"], subset["iterations_to_converge"], marker="s", linewidth=2)
    ax.set_title(f"Independent Phase Behavior: {phase_name}")
    ax.set_xlabel("Scenario Index")
    ax.set_ylabel("Iterations to Converge")
    ax.grid(alpha=0.3)
    return ax


def plot_sum_rate_trajectory(df_random: pd.DataFrame, df_graphrag: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot sum-rate by optimization iteration for two initializations.

    Args:
        df_random: DataFrame with columns iteration, sum_rate.
        df_graphrag: DataFrame with columns iteration, sum_rate.
        ax: Optional axis.

    Returns:
        Axis with rendered chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df_random["iteration"], df_random["sum_rate"], marker="o", label="Random Initialization")
    ax.plot(df_graphrag["iteration"], df_graphrag["sum_rate"], marker="^", label="Graph-RAG Initialization")
    ax.set_xlabel("Optimization Iteration")
    ax.set_ylabel("Sum-Rate (bps/Hz)")
    ax.set_title("Sum-Rate Trajectory")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_ee_progression(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot EEPSU progression over scenario index.

    Args:
        df: Metrics DataFrame containing eepsu.
        ax: Optional axis.

    Returns:
        Axis with rendered chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df["scenario_index"], df["eepsu"], color="#2a9d8f", marker="d", linewidth=2)
    ax.set_xlabel("Scenario Index")
    ax.set_ylabel("EEPSU")
    ax.set_title("Energy Efficiency Per Satisfied User Progression")
    ax.grid(alpha=0.3)
    return ax


def plot_interference_scatter(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot NOMA power allocation versus RIS user SNR.

    Args:
        df: DataFrame with noma_u1_power_ratio and ris_u3_snr.
        ax: Optional axis.

    Returns:
        Axis with rendered chart.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sns.regplot(
        data=df,
        x="noma_u1_power_ratio",
        y="ris_u3_snr",
        scatter_kws={"alpha": 0.6, "s": 35},
        line_kws={"color": "black", "linewidth": 1.5},
        ax=ax,
    )
    ax.set_xlabel("NOMA U1 Power Allocation Ratio")
    ax.set_ylabel("RIS U3 Received SNR")
    ax.set_title("Cross-Interference Validation")
    ax.grid(alpha=0.25)
    return ax


def generate_semantic_trace_log(rationale_dict: Dict[str, Any]) -> str:
    """Render a terminal-style semantic trace for XAI figure capture.

    Args:
        rationale_dict: Structured rationale payload.

    Returns:
        Multi-line formatted trace string.
    """
    scenario = rationale_dict.get("scenario_desc", "N/A")
    priors = rationale_dict.get("retrieved_priors", "N/A")
    final_params = rationale_dict.get("final_solver_params", {})
    rationale = rationale_dict.get("llm_rationale", "N/A")

    trace = (
        "+-----------------------------------------------------------+\n"
        "|  GRAPH-RAG SEMANTIC TRACE :: JOINT ORCHESTRATION TERMINAL |\n"
        "+-----------------------------------------------------------+\n"
        f"[SCENARIO] {scenario}\n"
        f"[PRIORS ] {priors}\n"
        f"[PARAMS ] {final_params}\n"
        f"[XAI    ] {rationale}\n"
        "+-----------------------------------------------------------+"
    )
    return trace


def generate_master_dashboard(data_dict: Dict[str, pd.DataFrame], output_dir: str = "figures") -> None:
    """Generate active plots in both dashboard and standalone form.

    Args:
        data_dict: Named dataframe collection from main pipeline.
        output_dir: Destination folder for image files.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    flat_axes = axes.flatten()
    idx = 0

    if "global_convergence" in ACTIVE_PLOTS and "metrics" in data_dict:
        plot_global_convergence(data_dict["metrics"], flat_axes[idx])
        idx += 1
    if "independent_phase" in ACTIVE_PLOTS and "metrics" in data_dict:
        phase = data_dict.get("independent_phase_name", "Phase 3 (Joint Overlap)")
        plot_independent_phase(data_dict["metrics"], phase, flat_axes[idx])
        idx += 1
    if "sum_rate" in ACTIVE_PLOTS and "sum_rate_random" in data_dict and "sum_rate_graphrag" in data_dict:
        plot_sum_rate_trajectory(data_dict["sum_rate_random"], data_dict["sum_rate_graphrag"], flat_axes[idx])
        idx += 1
    if "ee_progression" in ACTIVE_PLOTS and "metrics" in data_dict:
        plot_ee_progression(data_dict["metrics"], flat_axes[idx])
        idx += 1
    if "interference_scatter" in ACTIVE_PLOTS and "interference" in data_dict:
        plot_interference_scatter(data_dict["interference"], flat_axes[idx])
        idx += 1

    for j in range(idx, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.suptitle("Graph-RAG Continual Learning Dashboard", fontweight="bold")
    _save_figure(fig, os.path.join(output_dir, "master_dashboard.png"))
    plt.close(fig)

    # Standalone outputs.
    if "metrics" in data_dict:
        fig1, ax1 = plt.subplots(figsize=(9, 4.5))
        plot_global_convergence(data_dict["metrics"], ax1)
        _save_figure(fig1, os.path.join(output_dir, "global_convergence.png"))
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        phase = data_dict.get("independent_phase_name", "Phase 3 (Joint Overlap)")
        plot_independent_phase(data_dict["metrics"], phase, ax2)
        _save_figure(fig2, os.path.join(output_dir, "independent_phase.png"))
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        plot_ee_progression(data_dict["metrics"], ax3)
        _save_figure(fig3, os.path.join(output_dir, "ee_progression.png"))
        plt.close(fig3)

    if "sum_rate_random" in data_dict and "sum_rate_graphrag" in data_dict:
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        plot_sum_rate_trajectory(data_dict["sum_rate_random"], data_dict["sum_rate_graphrag"], ax4)
        _save_figure(fig4, os.path.join(output_dir, "sum_rate_trajectory.png"))
        plt.close(fig4)

    if "interference" in data_dict:
        fig5, ax5 = plt.subplots(figsize=(7, 4))
        plot_interference_scatter(data_dict["interference"], ax5)
        _save_figure(fig5, os.path.join(output_dir, "interference_scatter.png"))
        plt.close(fig5)
