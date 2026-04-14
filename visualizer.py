from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def _set_plot_style() -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.linewidth"] = 1.2


def generate_all_plots(csv_path: str, output_dir: str = "artifacts") -> None:
    _set_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Figure 1: Agent Convergence Latency
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    for phase_name, label, color in [
        ("Phase 1 RIS Learning", "Phase 1 (RIS)", "#1b5e20"),
        ("Phase 2 NOMA Learning", "Phase 2 (NOMA)", "#0d47a1"),
        ("Phase 4 Joint Mastery", "Phase 4 (Joint)", "#b71c1c"),
    ]:
        phase_df = df[(df["phase_name"] == phase_name) & (df["scenario_type"] == "train")]
        if not phase_df.empty:
            mean_curve = phase_df.groupby("epoch", as_index=False)["agent_iterations"].mean()
            ax1.plot(mean_curve["epoch"], mean_curve["agent_iterations"], marker="o", linewidth=2.0, label=label, color=color)
    ax1.set_xlabel("Training Epochs")
    ax1.set_ylabel("Mean Agent Iterations")
    ax1.set_title("Figure 1: Agent Convergence Latency")
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "figure1_agent_convergence_latency.png"))
    plt.close(fig1)

    # Figure 2: Zero-Shot vs Mastery (TEST_JOINT)
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    baseline_joint = df[(df["phase_name"] == "Phase 0 Baseline Evaluation") & (df["scenario_type"] == "test_joint")]
    zeroshot_joint = df[(df["phase_name"] == "Phase 3 Zero-Shot Composition") & (df["scenario_type"] == "test_joint_zero_shot")]
    phase4_joint = df[(df["phase_name"] == "Phase 4 Joint Mastery") & (df["scenario_type"] == "test_joint")]

    vals = [
        float(baseline_joint["agent_iterations"].iloc[-1]) if not baseline_joint.empty else 0.0,
        float(zeroshot_joint["agent_iterations"].iloc[-1]) if not zeroshot_joint.empty else 0.0,
        float(phase4_joint.sort_values("epoch")["agent_iterations"].iloc[-1]) if not phase4_joint.empty else 0.0,
    ]
    labels = ["Phase 0 Baseline", "Phase 3 Zero-Shot", "Phase 4 Final Epoch"]
    ax2.bar(labels, vals, color=["#6a1b9a", "#00838f", "#ef6c00"], width=0.6)
    ax2.set_ylabel("Absolute Agent Iterations")
    ax2.set_xlabel("Evaluation Phase")
    ax2.set_title("Figure 2: Zero-Shot vs Mastery (TEST_JOINT)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "figure2_zero_shot_vs_mastery.png"))
    plt.close(fig2)

    # Figure 3: Joint EEPSU Progression
    fig3, ax3 = plt.subplots(figsize=(7, 4.5))
    joint_eval = df[(df["phase_name"] == "Phase 4 Joint Mastery") & (df["scenario_type"] == "test_joint")].sort_values("epoch")
    if not joint_eval.empty:
        ax3.plot(joint_eval["epoch"], joint_eval["eepsu"], marker="o", linewidth=2.0, color="#ad1457")
    ax3.set_xlabel("Phase 4 Training Epochs")
    ax3.set_ylabel("EEPSU")
    ax3.set_title("Figure 3: Joint EEPSU Progression")
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "figure3_joint_utility_progression.png"))
    plt.close(fig3)

    # Figure 4: Energy Efficiency Distribution
    fig4, ax4 = plt.subplots(figsize=(7, 4.5))
    ax4.scatter(df["ris_u3_snr"], df["noma_u1_power_ratio"], alpha=0.7, color="#2e7d32", edgecolors="black", linewidths=0.4)
    ax4.set_xlabel("RIS SNR")
    ax4.set_ylabel("NOMA Power Ratio")
    ax4.set_title("Figure 4: Energy Efficiency Distribution")
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, "figure4_energy_efficiency_distribution.png"))
    plt.close(fig4)

    # Figure 5: Independent Domain Mastery
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 4.5))

    ris_baseline = df[(df["phase_name"] == "Phase 0 Baseline Evaluation") & (df["scenario_type"] == "test_ris")]
    ris_progress = df[(df["phase_name"] == "Phase 1 RIS Learning") & (df["scenario_type"] == "test_ris")].sort_values("epoch")
    ris_x = [0] + ris_progress["epoch"].tolist()
    ris_y = [float(ris_baseline["eepsu"].iloc[-1]) if not ris_baseline.empty else 0.0] + ris_progress["eepsu"].tolist()
    ax5a.plot(ris_x, ris_y, marker="o", color="#2e7d32", linewidth=2.0)
    ax5a.set_title("RIS Domain Mastery")
    ax5a.set_xlabel("Phase 1 Epochs")
    ax5a.set_ylabel("EEPSU")
    ax5a.grid(alpha=0.3)

    noma_baseline = df[(df["phase_name"] == "Phase 0 Baseline Evaluation") & (df["scenario_type"] == "test_noma")]
    noma_progress = df[(df["phase_name"] == "Phase 2 NOMA Learning") & (df["scenario_type"] == "test_noma")].sort_values("epoch")
    noma_x = [0] + noma_progress["epoch"].tolist()
    noma_y = [float(noma_baseline["pws"].iloc[-1]) if not noma_baseline.empty else 0.0] + noma_progress["pws"].tolist()
    ax5b.plot(noma_x, noma_y, marker="o", color="#1565c0", linewidth=2.0)
    ax5b.set_title("NOMA Domain Mastery")
    ax5b.set_xlabel("Phase 2 Epochs")
    ax5b.set_ylabel("PWS")
    ax5b.grid(alpha=0.3)

    fig5.suptitle("Figure 5: Independent Domain Mastery", fontweight="bold")
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, "figure5_independent_domain_mastery.png"))
    plt.close(fig5)
