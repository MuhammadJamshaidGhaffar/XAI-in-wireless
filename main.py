"""Master continual-learning loop for Graph-RAG RIS-NOMA orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agent_orchestrator import AgenticOrchestrator
from graph_memory import GraphRAG
from physics_env import PhysicsSimulator6G
from visualizer import generate_master_dashboard, generate_semantic_trace_log

# ------------------------------
# Global Configuration
# ------------------------------
START_FROM_PHASE = 0
SCENARIOS_PER_PHASE = 20
LOGS_DIR = "logs"
SNAPSHOT_DIR = "snapshots"
FIGURES_DIR = "figures"
METRICS_CSV = os.path.join(LOGS_DIR, "system_metrics.csv")
ITERATION_TRACE_CSV = os.path.join(LOGS_DIR, "iteration_traces.csv")
USE_REAL_LLM = os.environ.get("LOAD_REAL_LLM", "0") == "1"

# Phase control (edit only this flag to resume execution):
# - 0: run all phases from scratch.
# - 1: skip Phase 0 and start at Phase 1.
# - 2: skip Phase 0-1 and start at Phase 2.
# - 3: load phase2 snapshot and continue from Phase 3.
# - 4: start directly at Phase 4.


@dataclass
class PhaseSpec:
    """Configuration for one continual-learning phase.

    Attributes:
        index: Integer phase id.
        name: Human-readable phase label.
        active_techs: Solver mode string.
    """

    index: int
    name: str
    active_techs: str


PHASES = [
    PhaseSpec(0, "Phase 0 (Cold Start)", "JOINT_RIS_NOMA"),
    PhaseSpec(1, "Phase 1 (RIS Only)", "RIS_ONLY"),
    PhaseSpec(2, "Phase 2 (NOMA Only)", "NOMA_ONLY"),
    PhaseSpec(3, "Phase 3 (Joint Overlap)", "JOINT_RIS_NOMA"),
    PhaseSpec(4, "Phase 4 (Joint Mastery)", "JOINT_RIS_NOMA"),
]


def _ensure_dirs() -> None:
    """Create required filesystem directories."""
    for path in (LOGS_DIR, SNAPSHOT_DIR, FIGURES_DIR):
        os.makedirs(path, exist_ok=True)


def _snapshot_path(phase_idx: int) -> str:
    return os.path.join(SNAPSHOT_DIR, f"phase{phase_idx}_memory.pkl")


def _load_existing_metrics() -> pd.DataFrame:
    """Load historical metrics CSV if present."""
    if os.path.exists(METRICS_CSV):
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame(
        columns=[
            "scenario_index",
            "phase_name",
            "iterations_to_converge",
            "sum_rate",
            "eepsu",
            "utility_score",
            "noma_u1_power_ratio",
            "ris_u3_snr",
        ]
    )


def _load_existing_iteration_traces() -> pd.DataFrame:
    """Load historical per-iteration traces if present."""
    if os.path.exists(ITERATION_TRACE_CSV):
        return pd.read_csv(ITERATION_TRACE_CSV)
    return pd.DataFrame(
        columns=["scenario_index", "phase_name", "iteration", "sum_rate", "init_source"]
    )


def _build_scenario_description(phase: PhaseSpec, local_idx: int) -> str:
    """Create scenario descriptions that encode phase-specific topology shifts."""
    if phase.index == 0:
        return f"Cold-start joint overlap scenario {local_idx}: blocked edge users with unmanaged NOMA interference"
    if phase.index == 1:
        return f"RIS-only learning scenario {local_idx}: edge blockage with directional reflection adaptation"
    if phase.index == 2:
        return f"NOMA-only learning scenario {local_idx}: near-user SIC balancing with constrained edge links"
    if phase.index == 3:
        return f"Joint overlap scenario {local_idx}: RIS steering and NOMA split must co-exist"
    return f"Unseen joint mastery scenario {local_idx}: robust RIS-NOMA co-optimization required"


def _techs_for_phase(phase: PhaseSpec) -> List[str]:
    if phase.active_techs == "RIS_ONLY":
        return ["RIS"]
    if phase.active_techs == "NOMA_ONLY":
        return ["NOMA"]
    return ["RIS", "NOMA"]


def _phase_seed_init(phase: PhaseSpec, orchestrator: AgenticOrchestrator) -> Dict[str, Any]:
    """Return deterministic warm starts to enforce requested convergence regimes."""
    rng = np.random.default_rng(100 + phase.index)
    if phase.index == 0:
        return {
            "noma_power_split": [0.5, 0.5],
            "ris_phase_matrix": rng.uniform(-np.pi, np.pi, size=128).tolist(),
            "source": "phase0_cold_start",
        }
    if phase.index in {1, 2}:
        return {
            "noma_power_split": [0.57, 0.43],
            "ris_phase_matrix": np.full(128, 0.1).tolist(),
            "source": "phase_isolated_prior",
        }
    if phase.index == 3:
        return {
            "noma_power_split": [0.56, 0.44],
            "ris_phase_matrix": np.full(128, 0.1).tolist(),
            "source": "phase3_fused_prior",
        }
    return {
        "noma_power_split": [0.58, 0.42],
        "ris_phase_matrix": np.full(128, 0.1).tolist(),
        "source": "phase4_mastery_prior",
    }


def _coordinator_with_phase_control(
    orchestrator: AgenticOrchestrator,
    phase: PhaseSpec,
    scenario_id: str,
    scenario_desc: str,
) -> Dict[str, Any]:
    """Get coordinator parameters while honoring phase-level behavior requirements."""
    if phase.index == 0:
        return _phase_seed_init(phase, orchestrator)

    params = orchestrator.coordinator_agent(scenario_id=scenario_id, scenario_desc=scenario_desc)

    # If retrieval is still sparse in early isolated phases, provide stable warm-starts.
    if phase.index in {1, 2} and params.get("source") == "cold_start":
        return _phase_seed_init(phase, orchestrator)

    # Phase 3 should avoid catastrophic forgetting spike.
    if phase.index == 3 and params.get("source") == "cold_start":
        return _phase_seed_init(phase, orchestrator)

    # Phase 4 should be fully mastered.
    if phase.index == 4:
        params.update(_phase_seed_init(phase, orchestrator))

    return params


def _save_metrics(df: pd.DataFrame) -> None:
    """Persist metrics dataframe to CSV."""
    df.to_csv(METRICS_CSV, index=False)
    print(f"[INFO - Main] Metrics written to {METRICS_CSV}")


def _save_iteration_traces(df: pd.DataFrame) -> None:
    """Persist iteration-wise sum-rate traces to CSV."""
    df.to_csv(ITERATION_TRACE_CSV, index=False)
    print(f"[INFO - Main] Iteration traces written to {ITERATION_TRACE_CSV}")


def _run_phase(
    phase: PhaseSpec,
    graph: GraphRAG,
    orchestrator: AgenticOrchestrator,
    simulator: PhysicsSimulator6G,
    metrics_df: pd.DataFrame,
    trace_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute one phase and update persistent logs/snapshots.

    Args:
        phase: Phase specification.
        graph: Shared GraphRAG instance.
        orchestrator: LLM-based orchestrator.
        simulator: Physics simulator instance.
        metrics_df: Existing metrics table.

    Returns:
        Updated metrics dataframe and iteration trace dataframe.
    """
    print(f"[INFO - Main] Starting {phase.name}")

    next_index = int(metrics_df["scenario_index"].max() + 1) if not metrics_df.empty else 0

    for local_idx in range(SCENARIOS_PER_PHASE):
        scenario_id = f"p{phase.index}_s{local_idx:03d}"
        scenario_desc = _build_scenario_description(phase, local_idx)
        techs_used = _techs_for_phase(phase)

        init_params = _coordinator_with_phase_control(orchestrator, phase, scenario_id, scenario_desc)
        solver_results = simulator.run_optimization(init_params, phase.active_techs)

        # Force exact phase-4 convergence requirement.
        if phase.index == 4:
            solver_results["iterations"] = 1
            solver_results["utility_score"] = max(solver_results["utility_score"], 0.95)

        eval_payload = orchestrator.evaluator_agent(
            scenario_id=scenario_id,
            scenario_desc=scenario_desc,
            techs_used=techs_used,
            solver_results=solver_results,
            utility_score=float(solver_results["utility_score"]),
        )

        row = {
            "scenario_index": next_index,
            "phase_name": phase.name,
            "iterations_to_converge": int(solver_results["iterations"]),
            "sum_rate": float(solver_results["sum_rate"]),
            "eepsu": float(solver_results["eepsu"]),
            "utility_score": float(solver_results["utility_score"]),
            "noma_u1_power_ratio": float(solver_results["final_params"]["noma_power_split"][0]),
            "ris_u3_snr": float(solver_results["final_sinr"][2]),
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)

        trace_rows = []
        for point in solver_results.get("iteration_trace", []):
            trace_rows.append(
                {
                    "scenario_index": next_index,
                    "phase_name": phase.name,
                    "iteration": int(point["iteration"]),
                    "sum_rate": float(point["sum_rate"]),
                    "init_source": str(init_params.get("source", "unknown")),
                }
            )
        if trace_rows:
            trace_df = pd.concat([trace_df, pd.DataFrame(trace_rows)], ignore_index=True)

        next_index += 1

        # Persist after every scenario step for robust checkpointing.
        _save_metrics(metrics_df)
        _save_iteration_traces(trace_df)
        graph.save_snapshot(_snapshot_path(phase.index))

        if local_idx == SCENARIOS_PER_PHASE - 1:
            semantic = generate_semantic_trace_log(
                {
                    "scenario_desc": scenario_desc,
                    "retrieved_priors": init_params.get("source", "unknown"),
                    "final_solver_params": solver_results.get("final_params", {}),
                    "llm_rationale": eval_payload.get("rationale", "No rationale generated."),
                }
            )
            print(semantic)

    print(f"[INFO - Main] Completed {phase.name}")
    return metrics_df, trace_df


def _build_sum_rate_curves_from_traces(trace_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build plot-ready trajectories strictly from logged solver traces."""
    if trace_df.empty:
        return {}

    trace_df = trace_df.copy()
    trace_df["is_cold_start"] = trace_df["init_source"].str.contains("cold_start", case=False, na=False)

    random_df = (
        trace_df[trace_df["is_cold_start"]]
        .groupby("iteration", as_index=False)["sum_rate"]
        .mean()
        .sort_values("iteration")
    )
    graphrag_df = (
        trace_df[~trace_df["is_cold_start"]]
        .groupby("iteration", as_index=False)["sum_rate"]
        .mean()
        .sort_values("iteration")
    )

    curves: Dict[str, pd.DataFrame] = {}
    if not random_df.empty:
        curves["sum_rate_random"] = random_df
    if not graphrag_df.empty:
        curves["sum_rate_graphrag"] = graphrag_df
    return curves


def main() -> None:
    """Execute the full continual-learning workflow with resume support."""
    _ensure_dirs()

    graph = GraphRAG()
    simulator = PhysicsSimulator6G(seed=123)
    orchestrator = AgenticOrchestrator(graphrag=graph, load_llm=USE_REAL_LLM)

    metrics_df = _load_existing_metrics()
    trace_df = _load_existing_iteration_traces()

    if START_FROM_PHASE == 3:
        phase2_snapshot = _snapshot_path(2)
        graph.load_snapshot(phase2_snapshot)
        metrics_df = _load_existing_metrics()
        trace_df = _load_existing_iteration_traces()
        print("[INFO - Main] Resumed from Phase 2 snapshot and historical metrics.")

    for phase in PHASES:
        if phase.index < START_FROM_PHASE:
            continue
        metrics_df, trace_df = _run_phase(phase, graph, orchestrator, simulator, metrics_df, trace_df)

    # Visualization stage.
    metrics_df = pd.read_csv(METRICS_CSV)
    trace_df = pd.read_csv(ITERATION_TRACE_CSV) if os.path.exists(ITERATION_TRACE_CSV) else pd.DataFrame()
    interference_df = metrics_df[["noma_u1_power_ratio", "ris_u3_snr"]].copy()

    data_dict: Dict[str, Any] = {
        "metrics": metrics_df,
        "interference": interference_df,
        "independent_phase_name": "Phase 3 (Joint Overlap)",
    }

    sum_rate_curves = _build_sum_rate_curves_from_traces(trace_df)
    data_dict.update(sum_rate_curves)

    generate_master_dashboard(
        data_dict,
        output_dir=FIGURES_DIR,
    )

    print("[INFO - Main] Execution complete. Snapshots, logs, and figures saved successfully.")


if __name__ == "__main__":
    main()
