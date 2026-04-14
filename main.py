from __future__ import annotations

import copy
import csv
import os
from typing import Any, Dict, List, Tuple

from agent_orchestrator import AgentOrchestrator
from memory_manager import ConceptDatabase
from physics_env import PhysicsEnvironment
from visualizer import generate_all_plots


NUM_EPOCHS = 5
LOG_CSV_PATH = "artifacts/training_log.csv"


def _safe_import_llm() -> Any:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        use_4bit = bool(torch.cuda.is_available())
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            load_in_8bit=not use_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        llm_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return llm_pipe
    except Exception as exc:
        raise RuntimeError(f"Failed to load DeepSeek pipeline: {exc}") from exc


def _scenario_template(
    category: str,
    cluster_name: str,
    idx: int,
    description: str,
    d_far_m: float,
    d_near_m: float,
    d_bs_ris_m: float,
    d_ris_u3_m: float,
    blockage_factor: float,
    pathloss_exp: float,
    ris_elements: int,
    qos_target: float,
    snr_target: float,
    eepsu_target: float,
    pws_target: float,
    fairness_min: float,
) -> Dict:
    return {
        "scenario_id": f"{category}_{cluster_name}_{idx}",
        "category": category,
        "cluster": cluster_name,
        "description": description,
        "bs_power_dbm": 40.0,
        "base_power_dbm_min": 34.0,
        "base_power_dbm_max": 46.0,
        "noise_dbm": -94.0,
        "bandwidth_hz": 20e6,
        "d_far_m": d_far_m,
        "d_near_m": d_near_m,
        "d_bs_ris_m": d_bs_ris_m,
        "d_ris_u3_m": d_ris_u3_m,
        "blockage_factor": blockage_factor,
        "pathloss_exp": pathloss_exp,
        "ris_elements": ris_elements,
        "qos_target": qos_target,
        "snr_target": snr_target,
        "eepsu_target": eepsu_target,
        "pws_target": pws_target,
        "fairness_min": fairness_min,
        "max_power_budget": 12.0,
    }


def build_hierarchical_scenarios() -> Tuple[Dict[str, Dict[str, List[Dict]]], Dict[str, Dict]]:
    train: Dict[str, Dict[str, List[Dict]]] = {
        "RIS_ONLY": {
            "Cluster_Center_Blockage": [
                _scenario_template("RIS_ONLY", "Cluster_Center_Blockage", i, "RIS center blockage variations", 220, 90, 55, 95 + i, 0.78, 2.4, 160, 8.0, 25.0, 1.60, 3.00, 0.80)
                for i in range(1, 5)
            ],
            "Cluster_High_Correlation": [
                _scenario_template("RIS_ONLY", "Cluster_High_Correlation", i, "RIS high-correlation angular channel", 230, 95, 60, 100 + i, 0.88, 2.2, 192, 8.5, 26.0, 1.70, 3.30, 0.82)
                for i in range(1, 4)
            ],
            "Cluster_Low_Elevation": [
                _scenario_template("RIS_ONLY", "Cluster_Low_Elevation", i, "RIS low-elevation reflected path", 210, 85, 48, 90 + i, 0.82, 2.3, 144, 7.8, 24.0, 1.50, 2.80, 0.80)
                for i in range(1, 4)
            ],
        },
        "NOMA_ONLY": {
            "Cluster_Power_Imbalance": [
                _scenario_template("NOMA_ONLY", "Cluster_Power_Imbalance", i, "NOMA power imbalance with varied user offsets", 240 + i * 2, 80 + i, 50, 90, 0.90, 2.2, 64, 9.0, 20.0, 1.25, 13.50, 0.85)
                for i in range(1, 5)
            ],
            "Cluster_SIC_Sensitivity": [
                _scenario_template("NOMA_ONLY", "Cluster_SIC_Sensitivity", i, "NOMA SIC sensitivity under residual interference", 235 + i, 88 + i, 52, 92, 0.86, 2.3, 64, 9.2, 21.0, 1.20, 13.00, 0.84)
                for i in range(1, 4)
            ],
            "Cluster_Cell_Edge": [
                _scenario_template("NOMA_ONLY", "Cluster_Cell_Edge", i, "NOMA far-user cell-edge stress", 260 + i * 3, 92 + i, 56, 98, 0.76, 2.5, 64, 9.6, 19.0, 1.10, 12.60, 0.83)
                for i in range(1, 4)
            ],
        },
        "JOINT": {
            "Cluster_Interference_Canyon": [
                _scenario_template("JOINT", "Cluster_Interference_Canyon", i, "Joint RIS-NOMA urban canyon interference", 245 + i, 92 + i, 57, 102 + i, 0.80, 2.4, 192, 10.0, 24.0, 1.35, 15.50, 0.86)
                for i in range(1, 5)
            ],
            "Cluster_MultiPath_Shear": [
                _scenario_template("JOINT", "Cluster_MultiPath_Shear", i, "Joint sheared multipath with dynamic blockage", 238 + i, 86 + i, 54, 96 + i, 0.84, 2.3, 224, 10.5, 25.0, 1.40, 16.00, 0.87)
                for i in range(1, 4)
            ],
            "Cluster_Dense_Hotspot": [
                _scenario_template("JOINT", "Cluster_Dense_Hotspot", i, "Joint dense hotspot with reflected congestion", 250 + i * 2, 95 + i, 58, 106 + i, 0.79, 2.5, 256, 11.0, 23.5, 1.30, 15.20, 0.86)
                for i in range(1, 4)
            ],
        },
    }

    tests = {
        "TEST_RIS": _scenario_template(
            "RIS_ONLY",
            "Extreme_Diagonal_Blockage",
            1,
            "Unseen RIS test cluster: extreme diagonal blockage",
            265,
            100,
            68,
            122,
            0.62,
            2.7,
            256,
            8.8,
            27.0,
            1.75,
            3.60,
            0.82,
        ),
        "TEST_NOMA": _scenario_template(
            "NOMA_ONLY",
            "Asymmetric_Near_Far_Shadowing",
            1,
            "Unseen NOMA test cluster: asymmetric near-far shadowing",
            280,
            72,
            45,
            90,
            0.70,
            2.6,
            64,
            10.2,
            18.5,
            1.30,
            13.80,
            0.85,
        ),
        "TEST_JOINT": _scenario_template(
            "JOINT",
            "Cross_Polarized_MegaBlockage",
            1,
            "Unseen JOINT test cluster: cross-polarized mega blockage",
            275,
            94,
            64,
            130,
            0.66,
            2.8,
            320,
            11.5,
            28.0,
            1.45,
            16.40,
            0.88,
        ),
    }
    return train, tests


def _flatten_cluster_scenarios(category_clusters: Dict[str, List[Dict]]) -> List[Dict]:
    out: List[Dict] = []
    for _, scenarios in category_clusters.items():
        out.extend(scenarios)
    return out


def _log_header(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "scenario_index",
                "phase_name",
                "agent_iterations",
                "sum_rate",
                "noma_u1_power_ratio",
                "ris_u3_snr",
                "scenario_type",
                "eepsu",
                "pws",
                "jain_fairness",
                "domain_utility_score",
            ]
        )


def _append_log(csv_path: str, row: Dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["epoch"],
                row["scenario_index"],
                row["phase_name"],
                row["agent_iterations"],
                row["sum_rate"],
                row["noma_u1_power_ratio"],
                row["ris_u3_snr"],
                row["scenario_type"],
                row["eepsu"],
                row["pws"],
                row["jain_fairness"],
                row["domain_utility_score"],
            ]
        )


def _log_event(
    csv_path: str,
    epoch: int,
    scenario_index: int,
    phase_name: str,
    scenario_type: str,
    run_output: Dict,
) -> None:
    result = run_output["result"]
    params = run_output["params"]
    _append_log(
        csv_path,
        {
            "epoch": epoch,
            "scenario_index": scenario_index,
            "phase_name": phase_name,
            "agent_iterations": run_output["agent_iterations"],
            "sum_rate": result["sum_rate"],
            "noma_u1_power_ratio": params["noma_power_split"],
            "ris_u3_snr": result["ris_u3_snr"],
            "scenario_type": scenario_type,
            "eepsu": result["eepsu"],
            "pws": result["pws"],
            "jain_fairness": result["jain_fairness"],
            "domain_utility_score": result["domain_utility_score"],
        },
    )


def main() -> None:
    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    train_sets, tests = build_hierarchical_scenarios()
    llm_pipe = _safe_import_llm()

    physics = PhysicsEnvironment()
    orchestrator = AgentOrchestrator(llm_pipeline=llm_pipe, physics_env=physics)

    ris_db = ConceptDatabase(db_path="snapshots/ris_db.pkl")
    noma_db = ConceptDatabase(db_path="snapshots/noma_db.pkl")
    joint_db = ConceptDatabase(db_path="snapshots/joint_db.pkl")

    _log_header(LOG_CSV_PATH)

    # Phase 0: Baseline Evaluation (Epoch 0)
    phase = "Phase 0 Baseline Evaluation"
    p0_ris = orchestrator.run_agentic_evaluation(tests["TEST_RIS"], ["ris"], ris_db)
    _log_event(LOG_CSV_PATH, 0, 1, phase, "test_ris", p0_ris)

    p0_noma = orchestrator.run_agentic_evaluation(tests["TEST_NOMA"], ["noma"], noma_db)
    _log_event(LOG_CSV_PATH, 0, 2, phase, "test_noma", p0_noma)

    p0_joint = orchestrator.run_agentic_evaluation(tests["TEST_JOINT"], ["joint"], joint_db)
    _log_event(LOG_CSV_PATH, 0, 3, phase, "test_joint", p0_joint)

    # Phase 1: RIS Learning
    phase = "Phase 1 RIS Learning"
    ris_train = _flatten_cluster_scenarios(train_sets["RIS_ONLY"])
    for epoch in range(1, NUM_EPOCHS + 1):
        for idx, scenario in enumerate(ris_train, start=1):
            train_out = orchestrator.run_agentic_optimization(scenario, ["ris"], ris_db)
            _log_event(LOG_CSV_PATH, epoch, idx, phase, "train", train_out)

        eval_out = orchestrator.run_agentic_evaluation(tests["TEST_RIS"], ["ris"], ris_db)
        _log_event(LOG_CSV_PATH, epoch, 999, phase, "test_ris", eval_out)

    # Phase 2: NOMA Learning
    phase = "Phase 2 NOMA Learning"
    noma_train = _flatten_cluster_scenarios(train_sets["NOMA_ONLY"])
    for epoch in range(1, NUM_EPOCHS + 1):
        for idx, scenario in enumerate(noma_train, start=1):
            train_out = orchestrator.run_agentic_optimization(scenario, ["noma"], noma_db)
            _log_event(LOG_CSV_PATH, epoch, idx, phase, "train", train_out)

        eval_out = orchestrator.run_agentic_evaluation(tests["TEST_NOMA"], ["noma"], noma_db)
        _log_event(LOG_CSV_PATH, epoch, 999, phase, "test_noma", eval_out)

    # Phase 3: Zero-Shot Composition
    phase = "Phase 3 Zero-Shot Composition"
    joint_db.memory = [dict(x) for x in copy.deepcopy(ris_db.memory)]
    joint_db._save_to_disk()
    joint_db.merge_with(noma_db)

    p3_joint = orchestrator.run_agentic_evaluation(tests["TEST_JOINT"], ["joint", "ris", "noma"], joint_db)
    _log_event(LOG_CSV_PATH, 0, 1, phase, "test_joint_zero_shot", p3_joint)

    # Phase 4: Joint Mastery
    phase = "Phase 4 Joint Mastery"
    joint_train = _flatten_cluster_scenarios(train_sets["JOINT"])
    for epoch in range(1, NUM_EPOCHS + 1):
        for idx, scenario in enumerate(joint_train, start=1):
            train_out = orchestrator.run_agentic_optimization(scenario, ["joint", "ris", "noma"], joint_db)
            _log_event(LOG_CSV_PATH, epoch, idx, phase, "train", train_out)

        eval_out = orchestrator.run_agentic_evaluation(tests["TEST_JOINT"], ["joint", "ris", "noma"], joint_db)
        _log_event(LOG_CSV_PATH, epoch, 999, phase, "test_joint", eval_out)

    # Final artifact exports
    ris_db.export_to_markdown("artifacts/ris_rulebook.md", llm_pipeline=llm_pipe)
    noma_db.export_to_markdown("artifacts/noma_rulebook.md", llm_pipeline=llm_pipe)
    joint_db.export_to_markdown("artifacts/joint_rulebook.md", llm_pipeline=llm_pipe)

    # Publication plots
    generate_all_plots(LOG_CSV_PATH, output_dir="artifacts")

    print("Pipeline complete. Artifacts and logs written to ./artifacts")


if __name__ == "__main__":
    main()
