"""LLM-based coordinator/evaluator agents for 6G continual orchestration."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

import numpy as np
import torch

from graph_memory import GraphRAG


class AgenticOrchestrator:
    """Coordinator and evaluator interface around an instruction-tuned LLM."""

    def __init__(
        self,
        graphrag: GraphRAG,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_llm: bool = True,
    ) -> None:
        """Initialize orchestrator and optionally load the LLM.

        Args:
            graphrag: Shared GraphRAG memory object.
            model_name: Hugging Face model identifier.
            load_llm: Whether to load the actual model or run in mock mode.
        """
        self.graphrag = graphrag
        self.model_name = model_name
        self.pipeline = None
        self._mock_mode = not load_llm

        if load_llm:
            self._init_llm()
        else:
            print("[INFO - Orchestrator] Running in mock LLM mode.")

    def _init_llm(self) -> None:
        """Load Hugging Face pipeline with A100-friendly settings."""
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                task="text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._mock_mode = False
            print("[INFO - Orchestrator] LLM loaded with bfloat16 and device_map=auto.")
        except Exception as exc:  # pragma: no cover - environment dependent
            self.pipeline = None
            self._mock_mode = True
            print(f"[WARN - Orchestrator] Failed to load LLM; falling back to mock mode. Reason: {exc}")

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """Extract JSON object string from arbitrary model output.

        Args:
            text: Raw generated text.

        Returns:
            JSON candidate string if found.
        """
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fence_match:
            return fence_match.group(1)

        brace_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if brace_match:
            return brace_match.group(1)
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate and parse strict JSON output from the LLM.

        Args:
            system_prompt: High-level behavior prompt.
            user_prompt: Task-specific input.

        Returns:
            Parsed JSON dictionary.
        """
        if self._mock_mode:
            # Deterministic fallback for testability without model access.
            if "success" in user_prompt.lower() or "utility" in user_prompt.lower():
                return {
                    "success": True,
                    "rationale": "Utility and convergence indicate robust cross-tech synergy.",
                    "final_params": {
                        "noma_power_split": [0.58, 0.42],
                        "ris_phase_matrix": [0.1] * 128,
                    },
                }
            return {
                "noma_power_split": [0.56, 0.44],
                "ris_phase_matrix": [0.0] * 128,
                "retrieval_used": True,
            }

        prompt = (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt}\n"
            "Return only JSON without commentary."
        )
        output = self.pipeline(prompt, max_new_tokens=512, do_sample=False)
        generated = output[0]["generated_text"]
        json_text = self._extract_json_block(generated)
        if json_text is None:
            raise ValueError("Failed to locate JSON in model output.")

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Light cleanup for common trailing commas or markdown noise.
            cleaned = re.sub(r",\s*([}\]])", r"\1", json_text)
            return json.loads(cleaned)

    @staticmethod
    def _default_random_init() -> Dict[str, Any]:
        """Return cold-start initialization heuristic."""
        rng = np.random.default_rng()
        split = rng.uniform(0.3, 0.7)
        return {
            "noma_power_split": [float(split), float(1.0 - split)],
            "ris_phase_matrix": rng.uniform(-np.pi, np.pi, size=128).tolist(),
            "source": "cold_start",
        }

    def coordinator_agent(self, scenario_id: str, scenario_desc: str) -> Dict[str, Any]:
        """Produce initialization parameters for the physics solver.

        Args:
            scenario_id: Scenario identifier.
            scenario_desc: Natural language description.

        Returns:
            Initialization parameter dictionary.
        """
        retrieved = self.graphrag.retrieve_similar_strategy(scenario_desc)
        if retrieved is None:
            print(f"[INFO - Orchestrator] Coordinator cold start for {scenario_id}.")
            return self._default_random_init()

        system_prompt = (
            "You are a wireless optimization assistant. Combine retrieved priors "
            "into an initialization JSON with keys: noma_power_split, ris_phase_matrix."
        )
        user_prompt = (
            f"Scenario: {scenario_desc}\n"
            f"Retrieved rationale: {retrieved.get('rationale', '')}\n"
            f"Retrieved params: {json.dumps(retrieved.get('params', {}))}"
        )
        proposal = self.generate_json(system_prompt, user_prompt)
        proposal.setdefault("source", "graphrag_retrieval")
        print(f"[INFO - Orchestrator] Coordinator generated retrieval-based init for {scenario_id}.")
        return proposal

    def evaluator_agent(
        self,
        scenario_id: str,
        scenario_desc: str,
        techs_used: list[str],
        solver_results: Dict[str, Any],
        utility_score: float,
    ) -> Dict[str, Any]:
        """Evaluate solver output and store successful knowledge in GraphRAG.

        Args:
            scenario_id: Scenario identifier.
            scenario_desc: Scenario description.
            techs_used: Active technologies.
            solver_results: Physics solver output dictionary.
            utility_score: Composite utility score U.

        Returns:
            Evaluator decision and rationale dictionary.
        """
        system_prompt = (
            "You are an evaluator for continual 6G optimization. Return JSON with "
            "keys: success (bool), rationale (str), final_params (dict)."
        )
        user_prompt = (
            f"Scenario: {scenario_desc}\n"
            f"Iterations: {solver_results.get('iterations')}\n"
            f"SumRate: {solver_results.get('sum_rate')}\n"
            f"Utility: {utility_score}\n"
            "Mark success true when utility is strong and convergence is efficient."
        )
        decision = self.generate_json(system_prompt, user_prompt)

        success = bool(decision.get("success", utility_score > 0.8))
        if success:
            final_params = decision.get("final_params", solver_results.get("final_params", {}))
            rationale = decision.get("rationale", "Successful convergence with balanced interference tradeoff.")
            self.graphrag.insert_knowledge(
                scenario_id=scenario_id,
                scenario_desc=scenario_desc,
                techs_used=techs_used,
                strategy_params=final_params,
                rationale=rationale,
            )
            print(f"[INFO - Orchestrator] Evaluator stored successful strategy for {scenario_id}.")
        else:
            print(f"[INFO - Orchestrator] Evaluator rejected strategy for {scenario_id}.")

        decision["success"] = success
        return decision
