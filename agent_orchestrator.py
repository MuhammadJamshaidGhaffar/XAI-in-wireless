from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional


class AgentOrchestrator:
    """Coordinator/Evaluator/Librarian loop for cognitive optimization."""

    def __init__(self, llm_pipeline: Any, physics_env: Any) -> None:
        self.llm_pipeline = llm_pipeline
        self.physics_env = physics_env
        self._rng = random.Random(42)

    @staticmethod
    def strip_think_tags(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _call_llm(self, prompt: str, max_new_tokens: int = 350) -> str:
        if self.llm_pipeline is None:
            return ""
        out = self.llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
                if isinstance(text, str):
                    return text[len(prompt) :] if text.startswith(prompt) else text
        return ""

    def _parse_json(self, text: str) -> Dict:
        cleaned = self.strip_think_tags(text)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {}
        payload = cleaned[start : end + 1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}

    def _fallback_params(self, scenario: Dict, iteration: int) -> Dict:
        category = scenario.get("category", "JOINT")
        base = 0.65 if category in {"NOMA_ONLY", "JOINT"} else 0.5
        jitter = (self._rng.random() - 0.5) * 0.2
        noma_power = max(0.05, min(0.95, base + jitter - 0.02 * iteration))

        return {
            "noma_power_split": noma_power,
            "ris_phase_offset": max(-3.14, min(3.14, 0.4 - 0.1 * iteration + jitter)),
            "ris_reflection_amplitude": max(0.1, min(1.0, 0.92 - 0.04 * iteration)),
            "sic_residual": max(0.01, min(0.3, 0.12 - 0.01 * iteration)),
        }

    def coordinator_agent(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        iteration: int,
        feedback_log: List[Dict],
    ) -> Dict:
        retrieved = active_db.retrieve_concepts(
            query=scenario.get("description", ""),
            top_k=5,
            domain_filters=active_domains if active_domains else None,
        )
        coordinator_prompt = (
            "You are the Coordinator Agent for an O-RAN optimization system.\n"
            "Output STRICT JSON with numeric keys only:\n"
            "noma_power_split, ris_phase_offset, ris_reflection_amplitude, sic_residual.\n"
            "Keep values within physical bounds: noma_power_split:[0.05,0.95],"
            " ris_phase_offset:[-3.14,3.14], ris_reflection_amplitude:[0,1], sic_residual:[0,1].\n"
            f"Scenario: {json.dumps(scenario)}\n"
            f"Active Domains: {active_domains}\n"
            f"Retrieved Concepts: {json.dumps(retrieved)}\n"
            f"Failure Feedback: {json.dumps(feedback_log[-3:])}\n"
            "Return JSON only."
        )

        llm_text = self._call_llm(coordinator_prompt)
        parsed = self._parse_json(llm_text)
        if not parsed:
            parsed = self._fallback_params(scenario, iteration)

        params = {
            "noma_power_split": float(parsed.get("noma_power_split", self._fallback_params(scenario, iteration)["noma_power_split"])),
            "ris_phase_offset": float(parsed.get("ris_phase_offset", self._fallback_params(scenario, iteration)["ris_phase_offset"])),
            "ris_reflection_amplitude": float(
                parsed.get("ris_reflection_amplitude", self._fallback_params(scenario, iteration)["ris_reflection_amplitude"])
            ),
            "sic_residual": float(parsed.get("sic_residual", self._fallback_params(scenario, iteration)["sic_residual"])),
        }
        return params

    def evaluator_agent(self, scenario: Dict, params: Dict, result: Dict) -> Dict:
        evaluator_prompt = (
            "You are the Evaluator Agent in an O-RAN cognitive loop.\n"
            "Write one concise scientific concept explaining why this outcome occurred.\n"
            "Output STRICT JSON with keys: condition, rule, explanation, policy.\n"
            f"Scenario: {json.dumps(scenario)}\n"
            f"Params: {json.dumps(params)}\n"
            f"Physics Result: {json.dumps(result)}\n"
            "Return JSON only."
        )
        llm_text = self._call_llm(evaluator_prompt)
        parsed = self._parse_json(llm_text)
        if parsed:
            return parsed

        condition = (
            f"Category={scenario.get('category')} with blockage={scenario.get('blockage_factor')} "
            f"and ris_elements={scenario.get('ris_elements')} produced U={result.get('domain_utility_score', 0.0):.4f}."
        )
        rule = (
            f"Set noma_power_split={params.get('noma_power_split', 0.65):.3f}, "
            f"ris_reflection_amplitude={params.get('ris_reflection_amplitude', 0.9):.3f}, "
            f"sic_residual={params.get('sic_residual', 0.08):.3f} to balance fairness and interference."
        )
        return {
            "condition": condition,
            "rule": rule,
            "explanation": "Physics indicates utility responds to power split/SIC/reflection coupling.",
            "policy": "Adaptive Interference-Reflection Balancing",
        }

    @staticmethod
    def librarian_agent(concept: Dict, scenario: Dict) -> Dict:
        category = scenario.get("category", "JOINT")
        if category == "RIS_ONLY":
            domain = "ris"
        elif category == "NOMA_ONLY":
            domain = "noma"
        else:
            domain = "joint"
        out = dict(concept)
        out["domain"] = domain
        return out

    def run_agentic_optimization(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        max_iterations: int = 10,
        utility_success_threshold: float = 0.70,
    ) -> Dict:
        feedback_log: List[Dict] = []
        final_params: Optional[Dict] = None
        final_result: Optional[Dict] = None
        used_iterations = 0

        while used_iterations < max_iterations:
            used_iterations += 1
            params = self.coordinator_agent(
                scenario=scenario,
                active_domains=active_domains,
                active_db=active_db,
                iteration=used_iterations,
                feedback_log=feedback_log,
            )
            result = self.physics_env.evaluate(scenario, params)

            final_params = params
            final_result = result

            if float(result["domain_utility_score"]) >= utility_success_threshold:
                break

            feedback_log.append(
                {
                    "iteration": used_iterations,
                    "utility": result["domain_utility_score"],
                    "sum_rate": result["sum_rate"],
                    "params": params,
                }
            )

        assert final_params is not None and final_result is not None

        concept = self.evaluator_agent(scenario, final_params, final_result)
        tagged_concept = self.librarian_agent(concept, scenario)

        active_db.learn_concept(
            condition=tagged_concept.get("condition", ""),
            rule=tagged_concept.get("rule", ""),
            utility_score=float(final_result["domain_utility_score"]),
            domain=tagged_concept.get("domain", "joint"),
            policy=tagged_concept.get("policy", "Learned Policy"),
        )

        return {
            "agent_iterations": used_iterations,
            "params": final_params,
            "result": final_result,
            "concept": tagged_concept,
        }

    def run_agentic_evaluation(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        max_iterations: int = 10,
        utility_success_threshold: float = 0.70,
    ) -> Dict:
        feedback_log: List[Dict] = []
        final_params: Optional[Dict] = None
        final_result: Optional[Dict] = None
        used_iterations = 0

        while used_iterations < max_iterations:
            used_iterations += 1
            params = self.coordinator_agent(
                scenario=scenario,
                active_domains=active_domains,
                active_db=active_db,
                iteration=used_iterations,
                feedback_log=feedback_log,
            )
            result = self.physics_env.evaluate(scenario, params)

            final_params = params
            final_result = result

            if float(result["domain_utility_score"]) >= utility_success_threshold:
                break

            feedback_log.append(
                {
                    "iteration": used_iterations,
                    "utility": result["domain_utility_score"],
                    "sum_rate": result["sum_rate"],
                    "params": params,
                }
            )

        assert final_params is not None and final_result is not None
        return {
            "agent_iterations": used_iterations,
            "params": final_params,
            "result": final_result,
        }
