from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class AgentOrchestrator:
    """Coordinator/Evaluator/Librarian loop for solver-action optimization."""

    def __init__(self, llm_pipeline: Any, physics_env: Any) -> None:
        self.llm_pipeline = llm_pipeline
        self.physics_env = physics_env
        self.max_history_turns = 10
        self.max_history_chars = 12000

    @staticmethod
    def strip_think_tags(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_think_sections(text: str) -> List[str]:
        matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    def _call_llm(self, prompt: str, max_new_tokens: int = 350) -> str:
        if self.llm_pipeline is None:
            raise RuntimeError("LLM pipeline is not available. Program must stop.")
        out = self.llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(out, list) and out:
            first = out[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
                if isinstance(text, str):
                    return text[len(prompt) :] if text.startswith(prompt) else text
        raise RuntimeError("LLM output format is invalid. Program must stop.")

    def _parse_json(self, text: str) -> Dict:
        cleaned = self.strip_think_tags(text)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON object found in LLM output: {cleaned[:600]}")
        payload = cleaned[start : end + 1]
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}; payload={payload[:600]}") from exc

    @staticmethod
    def _require_keys(data: Dict, keys: List[str], agent_name: str) -> None:
        missing = [k for k in keys if k not in data]
        if missing:
            raise ValueError(f"{agent_name} output missing required keys: {missing}")

    @staticmethod
    def _ensure_allowed_value(value: str, allowed: List[str], field_name: str, agent_name: str) -> None:
        if value not in allowed:
            raise ValueError(f"{agent_name} invalid {field_name}={value}; allowed={allowed}")

    @staticmethod
    def _categorize_user_location(distance: float) -> str:
        if distance <= 100:
            return "NEAR"
        if distance <= 190:
            return "MID"
        return "FAR"

    @staticmethod
    def _categorize_channel_quality(blockage_factor: float) -> str:
        if blockage_factor < 0.7:
            return "BLOCKAGE"
        if blockage_factor < 0.85:
            return "WEAK"
        return "STRONG"

    @staticmethod
    def _categorize_interference(pathloss_exp: float) -> str:
        return "HIGH" if pathloss_exp >= 2.45 else "LOW"

    @staticmethod
    def _categorize_satisfaction(rates_mbps: List[float], qos_target: float) -> List[str]:
        out: List[str] = []
        for rate in rates_mbps:
            ratio = rate / (qos_target + 1e-12)
            if ratio >= 1.0:
                out.append("SATISFIED")
            elif ratio >= 0.7:
                out.append("UNSATISFIED")
            else:
                out.append("VERY_UNSATISFIED")
        return out

    def _build_categorized_scenario(self, scenario: Dict, latest_result: Optional[Dict]) -> Dict:
        d_near = float(scenario["d_near_m"])
        d_far = float(scenario["d_far_m"])
        user_location_state = [self._categorize_user_location(d_near), self._categorize_user_location(d_far)]
        channel_quality = [
            self._categorize_channel_quality(float(scenario["blockage_factor"])),
            self._categorize_channel_quality(float(scenario["blockage_factor"] * 0.95)),
        ]
        interference_profile = self._categorize_interference(float(scenario["pathloss_exp"]))

        if latest_result is None:
            satisfaction_status = ["VERY_UNSATISFIED", "VERY_UNSATISFIED"]
        else:
            rates = latest_result.get("scheduled_user_rates_mbps", [0.0, 0.0])
            satisfaction_status = self._categorize_satisfaction(rates, float(scenario["qos_target"]))

        return {
            "scenario_id": scenario["scenario_id"],
            "category": scenario["category"],
            "user_location_state": user_location_state,
            "satisfaction_status": satisfaction_status,
            "channel_quality": channel_quality,
            "interference_profile": interference_profile,
            "qos_target": scenario["qos_target"],
            "eepsu_target": scenario["eepsu_target"],
            "pws_target": scenario["pws_target"],
            "fairness_min": scenario["fairness_min"],
            "base_power_dbm_min": scenario["base_power_dbm_min"],
            "base_power_dbm_max": scenario["base_power_dbm_max"],
        }

    @staticmethod
    def _coordinator_system_prompt(category: str) -> str:
        base = (
            "You are the Coordinator Agent for a multi-user wireless network.\n"
            "You are a strategic decision-maker.\n"
            "Optimization priority: (1) Fairness, (2) EEPSU, (3) PWS.\n"
            "Use categorized scenario values only.\n"
            "Do not output raw CSI-level vectors.\n"
            "You must output STRICT JSON only with schema:\n"
            "{\n"
            "  \"actions\": [\n"
            "    {\"action_type\": \"...\", \"solver_name\": \"solve_noma|solve_ris|none\", \"params\": {...}, \"call\": \"CALL: ...\"}\n"
            "  ],\n"
            "  \"justification\": \"brief logic from satisfaction/fairness/EEPSU/PWS\"\n"
            "}\n"
            "You may include one or more actions in the same iteration.\n"
            "If you choose base-power control, use action_type='increase_base_power' and solver_name='none'.\n"
            "Base power must stay within [base_power_dbm_min, base_power_dbm_max].\n"
            "Near users that are SATISFIED should not consume extra RIS resources.\n"
        )
        if category == "RIS_ONLY":
            return (
                base
                + "Domain: RIS_ONLY. Do not discuss NOMA or hybrid solver.\n"
                "Allowed action_type: solve_ris, increase_base_power. You may call both in the same iteration.\n"
                "For solve_ris params keys: selected_user_indices, phase_resolution, reflection_elements, ris_algorithm.\n"
                "ris_algorithm must be one of: alternating_optimization, manifold, greedy, gradient_descent.\n"
                "Each entry in actions must use call text like: CALL: solve_ris(...) or CALL: increase_base_power(...)."
            )
        if category == "NOMA_ONLY":
            return (
                base
                + "Domain: NOMA_ONLY. Do not discuss RIS or hybrid solver.\n"
                "Allowed action_type: solve_noma, increase_base_power. You may call both in the same iteration.\n"
                "For solve_noma params keys must be exactly: h, P_max, sigma_sq, min_rate.\n"
                "Use function call form: solve_noma(h, P_max, sigma_sq, min_rate).\n"
                "Each entry in actions must use call text like: CALL: solve_noma(...) or CALL: increase_base_power(...)."
            )
        return (
            base
            + "Domain: JOINT. You may choose solve_noma, solve_ris, and/or increase_base_power in the same iteration.\n"
            "For solve_noma params keys: h, P_max, sigma_sq, min_rate.\n"
            "For solve_ris params keys: selected_user_indices, phase_resolution, reflection_elements, ris_algorithm.\n"
            "You can include any subset of these actions in one turn.\n"
            "Each entry in actions must use call text like: CALL: solve_noma(...) / CALL: solve_ris(...) / CALL: increase_base_power(...)."
        )

    @staticmethod
    def _evaluator_system_prompt(category: str) -> str:
        domain_line = (
            "RIS-only focus" if category == "RIS_ONLY" else "NOMA-only focus" if category == "NOMA_ONLY" else "Joint RIS+NOMA focus"
        )
        return (
            "You are the Evaluator Agent in a multi-turn optimization dialogue.\n"
            f"Domain context: {domain_line}.\n"
            "Do NOT output a concept/policy.\n"
            "You must guide the next coordinator turn and provide a refined RAG query.\n"
            "Optimization criteria: all users satisfied first, then maximize EEPSU and PWS.\n"
            "Output STRICT JSON only with schema:\n"
            "{\n"
            "  \"action\": \"continue|stop\",\n"
            "  \"guidance\": \"what coordinator should change next\",\n"
            "  \"rag_query\": \"optional scenario description for next retrieval\",\n"
            "  \"diagnosis\": \"brief reason based on EEPSU/PWS/fairness\"\n"
            "}"
        )

    def _summarize_history_if_needed(self, history: List[Dict]) -> List[Dict]:
        history_chars = len(json.dumps(history))
        if len(history) <= self.max_history_turns and history_chars <= self.max_history_chars:
            return history
        prompt = (
            "Summarize this coordinator-evaluator interaction while preserving optimization-critical details.\n"
            "Return STRICT JSON with keys: summary, preserved_actions, preserved_failures.\n"
            f"History: {json.dumps(history)}"
        )
        parsed = self._parse_json(self._call_llm(prompt, max_new_tokens=350))
        self._require_keys(parsed, ["summary", "preserved_actions", "preserved_failures"], "HistorySummarizer")
        return [
            {
                "type": "summary",
                "summary": parsed["summary"],
                "preserved_actions": parsed["preserved_actions"],
                "preserved_failures": parsed["preserved_failures"],
            }
        ]

    @staticmethod
    def _to_action_payload(solver_json: Dict, scenario: Dict) -> Dict:
        actions = solver_json.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Coordinator output must include non-empty 'actions' list")

        normalized_actions: List[Dict] = []
        for idx, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                raise ValueError(f"Action #{idx} must be an object")
            for required_key in ["action_type", "solver_name", "params", "call"]:
                if required_key not in action:
                    raise ValueError(f"Action #{idx} missing key: {required_key}")

            action_type = str(action["action_type"])
            solver_name = str(action["solver_name"])
            params = action["params"]

            if action_type == "increase_base_power":
                if "target_base_power_dbm" not in params:
                    raise ValueError("increase_base_power params must include target_base_power_dbm")
                power_dbm = float(params["target_base_power_dbm"])
                p_min = float(scenario["base_power_dbm_min"])
                p_max = float(scenario["base_power_dbm_max"])
                if power_dbm < p_min or power_dbm > p_max:
                    raise ValueError(f"target_base_power_dbm={power_dbm} outside range [{p_min}, {p_max}]")
                normalized_actions.append(
                    {
                        "action_type": "increase_base_power",
                        "target_base_power_dbm": power_dbm,
                    }
                )
                continue

            if action_type == "solve_noma":
                if solver_name != "solve_noma":
                    raise ValueError("solve_noma action must set solver_name='solve_noma'")
                required = ["h", "P_max", "sigma_sq", "min_rate"]
                missing = [k for k in required if k not in params]
                if missing:
                    raise ValueError(f"solve_noma params missing: {missing}")
                normalized_actions.append(
                    {
                        "action_type": "solve_noma",
                        "solve_noma": {
                            "h": params["h"],
                            "P_max": float(params["P_max"]),
                            "sigma_sq": float(params["sigma_sq"]),
                            "min_rate": float(params["min_rate"]),
                        },
                    }
                )
                continue

            if action_type == "solve_ris":
                if solver_name != "solve_ris":
                    raise ValueError("solve_ris action must set solver_name='solve_ris'")
                required = ["selected_user_indices", "phase_resolution", "reflection_elements", "ris_algorithm"]
                missing = [k for k in required if k not in params]
                if missing:
                    raise ValueError(f"solve_ris params missing: {missing}")
                normalized_actions.append(
                    {
                        "action_type": "solve_ris",
                        "solve_ris": {
                            "selected_user_indices": params["selected_user_indices"],
                            "phase_resolution": float(params["phase_resolution"]),
                            "reflection_elements": float(params["reflection_elements"]),
                            "ris_algorithm": str(params["ris_algorithm"]),
                        },
                    }
                )
                continue

            raise ValueError(f"Unsupported action_type: {action_type}")

        return {"actions": normalized_actions}

    @staticmethod
    def _validate_domain_action(category: str, action_type: str) -> None:
        if category == "NOMA_ONLY":
            allowed = ["solve_noma", "increase_base_power"]
        elif category == "RIS_ONLY":
            allowed = ["solve_ris", "increase_base_power"]
        else:
            allowed = ["solve_noma", "solve_ris", "increase_base_power"]
        if action_type not in allowed:
            raise ValueError(f"Action {action_type} not allowed in category {category}; allowed={allowed}")

    def coordinator_agent(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        iteration: int,
        rag_query: Optional[str],
        evaluator_guidance: str,
        chat_history: List[Dict],
        latest_result: Optional[Dict],
    ) -> Dict:
        if rag_query is None:
            retrieved = []
        else:
            retrieved = active_db.retrieve_concepts(
                query=rag_query,
                top_k=5,
                domain_filters=active_domains if active_domains else None,
            )
        categorized = self._build_categorized_scenario(scenario, latest_result)
        coordinator_prompt = (
            f"{self._coordinator_system_prompt(scenario['category'])}\n"
            f"Categorized Scenario: {json.dumps(categorized)}\n"
            f"Active Domains: {active_domains}\n"
            f"Retrieved Concepts: {json.dumps(retrieved)}\n"
            f"Evaluator Guidance: {evaluator_guidance}\n"
            f"Chat History: {json.dumps(chat_history)}\n"
            "Return JSON only."
        )

        llm_text = self._call_llm(coordinator_prompt)
        parsed = self._parse_json(llm_text)
        self._require_keys(parsed, ["actions", "justification"], "CoordinatorAgent")
        actions = parsed.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("CoordinatorAgent must output a non-empty actions list")
        for idx, action in enumerate(actions, start=1):
            if not isinstance(action, dict):
                raise ValueError(f"CoordinatorAgent action #{idx} is not an object")
            self._require_keys(action, ["action_type", "solver_name", "params", "call"], "CoordinatorAgent")
            self._validate_domain_action(scenario["category"], str(action["action_type"]))

            if str(action["action_type"]) == "solve_ris":
                ris_algo = str(action["params"].get("ris_algorithm", ""))
                self._ensure_allowed_value(
                    ris_algo,
                    ["alternating_optimization", "manifold", "greedy", "gradient_descent"],
                    "ris_algorithm",
                    "CoordinatorAgent",
                )

        physics_params = self._to_action_payload(parsed, scenario)
        return {
            "solver": parsed,
            "action_payload": physics_params,
            "retrieved": retrieved,
            "prompt": coordinator_prompt,
            "raw_response": llm_text,
            "think_sections": self._extract_think_sections(llm_text),
        }

    def evaluator_agent(self, scenario: Dict, coordinator_output: Dict, result: Dict, chat_history: List[Dict]) -> Dict:
        categorized = self._build_categorized_scenario(scenario, result)
        evaluator_prompt = (
            f"{self._evaluator_system_prompt(scenario['category'])}\n"
            f"Categorized Scenario: {json.dumps(categorized)}\n"
            f"Coordinator Decision: {json.dumps(coordinator_output['solver'])}\n"
            f"Physics Result: {json.dumps(result)}\n"
            f"Chat History: {json.dumps(chat_history)}\n"
            "Return JSON only."
        )
        llm_text = self._call_llm(evaluator_prompt)
        parsed = self._parse_json(llm_text)
        self._require_keys(parsed, ["action", "guidance", "diagnosis"], "EvaluatorAgent")
        if "rag_query" in parsed and parsed["rag_query"] is not None and not isinstance(parsed["rag_query"], str):
            raise ValueError("EvaluatorAgent rag_query must be a string or null when provided")
        return {
            "parsed": parsed,
            "prompt": evaluator_prompt,
            "raw_response": llm_text,
            "think_sections": self._extract_think_sections(llm_text),
        }

    def librarian_agent(self, scenario: Dict, final_result: Dict, chat_history: List[Dict]) -> Dict:
        category = scenario.get("category", "JOINT")
        if category == "RIS_ONLY":
            domain = "ris"
        elif category == "NOMA_ONLY":
            domain = "noma"
        else:
            domain = "joint"
        prompt = (
            "You are the Librarian Agent. Learn one reusable concept from the full coordinator-evaluator dialogue.\n"
            "Output STRICT JSON with keys: condition, rule, policy.\n"
            "Do not invent metrics not present in the chat.\n"
            f"Scenario: {json.dumps(scenario)}\n"
            f"Final Physics Result: {json.dumps(final_result)}\n"
            f"Dialogue History: {json.dumps(chat_history)}\n"
            "Return JSON only."
        )
        parsed = self._parse_json(self._call_llm(prompt, max_new_tokens=400))
        self._require_keys(parsed, ["condition", "rule", "policy"], "LibrarianAgent")
        parsed["domain"] = domain
        return parsed

    @staticmethod
    def _targets_met(result: Dict, scenario: Dict) -> bool:
        all_users_satisfied = all(rate >= float(scenario["qos_target"]) for rate in result["scheduled_user_rates_mbps"])
        return (
            all_users_satisfied
            and float(result["jain_fairness"]) >= float(scenario["fairness_min"])
            and float(result["eepsu"]) >= float(scenario["eepsu_target"])
            and float(result["pws"]) >= float(scenario["pws_target"])
        )

    def _run_loop(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        max_iterations: int,
        learn: bool,
    ) -> Dict:
        final_coordinator_output: Optional[Dict] = None
        final_result: Optional[Dict] = None
        used_iterations = 0
        rag_query: Optional[str] = scenario["description"]
        evaluator_guidance = "Start from fairness-first strategy."
        chat_history: List[Dict] = []
        full_chat_history: List[Dict] = []
        latest_result: Optional[Dict] = None

        while used_iterations < max_iterations:
            used_iterations += 1
            chat_history = self._summarize_history_if_needed(chat_history)
            coord_out = self.coordinator_agent(
                scenario=scenario,
                active_domains=active_domains,
                active_db=active_db,
                iteration=used_iterations,
                rag_query=rag_query,
                evaluator_guidance=evaluator_guidance,
                chat_history=chat_history,
                latest_result=latest_result,
            )
            result = self.physics_env.evaluate(scenario, coord_out["action_payload"])
            latest_result = result

            final_coordinator_output = coord_out
            final_result = result

            coordinator_turn = {
                "turn": used_iterations,
                "agent": "coordinator",
                "prompt": coord_out.get("prompt", ""),
                "raw_response": coord_out.get("raw_response", ""),
                "result_snapshot": {
                    "jain_fairness": result["jain_fairness"],
                    "eepsu": result["eepsu"],
                    "pws": result["pws"],
                    "scheduled_user_rates_mbps": result.get("scheduled_user_rates_mbps", []),
                },
            }
            full_chat_history.append(coordinator_turn)
            chat_history.append(coordinator_turn)

            if self._targets_met(result, scenario):
                break

            eval_out = self.evaluator_agent(scenario, coord_out, result, chat_history)
            evaluator_turn = {
                "turn": used_iterations,
                "agent": "evaluator",
                "prompt": eval_out.get("prompt", ""),
                "raw_response": eval_out.get("raw_response", ""),
            }
            full_chat_history.append(evaluator_turn)
            chat_history.append(evaluator_turn)

            evaluator_guidance = str(eval_out["parsed"]["guidance"])
            if "rag_query" in eval_out["parsed"]:
                next_rag_query = eval_out["parsed"]["rag_query"]
                if next_rag_query is None:
                    rag_query = None
                elif str(next_rag_query).strip():
                    rag_query = str(next_rag_query)
            if str(eval_out["parsed"]["action"]).lower() == "stop":
                break

        assert final_coordinator_output is not None and final_result is not None

        out: Dict[str, Any] = {
            "agent_iterations": used_iterations,
            "params": final_result["executed_params"],
            "result": final_result,
            "solver_call": final_coordinator_output["solver"],
            "chat_history": full_chat_history,
            "working_history": chat_history,
        }

        if learn:
            concept = self.librarian_agent(scenario, final_result, full_chat_history)
            active_db.learn_concept(
                condition=concept["condition"],
                rule=concept["rule"],
                utility_score=float(final_result["pws"] + final_result["eepsu"]),
                domain=concept["domain"],
                policy=concept["policy"],
            )
            out["concept"] = concept
        return out

    def run_agentic_optimization(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        max_iterations: int = 30,
    ) -> Dict:
        return self._run_loop(
            scenario=scenario,
            active_domains=active_domains,
            active_db=active_db,
            max_iterations=max_iterations,
            learn=True,
        )

    def run_agentic_evaluation(
        self,
        scenario: Dict,
        active_domains: List[str],
        active_db: Any,
        max_iterations: int = 30,
    ) -> Dict:
        return self._run_loop(
            scenario=scenario,
            active_domains=active_domains,
            active_db=active_db,
            max_iterations=max_iterations,
            learn=False,
        )
