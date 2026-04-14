from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PhysicsResult:
    sum_rate: float
    eepsu: float
    pws: float
    qos_achieved: float
    qos_target: float
    snr_achieved: float
    snr_target: float
    reflection_efficiency: float
    jain_fairness: float
    sic_error_penalty: float
    domain_utility_score: float
    noma_u1_power_ratio: float
    ris_u3_snr: float
    scheduled_user_rates_mbps: list


class PhysicsEnvironment:
    """Ground-truth physical layer calculator for RIS/NOMA/JOINT operation."""

    def __init__(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.2) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _dbm_to_watts(dbm: float) -> float:
        return 10 ** ((dbm - 30.0) / 10.0)

    @staticmethod
    def _path_gain(distance_m: float, exponent: float, blockage_factor: float) -> float:
        d = max(distance_m, 1.0)
        return blockage_factor * (d ** (-exponent))

    @staticmethod
    def _phase_alignment_efficiency(phase_offset_rad: float) -> float:
        # 1.0 when aligned, decays smoothly with phase mismatch.
        return float(0.5 * (1.0 + math.cos(phase_offset_rad)))

    @staticmethod
    def _jain_fairness(rates: np.ndarray) -> float:
        numerator = float(np.sum(rates) ** 2)
        denominator = float(len(rates) * np.sum(rates ** 2) + 1e-12)
        return numerator / denominator

    def calc_ris_utility(
        self,
        snr_achieved: float,
        snr_target: float,
        reflection_efficiency: float,
    ) -> float:
        return self.alpha * (snr_achieved / (snr_target + 1e-12)) + self.beta * reflection_efficiency

    def calc_noma_utility(
        self,
        fairness: float,
        sum_rate: float,
        sic_error_penalty: float,
    ) -> float:
        return self.alpha * fairness + self.beta * sum_rate - self.gamma * sic_error_penalty

    def calc_joint_utility(
        self,
        qos_achieved: float,
        qos_target: float,
        eepsu: float,
    ) -> float:
        return self.alpha * (qos_achieved / (qos_target + 1e-12)) + self.beta * eepsu

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    @staticmethod
    def _parse_noma_channel_state(h: object) -> Tuple[float, float]:
        if isinstance(h, dict):
            h_far = float(h.get("h_far", h.get("far", 1e-8)))
            h_near = float(h.get("h_near", h.get("near", max(1.8 * h_far, 1e-8))))
            return max(h_far, 1e-8), max(h_near, 1e-8)
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            return max(float(h[0]), 1e-8), max(float(h[1]), 1e-8)
        h_scalar = max(float(h), 1e-8)
        return h_scalar, max(1.8 * h_scalar, 1e-8)

    @staticmethod
    def _noma_rates_for_split(p_far: float, p_total: float, h_far: float, h_near: float, sigma_sq: float, sic_residual: float) -> Tuple[float, float]:
        p_far_w = p_far * p_total
        p_near_w = (1.0 - p_far) * p_total
        sinr_far = (p_far_w * h_far) / (p_near_w * h_far + sigma_sq + 1e-18)
        sinr_near = (p_near_w * h_near) / (sic_residual * p_far_w * h_near + sigma_sq + 1e-18)
        return float(math.log2(1.0 + sinr_far)), float(math.log2(1.0 + sinr_near))

    @staticmethod
    def _objective_noma(rate_far: float, rate_near: float, min_rate: float) -> float:
        rates = np.asarray([rate_far, rate_near], dtype=np.float64)
        fairness = (np.sum(rates) ** 2) / (2.0 * np.sum(rates ** 2) + 1e-12)
        deficit_far = max(0.0, min_rate - rate_far)
        deficit_near = max(0.0, min_rate - rate_near)
        deficit_penalty = deficit_far + deficit_near
        return float(1.8 * fairness + 0.2 * np.sum(rates) - 8.0 * deficit_penalty)

    def solve_noma(self, h: float, P_max: float, sigma_sq: float, min_rate: float) -> Dict:
        h_far, h_near = self._parse_noma_channel_state(h)
        p_total = max(float(P_max), 1e-8)
        sigma = max(float(sigma_sq), 1e-12)
        target = max(float(min_rate), 0.01)

        # Stage 1: coarse global search over feasible power splits.
        sic_residual = 0.05
        best_p = 0.6
        best_score = -1e18
        for p_far in np.linspace(0.05, 0.95, 361):
            r_far, r_near = self._noma_rates_for_split(p_far, p_total, h_far, h_near, sigma, sic_residual)
            score = self._objective_noma(r_far, r_near, target)
            if score > best_score:
                best_score = score
                best_p = float(p_far)

        # Stage 2: projected gradient refinement around the coarse optimum.
        p = best_p
        step = 0.04
        for _ in range(45):
            eps = 1e-3
            p_up = min(0.95, p + eps)
            p_dn = max(0.05, p - eps)
            r_far_up, r_near_up = self._noma_rates_for_split(p_up, p_total, h_far, h_near, sigma, sic_residual)
            r_far_dn, r_near_dn = self._noma_rates_for_split(p_dn, p_total, h_far, h_near, sigma, sic_residual)
            grad = (self._objective_noma(r_far_up, r_near_up, target) - self._objective_noma(r_far_dn, r_near_dn, target)) / (p_up - p_dn + 1e-12)
            p = max(0.05, min(0.95, p + step * grad))
            step *= 0.93

        final_far, final_near = self._noma_rates_for_split(p, p_total, h_far, h_near, sigma, sic_residual)
        imbalance = abs(final_far - final_near) / (max(final_far, final_near) + 1e-12)
        sic_residual = max(0.01, min(0.25, 0.04 + 0.16 * imbalance))

        return {
            "noma_power_split": float(p),
            "sic_residual": float(sic_residual),
        }

    @staticmethod
    def _deterministic_ris_channels(selected_user_indices: List[int], n_elements: int) -> np.ndarray:
        if not selected_user_indices:
            return np.ones((1, n_elements), dtype=np.complex128)
        channels: List[np.ndarray] = []
        for idx in selected_user_indices:
            seed = 1009 + int(idx) * 97 + n_elements * 13
            rng = np.random.default_rng(seed)
            amp = 0.6 + 0.35 * rng.random(n_elements)
            phase = rng.uniform(-math.pi, math.pi, size=n_elements)
            channels.append(amp * np.exp(1j * phase))
        return np.vstack(channels)

    @staticmethod
    def _ris_objective(theta: np.ndarray, h_users: np.ndarray) -> float:
        eff = h_users @ theta
        return float(np.sum(np.abs(eff) ** 2))

    def _solve_ris_alternating(self, h_users: np.ndarray, n_iters: int) -> np.ndarray:
        n_elements = h_users.shape[1]
        theta = np.exp(1j * np.zeros(n_elements))
        for _ in range(n_iters):
            combined = np.sum(np.conj(h_users), axis=0)
            theta = np.exp(1j * np.angle(combined + 1e-12))
        return theta

    def _solve_ris_manifold(self, h_users: np.ndarray, n_iters: int) -> np.ndarray:
        n_elements = h_users.shape[1]
        theta = np.exp(1j * np.zeros(n_elements))
        step = 0.12
        for _ in range(n_iters):
            eff = h_users @ theta
            grad = h_users.conj().T @ eff
            tangent = grad - np.real(grad * np.conj(theta)) * theta
            theta = theta + step * tangent
            theta = np.exp(1j * np.angle(theta))
            step *= 0.98
        return theta

    def _solve_ris_greedy(self, h_users: np.ndarray, phase_levels: int) -> np.ndarray:
        n_elements = h_users.shape[1]
        theta = np.ones(n_elements, dtype=np.complex128)
        quantized = np.exp(1j * np.linspace(-math.pi, math.pi, max(phase_levels, 8), endpoint=False))
        for n in range(n_elements):
            best_val = -1e18
            best_phase = 1 + 0j
            for q in quantized:
                theta_c = theta.copy()
                theta_c[n] = q
                val = self._ris_objective(theta_c, h_users)
                if val > best_val:
                    best_val = val
                    best_phase = q
            theta[n] = best_phase
        return theta

    def _solve_ris_gradient_descent(self, h_users: np.ndarray, n_iters: int) -> np.ndarray:
        n_elements = h_users.shape[1]
        theta = np.exp(1j * np.zeros(n_elements))
        step = 0.18
        for _ in range(n_iters):
            eff = h_users @ theta
            grad = h_users.conj().T @ eff
            theta = theta + step * grad
            theta = np.exp(1j * np.angle(theta))
            step *= 0.96
        return theta

    def solve_ris(
        self,
        selected_user_indices: list,
        phase_resolution: float,
        reflection_elements: float,
        ris_algorithm: str,
        max_ris_elements: float,
    ) -> Dict:
        if ris_algorithm not in ["alternating_optimization", "manifold", "greedy", "gradient_descent"]:
            raise ValueError(f"Unsupported RIS algorithm: {ris_algorithm}")

        n_elements = int(max(4, min(reflection_elements, max_ris_elements)))
        h_users = self._deterministic_ris_channels(list(selected_user_indices), n_elements)

        if ris_algorithm == "alternating_optimization":
            theta = self._solve_ris_alternating(h_users, n_iters=30)
        elif ris_algorithm == "manifold":
            theta = self._solve_ris_manifold(h_users, n_iters=45)
        elif ris_algorithm == "greedy":
            theta = self._solve_ris_greedy(h_users, phase_levels=int(max(8.0, phase_resolution)))
        else:
            theta = self._solve_ris_gradient_descent(h_users, n_iters=40)

        objective_val = self._ris_objective(theta, h_users)
        norm_obj = objective_val / (n_elements * max(len(selected_user_indices), 1) + 1e-12)
        reflection_ratio = self._clip01(n_elements / max(max_ris_elements, 1.0))
        reflection_amp = self._clip01(reflection_ratio * (1.0 - math.exp(-norm_obj)))
        phase_offset = float(np.angle(np.mean(theta)))

        return {
            "ris_phase_offset": float(phase_offset),
            "ris_reflection_amplitude": float(reflection_amp),
        }

    def _execute_action(self, scenario: Dict, action_bundle: Dict) -> Dict:
        actions = action_bundle.get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError("Action bundle must contain a non-empty 'actions' list")

        current_bs_power_dbm = float(scenario["bs_power_dbm"])

        executed = {
            "noma_power_split": None,
            "ris_phase_offset": None,
            "ris_reflection_amplitude": None,
            "sic_residual": None,
            "power_scale": 1.0,
            "executed_base_power_dbm": current_bs_power_dbm,
            "executed_action_types": [],
        }

        for action in actions:
            action_type = action["action_type"]

            if action_type == "increase_base_power":
                new_power_dbm = float(action["target_base_power_dbm"])
                p_min = float(scenario["base_power_dbm_min"])
                p_max = float(scenario["base_power_dbm_max"])
                if new_power_dbm < p_min or new_power_dbm > p_max:
                    raise ValueError(f"Base power request {new_power_dbm} out of range [{p_min}, {p_max}]")
                executed["power_scale"] = 10 ** ((new_power_dbm - current_bs_power_dbm) / 10.0)
                executed["executed_base_power_dbm"] = new_power_dbm
                executed["executed_action_types"].append("increase_base_power")
                continue

            if action_type == "solve_noma":
                s = action["solve_noma"]
                noma_out = self.solve_noma(
                    h=s["h"],
                    P_max=float(s["P_max"]),
                    sigma_sq=float(s["sigma_sq"]),
                    min_rate=float(s["min_rate"]),
                )
                executed["noma_power_split"] = float(noma_out["noma_power_split"])
                executed["sic_residual"] = float(noma_out["sic_residual"])
                executed["executed_action_types"].append("solve_noma")
                continue

            if action_type == "solve_ris":
                s = action["solve_ris"]
                ris_out = self.solve_ris(
                    selected_user_indices=list(s["selected_user_indices"]),
                    phase_resolution=float(s["phase_resolution"]),
                    reflection_elements=float(s["reflection_elements"]),
                    ris_algorithm=str(s["ris_algorithm"]),
                    max_ris_elements=float(scenario["ris_elements"]),
                )
                executed["ris_phase_offset"] = float(ris_out["ris_phase_offset"])
                executed["ris_reflection_amplitude"] = float(ris_out["ris_reflection_amplitude"])
                executed["executed_action_types"].append("solve_ris")
                continue

            raise ValueError(f"Unknown action_type: {action_type}")

        return executed

    def evaluate(self, scenario: Dict, params: Dict) -> Dict:
        executed = self._execute_action(scenario, params)
        category = scenario["category"]
        tx_power_w = self._dbm_to_watts(float(scenario["bs_power_dbm"])) * float(executed["power_scale"])
        noise_w = self._dbm_to_watts(float(scenario["noise_dbm"]))
        bandwidth_hz = float(scenario["bandwidth_hz"])
        pathloss_exp = float(scenario["pathloss_exp"])
        blockage = float(scenario["blockage_factor"])
        qos_target = float(scenario["qos_target"])
        snr_target = float(scenario["snr_target"])

        d_far = float(scenario["d_far_m"])
        d_near = float(scenario["d_near_m"])
        d_bs_ris = float(scenario["d_bs_ris_m"])
        d_ris_u3 = float(scenario["d_ris_u3_m"])
        ris_elements = float(scenario["ris_elements"])

        p_far = float(executed["noma_power_split"]) if executed["noma_power_split"] is not None else float(
            scenario.get("default_noma_power_split", 0.65)
        )
        p_far = max(0.05, min(0.95, p_far))
        p_near = 1.0 - p_far

        phase_offset = float(executed["ris_phase_offset"]) if executed["ris_phase_offset"] is not None else 0.0
        reflection_amplitude = (
            float(executed["ris_reflection_amplitude"]) if executed["ris_reflection_amplitude"] is not None else 0.0
        )
        reflection_amplitude = max(0.0, min(1.0, reflection_amplitude))
        sic_residual = float(executed["sic_residual"]) if executed["sic_residual"] is not None else float(
            scenario.get("default_sic_residual", 0.08)
        )
        sic_residual = max(0.0, min(1.0, sic_residual))

        g_far_direct = self._path_gain(d_far, pathloss_exp, blockage)
        g_near_direct = self._path_gain(d_near, pathloss_exp, blockage)
        g_bs_ris = self._path_gain(d_bs_ris, 2.0, 1.0)
        g_ris_u3 = self._path_gain(d_ris_u3, 2.0, blockage)

        phase_eff = self._phase_alignment_efficiency(phase_offset)
        reflection_eff = reflection_amplitude * phase_eff
        ris_gain_linear = ris_elements * g_bs_ris * g_ris_u3 * reflection_eff

        if category == "RIS_ONLY":
            snr_achieved = tx_power_w * ris_gain_linear / (noise_w + 1e-18)
            rate_ris = (bandwidth_hz * math.log2(1.0 + snr_achieved)) / 1e6
            sum_rate = rate_ris
            scheduled_rates = [rate_ris]
            weights = [1.0]
            pws = float(np.dot(np.asarray(weights), np.asarray(scheduled_rates)))
            eepsu = sum_rate / (len(scheduled_rates) * (tx_power_w + 0.01 * ris_elements))
            qos_achieved = sum_rate
            utility = self.calc_ris_utility(snr_achieved=snr_achieved, snr_target=snr_target, reflection_efficiency=reflection_eff)

            fairness = 1.0
            sic_penalty = 0.0
            noma_ratio = p_far

        elif category == "NOMA_ONLY":
            h_far = g_far_direct
            h_near = g_near_direct

            sinr_far = (p_far * tx_power_w * h_far) / (p_near * tx_power_w * h_far + noise_w + 1e-18)
            sinr_far_at_near = (p_far * tx_power_w * h_near) / (p_near * tx_power_w * h_near + noise_w + 1e-18)
            sinr_near = (p_near * tx_power_w * h_near) / (sic_residual * p_far * tx_power_w * h_near + noise_w + 1e-18)

            rate_far = (bandwidth_hz * math.log2(1.0 + sinr_far)) / 1e6
            rate_near = (bandwidth_hz * math.log2(1.0 + sinr_near)) / 1e6
            rates = np.array([rate_far, rate_near], dtype=np.float64)
            scheduled_rates = [float(rate_far), float(rate_near)]
            weights = [1.4 if rate_far < qos_target else 1.0, 1.4 if rate_near < qos_target else 1.0]
            pws = float(np.dot(np.asarray(weights), rates))

            sum_rate = float(np.sum(rates))
            fairness = self._jain_fairness(rates)
            sic_penalty = abs(sinr_far_at_near - sinr_far) / (1.0 + abs(sinr_far_at_near))

            eepsu = sum_rate / (len(scheduled_rates) * (tx_power_w + 0.2))
            qos_achieved = min(rate_far, rate_near)
            snr_achieved = max(sinr_far, sinr_near)
            utility = self.calc_noma_utility(fairness=fairness, sum_rate=sum_rate, sic_error_penalty=sic_penalty)

            noma_ratio = p_far

        else:  # JOINT
            h_far = g_far_direct + 0.35 * ris_gain_linear
            h_near = g_near_direct + 0.20 * ris_gain_linear

            sinr_far = (p_far * tx_power_w * h_far) / (p_near * tx_power_w * h_far + noise_w + 1e-18)
            sinr_far_at_near = (p_far * tx_power_w * h_near) / (p_near * tx_power_w * h_near + noise_w + 1e-18)
            sinr_near = (p_near * tx_power_w * h_near) / (sic_residual * p_far * tx_power_w * h_near + noise_w + 1e-18)

            rate_far = (bandwidth_hz * math.log2(1.0 + sinr_far)) / 1e6
            rate_near = (bandwidth_hz * math.log2(1.0 + sinr_near)) / 1e6
            sum_rate = float(rate_far + rate_near)
            scheduled_rates = [float(rate_far), float(rate_near)]
            rates_np = np.array([rate_far, rate_near], dtype=np.float64)
            fairness = self._jain_fairness(rates_np)
            weights = [1.5 if rate_far < qos_target else 1.0, 1.5 if rate_near < qos_target else 1.0]
            pws = float(np.dot(np.asarray(weights), rates_np))
            sic_penalty = abs(sinr_far_at_near - sinr_far) / (1.0 + abs(sinr_far_at_near))

            eepsu = sum_rate / (len(scheduled_rates) * (tx_power_w + 0.01 * ris_elements + 0.2))
            qos_achieved = min(rate_far, rate_near)
            snr_achieved = tx_power_w * (h_far + h_near) / (2.0 * noise_w + 1e-18)
            utility = self.calc_joint_utility(qos_achieved=qos_achieved, qos_target=qos_target, eepsu=eepsu)

            noma_ratio = p_far

        result = PhysicsResult(
            sum_rate=float(sum_rate),
            eepsu=float(eepsu),
            pws=float(pws),
            qos_achieved=float(qos_achieved),
            qos_target=float(qos_target),
            snr_achieved=float(snr_achieved),
            snr_target=float(snr_target),
            reflection_efficiency=float(reflection_eff),
            jain_fairness=float(fairness),
            sic_error_penalty=float(sic_penalty),
            domain_utility_score=float(utility),
            noma_u1_power_ratio=float(noma_ratio),
            ris_u3_snr=float(10.0 * math.log10(max(snr_achieved, 1e-12))),
            scheduled_user_rates_mbps=scheduled_rates,
        )
        out = result.__dict__
        out["executed_params"] = {
            "noma_power_split": p_far,
            "ris_phase_offset": phase_offset,
            "ris_reflection_amplitude": reflection_amplitude,
            "sic_residual": sic_residual,
            "executed_base_power_dbm": float(executed["executed_base_power_dbm"]),
            "action_types": list(executed.get("executed_action_types", [])),
        }
        return out
