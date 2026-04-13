from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PhysicsResult:
    sum_rate: float
    eepsu: float
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

    def evaluate(self, scenario: Dict, params: Dict) -> Dict:
        category = scenario["category"]
        tx_power_w = self._dbm_to_watts(float(scenario.get("bs_power_dbm", 40.0)))
        noise_w = self._dbm_to_watts(float(scenario.get("noise_dbm", -94.0)))
        bandwidth_hz = float(scenario.get("bandwidth_hz", 20e6))
        pathloss_exp = float(scenario.get("pathloss_exp", 2.2))
        blockage = float(scenario.get("blockage_factor", 1.0))
        qos_target = float(scenario.get("qos_target", 8.0))
        snr_target = float(scenario.get("snr_target", 25.0))

        d_far = float(scenario.get("d_far_m", 220.0))
        d_near = float(scenario.get("d_near_m", 90.0))
        d_bs_ris = float(scenario.get("d_bs_ris_m", 55.0))
        d_ris_u3 = float(scenario.get("d_ris_u3_m", 95.0))
        ris_elements = float(scenario.get("ris_elements", 128.0))

        p_far = float(params.get("noma_power_split", 0.65))
        p_far = max(0.05, min(0.95, p_far))
        p_near = 1.0 - p_far

        phase_offset = float(params.get("ris_phase_offset", 0.0))
        reflection_amplitude = float(params.get("ris_reflection_amplitude", 0.9))
        reflection_amplitude = max(0.0, min(1.0, reflection_amplitude))
        sic_residual = float(params.get("sic_residual", 0.08))
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
            sum_rate = (bandwidth_hz * math.log2(1.0 + snr_achieved)) / 1e6
            eepsu = sum_rate / (tx_power_w + 0.01 * ris_elements)
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

            sum_rate = float(np.sum(rates))
            fairness = self._jain_fairness(rates)
            sic_penalty = abs(sinr_far_at_near - sinr_far) / (1.0 + abs(sinr_far_at_near))

            eepsu = sum_rate / (tx_power_w + 0.2)
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
            fairness = self._jain_fairness(np.array([rate_far, rate_near], dtype=np.float64))
            sic_penalty = abs(sinr_far_at_near - sinr_far) / (1.0 + abs(sinr_far_at_near))

            eepsu = sum_rate / (tx_power_w + 0.01 * ris_elements + 0.2)
            qos_achieved = min(rate_far, rate_near)
            snr_achieved = tx_power_w * (h_far + h_near) / (2.0 * noise_w + 1e-18)
            utility = self.calc_joint_utility(qos_achieved=qos_achieved, qos_target=qos_target, eepsu=eepsu)

            noma_ratio = p_far

        result = PhysicsResult(
            sum_rate=float(sum_rate),
            eepsu=float(eepsu),
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
        )
        return result.__dict__
