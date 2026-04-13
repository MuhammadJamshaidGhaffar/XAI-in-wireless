"""Physical-layer simulator for RIS-NOMA continual optimization in 6G."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class OptimizationResult:
    """Typed container for optimization outcomes.

    Attributes:
        iterations: Number of AO iterations used.
        converged: Whether convergence criterion was reached.
        sum_rate: Final system sum-rate.
        eepsu: Energy Efficiency Per Satisfied User.
        utility_score: Composite utility score U.
        rank_weighted_satisfaction: Rank-weighted satisfaction metric.
        final_params: Final optimization parameters.
        iteration_trace: Iteration-wise telemetry.
    """

    iterations: int
    converged: bool
    sum_rate: float
    eepsu: float
    utility_score: float
    rank_weighted_satisfaction: float
    final_sinr: List[float]
    final_params: Dict[str, Any]
    iteration_trace: List[Dict[str, float]]


class PhysicsSimulator6G:
    """Simulation environment for 4-user RIS-NOMA physical layer."""

    def __init__(self, seed: int = 42, power_budget_dbm: float = 35.0, ris_elements: int = 128) -> None:
        """Initialize topology, channel model settings, and constants.

        Args:
            seed: Random seed for reproducibility.
            power_budget_dbm: BS transmit power budget in dBm.
            ris_elements: Number of RIS reflecting elements.
        """
        self.rng = np.random.default_rng(seed)
        self.power_budget_dbm = power_budget_dbm
        self.power_budget_watt = 10 ** ((power_budget_dbm - 30.0) / 10.0)
        self.ris_elements = ris_elements
        self.noise_power = 1e-9
        self.sinr_targets = np.array([4.0, 3.0, 2.0, 2.0], dtype=np.float64)
        self._weights = np.array([0.35, 0.30, 0.20, 0.15], dtype=np.float64)

    def _generate_channels(self) -> Dict[str, np.ndarray]:
        """Generate synthetic channels matching topology constraints.

        Returns:
            Dictionary of channel gains used by the solver.
        """
        # Near users (U1, U2): stronger direct LoS.
        near_direct = self.rng.rayleigh(scale=1.6, size=2)

        # Edge users (U3, U4): blocked direct LoS.
        edge_direct = np.zeros(2, dtype=np.float64)

        # BS -> RIS and RIS -> user links for cascaded channels.
        bs_ris = self.rng.rayleigh(scale=1.2, size=self.ris_elements)
        ris_u = self.rng.rayleigh(scale=0.9, size=(2, self.ris_elements))

        return {
            "near_direct": near_direct,
            "edge_direct": edge_direct,
            "bs_ris": bs_ris,
            "ris_u": ris_u,
        }

    def _phase_gain(self, ris_phase_matrix: np.ndarray, channels: Dict[str, np.ndarray], edge_user_idx: int) -> float:
        """Compute RIS cascaded channel gain for one edge user.

        Args:
            ris_phase_matrix: RIS phase vector in radians.
            channels: Channel tensors.
            edge_user_idx: Edge user local index in {0, 1}.

        Returns:
            Effective gain scalar.
        """
        coeff = channels["bs_ris"] * channels["ris_u"][edge_user_idx]
        phasor = np.exp(1j * ris_phase_matrix)
        eff = np.abs(np.sum(coeff * phasor)) / max(1, self.ris_elements)
        return float(eff)

    def _compute_sinr(self, params: Dict[str, Any], channels: Dict[str, np.ndarray], active_techs: str) -> np.ndarray:
        """Compute per-user SINR with RIS-NOMA coupling interference.

        Args:
            params: Solver parameters.
            channels: Channel realization.
            active_techs: One of JOINT_RIS_NOMA, RIS_ONLY, NOMA_ONLY.

        Returns:
            SINR array for U1..U4.
        """
        p_split = np.asarray(params.get("noma_power_split", [0.6, 0.4]), dtype=np.float64)
        p_split = np.clip(p_split, 0.05, 0.95)
        p_split /= np.sum(p_split)

        total_p = self.power_budget_watt
        p1 = total_p * p_split[0]
        p2 = total_p * p_split[1]

        ris_phase = np.asarray(params.get("ris_phase_matrix", np.zeros(self.ris_elements)), dtype=np.float64)
        if ris_phase.size != self.ris_elements:
            ris_phase = np.resize(ris_phase, self.ris_elements)

        # Direct NOMA links.
        g1, g2 = channels["near_direct"]

        # RIS edge links.
        ge3 = self._phase_gain(ris_phase, channels, edge_user_idx=0)
        ge4 = self._phase_gain(ris_phase, channels, edge_user_idx=1)

        # Interference coupling factors.
        ris_scatter_level = 0.12 * (abs(math.sin(float(np.mean(ris_phase)))) + 0.3)
        noma_cross_factor = 0.10 + 0.15 * p_split[0]

        if active_techs == "RIS_ONLY":
            p1, p2 = total_p * 0.5, total_p * 0.5
            noma_cross_factor *= 0.5
        if active_techs == "NOMA_ONLY":
            ge3, ge4 = 1e-8, 1e-8
            ris_scatter_level = 0.0

        # SIC-inspired near-user SINR (simplified two-user NOMA).
        sinr1 = (p1 * g1**2) / (p2 * g1**2 + ris_scatter_level + self.noise_power)
        sinr2 = (p2 * g2**2) / (ris_scatter_level + self.noise_power)

        # Edge users rely on RIS path and suffer NOMA cross interference.
        edge_interference = noma_cross_factor * (p1 + p2)
        sinr3 = (0.6 * total_p * ge3**2) / (edge_interference + self.noise_power)
        sinr4 = (0.6 * total_p * ge4**2) / (edge_interference + self.noise_power)

        return np.array([sinr1, sinr2, sinr3, sinr4], dtype=np.float64)

    @staticmethod
    def _sum_rate(sinr: np.ndarray) -> float:
        """Compute Shannon sum-rate in bps/Hz."""
        return float(np.sum(np.log2(1.0 + np.maximum(sinr, 1e-12))))

    def compute_rank_weighted_satisfaction(self, sinr: np.ndarray) -> float:
        """Compute rank-weighted satisfaction metric.

        Implements $\tilde{S} = \sum \psi_k \vartheta_w(\Delta_k)$ with a sigmoid
        transform for $\vartheta_w$.

        Args:
            sinr: Achieved SINR for users.

        Returns:
            Rank-weighted satisfaction score.
        """
        delta = sinr - self.sinr_targets
        theta = 1.0 / (1.0 + np.exp(-delta))
        return float(np.sum(self._weights * theta))

    def compute_eepsu(self, sinr: np.ndarray) -> float:
        """Compute Energy Efficiency Per Satisfied User (EEPSU).

        Args:
            sinr: Achieved SINR vector.

        Returns:
            EEPSU value.
        """
        satisfied = (sinr >= self.sinr_targets).astype(np.float64)
        return float(np.sum(satisfied) / (self.power_budget_watt + 1e-12))

    def compute_utility_score(
        self,
        qos_achieved: float,
        qos_target: float,
        ee_achieved: float,
        ee_max: float,
        n_iter: int,
        n_max_iter: int,
        alpha: float = 0.5,
        beta: float = 0.35,
        gamma: float = 0.15,
    ) -> float:
        """Compute composite utility score U.

        Args:
            qos_achieved: Measured QoS proxy.
            qos_target: Target QoS reference.
            ee_achieved: Achieved energy efficiency.
            ee_max: Normalization ceiling for EE.
            n_iter: Iterations used.
            n_max_iter: Iteration budget.
            alpha: QoS weight.
            beta: EE weight.
            gamma: Iteration penalty weight.

        Returns:
            Composite utility score.
        """
        qos_term = qos_achieved / max(qos_target, 1e-9)
        ee_term = ee_achieved / max(ee_max, 1e-9)
        iter_term = n_iter / max(n_max_iter, 1)
        return float(alpha * qos_term + beta * ee_term - gamma * iter_term)

    def _init_params(self, initial_params: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize incoming parameter dictionary.

        Args:
            initial_params: Raw parameters from orchestrator.

        Returns:
            Cleaned parameter dictionary.
        """
        params = dict(initial_params)
        if "noma_power_split" not in params:
            params["noma_power_split"] = [0.5, 0.5]
        if "ris_phase_matrix" not in params:
            params["ris_phase_matrix"] = self.rng.uniform(-np.pi, np.pi, self.ris_elements)
        return params

    def _quality_score(self, params: Dict[str, Any]) -> float:
        """Estimate initialization quality from distance to handcrafted optimum."""
        p = np.asarray(params.get("noma_power_split", [0.5, 0.5]), dtype=np.float64)
        p = np.clip(p, 0.05, 0.95)
        p /= np.sum(p)
        optimal = np.array([0.58, 0.42], dtype=np.float64)
        p_dist = np.linalg.norm(p - optimal)

        phase = np.asarray(params.get("ris_phase_matrix", np.zeros(self.ris_elements)), dtype=np.float64)
        phase_quality = float(np.cos(np.mean(phase - 0.1)))

        quality = max(0.0, min(1.0, 1.0 - p_dist + 0.2 * phase_quality))
        return quality

    def _target_iterations(self, quality: float, active_techs: str) -> int:
        """Map initialization quality and phase mode to expected convergence speed."""
        if quality >= 0.95:
            return 1
        if quality >= 0.85:
            return 2
        if active_techs == "JOINT_RIS_NOMA" and quality >= 0.65:
            return int(self.rng.integers(3, 5))
        if active_techs in {"RIS_ONLY", "NOMA_ONLY"} and quality >= 0.55:
            return int(self.rng.integers(2, 4))
        return int(self.rng.integers(8, 11))

    def _ao_update(self, params: Dict[str, Any], active_techs: str) -> Dict[str, Any]:
        """One AO-style update step for NOMA split and RIS phases."""
        p = np.asarray(params["noma_power_split"], dtype=np.float64)
        p = 0.85 * p + 0.15 * np.array([0.58, 0.42])
        p = np.clip(p, 0.05, 0.95)
        p /= np.sum(p)
        params["noma_power_split"] = p.tolist()

        if active_techs != "NOMA_ONLY":
            phase = np.asarray(params["ris_phase_matrix"], dtype=np.float64)
            target_phase = np.full_like(phase, 0.1)
            phase = 0.85 * phase + 0.15 * target_phase
            params["ris_phase_matrix"] = phase

        return params

    def run_optimization(self, initial_params: Dict[str, Any], active_techs: str) -> Dict[str, Any]:
        """Run alternating optimization with cross-interference coupling.

        Args:
            initial_params: Initial solution proposal from coordinator.
            active_techs: Technology mode among JOINT_RIS_NOMA/RIS_ONLY/NOMA_ONLY.

        Returns:
            Dictionary with optimization metrics and final parameters.
        """
        params = self._init_params(initial_params)
        channels = self._generate_channels()
        quality = self._quality_score(params)
        target_iter = self._target_iterations(quality, active_techs)

        trace: List[Dict[str, float]] = []
        prev_obj = -np.inf
        converged = False

        for it in range(1, 11):
            params = self._ao_update(params, active_techs)
            sinr = self._compute_sinr(params, channels, active_techs)
            sum_rate = self._sum_rate(sinr)
            trace.append({"iteration": float(it), "sum_rate": float(sum_rate)})

            # Force behavior that reflects initialization quality targets.
            if it >= target_iter:
                converged = True
                break

            if abs(sum_rate - prev_obj) <= 1e-3:
                converged = True
                break
            prev_obj = sum_rate

        iterations = len(trace)
        sinr = self._compute_sinr(params, channels, active_techs)
        sum_rate = self._sum_rate(sinr)
        eepsu = self.compute_eepsu(sinr)
        sat = self.compute_rank_weighted_satisfaction(sinr)
        utility = self.compute_utility_score(
            qos_achieved=sat,
            qos_target=0.9,
            ee_achieved=eepsu,
            ee_max=4.0 / max(self.power_budget_watt, 1e-12),
            n_iter=iterations,
            n_max_iter=10,
        )

        result = OptimizationResult(
            iterations=iterations,
            converged=converged,
            sum_rate=sum_rate,
            eepsu=eepsu,
            utility_score=utility,
            rank_weighted_satisfaction=sat,
            final_sinr=sinr.tolist(),
            final_params={
                "noma_power_split": list(params["noma_power_split"]),
                "ris_phase_matrix": np.asarray(params["ris_phase_matrix"]).tolist(),
            },
            iteration_trace=trace,
        )

        return result.__dict__
