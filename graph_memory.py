"""Graph-RAG structural memory for continual 6G orchestration.

This module stores scenario-strategy knowledge in a directed ontology graph and
supports embedding-based retrieval plus snapshot persistence.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


@dataclass
class RetrievalResult:
    """Container for a strategy retrieval result.

    Attributes:
        similarity: Cosine similarity score for the matched scenario.
        scenario_node: Scenario node identifier from the graph.
        strategy_node: Strategy node identifier from the graph.
        strategy_data: Payload from the strategy node.
    """

    similarity: float
    scenario_node: str
    strategy_node: str
    strategy_data: Dict[str, Any]


class GraphRAG:
    """Graph-backed semantic memory for continual learning.

    The ontology contains immutable technology root nodes and dynamic scenario
    and strategy nodes connected with typed edges.
    """

    TECH_RIS = "TECH_RIS"
    TECH_NOMA = "TECH_NOMA"

    def __init__(self, embedding_device: str = "cuda") -> None:
        """Initialize the graph and embedding model.

        Args:
            embedding_device: Preferred device for sentence embedding model.
        """
        self.graph = nx.DiGraph()
        self._init_tech_roots()
        self.embedding_device = embedding_device
        self.embedding_model = self._init_embedding_model(embedding_device)

    def _init_tech_roots(self) -> None:
        """Create immutable technology root nodes."""
        self.graph.add_node(self.TECH_RIS, node_type="tech", description="Reconfigurable Intelligent Surface")
        self.graph.add_node(self.TECH_NOMA, node_type="tech", description="Non-Orthogonal Multiple Access")

    def _init_embedding_model(self, device: str):
        """Initialize sentence encoder with graceful fallback.

        Args:
            device: Preferred accelerator string.

        Returns:
            Sentence transformer model if available; otherwise None.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            print("[INFO - GraphRAG] Loaded sentence embedding model on device:", device)
            return model
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[WARN - GraphRAG] Embedding model unavailable, using hash embedding fallback. Reason: {exc}")
            return None

    def _encode(self, text: str) -> np.ndarray:
        """Encode text into a numeric embedding vector.

        Args:
            text: Input text.

        Returns:
            Embedding as a 1-D numpy array.
        """
        if self.embedding_model is not None:
            vec = self.embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
            return vec.astype(np.float32)

        # Deterministic fallback embedding for environments without transformers.
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(0.0, 1.0, size=384).astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def insert_knowledge(
        self,
        scenario_id: str,
        scenario_desc: str,
        techs_used: List[str],
        strategy_params: Dict[str, Any],
        rationale: str,
    ) -> None:
        """Insert a scenario and linked strategy into the graph.

        Args:
            scenario_id: Unique scenario identifier.
            scenario_desc: Human-readable scenario description.
            techs_used: Active technologies, e.g. ["RIS", "NOMA"].
            strategy_params: Parameters used to solve the scenario.
            rationale: Explainable reason for strategy effectiveness.
        """
        scenario_node = f"SCENARIO::{scenario_id}"
        strategy_node = f"STRATEGY::{scenario_id}"
        embedding = self._encode(scenario_desc)

        self.graph.add_node(
            scenario_node,
            node_type="scenario",
            scenario_id=scenario_id,
            description=scenario_desc,
            embedding=embedding,
        )
        self.graph.add_node(
            strategy_node,
            node_type="strategy",
            scenario_id=scenario_id,
            params=strategy_params,
            rationale=rationale,
        )
        self.graph.add_edge(scenario_node, strategy_node, edge_type="RESOLVED_BY")

        tech_map = {"RIS": self.TECH_RIS, "NOMA": self.TECH_NOMA}
        for tech in techs_used:
            mapped = tech_map.get(tech.upper())
            if mapped is not None:
                self.graph.add_edge(scenario_node, mapped, edge_type="CONTAINS_TECH")

        print(f"[INFO - GraphRAG] Inserted knowledge for {scenario_id}.")

    def retrieve_similar_strategy(self, current_scenario_desc: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """Retrieve the strategy attached to the most similar scenario.

        Args:
            current_scenario_desc: Scenario query string.
            threshold: Minimum cosine similarity to accept a match.

        Returns:
            Strategy payload dict if a match is found, else None.
        """
        query_embedding = self._encode(current_scenario_desc)
        best: Optional[RetrievalResult] = None

        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "scenario":
                continue

            similarity = self._cosine_similarity(query_embedding, data["embedding"])
            if similarity < threshold:
                continue

            for neighbor in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor, default={})
                if edge_data.get("edge_type") != "RESOLVED_BY":
                    continue
                strategy_data = self.graph.nodes[neighbor]
                candidate = RetrievalResult(
                    similarity=similarity,
                    scenario_node=node_id,
                    strategy_node=neighbor,
                    strategy_data={
                        "scenario_id": strategy_data.get("scenario_id"),
                        "params": strategy_data.get("params", {}),
                        "rationale": strategy_data.get("rationale", ""),
                        "matched_similarity": similarity,
                        "matched_scenario": data.get("description", ""),
                    },
                )
                if best is None or candidate.similarity > best.similarity:
                    best = candidate

        if best is None:
            print("[INFO - GraphRAG] No similar strategy found.")
            return None

        print(
            "[INFO - GraphRAG] Retrieved strategy with similarity "
            f"{best.similarity:.3f} from {best.scenario_node}."
        )
        return best.strategy_data

    def save_snapshot(self, filepath: str) -> None:
        """Persist the full graph state with pickle.

        Args:
            filepath: Output snapshot path.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"[INFO - GraphRAG] Snapshot saved to disk: {filepath}")

    def load_snapshot(self, filepath: str) -> None:
        """Load a graph snapshot from pickle.

        Args:
            filepath: Snapshot path.

        Raises:
            FileNotFoundError: If the snapshot does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot not found: {filepath}")

        with open(filepath, "rb") as f:
            loaded_graph = pickle.load(f)

        self.graph = loaded_graph

        # Ensure technology roots exist for forward compatibility.
        if self.TECH_RIS not in self.graph:
            self.graph.add_node(self.TECH_RIS, node_type="tech", description="Reconfigurable Intelligent Surface")
        if self.TECH_NOMA not in self.graph:
            self.graph.add_node(self.TECH_NOMA, node_type="tech", description="Non-Orthogonal Multiple Access")

        print(f"[INFO - GraphRAG] Snapshot loaded from disk: {filepath}")
