import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class ConceptDatabase:
    """Concept-RAG memory with on-disk persistence and semantic retrieval."""

    def __init__(self, db_path: str, embedding_model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self.db_path = db_path
        self.memory: List[Dict[str, Any]] = []
        device = "cuda" if self._cuda_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model_name, device=device)
        self._load_from_disk()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _load_from_disk(self) -> None:
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, list):
                self.memory = loaded

    def _save_to_disk(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.memory, f)

    def merge_with(self, other_database: "ConceptDatabase") -> None:
        for concept in other_database.memory:
            self.memory.append(dict(concept))
        self._save_to_disk()

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode(text, convert_to_numpy=True)
        return vec.astype(np.float32)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def learn_concept(
        self,
        condition: str,
        rule: str,
        concept_score: float,
        domain: str = "joint",
        policy: str = "Learned Policy",
    ) -> None:
        composite = f"Condition: {condition}\nRule: {rule}"
        new_embedding = self._embed(composite)

        if not self.memory:
            self.memory.append(
                {
                    "condition": condition,
                    "rule": rule,
                    "concept_score": float(concept_score),
                    "domain": domain,
                    "policy": policy,
                    "embedding": new_embedding,
                }
            )
            self._save_to_disk()
            return

        sims: List[float] = []
        for concept in self.memory:
            emb = concept.get("embedding")
            if emb is None:
                emb = self._embed(f"Condition: {concept.get('condition', '')}\nRule: {concept.get('rule', '')}")
                concept["embedding"] = emb
            sims.append(self._cosine_similarity(new_embedding, np.asarray(emb, dtype=np.float32)))

        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        best_existing_score = float(
            self.memory[best_idx].get("concept_score", self.memory[best_idx].get("utility_score", -1e9))
        )

        if best_sim > 0.85:
            if float(concept_score) > best_existing_score:
                self.memory[best_idx].update(
                    {
                        "condition": condition,
                        "rule": rule,
                        "concept_score": float(concept_score),
                        "domain": domain,
                        "policy": policy,
                        "embedding": new_embedding,
                    }
                )
        else:
            self.memory.append(
                {
                    "condition": condition,
                    "rule": rule,
                    "concept_score": float(concept_score),
                    "domain": domain,
                    "policy": policy,
                    "embedding": new_embedding,
                }
            )

        self._save_to_disk()

    def retrieve_concepts(
        self,
        query: str,
        top_k: int = 5,
        domain_filters: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.memory:
            return []
        query_emb = self._embed(query)
        scored: List[Dict[str, Any]] = []
        for item in self.memory:
            if domain_filters and item.get("domain") not in domain_filters:
                continue
            emb = item.get("embedding")
            if emb is None:
                emb = self._embed(f"Condition: {item.get('condition', '')}\nRule: {item.get('rule', '')}")
                item["embedding"] = emb
            score = self._cosine_similarity(query_emb, np.asarray(emb, dtype=np.float32))
            item_copy = {k: v for k, v in item.items() if k != "embedding"}
            item_copy["similarity"] = score
            scored.append(item_copy)
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _run_llm_text(llm_pipeline: Any, prompt: str, max_new_tokens: int = 1500) -> str:
        if llm_pipeline is None:
            return ""
        output = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
                if isinstance(text, str):
                    return text[len(prompt) :] if text.startswith(prompt) else text
        return ""

    def export_to_markdown(self, output_path: str, llm_pipeline: Any = None) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        records = []
        for i, item in enumerate(self.memory, start=1):
            records.append(
                {
                    "id": i,
                    "policy": item.get("policy", "Learned Policy"),
                    "domain": item.get("domain", "joint"),
                    "condition": item.get("condition", ""),
                    "rule": item.get("rule", ""),
                    "concept_score": float(item.get("concept_score", item.get("utility_score", 0.0))),
                }
            )

        formatter_prompt = (
            "You are a formatting assistant for O-RAN concept rulebooks.\n"
            "CRITICAL: NEVER alter scientific wording.\n"
            "CRITICAL: ONLY add Markdown structure with headers for Policy, Condition, and Rule.\n"
            "Include domain and concept score fields, but do not paraphrase concepts.\n\n"
            f"Raw Concepts JSON:\n{json.dumps(records, indent=2)}\n\n"
            "Return Markdown only."
        )

        rendered = self._run_llm_text(llm_pipeline, formatter_prompt)
        if not rendered.strip():
            lines = ["# O-RAN Concept Rulebook", ""]
            for concept in records:
                lines.extend(
                    [
                        f"## Concept {concept['id']}",
                        f"**Policy:** {concept['policy']}",
                        f"**Domain:** {concept['domain']}",
                        f"**Concept Score:** {concept['concept_score']:.6f}",
                        "",
                        "### Condition",
                        concept["condition"],
                        "",
                        "### Rule",
                        concept["rule"],
                        "",
                    ]
                )
            rendered = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered.strip() + "\n")
