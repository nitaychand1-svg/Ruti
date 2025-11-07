"""
Vector-DB inspired episodic memory for ACTS v6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Episode:
    state: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    metadata: Dict[str, Any]
    embedding: np.ndarray


class EpisodicMemory:
    """
    Lightweight episodic memory using FAISS when available, otherwise a numpy
    nearest-neighbour fallback.
    """

    def __init__(self, embedding_dim: int = 512, backend: str = "auto") -> None:
        self.embedding_dim = embedding_dim
        self.episodes: List[Episode] = []
        self.backend = backend

        self._faiss_index = None
        if backend in ("auto", "faiss"):
            self._faiss_index = self._try_init_faiss()
            if self._faiss_index is None and backend == "faiss":
                raise RuntimeError("FAISS backend requested but unavailable")

    def store_episode(self, state: Dict[str, Any], action: Dict[str, Any], outcome: Dict[str, Any], metadata: Dict[str, Any]):
        embedding = self._build_embedding(state, action, outcome, metadata)
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch {embedding.shape[0]} != {self.embedding_dim}")

        episode = Episode(state=state, action=action, outcome=outcome, metadata=metadata, embedding=embedding)
        self.episodes.append(episode)
        self._add_to_index(embedding.astype("float32"))

    def recall_similar(self, query_state: Dict[str, Any], top_k: int = 5) -> List[Episode]:
        if not self.episodes:
            return []
        query_embedding = self._build_embedding(query_state, {}, {}, {})
        distances, indices = self._search_index(query_embedding.reshape(1, -1).astype("float32"), top_k)
        return [self.episodes[idx] for idx in indices[0] if idx < len(self.episodes)]

    def crisis_playbook(self) -> Dict[str, Any]:
        crises = [episode for episode in self.episodes if episode.metadata.get("regime") == "crisis"]
        winners = [ep for ep in crises if ep.outcome.get("pnl", 0) > 0]
        return {
            "n_crises": len(crises),
            "n_successful": len(winners),
            "success_rate": len(winners) / len(crises) if crises else 0.0,
            "top_strategies": [ep.action for ep in winners[:5]],
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_embedding(self, state: Dict[str, Any], action: Dict[str, Any], outcome: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        seed = hash(frozenset(state.items())) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, self.embedding_dim).astype("float32")

    def _try_init_faiss(self):
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("EpisodicMemory using FAISS backend")
            return index
        except Exception as exc:
            if self.backend == "faiss":
                raise
            logger.warning("FAISS unavailable, falling back to numpy search: %s", exc)
            return None

    def _add_to_index(self, embedding: np.ndarray):
        if self._faiss_index is not None:
            self._faiss_index.add(embedding.reshape(1, -1))

    def _search_index(self, embedding: np.ndarray, top_k: int):
        if self._faiss_index is not None:
            distances, indices = self._faiss_index.search(embedding, top_k)
            return distances, indices

        # numpy brute-force fallback
        all_embeddings = np.stack([ep.embedding for ep in self.episodes])
        dists = np.sum((all_embeddings[None, :, :] - embedding[:, None, :]) ** 2, axis=-1)
        idx = np.argsort(dists, axis=1)[:, :top_k]
        sorted_dists = np.take_along_axis(dists, idx, axis=1)
        return sorted_dists, idx
