"""
Causal world model and intervention engine for ACTS v6.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Entities and world model construction
# --------------------------------------------------------------------------- #


@dataclass
class WorldEntity:
    id: str
    type: str
    attributes: Dict[str, Any]
    embedding: np.ndarray


class WorldModelBuilder:
    """
    Builds a lightweight heterogeneous world model graph.

    In production this would combine NLP-driven entity extraction, knowledge
    graphs, and econometric signals. Here we provide a deterministic skeleton
    to support downstream components.
    """

    def __init__(self) -> None:
        self.entities: Dict[str, WorldEntity] = {}
        self.causal_graph: Optional[SimpleCausalGraph] = None

    async def build_world_graph(
        self,
        market_data: np.ndarray,
        news_corpus: Sequence[str],
        knowledge_base: Optional[Dict[str, Any]] = None,
    ) -> "SimpleCausalGraph":
        base_entities = ["BTC", "SPY", "GLD", "USDT"]
        for asset in base_entities:
            self.entities[asset] = WorldEntity(
                id=asset,
                type="asset",
                attributes={"market_cap": float(np.random.uniform(1e9, 5e11)), "volatility": 0.02},
                embedding=np.random.randn(128),
            )

        extracted = await self._extract_entities(news_corpus)
        self.entities.update(extracted)

        graph = await self._discover_relations(self.entities, market_data)
        self.causal_graph = graph

        logger.info("World graph built with %d entities and %d edges", len(graph.nodes), len(graph.edges))
        return graph

    async def _extract_entities(self, news_corpus: Sequence[str]) -> Dict[str, WorldEntity]:
        # Placeholder for NLP pipeline
        unique_tokens = {token for doc in news_corpus for token in doc.split() if token.istitle()}
        entities: Dict[str, WorldEntity] = {}
        for token in list(unique_tokens)[:5]:
            entities[token] = WorldEntity(
                id=token,
                type="event" if token.endswith(("Summit", "Meeting")) else "actor",
                attributes={"sentiment": float(np.random.normal(0, 0.2))},
                embedding=np.random.randn(128),
            )
        return entities

    async def _discover_relations(self, entities: Dict[str, WorldEntity], market_data: np.ndarray) -> "SimpleCausalGraph":
        graph = SimpleCausalGraph()
        for entity_id in entities:
            graph.add_node(entity_id, prior_mean=0.0, prior_std=1.0)

        entity_ids = list(entities.keys())
        rng = np.random.default_rng(42)
        for _ in range(min(len(entity_ids) * 2, 20)):
            src, dst = rng.choice(entity_ids, 2, replace=False)
            weight = float(rng.normal(0, 1))
            graph.add_edge(src, dst, weight=weight)

        return graph


# --------------------------------------------------------------------------- #
# Sequential interventions
# --------------------------------------------------------------------------- #


@dataclass(order=True)
class TemporalIntervention:
    timestep: int
    variable: str = field(compare=False)
    value: float = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


class SequentialInterventionEngine:
    def __init__(self, causal_graph: "SimpleCausalGraph", n_mc_samples: int = 10_000) -> None:
        self.graph = causal_graph
        self.n_mc_samples = n_mc_samples
        self.state_history: List[Dict[str, Any]] = []

    def apply_intervention_chain(
        self,
        interventions: Sequence[TemporalIntervention],
        horizon: int = 30,
    ) -> Dict[str, Any]:
        interventions_sorted = sorted(interventions)
        current_graph = self.graph.copy()
        self.state_history.clear()

        logger.info("Running intervention chain across %d timesteps", horizon)

        for t in range(horizon):
            for intervention in [i for i in interventions_sorted if i.timestep == t]:
                current_graph = self._apply_do(current_graph, intervention)

            node_samples = self._propagate(current_graph)
            regime_probs = self._infer_regime(node_samples)
            self.state_history.append({"timestep": t, "node_values": node_samples, "regime_probs": regime_probs})

            current_graph = self._update_priors(current_graph, node_samples)

        attribution = self._causal_attribution(self.state_history, interventions_sorted)
        convergence = self._convergence_metrics(self.state_history)

        return {
            "final_state": self.state_history[-1] if self.state_history else None,
            "trajectory": self.state_history,
            "causal_attribution": attribution,
            "convergence_metrics": convergence,
        }

    def _apply_do(self, graph: "SimpleCausalGraph", intervention: TemporalIntervention) -> "SimpleCausalGraph":
        modified = graph.copy()
        if intervention.variable in modified.nodes:
            node = modified.nodes[intervention.variable]
            node["intervened"] = True
            node["fixed_value"] = intervention.value
            node["parents"] = []
            node["metadata"] = {**node.get("metadata", {}), **intervention.metadata}
            logger.debug("Applied do(%s=%s) at t=%d", intervention.variable, intervention.value, intervention.timestep)
        return modified

    def _propagate(self, graph: "SimpleCausalGraph") -> Dict[str, np.ndarray]:
        samples: Dict[str, np.ndarray] = {}
        for node_name in graph.topological_sort():
            node = graph.nodes[node_name]
            if node.get("intervened"):
                samples[node_name] = np.full(self.n_mc_samples, node.get("fixed_value", 0.0))
                continue

            parent_values = np.zeros(self.n_mc_samples)
            for parent in node.get("parents", []):
                weight = node["weights"].get(parent, 0.0)
                parent_values += weight * samples[parent]

            noise = np.random.normal(0, node.get("noise_std", 0.1), size=self.n_mc_samples)
            baseline = np.random.normal(node.get("prior_mean", 0.0), node.get("prior_std", 1.0), size=self.n_mc_samples)
            samples[node_name] = baseline + parent_values + noise
        return samples

    def _update_priors(self, graph: "SimpleCausalGraph", samples: Dict[str, np.ndarray]) -> "SimpleCausalGraph":
        updated = graph.copy()
        for name, node in updated.nodes.items():
            if name not in samples:
                continue
            mean = float(np.mean(samples[name]))
            std = float(np.std(samples[name]) + 1e-6)
            node["prior_mean"] = 0.9 * node.get("prior_mean", 0.0) + 0.1 * mean
            node["prior_std"] = max(0.05, 0.9 * node.get("prior_std", 1.0) + 0.1 * std)
        return updated

    def _infer_regime(self, samples: Dict[str, np.ndarray]) -> np.ndarray:
        volatility = samples.get("volatility")
        if volatility is None:
            return np.array([0.34, 0.33, 0.33])
        mean_vol = float(np.mean(volatility))
        if mean_vol < 0.01:
            return np.array([0.75, 0.20, 0.05])
        if mean_vol < 0.03:
            return np.array([0.20, 0.60, 0.20])
        return np.array([0.10, 0.25, 0.65])

    def _causal_attribution(
        self,
        trajectory: Sequence[Dict[str, Any]],
        interventions: Sequence[TemporalIntervention],
    ) -> Dict[str, Dict[str, Any]]:
        attribution: Dict[str, Dict[str, Any]] = {}
        for intervention in interventions:
            t = intervention.timestep
            if t >= len(trajectory):
                continue
            state_before = trajectory[max(0, t - 1)]["node_values"]
            state_after = trajectory[t]["node_values"]
            pnl_before = np.mean(state_before.get("pnl", np.zeros(self.n_mc_samples)))
            pnl_after = np.mean(state_after.get("pnl", np.zeros(self.n_mc_samples)))
            attribution[f"{intervention.variable}_t{t}"] = {
                "intervention": intervention,
                "pnl_effect": float(pnl_after - pnl_before),
            }
        return attribution

    def _convergence_metrics(self, trajectory: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(trajectory) < 5:
            return {"rhat": 1.0, "converged": True}
        pnl_chains = [state["node_values"].get("pnl", np.zeros(self.n_mc_samples)) for state in trajectory[-5:]]
        if len(pnl_chains) < 2:
            return {"rhat": 1.0, "converged": True}
        rhat = self._gelman_rubin(pnl_chains)
        return {"rhat": rhat, "converged": rhat < 1.05}

    def _gelman_rubin(self, chains: Sequence[np.ndarray]) -> float:
        chains = [np.asarray(chain) for chain in chains]
        n_chains = len(chains)
        n_samples = min(chain.shape[0] for chain in chains)
        truncated = np.asarray([chain[:n_samples] for chain in chains])
        chain_means = truncated.mean(axis=1)
        overall_mean = chain_means.mean()
        B = n_samples / (n_chains - 1) * np.sum((chain_means - overall_mean) ** 2)
        W = np.mean(np.var(truncated, axis=1, ddof=1))
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        return float(np.sqrt(var_hat / (W + 1e-8)))


# --------------------------------------------------------------------------- #
# Simple causal graph representation
# --------------------------------------------------------------------------- #


class SimpleCausalGraph:
    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[tuple] = []

    def add_node(self, name: str, parents: Optional[List[str]] = None, prior_mean: float = 0.0, prior_std: float = 1.0):
        self.nodes.setdefault(
            name,
            {
                "name": name,
                "parents": parents[:] if parents else [],
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "weights": {},
                "noise_std": 0.1,
                "intervened": False,
            },
        )

    def add_edge(self, from_node: str, to_node: str, weight: float = 1.0):
        if to_node not in self.nodes:
            self.add_node(to_node)
        if from_node not in self.nodes:
            self.add_node(from_node)

        node = self.nodes[to_node]
        node["weights"][from_node] = weight
        if from_node not in node["parents"]:
            node["parents"].append(from_node)
        self.edges.append((from_node, to_node, weight))

    def topological_sort(self) -> List[str]:
        visited = set()
        order: List[str] = []

        def dfs(node_name: str) -> None:
            if node_name in visited:
                return
            visited.add(node_name)
            for parent in self.nodes.get(node_name, {}).get("parents", []):
                dfs(parent)
            order.append(node_name)

        for node_name in self.nodes:
            dfs(node_name)
        return order

    def copy(self) -> "SimpleCausalGraph":
        new_graph = SimpleCausalGraph()
        new_graph.nodes = copy.deepcopy(self.nodes)
        new_graph.edges = copy.deepcopy(self.edges)
        return new_graph
