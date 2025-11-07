"""
High-level orchestration for ACTS v6.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .adaptive_sampling import AdaptiveMCSampler
from .debate import MultiAgentDebateSystem
from .federated import FederatedTrainingCoordinator
from .interface import HumanAIInterface
from .marl import HierarchicalMARLSwarm
from .memory import EpisodicMemory
from .perception import ELBOOptimizer, MultiModalFusionEngine, BayesianRegimePredictor
from .risk import ExistentialRiskSimulator, ImportanceSampler
from .self_evolution import PerformanceSnapshot, SelfEvolutionOracle
from .world_model import SequentialInterventionEngine, TemporalIntervention, WorldModelBuilder

logger = logging.getLogger(__name__)


class ACTSv6Complete:
    def __init__(
        self,
        input_dim: int,
        n_assets: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_assets = n_assets

        logger.info("Initializing ACTS v6 on %s", self.device)

        # Layer 1
        self.multi_modal_fusion = MultiModalFusionEngine(device=self.device)
        self.regime_predictor = BayesianRegimePredictor(input_dim=4096, hidden_dim=128, n_regimes=3).to(self.device)
        self.regime_predictor.elbo_optimizer = ELBOOptimizer(self.regime_predictor, lr=1e-4)

        # Layer 2
        self.multi_agent_debate = MultiAgentDebateSystem()

        # Layer 3
        self.execution_swarm = HierarchicalMARLSwarm(n_agents=5)

        # Core causal engine
        self.world_model_builder = WorldModelBuilder()
        self.intervention_engine: Optional[SequentialInterventionEngine] = None
        self.adaptive_sampler = AdaptiveMCSampler()

        # Memory
        self.episodic_memory = EpisodicMemory()

        # Risk
        target_regime = np.array([0.6, 0.3, 0.1])
        proposal_regime = np.array([0.3, 0.4, 0.3])
        self.importance_sampler = ImportanceSampler(target_regime, proposal_regime)
        self.existential_risk = ExistentialRiskSimulator()

        # Self-improvement
        self.self_evolution = SelfEvolutionOracle()

        # Federated learning
        self.federated_coordinator = FederatedTrainingCoordinator()

        # Human interface
        self.human_interface = HumanAIInterface()

    # ------------------------------------------------------------------ #
    # Perception + Regime classification
    # ------------------------------------------------------------------ #
    async def perceive_environment(
        self,
        market_data: np.ndarray,
        news_articles: Sequence[str],
        social_posts: Sequence[str],
        geo_images: Optional[Sequence[Any]] = None,
        audio_speeches: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        perception = await self.multi_modal_fusion.perceive_world(
            market_data=market_data,
            news_articles=news_articles,
            social_posts=social_posts,
            geo_images=geo_images,
            audio_speeches=audio_speeches,
        )

        features_tensor = torch.from_numpy(perception.holistic_features).float().to(self.device)
        probs, epistemic, entropy = self.regime_predictor.predict_proba(features_tensor.unsqueeze(0))
        regime_probs = probs.squeeze(0).detach().cpu().numpy()

        return {
            "perception": perception,
            "regime_probs": {
                "expansion": float(regime_probs[0]),
                "neutral": float(regime_probs[1]),
                "contraction": float(regime_probs[2]),
            },
            "uncertainty": {
                "epistemic": float(epistemic.mean().item()),
                "entropy": float(entropy.mean().item()),
            },
        }

    # ------------------------------------------------------------------ #
    # Strategy debate + execution
    # ------------------------------------------------------------------ #
    async def formulate_strategy(
        self,
        features: np.ndarray,
        regime_probs: Dict[str, float],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        constraints = constraints or {"n_assets": self.n_assets, "max_var": 0.02}
        debate_outcome = await self.multi_agent_debate.orchestrate_debate(features, regime_probs, constraints)
        return debate_outcome

    async def execute_strategy(
        self,
        consensus_strategy: Dict[str, Any],
        market_state: Dict[str, Any],
        urgency: str = "normal",
    ) -> Dict[str, Any]:
        execution_report = await self.execution_swarm.swarm_execute(consensus_strategy, market_state, urgency)
        return execution_report

    # ------------------------------------------------------------------ #
    # Causal modelling
    # ------------------------------------------------------------------ #
    async def rebuild_world_model(self, market_data: np.ndarray, news_corpus: Sequence[str]) -> None:
        graph = await self.world_model_builder.build_world_graph(market_data, news_corpus, knowledge_base=None)
        self.intervention_engine = SequentialInterventionEngine(graph, n_mc_samples=self.adaptive_sampler.min_samples)

    def run_interventions(self, interventions: Sequence[TemporalIntervention], entropy: float) -> Dict[str, Any]:
        if self.intervention_engine is None:
            raise RuntimeError("World model not built yet")
        n_samples = self.adaptive_sampler.determine_n_samples(entropy)
        self.intervention_engine.n_mc_samples = n_samples
        return self.intervention_engine.apply_intervention_chain(interventions)

    # ------------------------------------------------------------------ #
    # Memory + Human interface
    # ------------------------------------------------------------------ #
    def log_episode(self, state: Dict[str, Any], action: Dict[str, Any], outcome: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        self.episodic_memory.store_episode(state, action, outcome, metadata)

    async def explain_to_human(self, decision: Dict[str, Any], user_query: str) -> str:
        return await self.human_interface.explain_decision(decision, user_query)

    # ------------------------------------------------------------------ #
    # Risk + Self evolution
    # ------------------------------------------------------------------ #
    def run_risk_scenarios(self, portfolio: Dict[str, float]) -> Dict[str, Any]:
        results = {}
        for scenario_name in self.existential_risk.scenarios:
            results[scenario_name] = self.existential_risk.simulate(scenario_name, portfolio).__dict__
        return results

    async def monitor_performance(self, snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        self.self_evolution.record_performance(snapshot)
        return await self.self_evolution.monitor_and_improve()

    # ------------------------------------------------------------------ #
    # Federated learning
    # ------------------------------------------------------------------ #
    async def federated_round(self) -> Dict[str, Any]:
        return await self.federated_coordinator.training_round()

    # ------------------------------------------------------------------ #
    # End-to-end cycle
    # ------------------------------------------------------------------ #
    async def full_cycle(
        self,
        market_data: np.ndarray,
        news: Sequence[str],
        posts: Sequence[str],
        market_state: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        perception_bundle = await self.perceive_environment(market_data, news, posts)
        await self.rebuild_world_model(market_data, news)

        debate_bundle = await self.formulate_strategy(
            features=perception_bundle["perception"].holistic_features,
            regime_probs=perception_bundle["regime_probs"],
            constraints=constraints,
        )

        execution_report = await self.execute_strategy(debate_bundle["consensus_strategy"], market_state)

        decision = {
            "regime": max(perception_bundle["regime_probs"], key=perception_bundle["regime_probs"].get),
            "expected_return": debate_bundle["consensus_strategy"]["expected_return"],
            "rationale": "Consensus across multi-agent debate.",
        }

        self.log_episode(
            state={"market_data": market_data.tolist()},
            action=debate_bundle["consensus_strategy"],
            outcome=execution_report["aggregate_metrics"],
            metadata={"regime": decision["regime"]},
        )

        return {
            "perception": perception_bundle,
            "debate": debate_bundle,
            "execution": execution_report,
            "decision_summary": decision,
        }
