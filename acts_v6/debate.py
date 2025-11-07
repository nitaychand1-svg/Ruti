"""
Multi-agent debate subsystem for ACTS v6.

The debate layer orchestrates specialist agents to discuss strategy proposals,
critique one another, and converge on a consensus portfolio allocation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


ProposalGenerator = Callable[[str, np.ndarray, Dict[str, float], Dict[str, Any]], "StrategyProposal"]
CritiqueGenerator = Callable[[str, "StrategyProposal", Sequence["StrategyProposal"]], str]


@dataclass
class StrategyProposal:
    agent_id: str
    weights: np.ndarray
    expected_return: float
    sharpe: float
    rationale: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "StrategyProposal":
        weights = self.weights.copy()
        weights /= np.sum(np.abs(weights)) + 1e-8
        return dataclasses.replace(self, weights=weights)


@dataclass
class DebateRound:
    critiques: Dict[str, str]
    updated_proposals: List[StrategyProposal]


class MultiAgentDebateSystem:
    """
    Simple-yet-extensible multi-agent debate system.

    LLM calls are abstracted behind user-provided proposal/critique generators.
    Defaults rely on heuristics to remain functional without network access.
    """

    def __init__(
        self,
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        proposal_generator: Optional[ProposalGenerator] = None,
        critique_generator: Optional[CritiqueGenerator] = None,
    ) -> None:
        self.agent_configs = agent_configs or self._default_agent_configs()
        self._proposal_generator = proposal_generator or self._heuristic_proposal
        self._critique_generator = critique_generator or self._heuristic_critique
        logger.info("Multi-Agent Debate System initialized with %d agents", len(self.agent_configs))

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    async def orchestrate_debate(
        self,
        features: np.ndarray,
        regime_probs: Dict[str, float],
        constraints: Dict[str, Any],
        rounds: int = 3,
    ) -> Dict[str, Any]:
        proposals = await self._collect_proposals(features, regime_probs, constraints)
        history: List[DebateRound] = []

        for round_idx in range(rounds):
            critiques = await self._collect_critiques(proposals)
            updated = await self._revise_proposals(proposals, critiques)
            history.append(DebateRound(critiques=critiques, updated_proposals=updated))
            proposals = updated
            logger.debug("Finished debate round %d", round_idx + 1)

        consensus = self._compute_consensus(proposals, constraints)
        alignment = self._alignment_score(consensus, constraints)
        pareto = self._pareto_frontier(proposals)

        return {
            "consensus_strategy": consensus,
            "agent_proposals": [p.normalized() for p in proposals],
            "debate_history": history,
            "alignment_score": alignment,
            "pareto_frontier": pareto,
        }

    # ------------------------------------------------------------------ #
    # Internal steps
    # ------------------------------------------------------------------ #
    async def _collect_proposals(
        self,
        features: np.ndarray,
        regime_probs: Dict[str, float],
        constraints: Dict[str, Any],
    ) -> List[StrategyProposal]:
        tasks = [
            asyncio.to_thread(self._proposal_generator, agent_id, features, regime_probs, constraints)
            for agent_id in self.agent_configs
        ]
        proposals = await asyncio.gather(*tasks)
        return [proposal.normalized() for proposal in proposals]

    async def _collect_critiques(self, proposals: Sequence[StrategyProposal]) -> Dict[str, str]:
        async def _critique(agent_id: str, proposal: StrategyProposal) -> str:
            peer_proposals = [p for p in proposals if p.agent_id != agent_id]
            return await asyncio.to_thread(self._critique_generator, agent_id, proposal, peer_proposals)

        results = await asyncio.gather(*[_critique(p.agent_id, p) for p in proposals])
        return {proposal.agent_id: critique for proposal, critique in zip(proposals, results)}

    async def _revise_proposals(
        self,
        proposals: Sequence[StrategyProposal],
        critiques: Dict[str, str],
    ) -> List[StrategyProposal]:
        revised: List[StrategyProposal] = []
        for proposal in proposals:
            critique = critiques.get(proposal.agent_id, "")
            modifier = -0.05 if "risk" in critique.lower() else 0.03 if "opportunity" in critique.lower() else 0.0
            updated_confidence = np.clip(proposal.confidence + modifier, 0.5, 0.99)
            revised.append(
                dataclasses.replace(
                    proposal,
                    confidence=float(updated_confidence),
                    metadata={**proposal.metadata, "critique": critique},
                )
            )
        return revised

    # ------------------------------------------------------------------ #
    # Consensus logic
    # ------------------------------------------------------------------ #
    def _compute_consensus(self, proposals: Sequence[StrategyProposal], constraints: Dict[str, Any]) -> Dict[str, Any]:
        weights = np.zeros_like(proposals[0].weights)
        total_conf = sum(p.confidence for p in proposals) + 1e-8
        for proposal in proposals:
            weights += proposal.weights * (proposal.confidence / total_conf)

        weights = self._apply_constraints(weights, constraints)

        return {
            "weights": weights,
            "expected_return": float(np.mean([p.expected_return for p in proposals])),
            "sharpe": float(np.mean([p.sharpe for p in proposals])),
        }

    def _alignment_score(self, consensus: Dict[str, Any], constraints: Dict[str, Any]) -> float:
        target_risk = constraints.get("max_var", 0.02)
        realized = np.sum(np.square(consensus["weights"]))
        alignment = np.clip(1.0 - abs(realized - target_risk) / (target_risk + 1e-6), 0.0, 1.0)
        return float(alignment)

    def _pareto_frontier(self, proposals: Sequence[StrategyProposal]) -> List[StrategyProposal]:
        frontier = []
        for proposal in proposals:
            dominated = False
            for other in proposals:
                if other is proposal:
                    continue
                if other.expected_return >= proposal.expected_return and other.sharpe >= proposal.sharpe:
                    if other.expected_return > proposal.expected_return or other.sharpe > proposal.sharpe:
                        dominated = True
                        break
            if not dominated:
                frontier.append(proposal.normalized())
        return frontier

    # ------------------------------------------------------------------ #
    # Heuristics
    # ------------------------------------------------------------------ #
    def _default_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            "bull": {"persona": "Optimistic growth trader", "risk_tolerance": "high"},
            "bear": {"persona": "Defensive macro strategist", "risk_tolerance": "low"},
            "risk": {"persona": "Risk officer", "risk_tolerance": "minimal"},
            "ethical": {"persona": "ESG advocate", "risk_tolerance": "medium"},
            "innovation": {"persona": "Contrarian DeFi analyst", "risk_tolerance": "very_high"},
            "macro": {"persona": "Global macro economist", "risk_tolerance": "medium"},
        }

    def _heuristic_proposal(
        self,
        agent_id: str,
        features: np.ndarray,
        regime_probs: Dict[str, float],
        constraints: Dict[str, Any],
    ) -> StrategyProposal:
        rng = np.random.default_rng(abs(hash(agent_id)) % (2**32))
        n_assets = constraints.get("n_assets", len(features))
        weights = rng.random(n_assets)
        weights /= np.sum(weights) + 1e-8
        expected_return = float(0.05 + 0.10 * regime_probs.get("expansion", 0.3))
        sharpe = float(1.0 + rng.random() * 1.5)
        rationale = f"{agent_id} heuristic proposal under {regime_probs}"
        confidence = float(np.clip(0.7 + rng.normal(0, 0.05), 0.55, 0.95))
        return StrategyProposal(agent_id, weights, expected_return, sharpe, rationale, confidence)

    def _heuristic_critique(
        self,
        agent_id: str,
        proposal: StrategyProposal,
        peer_proposals: Sequence[StrategyProposal],
    ) -> str:
        avg_peer_conf = np.mean([peer.confidence for peer in peer_proposals]) if peer_proposals else proposal.confidence
        if proposal.confidence > avg_peer_conf:
            return "Opportunity: reinforce conviction but monitor tail risk."
        return "Risk flagged: rebalance exposure and tighten hedges."

    def _apply_constraints(self, weights: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        clipped = np.clip(weights, constraints.get("min_weight", -0.3), constraints.get("max_weight", 0.5))
        clipped /= np.sum(np.abs(clipped)) + 1e-8
        return clipped
