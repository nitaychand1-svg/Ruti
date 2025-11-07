"""
Hierarchical multi-agent RL execution swarm for ACTS v6.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionTask:
    asset: str
    size: float
    urgency: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    asset: str
    requested: float
    filled: float
    avg_price: float
    slippage: float
    latency_ms: float
    agent: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalMARLSwarm:
    """
    Simplified execution swarm providing hooks for MARL integration.

    The class keeps a replay buffer of recent tasks to support continual
    learning. Actual RL policies can be plugged in by overriding `_execute_task`.
    """

    def __init__(self, n_agents: int = 5, buffer_size: int = 50_000) -> None:
        self.agent_configs = self._default_agents()[:n_agents]
        self.shared_buffer: deque = deque(maxlen=buffer_size)
        self.latency_budget_ms = 500.0
        logger.info("Hierarchical MARL Swarm initialized with %d agents", len(self.agent_configs))

    async def swarm_execute(
        self,
        strategy: Dict[str, Any],
        market_state: Dict[str, Any],
        urgency: str = "normal",
    ) -> Dict[str, Any]:
        tasks = self._decompose_order(strategy, urgency)
        execution_results = await asyncio.gather(
            *[self._execute_with_agent(task, market_state) for task in tasks],
            return_exceptions=False,
        )
        for task, result in zip(tasks, execution_results):
            self.shared_buffer.append((task, result))

        aggregate = self._aggregate_results(execution_results)
        tca = self._transaction_cost_analysis(strategy, execution_results)

        return {
            "execution_results": execution_results,
            "aggregate_metrics": aggregate,
            "tca_report": tca,
        }

    # ------------------------------------------------------------------ #
    # Task pipeline
    # ------------------------------------------------------------------ #
    def _decompose_order(self, strategy: Dict[str, Any], urgency: str) -> List[ExecutionTask]:
        weights = strategy.get("weights", np.array([]))
        total_capital = strategy.get("notional", 1_000_000)
        tickers = strategy.get("tickers") or [f"Asset_{i}" for i in range(len(weights))]

        tasks: List[ExecutionTask] = []
        for weight, ticker in zip(weights, tickers):
            notional = float(weight) * total_capital
            if abs(notional) < 1e-6:
                continue
            tasks.append(
                ExecutionTask(
                    asset=ticker,
                    size=notional,
                    urgency=urgency,
                    metadata={"target_weight": float(weight)},
                )
            )
        return tasks

    async def _execute_with_agent(self, task: ExecutionTask, market_state: Dict[str, Any]) -> ExecutionResult:
        agent = self._assign_agent(task)
        result = await self._execute_task(agent, task, market_state)
        return result

    def _assign_agent(self, task: ExecutionTask) -> str:
        if task.urgency == "urgent":
            return "timing_optimizer"
        if abs(task.size) > 0.2 * task.metadata.get("notional_bounds", 1_000_000):
            return "impact_minimizer"
        return "liquidity_seeker"

    async def _execute_task(self, agent: str, task: ExecutionTask, market_state: Dict[str, Any]) -> ExecutionResult:
        await asyncio.sleep(0.01)  # simulate latency
        mid_price = market_state.get(task.asset, {}).get("mid", 100.0)
        volatility = market_state.get(task.asset, {}).get("volatility", 0.02)
        slippage = float(np.clip(np.random.normal(0.0005, volatility / 10), 0, 0.01))
        filled = task.size * (1 - np.random.uniform(0, 0.02))
        avg_price = mid_price * (1 + slippage * np.sign(task.size))

        return ExecutionResult(
            asset=task.asset,
            requested=task.size,
            filled=filled,
            avg_price=avg_price,
            slippage=slippage,
            latency_ms=float(np.random.uniform(5, self.latency_budget_ms)),
            agent=agent,
            metadata={"urgency": task.urgency},
        )

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #
    def _aggregate_results(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        if not results:
            return {"total_filled": 0.0, "avg_slippage": 0.0, "fill_quality": 0.0, "latency_ms_p95": 0.0}

        total_filled = float(sum(r.filled for r in results))
        avg_slippage = float(np.mean([r.slippage for r in results]))
        fill_quality = float(np.mean([abs(r.filled / (r.requested + 1e-8)) for r in results]))
        latency_ms_p95 = float(np.percentile([r.latency_ms for r in results], 95))

        return {
            "total_filled": total_filled,
            "avg_slippage": avg_slippage,
            "fill_quality": fill_quality,
            "latency_ms_p95": latency_ms_p95,
        }

    def _transaction_cost_analysis(self, strategy: Dict[str, Any], results: List[ExecutionResult]) -> Dict[str, Any]:
        if not results:
            return {}
        benchmark_prices = strategy.get("benchmark_prices", {})
        shortfall = []
        for result in results:
            benchmark = benchmark_prices.get(result.asset, result.avg_price)
            shortfall.append((result.avg_price - benchmark) * np.sign(result.requested))
        return {
            "implementation_shortfall": float(np.mean(shortfall)),
            "market_impact": float(np.mean([r.slippage for r in results])),
            "timing_cost": float(np.std([r.latency_ms for r in results]) / 1000.0),
        }

    # ------------------------------------------------------------------ #
    # Defaults
    # ------------------------------------------------------------------ #
    def _default_agents(self) -> List[Dict[str, Any]]:
        return [
            {"id": "liquidity_seeker", "role": "Find natural liquidity", "algo": "SAC"},
            {"id": "impact_minimizer", "role": "Minimize adverse selection", "algo": "SAC"},
            {"id": "timing_optimizer", "role": "Optimal slicing", "algo": "SAC"},
            {"id": "venue_router", "role": "Venue selection", "algo": "SAC"},
            {"id": "adversarial_defender", "role": "Mitigate HFT adversaries", "algo": "SAC"},
        ]
