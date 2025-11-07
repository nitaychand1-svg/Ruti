"""
Self-evolution oracle for ACTS v6.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    timestamp: float
    sharpe: float
    drawdown: float
    latency_ms_p95: float
    notes: Dict[str, Any] = field(default_factory=dict)


class SelfEvolutionOracle:
    """
    Monitors performance metrics and triggers improvement actions.
    """

    def __init__(self) -> None:
        self.performance_history: List[PerformanceSnapshot] = []
        self.improvement_log: List[Dict[str, Any]] = []

    def record_performance(self, snapshot: PerformanceSnapshot) -> None:
        self.performance_history.append(snapshot)
        logger.debug("Recorded performance snapshot: %s", snapshot)

    async def monitor_and_improve(self) -> Dict[str, Any]:
        if len(self.performance_history) < 5:
            return {"status": "insufficient_data"}

        recent = self.performance_history[-30:]
        avg_sharpe = float(np.mean([snap.sharpe for snap in recent]))
        avg_latency = float(np.mean([snap.latency_ms_p95 for snap in recent]))

        improvements: List[Dict[str, Any]] = []

        if avg_sharpe < 1.5:
            improvements.append(
                {
                    "type": "architecture_search",
                    "reason": f"Sharpe {avg_sharpe:.2f} below 1.5",
                    "action": "Trigger NAS exploration",
                }
            )
        if avg_latency > 450:
            improvements.append(
                {
                    "type": "execution_optimization",
                    "reason": f"Latency {avg_latency:.1f} ms exceeds budget",
                    "action": "Optimize MARL swarm parameters",
                }
            )
        if max(snapshot.drawdown for snapshot in recent) > 0.08:
            improvements.append(
                {
                    "type": "risk_review",
                    "reason": "Drawdown exceeding 8%",
                    "action": "Run stress scenarios and rebalance risk limits",
                }
            )

        if improvements:
            self.improvement_log.extend(improvements)
            status = "actions_triggered"
        else:
            status = "stable"

        return {
            "status": status,
            "improvements_made": improvements,
            "latest_performance": dataclasses.asdict(self.performance_history[-1]),
        }
