"""
Human-AI symbiosis interface for ACTS v6.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    query: str
    response: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class HumanAIInterface:
    def __init__(self) -> None:
        self.conversation_history: List[ConversationTurn] = []

    async def explain_decision(self, decision: Dict[str, Any], user_query: str) -> str:
        intent = self._parse_intent(user_query)
        if intent == "why":
            response = self._explain_why(decision)
        elif intent == "what_if":
            response = self._explain_counterfactual(decision, user_query)
        else:
            response = "I can explain my reasoning. Ask 'Why did you...?' or 'What if...?'"

        self.conversation_history.append(ConversationTurn(query=user_query, response=response))
        return response

    def _parse_intent(self, query: str) -> str:
        lower = query.lower()
        if "why" in lower:
            return "why"
        if "what if" in lower:
            return "what_if"
        return "unknown"

    def _explain_why(self, decision: Dict[str, Any]) -> str:
        regime = decision.get("regime", "unknown")
        expected_return = decision.get("expected_return", 0.0)
        rationale = decision.get("rationale", "Data-driven opportunity detected.")
        return f"I acted because regime={regime}, expected_return={expected_return:.2%}. {rationale}"

    def _explain_counterfactual(self, decision: Dict[str, Any], query: str) -> str:
        return (
            "If that event occurred, I would recompute causal interventions, "
            "stress scenarios, and adjust exposures accordingly."
        )
