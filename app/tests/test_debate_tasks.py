import pytest
import asyncio
from app.tasks.debate_tasks import create_debate_graph

@pytest.mark.asyncio
async def test_debate_graph():
    tg, context = create_debate_graph("AAPL")
    results = await tg.run(start_nodes=["fetch_news"], context=context)

    # ????????
    assert "rl_decision" in results
    decision = results["rl_decision"]["decision"]
    assert isinstance(decision, dict)
    assert "action" in decision
    assert "reason" in decision
    print("? Test passed for debate_task")

@pytest.mark.asyncio
async def test_debate_graph_error():
    tg, context = create_debate_graph("invalid123")  # Invalid: lowercase and has numbers
    with pytest.raises(ValueError):
        await tg.run(start_nodes=["fetch_news"], context=context)
    print("? Test passed for error handling")
