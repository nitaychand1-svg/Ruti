import pytest
import asyncio
from app.modules.taskgraph import Node, TaskGraph
from hypothesis import given, strategies as st

async def dummy_task(context, deps):
    return {"result": 42}

async def failing_task(context, deps):
    raise ValueError("Fail")

@pytest.mark.asyncio
async def test_taskgraph_simple():
    tg = TaskGraph()
    tg.add(Node("a", dummy_task))
    results = await tg.run(start_nodes=["a"], context={})
    assert results["a"]["result"] == 42

@pytest.mark.asyncio
async def test_taskgraph_deps():
    tg = TaskGraph()
    tg.add(Node("a", dummy_task))
    tg.add(Node("b", dummy_task, depends_on=["a"]))
    results = await tg.run(start_nodes=["a"], context={})
    assert "b" in results

@pytest.mark.asyncio
async def test_taskgraph_error():
    tg = TaskGraph()
    tg.add(Node("a", failing_task))
    with pytest.raises(ValueError):
        await tg.run(start_nodes=["a"], context={})

@given(st.text(min_size=1))
def test_hypothesis_dummy(input_str):
    # Simple fuzz
    assert len(input_str) >= 1
