from fastapi import APIRouter, Request, Query
from app.tasks.debate_tasks import create_debate_graph
import asyncio
from app.modules.logging_config import logger

router = APIRouter()

@router.get("/debate/{ticker}")
async def debate(ticker: str, request: Request, debug: bool = Query(False)):
    corr_id = getattr(request.state, "corr_id", "unknown")
    if not ticker.isupper() or not ticker.isalpha():
        raise ValueError("Invalid ticker: must be uppercase letters")
    try:
        tg, context = create_debate_graph(ticker)
        results = await tg.run(start_nodes=["fetch_news"], context=context)
        if debug:
            return {"results": results, "corr_id": corr_id}
        return {"decision": results["rl_decision"]["decision"], "corr_id": corr_id}
    except Exception as e:
        logger.error(f"Debate error: {e}", extra={"corr_id": corr_id})
        raise
