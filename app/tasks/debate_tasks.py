from app.modules.taskgraph import Node, TaskGraph
from app.modules.llm_wrapper import async_llm_infer
from app.modules.data_sources import async_fetch_news
from app.modules.cognitive_middleware import cognitive
from app.modules.rl_agent import PPO
from opentelemetry import trace
import asyncio
from app.modules.logging_config import logger

tracer = trace.get_tracer(__name__)

# --- ??????????? ????? ---
async def fetch_news_task(context, deps):
    ticker = context['ticker']
    if not ticker.isupper() or not ticker.isalpha():
        raise ValueError("Invalid ticker")
    try:
        async with asyncio.timeout(5):  # 5s timeout
            news = await async_fetch_news(ticker)
        return {"news": news}
    except Exception as e:
        logger.error(f"Fetch news error: {e}")
        raise

async def llm_analysis_task(context, deps):
    news = deps['fetch_news']['news']
    with tracer.start_as_current_span("llm_analysis"):
        try:
            async with asyncio.timeout(10):
                result = await async_llm_infer(f"Analyze news for {context['ticker']}: {news}")
            return {"raw": result}
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

async def cognitive_task(context, deps):
    raw = deps['llm_analysis']['raw']
    try:
        processed = cognitive.process_reasoning(raw, context)
        return {"processed": processed}
    except Exception as e:
        logger.error(f"Cognitive error: {e}")
        raise

async def rl_decision_task(context, deps):
    processed = deps['cognitive']['processed']
    try:
        decision = await asyncio.to_thread(PPO.predict, processed)
        return {"decision": decision}
    except Exception as e:
        logger.error(f"RL error: {e}")
        raise

# --- ?????? TaskGraph ---
def create_debate_graph(ticker: str):
    tg = TaskGraph()
    tg.add(Node("fetch_news", fetch_news_task))
    tg.add(Node("llm_analysis", llm_analysis_task, depends_on=["fetch_news"]))
    tg.add(Node("cognitive", cognitive_task, depends_on=["llm_analysis"]))
    tg.add(Node("rl_decision", rl_decision_task, depends_on=["cognitive"]))
    context = {"ticker": ticker}
    return tg, context
